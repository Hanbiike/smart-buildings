"""Observation/command adapter for the isolated equipment balances.

Zone temperatures latch BEFORE the simulator's thermal solve; actuator and
flow observations arrive AFTER it. At observation t, the pending temperature
transition belongs to command t-1, with outdoor/water inputs from t-1. A new
command t must not replace that pending transition. No thermal model is fitted
here and nothing is connected to a controller.
"""

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from smart_buildings.baselines.equipment_balance import (
    AIR_CAPACITY, air_flows, air_handler, reheat_coil, water_flows,
)


class UnobservedCapacityError(ValueError):
    """No finite handler capacity is available for a proposed active command."""


class UnidentifiedActuatorError(ValueError):
    """A proposed command may hit an actuator floor that has not been identified."""


def _number(name):
    match = re.search(r'(\d+)$', name)
    if match is None:
        raise ValueError(f'Device lacks a numeric suffix: {name}')
    return int(match.group(1))


def component_zone_areas(zone_map, room_ids):
    """Reconstruct component labels, not counts of material values 0/1/2.

    This follows the public regular-zone convention, not arbitrary custom zone
    mappings. A caller with a custom mapping must supply validated areas itself.
    Areas are grid-cell counts; only relative areas are used for return mixing.
    """
    import cv2

    array = np.asarray(zone_map)
    if array.ndim != 2 or not np.isfinite(array).all() or not np.isin(array, [0, 1, 2]).all():
        raise ValueError('Expected a two-dimensional material map with values 0/1/2')
    count, _, stats, _ = cv2.connectedComponentsWithStats((array == 0).astype(np.uint8), connectivity=4)
    if set(room_ids) != set(range(1, count)):
        raise ValueError('Geometry components do not match the observed zone labels')
    return {room: float(stats[room, cv2.CC_STAT_AREA]) for room in room_ids}


@dataclass(frozen=True)
class EquipmentAssumptions:
    """Explicit global priors, NOT inferred hidden equipment specifications."""
    design_air_pressure_pa: float = 2000.0
    valve_kv: float = 0.108
    header_resistance: float = 0.0
    max_discharge_temperature: float = 308.15
    fan_efficiency: float = 0.9

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError('Equipment assumptions must be finite')
        if (self.design_air_pressure_pa <= 0 or self.valve_kv < 0
                or self.header_resistance < 0 or self.max_discharge_temperature <= 0
                or not 0 < self.fan_efficiency <= 1):
            raise ValueError('Invalid equipment assumptions')


@dataclass(frozen=True)
class ZoneDuty:
    air_flow: float
    water_flow: float
    supply_temperature: float  # BEFORE reheat
    discharge_temperature: float
    air_power: float  # instantaneous at the supplied zone state, not a step-average
    reheat_power: float

    @property
    def total_power(self):
        return self.air_power + self.reheat_power


@dataclass(frozen=True)
class EquipmentEstimate:
    control_step: int
    outdoor_temperature: Mapping[int, float]
    water_temperature: float
    zones: Mapping[int, ZoneDuty]
    handler_flows: Mapping[int, float]
    cooling_power: float
    intake_fan_power: float  # excludes exhaust fans, pumps and chiller electricity
    temperature_basis: str
    uncertain_damper_floors: tuple[int, ...]
    uncertain_pressure_floors: tuple[int, ...]


@dataclass(frozen=True)
class DelayedTransition:
    start_observation: int
    end_observation: int
    start_temperatures: Mapping[int, float]
    end_temperatures: Mapping[int, float]
    forcing: EquipmentEstimate


class EquipmentObserver:
    def __init__(self, metadata, *, assumptions, zone_areas=None):
        self.metadata = metadata
        self.assumptions = assumptions
        self.positions = {name: i for i, name in enumerate(metadata['observation_field_names'])}
        self.norm = metadata['observation_normalization_metadata']
        self.action_norm = metadata['action_normalization_metadata']
        self.zone_actions = {_number(name): indices for name, indices in metadata['zone_action_indices'].items()}
        self.rooms = sorted(self.zone_actions)
        self.handlers = {_number(name): [_number(room) for room in rooms]
                         for name, rooms in metadata['ahu_zone_assignments'].items()}
        self.ahu_ids = sorted(self.handlers)
        assigned = [room for rooms in self.handlers.values() for room in rooms]
        if sorted(assigned) != self.rooms or len(set(assigned)) != len(assigned):
            raise ValueError('Every observed zone must belong to exactly one handler')
        if len(metadata['pressure_indices']) != len(self.ahu_ids) + 1:
            raise ValueError('Expected handler pressures and one common water loop')
        if metadata['n_discrete_action'] != len(self.ahu_ids) + 1:
            raise ValueError('Expected one discrete command per handler and water loop')
        self.zone_indices = {room: {key: self._index(f'VAV_{room}_{key}') for key in (
            'zone_air_temperature_sensor', 'supply_air_damper_percentage_command',
            'reheat_valve_percentage_command', 'supply_air_flowrate_setpoint')}
            for room in self.rooms}
        self.ahu_indices = {ahu: {key: self._index(f'ahu_{ahu}_{key}') for key in (
            'outside_air_temperature_sensor', 'supply_air_temperature_sensor',
            'supply_air_static_pressure_sensor', 'recirculation_percentage_command',
            'supervisor_run_command', 'supply_air_flowrate_sensor',
            'supply_fan_speed_percentage_command')} for ahu in self.ahu_ids}
        self.water_idx = self._index('_supply_water_temperature_sensor')
        self.water_pressure_idx = self._index('_differential_pressure_sensor')
        if zone_areas is None and metadata.get('zone_map') is not None:
            zone_areas = component_zone_areas(metadata['zone_map'], self.rooms)
        if zone_areas is not None and set(zone_areas) != set(self.rooms):
            raise ValueError('Area mapping must cover exactly the observed zones')
        self.area_weighting = 'provided_areas' if zone_areas is not None else 'equal_area_proxy'
        self.areas = {room: float(zone_areas[room]) if zone_areas is not None else 1.0
                      for room in self.rooms}
        if any(not np.isfinite(v) or v <= 0 for v in self.areas.values()):
            raise ValueError('Zone areas must be finite and positive')
        self.damper_floors, self.pressure_floors, self.capacities = {}, {}, {}
        self.minimum_dampers, self.minimum_running_pressures = {}, {}
        self.capacity_samples = {ahu: [] for ahu in self.ahu_ids}
        self.water_heat_step = self.water_cool_step = 0.0
        self.step = -1
        self.frame = self.pending = self.last_action = None

    def _index(self, suffix):
        matches = [i for name, i in self.positions.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one observation ending in {suffix}')
        return matches[0]

    def _zone_value(self, obs, room, field):
        stats = self.norm[field]
        return float(obs[self.zone_indices[room][field]]) * stats['sample_std'] + stats['sample_mean']

    def _native(self, value, key):
        stats = self.action_norm[key]
        low, high = stats['min_normalized_value'], stats['max_normalized_value']
        if not low - 1e-6 <= value <= high + 1e-6 or high <= low:
            raise ValueError(f'Invalid normalized command for {key}')
        fraction = np.clip((value - low) / (high - low), 0, 1)
        return float(stats['min_native_value'] + fraction *
                     (stats['max_native_value'] - stats['min_native_value']))

    def _decode(self, obs):
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (len(self.positions),) or not np.isfinite(obs).all():
            raise ValueError('Invalid observation shape or nonfinite observation')
        zones = {room: dict(
            temperature=self._zone_value(obs, room, 'zone_air_temperature_sensor'),
            damper=self._zone_value(obs, room, 'supply_air_damper_percentage_command'),
            valve=self._zone_value(obs, room, 'reheat_valve_percentage_command'),
            design_flow=float(obs[self.zone_indices[room]['supply_air_flowrate_setpoint']]))
            for room in self.rooms}
        handlers = {ahu: {key: float(obs[index]) for key, index in indices.items()}
                    for ahu, indices in self.ahu_indices.items()}
        return dict(zones=zones, handlers=handlers, water=float(obs[self.water_idx]),
                    water_pressure=float(obs[self.water_pressure_idx]))

    def set_action(self, continuous, discrete):
        continuous, discrete = np.asarray(continuous, float), np.asarray(discrete)
        if (continuous.shape != (self.metadata['n_continuous_action'],)
                or discrete.shape != (self.metadata['n_discrete_action'],)
                or not np.isfinite(continuous).all() or not np.isin(discrete, [0, 1]).all()):
            raise ValueError('Invalid action shapes or values')
        self.last_action = continuous.copy(), discrete.copy()

    def _settings(self, frame, action):
        zones = {room: values.copy() for room, values in frame['zones'].items()}
        handlers = {ahu: values.copy() for ahu, values in frame['handlers'].items()}
        water_pressure = frame['water_pressure']
        if action is not None:
            continuous, discrete = action
            for room in self.rooms:
                indices = self.zone_actions[room]
                zones[room]['damper'] = max(self.damper_floors.get(room, 0.0), self._native(
                    continuous[indices['supply_air_damper_percentage_command']],
                    'supply_air_damper_percentage_command'))
                zones[room]['valve'] = self._native(continuous[indices['reheat_valve_percentage_command']],
                                                  'reheat_valve_percentage_command')
            for pos, ahu in enumerate(self.ahu_ids):
                settings = handlers[ahu]
                settings['supervisor_run_command'] = float(discrete[pos])
                for field, index_list in [('supply_air_temperature_setpoint', 'temp_indices'),
                                           ('recirculation_percentage_command', 'recirculation_indices'),
                                           ('supply_air_static_pressure_setpoint', 'pressure_indices')]:
                    value = self._native(continuous[self.metadata[index_list][pos]], f'ahu_{ahu}_{field}')
                    target = field.replace('setpoint', 'sensor') if field.endswith('setpoint') else field
                    settings[target] = value
                settings['supply_air_static_pressure_sensor'] = max(
                    settings['supply_air_static_pressure_sensor'], self.pressure_floors.get(ahu, 0.0))
            water_pressure = (self._native(continuous[self.metadata['pressure_indices'][-1]],
                                          'differential_pressure_setpoint') if discrete[-1] else 0.0)
        return zones, handlers, water_pressure

    def _balance(self, frame, action, *, control_step, outdoor, water,
                 temperatures, basis):
        zones, handlers, water_pressure = self._settings(frame, action)
        ordered = self.rooms
        water_vector = water_flows([zones[r]['valve'] for r in ordered],
                                   [self.assumptions.valve_kv] * len(ordered), water_pressure,
                                   self.assumptions.header_resistance, running=water_pressure > 0)
        water_by_room = dict(zip(ordered, water_vector))
        duties, totals = {}, {}
        uncertain_dampers, uncertain_pressures = [], []
        cooling_power = fan_power = 0.0
        for ahu, members in self.handlers.items():
            settings = handlers[ahu]
            running = settings['supervisor_run_command'] > 0.5
            if action is not None and running:
                for room in members:
                    if (room not in self.damper_floors and zones[room]['damper'] <
                            self.minimum_dampers.get(room, 1.0) - 1e-6):
                        uncertain_dampers.append(room)
                if (ahu not in self.pressure_floors and settings['supply_air_static_pressure_sensor'] <
                        self.minimum_running_pressures.get(ahu, float('inf')) - 1e-6):
                    uncertain_pressures.append(ahu)
            capacity = self.capacities.get(ahu)
            if capacity is None:
                if action is not None and running:
                    raise UnobservedCapacityError(f'Handler {ahu} capacity has not yet been observed')
                capacity = settings['supply_air_flowrate_sensor']
            flows = air_flows([zones[r]['damper'] for r in members],
                              [zones[r]['design_flow'] for r in members],
                              settings['supply_air_static_pressure_sensor'],
                              self.assumptions.design_air_pressure_pa, capacity, running=running)
            total = float(flows.sum())
            mean = sum(temperatures[r] * self.areas[r] for r in members) / sum(self.areas[r] for r in members)
            # Outdoor input is explicit: pending physics uses the preceding
            # observation's weather, not weather at the new sensor timestamp.
            balance = air_handler(mean, outdoor[ahu], settings['recirculation_percentage_command'],
                                  settings['supply_air_temperature_sensor'], total,
                                  settings['supply_air_static_pressure_sensor'],
                                  self.assumptions.fan_efficiency, running=running)
            cooling_power += balance.cooling_power
            fan_power += balance.fan_power
            totals[ahu] = total
            for room, flow in zip(members, flows):
                coil = reheat_coil(balance.supply_temperature, water, float(flow),
                                   float(water_by_room[room]), self.assumptions.max_discharge_temperature)
                duties[room] = ZoneDuty(float(flow), float(water_by_room[room]),
                                       balance.supply_temperature, coil.discharge_temperature,
                                       AIR_CAPACITY * flow * (balance.supply_temperature - temperatures[room]),
                                       coil.heat_to_air)
        return EquipmentEstimate(control_step, MappingProxyType(dict(outdoor)), water,
                                 MappingProxyType(duties), MappingProxyType(totals),
                                 cooling_power, fan_power, basis,
                                 tuple(uncertain_dampers), tuple(uncertain_pressures))

    def observe(self, obs):
        frame = self._decode(obs)
        previous, previous_pending = self.frame, self.pending
        if previous is not None and self.last_action is not None:
            continuous, discrete = self.last_action
            for room in self.rooms:
                requested = self._native(continuous[self.zone_actions[room][
                    'supply_air_damper_percentage_command']], 'supply_air_damper_percentage_command')
                actual = frame['zones'][room]['damper']
                if actual > requested + 1e-5:
                    self.damper_floors[room] = min(self.damper_floors.get(room, actual), actual)
            for pos, ahu in enumerate(self.ahu_ids):
                requested = self._native(continuous[self.metadata['pressure_indices'][pos]],
                                         f'ahu_{ahu}_supply_air_static_pressure_setpoint')
                actual = frame['handlers'][ahu]['supply_air_static_pressure_sensor']
                if discrete[pos] and actual > requested + 1e-5:
                    self.pressure_floors[ahu] = min(self.pressure_floors.get(ahu, actual), actual)
            target = self._native(continuous[self.metadata['water_indices'][0]],
                                  'supply_water_temperature_setpoint')
            change = frame['water'] - previous['water']
            if change > 0 and target > previous['water']:
                self.water_heat_step = max(self.water_heat_step, change)
            if change < 0 and target < previous['water']:
                self.water_cool_step = max(self.water_cool_step, -change)
        for ahu, settings in frame['handlers'].items():
            if settings['supervisor_run_command'] > 0.5:
                pressure = settings['supply_air_static_pressure_sensor']
                self.minimum_running_pressures[ahu] = min(
                    self.minimum_running_pressures.get(ahu, pressure), pressure)
            flow, fan = settings['supply_air_flowrate_sensor'], settings['supply_fan_speed_percentage_command']
            if flow > 0 and fan > 1e-6:
                self.capacity_samples[ahu].append(flow / fan)
                self.capacities[ahu] = float(np.median(self.capacity_samples[ahu][-32:]))
        for room, settings in frame['zones'].items():
            self.minimum_dampers[room] = min(self.minimum_dampers.get(room, settings['damper']),
                                             settings['damper'])
        self.step += 1
        self.frame, self.last_action = frame, None
        temperatures = {r: frame['zones'][r]['temperature'] for r in self.rooms}
        self.pending = (self._balance(
            frame, None, control_step=self.step - 1,
            outdoor={ahu: previous['handlers'][ahu]['outside_air_temperature_sensor'] for ahu in self.ahu_ids},
            water=previous['water'], temperatures=temperatures, basis='latched_step_start')
            if previous is not None else None)
        if previous_pending is None:
            return None  # Skip the reset's duplicate initial temperature.
        return DelayedTransition(self.step - 1, self.step,
                                 MappingProxyType({r: previous['zones'][r]['temperature'] for r in self.rooms}),
                                 MappingProxyType(temperatures), previous_pending)

    def project_action(self, continuous, discrete, *, zone_temperatures=None,
                       require_identified_floors=False, outdoor_temperatures=None,
                       water_temperature=None):
        if self.frame is None:
            raise ValueError('Observe a state before projecting commands')
        # Validate/copy without replacing the command recorded for calibration.
        saved = self.last_action
        try:
            self.set_action(continuous, discrete)
            action = self.last_action
        finally:
            self.last_action = saved
        temperatures = (dict(zone_temperatures) if zone_temperatures is not None else
                        {r: self.frame['zones'][r]['temperature'] for r in self.rooms})
        if set(temperatures) != set(self.rooms):
            raise ValueError('A temperature is required for each zone')
        outdoor = (dict(outdoor_temperatures) if outdoor_temperatures is not None else
                   {ahu: self.frame['handlers'][ahu]['outside_air_temperature_sensor']
                    for ahu in self.ahu_ids})
        if set(outdoor) != set(self.ahu_ids):
            raise ValueError('An outdoor temperature is required for each handler')
        estimate = self._balance(self.frame, action, control_step=self.step,
                             outdoor=outdoor,
                             water=self.frame['water'] if water_temperature is None else float(water_temperature),
                             temperatures=temperatures,
                             basis='provided_state' if zone_temperatures is not None else 'lagged_observation_proxy')
        if require_identified_floors and (estimate.uncertain_damper_floors or estimate.uncertain_pressure_floors):
            raise UnidentifiedActuatorError('Candidate may encounter an unidentified actuator floor')
        return estimate

    def project_water_sensor(self, continuous, *, start_temperature=None):
        if self.frame is None:
            raise ValueError('Observe a state before projecting water temperature')
        continuous = np.asarray(continuous, float)
        if continuous.shape != (self.metadata['n_continuous_action'],):
            raise ValueError('Invalid continuous action shape')
        target = self._native(float(continuous[self.metadata['water_indices'][0]]),
                              'supply_water_temperature_setpoint')
        water = self.frame['water'] if start_temperature is None else float(start_temperature)
        if not np.isfinite(water):
            raise ValueError('Water temperature must be finite')
        gap = target - water
        return water + (min(gap, self.water_heat_step) if gap > 0 else max(gap, -self.water_cool_step))
