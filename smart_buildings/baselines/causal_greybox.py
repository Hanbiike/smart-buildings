"""Experimental action-conditioned predictor; never attached to a controller.

AHU temperature sensors report setpoints, not discharge temperature. Reconstruct
the latter from recirculation and outdoor air before applying the cooling cap.
Only observation history, proposed actions and participant metadata are used.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from smart_buildings.baselines.greybox import GreyBoxModel, _trailing_room_number


class CausalGreyBoxModel(GreyBoxModel):
    """Bounded delta-temperature ARX with a lightweight physical input map."""

    def __init__(self, metadata, *, fit_window=144, use_actions=True, sensor_delay=0,
                 calibrate_plant=False):
        self._suffix_cache = {}
        super().__init__(metadata, fit_window=fit_window)
        self.use_actions = use_actions
        if sensor_delay not in (0, 1):
            raise ValueError('sensor_delay must be 0 or 1')
        self.sensor_delay = sensor_delay
        self.calibrate_plant = calibrate_plant
        self.damper_floor = {}
        self.pressure_floor = {}
        self.water_heating_step = 0.0
        self.water_cooling_step = 0.0
        self._prediction_paths = {}
        self.lagged_action = None
        self.rooms = sorted(self.room_temp_obs)
        self.ahu_position = {ahu: i for i, ahu in enumerate(self.ahu_ids)}
        self.actions = {
            _trailing_room_number(name): indices
            for name, indices in metadata['zone_action_indices'].items()
        }
        self.history = {room: [] for room in self.rooms}
        self.coefficients = {room: np.zeros(7) for room in self.rooms}
        self.feature_prior = np.array([0, 0, 0, 0, 0, 0, 0.5])
        self.feature_lower = np.array([0, 0, 0, 0, -1, -1, 0])
        self.feature_upper = np.array([2, 1, 2, 2, 1, 1, 0.98])
        self.ridge_strength = 0.01
        self.previous_obs = None
        self.previous_action = None
        self.delta = {room: 0.0 for room in self.rooms}
        self.observation = None
        self._forecast_inputs = None

    def _first_suffix(self, suffix):
        # Field layout is immutable for one model instance. Rollouts query these
        # same suffixes thousands of times; never rescan the complete layout.
        if suffix not in self._suffix_cache:
            self._suffix_cache[suffix] = super()._first_suffix(suffix)
        return self._suffix_cache[suffix]

    def _field(self, obs, ahu, suffix, default):
        index = self._first_suffix(f'ahu_{ahu}_{suffix}')
        return float(obs[index]) if index is not None else default

    def _native(self, value, key):
        stats = self.metadata['action_normalization_metadata'][key]
        lo, hi = stats['min_normalized_value'], stats['max_normalized_value']
        fraction = np.clip((float(value) - lo) / (hi - lo), 0, 1)
        return float(stats['min_native_value'] + fraction * (
            stats['max_native_value'] - stats['min_native_value']))

    def inputs(self, obs, action=None, *, temperatures=None):
        """Construct same-step plant inputs without reading the next observation.

        Flow capacity is inferred from reported flow/fan ratio when available.
        Minimum fan pressure is conservatively estimated from the last sensor;
        neither the hidden fan floor nor coil water distribution is observable.
        """
        if temperatures is None:
            temperatures = {r: self.zone_temp(obs, r) for r in self.rooms}
        ambient = self._ambient(obs)
        result = {}
        for ahu in self.ahu_ids:
            members = [r for r in self.rooms if self.room_ahu.get(r) == ahu]
            mean = float(np.mean([temperatures[r] for r in members]))
            pos = self.ahu_position[ahu]
            pressure = self._field(obs, ahu, 'supply_air_static_pressure_sensor', 0)
            recirc = self._field(obs, ahu, 'recirculation_percentage_command', 1)
            sat = self._ahu_supply(obs, ahu)
            run = self._ahu_running(obs, ahu)
            if action is not None and self.use_actions:
                continuous, discrete = action
                run = float(discrete[pos] > 0)
                sat = self._native(continuous[self.metadata['temp_indices'][pos]],
                                   f'ahu_{ahu}_supply_air_temperature_setpoint')
                command_pressure = self._native(
                    continuous[self.metadata['pressure_indices'][pos]],
                    f'ahu_{ahu}_supply_air_static_pressure_setpoint')
                floor = (self.pressure_floor.get(ahu, 0.0) if self.calibrate_plant
                         else min(pressure, 100.0))
                pressure = max(command_pressure, floor)
                recirc = self._native(
                    continuous[self.metadata['recirculation_indices'][pos]],
                    f'ahu_{ahu}_recirculation_percentage_command')
            supply = min(recirc * mean + (1 - recirc) * ambient, sat)
            flows, reheats = {}, {}
            for room in members:
                damper = self._damper_native(obs, room)
                reheat = self._reheat_native(obs, room)
                if action is not None and self.use_actions:
                    indices = self.actions[room]
                    damper = self._native(action[0][indices['supply_air_damper_percentage_command']],
                                          'supply_air_damper_percentage_command')
                    reheat = self._native(action[0][indices['reheat_valve_percentage_command']],
                                          'reheat_valve_percentage_command')
                    if self.calibrate_plant:
                        damper = max(damper, self.damper_floor.get(room, 0.0))
                design_idx = self._first_suffix(f'VAV_{room}_supply_air_flowrate_setpoint')
                design = float(obs[design_idx]) if design_idx is not None else 1.0
                flows[room] = damper * design * np.sqrt(max(pressure, 0) / 2000) * run
                reheats[room] = reheat
            total = sum(flows.values())
            fan = self._field(obs, ahu, 'supply_fan_speed_percentage_command', 0)
            reported_flow = self._field(obs, ahu, 'supply_air_flowrate_sensor', 0)
            capacity = reported_flow / fan if fan > 1e-6 else float('inf')
            ratio = min(1.0, capacity / total) if total > 0 else 1.0
            for room in members:
                result[room] = dict(ambient=ambient, mean=mean, supply=supply,
                                    flow=flows[room] * ratio, reheat=reheats[room],
                                    water=self._water_temp(obs),
                                    comfort=float(obs[self.field_pos['comfort_mode_now']]))
        return result

    @staticmethod
    def features(temp, delta, inputs):
        flow, supply = inputs['flow'], inputs['supply']
        # A bounded proxy, not an exact coil model: shared water flow is hidden.
        reheat_gap = min(inputs['reheat'] * max(0, inputs['water'] - supply),
                         max(0, 308.15 - supply))
        return np.array([(inputs['ambient'] - temp) / 40,
                         (inputs['mean'] - temp) / 5,
                         flow * (supply - temp) / 10,
                         flow * reheat_gap / 10,
                         inputs['comfort'], 1.0, delta])

    def update(self, obs):
        if self.calibrate_plant and self.previous_obs is not None and self.previous_action is not None:
            continuous, discrete = self.previous_action
            for room in self.rooms:
                command = self._native(continuous[self.actions[room][
                    'supply_air_damper_percentage_command']], 'supply_air_damper_percentage_command')
                measured = self._damper_native(obs, room)
                if measured > command + 1e-5:
                    self.damper_floor[room] = min(self.damper_floor.get(room, measured), measured)
            for ahu, pos in self.ahu_position.items():
                command = self._native(continuous[self.metadata['pressure_indices'][pos]],
                                       f'ahu_{ahu}_supply_air_static_pressure_setpoint')
                measured = self._field(obs, ahu, 'supply_air_static_pressure_sensor', 0)
                if discrete[pos] > 0 and measured > command + 1e-5:
                    self.pressure_floor[ahu] = min(self.pressure_floor.get(ahu, measured), measured)
            water_indices = self.metadata.get('water_indices', [])
            if water_indices:
                gap = self._native(continuous[water_indices[0]], 'supply_water_temperature_setpoint') - self._water_temp(self.previous_obs)
                change = self._water_temp(obs) - self._water_temp(self.previous_obs)
                if gap > 0 and change > 0:
                    self.water_heating_step = max(self.water_heating_step, change)
                if gap < 0 and change < 0:
                    self.water_cooling_step = max(self.water_cooling_step, -change)
        if self.previous_obs is not None:
            transition_action = self.lagged_action if self.sensor_delay else self.previous_action
            inputs = self.inputs(self.previous_obs, transition_action)
            for room in self.rooms:
                temp = self.zone_temp(self.previous_obs, room)
                target = self.zone_temp(obs, room) - temp
                self.history[room].append((self.features(temp, self.delta[room], inputs[room]), target))
                self.delta[room] = target
                count = len(self.history[room])
                if count >= self.fit_window and (count == self.fit_window or count % 12 == 0):
                    self._fit_room(room)
        self.lagged_action = self.previous_action
        self.observation = np.asarray(obs).copy()
        self.previous_obs = self.observation
        self.previous_action = None
        self._forecast_inputs = None
        self._prediction_paths = {}

    def _fit_room(self, room):
        x = np.array([row[0] for row in self.history[room]])
        y = np.array([row[1] for row in self.history[room]])
        # Fixed feature scales, ridge prior and non-negative physical gains.
        prior = self.feature_prior
        root_ridge = np.sqrt(self.ridge_strength)
        fit = lsq_linear(np.vstack([x, root_ridge * np.eye(len(prior))]),
                         np.r_[y, root_ridge * prior],
                         bounds=(self.feature_lower, self.feature_upper))
        self.coefficients[room] = fit.x

    def set_action(self, continuous, discrete):
        self.previous_action = (np.asarray(continuous).copy(), np.asarray(discrete).copy())
        self._forecast_inputs = None
        self._prediction_paths = {}

    def predict(self, room, steps, *, temp0=None, hold=None):
        if hold is not None:
            raise ValueError('Use set_action for candidate commands')
        if self.calibrate_plant:
            actual_temp = self.zone_temp(self.observation, room)
            if temp0 is not None and abs(temp0 - actual_temp) > 1e-9:
                raise ValueError('Calibrated forecasts start at the observed state')
            if steps == 0:
                return actual_temp
            if steps not in self._prediction_paths:
                commands = [self.previous_action] * max(0, steps - self.sensor_delay)
                self._prediction_paths[steps] = self.predict_action_path(commands, steps)
            return self._prediction_paths[steps][-1][room]
        if self._forecast_inputs is None:
            proposed = self.inputs(self.observation, self.previous_action)
            pending = (self.inputs(self.observation, self.lagged_action)
                       if self.sensor_delay else proposed)
            self._forecast_inputs = proposed, pending
        inputs = self._forecast_inputs[0][room]
        pending_inputs = self._forecast_inputs[1][room]
        temp = self.zone_temp(self.observation, room) if temp0 is None else float(temp0)
        delta = self.delta[room]
        for step in range(steps):
            step_inputs = pending_inputs if step == 0 else inputs
            delta = float(self.features(temp, delta, step_inputs) @ self.coefficients[room])
            temp += delta
        return temp

    def trusted(self, room, residual_limit=0.5):
        # Forecast improvements do not establish counterfactual control accuracy.
        return False

    def predict_action_path(self, actions, steps):
        """Diagnostic rollout of a supplied command path with frozen parameters.

        No future observations are read. Supplying recorded future commands is
        an OFFLINE ORACLE diagnostic, not an available online forecast.
        Weather, return-air proxy and water temperature remain held at origin.
        """
        needed = max(0, steps - self.sensor_delay)
        if len(actions) < needed:
            raise ValueError('Insufficient commands for the requested horizon')
        temperatures = {r: self.zone_temp(self.observation, r) for r in self.rooms}
        deltas = self.delta.copy()
        path = []
        water = self._water_temp(self.observation)
        for step in range(steps):
            command = (self.lagged_action if self.sensor_delay and step == 0
                       else actions[step - self.sensor_delay])
            inputs = self.inputs(self.observation, command)
            if self.calibrate_plant:
                indices = self.metadata.get('water_indices', [])
                if command is not None and indices and not (self.sensor_delay and step == 0):
                    target = self._native(command[0][indices[0]], 'supply_water_temperature_setpoint')
                    gap = target - water
                    water += (min(gap, self.water_heating_step) if gap > 0
                              else max(gap, -self.water_cooling_step))
                for values in inputs.values():
                    values['water'] = water
            for room in self.rooms:
                deltas[room] = float(self.features(
                    temperatures[room], deltas[room], inputs[room]) @ self.coefficients[room])
                temperatures[room] += deltas[room]
            path.append(temperatures.copy())
        return path

    def steps_to_reach(self, *args, **kwargs):
        raise NotImplementedError('Warm-up scheduling has not been validated')
