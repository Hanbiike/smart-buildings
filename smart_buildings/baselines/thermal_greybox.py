"""Offline-only ablations of coil physics and a causal slow thermal state.

No scenario identities/configuration are read. The Kv and discharge limit are
global physical priors, not inferred hidden equipment specifications. The slow
state is a filtered air-temperature proxy, not a measured wall temperature.
"""

from __future__ import annotations

import numpy as np

from smart_buildings.baselines.causal_greybox import CausalGreyBoxModel


class ThermalGreyBoxModel(CausalGreyBoxModel):
    """Keep water and wall-state changes independently switchable for ablation."""

    def __init__(self, metadata, *, fit_window=144, physical_water=True,
                 thermal_mass=True, mass_time_steps=24.0, water_kv=0.108,
                 header_resistance=0.0):
        if mass_time_steps < 1 or water_kv < 0 or header_resistance < 0:
            raise ValueError('Invalid physical prior')
        super().__init__(metadata, fit_window=fit_window, sensor_delay=1,
                         calibrate_plant=True)
        self.physical_water = physical_water
        self.thermal_mass = thermal_mass
        self.mass_time_steps = mass_time_steps
        self.water_kv = water_kv
        self.header_resistance = header_resistance
        self.mass = {}
        if thermal_mass:
            self.feature_prior = np.r_[self.feature_prior, 0.0]
            self.feature_lower = np.r_[self.feature_lower, 0.0]
            self.feature_upper = np.r_[self.feature_upper, 1.0]
            self.coefficients = {room: np.zeros(8) for room in self.rooms}

    def inputs(self, obs, action=None, *, temperatures=None):
        result = super().inputs(obs, action, temperatures=temperatures)
        if self.physical_water:
            pressure_idx = self._first_suffix('_differential_pressure_sensor')
            pressure = float(obs[pressure_idx]) if pressure_idx is not None else 0.0
            if action is not None:
                pressure_indices = self.metadata['pressure_indices']
                if len(pressure_indices) > len(self.ahu_ids):
                    pressure = self._native(action[0][pressure_indices[-1]],
                                            'differential_pressure_setpoint')
                if len(action[1]) > len(self.ahu_ids) and action[1][-1] <= 0:
                    pressure = 0.0
            total_kv = sum(values['reheat'] * self.water_kv for values in result.values())
            pressure_factor = np.sqrt(max(0.0, pressure) /
                                      (1 + total_kv ** 2 * self.header_resistance))
            for values in result.values():
                # Kv * sqrt(bar) is m3/hour, NOT m3/second.
                values['water_flow'] = (values['reheat'] * self.water_kv *
                                        pressure_factor / 3600.0)
        for room, values in result.items():
            values['mass'] = self.mass.get(room, self.zone_temp(obs, room))
        return result

    @staticmethod
    def coil_power(inputs):
        """Coil duty in W, bounded by water, air and discharge headroom.

        With zero header resistance, each valve flow is independent of the
        number of other open valves. Artificial fixed-total-flow sharing would
        contradict that balance. Nonzero resistance provides genuine coupling.
        """
        air_capacity = 1.2 * 1006.0 * max(0.0, inputs['flow'])
        water_capacity = 1000.0 * 4180.0 * max(0.0, inputs['water_flow'])
        if air_capacity == 0 or water_capacity == 0:
            return 0.0
        gap = inputs['water'] - inputs['supply']
        effectiveness = min(1.0, water_capacity / air_capacity)
        if gap > 0:
            effectiveness = min(effectiveness,
                                max(0, 308.15 - inputs['supply']) / gap)
        return effectiveness * air_capacity * gap

    def features(self, temp, delta, inputs):
        features = super().features(temp, delta, inputs)
        if self.physical_water:
            features[3] = self.coil_power(inputs) / 10000.0
        if self.thermal_mass:
            features = np.r_[features, (inputs['mass'] - temp) / 5.0]
        return features

    def update(self, obs):
        if not self.mass:
            self.mass = {room: self.zone_temp(obs, room) for room in self.rooms}
        # Parent fits the transition starting at the preceding observed state.
        # Keep its matching mass proxy until that transition has been absorbed.
        super().update(obs)
        for room in self.rooms:
            self.mass[room] += (self.zone_temp(obs, room) - self.mass[room]) / self.mass_time_steps

    def predict_action_path(self, actions, steps):
        needed = max(0, steps - self.sensor_delay)
        if len(actions) < needed:
            raise ValueError('Insufficient commands for the requested horizon')
        temperatures = {room: self.zone_temp(self.observation, room) for room in self.rooms}
        masses = self.mass.copy()
        deltas = self.delta.copy()
        water = self._water_temp(self.observation)
        path = []
        for step in range(steps):
            command = self.lagged_action if step == 0 else actions[step - 1]
            # Preserve the calibrated model's return-air assumption so the
            # water and mass ablations do not silently add another change.
            inputs = self.inputs(self.observation, command)
            indices = self.metadata.get('water_indices', [])
            if step > 0 and command is not None and indices:
                target = self._native(command[0][indices[0]], 'supply_water_temperature_setpoint')
                gap = target - water
                water += (min(gap, self.water_heating_step) if gap > 0
                          else max(gap, -self.water_cooling_step))
            for room in self.rooms:
                inputs[room]['water'] = water
                inputs[room]['mass'] = masses[room]
                deltas[room] = float(self.features(temperatures[room], deltas[room],
                                                  inputs[room]) @ self.coefficients[room])
                temperatures[room] += deltas[room]
                masses[room] += (temperatures[room] - masses[room]) / self.mass_time_steps
            path.append(temperatures.copy())
        return path
