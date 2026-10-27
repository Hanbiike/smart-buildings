"""Tests of coil energy bounds, hydraulic coupling and causal mass state."""

import copy
import unittest

import numpy as np

from smart_buildings.baselines.thermal_greybox import ThermalGreyBoxModel
from tools.test_causal_greybox import case


def water_case(two_rooms=False):
    metadata, obs, action = case()
    metadata['observation_field_names'].append('hws_differential_pressure_sensor')
    metadata['pressure_indices'].append(5)
    metadata['action_normalization_metadata']['differential_pressure_setpoint'] = dict(
        min_native_value=0, max_native_value=20,
        min_normalized_value=-1, max_normalized_value=1)
    obs = np.r_[obs, 20.0]
    continuous = np.r_[action[0], 1.0]
    continuous[4] = 1
    if two_rooms:
        metadata['observation_field_names'].extend([
            'VAV_2_zone_air_temperature_sensor',
            'VAV_2_supply_air_damper_percentage_command',
            'VAV_2_reheat_valve_percentage_command',
            'VAV_2_supply_air_flowrate_setpoint'])
        metadata['ahu_zone_assignments']['ahu_1'].append('room_2')
        metadata['zone_action_indices']['VAV_2'] = dict(
            supply_air_damper_percentage_command=6,
            reheat_valve_percentage_command=7)
        obs = np.r_[obs, -0.2, 1.0, 1.0, 2.0]
        continuous = np.r_[continuous, 1.0, 1.0]
    return metadata, obs, (continuous, np.array([1, 1]))


class ThermalGreyBoxTest(unittest.TestCase):
    def test_coil_obeys_water_air_and_discharge_bounds(self):
        inputs = dict(flow=0.5, water_flow=0.001, water=350, supply=295)
        power = ThermalGreyBoxModel.coil_power(inputs)
        self.assertAlmostEqual(power, 1.2 * 1006 * 0.5 * (308.15 - 295))
        inputs['water_flow'] = 1e-6
        self.assertAlmostEqual(ThermalGreyBoxModel.coil_power(inputs),
                               1000 * 4180 * 1e-6 * (350 - 295))

    def test_no_heat_without_air_or_water(self):
        for air, water in [(0, 0.001), (0.5, 0)]:
            self.assertEqual(ThermalGreyBoxModel.coil_power(
                dict(flow=air, water_flow=water, water=350, supply=295)), 0)

    def test_pump_off_removes_water_flow(self):
        metadata, obs, action = water_case()
        model = ThermalGreyBoxModel(metadata)
        self.assertGreater(model.inputs(obs, action)[1]['water_flow'], 0)
        action[1][-1] = 0
        self.assertEqual(model.inputs(obs, action)[1]['water_flow'], 0)

    def test_header_zero_does_not_invent_competition(self):
        metadata, obs, action = water_case(two_rooms=True)
        for resistance in [0.0, 1.0]:
            model = ThermalGreyBoxModel(metadata, header_resistance=resistance)
            both = model.inputs(obs, action)[1]['water_flow']
            changed = (action[0].copy(), action[1].copy())
            changed[0][7] = -1
            one = model.inputs(obs, changed)[1]['water_flow']
            if resistance == 0:
                self.assertEqual(both, one)
            else:
                self.assertLess(both, one)

    def test_kv_units_are_cubic_metres_per_hour(self):
        metadata, obs, action = water_case()
        model = ThermalGreyBoxModel(metadata, water_kv=0.108)
        self.assertAlmostEqual(model.inputs(obs, action)[1]['water_flow'],
                               0.108 * np.sqrt(20) / 3600)

    def test_mass_is_causal_and_rollouts_do_not_change_fitted_state(self):
        metadata, obs, action = water_case()
        model = ThermalGreyBoxModel(metadata, fit_window=12)
        for step in range(30):
            current = obs.copy()
            current[0] += 0.001 * step
            model.update(current)
            model.set_action(*action)
        state = copy.deepcopy(model.mass)
        coefficients = model.coefficients[1].copy()
        temperature = model.zone_temp(current, 1)
        self.assertLess(model.mass[1], temperature)
        self.assertEqual(len(coefficients), 8)
        self.assertTrue(np.isfinite(model.predict(1, 12)))
        self.assertEqual(state, model.mass)
        np.testing.assert_array_equal(coefficients, model.coefficients[1])
        self.assertFalse(model.trusted(1))

    def test_disabling_both_ablations_matches_calibrated_parent(self):
        from smart_buildings.baselines.causal_greybox import CausalGreyBoxModel
        metadata, obs, action = water_case()
        parent = CausalGreyBoxModel(metadata, sensor_delay=1, calibrate_plant=True,
                                   fit_window=12)
        ablation = ThermalGreyBoxModel(metadata, physical_water=False, thermal_mass=False,
                                       fit_window=12)
        for step in range(30):
            current = obs.copy()
            current[0] += 0.001 * step
            for model in (parent, ablation):
                model.update(current)
                model.set_action(*action)
        self.assertAlmostEqual(parent.predict(1, 12), ablation.predict(1, 12))

    def test_cold_water_cools_air_instead_of_becoming_a_heat_source(self):
        inputs = dict(flow=0.5, water_flow=0.0001, water=285, supply=295)
        self.assertLess(ThermalGreyBoxModel.coil_power(inputs), 0)

    def test_mass_prior_and_hydraulic_parameters_are_validated(self):
        metadata, _, _ = water_case()
        for kwargs in [dict(mass_time_steps=0), dict(water_kv=-1),
                       dict(header_resistance=-1)]:
            with self.assertRaises(ValueError):
                ThermalGreyBoxModel(metadata, **kwargs)


if __name__ == '__main__':
    unittest.main()
