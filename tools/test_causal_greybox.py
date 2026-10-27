"""Small synthetic tests; no simulator rollouts or result artifacts required."""

import unittest

import numpy as np

from smart_buildings.baselines.causal_greybox import CausalGreyBoxModel


def case():
    names = ['VAV_1_zone_air_temperature_sensor',
             'VAV_1_supply_air_damper_percentage_command',
             'VAV_1_reheat_valve_percentage_command',
             'VAV_1_supply_air_flowrate_setpoint',
             'ahs_ahu_1_outside_air_temperature_sensor',
             'ahs_ahu_1_supply_air_temperature_sensor',
             'ahs_ahu_1_supervisor_run_command',
             'ahs_ahu_1_supply_air_static_pressure_sensor',
             'ahs_ahu_1_recirculation_percentage_command',
             'ahs_ahu_1_supply_air_flowrate_sensor',
             'ahs_ahu_1_supply_fan_speed_percentage_command',
             'hws_supply_water_temperature_sensor', 'comfort_mode_now']
    limits = {'ahu_1_supply_air_temperature_setpoint': (285, 305),
              'ahu_1_supply_air_static_pressure_setpoint': (0, 2000),
              'ahu_1_recirculation_percentage_command': (0, 1),
              'supply_air_damper_percentage_command': (0, 1),
              'reheat_valve_percentage_command': (0, 1)}
    metadata = dict(observation_field_names=names,
                    observation_normalization_metadata={},
                    ahu_zone_assignments={'ahu_1': ['room_1']},
                    zone_action_indices={'VAV_1': {
                        'supply_air_damper_percentage_command': 3,
                        'reheat_valve_percentage_command': 4}},
                    temp_indices=[0], pressure_indices=[1], recirculation_indices=[2],
                    action_normalization_metadata={k: dict(
                        min_native_value=lo, max_native_value=hi,
                        min_normalized_value=-1, max_normalized_value=1)
                        for k, (lo, hi) in limits.items()})
    obs = np.array([-0.2, 1, -1, 2, 254, 305, 1, 2000, 0.7, 2, 0.1, 340, 0.])
    action = (np.array([1., 1., 0.4, 1., -1.]), np.array([1]))
    return metadata, obs, action


class CausalGreyBoxTest(unittest.TestCase):
    def test_suffix_cache_matches_original_field_lookup(self):
        from smart_buildings.baselines.greybox import GreyBoxModel
        metadata, _, _ = case()
        model = CausalGreyBoxModel(metadata)
        for suffix in ['_outside_air_temperature_sensor',
                       'VAV_1_supply_air_flowrate_setpoint', '_missing_field']:
            expected = GreyBoxModel._first_suffix(model, suffix)
            self.assertEqual(model._first_suffix(suffix), expected)
            self.assertEqual(model._first_suffix(suffix), expected)

    def test_sensor_is_not_discharge_temperature(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata)
        self.assertAlmostEqual(model.inputs(obs, action)[1]['supply'], 282)

    def test_pressure_and_run_gate_airflow(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata)
        full = model.inputs(obs, action)[1]['flow']
        obs[7] = 0
        action[0][1] = -0.5  # 500 Pa instead of 2000 Pa
        self.assertAlmostEqual(model.inputs(obs, action)[1]['flow'], full / 2)
        action[1][0] = 0
        self.assertEqual(model.inputs(obs, action)[1]['flow'], 0)

    def test_training_uses_preceding_action(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata, fit_window=2)
        model.update(obs)
        expected = model.features(model.zone_temp(obs, 1), 0,
                                  model.inputs(obs, action)[1])
        model.set_action(*action)
        action[0][:] = -1  # stored commands must be copied, not aliased
        following = obs.copy()
        following[0] += 0.02
        model.update(following)
        np.testing.assert_allclose(model.history[1][0][0], expected)
        self.assertAlmostEqual(model.history[1][0][1], 0.1)

    def test_reheat_proxy_is_capped(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata)
        inputs = model.inputs(obs, action)[1]
        inputs.update(supply=300, water=350, reheat=1, flow=2)
        features = model.features(294, 0, inputs)
        self.assertAlmostEqual(features[3], 2 * 8.15 / 10)

    def test_fit_is_finite_but_not_trusted_for_control(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata, fit_window=12)
        for step in range(30):
            current = obs.copy()
            current[0] += step * 0.001
            model.update(current)
            model.set_action(*action)
        self.assertTrue(np.isfinite(model.predict(1, 12)))
        self.assertFalse(model.trusted(1))
        with self.assertRaises(NotImplementedError):
            model.steps_to_reach(1, 295, 12)

    def test_delayed_sensor_uses_older_command(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata, fit_window=12, sensor_delay=1)
        model.update(obs)
        model.set_action(*action)
        following = obs.copy()
        following[0] += 0.02
        model.update(following)
        expected = model.features(model.zone_temp(following, 1), 0.1,
                                  model.inputs(following, action)[1])
        changed = action[0].copy()
        changed[3] = -1
        model.set_action(changed, action[1])
        model.update(following)
        np.testing.assert_allclose(model.history[1][-1][0], expected)

    def test_calibration_learns_floor_from_past_commands(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata, calibrate_plant=True, sensor_delay=1)
        model.update(obs)
        action[0][3] = -1
        action[0][1] = -1
        model.set_action(*action)
        following = obs.copy()
        following[1] = -0.8  # effective damper 0.1 despite command 0
        following[7] = 50   # effective fan floor despite command 0 Pa
        model.update(following)
        model.set_action(*action)
        self.assertAlmostEqual(model.damper_floor[1], 0.1)
        self.assertEqual(model.pressure_floor[1], 50)
        self.assertGreater(model.inputs(following, action)[1]['flow'], 0)

    def test_online_water_rate_calibration(self):
        metadata, obs, action = case()
        metadata['water_indices'] = [5]
        metadata['action_normalization_metadata']['supply_water_temperature_setpoint'] = dict(
            min_native_value=310, max_native_value=350,
            min_normalized_value=-1, max_normalized_value=1)
        command = np.r_[action[0], 1.0]
        model = CausalGreyBoxModel(metadata, calibrate_plant=True)
        model.update(obs)
        model.set_action(command, action[1])
        following = obs.copy()
        following[11] += 2.5
        model.update(following)
        self.assertEqual(model.water_heating_step, 2.5)
        command[-1] = -1
        model.set_action(command, action[1])
        following[11] -= 0.5
        model.update(following)
        self.assertEqual(model.water_cooling_step, 0.5)

    def test_pending_sensor_prediction_cannot_see_new_command(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata, sensor_delay=1)
        model.update(obs)
        model.set_action(*action)
        model.update(obs)
        model.coefficients[1][2] = 0.1
        on = (action[0].copy(), np.array([1]))
        off = (action[0].copy(), np.array([0]))
        self.assertEqual(model.predict_action_path([on], 1),
                         model.predict_action_path([off], 1))
        self.assertNotEqual(model.predict_action_path([on], 2)[-1],
                            model.predict_action_path([off], 2)[-1])

    def test_candidate_change_invalidates_calibrated_forecast_cache(self):
        metadata, obs, action = case()
        model = CausalGreyBoxModel(metadata, sensor_delay=1, calibrate_plant=True)
        model.update(obs)
        model.set_action(*action)
        model.update(obs)
        model.coefficients[1][2] = 0.1
        model.set_action(action[0], np.array([1]))
        on = model.predict(1, 2)
        model.set_action(action[0], np.array([0]))
        self.assertNotEqual(on, model.predict(1, 2))


if __name__ == '__main__':
    unittest.main()
