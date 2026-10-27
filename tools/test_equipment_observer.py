"""Independent adapter fixtures: decoding, observation delays and no lookahead."""

import unittest

import numpy as np

from smart_buildings.baselines.equipment_observer import (
    EquipmentAssumptions, EquipmentObserver, UnidentifiedActuatorError, component_zone_areas,
)


def observer_case():
    fields = []
    for room in (1, 2):
        fields += [f'VAV_{room}_{key}' for key in (
            'zone_air_temperature_sensor', 'supply_air_damper_percentage_command',
            'reheat_valve_percentage_command', 'supply_air_flowrate_setpoint')]
    fields += ['ahs_ahu_1_' + key for key in (
        'outside_air_temperature_sensor', 'supply_air_temperature_sensor',
        'supply_air_static_pressure_sensor', 'recirculation_percentage_command',
        'supervisor_run_command', 'supply_air_flowrate_sensor',
        'supply_fan_speed_percentage_command')]
    fields += ['hws_supply_water_temperature_sensor', 'hws_differential_pressure_sensor']
    bounds = {
        'ahu_1_supply_air_temperature_setpoint': (285, 305),
        'ahu_1_supply_air_static_pressure_setpoint': (0, 2000),
        'ahu_1_recirculation_percentage_command': (0, 1),
        'supply_water_temperature_setpoint': (310, 350),
        'differential_pressure_setpoint': (0, 20),
        'supply_air_damper_percentage_command': (0, 1),
        'reheat_valve_percentage_command': (0, 1),
    }
    meta = dict(n_continuous_action=9, n_discrete_action=2,
                observation_field_names=fields,
                observation_normalization_metadata={
                    'zone_air_temperature_sensor': dict(sample_mean=295, sample_std=5),
                    'supply_air_damper_percentage_command': dict(sample_mean=.5, sample_std=.5),
                    'reheat_valve_percentage_command': dict(sample_mean=.5, sample_std=.5)},
                action_normalization_metadata={key: dict(
                    min_native_value=lo, max_native_value=hi,
                    min_normalized_value=-1, max_normalized_value=1)
                    for key, (lo, hi) in bounds.items()},
                zone_action_indices={
                    'VAV_1': dict(supply_air_damper_percentage_command=5,
                                  reheat_valve_percentage_command=6),
                    'VAV_2': dict(supply_air_damper_percentage_command=7,
                                  reheat_valve_percentage_command=8)},
                ahu_zone_assignments={'ahu_1': ['room_1', 'room_2']},
                temp_indices=[0], pressure_indices=[1, 4], recirculation_indices=[2],
                water_indices=[3])
    obs = np.array([-0.2, 1, 1, 2, 0.2, 1, 1, 2,
                    254, 305, 2000, .7, 1, 4, .2, 340, 20.])
    action = (np.array([1, 1, .4, 1, 1, 1, 1, 1, 1.]), np.array([1, 1]))
    return meta, obs, action


class EquipmentObserverTest(unittest.TestCase):
    def make(self, **kwargs):
        meta, obs, action = observer_case()
        return EquipmentObserver(meta, assumptions=EquipmentAssumptions(), **kwargs), obs, action

    def test_decoding_setpoint_and_return_air_weights(self):
        observer, obs, action = self.make(zone_areas={1: 1, 2: 3})
        observer.observe(obs)
        estimate = observer.project_action(*action)
        # Native zone temperatures: 294,296; weighted return:295.5.
        self.assertAlmostEqual(estimate.zones[1].supply_temperature, .7 * 295.5 + .3 * 254)
        self.assertNotEqual(estimate.zones[1].supply_temperature, 305)
        self.assertEqual(estimate.temperature_basis, 'lagged_observation_proxy')

    def test_capacity_and_new_command_flow_cap(self):
        observer, obs, action = self.make()
        obs[13:15] = [1, 1]  # independently reported capacity = 1 m3/s
        observer.observe(obs)
        self.assertEqual(observer.project_action(*action).handler_flows[1], 1)

    def test_unobserved_capacity_is_not_assumed_infinite(self):
        observer, obs, action = self.make()
        obs[13:15] = 0
        observer.observe(obs)
        with self.assertRaises(ValueError):
            observer.project_action(*action)

    def test_first_unseen_closing_limit_is_marked_not_guessed(self):
        observer, obs, action = self.make()
        observer.observe(obs)
        action[0][5] = action[0][7] = -1
        estimate = observer.project_action(*action)
        self.assertEqual(set(estimate.uncertain_damper_floors), {1, 2})
        with self.assertRaises(UnidentifiedActuatorError):
            observer.project_action(*action, require_identified_floors=True)
        observer.set_action(*action)
        following = obs.copy()
        following[[1, 5]] = -.8
        following[13:15] = [.4, .02]
        observer.observe(following)
        identified = observer.project_action(*action, require_identified_floors=True)
        self.assertFalse(identified.uncertain_damper_floors)

    def test_actuator_floors_are_learned_from_only_past_pairs(self):
        observer, obs, action = self.make()
        observer.observe(obs)
        action[0][1] = action[0][5] = action[0][7] = -1
        observer.set_action(*action)
        following = obs.copy()
        following[[1, 5]] = -.8  # native damper 0.1
        following[10] = 200
        following[13] = .4 / np.sqrt(10)
        following[14] = following[13] / 20
        observer.observe(following)
        self.assertAlmostEqual(observer.damper_floors[1], .1)
        self.assertEqual(observer.pressure_floors[1], 200)
        self.assertGreater(observer.project_action(*action).handler_flows[1], 0)

    def test_pending_transition_uses_old_weather_water_and_current_latched_temperature(self):
        observer, obs, action = self.make()
        self.assertIsNone(observer.observe(obs))
        observer.set_action(*action)
        following = obs.copy()
        following[8] = 280
        following[15] = 342.5
        self.assertIsNone(observer.observe(following))
        pending = observer.pending
        self.assertEqual(pending.control_step, 0)
        self.assertEqual(pending.outdoor_temperature[1], 254)
        self.assertEqual(pending.water_temperature, 340)
        action[1][:] = 0
        observer.set_action(*action)
        newer = following.copy()
        newer[0] += .02
        newer[12:15] = 0
        newer[16] = 0
        transition = observer.observe(newer)
        self.assertIs(transition.forcing, pending)
        self.assertEqual((transition.start_observation, transition.end_observation), (1, 2))
        self.assertAlmostEqual(transition.end_temperatures[1] - transition.start_temperatures[1], .1)

    def test_candidate_does_not_change_pending_or_recorded_action(self):
        observer, obs, action = self.make()
        observer.observe(obs)
        observer.set_action(*action)
        observer.observe(obs)
        observer.set_action(*action)
        pending, recorded = observer.pending, observer.last_action
        off = action[1].copy()
        off[:] = 0
        candidate = observer.project_action(action[0], off, zone_temperatures={1: 295, 2: 295})
        self.assertEqual(candidate.zones[1].total_power, 0)
        self.assertIs(observer.pending, pending)
        self.assertIs(observer.last_action, recorded)
        self.assertEqual(candidate.temperature_basis, 'provided_state')
        with self.assertRaises(TypeError):
            pending.zones[1] = candidate.zones[1]

    def test_water_sensor_prediction_uses_only_learned_rates(self):
        observer, obs, action = self.make()
        observer.observe(obs)
        self.assertEqual(observer.project_water_sensor(action[0]), 340)
        observer.set_action(*action)
        following = obs.copy()
        following[15] += 2.5
        observer.observe(following)
        self.assertEqual(observer.project_water_sensor(action[0]), 345)

    def test_geometry_is_connected_components_not_material_counts(self):
        material = np.full((5, 8), 2)
        material[1:3, 1:3] = 0
        material[1:4, 5:7] = 0
        areas = component_zone_areas(material, [1, 2])
        self.assertEqual(sorted(areas.values()), [4, 6])
        with self.assertRaises(ValueError):
            component_zone_areas(material, [1, 2, 3])

    def test_shapes_ranges_and_topology_fail_explicitly(self):
        observer, obs, action = self.make()
        with self.assertRaises(ValueError):
            observer.observe(obs[:-1])
        observer.observe(obs)
        with self.assertRaises(ValueError):
            observer.project_action(np.full(9, 2.), action[1])
        meta, _, _ = observer_case()
        meta['ahu_zone_assignments']['ahu_1'].append('room_1')
        with self.assertRaises(ValueError):
            EquipmentObserver(meta, assumptions=EquipmentAssumptions())


if __name__ == '__main__':
    unittest.main()
