"""Focused behavior tests for the occupancy-gated gen-3.1 chain."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from smart_buildings.baselines.occ_gated import OccGatedPolicy, OccSatPolicy
from smart_buildings.baselines.occ_gated2 import OccGated2Policy, OccSat2Policy
from smart_buildings.baselines.occ_gated3 import OccGated3Policy, OccSat3Policy
from smart_buildings.baselines.occ_duty import OccDutyPolicy, OccDutyLeanPolicy
from smart_buildings.baselines.occ_gated4 import OccGated4Policy


def _fake_building():
    fields = [
        "VAV_1_zone_air_temperature_sensor",
        "VAV_2_zone_air_temperature_sensor",
        "VAV_1_zone_air_heating_temperature_setpoint_sensor",
        "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
        "VAV_2_zone_air_heating_temperature_setpoint_sensor",
        "VAV_2_zone_air_cooling_temperature_setpoint_sensor",
        "hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
        "num_occupants", "comfort_mode_now", "comfort_mode_soon",
        "ahs_ahu_1_outside_air_temperature_sensor",
    ]
    meta = {
        "comfort_temp_window": (294.0, 297.0),
        "eco_temp_window": (289.0, 298.0),
        "morning_start_hour": 6,
        "evening_start_hour": 19,
        "n_discrete_action": 2,
        "n_continuous_action": 9,
        "temp_indices": [0],
        "pressure_indices": [1, 3],
        "water_indices": [2],
        "recirculation_indices": [4],
        "zone_action_indices": {
            "VAV_1": {"supply_air_damper_percentage_command": 5,
                      "reheat_valve_percentage_command": 6},
            "VAV_2": {"supply_air_damper_percentage_command": 7,
                      "reheat_valve_percentage_command": 8},
        },
        "observation_field_names": fields,
        "ahu_zone_assignments": {"ahu_1": ["room_1", "room_2"]},
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295.0, "sample_std": 5.0},
        },
    }
    spec = {
        "continuous_action": SimpleNamespace(low=-np.ones(9), high=np.ones(9)),
        "discrete_action": SimpleNamespace(low=np.zeros(2), high=np.ones(2)),
    }
    return meta, spec, {name: i for i, name in enumerate(fields)}


def _obs(index, hour, day, zone_k=295.0, occ=0.0, now=0.0, soon=0.0, oat_k=280.0):
    obs = np.zeros(len(index), dtype=np.float32)
    for vav in ("VAV_1", "VAV_2"):
        obs[index[f"{vav}_zone_air_temperature_sensor"]] = (zone_k - 295.0) / 5.0
        obs[index[f"{vav}_zone_air_heating_temperature_setpoint_sensor"]] = 294.0
        obs[index[f"{vav}_zone_air_cooling_temperature_setpoint_sensor"]] = 297.0
    obs[index["hod_sin_000"]] = math.sin(2 * math.pi * hour / 24)
    obs[index["hod_cos_000"]] = math.cos(2 * math.pi * hour / 24)
    obs[index["dow_sin_000"]] = math.sin(2 * math.pi * day / 7)
    obs[index["dow_cos_000"]] = math.cos(2 * math.pi * day / 7)
    obs[index["num_occupants"]] = occ
    obs[index["comfort_mode_now"]] = now
    obs[index["comfort_mode_soon"]] = soon
    obs[index["ahs_ahu_1_outside_air_temperature_sensor"]] = oat_k
    return obs


class OccGatedTest(unittest.TestCase):
    def _policy(self, cls=OccGatedPolicy):
        meta, spec, index = _fake_building()
        return cls(meta, spec), index

    def test_night_empty_shuts_down(self):
        policy, index = self._policy()
        action = policy.get_action(_obs(index, 2, 2, zone_k=292.0))
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])
        self.assertEqual(action["continuous_action"][2], -1.0)  # water cold
        self.assertEqual(action["continuous_action"][5], -1.0)  # damper min

    def test_occupied_comfort_runs(self):
        policy, index = self._policy()
        action = policy.get_action(
            _obs(index, 10, 2, zone_k=292.0, occ=10.0, now=1.0))
        self.assertEqual(action["discrete_action"][0], 1)  # AHU on

    def test_holiday_shuts_down_after_grace(self):
        policy, index = self._policy()
        obs = _obs(index, 10, 0, zone_k=292.0, occ=0.0, now=1.0)
        for _ in range(OccGatedPolicy.EMPTY_GRACE_STEPS + 1):
            action = policy.get_action(obs)
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])

    def test_departed_afternoon_shuts_down(self):
        policy, index = self._policy()
        policy.get_action(_obs(index, 10, 2, zone_k=295.0, occ=10.0, now=1.0))
        action = policy.get_action(
            _obs(index, 15, 2, zone_k=295.0, occ=0.0, now=1.0))
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])

    def test_prestart_runs_before_comfort(self):
        policy, index = self._policy()
        # 05:00 Monday, zones cold, comfort within the hour.
        action = policy.get_action(
            _obs(index, 5, 0, zone_k=288.0, soon=1.0, oat_k=250.0))
        self.assertEqual(action["discrete_action"][0], 1)

    def test_occupied_late_night_runs(self):
        policy, index = self._policy()
        action = policy.get_action(
            _obs(index, 2, 2, zone_k=286.0, occ=3.0))
        self.assertEqual(action["discrete_action"][0], 1)

    def test_normalized_occupancy_decodes_from_running_min(self):
        policy, index = self._policy()
        # Normalized signal: empty reads the running minimum; occupancy is
        # anything more than OCC_EPS above it.
        policy.get_action(_obs(index, 2, 2, zone_k=292.0, occ=-0.9))
        action = policy.get_action(_obs(index, 2, 2, zone_k=292.0, occ=-0.9))
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])
        action = policy.get_action(
            _obs(index, 10, 2, zone_k=292.0, occ=-0.2, now=1.0))
        self.assertEqual(action["discrete_action"][0], 1)


class OccSatTest(unittest.TestCase):
    def test_saturation_caps_heating_dampers(self):
        meta, spec, index = _fake_building()
        policy = OccSatPolicy(meta, spec)
        policy._saturated = True
        policy._zone_u = {"VAV_1": -0.8, "VAV_2": 0.1}
        continuous = np.zeros(9, dtype=np.float32)
        zones = [("VAV_1", 0, meta["zone_action_indices"]["VAV_1"]),
                 ("VAV_2", 1, meta["zone_action_indices"]["VAV_2"])]
        obs = _obs(index, 10, 2, zone_k=288.0, occ=10.0, now=1.0)
        policy._pi_active = True
        policy._zone_integral = {}
        policy._max_heat = 0.0
        policy._command_zones(continuous, zones, obs, 294.0, 297.0)
        heat_damper = (float(continuous[5]) + 1.0) * 0.5
        self.assertLessEqual(heat_damper, policy.SAT_DAMPER_CAP + 1e-6)


class OccGated2Test(unittest.TestCase):
    def _policy(self, cls=OccGated2Policy):
        meta, spec, index = _fake_building()
        return cls(meta, spec), index

    def test_cold_floor_raises_water_while_heating(self):
        policy, index = self._policy()
        obs = _obs(index, 10, 2, zone_k=292.0, occ=10.0, now=1.0, oat_k=250.0)
        action = policy.get_action(obs)
        frac = (float(action["continuous_action"][2]) + 1.0) * 0.5
        self.assertGreaterEqual(frac, 0.7)

    def test_mild_weather_leaves_water_reset(self):
        policy, index = self._policy()
        obs = _obs(index, 10, 2, zone_k=293.5, occ=10.0, now=1.0, oat_k=285.0)
        action = policy.get_action(obs)
        frac = (float(action["continuous_action"][2]) + 1.0) * 0.5
        self.assertLess(frac, 0.7)

    def test_no_heat_no_floor(self):
        policy, index = self._policy()
        obs = _obs(index, 10, 2, zone_k=296.0, occ=10.0, now=1.0, oat_k=250.0)
        action = policy.get_action(obs)
        frac = (float(action["continuous_action"][2]) + 1.0) * 0.5
        self.assertAlmostEqual(frac, 0.0, places=2)

    def test_occ_sat2_carries_damper_cap(self):
        meta, spec, index = _fake_building()
        policy = OccSat2Policy(meta, spec)
        self.assertEqual(policy.SAT_DAMPER_CAP, 0.5)
        policy._saturated = True
        policy._zone_u = {"VAV_1": -0.8, "VAV_2": 0.1}
        continuous = np.zeros(9, dtype=np.float32)
        zones = [("VAV_1", 0, meta["zone_action_indices"]["VAV_1"]),
                 ("VAV_2", 1, meta["zone_action_indices"]["VAV_2"])]
        obs = _obs(index, 10, 2, zone_k=288.0, occ=10.0, now=1.0)
        policy._pi_active = True
        policy._zone_integral = {}
        policy._max_heat = 0.0
        policy._command_zones(continuous, zones, obs, 294.0, 297.0)
        heat_damper = (float(continuous[5]) + 1.0) * 0.5
        self.assertLessEqual(heat_damper, 0.5 + 1e-6)


class OccGated3Test(unittest.TestCase):
    def _policy(self, cls=OccGated3Policy):
        meta, spec, index = _fake_building()
        return cls(meta, spec), index

    def test_prestart_latches_until_comfort(self):
        policy, index = self._policy()
        # Cold pre-dawn: prestart triggers and runs the AHU.
        action = policy.get_action(
            _obs(index, 5, 0, zone_k=288.0, soon=1.0, oat_k=250.0))
        self.assertEqual(action["discrete_action"][0], 1)
        # Zones recover to in-band with arrival still 55 min out: a
        # threshold re-evaluation would shut down here (need < steps left).
        need = policy._needed_prestart(294.0)
        self.assertLess(need, 12)
        action = policy.get_action(
            _obs(index, 5, 0, zone_k=294.0, soon=1.0, oat_k=250.0))
        self.assertEqual(action["discrete_action"][0], 1)

    def test_floor_bounded_by_eco_band(self):
        policy, index = self._policy()
        policy._rate_heat = 0.0005  # slowest: computed floor ~ band edge
        floor_heat, _ = policy._floor_window()
        self.assertLessEqual(floor_heat, 289.0 - 1.0)

    def test_floor_hysteresis(self):
        policy, index = self._policy()
        policy._rate_heat = 0.02
        floor_heat, _ = policy._floor_window()
        # Deep breach on a weekend night (no prestart possible) -> heat.
        action = policy.get_action(
            _obs(index, 2, 6, zone_k=floor_heat - 1.0, oat_k=250.0))
        self.assertEqual(policy._floor_mode, "heat")
        # Recovered halfway into the deadband -> still heating (no chatter).
        action = policy.get_action(
            _obs(index, 2, 6, zone_k=floor_heat - 0.2, oat_k=250.0))
        self.assertEqual(policy._floor_mode, "heat")
        # Above floor + exit deadband -> shutdown.
        action = policy.get_action(
            _obs(index, 2, 6, zone_k=floor_heat + 0.6, oat_k=250.0))
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])

    def test_occ_sat3_imports(self):
        meta, spec, _ = _fake_building()
        policy = OccSat3Policy(meta, spec)
        self.assertEqual(policy.KEY, "occ_sat3")

    def test_rates_learn_only_in_prestart(self):
        policy, index = self._policy()
        before = policy._rate_heat
        policy._max_heat = 0.9
        # Occupied steady-state micro-rise: must not pollute the rate.
        policy._update_rates(294.0)
        policy._update_rates(294.001)
        self.assertEqual(policy._rate_heat, before)
        # Same rise during prestart: learned (EMA moves toward 0.05).
        policy._prestart_active = True
        policy._update_rates(294.0)
        policy._update_rates(294.05)
        self.assertAlmostEqual(policy._rate_heat, 0.8 * before + 0.2 * 0.05)


class OccDutyTest(unittest.TestCase):
    def _policy(self, cls=OccDutyPolicy):
        meta, spec, index = _fake_building()
        return cls(meta, spec), index

    def test_cycles_off_in_band(self):
        policy, index = self._policy()
        obs = _obs(index, 10, 2, zone_k=295.5, occ=10.0, now=1.0)
        runs = [policy.get_action(obs)["discrete_action"][0]
                for _ in range(policy.DUTY_ON_STEPS + policy.DUTY_OFF_STEPS)]
        self.assertEqual(runs[:policy.DUTY_ON_STEPS], [1] * policy.DUTY_ON_STEPS)
        self.assertEqual(runs[policy.DUTY_ON_STEPS:], [0] * policy.DUTY_OFF_STEPS)

    def test_breach_restarts_on_phase(self):
        policy, index = self._policy()
        in_band = _obs(index, 10, 2, zone_k=295.5, occ=10.0, now=1.0)
        for _ in range(policy.DUTY_ON_STEPS):
            policy.get_action(in_band)
        # Next step would coast, but a zone leaves the inner band.
        cold = _obs(index, 10, 2, zone_k=294.5, occ=10.0, now=1.0)
        action = policy.get_action(cold)
        self.assertEqual(action["discrete_action"][0], 1)

    def test_no_cycling_out_of_band(self):
        policy, index = self._policy()
        obs = _obs(index, 10, 2, zone_k=292.0, occ=10.0, now=1.0)
        for _ in range(policy.DUTY_ON_STEPS + policy.DUTY_OFF_STEPS + 2):
            action = policy.get_action(obs)
            self.assertEqual(action["discrete_action"][0], 1)

    def test_lean_variant(self):
        meta, spec, _ = _fake_building()
        policy = OccDutyLeanPolicy(meta, spec)
        self.assertEqual((policy.DUTY_ON_STEPS, policy.DUTY_OFF_STEPS), (4, 4))


class OccGated4Test(unittest.TestCase):
    def test_water_floor_disabled(self):
        meta, spec, index = _fake_building()
        policy = OccGated4Policy(meta, spec)
        obs = _obs(index, 10, 2, zone_k=293.5, occ=10.0, now=1.0, oat_k=250.0)
        action = policy.get_action(obs)
        frac = (float(action["continuous_action"][2]) + 1.0) * 0.5
        # Round-2/3 would pin the floor (~0.77 here); round-4 leaves the
        # chain's reset value (below the floor for a small shortfall).
        self.assertLess(frac, 0.77)


if __name__ == "__main__":
    unittest.main()
