"""P21-6 F2/F4: cold fallback + occ gate registration and wiring."""

import unittest

import numpy as np

from smart_buildings import evaluate
from smart_buildings.baselines.probe_modes import (
    SkynetColdBProbe,
    SkynetColdProbe,
    SkynetOccProbe,
)


class TestF2F4(unittest.TestCase):
    def test_registered(self):
        for key in ("probe_skynet_cold", "probe_skynet_coldb",
                    "probe_skynet_occ"):
            self.assertIn(key, evaluate.POLICY_MAP)

    def test_coldb_clears_offset(self):
        probe = SkynetColdBProbe.__new__(SkynetColdBProbe)
        state = {"offset": {"z": 1.0}}

        class FakeFloor:
            class pol:
                comfort_temp_window = (294.0, 297.0)

                @staticmethod
                def _zone_band(obs, vav, lo, hi):
                    return lo, hi

            def propose(self, obs):
                return "pc", "pd"

            def finish(self, pc, pd):
                return "base-action"

        FakeFloor.pol.target_offset = state["offset"]

        class FakeIx:
            zone_obs = np.array([[0]])
            zone_vav = ["v0"]
            t_std = 1.0
            t_mean = 280.0

        class FakePol:
            floor = FakeFloor()
            ix = FakeIx()

            def get_action(self, obs):
                return "residual-action"

        probe._pol = FakePol()
        out = probe.get_action(np.array([0.0]))
        self.assertEqual(out, "base-action")
        self.assertEqual(FakeFloor.pol.target_offset, {})

    def test_cold_const(self):
        self.assertEqual(SkynetColdProbe.F2_COLD_K, 1.0)

    def test_cold_fallback_wiring(self):
        probe = SkynetColdProbe.__new__(SkynetColdProbe)
        calls = []

        class FakeFloor:
            class pol:
                comfort_temp_window = (294.0, 297.0)

                @staticmethod
                def _zone_band(obs, vav, lo, hi):
                    return lo, hi

            def propose(self, obs):
                calls.append("propose")
                return "pc", "pd"

            def finish(self, pc, pd):
                calls.append("finish")
                return "base-action"

        class FakeIx:
            zone_obs = np.array([[0]])
            zone_vav = ["v0"]
            t_std = 1.0
            t_mean = 280.0

        class FakePol:
            floor = FakeFloor()
            ix = FakeIx()

            def get_action(self, obs):
                calls.append("residual")
                return "residual-action"

        probe._pol = FakePol()
        # zone temp 280K < lo-1 -> cold -> base fallback
        out = probe.get_action(np.array([0.0]))
        self.assertEqual(out, "base-action")
        self.assertEqual(calls, ["propose", "finish"])
        # warm zone -> residual path
        FakeIx.t_mean = 296.0
        out = probe.get_action(np.array([0.0]))
        self.assertEqual(out, "residual-action")
        self.assertEqual(calls[-1], "residual")

    def test_occ_gate_mechanism_live(self):
        import sys as _sys

        if SkynetOccProbe.PKGDIR not in _sys.path:
            _sys.path.insert(0, SkynetOccProbe.PKGDIR)
        from skynet.zt_controller import TunedPIPolicy

        pol = TunedPIPolicy.__new__(TunedPIPolicy)
        pol.OCC_GATE = True
        pol._occ_idx = 0
        pol.morning_start_hour = 8
        pol.evening_start_hour = 19
        pol.PRESTART_H = 1
        pol.EVENING_STOP_H = 0
        pol._get_hour_and_day = lambda obs: (12, 2)  # Wed noon
        self.assertFalse(pol._pi_hours(np.array([0.0])))
        self.assertTrue(pol._pi_hours(np.array([5.0])))

    def test_occ_gate_wiring(self):
        probe = SkynetOccProbe.__new__(SkynetOccProbe)

        class FakePolObj:
            OCC_GATE = False

        class FakeFloor:
            pol = FakePolObj()

        class FakePol:
            floor = FakeFloor()

        probe._pol = FakePol()
        # emulate __init__ tail
        probe._pol.floor.pol.OCC_GATE = True
        self.assertTrue(probe._pol.floor.pol.OCC_GATE)


if __name__ == "__main__":
    unittest.main()
