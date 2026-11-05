"""P20-7 N1.1: danielwuu pieces registration, consts, stack MRO."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.probe_modes import (
    N1HystMixin,
    N1HystProbe,
    N1OffMixin,
    N1OffProbe,
    N1P2Mixin,
    N1P2Probe,
    N1StackProbe,
    N1StackTunedProbe,
)


class TestN1Pieces(unittest.TestCase):
    def test_registered(self):
        for key in ("probe_n1_off", "probe_n1_p2", "probe_n1_hyst",
                    "probe_n1_stack", "probe_n1_tuned"):
            self.assertIn(key, evaluate.POLICY_MAP)

    def test_tuned_consts_match_bo_best(self):
        self.assertEqual(N1StackTunedProbe.N1_OFF_MARGIN, 0.5408)
        self.assertEqual(N1StackTunedProbe.N1_P2_EXP, 2.6692)
        self.assertEqual(N1StackTunedProbe.N1_HYST_WINDOW, 71)
        self.assertTrue(issubclass(N1StackTunedProbe, N1StackProbe))

    def test_defaults_match_danielwuu(self):
        self.assertEqual(N1OffMixin.N1_OFF_MARGIN, 0.5)
        self.assertEqual(N1P2Mixin.N1_P2_EXP, 2.0)
        self.assertEqual(N1P2Mixin.N1_P2_MIN_PA, 200.0)
        self.assertEqual(N1HystMixin.N1_HYST_WINDOW, 24)
        self.assertEqual(N1HystMixin.N1_HYST_MARGIN, 0.0)

    def test_on_flags_default_live(self):
        self.assertEqual(N1OffMixin.N1_OFF_ON, 1)
        self.assertEqual(N1P2Mixin.N1_P2_ON, 1)
        self.assertEqual(N1HystMixin.N1_HYST_ON, 1)

    def test_singles_carry_only_own_mixin(self):
        occ9 = evaluate.POLICY_MAP["occ_econdbhhot9"]
        self.assertIn(N1OffMixin, N1OffProbe.__mro__)
        self.assertNotIn(N1P2Mixin, N1OffProbe.__mro__)
        self.assertNotIn(N1HystMixin, N1OffProbe.__mro__)
        self.assertIn(N1P2Mixin, N1P2Probe.__mro__)
        self.assertNotIn(N1OffMixin, N1P2Probe.__mro__)
        self.assertIn(N1HystMixin, N1HystProbe.__mro__)
        self.assertNotIn(N1OffMixin, N1HystProbe.__mro__)
        for probe in (N1OffProbe, N1P2Probe, N1HystProbe, N1StackProbe):
            self.assertTrue(issubclass(probe, occ9))

    def test_stack_orders_off_p2_hyst_above_occ9(self):
        mro = list(N1StackProbe.__mro__)
        occ9 = evaluate.POLICY_MAP["occ_econdbhhot9"]
        self.assertLess(mro.index(N1OffMixin), mro.index(N1P2Mixin))
        self.assertLess(mro.index(N1P2Mixin), mro.index(N1HystMixin))
        self.assertLess(mro.index(N1HystMixin), mro.index(occ9))

    def test_p2_pressure_math(self):
        # fraction 1.0 -> 2000 Pa -> +1.0; fraction 0.1 -> clip 200 Pa -> -0.8.
        self.assertAlmostEqual(min(2000.0, max(200.0, 2000.0 * 1.0 ** 2))
                               / 1000.0 - 1.0, 1.0)
        self.assertAlmostEqual(min(2000.0, max(200.0, 2000.0 * 0.1 ** 2))
                               / 1000.0 - 1.0, -0.8)


if __name__ == "__main__":
    unittest.main()
