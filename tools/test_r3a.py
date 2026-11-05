"""P20-8 R3a: menu arm consts, registration, neutral-default parity."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.occ_econ import OccEconMixin
from smart_buildings.baselines.probe_modes import (
    R3aHwsProbe,
    R3aPcapProbe,
    R3aRecProbe,
    R3aSatProbe,
    _PressureFloorMixin,
)


class TestR3aMenu(unittest.TestCase):
    def test_registered(self):
        for key in ("probe_r3a_sat", "probe_r3a_pcap", "probe_r3a_rec",
                    "probe_r3a_hws"):
            self.assertIn(key, evaluate.POLICY_MAP)

    def test_arm_consts(self):
        self.assertEqual(R3aSatProbe.MILD_SAT, -1.0)
        self.assertEqual(R3aPcapProbe.PRESSURE_FLOOR, -0.6)
        self.assertEqual(R3aRecProbe.ECON_FOUT_HI, 0.7)
        self.assertEqual(R3aHwsProbe.WATER_SCALE, 0.7)

    def test_neutral_defaults_untouched(self):
        occ9 = evaluate.POLICY_MAP["occ_econdbhhot9"]
        self.assertEqual(occ9.MILD_SAT, -0.4)
        self.assertEqual(occ9.WATER_SCALE, 1.0)
        self.assertEqual(OccEconMixin.ECON_FOUT_LO, 0.3)
        self.assertEqual(OccEconMixin.ECON_FOUT_HI, 1.0)
        self.assertEqual(_PressureFloorMixin.PRESSURE_FLOOR, -1.0)

    def test_each_arm_touches_one_knob(self):
        occ9 = evaluate.POLICY_MAP["occ_econdbhhot9"]
        arms = (
            (R3aSatProbe, "MILD_SAT"),
            (R3aRecProbe, "ECON_FOUT_HI"),
            (R3aHwsProbe, "WATER_SCALE"),
        )
        knobs = ("MILD_SAT", "ECON_FOUT_HI", "WATER_SCALE")
        for arm, own in arms:
            for knob in knobs:
                if knob == own:
                    self.assertNotEqual(getattr(arm, knob),
                                        getattr(occ9, knob))
                else:
                    self.assertEqual(getattr(arm, knob),
                                     getattr(occ9, knob))
        self.assertNotIn("PRESSURE_FLOOR", dir(occ9))

    def test_pcap_mixin_sits_above_occ9(self):
        mro = list(R3aPcapProbe.__mro__)
        occ9 = evaluate.POLICY_MAP["occ_econdbhhot9"]
        self.assertLess(mro.index(_PressureFloorMixin), mro.index(occ9))


if __name__ == "__main__":
    unittest.main()
