"""P21-13: slot DAgger probe registration and MRO."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.probe_modes import (
    ResidPCProbe,
    ResidS05Probe,
    ResidS100Probe,
    ResidS30Probe,
    ResidS50Probe,
    ResidualSlotProbe,
    SlotDag1Probe,
    SlotDag2Probe,
    SlotDaggerProbe,
    SlotT1SoftProbe,
)


class TestSlotDag(unittest.TestCase):
    def test_registered(self):
        self.assertIn("probe_slot_dag", evaluate.POLICY_MAP)
        self.assertIs(evaluate.POLICY_MAP["probe_slot_dag"],
                      SlotDaggerProbe)

    def test_key(self):
        self.assertEqual(SlotDaggerProbe.KEY, "probe_slot_dag")

    def test_mro(self):
        self.assertTrue(issubclass(SlotDaggerProbe, SlotT1SoftProbe))
        # same heads as soft7 (inherits WEIGHTS)
        self.assertEqual(SlotDaggerProbe.WEIGHTS, SlotT1SoftProbe.WEIGHTS)

    def test_dag1_registered(self):
        self.assertIn("probe_slot_dag1", evaluate.POLICY_MAP)
        self.assertEqual(SlotDag1Probe.KEY, "probe_slot_dag1")
        self.assertTrue(SlotDag1Probe.WEIGHTS.endswith("slot_dag1.npz"))

    def test_dag2_registered(self):
        self.assertIn("probe_slot_dag2", evaluate.POLICY_MAP)
        self.assertEqual(SlotDag2Probe.KEY, "probe_slot_dag2")
        self.assertTrue(SlotDag2Probe.WEIGHTS.endswith("slot_dag2.npz"))

    def test_resid_registered(self):
        self.assertIn("probe_slot_resid", evaluate.POLICY_MAP)
        self.assertEqual(ResidualSlotProbe.KEY, "probe_slot_resid")
        self.assertEqual(ResidualSlotProbe.RESID_SCALE, 0.15)

    def test_scale_arms_registered(self):
        for key, cls, scale in (
                ("probe_slot_res05", ResidS05Probe, 0.05),
                ("probe_slot_res30", ResidS30Probe, 0.30),
                ("probe_slot_res50", ResidS50Probe, 0.50),
                ("probe_slot_res100", ResidS100Probe, 1.00)):
            self.assertIn(key, evaluate.POLICY_MAP)
            self.assertEqual(cls.RESID_SCALE, scale)
        self.assertIn("probe_slot_respc", evaluate.POLICY_MAP)
        self.assertEqual(ResidPCProbe.RESID_SCALES["press"], 0.0)
        self.assertEqual(ResidPCProbe.RESID_SCALES["sat"], 0.224)
        probe = ResidPCProbe.__new__(ResidPCProbe)
        self.assertEqual(probe._resid_scale("damp"), 0.129)
        base = ResidualSlotProbe.__new__(ResidualSlotProbe)
        self.assertEqual(base._resid_scale("damp"), 0.15)

    def test_teacher_import(self):
        from smart_buildings.baselines.occ_econdbhhot9 import (
            OccEconDbhHot9Policy)
        self.assertTrue(callable(OccEconDbhHot9Policy))


if __name__ == "__main__":
    unittest.main()
