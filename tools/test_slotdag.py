"""P21-13: slot DAgger probe registration and MRO."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.probe_modes import (
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

    def test_teacher_import(self):
        from smart_buildings.baselines.occ_econdbhhot9 import (
            OccEconDbhHot9Policy)
        self.assertTrue(callable(OccEconDbhHot9Policy))


if __name__ == "__main__":
    unittest.main()
