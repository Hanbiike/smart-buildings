"""P22-4: DMAX probe registration + map check."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.occ_dmax1 import OccDmax9Policy
from smart_buildings.baselines.occ_dualmax import DualMaxOutputMixin
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)


class TestDmax1(unittest.TestCase):
    def test_registered(self):
        self.assertIn("occ_dmax9", evaluate.POLICY_MAP)
        self.assertIs(evaluate.POLICY_MAP["occ_dmax9"], OccDmax9Policy)

    def test_mro(self):
        self.assertTrue(issubclass(OccDmax9Policy, DualMaxOutputMixin))
        self.assertTrue(issubclass(OccDmax9Policy,
                                   OccEconDbhHot9Policy))

    def test_map_staging(self):
        probe = OccDmax9Policy.__new__(OccDmax9Policy)
        probe._dm_on = 1
        probe._dm_split = 0.5
        probe._dm_heat_min = 0.1
        # heating demand below split: min airflow, partial valve
        damper, valve = DualMaxOutputMixin._zone_output_map(probe, -0.25)
        self.assertAlmostEqual(damper, 0.1)
        self.assertAlmostEqual(valve, 0.5)
        # heating demand above split: air opens, valve pinned
        damper, valve = DualMaxOutputMixin._zone_output_map(probe, -0.75)
        self.assertGreater(damper, 0.1)
        self.assertAlmostEqual(valve, 1.0)


if __name__ == "__main__":
    unittest.main()
