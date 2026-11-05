"""P22-1: UGAS floor-cap probe registration."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)
from smart_buildings.baselines.occ_ugas1 import UgasCap20Policy


class TestUgas1(unittest.TestCase):
    def test_registered(self):
        self.assertIn("occ_ugascap20", evaluate.POLICY_MAP)
        self.assertIs(evaluate.POLICY_MAP["occ_ugascap20"],
                      UgasCap20Policy)

    def test_const(self):
        self.assertEqual(UgasCap20Policy.FLOOR_HEAT_CAP_BELOW_ECO, 2.0)
        self.assertEqual(
            OccEconDbhHot9Policy.FLOOR_HEAT_CAP_BELOW_ECO, 1.0)

    def test_inheritance(self):
        self.assertTrue(issubclass(UgasCap20Policy,
                                   OccEconDbhHot9Policy))


if __name__ == "__main__":
    unittest.main()
