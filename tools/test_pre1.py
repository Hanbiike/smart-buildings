"""P22-2: PRE margin probe registration."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)
from smart_buildings.baselines.occ_pre1 import PreMargin12Policy


class TestPre1(unittest.TestCase):
    def test_registered(self):
        self.assertIn("occ_prem12", evaluate.POLICY_MAP)
        self.assertIs(evaluate.POLICY_MAP["occ_prem12"],
                      PreMargin12Policy)

    def test_const(self):
        self.assertEqual(PreMargin12Policy.PRESTART_MARGIN_STEPS, 12)
        self.assertEqual(
            OccEconDbhHot9Policy.PRESTART_MARGIN_STEPS, 6)

    def test_inheritance(self):
        self.assertTrue(issubclass(PreMargin12Policy,
                                   OccEconDbhHot9Policy))


if __name__ == "__main__":
    unittest.main()
