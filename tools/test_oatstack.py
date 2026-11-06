# -*- coding: utf-8 -*-
"""P23-7: OAT-stack probe registration, attrs, and MRO (0 sim)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_buildings import evaluate as E
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy)
from smart_buildings.baselines.occ_pre1 import PreMargin12Policy
from smart_buildings.baselines.occ_ugas1 import UgasCap20Policy
from smart_buildings.baselines.probe_modes import (
    N1OffMixin, OatStackAProbe, OatStackBProbe,
)


class TestOatStack(unittest.TestCase):
    def test_registered(self):
        self.assertIs(E.POLICY_MAP["probe_oat_stacka"], OatStackAProbe)
        self.assertIs(E.POLICY_MAP["probe_oat_stackb"], OatStackBProbe)

    def test_stacka_attrs(self):
        self.assertEqual(OatStackAProbe.PRESTART_MARGIN_STEPS, 12)
        self.assertEqual(OatStackAProbe.N1_OFF_OAT_GATE, 1)
        self.assertTrue(issubclass(OatStackAProbe, N1OffMixin))
        self.assertTrue(issubclass(OatStackAProbe, PreMargin12Policy))
        self.assertTrue(issubclass(OatStackAProbe, OccEconDbhHot9Policy))

    def test_stackb_attrs(self):
        self.assertEqual(OatStackBProbe.PRESTART_MARGIN_STEPS, 12)
        self.assertEqual(OatStackBProbe.FLOOR_HEAT_CAP_BELOW_ECO, 2.0)
        self.assertEqual(OatStackBProbe.N1_OFF_OAT_GATE, 1)
        self.assertTrue(issubclass(OatStackBProbe, N1OffMixin))
        self.assertTrue(issubclass(OatStackBProbe, PreMargin12Policy))
        self.assertTrue(issubclass(OatStackBProbe, UgasCap20Policy))

    def test_mro_base_action_from_mixin(self):
        self.assertIs(OatStackAProbe._base_action,
                      N1OffMixin._base_action)
        self.assertIs(OatStackBProbe._base_action,
                      N1OffMixin._base_action)


if __name__ == "__main__":
    unittest.main()
