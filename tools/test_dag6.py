# -*- coding: utf-8 -*-
"""P23-6: dagger_collect key-resolution tests (0 sim)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_buildings import evaluate as E
from tools import dagger_collect as D


class TestDaggerKeys(unittest.TestCase):
    def test_p23_roles_resolve(self):
        from smart_buildings.baselines.probe_modes import ResidS30Probe
        from smart_buildings.baselines.occ_econdbhhot9 import (
            OccEconDbhHot9Policy)
        self.assertIs(E.POLICY_MAP["probe_slot_res30"], ResidS30Probe)
        self.assertIs(E.POLICY_MAP["occ_econdbhhot9"], OccEconDbhHot9Policy)

    def test_collect_signature(self):
        import inspect
        params = list(inspect.signature(D._collect_proc).parameters)
        self.assertEqual(params, ["scenario", "gin_path", "student_key",
                                 "student_weights", "teacher_key",
                                 "steps", "tag"])


if __name__ == "__main__":
    unittest.main()
