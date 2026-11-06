# -*- coding: utf-8 -*-
"""P23-6: replay_base writer tests (determinism + shipped-B parity)."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
PAIR = "building_train_8/fairbanks_8"
DD = ROOT / "smart_buildings" / "results" / PAIR / "nn_data_occ_econdbhhot9"
DUMP = DD / "obs_dump.npz"
SHIPPED_B = DD / "B_occ_gated7.npy"
# /tmp is quota-capped; test artifacts go to the (git-ignored) slot_runs.
RUNS = ROOT / "smart_buildings" / "results" / "slot_runs"


def _replay(out):
    cmd = [PY, "-B", str(ROOT / "tools" / "replay_base.py"),
           "--pairs", PAIR, "--out", out]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          timeout=1200)


@unittest.skipUnless(DUMP.exists(), "needs teacher dump (local artifact)")
class TestReplayBase(unittest.TestCase):
    def test_determinism_and_shape(self):
        r1 = _replay(str(RUNS / "test_rb1.npy"))
        self.assertEqual(r1.returncode, 0, r1.stderr[-2000:])
        r2 = _replay(str(RUNS / "test_rb2.npy"))
        self.assertEqual(r2.returncode, 0, r2.stderr[-2000:])
        a = np.load(RUNS / "test_rb1.npy")
        b = np.load(RUNS / "test_rb2.npy")
        self.assertTrue(np.array_equal(a, b))
        z = np.load(DUMP)
        self.assertEqual(a.shape, z["action_continuous"].shape)
        self.assertTrue(np.isfinite(a).all())

    @unittest.skipUnless(SHIPPED_B.exists(), "needs shipped B-file")
    def test_shipped_parity(self):
        r = _replay(str(RUNS / "test_rb3.npy"))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        a = np.load(RUNS / "test_rb3.npy")
        self.assertTrue(np.array_equal(a, np.load(SHIPPED_B)))


if __name__ == "__main__":
    unittest.main()
