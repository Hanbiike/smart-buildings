# -*- coding: utf-8 -*-
"""P23-4: K=2 climate-mode resid — unit + train/liveness tests.

Covers: OAT mode mapping/hysteresis, K=1 legacy parity (bit-exact,
legacy format), K=2 shapes/modes-key, deploy-path rollout liveness.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv" / "bin" / "python")

from smart_buildings.baselines.slot_linear import (  # noqa: E402
    MODE_HYST_K, SlotFeatures, mode_of_oat,
)

PAIR = "building_train_8/fairbanks_8"
RUNS = ROOT / "smart_buildings" / "results" / "slot_runs"


def _run_train(out, modes, bound="283.15"):
    cmd = [PY, "-B", str(ROOT / "tools" / "slot_train.py"),
           "--teacher", "occ_econdbhhot9",
           "--residual-base", "occ_gated7",
           "--scenarios", PAIR, "--no-holdout", "--out", out,
           "--modes", str(modes), "--mode-bounds", bound]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          timeout=1200)


class TestOatMode(unittest.TestCase):
    def test_mapping(self):
        b = (283.15,)
        self.assertEqual(mode_of_oat(260.0, b), 0)
        self.assertEqual(mode_of_oat(283.14, b), 0)
        self.assertEqual(mode_of_oat(283.15, b), 1)
        self.assertEqual(mode_of_oat(310.0, b), 1)
        self.assertEqual(mode_of_oat(310.0, ()), 0)  # K=1 legacy

    def test_hysteresis_roundtrip(self):
        meta = json.load(open(ROOT / "smart_buildings" / "results"
                              / "building_train_8" / "fairbanks_8"
                              / "scenario_meta.json"))
        feat = SlotFeatures(meta)
        feat.mode_bounds = (283.15,)
        feat.reset()
        self.assertIsNone(feat._oat_mode)
        feat._update_oat_mode(270.0)
        self.assertEqual(feat.mode, 0)
        feat._update_oat_mode(283.15 + MODE_HYST_K - 0.01)  # below hyst edge
        self.assertEqual(feat.mode, 0)
        feat._update_oat_mode(283.15 + MODE_HYST_K)  # cross up
        self.assertEqual(feat.mode, 1)
        feat._update_oat_mode(283.15 - MODE_HYST_K + 0.01)  # below, in band
        self.assertEqual(feat.mode, 1)
        feat._update_oat_mode(283.15 - MODE_HYST_K - 0.01)  # cross down
        self.assertEqual(feat.mode, 0)


class TestK1Legacy(unittest.TestCase):
    def test_k1_format_and_determinism(self):
        r1 = _run_train("test_resk2_k1a", 1)
        self.assertEqual(r1.returncode, 0, r1.stderr[-2000:])
        r2 = _run_train("test_resk2_k1b", 1)
        self.assertEqual(r2.returncode, 0, r2.stderr[-2000:])
        a = np.load(RUNS / "test_resk2_k1a.npz")
        b = np.load(RUNS / "test_resk2_k1b.npz")
        self.assertEqual(a["Wz"].ndim, 4)  # legacy (2,288,F,Out)
        self.assertNotIn("modes", set(a.files))
        for k in ("Wz", "bz", "Wa", "ba", "Wg", "bg"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)


class TestK2Deploy(unittest.TestCase):
    def test_k2_shapes_modeskey_and_rollout(self):
        from smart_buildings import evaluate as E
        from smart_buildings.baselines.probe_modes import ResidK2Probe

        r = _run_train("test_resk2_k2", 2)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("mode0:", r.stdout)
        self.assertIn("mode1:", r.stdout)
        z = np.load(RUNS / "test_resk2_k2.npz")
        self.assertEqual(z["Wz"].ndim, 5)
        self.assertEqual(z["Wz"].shape[0], 2)
        self.assertIn("modes", set(z.files))
        self.assertTrue(np.allclose(z["modes"], [283.15]))

        # deploy path: hybrid loads K=2, bounds wired, rollout lives
        class _T(ResidK2Probe):
            WEIGHTS = str(RUNS / "test_resk2_k2.npz")

        self.assertIn("probe_slot_resk2", E.POLICY_MAP)
        self.assertIs(E.POLICY_MAP["probe_slot_resk2"], ResidK2Probe)
        # POLICY_MAP points at the full-refit WEIGHTS (built after this
        # test); run the test-weights subclass directly for liveness.
        scen = (ROOT / "smart_buildings" / "training_scenarios"
                / "building_train_8" / "fairbanks_8")
        summary = E.evaluate_policy_for_config(
            Path("/tmp/resk2live"), scen, "test_resk2_k2", _T, 300)
        self.assertTrue(np.isfinite(summary["total_reward"]))
        import pandas as pd
        df = pd.read_csv(Path("/tmp/resk2live") / "probe_slot_resk2"
                         / "reward_components.csv")
        self.assertEqual(len(df), 300)
