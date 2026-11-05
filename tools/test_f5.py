# -*- coding: utf-8 -*-
"""P23-5: augmented-feature unit tests (dims, scaling, fallbacks)."""
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_buildings.baselines.slot_linear import SlotFeatures
from tools.test_occ_gated import _fake_building, _obs


class TestF5Dims(unittest.TestCase):
    def test_dims(self):
        meta, _, _ = _fake_building()
        feat = SlotFeatures(meta)
        self.assertEqual(feat.FZ, 20)
        self.assertEqual(feat.FA, 18)
        self.assertEqual(feat.FG, 12)

    def test_new_channels_finite(self):
        meta, _, index = _fake_building()
        feat = SlotFeatures(meta)
        _, _, (zf, af, gf) = feat.step(
            _obs(index, 10, 2, zone_k=295.0))
        self.assertTrue(np.isfinite(zf[:, 18:]).all())
        self.assertTrue(np.isfinite(af[:, 14:]).all())
        self.assertTrue(np.isfinite(gf[11:]).all())

    def test_fallbacks(self):
        meta, _, _ = _fake_building()
        feat = SlotFeatures(meta)
        obs = np.zeros(len(meta["observation_field_names"]))
        # fake building lacks setpoint/fan columns -> defaults
        self.assertEqual(feat._ahu_kelvin(
            obs, 0, "supply_air_temperature_setpoint", 291.0), 291.0)
        if feat.hws_twsp < 0:
            self.assertEqual(feat._twsp_kelvin(obs), 330.0)


class TestF5RealObs(unittest.TestCase):
    def test_scales_on_real_obs(self):
        wdir = (ROOT / "smart_buildings" / "results"
                / "building_train_8" / "fairbanks_8")
        meta = json.load(open(wdir / "scenario_meta.json"))
        feat = SlotFeatures(meta)
        d = np.load(wdir / "nn_data_occ_econdbhhot9" / "obs_dump.npz")
        n_bad = 0
        for t in range(0, 8640, 97):
            _, _, (zf, af, gf) = feat.step(d["obs"][t], update=True)
            if not (np.isfinite(zf).all() and np.isfinite(af).all()
                    and np.isfinite(gf).all()):
                n_bad += 1
            # unit-ish scales on the new channels
            self.assertTrue((np.abs(af[:, 14]) < 3).all())   # satsp
            self.assertTrue((np.abs(af[:, 15]) <= 2).all())  # sat err
            self.assertTrue((np.abs(af[:, 16]) <= 2).all())  # dp err
            self.assertTrue(((af[:, 17] >= 0)
                             & (af[:, 17] < 5)).all())       # fan x4
            self.assertTrue((np.abs(zf[:, 18:]) < 10).all())
            self.assertTrue(abs(float(gf[11])) < 5)          # water sp gap
        self.assertEqual(n_bad, 0)


if __name__ == "__main__":
    unittest.main()
