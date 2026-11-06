# -*- coding: utf-8 -*-
"""P23-8: ship-policy parity, P21-3 bars, determinism (0 sim).

Replays fb8 teacher obs offline through probe_slot_res30 (f64),
probe_slot_res30 with f32-cast heads, and occ_resid30 (embedded
f32): ship-vs-f32 must be bit-exact (code-mirror proof), ship-vs-f64
within the P21-3 bars (cont < 1e-4, discrete equality).
"""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_buildings import evaluate as E
from smart_buildings.baselines.occ_resid30 import (
    RESID30_W_B64, Resid30ShipPolicy,
)
from smart_buildings.baselines.probe_modes import ResidS30Probe

PAIR = ("building_train_8", "fairbanks_8")
N_STEPS = 8640


def _env():
    gin = (ROOT / "smart_buildings" / "training_scenarios"
           / PAIR[0] / PAIR[1] / "config.gin")
    env = E.load_hybrid_environment(gin)
    env.metrics_path = None
    meta = E.create_building_meta_data(env, gin.parent)
    return meta, env.action_spec()


def _replay(pol, obs, n):
    conts, discs = [], []
    for t in range(n):
        a = E.validate_action(E.get_policy_action(pol, obs[t]),
                              _SPEC)
        conts.append(a["continuous_action"])
        discs.append(a["discrete_action"])
    return np.stack(conts), np.stack(discs)


_SPEC = None


class TestShip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global _SPEC
        meta, _SPEC = _env()
        cls.meta = meta
        d = np.load(ROOT / "smart_buildings" / "results" / PAIR[0]
                    / PAIR[1] / "nn_data_occ_econdbhhot9" / "obs_dump.npz")
        cls.obs = d["obs"]

    def test_registered(self):
        self.assertIs(E.POLICY_MAP["occ_resid30"], Resid30ShipPolicy)
        self.assertEqual(Resid30ShipPolicy.KEY, "occ_resid30")
        self.assertEqual(Resid30ShipPolicy.RESID_SCALE, 0.30)

    def test_no_heavy_imports(self):
        src = (ROOT / "smart_buildings" / "baselines"
               / "occ_resid30.py").read_text()
        imports = [l for l in src.splitlines()
                   if l.startswith("import ") or l.startswith("from ")]
        blob = "\n".join(imports)
        for ban in ("torch", "nn_model", "slot_linear", "slot_hybrid",
                    "probe_modes", "base64", "zlib"):
            self.assertNotIn(ban, blob)
        self.assertIn("occ_gated7", blob)
        self.assertIn("import numpy", blob)

    def test_payload_size(self):
        self.assertLess(len(RESID30_W_B64), 200 * 1024)

    def test_consts_extraction_clean(self):
        import re
        src = (ROOT / "smart_buildings" / "baselines"
               / "occ_resid30.py").read_text()
        consts = [l for l in src.splitlines()
                  if re.match(r"^[A-Z_][A-Z_0-9]* *=", l)]
        self.assertEqual([c.split("=")[0].strip() for c in consts],
                         ["RESID30_W_B64"])

    def test_parity_and_bars(self):
        ship = Resid30ShipPolicy(self.meta, _SPEC)
        probe = ResidS30Probe(self.meta, _SPEC)
        # f32-cast twin of the probe (same code path, ship precision)
        for k in ("Wz", "bz", "Wa", "ba", "Wg", "bg"):
            setattr(probe, k, getattr(probe, k).astype(np.float32))
        ship.reset()
        probe.reset()
        n = N_STEPS
        cs, ds = _replay(ship, self.obs, n)
        cp, dp = _replay(probe, self.obs, n)
        self.assertTrue(np.array_equal(cs, cp),
                        f"maxdiff={np.abs(cs - cp).max()}")
        self.assertTrue(np.array_equal(ds, dp))
        # P21-3 bars vs the f64 reference
        probe64 = ResidS30Probe(self.meta, _SPEC)
        probe64.reset()
        c64, d64 = _replay(probe64, self.obs, n)
        self.assertLess(np.abs(cs.astype(float) - c64).max(), 1e-4)
        self.assertTrue(np.array_equal(ds, d64))

    def test_determinism_and_time(self):
        a = Resid30ShipPolicy(self.meta, _SPEC)
        a.reset()
        t0 = time.time()
        c1, d1 = _replay(a, self.obs, 2000)
        el = (time.time() - t0) / 2000
        b = Resid30ShipPolicy(self.meta, _SPEC)
        b.reset()
        c2, d2 = _replay(b, self.obs, 2000)
        self.assertTrue(np.array_equal(c1, c2))
        self.assertTrue(np.array_equal(d1, d2))
        self.assertLess(el, 0.05)  # 50ms/step p99-style bar (mean proxy)


if __name__ == "__main__":
    unittest.main()
