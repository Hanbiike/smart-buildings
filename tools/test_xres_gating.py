"""Gating-audit tests: our `free` flag + fr masking (plan F/D3).

Weights-free by design: the predicate is tested with a real
OccGated7Policy plus a fake index map, and the mask algebra with
synthetic tensors. No torch checkpoint or /tmp path is touched.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.probe_modes import any_cold, residual_free
from tools.skynet_ppo import (GAMMA, LAM, clamp_logstd, clip_frac,
                             compute_gae, mean_kl, normalize_adv,
                             pg_objective)
from tools.skynet_train import collate, pool_checksum
from tools.test_occ_gated import _fake_building, _obs


def _fake_ix(index):
    t1 = index["VAV_1_zone_air_temperature_sensor"]
    t2 = index["VAV_2_zone_air_temperature_sensor"]
    return SimpleNamespace(zone_obs=np.array([[t1, 0, 0, 0, 0, 0],
                                              [t2, 0, 0, 0, 0, 0]]),
                           t_std=5.0, t_mean=295.0,
                           zone_vav=["VAV_1", "VAV_2"], V=2)


class ResidualFreeTest(unittest.TestCase):
    def _run(self, hour, day, **kw):
        meta, spec, index = _fake_building()
        policy = OccGated7Policy(meta, spec)
        obs = _obs(index, hour, day, **kw)
        policy.get_action(obs)  # sets _shutdown / _floor_mode
        return policy, _fake_ix(index), obs

    def test_shutdown_not_free(self):
        policy, ix, obs = self._run(2.5, 2, zone_k=292.0)
        self.assertTrue(policy._shutdown)
        self.assertFalse(residual_free(policy, ix, obs, 1.0))

    def test_floor_mode_not_free(self):
        policy, ix, obs = self._run(10.5, 2, zone_k=295.0,
                                    occ=10.0, now=1.0)
        self.assertTrue(residual_free(policy, ix, obs, 1.0))
        policy._floor_mode = "floor_heat"
        self.assertFalse(residual_free(policy, ix, obs, 1.0))

    def test_cold_prestart_not_free(self):
        policy, ix, obs = self._run(4.5, 2, zone_k=288.0, soon=1.0,
                                    oat_k=250.0)
        self.assertFalse(policy._shutdown)
        self.assertTrue(any_cold(policy, ix, obs, 1.0))
        self.assertFalse(residual_free(policy, ix, obs, 1.0))

    def test_warm_conditioning_free(self):
        policy, ix, obs = self._run(10.5, 2, zone_k=295.0,
                                    occ=10.0, now=1.0)
        self.assertTrue(residual_free(policy, ix, obs, 1.0))

    def test_cold_term_disabled_by_none(self):
        policy, ix, obs = self._run(4.5, 2, zone_k=288.0, soon=1.0,
                                    oat_k=250.0)
        self.assertFalse(policy._shutdown)
        self.assertTrue(residual_free(policy, ix, obs, None))


class FrMaskTest(unittest.TestCase):
    def test_gated_rows_exactly_zero(self):
        torch.manual_seed(0)
        T = 64
        r = torch.exp(torch.randn(T))
        a = torch.randn(T)
        one = torch.zeros(T)
        one[7] = 1.0
        pol = -torch.minimum(r * a, torch.clamp(r, 0.8, 1.2) * a)
        self.assertTrue(torch.equal(pg_objective(r, a, one, 0.2), pol[7]))
        self.assertEqual(float(pg_objective(r, a, torch.zeros(T))), 0.0)

    def test_masked_mean_matches_free_subset(self):
        torch.manual_seed(1)
        T = 64
        r = torch.exp(torch.randn(T))
        a = torch.randn(T)
        fr = torch.ones(T)
        fr[::2] = 0.0
        got = float(pg_objective(r, a, fr))
        want = float(pg_objective(r[1::2], a[1::2], fr[1::2]))
        self.assertLess(abs(got - want), 1e-6)

    def test_kl_masked(self):
        torch.manual_seed(2)
        T = 64
        old_lp = torch.randn(T)
        old_nd = torch.full((T,), 10.0)
        new_lp = old_lp + 1.0
        fr = torch.ones(T)
        fr[::2] = 0.0
        k_all = mean_kl(old_lp, old_nd, new_lp, old_nd, fr)
        k_free = mean_kl(old_lp[1::2], old_nd[1::2], new_lp[1::2],
                         old_nd[1::2], fr[1::2])
        self.assertLess(abs(k_all - k_free), 1e-6)


class CollateStackedTest(unittest.TestCase):
    def _traj(self, T, V, K, fr0):
        rng = np.random.RandomState(V * 100 + K)
        fr = np.ones(T, dtype=np.float32)
        fr[:fr0] = 0.0
        return {"zone": rng.randn(T, V, 23).astype(np.float32),
                "ahu": rng.randn(T, K, 34).astype(np.float32),
                "plant": rng.randn(T, 22).astype(np.float32),
                "za": rng.randn(T, V, 3).astype(np.float32),
                "aa": rng.randn(T, K, 4).astype(np.float32),
                "pa": rng.randn(T, 3).astype(np.float32),
                "lpz": rng.randn(T, V).astype(np.float32),
                "lpa": rng.randn(T, K).astype(np.float32),
                "lpp": rng.randn(T).astype(np.float32),
                "r": rng.randn(T).astype(np.float64),
                "v": rng.randn(T).astype(np.float64),
                "fr": fr}

    def test_stacked_arrays_collate_with_fr(self):
        t1 = self._traj(4, 3, 2, 1)
        t2 = self._traj(6, 5, 3, 2)
        F, bounds = collate([t1, t2])
        self.assertEqual(bounds, [4, 6])
        self.assertEqual(F["zone"].shape, (10, 5, 23))
        self.assertEqual(F["ahu"].shape, (10, 3, 34))
        self.assertTrue(np.array_equal(F["fr"][:4], t1["fr"]))
        self.assertTrue(np.array_equal(F["fr"][4:], t2["fr"]))
        # Padding masks cover real entities only.
        self.assertEqual(float(F["zmask"][:4, :3].sum()), 12.0)
        self.assertEqual(float(F["zmask"][:4, 3:].sum()), 0.0)
        self.assertEqual(float(F["zmask"][4:].sum()), 30.0)


class NormalizeAdvTest(unittest.TestCase):
    def test_scales_stay_separate(self):
        # Cold traj (100x scale) must not set the mild traj's scale.
        rng = np.random.RandomState(0)
        cold = rng.randn(64) * 100.0
        mild = rng.randn(64)
        fr = np.ones(64, dtype=np.float32)
        normalize_adv(cold, fr)
        normalize_adv(mild, fr)
        self.assertAlmostEqual(float(cold.std()), 1.0, places=6)
        self.assertAlmostEqual(float(mild.std()), 1.0, places=6)
        self.assertAlmostEqual(float(cold.mean()), 0.0, places=6)

    def test_gated_entries_untouched(self):
        adv = np.arange(8, dtype=np.float64)
        fr = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float32)
        out = normalize_adv(adv, fr)
        self.assertIs(out, adv)
        self.assertTrue(np.array_equal(adv[:2], [0.0, 1.0]))
        self.assertTrue(np.array_equal(adv[6:], [6.0, 7.0]))
        self.assertAlmostEqual(float(adv[2:6].mean()), 0.0, places=9)
        self.assertAlmostEqual(float(adv[2:6].std()), 1.0, places=6)

    def test_all_gated_untouched(self):
        adv = np.arange(4, dtype=np.float64)
        out = normalize_adv(adv, np.zeros(4, dtype=np.float32))
        self.assertTrue(np.array_equal(out, [0.0, 1.0, 2.0, 3.0]))


class BestCkptTest(unittest.TestCase):
    def test_extra_roundtrip(self):
        # E3: the .best artifact must carry cycle/lr/best_r metadata.
        import tempfile
        import torch
        from tools.skynet_ppo import save_ckpt

        class StubNet:
            cfg = {"d": 64, "depth": 2}

            def state_dict(self):
                return {"w": torch.zeros(2)}

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.best")
            extra = {"cycle": 3, "lr": 1.5e-4, "best_r": -0.0123}
            save_ckpt(StubNet(), None, p, extra=extra)
            ck = torch.load(p, map_location="cpu", weights_only=False)
        self.assertEqual(ck["extra"]["cycle"], 3)
        self.assertAlmostEqual(ck["extra"]["lr"], 1.5e-4)
        self.assertAlmostEqual(ck["extra"]["best_r"], -0.0123)
        self.assertIsNone(ck["opt"])


class PoolChecksumTest(unittest.TestCase):
    def test_order_insensitive_content_sensitive(self):
        # E4: same pool in any order -> same checksum; one swap changes it.
        import hashlib
        a = ["building_train_8/atlanta_1", "building_xfer_1/minneapolis_1"]
        b = list(reversed(a))
        self.assertEqual(pool_checksum(a), pool_checksum(b))
        want = hashlib.sha256("\n".join(sorted(a)).encode()).hexdigest()
        self.assertEqual(pool_checksum(a), want)
        c = ["building_train_8/atlanta_1", "building_xfer_3/singapore_2"]
        self.assertNotEqual(pool_checksum(a), pool_checksum(c))


class ClampLogstdTest(unittest.TestCase):
    def test_floor_binds_below_noop_above(self):
        # H5: collapsed logstds clamp to -2.0, healthy ones untouched.
        class StubNet:
            def __init__(self):
                self._s = [torch.nn.Parameter(torch.tensor([-3.5, -1.2])),
                           torch.nn.Parameter(torch.tensor([-1.4])),
                           torch.nn.Parameter(torch.tensor([-0.9, -4.0]))]

            def stds(self):
                return self._s

        net = StubNet()
        clamp_logstd(net)
        got = [s.detach().tolist() for s in net.stds()]
        want = [[-2.0, -1.2], [-1.4], [-0.9, -2.0]]
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=1e-6)


class MCReturnsTest(unittest.TestCase):
    def test_lam1_zeros_is_discounted_rewards(self):
        # H2 relies on this: lam=1 over zero values = MC gamma-returns.
        r = np.array([-1.0, -2.0, -3.0, -4.0])
        _, ret = compute_gae(r, np.zeros(4), lam=1.0)
        want = np.array([r[t] + sum(GAMMA ** k * r[t + k]
                                    for k in range(1, 4 - t))
                         for t in range(4)])
        np.testing.assert_allclose(ret, want, rtol=1e-9)

    def test_bootstrap_default_zero_lastvalue_opt_in(self):
        # H7: terminal default bootstraps 0; truncated passes last v.
        r = np.zeros(3)
        v = np.ones(3)
        adv0, _ = compute_gae(r, v)
        self.assertAlmostEqual(adv0[-1], -1.0)
        adv5, _ = compute_gae(r, v, bootstrap=5.0)
        self.assertAlmostEqual(adv5[-1], GAMMA * 5.0 - 1.0)
        # Bootstrap only touches the tail (horizon ~18 steps).
        np.testing.assert_allclose(adv5[:-1] - adv0[:-1],
                                   (GAMMA * LAM) ** np.array([2, 1])
                                   * GAMMA * 5.0, rtol=1e-9)


class ClipFracTest(unittest.TestCase):
    def test_band_and_mask(self):
        # E5: ratios outside [0.8, 1.2] count; gated rows don't.
        r = torch.tensor([1.0, 1.1, 1.5, 0.5, 2.0, 0.9])
        fr = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        # Free rows: 1.5 and 0.5 out of 4 -> 0.5 (2.0 is gated).
        self.assertAlmostEqual(clip_frac(r, fr), 0.5)
        self.assertAlmostEqual(
            clip_frac(r, torch.ones(6), clip=0.9), 1.0 / 6.0)
        self.assertAlmostEqual(clip_frac(r, torch.zeros(6)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
