"""Focused tests for the slot-linear controller."""

import math
import unittest

import numpy as np

from smart_buildings.baselines.slot_linear import (
    N_SLOTS,
    SlotFeatures,
    SlotLinearPolicy,
    bank_of_day,
    blend_weights,
    slot_of_day,
)
from tools.test_occ_gated import _fake_building, _obs


def _hod(h):
    a = 2 * math.pi * h / 24.0
    return math.sin(a), math.cos(a)


def _dow(d):
    a = 2 * math.pi * d / 7.0
    return math.sin(a), math.cos(a)


class SlotDecodeTest(unittest.TestCase):
    def test_slot_boundaries(self):
        self.assertEqual(slot_of_day(*_hod(0.0)), 0)
        self.assertEqual(slot_of_day(*_hod(6.0)), 72)
        self.assertEqual(slot_of_day(*_hod(23.99)), N_SLOTS - 1)

    def test_bank_weekday_weekend(self):
        self.assertEqual(bank_of_day(*_dow(0)), 0)
        self.assertEqual(bank_of_day(*_dow(4)), 0)
        self.assertEqual(bank_of_day(*_dow(5)), 1)
        self.assertEqual(bank_of_day(*_dow(6)), 1)

    def test_blend_normalized(self):
        for kind in ((1, 2, 1), (1, 4, 6, 4, 1)):
            w = blend_weights(kind)
            self.assertAlmostEqual(w.sum(), 1.0)
            self.assertTrue((w > 0).all())


class SlotFeaturesTest(unittest.TestCase):
    def test_shapes_and_finite(self):
        meta, _, index = _fake_building()
        feat = SlotFeatures(meta)
        bank, slot, (zf, af, gf) = feat.step(_obs(index, 10, 2, zone_k=295.0))
        self.assertEqual(bank, 0)
        self.assertEqual(zf.shape, (2, feat.FZ))
        self.assertEqual(af.shape, (1, feat.FA))
        self.assertEqual(gf.shape, (feat.FG,))
        for arr in (zf, af, gf):
            self.assertTrue(np.isfinite(arr).all())

    def test_memory_causality(self):
        meta, _, index = _fake_building()
        feat = SlotFeatures(meta)
        _, _, (zf1, _, _) = feat.step(_obs(index, 10, 2, zone_k=295.0))
        # First step: prev/EMA equal current temp.
        self.assertAlmostEqual(zf1[0, 14], (295.0 - 294.0) / 6.0)
        _, _, (zf2, _, _) = feat.step(_obs(index, 10, 2, zone_k=296.0))
        # Second step: prev is the first temp, EMA still the first (1-step lag).
        self.assertAlmostEqual(zf2[0, 14], (295.0 - 294.0) / 6.0)
        self.assertAlmostEqual(zf2[0, 15], zf1[0, 15])
        _, _, (zf3, _, _) = feat.step(_obs(index, 10, 2, zone_k=296.0))
        # Third step: EMA has moved toward the new temperature.
        self.assertGreater(zf3[0, 15], zf2[0, 15])
        self.assertLess(zf3[0, 15], (296.0 - 294.0) / 6.0)

    def test_integral_accumulates(self):
        meta, _, index = _fake_building()
        feat = SlotFeatures(meta)
        # Cold zones (below center 295.5): positive center error accumulates.
        _, _, (zf1, af1, _) = feat.step(_obs(index, 10, 2, zone_k=293.0))
        self.assertAlmostEqual(zf1[0, 16], 0.0)
        _, _, (zf2, af2, _) = feat.step(_obs(index, 10, 2, zone_k=293.0))
        self.assertGreater(zf2[0, 16], 0.0)
        self.assertGreater(af2[0, 12], 0.0)  # heat EMA rises
        self.assertAlmostEqual(af2[0, 13], 0.0)  # cool EMA stays


class SlotPolicyTest(unittest.TestCase):
    def _policy(self):
        meta, spec, index = _fake_building()
        feat = SlotFeatures(meta)
        rng = np.random.default_rng(0)
        FZ, FA, FG = feat.FZ, feat.FA, feat.FG
        np.savez("/tmp/slot_test_weights.npz",
                 Wz=rng.normal(0, 0.1, (2, N_SLOTS, FZ, 2)),
                 bz=np.zeros((2, N_SLOTS, 2)),
                 Wa=rng.normal(0, 0.1, (2, N_SLOTS, FA, 4)),
                 ba=np.zeros((2, N_SLOTS, 4)),
                 Wg=rng.normal(0, 0.1, (2, N_SLOTS, FG, 3)),
                 bg=np.zeros((2, N_SLOTS, 3)))
        pol = SlotLinearPolicy(meta, spec,
                               weights_path="/tmp/slot_test_weights.npz")
        return pol, index

    def test_action_contract(self):
        pol, index = self._policy()
        action = pol.get_action(_obs(index, 10, 2, zone_k=295.0, occ=5.0))
        cont, disc = action["continuous_action"], action["discrete_action"]
        self.assertEqual(cont.dtype, np.float32)
        self.assertEqual(disc.dtype, np.int32)
        self.assertTrue((cont >= -1.0).all() and (cont <= 1.0).all())
        self.assertTrue(set(np.unique(disc)) <= {0, 1})

    def test_blend_is_neighbor_average(self):
        pol, _ = self._policy()
        pol.Wz[:] = 0.0
        pol.Wz[0, 71, :, 0] = 1.0
        pol.Wz[0, 72, :, 0] = 2.0
        pol.Wz[0, 73, :, 0] = 3.0
        Wm, _ = pol._blended(pol.Wz, pol.bz, 0, 72)
        self.assertAlmostEqual(Wm[0, 0], (1 + 4 + 3) / 4.0)

    def test_run_bit_threshold(self):
        pol, index = self._policy()
        pol.Wa[:] = 0.0
        pol.ba[:] = 0.0
        pol.ba[:, :, 3] = 0.9  # run logit high everywhere
        action = pol.get_action(_obs(index, 10, 2, zone_k=295.0, occ=5.0))
        self.assertEqual(action["discrete_action"][0], 1)
        pol.ba[:, :, 3] = 0.1
        action = pol.get_action(_obs(index, 10, 2, zone_k=295.0, occ=5.0))
        self.assertEqual(action["discrete_action"][0], 0)


if __name__ == "__main__":
    unittest.main()
