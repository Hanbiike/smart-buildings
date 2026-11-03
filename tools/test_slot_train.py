# -*- coding: utf-8 -*-
"""Tests for slot_train.py soft-assignment mechanics (MoE T1)."""
import unittest

import numpy as np

from tools.slot_train import SOFT_KERNELS, Accum


class SoftKernelTest(unittest.TestCase):
    def test_kernels_normalized_symmetric(self):
        for name, (offs, w) in SOFT_KERNELS.items():
            self.assertAlmostEqual(sum(w), 1.0, msg=name)
            self.assertEqual(tuple(-o for o in reversed(offs)), offs,
                             msg=name)

    def test_k3_exact(self):
        offs, w = SOFT_KERNELS["k3"]
        self.assertEqual(offs, (-1, 0, 1))
        self.assertEqual(w, (0.25, 0.5, 0.25))

    def test_k5_exact(self):
        offs, w = SOFT_KERNELS["k5"]
        self.assertEqual(offs, (-2, -1, 0, 1, 2))
        self.assertEqual(w, (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16))

    def test_circular_wrap(self):
        # slot 0 spreads to 287/0/1; slot 287 to 286/287/0 (mod 288)
        from smart_buildings.baselines.slot_linear import N_SLOTS
        offs, _ = SOFT_KERNELS["k3"]
        self.assertEqual(sorted((0 + o) % N_SLOTS for o in offs),
                         [0, 1, 287])
        self.assertEqual(sorted((287 + o) % N_SLOTS for o in offs),
                         [0, 286, 287])


class WeightedAccumTest(unittest.TestCase):
    def test_weight_splits(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 4))
        Y = rng.normal(size=(10, 2))
        a = Accum(4, 2)
        a.add(X, Y, w=1.0)
        b = Accum(4, 2)
        b.add(X, Y, w=0.25)
        b.add(X, Y, w=0.75)
        np.testing.assert_allclose(a.A, b.A, rtol=1e-12)
        np.testing.assert_allclose(a.B, b.B, rtol=1e-12)
        self.assertAlmostEqual(a.n, b.n)

    def test_zero_weight_adds_nothing(self):
        a = Accum(2, 1)
        a.add(np.ones((3, 2)), np.ones((3, 1)), w=0.0)
        self.assertEqual(a.n, 0)
        W, b = a.solve(1.0)
        self.assertIsNone(W)
        self.assertIsNone(b)


if __name__ == "__main__":
    unittest.main()
