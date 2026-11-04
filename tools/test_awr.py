# -*- coding: utf-8 -*-
"""Focused tests for the P10 offline stack (plan-19 P10.1)."""
import sys
import unittest

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
sys.path.insert(0, "/tmp/pkg")

import numpy as np

import vrule_fit as vf


class TestMcReturnsGamma(unittest.TestCase):
    def test_gamma_forwarded(self):
        # Regression: mc_returns accepted gamma but compute_gae ran at
        # the module default (P10.1 fix; BCOOLER-style MC needs gamma=1).
        r = np.array([1.0, 1.0, 1.0, 1.0])
        m1 = vf.mc_returns(r, [4], gamma=1.0)
        m0 = vf.mc_returns(r, [4], gamma=0.0)
        np.testing.assert_allclose(m1, [4.0, 3.0, 2.0, 1.0], rtol=1e-9)
        np.testing.assert_allclose(m0, [1.0, 1.0, 1.0, 1.0], rtol=1e-9)

    def test_default_is_ppo_gamma(self):
        import skynet_ppo as ppo
        r = np.array([1.0, 2.0, 3.0])
        a = vf.mc_returns(r, [3])
        b = vf.mc_returns(r, [3], gamma=ppo.GAMMA)
        np.testing.assert_allclose(a, b, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
