"""Tests for the G8 worst-climate promotion screen."""
import os
import tempfile
import unittest

from tools.promote_screen import main, screen


def _csv(d, building, weather, policy, rewards):
    p = os.path.join(d, building, weather, policy)
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "reward_components.csv"), "w") as f:
        f.write("agent_reward\n")
        for r in rewards:
            f.write(f"{r}\n")


class PromoteScreenTest(unittest.TestCase):
    def _layout(self, d, cand_rewards):
        # Rule policy at 0.01/pair on two pairs; candidate varies.
        for w in ("atlanta_1", "fairbanks_8"):
            _csv(d, "b8", w, "rule", [-0.01] * 100)
        for w, rs in zip(("atlanta_1", "fairbanks_8"), cand_rewards):
            _csv(d, "b8", w, "cand", rs)

    def test_pass(self):
        with tempfile.TemporaryDirectory() as d:
            self._layout(d, ([-0.01] * 100, [-0.029] * 100))
            ok, rows, missing = screen(d, "cand", "rule")
            self.assertTrue(ok)
            self.assertEqual(missing, [])
            self.assertEqual(len(rows), 2)
            self.assertEqual(main([d, "cand", "rule"]), 0)

    def test_fail_one_blowup(self):
        with tempfile.TemporaryDirectory() as d:
            self._layout(d, ([-0.01] * 100, [-0.035] * 100))
            ok, rows, _ = screen(d, "cand", "rule")
            self.assertFalse(ok)
            worst = max(rows, key=lambda r: r[3])
            self.assertEqual(worst[0], "b8/fairbanks_8")
            self.assertAlmostEqual(worst[3], 3.5)
            self.assertEqual(main([d, "cand", "rule"]), 1)

    def test_boundary_3x_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self._layout(d, ([-0.03] * 100, [-0.03] * 100))
            ok, _, _ = screen(d, "cand", "rule")
            self.assertTrue(ok)

    def test_missing_rule_data(self):
        with tempfile.TemporaryDirectory() as d:
            _csv(d, "b8", "atlanta_1", "cand", [-0.01] * 10)
            ok, rows, missing = screen(d, "cand", "rule")
            self.assertFalse(ok)
            self.assertEqual(missing, ["b8/atlanta_1"])
            self.assertEqual(main([d, "cand", "rule"]), 2)


if __name__ == "__main__":
    unittest.main()
