"""Unit test for tools/oracle_bound.py on synthetic CSVs."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.oracle_bound import bin_losses, oracle_bound


def _csv(path: Path, rewards: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("agent_reward\n")
        for r in rewards:
            f.write(f"{r}\n")


class TestOracleBound(unittest.TestCase):
    def test_bin_losses(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "r.csv"
            _csv(p, [-1.0, -3.0, 2.0, -2.0])
            self.assertEqual(bin_losses(p, 2), [2.0, 0.0])

    def test_bound_math(self):
        # 1 pair, 2 bins of 2 steps, arms A/B + control C.
        # bin0 losses: A=1.0 B=3.0 C=2.0 -> best A=1.0
        # bin1 losses: A=1.0 B=0.0 C=2.0 -> best B=0.0
        # fixed-best: A=1.0; oracle=(1.0+0.0)/2=0.5; bound=0.5.
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _csv(root / "arms" / "b" / "w" / "A" / "reward_components.csv",
                 [-1.0, -1.0, -1.0, -1.0])
            _csv(root / "arms" / "b" / "w" / "B" / "reward_components.csv",
                 [-3.0, -3.0, 1.0, -1.0])
            _csv(root / "ctl" / "b" / "w" / "C" / "reward_components.csv",
                 [-2.0, -2.0, -2.0, -2.0])
            res = oracle_bound(root / "arms", root / "ctl", [("b", "w")],
                               ["A", "B"], "C", 2)
            self.assertAlmostEqual(res["fixed_best"], 1.0)
            self.assertAlmostEqual(res["oracle"], 0.5)
            self.assertAlmostEqual(res["bound"], 0.5)
            self.assertEqual(res["n_bins"], 2)

    def test_oracle_never_worse_than_fixed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _csv(root / "arms" / "b" / "w" / "A" / "reward_components.csv",
                 [-2.0] * 8)
            _csv(root / "ctl" / "b" / "w" / "C" / "reward_components.csv",
                 [-1.0] * 8)
            res = oracle_bound(root / "arms", root / "ctl", [("b", "w")],
                               ["A"], "C", 4)
            self.assertAlmostEqual(res["bound"], 0.0)


if __name__ == "__main__":
    unittest.main()
