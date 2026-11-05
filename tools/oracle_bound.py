#!/usr/bin/env python3
"""J5-style hindsight oracle bound over per-bin arm losses.

Usage: oracle_bound.py <arms_dir> <control_dir> <pairs_csv> <arms_csv>
       [--bin-steps 96] [--control-policy occ_econdbhhot9]

Layout: <dir>/<building>/<weather>/<policy>/reward_components.csv
with an agent_reward column (per-step). Loss = abs(sum)/len, sliced
into bins of --bin-steps; the oracle takes the best arm per bin.
Bound = fixed-best mean - oracle mean over all bins x pairs.
"""

import sys
from pathlib import Path

import pandas as pd


def bin_losses(csv: Path, bin_steps: int) -> list:
    df = pd.read_csv(csv, usecols=["agent_reward"])
    r = df.agent_reward.to_numpy()
    n = len(r) // bin_steps * bin_steps
    return [abs(r[i:i + bin_steps].sum()) / bin_steps
            for i in range(0, n, bin_steps)]


def oracle_bound(arms_dir: Path, control_dir: Path, pairs: list,
                 arms: list, control_policy: str,
                 bin_steps: int) -> dict:
    cands = arms + [control_policy]
    per_pair_fixed: dict = {}
    oracle_sum = 0.0
    n_bins = 0
    for b, w in pairs:
        legs = {}
        for c in cands:
            root = control_dir if c == control_policy else arms_dir
            legs[c] = bin_losses(root / b / w / c / "reward_components.csv",
                                 bin_steps)
        nb = len(legs[cands[0]])
        assert all(len(v) == nb for v in legs.values()), (b, w)
        fixed = min(sum(v) / nb for v in legs.values())
        per_pair_fixed[f"{b}/{w}"] = fixed
        for i in range(nb):
            oracle_sum += min(v[i] for v in legs.values())
        n_bins += nb
    fixed_best = sum(per_pair_fixed.values()) / len(per_pair_fixed)
    oracle = oracle_sum / n_bins
    return {"fixed_best": fixed_best, "oracle": oracle,
            "bound": fixed_best - oracle, "n_bins": n_bins,
            "per_pair_fixed": per_pair_fixed}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = [a for a in sys.argv[1:] if a.startswith("--")]
    bin_steps = 96
    control_policy = "occ_econdbhhot9"
    for o in opts:
        if o.startswith("--bin-steps"):
            bin_steps = int(o.split("=", 1)[1])
        elif o.startswith("--control-policy"):
            control_policy = o.split("=", 1)[1]
    arms_dir, control_dir = Path(args[0]), Path(args[1])
    pairs = [tuple(p.split("/")) for p in args[2].split(",")]
    arms = args[3].split(",")
    res = oracle_bound(arms_dir, control_dir, pairs, arms, control_policy,
                       bin_steps)
    print(f"bins: {res['n_bins']} x {bin_steps} steps")
    for pair, fixed in res["per_pair_fixed"].items():
        print(f"{pair}: fixed-best={fixed:.6f}")
    print(f"fixed-best mean: {res['fixed_best']:.6f}")
    print(f"oracle mean:     {res['oracle']:.6f}")
    print(f"BOUND:           {res['bound']:.6f}")


if __name__ == "__main__":
    main()
