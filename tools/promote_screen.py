#!/usr/bin/env python3
"""Promotion screen: fail if any single pair exceeds 3x its rule value.

Worst-climate guard (plan F/G8, their 0.96-Fairbanks near-miss 9.8).
Reads a fullrun/fullrun_t results dir and compares one candidate policy
against one rule policy pair-by-pair.

Usage: promote_screen.py <results_dir> <candidate> <rule> [max_ratio]
Exit 0 on PASS, 1 on FAIL (any ratio > max_ratio), 2 on missing data.
"""
import glob
import os
import sys

import pandas as pd

DEFAULT_MAX_RATIO = 3.0


def pair_loss(csv):
    df = pd.read_csv(csv)
    return abs(df.agent_reward.sum()) / len(df)


def screen(results_dir, candidate, rule, max_ratio=DEFAULT_MAX_RATIO):
    """Return (ok, rows, missing). rows: [(pair, cand, rule, ratio)]."""
    rows, missing = [], []
    cands = glob.glob(os.path.join(results_dir, "*", "*", candidate,
                                   "reward_components.csv"))
    for csv in sorted(cands):
        p = csv.split(os.sep)
        pair = f"{p[-4]}/{p[-3]}"
        rcsv = os.path.join(results_dir, p[-4], p[-3], rule,
                            "reward_components.csv")
        if not os.path.exists(rcsv):
            missing.append(pair)
            continue
        cl, rl = pair_loss(csv), pair_loss(rcsv)
        rows.append((pair, cl, rl, cl / rl if rl > 0 else float("inf")))
    # Relative epsilon: an exact-3x ratio must not fail on fp noise.
    ok = (bool(rows) and not missing
          and all(r[3] <= max_ratio * (1 + 1e-9) for r in rows))
    return ok, rows, missing


def main(argv=None):
    a = argv or sys.argv[1:]
    results_dir, candidate, rule = a[0], a[1], a[2]
    max_ratio = float(a[3]) if len(a) > 3 else DEFAULT_MAX_RATIO
    ok, rows, missing = screen(results_dir, candidate, rule, max_ratio)
    for pair, cl, rl, ratio in sorted(rows, key=lambda r: -r[3]):
        flag = "FAIL" if ratio > max_ratio * (1 + 1e-9) else "ok"
        print(f"{pair:<34} cand={cl:.6f} rule={rl:.6f} "
              f"ratio={ratio:.3f} {flag}")
    if missing:
        print(f"missing rule data: {sorted(missing)}")
        return 2
    if not rows:
        print("no candidate data")
        return 2
    worst = max(r[3] for r in rows)
    print(f"G8 screen: {'PASS' if ok else 'FAIL'} "
          f"(n={len(rows)} worst_ratio={worst:.3f} limit={max_ratio})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
