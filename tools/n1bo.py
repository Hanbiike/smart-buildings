# -*- coding: utf-8 -*-
"""Bayesian-optimization ask-tell driver for the N1 danielwuu stack (P20-7).

G4bo-style protocol: bayes_opt ask-tell over 3 dims (off_margin, p2_exp,
hyst_window), 5-scenario screen (same SCEN as g4bo) x 864 steps, HONEST
billing (hybrid-arm ledger in honest steps). No PG sims in this arm.

Usage: n1bo.py <fresh_iters> [init_points]
State: smart_buildings/results/n1bo_state.json (git-ignored).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from bayes_opt import BayesianOptimization

ROOT = Path(__file__).resolve().parent.parent
PARAMS_JSON = ROOT / "smart_buildings" / "results" / "n1bo_params.json"
BO_STATE = ROOT / "smart_buildings" / "results" / "n1bo_state.json"

PBOUNDS = {
    "off_margin": (0.0, 1.5),
    "p2_exp": (1.0, 3.0),
    "hyst_window": (6.0, 72.0),
}
SCEN = [
    ("building_train_1", "fairbanks_1"),
    ("building_train_1", "atlanta_2"),
    ("building_train_2", "fairbanks_2"),
    ("building_train_8", "fairbanks_8"),
    ("building_train_8", "singapore_7"),
]
STEPS = 864
SCREEN_SIMS = len(SCEN) * STEPS


def run_screen(tag: str) -> float:
    """Evaluate probe_n1bo on the 5-scenario honest screen; return sum loss."""
    pairs = ",".join(f"{b}/{w}" for b, w in SCEN)
    out = ROOT / "smart_buildings" / "results" / tag
    env = dict(os.environ, HONEST_BILL="1")
    subprocess.run(
        [sys.executable, "tools/fullrun.py", str(out), str(STEPS),
         "5", "1500", "probe_n1bo", pairs],
        cwd=ROOT, check=True, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    total = 0.0
    for b, w in SCEN:
        csv = out / b / w / "probe_n1bo" / "reward_components.csv"
        df = pd.read_csv(csv)
        total += abs(df.agent_reward.sum()) / len(df)
    return total


def main() -> None:
    fresh = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    init = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    bo = BayesianOptimization(f=None, pbounds=PBOUNDS, random_state=7,
                              verbose=0)
    sims = 0
    for it in range(fresh):
        params = bo.suggest()
        PARAMS_JSON.write_text(json.dumps(
            {k: round(float(v), 4) for k, v in params.items()}))
        target = run_screen(f"n1bo_it{it:02d}")
        sims += SCREEN_SIMS
        bo.register(params, -target)
        best = bo.max
        print(f"  it{it:02d} sum={target:.6f} params="
              + json.dumps({k: round(float(v), 4)
                            for k, v in params.items()})
              + f" best_sum={-best['target']:.6f} sims={sims}", flush=True)
        BO_STATE.write_text(json.dumps({
            "it": it, "sims": sims,
            "best_params": best["params"], "best_sum": -best["target"],
        }))
    best = bo.max
    print("BEST_SUM", -best["target"], "PARAMS",
          json.dumps(best["params"]), "SIMS", sims)


if __name__ == "__main__":
    main()
