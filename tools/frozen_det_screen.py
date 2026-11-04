# -*- coding: utf-8 -*-
"""Deterministic-mean screen driver (plan-20 P20-1, P20-11).

Serial full-episode screen: deterministic-mean residual arms vs
force_rule (pure rule) arms on fixed pairs, honest bill. Committed
successor of the P10 scratch driver (/tmp/p10_screen.py).

Usage: frozen_det_screen.py [--weights PT] [--tag NAME]
  [--seed 31] [--steps 8640]
Prints per-pair reward_sum lines; gaps are computed downstream.
"""
import sys

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS")
sys.path.insert(0, "/tmp/pkg")

from pathlib import Path

from smart_buildings import evaluate as E
from smart_buildings.evaluate import POLICY_MAP
import tools.skynet_rollout as roll
from tools.honest_bill import apply_honest_bill

PAIRS = [("building_train_8", "fairbanks_8"),
         ("building_train_7", "phoenix_3")]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/tmp/opencode/sb-rl/skynet.pt")
    ap.add_argument("--tag", default="det")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--steps", type=int, default=8640)
    a = ap.parse_args()
    apply_honest_bill()
    for b, w in PAIRS:
        d = Path("smart_buildings/training_scenarios") / b / w
        for tag, kw in (("base", dict(force_rule=True)),
                        (a.tag, dict(weights=a.weights,
                                     deterministic=True))):
            env = E.load_hybrid_environment(d / "config.gin")
            env.metrics_path = None
            meta = E.create_building_meta_data(env, d)
            spec = env.action_spec()
            base = E.initialize_policy(POLICY_MAP["occ_gated7"], meta,
                                       spec)
            tr = roll.rollout_env(env, meta, spec, base, a.steps,
                                  seed=a.seed, **kw)
            print(f"{b}/{w} {tag}: steps={len(tr['r'])} "
                  f"reward_sum={float(sum(tr['r'])):.4f}", flush=True)


if __name__ == "__main__":
    main()
