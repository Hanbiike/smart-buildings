# -*- coding: utf-8 -*-
"""P23-6: offline base-policy replay writer (B_<base>.npy).

Replays a base policy (default occ_gated7) over stored teacher obs
and writes continuous actions next to each dump, in the layout
tools/slot_train.py --residual-base consumes. Pure offline (no sim):
building meta and action spec come from a fresh env init (no
rollout steps are simulated).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings import evaluate as E  # noqa: E402

RESULTS = PROJECT_ROOT / "smart_buildings" / "results"
SCEN_DIR = PROJECT_ROOT / "smart_buildings" / "training_scenarios"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="all",
                    help="'all' or comma list building/weather")
    ap.add_argument("--teacher", default="occ_econdbhhot9")
    ap.add_argument("--tag", default=None,
                    help="dump dir tag (default nn_data_<teacher>)")
    ap.add_argument("--base", default="occ_gated7")
    ap.add_argument("--out", default=None,
                    help="output name (default B_<base>.npy)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.pairs == "all":
        pairs = []
        for b in sorted(SCEN_DIR.glob("building_train_*")):
            for w in sorted(b.iterdir()):
                if w.is_dir():
                    pairs.append(f"{b.name}/{w.name}")
    else:
        pairs = args.pairs.split(",")
    tag = args.tag or f"nn_data_{args.teacher}"
    out_name = args.out or f"B_{args.base}.npy"
    cls = E.POLICY_MAP[args.base]
    for pair in pairs:
        ddir = RESULTS / pair / tag
        dump = ddir / "obs_dump.npz"
        if not dump.exists():
            print(f"  {pair}: no dump, skip", flush=True)
            continue
        b, w = pair.split("/")
        gin = SCEN_DIR / b / w / "config.gin"
        env = E.load_hybrid_environment(gin)
        env.metrics_path = None
        meta = E.create_building_meta_data(env, gin.parent)
        spec = env.action_spec()
        z = np.load(dump)
        obs, cont = z["obs"], z["action_continuous"]
        pol = E.initialize_policy(cls, meta, spec)
        pol.reset()
        T = obs.shape[0]
        B = np.zeros((T, cont.shape[1]), np.float32)
        for t in range(T):
            a = E.validate_action(E.get_policy_action(pol, obs[t]), spec)
            B[t] = a["continuous_action"]
        np.save(ddir / out_name, B)
        print(f"  {pair}: T={T} -> {out_name}", flush=True)


if __name__ == "__main__":
    main()
