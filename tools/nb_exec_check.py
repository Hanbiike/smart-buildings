# -*- coding: utf-8 -*-
"""P23-8: local notebook exec-check (SubmissionPolicy from ipynb).

Execs the notebook controller cells (EDA + inference-server cells
skipped, kaggle_evaluation stubbed) and runs SubmissionPolicy on two
scenarios x --steps (default 500). The default episode start is a
Sunday 06:00, so 300+ steps cover shutdown (weekend) + night +
Monday prestart. Verifies: no exception, full step count, finite
loss. Results go to --results-dir (default ~/sim_scratch/nbcheck).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings import evaluate as E  # noqa: E402

# Whitelist: chain cell + assembled cell + SubmissionPolicy cell only
# (EDA and inference-server cells never execute locally).
KEEP_MARKS = ("Full controller chain",
              ">>> assembled occ cells",
              "class SubmissionPolicy")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--pairs", default="building_train_8/fairbanks_8,"
                    "building_train_8/singapore_7")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--results-dir", default=None)
    return ap.parse_args()


def load_policy(nb_path):
    nb = json.load(open(nb_path))
    stub = types.ModuleType("kaggle_evaluation")
    stub_srv = types.ModuleType("kaggle_evaluation."
                                "smart_buildings_inference_server")
    stub_srv.Policy = object
    stub.smart_buildings_inference_server = stub_srv
    sys.modules["kaggle_evaluation"] = stub
    sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = (
        stub_srv)
    ns = {}
    exec("import numpy as np", ns)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not any(m in src for m in KEEP_MARKS):
            continue
        exec(compile(src, nb_path, "exec"), ns)
    return ns["SubmissionPolicy"]


def main():
    args = parse_args()
    policy_cls = load_policy(args.notebook)
    print(f"SubmissionPolicy: {policy_cls.__mro__[1].__name__}", flush=True)
    results = (Path(args.results_dir) if args.results_dir
               else Path.home() / "sim_scratch" / "nbcheck")
    ok = True
    for pair in args.pairs.split(","):
        b, w = pair.split("/")
        scen = (PROJECT_ROOT / "smart_buildings" / "training_scenarios"
                / b / w)
        sdir = results / b / w
        sdir.mkdir(parents=True, exist_ok=True)
        summary = E.evaluate_policy_for_config(sdir, scen, "nbcheck",
                                               policy_cls, args.steps)
        import pandas as pd
        df = pd.read_csv(sdir / policy_cls.KEY / "reward_components.csv")
        n = len(df)
        loss = abs(df.agent_reward.sum()) / n
        import math
        good = n == args.steps and math.isfinite(loss)
        ok = ok and good
        print(f"{pair}: n={n} loss={loss:.6f} "
              f"{'OK' if good else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("exec-check FAILED")
    print("exec-check PASS", flush=True)


if __name__ == "__main__":
    main()
