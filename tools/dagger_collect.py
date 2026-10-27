# -*- coding: utf-8 -*-
"""DAgger collection: student (slot-hybrid) drives, teacher labels every obs.

Writes results/<b>/<w>/nn_data_<tag>/obs_dump.npz with obs + TEACHER
actions, in the same layout tools/slot_train.py consumes. slot_train.py
accepts a comma-separated --teacher list so rounds aggregate.
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.baselines.slot_hybrid import SlotHybridPolicy  # noqa: E402
from smart_buildings.baselines.occ_gated4 import OccGated4Policy  # noqa: E402

SCEN_DIR = PROJECT_ROOT / "smart_buildings" / "training_scenarios"
RESULTS = PROJECT_ROOT / "smart_buildings" / "results"

SUBSET_8 = [
    ("building_train_1", "fairbanks_1"),
    ("building_train_1", "atlanta_2"),
    ("building_train_1", "singapore_8"),
    ("building_train_1", "phoenix_5"),
    ("building_train_8", "fairbanks_8"),
    ("building_train_8", "atlanta_1"),
    ("building_train_8", "singapore_7"),
    ("building_train_8", "phoenix_4"),
]

def _collect_proc(scenario, gin_path, student_weights, steps, tag):
    """One process per scenario: init exactly its env, collect, save."""
    from smart_buildings import evaluate as E
    env = E.load_hybrid_environment(Path(gin_path))
    env.metrics_path = None
    meta = E.create_building_meta_data(env, Path(gin_path).parent)
    spec = env.action_spec()
    tracker = E.SimulationTracker(env, capture_snapshots=False)
    stu = SlotHybridPolicy(meta, spec, student_weights)
    tea = OccGated4Policy(meta, spec)
    stu.reset()
    tea.reset()
    tracker.reset()
    obs = tracker.last_obs
    D = int(np.asarray(obs).shape[0])
    nc = int(meta["n_continuous_action"])
    nd = int(meta["n_discrete_action"])
    O = np.zeros((steps, D), np.float32)
    C = np.zeros((steps, nc), np.float32)
    G = np.zeros((steps, nd), np.int64)
    R = np.zeros(steps, np.float32)
    n = 0
    for t in range(steps):
        ah = E.validate_action(E.get_policy_action(stu, obs), spec)
        at = E.validate_action(tea.get_action(obs), spec)
        O[t] = np.asarray(obs, dtype=np.float32)
        C[t] = at["continuous_action"]
        G[t] = at["discrete_action"]
        obs, _, term, trunc, _ = tracker.step(ah)
        R[t] = float(tracker.rewards[-1])
        n = t + 1
        if term or trunc:
            break
    b, w = scenario
    outdir = RESULTS / b / w / f"nn_data_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "obs_dump.npz", obs=O[:n],
        action_continuous=C[:n], action_discrete=G[:n], rewards=R[:n],
        state=np.zeros((n, nc), np.float32))
    print(f"  {b}/{w}: n={n} student_loss={-R[:n].mean():.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--tag", default="dag1")
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenarios", default="subset8")
    args = ap.parse_args()
    if args.scenarios == "subset8":
        scenarios = SUBSET_8
    elif args.scenarios == "all":
        scenarios = []
        for b in sorted(SCEN_DIR.glob("building_train_*")):
            for w in sorted(b.iterdir()):
                if w.is_dir():
                    scenarios.append((b.name, w.name))
    else:
        scenarios = [tuple(p.split("/")) for p in args.scenarios.split(",")]
    print(f"scenarios: {len(scenarios)} workers: {args.workers}",
          flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    procs = []
    for s in scenarios:
        gin = str(SCEN_DIR / s[0] / s[1] / "config.gin")
        p = ctx.Process(target=_collect_proc,
                        args=(s, gin, args.student, args.steps, args.tag))
        procs.append(p)
    # Throttle to --workers concurrent processes.
    running = []
    pending = list(procs)
    while pending or running:
        while pending and len(running) < args.workers:
            p = pending.pop(0)
            p.start()
            running.append(p)
        for p in running:
            p.join(timeout=5)
        alive = [p for p in running if p.is_alive()]
        for p in running:
            if not p.is_alive() and p.exitcode != 0:
                print(f"worker exitcode={p.exitcode}", flush=True)
        running = alive
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
