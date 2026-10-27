# -*- coding: utf-8 -*-
"""Run several (policy, building, weather) evaluations in parallel.

Each job is a screen of at most 288 steps; the official evaluate module is invoked in its
own process so the results directory stays race-free (one subdir per
scenario/policy). Progress prints as jobs finish; a final table summarizes
every episode in the LB metric (abs(mean reward) = |total| / steps).
"""
import argparse
import concurrent.futures
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REWARD_RE = re.compile(r"(\S+)/(\S+):\s*$")
REWARD_LINE_RE = re.compile(r"^\s*(\S+)\s*:\s*reward\s*=\s*(-?[\d.]+)\s*$")


def read_exact_return(results_dir, building, weather, policy, expected_steps):
    """Never fall back to rounded stdout or silently accept a partial run."""
    path = Path(results_dir) / building / weather / policy / 'summary.json'
    summary = json.loads(path.read_text())
    if summary['total_steps'] != expected_steps:
        raise ValueError(f'Expected {expected_steps} steps, got {summary["total_steps"]}')
    reward = float(summary['total_reward'])
    if not math.isfinite(reward):
        raise ValueError('Nonfinite episode return')
    return reward


def one_job(args_pack) -> tuple[str, float | None, str]:
    policy, building, weather, steps, timeout, dump_obs, results_dir, *options = args_pack
    start_timestamp = options[0] if options else None
    cmd = [
        sys.executable, "-m", "smart_buildings.evaluate",
        "--policies", policy, "--buildings", building, "--weathers", weather,
        "--results-dir", str(results_dir),
    ]
    if steps:
        cmd += ["--steps", str(steps)]
    if dump_obs:
        cmd += ["--dump-obs"]
    if start_timestamp is not None:
        cmd += ['--start-timestamp', start_timestamp]
    label = f"{building}/{weather}:{policy}"
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=Path(__file__).resolve().parent.parent,
        )
        out = proc.stdout
        if proc.returncode != 0:
            return label, None, f"exit {proc.returncode}: {proc.stderr[-400:]}"
    except subprocess.TimeoutExpired:
        return label, None, f"timeout after {timeout}s"
    try:
        reward = read_exact_return(results_dir, building, weather, policy, steps)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return label, None, f'invalid summary: {error}'
    elapsed = time.time() - t0
    note = f"{elapsed:.0f}s"
    return label, reward, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--weathers", required=True,
                    help="comma list or 'all' (per building!_see training_scenarios)")
    ap.add_argument("--steps", type=int, default=288)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--dump-obs", action="store_true",
                     help="pass --dump-obs to each eval job (record obs/action trajectories)")
    ap.add_argument('--results-dir', type=Path,
                    help='New output directory; existing paths are rejected')
    ap.add_argument('--start-timestamp', help='Fresh-reset start passed to every policy')
    args = ap.parse_args()
    if not 1 <= args.steps <= 288:
        ap.error("--steps must be between 1 and 288 (local simulation budget)")
    if args.results_dir is None:
        parent = Path(__file__).resolve().parent.parent / 'smart_buildings/results/screens'
        parent.mkdir(parents=True, exist_ok=True)
        args.results_dir = Path(tempfile.mkdtemp(prefix='screen_', dir=parent))
    else:
        args.results_dir = args.results_dir.resolve()
        if args.results_dir.exists():
            ap.error('Refusing to reuse an existing results directory')
        args.results_dir.mkdir(parents=True)
    print(f'Results: {args.results_dir}', flush=True)

    policies = list(dict.fromkeys(args.policies.split(",")))
    buildings = list(dict.fromkeys(args.buildings.split(",")))
    scenarios_dir = Path("smart_buildings/training_scenarios")

    jobs = []
    skipped = []
    for building in buildings:
        if args.weathers == "all":
            weathers = sorted(p.name for p in (scenarios_dir / building).iterdir()
                              if p.is_dir())
        else:
            weathers = list(dict.fromkeys(args.weathers.split(",")))
        for weather in weathers:
            if not (scenarios_dir / building / weather).is_dir():
                skipped.append(f"{building}/{weather}")
                continue
            for policy in policies:
                jobs.append((policy, building, weather, args.steps, args.timeout,
                              args.dump_obs, args.results_dir, args.start_timestamp))
    if skipped:
        print(f"skipped missing scenarios: {', '.join(sorted(set(skipped)))}\n",
              flush=True)

    print(f"{len(jobs)} jobs, concurrency {args.concurrency}\n", flush=True)
    results: list[tuple[str, float | None, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(one_job, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            label, reward, note = fut.result()
            results.append((label, reward, note))
            shown = f"{reward:.2f}" if reward is not None else note
            print(f"[{len(results)}/{len(jobs)}] {label:<44} {shown}", flush=True)

    print("\n==== SUMMARY (LB metric: |total| / steps) ====")
    rows = []
    for policy, building, weather, _, _, _, _, _ in jobs:
        reward = next((r for lab, r, _ in results if lab == f"{building}/{weather}:{policy}"), None)
        rows.append((building, weather, policy, reward))
    for building in sorted({r[0] for r in rows}):
        for weather in sorted({r[1] for r in rows if r[0] == building}):
            per_policy = [(p, r) for b, w, p, r in rows
                          if b == building and w == weather]
            line = f"{building}/{weather:<13} "
            vals = []
            for p, r in per_policy:
                if r is None:
                    vals.append(f"{p}=?")
                    continue
                steps = args.steps or 8640
                vals.append(f"{p}={abs(r) / steps:.5f}")
            print(line + "  ".join(vals))


if __name__ == "__main__":
    main()
