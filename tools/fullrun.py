#!/usr/bin/env python3
"""Scratch full-episode parallel driver (evaluate.py has no step cap).

Usage: fullrun.py <results_dir> <steps> <concurrency> <timeout> <policies_csv> <pairs_csv>
pairs_csv: building/weather,building/weather,...
"""
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def one_job(pack):
    policy, building, weather, steps, timeout, results_dir, dump = pack
    cmd = [sys.executable, "-m", "smart_buildings.evaluate",
           "--policies", policy, "--buildings", building, "--weathers", weather,
           "--results-dir", str(results_dir), "--steps", str(steps)]
    if dump:
        cmd.append("--dump-obs")
    label = f"{building}/{weather}:{policy}"
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT)
        if proc.returncode != 0:
            return label, None, f"exit {proc.returncode}: {proc.stderr[-300:]}"
    except subprocess.TimeoutExpired:
        return label, None, f"timeout after {timeout}s"
    el = time.time() - t0
    # read exact loss
    import pandas as pd
    csv = Path(results_dir) / building / weather / policy / "reward_components.csv"
    try:
        df = pd.read_csv(csv)
        loss = abs(df.agent_reward.sum()) / len(df)
        return label, loss, f"{el:.0f}s n={len(df)}"
    except Exception as e:
        return label, None, f"bad csv: {e}"


def main():
    results_dir, steps, conc, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    policies = sys.argv[5].split(",")
    pairs = [p.split("/") for p in sys.argv[6].split(",")]
    dump = len(sys.argv) > 7 and sys.argv[7] == "dump"
    jobs = [(p, b, w, steps, timeout, results_dir, dump) for b, w in pairs for p in policies]
    print(f"{len(jobs)} jobs, concurrency {conc}", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(one_job, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            label, loss, note = fut.result()
            results.append((label, loss, note))
            shown = f"{loss:.6f}" if loss is not None else note
            print(f"[{len(results)}/{len(jobs)}] {label:<46} {shown} {note if loss is not None else ''}", flush=True)
    print("\n==== SUMMARY ====")
    for b, w in pairs:
        line = f"{b}/{w:<13} "
        for p in policies:
            r = next((x for x in results if x[0] == f"{b}/{w}:{p}"), None)
            line += f"  {p}=" + (f"{r[1]:.6f}" if r and r[1] is not None else "?")
        print(line)


if __name__ == "__main__":
    main()
