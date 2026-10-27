#!/usr/bin/env python3
"""Analyze transfer-suite results: means, breakdowns, worst runs.

Usage: xfer_analyze.py <results_dir>
"""
import glob
import os
import sys

import pandas as pd

ROOT = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(ROOT, "*", "*", "*", "reward_components.csv")):
    parts = f.split(os.sep)
    b, w, p = parts[-4], parts[-3], parts[-2]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"BAD {b}/{w}:{p}: {e}")
        continue
    n = len(df)
    loss = abs(df.agent_reward.sum()) / n
    rows.append(dict(
        building=b, weather=w, policy=p, loss=loss, n=n,
        prod_regret=df.normalized_productivity_regret.mean(),
        energy=df.normalized_energy_cost.mean(),
        carbon=df.normalized_carbon_emission.mean(),
        gas=df.natural_gas_energy_cost.sum(),
        elec=df.electricity_energy_cost.sum(),
        worst_step=df.agent_reward.min(),
    ))

r = pd.DataFrame(rows)
print(f"runs: {len(r)}")
print("\n== per-policy mean loss ==")
print(r.groupby("policy").loss.agg(["mean", "min", "max", "count"]).round(6).to_string())
print("\n== per-policy component means (normalized) ==")
print(r.groupby("policy")[["prod_regret", "energy", "carbon"]].mean().round(5).to_string())
print("\n== occ vs guard vs baselines per building ==")
pv = r.pivot_table(index="building", columns="policy", values="loss", aggfunc="mean")
print(pv.round(6).to_string())
print("\n== per weather ==")
pv = r.pivot_table(index="weather", columns="policy", values="loss", aggfunc="mean")
print(pv.round(6).to_string())
print("\n== worst 10 runs ==")
w = r.sort_values("loss", ascending=False).head(10)
print(w[["building", "weather", "policy", "loss", "prod_regret", "energy", "worst_step"]].round(5).to_string(index=False))
print("\nXFER_ANALYZE_OK")
