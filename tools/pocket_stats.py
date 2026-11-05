#!/usr/bin/env python3
"""Repro stats for round-5 angle-4 loss pockets (P22-10 close).

Reads the 30 train occ9 reward_components.csv + timeseries.csv under
smart_buildings/results/ and prints: total/mean loss, weighted component
splits, cold unoccupied gas, hour-6+7 comfort share. Expected (11-05):
total -2276.3, comfort -114.1, h67 76.7%, h6 57.7%.

Usage: ./.venv/bin/python -B tools/pocket_stats.py
"""

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "smart_buildings" / "results"


def main() -> None:
    tot = comf = en = co = unocc = h67 = h6 = 0.0
    seen = set()
    files = sorted(glob.glob(
        str(RES / "*" / "*" / "occ_econdbhhot9" / "reward_components.csv")))
    for f in files:
        key = "/".join(f.split("/")[-4:-2])
        if key in seen:
            continue
        seen.add(key)
        df = pd.read_csv(f)
        ts = pd.read_csv(f.replace("reward_components.csv", "timeseries.csv"),
                         usecols=["timestamp", "occupancy"])
        d = pd.to_datetime(ts.timestamp)
        df["h"] = d.dt.hour
        df["occ"] = ts.occupancy
        tot += df.agent_reward.sum()
        comf += df.normalized_productivity_regret.sum()
        en += df.normalized_energy_cost.sum()
        co += df.normalized_carbon_emission.sum()
        u = df[df.occ < 0.5]
        unocc += (0.2 * u.normalized_energy_cost
                  + 0.2 * u.normalized_carbon_emission).sum()
        h67 += df[df.h.isin([6, 7])].normalized_productivity_regret.sum()
        h6 += df[df.h == 6].normalized_productivity_regret.sum()
    n = len(seen)
    print(f"pairs: {n}")
    print(f"total={tot:.1f} mean_loss={abs(tot) / (n * 8640):.5f}")
    print(f"comfort_w={0.6 * comf:.1f} energy_w={0.2 * en:.1f} "
          f"carbon_w={0.2 * co:.1f}")
    print(f"unocc_gas_w={unocc:.1f}")
    print(f"h67={h67:.1f} ({100 * h67 / comf:.1f}% of comfort) "
          f"h6={h6:.1f} ({100 * h6 / comf:.1f}%)")


if __name__ == "__main__":
    main()
