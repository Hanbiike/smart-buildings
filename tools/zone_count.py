#!/usr/bin/env python3
"""Zone-count monster check for generated buildings (plan 18/K2).

Loads one weather per building and reports AHUs / total zones /
max zones per AHU. Buildings are valid iff max zones per AHU <= 10;
anything near the 125-zone xfer_4 monster is out of spec for suites.

Usage: zone_count.py <scenarios_dir> <building> [<building> ...]
Example: zone_count.py smart_buildings/transfer_scenarios
             building_xfer_5 building_xfer_6
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smart_buildings import evaluate as E


def main(argv=None):
    a = argv or sys.argv[1:]
    sdir, buildings = a[0], a[1:]
    for b in buildings:
        d = sorted((Path(sdir) / b).iterdir())[0]
        t0 = time.time()
        env = E.load_hybrid_environment(d / "config.gin")
        az = E.get_ahu_zone_assignments(env, rename_rooms_to_vavs=False)
        nz = sum(len(v) for v in az.values())
        mx = max(len(v) for v in az.values())
        el = time.time() - t0
        print(f"{b}: AHUs={len(az)} zones={nz} "
              f"max_zones_per_ahu={mx} load_s={el:.0f}", flush=True)


if __name__ == "__main__":
    main()
