#!/usr/bin/env python3
"""Generate transfer-validation suite: unseen buildings x unseen weather.

Buildings: seeds 50000+i (train uses 0,1,8,3001,4007,9008), paper-like
ranges, materials rotated over the 4 named constructions.
Weather: minneapolis_1/_5 (unseen STATION) + atlanta_4/singapore_2
(unseen MONTHS of seen stations).
Output: smart_buildings/transfer_scenarios/<building>/<weather>/config.gin
"""
import dataclasses
import os
import random
import sys
from pathlib import Path

import yaml

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from smart_buildings.smart_control.utils.config_generator import generate_config
from smart_buildings.smart_control.utils.scenario_generator import (
    CONSTRUCTIONS,
    generate_scenario_from_config,
    load_scenario_from_parts,
)

BUILDINGS_DIR = os.path.join(ROOT, "smart_buildings", "buildings_xfer")
SCENARIOS_DIR = os.path.join(ROOT, "smart_buildings", "transfer_scenarios")
WEATHER_DIR = os.path.join(ROOT, "smart_buildings", "weather")

SHAPES = ["rectangle", "h_shape", "t_shape", "u_shape", "composite"]
CONSTRUCTIONS_ORDER = ["heavy_masonry", "medium_concrete", "light_steel_frame", "curtain_wall"]
WEATHERS = ["minneapolis_1", "minneapolis_5", "atlanta_4", "singapore_2"]


def main():
    os.makedirs(BUILDINGS_DIR, exist_ok=True)
    rng = random.Random(777)
    building_paths = []
    for i in range(4):
        seed = 50000 + i
        cfg = generate_config(
            name=f"building_xfer_{i + 1}",
            seed=seed,
            building_shape=SHAPES[i % len(SHAPES)] if i < 4 else rng.choice(SHAPES),
            width=(250, 700),
            height=(180, 450),
            num_floors=(2, 4),
            min_room_size=(30, 55),
            room_size_span=(15, 25),
            num_ahus=(2, 8),
            occupants_per_zone=(1, 6),
        )
        cfg["ahu"]["max_zones_per_ahu"] = 10
        con = CONSTRUCTIONS_ORDER[i % len(CONSTRUCTIONS_ORDER)]
        cfg["materials"] = dataclasses.asdict(CONSTRUCTIONS[con])
        cfg["construction_name"] = con
        path = os.path.join(BUILDINGS_DIR, f"building_xfer_{i + 1}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        building_paths.append(path)
        fp = cfg["floor_plan"]
        print(f"{cfg['name']}: seed={seed} shape={fp['building_shape']} "
              f"floors={fp['num_floors']} ahus={cfg['ahu']['num_ahus']} mat={con}",
              flush=True)

    n = 0
    for bp in building_paths:
        for w in WEATHERS:
            wp = os.path.join(WEATHER_DIR, f"{w}.yaml")
            sc = load_scenario_from_parts(bp, wp, output_base_dir=SCENARIOS_DIR)
            result = generate_scenario_from_config(sc)
            n += 1
            print(f"[{n}/16] {result['output_dir']}", flush=True)
    print(f"TRANSFER_SUITE_OK {n} scenarios in {SCENARIOS_DIR}")


if __name__ == "__main__":
    main()
