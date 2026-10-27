# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Debug: replay Fairbanks and dump internals during day-5 work hours (8-14h).

Hypothesis under test: reheat deadband (0.15 K) + tiny dampers leave zones
hovering just under the comfort floor (294 K) all workday -> -30..41 regret
per workday, which the 576-step slices (days 1-2, zero deficit) never saw.
"""
from pathlib import Path

import numpy as np

from smart_buildings.evaluate import (
    create_building_meta_data,
    load_hybrid_environment,
)
from smart_buildings.baselines.ode_thermal import OdeThermalPolicy

config_path = Path(
    'smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin'
)
env = load_hybrid_environment(config_path)
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = create_building_meta_data(env, config_path)
spec = env.action_spec()
policy = OdeThermalPolicy(meta, spec)

obs, _ = env.reset()
for step in range(1500):
    action = policy.get_action(obs)
    cont = action['continuous_action']
    obs = np.asarray(obs, dtype=np.float32).ravel()
    hour, day = policy._hour_and_day(obs)
    # day 5 (index 4) starts at step 4*288 = 1152
    if step >= 1240 and step <= 1420 and step % 2 == 0:
        zones0 = policy.ahu_controllers[0]['zones'][:3]
        amb = policy._read_ambient(obs)
        spt_low, spt_high = policy.comfort_temp_window
        parts = []
        for r in zones0:
            oi = policy.room_temp_obs.get(r, None)
            if oi is None:
                continue
            t = policy._zone_temperature(obs, oi)
            cold = max(0.0, spt_low - 0.05 - t)
            valve = cont[policy.room_reheat_action_idx[r]] if r in policy.room_reheat_action_idx else None
            damp = cont[policy.room_damper_action_idx[r]] if r in policy.room_damper_action_idx else None
            parts.append((t, cold, valve, damp))
        u0 = float(cont[policy.ahu_controllers[0]['temp_idx']])
        p0 = float(cont[policy.ahu_controllers[0]['pressure_idx']]) if policy.ahu_controllers[0]['pressure_idx'] is not None else None
        water = float(cont[policy.water_temp_idx]) if policy.water_temp_idx is not None else None
        occ = float(obs[policy.occupancy_idx]) if policy.occupancy_idx is not None else -1
        zs = '  '.join(f"T{t:.2f}/c{c:.2f}/v{float(v):+.1f}/d{float(d):+.2f}" for t, c, v, d in parts)
        print(f"{step:5d} d{day} h{hour:4.1f} amb{amb:6.1f} occ{occ:4.1f} "
              f"u{u0:+.2f} p{p0:+.2f} w{water:+.1f} | {zs}")
    if step % 288 == 287:
        dN = step // 288 + 1
        # day summary: rewards are printed only from the csv; keep it light
    if step < 1499:
        pass
    obs, *_ = env.step(action)

# hourly mean zone temps for days 4 and 5
print("\nhourly mean zone T, day-by-day (K):")
ts = []
obs, _ = env.reset()
