# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Debug: compare pid_water vs ode_thermal AHU commands at 06:00 Singapore."""

from pathlib import Path

import numpy as np

from smart_buildings.evaluate import (
    create_building_meta_data,
    load_hybrid_environment,
)
from smart_buildings.baselines.ode_thermal import OdeThermalPolicy
from smart_buildings.baselines.pid import PIDPolicy

config_path = Path(
    'smart_buildings/training_scenarios/building_train_1/singapore_8/config.gin'
)

for policy_class in (OdeThermalPolicy, PIDPolicy):
    env = load_hybrid_environment(config_path)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, config_path)
    spec = env.action_spec()
    policy = policy_class(meta, spec)
    print(f"\n===== {policy_class.__name__} =====")
    if policy_class is OdeThermalPolicy:
        t_idx_all = [
            c['temp_idx'] for c in policy.ahu_controllers
        ]
        p_idx_all = [
            c['pressure_idx'] for c in policy.ahu_controllers
        ]
        d_idx_all = [
            c['run_discrete_idx'] for c in policy.ahu_controllers
        ]
    else:
        t_idx_all = [c['temp_idx'] for c in policy.ahu_controllers]
        p_idx_all = [c['pressure_idx'] for c in policy.ahu_controllers]
        d_idx_all = [c['run_discrete_idx'] for c in policy.ahu_controllers]
    obs, _ = env.reset()
    for step in range(360):
        action = policy.get_action(obs)
        cont = action['continuous_action']
        disc = action['discrete_action']
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        if hasattr(policy, '_hour_and_day'):
            hour, _ = policy._hour_and_day(obs_arr)
        else:
            hour, _ = policy._get_hour_and_day(obs_arr)
        if 5.8 <= hour <= 6.8 and step % 2 == 0:
            us = [float(cont[i]) for i in t_idx_all]
            ps = [
                float(cont[i]) if i is not None else None for i in p_idx_all
            ]
            ds = [int(disc[i]) for i in d_idx_all]
            w = (
                float(cont[policy.water_temp_idx])
                if policy.water_temp_idx is not None
                else None
            )
            print(
                f"step {step:3d} h={hour:5.2f} u={us} "
                f"p={ps} disc={ds} water={w}"
            )
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
