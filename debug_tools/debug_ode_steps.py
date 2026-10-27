# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Debug: run ode_thermal for 2 days, print internals around 06:00."""

from pathlib import Path

import numpy as np

from smart_buildings.evaluate import (
    create_building_meta_data,
    load_hybrid_environment,
)
from smart_buildings.baselines.ode_thermal import OdeThermalPolicy

config_path = Path(
    'smart_buildings/training_scenarios/building_train_1/singapore_8/config.gin'
)
env = load_hybrid_environment(config_path)
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = create_building_meta_data(env, config_path)
spec = env.action_spec()
policy = OdeThermalPolicy(meta, spec)

obs, _ = env.reset()
for step in range(576):
    action = policy.get_action(obs)
    cont = action['continuous_action']
    hour, _ = policy._hour_and_day(np.asarray(obs, dtype=np.float32).ravel())
    if 5.5 <= hour <= 8.0 and step % 1 == 0:
        a0 = policy.ahu_controllers[0]
        t_idx = a0['temp_idx']
        zones0 = a0['zones'][:4]
        zt = [
            round(policy._zone_temperature(obs, policy.room_temp_obs[r]), 2)
            for r in zones0
        ]
        r0 = zones0[0]
        trusted = policy._model_trusted(r0)
        pred0 = None
        if trusted:
            pred0 = round(
                policy._predict_horizon(
                    r0, zt[0], policy._read_ambient(obs),
                    float(obs[policy.occupancy_idx]) if policy.occupancy_idx is not None else 0.0,
                    0.3, 295.0, 0.0,
                ), 2
            )
        # reheat valve levels of the first zones
        rh = [
            round(float(cont[policy.room_reheat_action_idx[r]]), 2)
            if r in policy.room_reheat_action_idx else None
            for r in zones0
        ]
        print(
            f"step {step:4d} h={hour:5.2f} u0={float(cont[t_idx]):+.2f} "
            f"int0={policy.pid_state[0]['integral']:6.2f} "
            f"water={float(cont[policy.water_temp_idx]):+.1f} "
            f"T={zt} reheat={rh}"
        )
        print(
            f"         trusted0={trusted} pred0={pred0} "
            f"resid0={policy.residual[r0]:.3f} bias0={policy.bias[r0]:+.3f} "
            f"hrz0={policy.horizon_residual[r0]:.3f} amb={policy._read_ambient(obs):.1f}"
        )
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
