# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Debug: replay Fairbanks; log per-zone pred vs actual and command details."""
from pathlib import Path

import numpy as np

from smart_buildings.evaluate import (
    create_building_meta_data,
    load_hybrid_environment,
)
from smart_buildings.baselines import ode_thermal as ot

PRED_LOG = []
CMD_LOG = []

_orig_predict = ot.OdeThermalPolicy._predict_horizon
def _predict(self, room, temperature, ambient, occupancy, damper, supply, reheat_hold):
    out = _orig_predict(self, room, temperature, ambient, occupancy, damper, supply, reheat_hold)
    PRED_LOG.append((self.step_count, room, temperature, out,
                     float(self.bias.get(room, 0.0)), bool(self._model_trusted(room))))
    return out
ot.OdeThermalPolicy._predict_horizon = _predict

_orig_cmd = ot.OdeThermalPolicy._command_zone_actions
def _cmd(self, continuous, room, cold_error, hot_error):
    valve = _orig_cmd(self, continuous, room, cold_error, hot_error)
    idx = self.room_damper_action_idx.get(room)
    damper = float(continuous[idx]) if idx is not None else None
    CMD_LOG.append((self.step_count, room, cold_error, hot_error, valve, damper))
    return valve
ot.OdeThermalPolicy._command_zone_actions = _cmd

config_path = Path(
    'smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin'
)
env = load_hybrid_environment(config_path)
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = create_building_meta_data(env, config_path)
spec = env.action_spec()
policy = ot.OdeThermalPolicy(meta, spec)

obs, _ = env.reset()
for step in range(1500):
    action = policy.get_action(obs)
    obs = np.asarray(obs, dtype=np.float32).ravel()
    if step < 1499:
        obs, *_ = env.step(action)

# Steps 1248-1368 = episode-day 5 hours ~8-14 (day5 starts 4*288=1152)
sel_steps = set(range(1248, 1370))
by_room = {}
for s, room, temp, pred, bias, trusted in PRED_LOG:
    if s in sel_steps:
        a, b = by_room.get(room, (0.0, 0.0))
        by_room[room] = (a + (pred - temp), b + 1)
rows = []
for room, (diff_sum, n) in sorted(by_room.items()):
    diff = diff_sum / max(n, 1)
    rows.append((room, round(float(diff), 3), n))
print("per-zone mean(pred - actual) over steps 1248-1368 (day5 8-14h):")
print(f"{'room':>4} {'pred-act':>9} {'n':>4}")
for room, diff, _ in rows:
    mark = ' <<<' if abs(diff) > 0.2 else ''
    print(f"{room:>4} {diff:>9} {mark}")

errs = [(s, room, ce, he, v) for s, room, ce, he, v, dmp in CMD_LOG if s in sel_steps]
rooms_ce = {}
for s, room, ce, he, v in errs:
    a, b = rooms_ce.get(room, (0.0, 0))
    rooms_ce[room] = (a + ce, b + 1)
print("\nper-zone mean cold_error (valve driver) same window:")
for room, (ce_sum, _) in sorted(rooms_ce.items()):
    if ce_sum > 0.5:
        print(f"room {room}: mean cold_err {ce_sum:6.2f}")
