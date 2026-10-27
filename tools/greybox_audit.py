"""Audit actuator alignment and online water-rate calibration on saved dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.baselines.causal_greybox import CausalGreyBoxModel  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / 'smart_buildings' / 'results'


def audit(scenario, policy, steps):
    base = RESULTS / scenario
    meta = json.loads((base / 'scenario_meta.json').read_text())
    model = CausalGreyBoxModel(meta, calibrate_plant=True, sensor_delay=1,
                               fit_window=steps + 1)  # calibration only; no fit
    with np.load(base / f'nn_data_{policy}' / 'obs_dump.npz') as dump:
        obs = dump['obs'][:steps]
        continuous = dump['action_continuous'][:steps]
        discrete = dump['action_discrete'][:steps]
    report = {'scenario': scenario, 'policy': policy, 'observations': len(obs)}
    for key, getter in [('supply_air_damper_percentage_command', model._damper_native),
                        ('reheat_valve_percentage_command', model._reheat_native)]:
        timing = {}
        # Equal sample counts for all candidate observation delays.
        length = len(obs) - 2
        for lag in (0, 1, 2):
            errors = []
            for room in model.rooms:
                index = model.actions[room][key]
                measured = np.array([getter(row, room) for row in obs[lag:lag + length]])
                command = np.array([model._native(row[index], key)
                                    for row in continuous[:length]])
                errors.extend(abs(measured - command))
            timing[str(lag)] = {'native_mae': float(np.mean(errors)),
                                'match_fraction': float(np.mean(np.array(errors) < 1e-5))}
        report[key] = timing

    hold_errors, rate_errors, floor_errors = [], [], []
    for t in range(len(obs)):
        model.update(obs[t])
        model.set_action(continuous[t], discrete[t])
        if t + 1 >= len(obs) or t < min(144, steps // 2):
            continue
        for room in model.rooms:
            index = model.actions[room]['supply_air_damper_percentage_command']
            command = model._native(continuous[t, index], 'supply_air_damper_percentage_command')
            predicted = max(command, model.damper_floor.get(room, 0))
            floor_errors.append(abs(predicted - model._damper_native(obs[t + 1], room)))
        indices = meta.get('water_indices', [])
        if indices and model.water_temp_idx is not None:
            water = model._water_temp(obs[t])
            target = model._native(continuous[t, indices[0]], 'supply_water_temperature_setpoint')
            gap = target - water
            change = (min(gap, model.water_heating_step) if gap > 0
                      else max(gap, -model.water_cooling_step))
            actual = model._water_temp(obs[t + 1])
            hold_errors.append(abs(water - actual))
            rate_errors.append(abs(water + change - actual))
    report['causal_calibration'] = {
        'damper_mae_after_learned_floor': float(np.mean(floor_errors)),
        'water_hold_mae_K': float(np.mean(hold_errors)) if hold_errors else None,
        'water_rate_mae_K': float(np.mean(rate_errors)) if rate_errors else None,
        'water_samples': len(hold_errors),
        'final_heating_K_per_step': model.water_heating_step,
        'final_cooling_K_per_step': model.water_cooling_step,
        'learned_damper_floors': sorted(set(round(v, 6) for v in model.damper_floor.values())),
        'learned_pressure_floors_Pa': sorted(set(model.pressure_floor.values())),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', default=[
        'building_train_2/fairbanks_2', 'building_train_8/singapore_7'])
    parser.add_argument('--policy', default='dsat_gen31')
    parser.add_argument('--steps', type=int, default=288)
    args = parser.parse_args()
    if not 4 <= args.steps <= 288:
        parser.error('require 4 <= steps <= 288')
    print(json.dumps([audit(s, args.policy, args.steps) for s in args.scenarios],
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
