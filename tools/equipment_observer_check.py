"""Causal equipment-input audit on saved windows; never invokes a simulator."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.baselines.equipment_observer import (  # noqa: E402
    EquipmentAssumptions, EquipmentObserver, UnobservedCapacityError,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'smart_buildings/results'


def _errors(values):
    array = np.asarray(values, dtype=float)
    return (dict(samples=len(values), mae=float(np.mean(np.abs(array))),
                 rmse=float(np.sqrt(np.mean(np.square(array)))),
                 maximum=float(np.max(np.abs(array)))) if values else
            dict(samples=0, mae=None, rmse=None, maximum=None))


def check(scenario, *, policy='dsat_gen31', steps=288, offset=576, warmup=72,
          assumptions=None):
    if not 3 <= steps <= 288 or offset < 0 or not 0 <= warmup < steps - 1:
        raise ValueError('Require 3 <= steps <= 288, offset >= 0 and 0 <= warmup < steps-1')
    base = RESULTS / scenario
    metadata = json.loads((base / 'scenario_meta.json').read_text())
    map_path = ROOT / 'smart_buildings/training_scenarios' / scenario / 'zone_map.npy'
    if map_path.exists():
        metadata['zone_map'] = np.load(map_path)
    with np.load(base / f'nn_data_{policy}/obs_dump.npz') as dump:
        window = slice(offset, offset + steps)
        obs = dump['obs'][window]
        continuous = dump['action_continuous'][window]
        discrete = dump['action_discrete'][window]
    if any(len(array) != steps for array in (obs, continuous, discrete)):
        raise ValueError('Saved dump does not contain the full requested window')
    parameters = assumptions or EquipmentAssumptions()
    observer = EquipmentObserver(metadata, assumptions=parameters)
    flow_errors, identified_flow_errors, water_errors, water_hold_errors = [], [], [], []
    projected_power = []
    skipped = transitions = uncertain_projections = uncertain_flow_samples = 0
    for t in range(steps):
        transition = observer.observe(obs[t])
        if transition is not None:
            transitions += 1
            assert transition.forcing.control_step == transition.start_observation - 1
        if t >= warmup and t + 1 < steps:
            try:
                proposal = observer.project_action(continuous[t], discrete[t])
            except UnobservedCapacityError:
                skipped += 1
            else:
                if proposal.uncertain_damper_floors or proposal.uncertain_pressure_floors:
                    uncertain_projections += 1
                for ahu in observer.ahu_ids:
                    actual = float(obs[t + 1, observer.ahu_indices[ahu]['supply_air_flowrate_sensor']])
                    error = proposal.handler_flows[ahu] - actual
                    flow_errors.append(error)
                    if (ahu in proposal.uncertain_pressure_floors or any(
                            room in proposal.uncertain_damper_floors for room in observer.handlers[ahu])):
                        uncertain_flow_samples += 1
                    else:
                        identified_flow_errors.append(error)
                projected_power.extend(duty.total_power for duty in proposal.zones.values())
            actual_water = float(obs[t + 1, observer.water_idx])
            water_errors.append(observer.project_water_sensor(continuous[t]) - actual_water)
            water_hold_errors.append(observer.frame['water'] - actual_water)
        observer.set_action(continuous[t], discrete[t])
    occupancy_idx = observer.positions.get('num_occupants')
    return dict(
        scenario=scenario, policy=policy, steps=steps, offset=offset, warmup=warmup,
        occupied_observations=int(np.count_nonzero(obs[:, occupancy_idx] > 0))
        if occupancy_idx is not None else None,
        return_air_weighting=observer.area_weighting,
        geometry_mapping='regular connected-component labels assumed' if 'zone_map' in metadata
        else 'unavailable; equal-area proxy',
        assumptions=asdict(parameters),
        delayed_transitions=transitions, skipped_unobserved_capacity=skipped,
        handler_flow_m3_s=_errors(flow_errors),
        identified_floor_flow_m3_s=_errors(identified_flow_errors),
        projections_with_uncertain_floors=uncertain_projections,
        uncertain_floor_flow_samples=uncertain_flow_samples,
        water_sensor_K=_errors(water_errors), water_hold_K=_errors(water_hold_errors),
        learned_capacities_m3_s=observer.capacities,
        learned_damper_floors=sorted(set(round(value, 6) for value in observer.damper_floors.values())),
        learned_pressure_floors_Pa=sorted(set(observer.pressure_floors.values())),
        observed_water_step_K=dict(heating=observer.water_heat_step, cooling=observer.water_cool_step),
        projected_zone_power_W=(dict(minimum=float(np.min(projected_power)),
                                     maximum=float(np.max(projected_power))) if projected_power else None),
        power_validation='NOT validated against measured heat; lagged-temperature proxy and global equipment priors',
        temperature_forecast_validation='NOT performed; this adapter does not predict zone temperature',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', default=[
        'building_train_1/fairbanks_1', 'building_train_2/fairbanks_2',
        'building_train_8/fairbanks_8', 'building_train_1/atlanta_2',
        'building_train_8/singapore_7'])
    parser.add_argument('--all-scenarios', action='store_true')
    parser.add_argument('--policy', default='dsat_gen31')
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--offset', type=int, default=576)
    parser.add_argument('--warmup', type=int, default=72)
    parser.add_argument('--output', type=Path, help='New JSON report; existing files are never overwritten')
    parser.add_argument('--summary', action='store_true', help='Print compact metrics; the saved report stays complete')
    args = parser.parse_args()
    if not 3 <= args.steps <= 288 or args.offset < 0 or not 0 <= args.warmup < args.steps - 1:
        parser.error('Require 3 <= steps <= 288, offset >= 0 and 0 <= warmup < steps-1')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite an existing report')
    if args.all_scenarios:
        args.scenarios = sorted(str(path.parent.parent.relative_to(RESULTS))
                                for path in RESULTS.glob(f'*/*/nn_data_{args.policy}/obs_dump.npz'))
    if not args.scenarios:
        parser.error('No saved cases found')
    rows = [check(scenario, policy=args.policy, steps=args.steps,
                  offset=args.offset, warmup=args.warmup) for scenario in args.scenarios]
    text = json.dumps(rows, indent=2, allow_nan=False)
    if args.summary:
        for row in rows:
            print(row['scenario'], 'flow_MAE_m3_s=', row['handler_flow_m3_s']['mae'],
                  'water_MAE_K=', row['water_sensor_K']['mae'],
                  'water_hold_MAE_K=', row['water_hold_K']['mae'],
                  'skipped=', row['skipped_unobserved_capacity'])
    else:
        print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            file.write(text + '\n')
        if args.summary:
            print('Report:', args.output)


if __name__ == '__main__':
    main()
