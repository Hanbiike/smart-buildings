"""Validate physical zone dynamics on a bounded archived workday window.

Fits use only the initial observation prefix; parameters stay frozen during
validation. Latent mass and equipment calibration assimilate only observations
already received. No simulator or control policy is invoked.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.baselines.equipment_observer import EquipmentAssumptions, EquipmentObserver  # noqa: E402
from smart_buildings.baselines.zone_thermal import (  # noqa: E402
    ThermalSeries, condition_mass, condition_mass_kalman, fit_zone, forecast_path,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'smart_buildings/results'


def fit_prefix(metadata, obs, continuous, discrete, *, two_state, max_evaluations=40,
               mass_observer='boundary', passive_multistep=False):
    """The caller supplies ONLY its training prefix, not validation arrays."""
    if not 10 <= len(obs) <= 288 or len(continuous) != len(obs) or len(discrete) != len(obs):
        raise ValueError('A complete prefix of 10..288 observations is required')
    observer = EquipmentObserver(metadata, assumptions=EquipmentAssumptions())
    comfort_idx = observer.positions['comfort_mode_now']
    room_ahu = {room: ahu for ahu, rooms in observer.handlers.items() for room in rooms}
    records = {room: [] for room in observer.rooms}
    for t, observation in enumerate(obs):
        transition = observer.observe(observation)
        if transition is not None:
            forcing = transition.forcing
            for room, duty in forcing.zones.items():
                records[room].append((transition.start_temperatures[room],
                                      transition.end_temperatures[room],
                                      forcing.outdoor_temperature[room_ahu[room]],
                                      duty.supply_temperature, duty.air_flow, duty.reheat_power,
                                      float(obs[forcing.control_step, comfort_idx])))
        observer.set_action(continuous[t], discrete[t])
    median_area = float(np.median(list(observer.areas.values())))
    fits = {room: fit_zone(ThermalSeries(*np.array(rows).T), two_state=two_state,
                           area_scale=observer.areas[room] / median_area,
                           max_evaluations=max_evaluations, mass_observer=mass_observer,
                           passive_multistep=passive_multistep)
            for room, rows in records.items()}
    return fits


def _metrics(errors):
    if not errors:
        return None
    values = np.array(errors)
    means = values.mean(axis=0)
    return dict(samples=len(errors), model_mae_K=float(means[0]),
                persistence_mae_K=float(means[1]), linear_drift_mae_K=float(means[2]),
                model_p95_K=float(np.percentile(values[:, 0], 95)),
                persistence_p95_K=float(np.percentile(values[:, 1], 95)),
                relative_reduction=float(1 - means[0] / means[1]) if means[1] > 1e-9 else None)


def evaluate_arrays(metadata, obs, continuous, discrete, *, two_state=True,
                    fit_observations=72, horizons=(1, 6, 12), command_path='recorded',
                    max_evaluations=40, mass_observer='boundary', passive_multistep=False):
    n = len(obs)
    if (not 10 <= fit_observations < n <= 288 or len(continuous) != n or len(discrete) != n
            or not horizons or min(horizons) < 1 or max(horizons) >= n - fit_observations):
        raise ValueError('Invalid bounded window, training prefix or horizons')
    if command_path not in ('recorded', 'held'):
        raise ValueError('Unknown command-path protocol')
    if mass_observer not in ('boundary', 'kalman'):
        raise ValueError('Unknown mass-state observer')
    fits = fit_prefix(metadata, obs[:fit_observations], continuous[:fit_observations],
                      discrete[:fit_observations], two_state=two_state,
                      max_evaluations=max_evaluations, mass_observer=mass_observer,
                      passive_multistep=passive_multistep)
    observer = EquipmentObserver(metadata, assumptions=EquipmentAssumptions())
    comfort_idx = observer.positions['comfort_mode_now']
    occupancy_idx = observer.positions.get('num_occupants')
    room_ahu = {room: ahu for ahu, rooms in observer.handlers.items() for room in rooms}
    masses = {room: fit.initial_mass for room, fit in fits.items()}
    variances = {room: 4.0 for room in observer.rooms}  # fixed 2 K prior std
    minimum_mass, maximum_mass = min(masses.values()), max(masses.values())
    envelope_low, envelope_high = masses.copy(), masses.copy()
    mass_envelope_violations = 0
    maximum_envelope_violation = 0.0
    errors = {h: {group: [] for group in ('all', 'active', 'passive', 'occupied', 'empty',
                                         'identified_actuators', 'uncertain_actuators')}
              for h in horizons}
    uncertain_origins = 0
    for t in range(n):
        transition = observer.observe(obs[t])
        for room in observer.rooms:
            temperature = observer.frame['zones'][room]['temperature']
            envelope_low[room] = min(envelope_low[room], temperature)
            envelope_high[room] = max(envelope_high[room], temperature)
        if transition is not None:
            for room in observer.rooms:
                fit = fits[room]
                if mass_observer == 'kalman':
                    duty = transition.forcing.zones[room]
                    masses[room], variances[room] = condition_mass_kalman(
                        fit.parameters, transition.start_temperatures[room], masses[room], variances[room],
                        transition.end_temperatures[room],
                        outdoor=transition.forcing.outdoor_temperature[room_ahu[room]],
                        supply=duty.supply_temperature, flow=duty.air_flow, reheat=duty.reheat_power,
                        internal=fit.internal_power(float(obs[transition.forcing.control_step, comfort_idx])))
                else:
                    masses[room] = condition_mass(fit.parameters, masses[room],
                                                  transition.end_temperatures[room],
                                                  transition.forcing.outdoor_temperature[room_ahu[room]])
            minimum_mass = min(minimum_mass, min(masses.values()))
            maximum_mass = max(maximum_mass, max(masses.values()))
            for room, mass in masses.items():
                # In this topology the mass has no independent heat source or
                # outside path; a passive trajectory stays in this convex hull.
                violation = max(envelope_low[room] - mass, mass - envelope_high[room], 0)
                if violation > 1e-6:
                    mass_envelope_violations += 1
                    maximum_envelope_violation = max(maximum_envelope_violation, violation)
        if fit_observations <= t < n - 1:
            length = min(max(horizons), n - t - 1)
            commands = [(continuous[t + i] if command_path == 'recorded' else continuous[t],
                         discrete[t + i] if command_path == 'recorded' else discrete[t])
                        for i in range(length - 1)]
            path = forecast_path(observer, fits, masses, commands, length,
                                 pending_comfort=float(obs[t - 1, comfort_idx]),
                                 current_comfort=float(obs[t, comfort_idx]))
            uncertain_origins += bool(path.uncertain_damper_floors or path.uncertain_pressure_floors)
            for room in observer.rooms:
                now = observer.frame['zones'][room]['temperature']
                idx = observer.zone_indices[room]['zone_air_temperature_sensor']
                stats = metadata['observation_normalization_metadata']['zone_air_temperature_sensor']
                previous = float(obs[t - 1, idx]) * stats['sample_std'] + stats['sample_mean']
                active = observer.frame['handlers'][room_ahu[room]]['supervisor_run_command'] > .5
                for h in horizons:
                    if t + h >= n:
                        continue
                    actual = float(obs[t + h, idx]) * stats['sample_std'] + stats['sample_mean']
                    value = (abs(path.temperatures[h - 1][room] - actual), abs(now - actual),
                             abs(now + h * (now - previous) - actual))
                    errors[h]['all'].append(value)
                    # Conservative origin-wide label: uncertainty anywhere in
                    # the longest path keeps this origin out of the clean set.
                    uncertain = bool(path.uncertain_damper_floors or path.uncertain_pressure_floors)
                    errors[h]['uncertain_actuators' if uncertain else 'identified_actuators'].append(value)
                    errors[h]['active' if active else 'passive'].append(value)
                    if occupancy_idx is not None:
                        errors[h]['occupied' if obs[t, occupancy_idx] > 0 else 'empty'].append(value)
        observer.set_action(continuous[t], discrete[t])
    return dict(
        model=('two_state' if two_state else 'one_state') + ('_kalman' if mass_observer == 'kalman' else ''), observations=n,
        mass_observer=mass_observer, estimated_mass_range_K=[minimum_mass, maximum_mass],
        passive_multistep=passive_multistep,
        mass_passive_envelope_violations=mass_envelope_violations,
        maximum_mass_envelope_violation_K=maximum_envelope_violation,
        state_estimation_assumptions=(dict(initial_mass_variance_K2=4.0,
                                           air_process_std_K_per_step=.05,
                                           mass_process_std_K_per_step=.01,
                                           measured_air_is_exact=True)
                                     if mass_observer == 'kalman' else None),
        fitting_mass_update=mass_observer + '; parameters frozen before validation',
        fit_observations=fit_observations, frozen_thermal_parameters=True,
        conditional_recorded_commands=command_path == 'recorded',
        future_weather='held at origin', future_comfort='held at origin',
        future_water='causal rate projection', step_seconds=300,
        area_weighting=observer.area_weighting,
        occupied_observations=int(np.count_nonzero(obs[:, occupancy_idx] > 0))
        if occupancy_idx is not None else None,
        origins_with_uncertain_actuator_floors=uncertain_origins,
        solver_failures=sum(not fit.diagnostics['solver_success'] for fit in fits.values()),
        fitted_zones={str(room): asdict(fit) for room, fit in fits.items()},
        metrics={f'h{h}': {group: _metrics(values) for group, values in groups.items()}
                 for h, groups in errors.items()},
        control_integration=False,
    )


def evaluate_case(scenario, *, policy='dsat_gen31', steps=288, offset=576, **kwargs):
    if not 11 <= steps <= 288 or offset < 0:
        raise ValueError('Require 11 <= steps <= 288 and offset >= 0')
    base = RESULTS / scenario
    metadata = json.loads((base / 'scenario_meta.json').read_text())
    map_path = ROOT / 'smart_buildings/training_scenarios' / scenario / 'zone_map.npy'
    if map_path.exists():
        metadata['zone_map'] = np.load(map_path)
    with np.load(base / f'nn_data_{policy}/obs_dump.npz') as dump:
        window = slice(offset, offset + steps)
        arrays = [dump[name][window] for name in ('obs', 'action_continuous', 'action_discrete')]
    if any(len(array) != steps for array in arrays):
        raise ValueError('Saved dump lacks the complete requested window')
    row = evaluate_arrays(metadata, *arrays, **kwargs)
    row.update(scenario=scenario, policy=policy, offset=offset)
    return row


def summarize(rows, horizons):
    result = {}
    for model in sorted({row['model'] for row in rows}):
        selected = [row for row in rows if row['model'] == model]
        result[model] = {}
        for h in horizons:
            table = {}
            for group in ('all', 'active', 'occupied', 'identified_actuators', 'uncertain_actuators'):
                metrics = [row['metrics'][f'h{h}'][group] for row in selected
                           if row['metrics'][f'h{h}'][group] is not None]
                if not metrics:
                    table[group] = None
                    continue
                mean = float(np.mean([m['model_mae_K'] for m in metrics]))
                baseline = float(np.mean([m['persistence_mae_K'] for m in metrics]))
                table[group] = dict(cases=len(metrics), model_mae_K=mean,
                                    persistence_mae_K=baseline,
                                    relative_reduction=1 - mean / baseline if baseline > 1e-9 else None)
            result[model][f'h{h}'] = table
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', default=[
        'building_train_1/fairbanks_1', 'building_train_2/fairbanks_2',
        'building_train_8/fairbanks_8', 'building_train_1/atlanta_2', 'building_train_8/singapore_7'])
    parser.add_argument('--all-scenarios', action='store_true')
    parser.add_argument('--models', nargs='+', choices=['one_state', 'two_state'], default=['one_state', 'two_state'])
    parser.add_argument('--policy', default='dsat_gen31')
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--offset', type=int, default=576)
    parser.add_argument('--fit-observations', type=int, default=72)
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 2, 3, 6, 12])
    parser.add_argument('--command-path', choices=['recorded', 'held'], default='recorded')
    parser.add_argument('--mass-observer', choices=['boundary', 'kalman'], default='boundary')
    parser.add_argument('--passive-multistep', action='store_true')
    parser.add_argument('--max-evaluations', type=int, default=40)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.passive_multistep and args.mass_observer != 'boundary':
        parser.error('Passive multistep requires --mass-observer boundary')
    if not 10 <= args.fit_observations < args.steps <= 288 or args.offset < 0:
        parser.error('Invalid observation budget or fitting prefix')
    if min(args.horizons) < 1 or max(args.horizons) >= args.steps - args.fit_observations:
        parser.error('Horizons must fit in the validation window')
    if not 1 <= args.max_evaluations <= 80:
        parser.error('Fitter budget must be between 1 and 80')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite an existing report')
    if args.all_scenarios:
        args.scenarios = sorted(str(p.parent.parent.relative_to(RESULTS))
                                for p in RESULTS.glob(f'*/*/nn_data_{args.policy}/obs_dump.npz'))
    if not args.scenarios:
        parser.error('No saved cases found')
    rows = []
    print('OFFLINE thermal validation, not a policy score; command path:', args.command_path, flush=True)
    for scenario in args.scenarios:
        for model in args.models:
            row = evaluate_case(scenario, policy=args.policy, steps=args.steps, offset=args.offset,
                                two_state=model == 'two_state', fit_observations=args.fit_observations,
                                horizons=tuple(args.horizons), command_path=args.command_path,
                                max_evaluations=args.max_evaluations, mass_observer=args.mass_observer,
                                passive_multistep=args.passive_multistep)
            rows.append(row)
            print(row['model'], scenario, {f'h{h}': round(row['metrics'][f'h{h}']['all']['model_mae_K'], 4)
                                    for h in args.horizons}, 'solver failures', row['solver_failures'], flush=True)
    report = dict(rows=rows, summary=summarize(rows, args.horizons))
    print(json.dumps(report['summary'], indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            json.dump(report, file, indent=2, allow_nan=False)
        print('Report:', args.output)


if __name__ == '__main__':
    main()
