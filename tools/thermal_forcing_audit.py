"""Offline oracle-forcing ablation; never an online forecast or policy score."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.baselines.equipment_balance import ZoneThermalParameters
from smart_buildings.baselines.equipment_observer import EquipmentAssumptions, EquipmentObserver
from smart_buildings.baselines.zone_thermal import ThermalFit, advance_zone, condition_mass, forecast_path
from tools.modular_thermal_check import ROOT, RESULTS, _metrics


def oracle_path(fits, masses, temperatures, transitions, room_ahu, comfort):
    """Future recorded forcing and comfort, but NO future zone-state correction."""
    air, mass = dict(temperatures), dict(masses)
    path = []
    for transition, flag in zip(transitions, comfort):
        forcing = transition.forcing
        for room, fit in fits.items():
            duty = forcing.zones[room]
            air[room], mass[room] = advance_zone(
                fit.parameters, air[room], mass[room],
                outdoor=forcing.outdoor_temperature[room_ahu[room]],
                supply=duty.supply_temperature, flow=duty.air_flow,
                reheat=duty.reheat_power, internal=fit.internal_power(flag))
        path.append(air.copy())
    return path


def audit(metadata, obs, continuous, discrete, fits, *, prefix=72, horizons=(1, 2, 6, 12)):
    n = len(obs)
    if not 10 <= prefix < n <= 288 or len(continuous) != n or len(discrete) != n:
        raise ValueError('Require a complete bounded archive window')
    if not horizons or min(horizons) < 1 or max(horizons) >= n - prefix:
        raise ValueError('Invalid horizons')
    # Explicit noncausal diagnostic tape, isolated from the forecast observer.
    tape_observer = EquipmentObserver(metadata, assumptions=EquipmentAssumptions())
    tape = {}
    for t in range(n):
        transition = tape_observer.observe(obs[t])
        if transition is not None:
            tape[t] = transition
        tape_observer.set_action(continuous[t], discrete[t])
    observer = EquipmentObserver(metadata, assumptions=EquipmentAssumptions())
    room_ahu = {room: ahu for ahu, rooms in observer.handlers.items() for room in rooms}
    comfort_idx = observer.positions['comfort_mode_now']
    masses = {room: fit.initial_mass for room, fit in fits.items()}
    errors = {h: {'projected': [], 'oracle': []} for h in horizons}
    for t in range(n):
        transition = observer.observe(obs[t])
        if transition is not None:
            for room, fit in fits.items():
                masses[room] = condition_mass(fit.parameters, masses[room],
                    transition.end_temperatures[room],
                    transition.forcing.outdoor_temperature[room_ahu[room]])
        if prefix <= t < n - max(horizons):
            length = max(horizons)
            commands = [(continuous[t + k], discrete[t + k]) for k in range(length - 1)]
            projected = forecast_path(observer, fits, masses, commands, length,
                pending_comfort=float(obs[t - 1, comfort_idx]), current_comfort=float(obs[t, comfort_idx]))
            future = [tape[t + k] for k in range(1, length + 1)]
            flags = [float(obs[x.forcing.control_step, comfort_idx]) for x in future]
            temperatures = {r: observer.frame['zones'][r]['temperature'] for r in observer.rooms}
            oracle = oracle_path(fits, masses, temperatures, future, room_ahu, flags)
            for h in horizons:
                for room, now in temperatures.items():
                    actual = tape[t + h].end_temperatures[room]
                    baseline = abs(now - actual)
                    for name, prediction in [('projected', projected.temperatures), ('oracle', oracle)]:
                        errors[h][name].append((abs(prediction[h - 1][room] - actual), baseline, baseline))
        observer.set_action(continuous[t], discrete[t])
    def metrics(values):
        result = _metrics(values)
        if result is not None:
            result.pop('linear_drift_mae_K')
        return result
    return {f'h{h}': {name: metrics(values) for name, values in groups.items()}
            for h, groups in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, default=RESULTS / 'modular_thermal_passive_multistep_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite a report')
    source = json.loads(args.fits.read_text())
    rows = []
    for row in source['rows']:
        if row['mass_observer'] != 'boundary' or not row.get('passive_multistep'):
            raise ValueError('Audit requires frozen passive multistep fits')
        scenario = row['scenario']
        base = RESULTS / scenario
        metadata = json.loads((base / 'scenario_meta.json').read_text())
        map_path = ROOT / 'smart_buildings/training_scenarios' / scenario / 'zone_map.npy'
        if map_path.exists():
            metadata['zone_map'] = np.load(map_path)
        fits = {}
        for room, values in row['fitted_zones'].items():
            values = dict(values)
            values['parameters'] = ZoneThermalParameters(**values['parameters'])
            fits[int(room)] = ThermalFit(**values)
        with np.load(base / f"nn_data_{row['policy']}/obs_dump.npz") as dump:
            window = slice(row['offset'], row['offset'] + row['observations'])
            arrays = [dump[k][window] for k in ('obs', 'action_continuous', 'action_discrete')]
        metrics = audit(metadata, *arrays, fits, prefix=row['fit_observations'])
        rows.append(dict(scenario=scenario, metrics=metrics))
        print(scenario, {h: {name: round(m['model_mae_K'], 4) for name, m in groups.items()}
                         for h, groups in metrics.items()}, flush=True)
    with args.output.open('x') as file:
        json.dump(dict(protocol='OFFLINE oracle future forcing, no future state correction; '
                       'same complete-horizon origins in both arms; frozen fits',
                       source_fits=str(args.fits), rows=rows), file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
