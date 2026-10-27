"""Frozen-observer parameter/forcing/noise stress audit; synthetic only."""

import argparse
from copy import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_system_modes_check import initial_state, system_basis
from tools.thermal_wall_modes_check import ModalResponse
from tools.thermal_state_observer_check import DelayedObserver, HORIZONS, _metrics, command_path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ('mixed', 'late_fixed_pulse', 'late_air_pulse')


@dataclass(frozen=True)
class Mismatch:
    name: str
    capacity_scale: float = 1.
    wall_link_scale: float = 1.
    exterior_scale: float = 1.
    airflow_scale: float = 1.
    hidden_room_1_heat_W: float = 0.
    sensor_noise_std_K: float = 0.


CASES = (
    Mismatch('nominal'),
    Mismatch('capacity_minus20', capacity_scale=.8),
    Mismatch('capacity_plus20', capacity_scale=1.2),
    Mismatch('wall_links_minus20', wall_link_scale=.8),
    Mismatch('wall_links_plus20', wall_link_scale=1.2),
    Mismatch('exterior_minus20', exterior_scale=.8),
    Mismatch('exterior_plus20', exterior_scale=1.2),
    Mismatch('airflow_minus20', airflow_scale=.8),
    Mismatch('airflow_plus20', airflow_scale=1.2),
    Mismatch('hidden_heat_5W', hidden_room_1_heat_W=5.),
    Mismatch('sensor_noise_0p05K', sensor_noise_std_K=.05),
    Mismatch('combined', capacity_scale=1.2, wall_link_scale=.8, exterior_scale=1.2,
             airflow_scale=.8, hidden_room_1_heat_W=5., sensor_noise_std_K=.05),
)


def perturb_network(nominal, spec):
    """Positive passive network perturbations, NOT a new material calibration.

    Wall-link and film conductances are independently scaled network terms;
    this is not claimed to equal changing one material conductivity in GIN.
    The nominal matrices/basis/observer are never modified.
    """
    scales = [spec.capacity_scale, spec.wall_link_scale, spec.exterior_scale, spec.airflow_scale]
    if (not np.isfinite([*scales, spec.hidden_room_1_heat_W, spec.sensor_noise_std_K]).all()
            or min(scales) <= 0 or spec.sensor_noise_std_K < 0):
        raise ValueError('Require positive finite network scales and nonnegative noise')
    truth = copy(nominal)
    truth.capacity = nominal.capacity.copy() * spec.capacity_scale
    truth.exposure = nominal.exposure.copy() * spec.exterior_scale
    walls = np.zeros(len(nominal.capacity), dtype=bool)
    for coordinate, i in nominal.index.items():
        walls[i] = coordinate not in nominal.mass_index
    links = -nominal.laplacian.copy()
    np.fill_diagonal(links, 0)
    links[walls[:, None] | walls[None, :]] *= spec.wall_link_scale
    truth.laplacian = np.diag(links.sum(axis=1)) - links
    return truth


def realized_input(command, step, spec):
    actual = dict(command)
    actual['conductance'] = tuple(spec.airflow_scale * np.asarray(command['conductance']))
    heat = np.asarray(command['fixed'], dtype=float).copy()
    if 96 <= step < 180:
        heat[0] += spec.hidden_room_1_heat_W
    actual['fixed'] = tuple(heat)
    return actual


def generate_mismatch(nominal, protocol, steps, spec, *, diagnostics=None):
    """Truth and noisy received data are separated from observer commands."""
    commands = command_path(protocol, steps)  # validates bound before any solve
    truth = perturb_network(nominal, spec)
    reference = ModalResponse(truth, np.eye(len(truth.capacity)))
    state = initial_state(truth, protocol)
    minimum, maximum = float(state.min()), float(state.max())
    temperatures = [truth.sensed.T @ state]
    for step, command in enumerate(commands):
        state = reference.advance(state, realized_input(command, step, spec))
        minimum, maximum = min(minimum, float(state.min())), max(maximum, float(state.max()))
        temperatures.append(truth.sensed.T @ state)
    clean = np.array([temperatures[0], *temperatures[:-1]])
    received = clean + np.random.default_rng(202610).normal(0, spec.sensor_noise_std_K, clean.shape)
    received[1] = received[0]  # same latched reset measurement, not new noise
    if diagnostics is not None:
        diagnostics['true_physical_temperature_range_K'] = [minimum, maximum]
    return received, clean, commands


def evaluate_received(nominal, model, received, clean_targets, commands, *, prefix):
    """No truth parameters/fields/actual heat are supplied to the observer."""
    steps = len(commands)
    if (not 1 <= prefix < steps <= 288 or steps - prefix < max(HORIZONS)
            or len(received) != steps + 1 or np.shape(clean_targets) != np.shape(received)):
        raise ValueError('Invalid bounded observation/target/input window')
    filters = {'kalman': DelayedObserver(nominal, model),
               'open_loop': DelayedObserver(nominal, model, correction=False)}
    groups = {name: {h: {kind: [] for kind in (*filters, 'persistence')} for h in HORIZONS}
              for name in ('all', 'pulse_window', 'hidden_heat_window')}
    minimum, maximum = float('inf'), float('-inf')
    innovations = []
    for t, temperatures in enumerate(received):
        for kind, observer in filters.items():
            observer.receive(temperatures, completed_input=commands[t - 2] if t >= 2 else None)
            if kind == 'kalman':
                reconstructed = model.basis @ observer.state
                minimum = min(minimum, float(reconstructed.min()))
                maximum = max(maximum, float(reconstructed.max()))
                innovations.append(float(np.max(np.abs(model.s @ observer.state - temperatures))))
        if prefix <= t <= steps - max(HORIZONS):
            path = commands[t - 1:t - 1 + max(HORIZONS)]
            predictions = {kind: observer.forecast(path) for kind, observer in filters.items()}
            active_groups = ['all']
            if 84 <= t < 120:
                active_groups.append('pulse_window')
            if 84 <= t < 192:
                active_groups.append('hidden_heat_window')
            for h in HORIZONS:
                target = clean_targets[t + h]  # never passed into receive/forecast
                errors = {kind: np.abs(values[h - 1] - target).tolist() for kind, values in predictions.items()}
                errors['persistence'] = np.abs(temperatures - target).tolist()
                for group in active_groups:
                    for kind, values in errors.items():
                        groups[group][h][kind].extend(values)
    return dict(states=len(filters['kalman'].state),
        reconstructed_temperature_range_K=[minimum, maximum],
        maximum_post_update_sensor_residual_K=max(innovations),
        metrics={name: {f'h{h}': {kind: _metrics(values) if values else None for kind, values in errors.items()}
                        for h, errors in horizons.items()} for name, horizons in groups.items()})


def check(*, steps=288, prefix=72):
    if (not isinstance(steps, int) or not isinstance(prefix, int)
            or not 1 <= prefix < steps <= 288 or steps - prefix < max(HORIZONS)):
        raise ValueError('Require a bounded response and complete forecast horizon')
    nominal = fixture(room_rows=6, room_columns=5)
    # All bases, priors and noise assumptions are fixed BEFORE generating truth.
    models = {count + 5: ModalResponse(nominal, system_basis(nominal, count, adjoint=True)[0])
              for count in (8, 16, 24)}
    rows = []
    for spec in CASES:
        for protocol in PROTOCOLS:
            diagnostics = {}
            received, clean, commands = generate_mismatch(nominal, protocol, steps, spec, diagnostics=diagnostics)
            for count, model in models.items():
                row = evaluate_received(nominal, model, received, clean, commands, prefix=prefix)
                row.update(mismatch=spec.name, protocol=protocol)
                # Generation diagnostics are attached AFTER evaluation; never
                # passed into the causal observer or used as clipping bounds.
                row.update(diagnostics)
                low, high = diagnostics['true_physical_temperature_range_K']
                estimated_low, estimated_high = row['reconstructed_temperature_range_K']
                row['estimated_excursion_outside_true_global_range_K'] = max(
                    low - estimated_low, estimated_high - high, 0.)
                rows.append(row)
    summary = {}
    for count in models:
        selected = [row for row in rows if row['states'] == count]
        summary[str(count)] = {}
        for h in ('h2', 'h6', 'h12'):
            ratios = [(row['metrics']['all'][h]['kalman']['mae_K']
                       / max(row['metrics']['all'][h]['persistence']['mae_K'], 1e-12), row)
                      for row in selected]
            ratio, worst = max(ratios, key=lambda item: item[0])
            summary[str(count)][h] = dict(cases=len(selected), cases_beating_persistence=sum(r < 1 for r, _ in ratios),
                worst_relative_to_persistence=ratio,
                worst_case=[worst['mismatch'], worst['protocol']])
    files = ['tools/thermal_mismatch_check.py', 'tools/thermal_state_observer_check.py',
             'tools/thermal_system_modes_check.py', 'tools/thermal_grid_reference.py',
             'tools/thermal_wall_modes_check.py', 'tools/thermal_reduction_check.py']
    return dict(protocol='Synthetic plant perturbed; nominal bases, observer, priors and noise frozen; '
                'no refitting or truth-parameter/state updates; forecasts use nominal conditional future inputs',
        steps_per_response=steps, prefix_observations=prefix, noise_seed=202610,
        evaluation_target='Noiseless latched physical zone temperature, not future sensor noise',
        fixed_observer_assumed_sensor_std_K=.0001, fixed_process_std_K=.01,
        perturbations=[asdict(spec) for spec in CASES],
        origin_windows=dict(pulse_window=[84, 120], hidden_heat_window=[84, 192]),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--prefix', type=int, default=72)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.prefix < args.steps <= 288 or args.steps - args.prefix < max(HORIZONS):
        parser.error('Invalid bounded budget/prefix/horizon')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report')
    result = check(steps=args.steps, prefix=args.prefix)
    print(json.dumps(result['summary'], indent=2, allow_nan=False))
    for row in result['rows']:
        if row['states'] == 13:
            print(row['mismatch'], row['protocol'], {h: {kind: round(value['mae_K'], 6)
                for kind, value in row['metrics']['all'][h].items()} for h in ('h2', 'h12')})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
