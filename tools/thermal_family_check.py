"""Finite prefix-compatible parameter/state family; empirical envelopes only.

No calibrated confidence claim, true parameter access in selection, coefficient
updates after the prefix, environment rollout or participant policy.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_mismatch_check import CASES, Mismatch, generate_mismatch
from tools.thermal_prefix_identification_check import candidate, fit_prefix, prefix_residual
from tools.thermal_state_observer_check import DelayedObserver, HORIZONS, _metrics
from tools.thermal_system_modes_check import system_basis
from tools.thermal_wall_modes_check import ModalResponse

ROOT = Path(__file__).resolve().parents[1]
SCORE_FLOOR = .001  # K per horizon-normalized prefix residual, fixed in advance


def family_vectors():
    """Declared prior grid; missing combinations are not silently implied."""
    loads = ((0., 0.), (5., 0.), (-5., 0.), (0., 5.), (0., -5.))
    return [np.array([*np.log(scales), *(np.asarray(load) / 5)])
            for scales in product((.8, 1., 1.2), repeat=3) for load in loads]


def select_family(nominal, basis, received, commands):
    if (not 30 <= len(received) <= 72 or len(commands) != len(received)
            or np.shape(received) != (len(received), 2) or not np.isfinite(received).all()):
        raise ValueError('Selection requires only a finite 30..72-observation prefix')
    vectors = family_vectors()
    losses = []
    nominal_residual = prefix_residual(nominal, basis, received, commands, np.zeros(5))
    median = np.median(nominal_residual)
    scale = max(SCORE_FLOOR, float(1.4826 * np.median(np.abs(nominal_residual - median))))
    for vector in vectors:
        residual = prefix_residual(nominal, basis, received, commands, vector)
        losses.append(float(np.mean(residual ** 2)))
    losses = np.array(losses)
    threshold = float(losses.min() + (2 * scale) ** 2)
    indices = np.flatnonzero(losses <= threshold)
    weights = np.exp(-(losses[indices] - losses.min()) / scale ** 2)
    weights /= weights.sum()
    selected = [vectors[index] for index in indices]
    return selected, weights, dict(total_candidates=len(vectors), retained=len(indices),
        retained_indices=indices.tolist(), retained_vectors=[v.tolist() for v in selected],
        weights=weights.tolist(), prefix_losses=losses.tolist(), prefix_residual_scale_K=scale,
        acceptance_mse_threshold=threshold, effective_model_count=float(1 / np.sum(weights ** 2)),
        interpretation='Finite prefix compatibility heuristic, not posterior probabilities or confidence levels')


def forecast_distribution(observer, path):
    """Latent physical-output uncertainty under the fixed assumed Q/R.

    Excludes future measurement variance because the target is noiseless zone
    temperature. Does not account for missing physics or unmodeled future heat.
    """
    predictions = observer.forecast(path)
    covariance = observer.p.copy()
    deviations = []
    for inputs in path:
        f = observer.transition(inputs)
        covariance = f @ covariance @ f.T + observer.q
        variances = np.diag(observer.h @ covariance @ observer.h.T)
        deviations.append(np.sqrt(np.maximum(variances, 0)))
    return predictions, np.array(deviations)


def envelope_metrics(targets, lows, highs):
    targets, lows, highs = [np.asarray(values) for values in (targets, lows, highs)]
    gap = np.maximum(np.maximum(lows - targets, targets - highs), 0)
    return dict(samples=targets.size, coverage_fraction=float(np.mean(gap <= 1e-10)),
        mean_width_K=float(np.mean(highs - lows)), maximum_miss_K=float(gap.max()))


def evaluate(nominal, basis, received, clean, commands, *, prefix, vectors, weights, point):
    members = [candidate(nominal, basis, vector, commands) for vector in vectors]
    observers = [DelayedObserver(grid, model) for grid, model, _ in members]
    nominal_member = candidate(nominal, basis, np.zeros(5), commands)
    point_member = candidate(nominal, basis, point, commands)
    comparisons = {'nominal': (DelayedObserver(*nominal_member[:2]), nominal_member[2]),
                   'point_fit': (DelayedObserver(*point_member[:2]), point_member[2])}
    groups = {name: {h: dict(errors={key: [] for key in (*comparisons, 'family_mean', 'persistence')},
        targets=[], lower=[], upper=[], augmented_lower=[], augmented_upper=[]) for h in HORIZONS}
        for name in ('all', 'pulse_window')}
    minimum, maximum = float('inf'), float('-inf')
    steps = len(commands)
    for t, temperatures in enumerate(received):
        for observer, (_, _, path) in zip(observers, members):
            observer.receive(temperatures, completed_input=path[t - 2] if t >= 2 else None)
        for observer, path in comparisons.values():
            observer.receive(temperatures, completed_input=path[t - 2] if t >= 2 else None)
        mean_field = sum(weight * (observer.model.basis @ observer.state)
                         for weight, observer in zip(weights, observers))
        minimum, maximum = min(minimum, float(mean_field.min())), max(maximum, float(mean_field.max()))
        if prefix <= t <= steps - max(HORIZONS):
            forecasts, deviations = [], []
            for observer, (_, _, path) in zip(observers, members):
                prediction, std = forecast_distribution(observer, path[t - 1:t + 11])
                forecasts.append(prediction)
                deviations.append(std)
            forecasts, deviations = np.array(forecasts), np.array(deviations)
            mean = np.einsum('m,mhz->hz', weights, forecasts)
            lower, upper = forecasts.min(axis=0), forecasts.max(axis=0)
            augmented_lower = (forecasts - 2 * deviations).min(axis=0)
            augmented_upper = (forecasts + 2 * deviations).max(axis=0)
            points = {name: observer.forecast(path[t - 1:t + 11])
                      for name, (observer, path) in comparisons.items()}
            points['family_mean'] = mean
            active = ['all'] + (['pulse_window'] if 84 <= t < 120 else [])
            for h in HORIZONS:
                target = clean[t + h]
                for name in active:
                    row = groups[name][h]
                    for kind, prediction in points.items():
                        row['errors'][kind].extend(np.abs(prediction[h - 1] - target).tolist())
                    row['errors']['persistence'].extend(np.abs(temperatures - target).tolist())
                    for key, values in (('targets', target), ('lower', lower[h - 1]), ('upper', upper[h - 1]),
                                        ('augmented_lower', augmented_lower[h - 1]), ('augmented_upper', augmented_upper[h - 1])):
                        row[key].extend(values.tolist())
    result = {}
    for name, horizons in groups.items():
        result[name] = {}
        for h, values in horizons.items():
            if not values['targets']:
                result[name][f'h{h}'] = None
                continue
            result[name][f'h{h}'] = dict(point_errors={key: _metrics(errors) for key, errors in values['errors'].items()},
                family_envelope=envelope_metrics(values['targets'], values['lower'], values['upper']),
                covariance_augmented_envelope=envelope_metrics(values['targets'], values['augmented_lower'], values['augmented_upper']))
    return dict(metrics=result, weighted_reconstructed_field_range_K=[minimum, maximum])


def check(*, steps=288, prefix=72):
    if (not isinstance(steps, int) or not isinstance(prefix, int)
            or not 30 <= prefix <= 72 or not prefix + 12 <= steps <= 288):
        raise ValueError('Require a 30..72-observation prefix inside <=288 transitions')
    nominal = fixture(room_rows=6, room_columns=5)
    basis = system_basis(nominal, 8, adjoint=True)[0]
    specs = [spec for spec in CASES if spec.name in ('nominal', 'capacity_plus20', 'exterior_minus20',
             'airflow_minus20', 'hidden_heat_5W', 'sensor_noise_0p05K')]
    specs += [Mismatch('between_grid', capacity_scale=1.1, exterior_scale=.9, airflow_scale=1.1),
              Mismatch('airflow_outside_prior', airflow_scale=1.5)]
    rows = []
    for spec in specs:
        for protocol in ('mixed', 'late_air_pulse'):
            diagnostics = {}
            received, clean, commands = generate_mismatch(nominal, protocol, steps, spec, diagnostics=diagnostics)
            vectors, weights, selection = select_family(nominal, basis, received[:prefix], commands[:prefix])
            point, point_info = fit_prefix(nominal, basis, received[:prefix], commands[:prefix])
            row = evaluate(nominal, basis, received, clean, commands, prefix=prefix,
                           vectors=vectors, weights=weights, point=point)
            row.update(mismatch=spec.name, protocol=protocol, selection=selection, point_fit=point_info,
                       truth_evaluation_only=diagnostics)
            rows.append(row)
            m = row['metrics']['all']['h12']
            print(spec.name, protocol, 'retained', len(vectors), 'MAE',
                {k: round(v['mae_K'], 6) for k, v in m['point_errors'].items()},
                'envelope coverage/width', round(m['family_envelope']['coverage_fraction'], 4),
                round(m['family_envelope']['mean_width_K'], 6), flush=True)
    files = ['tools/thermal_family_check.py', 'tools/thermal_prefix_identification_check.py',
             'tools/thermal_state_observer_check.py', 'tools/thermal_mismatch_check.py',
             'tools/thermal_grid_reference.py', 'tools/thermal_system_modes_check.py',
             'tools/thermal_wall_modes_check.py', 'tools/thermal_reduction_check.py']
    return dict(protocol='Finite parameter/state family selected on prefix only; membership and weights '
                'frozen thereafter; causal state updates; nominal conditional future inputs; no environment or policy',
        steps_per_response=steps, prefix_observations=prefix, states_per_member=13,
        prior_scales=[.8, 1., 1.2], prior_load_pairs_W=[[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]],
        prefix_score_floor_K=SCORE_FLOOR, covariance_envelope_sigma_multiplier=2,
        coverage_claim='Empirical finite-family coverage only, NOT a calibrated confidence interval',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--prefix', type=int, default=72)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 30 <= args.prefix <= 72 or not args.prefix + 12 <= args.steps <= 288:
        parser.error('Invalid bounded budget/prefix')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report')
    result = check(steps=args.steps, prefix=args.prefix)
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
