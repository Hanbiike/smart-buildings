"""Conservative prefix-only scalar identification on synthetic unknown plants.

No truth labels, future temperatures or latent fields enter the fitter. Local
sensitivity support is conditional on the chosen nominal model, not proof of
physical identification with arbitrary unknown initial wall states.
"""

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_mismatch_check import CASES, evaluate_received, generate_mismatch
from tools.thermal_state_observer_check import DelayedObserver
from tools.thermal_system_modes_check import system_basis
from tools.thermal_wall_modes_check import ModalResponse

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('log_capacity', 'log_exterior', 'log_airflow', 'room_1_load_5W', 'room_2_load_5W')
LOWER = np.array([np.log(.5)] * 3 + [-2., -2.])
UPPER = np.array([np.log(2.)] * 3 + [2., 2.])
SENSITIVITY_RMS_GATE = .001
RELATIVE_RANK_GATE = .001
CASE_NAMES = ('nominal', 'capacity_plus20', 'exterior_minus20', 'airflow_minus20',
              'hidden_heat_5W', 'sensor_noise_0p05K')


def candidate(nominal, basis, vector, commands):
    """Only nominal matrices and candidate values; no perturbed truth object."""
    grid = copy(nominal)
    grid.capacity = nominal.capacity * np.exp(vector[0])
    grid.exposure = nominal.exposure * np.exp(vector[1])
    model = ModalResponse(grid, basis)
    path = []
    for command in commands:
        values = dict(command)
        values['conductance'] = tuple(np.asarray(command['conductance']) * np.exp(vector[2]))
        values['fixed'] = tuple(np.asarray(command['fixed']) + 5 * np.asarray(vector[3:]))
        path.append(values)
    return grid, model, path


def prefix_residual(nominal, basis, received, commands, vector, *, initial_offset=None):
    grid, model, path = candidate(nominal, basis, vector, commands)
    observer = DelayedObserver(grid, model)
    errors = []
    # Initial 12 observations are assimilated inside the supplied prefix, not
    # hidden extra warm-up. Every target and forecast input stays in the prefix.
    origins = set(range(12, len(received) - 12, 6))
    for t, temperatures in enumerate(received):
        observer.receive(temperatures, completed_input=path[t - 2] if t >= 2 else None)
        if t == 0 and initial_offset is not None:
            # Hypothetical unmeasured initial-state direction, never a true
            # simulator state. The caller constrains H*offset=0.
            observer.state += initial_offset
        if t in origins:
            prediction = observer.forecast(path[t - 1:t + 11])
            for h in (2, 6, 12):
                errors.extend(((prediction[h - 1] - received[t + h]) / h).tolist())
    return np.array(errors)


def excitation_support(received, commands):
    variation = float(np.max(np.ptp(received, axis=0)))
    gradient = max(abs(c['ambient'] - float(np.mean(y))) for c, y in zip(commands, received))
    drivers = [np.asarray(c['conductance']) * (np.asarray(c['supply']) - y) + c['fixed']
               for c, y in zip(commands, received)]
    power = float(np.max(np.abs(drivers)))
    thermal = variation >= .05 and (gradient >= 1. or power >= .1)
    airflow = thermal and max(max(c['conductance']) for c in commands) >= .1
    supported = np.array([thermal, thermal and gradient >= 1., airflow, thermal, thermal])
    return supported, dict(zone_range_K=variation, maximum_outdoor_gradient_K=gradient,
                           maximum_nominal_thermal_driver_W=power,
                           interpretation='Necessary excitation checks, not sufficient identification')


def fit_prefix(nominal, basis, received, commands, *, max_evaluations=40):
    received = np.asarray(received)
    if (received.ndim != 2 or received.shape[1] != 2 or not np.isfinite(received).all()
            or not 30 <= len(received) <= 72 or len(commands) != len(received)
            or not 1 <= max_evaluations <= 40):
        raise ValueError('Require only a finite 30..72 observation prefix and matching commands')
    zero = np.zeros(5)
    calls = 0

    def data(vector, initial_offset=None):
        nonlocal calls
        calls += 1
        return prefix_residual(nominal, basis, received, commands, vector, initial_offset=initial_offset)

    def sensitivity(vector, indices):
        columns = []
        for index in indices:
            step = np.zeros(5)
            step[index] = 1e-4
            columns.append((data(vector + step) - data(vector - step)) / 2e-4)
        matrix = np.array(columns).T
        norms = np.linalg.norm(matrix, axis=0)
        singular = np.linalg.svd(matrix / np.maximum(norms, 1e-30), compute_uv=False)
        return matrix, singular

    supported, excitation = excitation_support(received, commands)
    baseline = data(zero)
    matrix, singular = sensitivity(zero, range(5))
    rms = np.sqrt(np.mean(matrix ** 2, axis=0))
    eligible = np.flatnonzero(supported & (rms >= SENSITIVITY_RMS_GATE))
    diagnostics = dict(excitation=excitation, sensitivity_rms_K_per_coordinate=dict(zip(NAMES, rms.tolist())),
        necessary_excitation_support=dict(zip(NAMES, supported.tolist())),
        eligible_parameters=[NAMES[i] for i in eligible], baseline_prefix_mae=float(np.mean(np.abs(baseline))),
        data_residual_definition='Conditional h2/h6/h12 prediction errors divided by horizon, stride 6',
        initial_state_access='Two received sensors only; frozen Kalman assumptions',
        load_scope='Constant prefix bias only; does not identify future occupancy or load changes',
        all_parameter_normalized_singular_values=singular.tolist())

    def finish(status, vector=zero):
        diagnostics.update(status=status, residual_function_calls=calls,
            applied_vector=vector.tolist(),
            unidentified_parameters=[name for i, name in enumerate(NAMES)
                                     if status != 'applied_local_fit' or i not in eligible])
        return vector.copy(), diagnostics

    if not len(eligible):
        return finish('no_prefix_excitation_or_sensitivity')
    # Profile out arbitrary unmeasured initial-state changes. Ignoring these
    # nuisance directions can make a well-conditioned parameter-only Jacobian
    # look identifiable when it is just correcting an initial-history error.
    observer = DelayedObserver(nominal, ModalResponse(nominal, basis))
    values, vectors = np.linalg.eigh(observer.p)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0)))
    nuisance = root @ null_space(observer.h @ root)

    def profiled(vector, parameter_matrix):
        columns = []
        for direction in nuisance.T:
            columns.append((data(vector, direction * .001) - data(vector, -direction * .001)) / .002)
        initial_matrix = np.array(columns).T
        u, singular, _ = np.linalg.svd(initial_matrix, full_matrices=False)
        keep = singular > max(singular[0] * 1e-6, 1e-9)
        q = u[:, keep]
        return parameter_matrix - q @ (q.T @ parameter_matrix), singular, int(keep.sum())

    profile_matrix, initial_singular, initial_rank = profiled(zero, matrix)
    profile_rms = np.sqrt(np.mean(profile_matrix ** 2, axis=0))
    diagnostics.update(initial_nuisance_directions=nuisance.shape[1],
        initial_nuisance_effective_rank=initial_rank,
        initial_nuisance_singular_values=initial_singular.tolist(),
        state_profiled_sensitivity_rms=dict(zip(NAMES, profile_rms.tolist())),
        nuisance_profile_assumption='Conservative unconstrained local null-sensor initial offsets; '
            '2 K prior sets coordinates, not an amplitude constraint')
    eligible = np.array([i for i in eligible if profile_rms[i] >= SENSITIVITY_RMS_GATE], dtype=int)
    diagnostics['eligible_parameters_after_state_profile'] = [NAMES[i] for i in eligible]
    if not len(eligible):
        return finish('confounded_with_initial_state')
    normalized = profile_matrix[:, eligible]
    normalized = normalized / np.maximum(np.linalg.norm(normalized, axis=0), 1e-30)
    profile_singular = np.linalg.svd(normalized, compute_uv=False)
    diagnostics['state_profiled_normalized_singular_values'] = profile_singular.tolist()
    if profile_singular[-1] / profile_singular[0] < RELATIVE_RANK_GATE:
        return finish('rank_deficient_after_state_profile')
    _, singular = sensitivity(zero, eligible)
    diagnostics['eligible_normalized_singular_values'] = singular.tolist()
    if singular[-1] / singular[0] < RELATIVE_RANK_GATE:
        return finish('confounded_prefix_sensitivities')

    def objective(active):
        vector = zero.copy()
        vector[eligible] = active
        prior_scale = np.array([np.log(2.)] * 3 + [1., 1.])[eligible]
        return np.r_[data(vector), .001 * active / prior_scale]

    result = least_squares(objective, zero[eligible], bounds=(LOWER[eligible], UPPER[eligible]),
        max_nfev=max_evaluations, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    fitted = zero.copy()
    fitted[eligible] = result.x
    final_error = data(fitted)
    relative = (result.x - LOWER[eligible]) / (UPPER[eligible] - LOWER[eligible])
    _, final_singular = sensitivity(fitted, eligible)
    final_matrix, _ = sensitivity(fitted, range(5))
    final_profile, _, _ = profiled(fitted, final_matrix)
    final_profile_rms = np.sqrt(np.mean(final_profile[:, eligible] ** 2, axis=0))
    normalized = final_profile[:, eligible] / np.maximum(np.linalg.norm(final_profile[:, eligible], axis=0), 1e-30)
    final_profile_singular = np.linalg.svd(normalized, compute_uv=False)
    diagnostics.update(solver_success=bool(result.success), evaluations=int(result.nfev),
        proposed_vector=fitted.tolist(), fitted_prefix_mae=float(np.mean(np.abs(final_error))),
        final_normalized_singular_values=final_singular.tolist())
    diagnostics['final_state_profiled_sensitivity_rms'] = final_profile_rms.tolist()
    if not result.success:
        return finish('solver_budget_or_convergence_failure')
    if np.any((relative < .001) | (relative > .999)):
        return finish('parameter_at_bound')
    if final_singular[-1] / final_singular[0] < RELATIVE_RANK_GATE:
        return finish('confounded_at_fitted_point')
    if (np.any(final_profile_rms < SENSITIVITY_RMS_GATE)
            or final_profile_singular[-1] / final_profile_singular[0] < RELATIVE_RANK_GATE):
        return finish('initial_state_confounding_at_fitted_point')
    if np.mean(np.abs(final_error)) > .9 * np.mean(np.abs(baseline)):
        return finish('insufficient_prefix_improvement')
    return finish('applied_local_fit', fitted)


def check(*, steps=288, prefix=72):
    if (not isinstance(steps, int) or not isinstance(prefix, int)
            or not 30 <= prefix <= 72 or not prefix + 12 <= steps <= 288):
        raise ValueError('Require a 30..72 observation prefix inside a <=288-step response')
    nominal = fixture(room_rows=6, room_columns=5)
    basis = system_basis(nominal, 8, adjoint=True)[0]
    baseline_model = ModalResponse(nominal, basis)
    rows = []
    for spec in CASES:
        if spec.name not in CASE_NAMES:
            continue
        for protocol in ('mixed', 'late_air_pulse'):
            received, clean, commands = generate_mismatch(nominal, protocol, steps, spec)
            # The fitter gets no validation arrays or truth labels.
            vector, diagnostics = fit_prefix(nominal, basis, received[:prefix], commands[:prefix])
            fitted_grid, fitted_model, fitted_commands = candidate(nominal, basis, vector, commands)
            baseline = evaluate_received(nominal, baseline_model, received, clean, commands, prefix=prefix)
            fitted = evaluate_received(fitted_grid, fitted_model, received, clean, fitted_commands, prefix=prefix)
            rows.append(dict(mismatch=spec.name, protocol=protocol, fit=diagnostics,
                baseline=baseline, prefix_identified=fitted,
                truth_parameters_evaluation_only=dict(capacity_scale=spec.capacity_scale,
                    exterior_scale=spec.exterior_scale, airflow_scale=spec.airflow_scale,
                    hidden_load_starts_at_step=96 if spec.hidden_room_1_heat_W else None)))
    files = ['tools/thermal_prefix_identification_check.py', 'tools/thermal_state_observer_check.py',
             'tools/thermal_mismatch_check.py', 'tools/thermal_grid_reference.py',
             'tools/thermal_system_modes_check.py', 'tools/thermal_wall_modes_check.py']
    return dict(protocol='Prefix-only conditional local scalar fit; nominal basis; no true labels/states '
                'or validation targets in fitting; frozen coefficients after prefix; causal latent updates only',
        steps_per_response=steps, prefix_observations=prefix,
        gates=dict(sensitivity_rms_K_per_coordinate=SENSITIVITY_RMS_GATE,
                   normalized_singular_ratio=RELATIVE_RANK_GATE, minimum_prefix_improvement=.1,
                   regularization_K=.001, optimizer_max_nfev=40),
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
    for row in result['rows']:
        print(row['mismatch'], row['protocol'], row['fit']['status'],
            {h: [round(row[name]['metrics']['all'][h]['kalman']['mae_K'], 7)
                 for name in ('baseline', 'prefix_identified')] for h in ('h2', 'h6', 'h12')})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
