"""Reduce the entire known synthetic system using input/adjoint responses.

No full air/slab state block is retained. Materials and initial history remain
oracle fixture data; this is NOT an identified online building predictor.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_reduction_check import PROTOCOLS, balance_residual, forcing
from tools.thermal_port_modes_check import CHECK_PERIODS, DESIGN_PERIODS
from tools.thermal_wall_modes_check import ModalResponse, transfer_audit

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (0, 4, 8, 12, 16, 24)
FEEDBACK = [('no_air_feedback', (0., 0.)), ('air_feedback', (4., 3.)),
            ('off_design_feedback', (1., 5.))]


def system_basis(grid, additional_modes, *, adjoint=False):
    """DC input span, then normalized input and optional output-adjoint snapshots.

    Design periods and feedback are fixed, and no response trajectory or initial
    condition enters this construction. The uniform vector is included first
    to conserve total energy. Adjoint directions use the conjugate-transposed
    full operator and the actual zone-average sensor columns.
    """
    if not isinstance(additional_modes, int) or not 0 <= additional_modes <= 32:
        raise ValueError('Require 0..32 enrichment directions')
    root = np.sqrt(grid.capacity)
    k = grid.laplacian + np.diag(grid.exposure)
    b = np.column_stack((grid.uniform, grid.applied, grid.exposure))
    static = []

    def append_independent(vector, accepted):
        vector = vector.copy()
        for _ in range(2):
            for previous in accepted:
                vector -= previous * (previous @ vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            accepted.append(vector / norm)

    append_independent(root, static)
    dc = np.linalg.solve(k, b)
    for column in (root[:, None] * dc).T:
        append_independent(column / max(np.linalg.norm(column), 1e-30), static)
    static_matrix = np.array(static).T
    snapshots = []
    for g in ((0., 0.), (4., 3.)):
        base = k + (grid.applied * g) @ grid.sensed.T
        for period in DESIGN_PERIODS:
            dynamic = (1 - np.exp(-2j * np.pi * grid.seconds / period)) / grid.seconds
            matrix = base + np.diag(grid.capacity * dynamic)
            responses = [np.linalg.solve(matrix, b)]
            if adjoint:
                responses.append(np.linalg.solve(matrix.conj().T, grid.sensed))
            for response in responses:
                weighted = root[:, None] * response
                for component in (weighted.real, weighted.imag):
                    for column in component.T:
                        column = column - static_matrix @ (static_matrix.T @ column)
                        norm = np.linalg.norm(column)
                        if norm > 1e-10:
                            snapshots.append(column / norm)
    u, singular, _ = np.linalg.svd(np.array(snapshots).T, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    if additional_modes > rank:
        raise ValueError('Requested directions exceed numerical snapshot rank')
    accepted = list(static)
    for column in u.T:
        if len(accepted) == len(static) + additional_modes:
            break
        append_independent(column, accepted)
    if len(accepted) != len(static) + additional_modes:
        raise ValueError('Insufficient independent response directions')
    return np.array(accepted).T / root[:, None], dict(
        static_directions=len(static), additional_directions=additional_modes,
        includes_output_adjoint=adjoint, design_periods_seconds=list(DESIGN_PERIODS),
        design_feedback=[[0., 0.], [4., 3.]], numerical_snapshot_rank=rank,
        snapshot_singular_values=singular.tolist())


def input_at(protocol, step):
    if protocol == 'initial_only':
        return dict(ambient=295., fixed=(0., 0.), conductance=(0., 0.), supply=(295., 295.))
    return forcing(protocol, step)


def initial_state(grid, protocol):
    initial = np.full(len(grid.capacity), 295.)
    if protocol == 'mixed':
        initial += np.linspace(-2., 2., len(initial))
    elif protocol == 'initial_only':
        delta = np.random.default_rng(2026).uniform(-2., 2., len(initial))
        delta -= (grid.capacity @ delta) / grid.capacity.sum()
        initial += delta
    return initial


def stability_audit(model):
    result = {}
    for g in ((0., 0.), (4., 3.), (1., 5.), (0., 3.), (4., 0.)):
        matrix = model.rate + model.k + (model.p * g) @ model.s
        propagation = np.linalg.solve(matrix, model.rate)
        result[str(g)] = float(np.max(np.abs(np.linalg.eigvals(propagation))))
    return result


def check(*, steps=288):
    if not isinstance(steps, int) or not 1 <= steps <= 288:
        raise ValueError('Require an integer budget in 1..288')
    grid = fixture(room_rows=6, room_columns=5)
    bases, metadata = {}, {}
    for adjoint in (False, True):
        for count in COUNTS:
            name = ('input_output' if adjoint else 'input') + f'_{count}'
            bases[name], metadata[name] = system_basis(grid, count, adjoint=adjoint)
    models = {name: ModalResponse(grid, basis) for name, basis in bases.items()}
    reference = ModalResponse(grid, np.eye(len(grid.capacity)))
    transfer = transfer_audit(grid, bases, periods=CHECK_PERIODS, feedback=FEEDBACK)
    stability = {name: stability_audit(model) for name, model in models.items()}
    cases = []
    for protocol in (*PROTOCOLS, 'initial_only'):
        initial = initial_state(grid, protocol)
        full = initial.copy()
        states = {name: model.initial(grid, initial) for name, model in models.items()}
        initial_error = {name: (grid.sensed.T @ (model.basis @ states[name] - initial)).tolist()
                         for name, model in models.items()}
        errors = {name: [] for name in models}
        energy = {name: 0. for name in models}
        for step in range(steps):
            inputs = input_at(protocol, step)
            full = reference.advance(full, inputs)
            for name, model in models.items():
                previous = model.basis @ states[name]
                states[name] = model.advance(states[name], inputs)
                updated = model.basis @ states[name]
                residual = balance_residual(grid, previous, updated, inputs)
                energy[name] = max(energy[name], abs(float(residual.sum())))
                errors[name].append(grid.sensed.T @ (updated - full))
        cases.append(dict(protocol=protocol, reductions={name: dict(states=bases[name].shape[1],
            zone_mae_K=float(np.mean(np.abs(values))), maximum_zone_error_K=float(np.max(np.abs(values))),
            first_12_steps_zone_mae_K=float(np.mean(np.abs(values[:12]))),
            initial_zone_error_K=initial_error[name],
            checkpoints={str(h): np.asarray(values[h - 1]).tolist() for h in (1, 2, 6, 12, 72, 288) if h <= steps},
            maximum_total_energy_residual_W=energy[name]) for name, values in errors.items()}))
    files = ['tools/thermal_system_modes_check.py', 'tools/thermal_wall_modes_check.py',
             'tools/thermal_grid_reference.py', 'tools/thermal_reduction_check.py',
             'tools/thermal_port_modes_check.py']
    return dict(protocol='Whole-system reduction of known synthetic matrices; independent bounded '
                'responses with oracle initial projection, no fitting, convection or environment',
        steps_per_case=steps, full_states=len(grid.capacity), design=metadata,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        transfer=transfer, stability_spectral_radius=stability, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 288:
        parser.error('Require 1 <= steps <= 288')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report')
    result = check(steps=args.steps)
    for case in result['cases']:
        print(case['protocol'], {name: dict(states=m['states'], mae_K=round(m['zone_mae_K'], 7),
            max_K=round(m['maximum_zone_error_K'], 7)) for name, m in case['reductions'].items()})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
