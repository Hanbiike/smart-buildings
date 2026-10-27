"""Frequency-response-enriched wall bases, using known synthetic matrices only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_reduction_check import PROTOCOLS, balance_residual, forcing
from tools.thermal_spatial_check import partitions
from tools.thermal_wall_modes_check import (
    ModalResponse, orthonormalize_capacity, transfer_audit, wall_basis,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN_PERIODS = (600., 1800., 3600.)
CHECK_PERIODS = (None, 600., 900., 1200., 1800., 2700., 3600., 7200., 21600., 86400.)
COUNTS = (0, 2, 4, 8, 12, 16)


def port_basis(grid, additional_modes):
    """Static/DC lift plus SVD of input-driven wall response residuals.

    Full air/slab ports are retained. Design responses are obtained from the
    known matrices, NEVER from the evaluation trajectories. Per-real/imaginary
    input-column normalization prevents watts/ambient units dominating the SVD.
    This is input-response enrichment, not balanced or output-optimal reduction.
    """
    walls = [i for p, i in grid.index.items() if p not in grid.mass_index]
    wall_set = set(walls)
    ports = [i for i in range(len(grid.capacity)) if i not in wall_set]
    if not isinstance(additional_modes, int) or not 0 <= additional_modes < len(walls):
        raise ValueError('Invalid response-mode budget')
    k = grid.laplacian + np.diag(grid.exposure)
    kw = k[np.ix_(walls, walls)]
    lift = np.linalg.solve(kw, -k[np.ix_(walls, ports)])
    root = np.sqrt(grid.capacity[walls])
    ambient = root * np.linalg.solve(kw, grid.exposure[walls])
    ambient /= np.linalg.norm(ambient)
    b = np.column_stack((grid.uniform, grid.applied, grid.exposure))
    snapshots = []
    for g in ((0., 0.), (4., 3.)):
        base = k + (grid.applied * g) @ grid.sensed.T
        for period in DESIGN_PERIODS:
            dynamic = (1 - np.exp(-2j * np.pi * grid.seconds / period)) / grid.seconds
            response = np.linalg.solve(base + np.diag(grid.capacity * dynamic), b)
            residual = root[:, None] * (response[walls] - lift @ response[ports])
            for component in (residual.real, residual.imag):
                for column in component.T:
                    column = column - ambient * (ambient @ column)
                    norm = np.linalg.norm(column)
                    if norm > 1e-10:
                        snapshots.append(column / norm)
    u, singular, _ = np.linalg.svd(np.array(snapshots).T, full_matrices=True)
    accepted = [ambient]
    for column in u.T:
        vector = column.copy()
        for _ in range(2):
            for previous in accepted:
                vector -= previous * (previous @ vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            accepted.append(vector / norm)
    modes = np.array(accepted[:additional_modes + 1]).T / root[:, None]
    basis = np.zeros((len(grid.capacity), len(ports) + modes.shape[1]))
    basis[ports, :len(ports)] = np.eye(len(ports))
    basis[walls, :len(ports)] = lift
    basis[np.ix_(walls, range(len(ports), basis.shape[1]))] = modes
    return orthonormalize_capacity(grid, basis), dict(
        design_periods_seconds=list(DESIGN_PERIODS), design_feedback_conductances=[[0., 0.], [4., 3.]],
        normalized_real_snapshot_columns=len(snapshots), snapshot_singular_values=singular.tolist(),
        additional_response_modes=additional_modes,
        numerical_snapshot_rank=int(np.count_nonzero(singular > singular[0] * 1e-10)))


def check(*, steps=288):
    if not isinstance(steps, int) or not 1 <= steps <= 288:
        raise ValueError('Require an integer budget in 1..288')
    grid = fixture(room_rows=6, room_columns=5)
    bases, metadata = {}, {}
    for count in COUNTS:
        bases[f'port_modes_{count}'], metadata[f'port_modes_{count}'] = port_basis(grid, count)
    bases['slow_modes_8'] = wall_basis(grid, 8)[0]
    bases['isothermal_coarse'] = partitions(grid, air_level='full', wall_level='coarse')[0]
    models = {name: ModalResponse(grid, basis) for name, basis in bases.items()}
    reference = ModalResponse(grid, np.eye(len(grid.capacity)))
    feedback = [('no_air_feedback', (0., 0.)), ('air_feedback', (4., 3.)),
                ('off_design_feedback', (1., 5.))]
    transfer = transfer_audit(grid, bases, periods=CHECK_PERIODS, feedback=feedback)
    cases = []
    for protocol in PROTOCOLS:
        initial = np.full(len(grid.capacity), 295.)
        if protocol == 'mixed':
            initial += np.linspace(-2., 2., len(initial))
        full = initial.copy()
        states = {name: model.initial(grid, initial) for name, model in models.items()}
        initial_errors = {name: float(np.max(np.abs(grid.sensed.T @ (model.basis @ states[name] - initial))))
                          for name, model in models.items()}
        errors = {name: [] for name in models}
        energy = {name: 0. for name in models}
        for step in range(steps):
            inputs = forcing(protocol, step)
            full = reference.advance(full, inputs)
            for name, model in models.items():
                previous = model.basis @ states[name]
                states[name] = model.advance(states[name], inputs)
                updated = model.basis @ states[name]
                energy[name] = max(energy[name], abs(float(balance_residual(grid, previous, updated, inputs).sum())))
                errors[name].append(grid.sensed.T @ (updated - full))
        cases.append(dict(protocol=protocol, reductions={name: dict(states=bases[name].shape[1],
            zone_mae_K=float(np.mean(np.abs(values))), maximum_zone_error_K=float(np.max(np.abs(values))),
            first_12_steps_zone_mae_K=float(np.mean(np.abs(values[:12]))),
            initial_projected_zone_error_K=initial_errors[name], maximum_total_energy_residual_W=energy[name])
            for name, values in errors.items()}))
    files = ['tools/thermal_port_modes_check.py', 'tools/thermal_wall_modes_check.py',
             'tools/thermal_grid_reference.py', 'tools/thermal_spatial_check.py', 'tools/thermal_reduction_check.py']
    return dict(protocol='Known synthetic input-response bases; independent bounded responses; '
                'no trajectory fitting, convection, environment or policy',
        steps_per_case=steps, full_states=len(grid.capacity), design=metadata,
        check_periods_seconds=list(CHECK_PERIODS), off_design_feedback=[1., 5.],
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        transfer=transfer, cases=cases)


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
