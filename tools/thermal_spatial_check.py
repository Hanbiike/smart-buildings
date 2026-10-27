"""Factorial air/core and wall reduction audit on a known synthetic system.

No environment, archive, fitting or participant policy. Each response is capped
at 288 steps including its initial excitation; there is no hidden warm-up.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import lu_factor, lu_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture, solve_source
from tools.thermal_reduction_check import PROTOCOLS, balance_residual, forcing

ROOT = Path(__file__).resolve().parents[1]
AIR_LEVELS = ('room', 'spatial', 'full')
WALL_LEVELS = ('coarse', 'segments', 'full')


def partitions(grid, *, air_level, wall_level):
    if air_level not in AIR_LEVELS or wall_level not in WALL_LEVELS:
        raise ValueError('Unknown spatial partition')
    groups = []
    core_counts = []
    for room, mask in enumerate(grid.rooms):
        parts = {}
        core_count = 0
        for r, c in zip(*np.where(mask)):
            partition_side = abs(c - grid.partition_column) == 1
            facade_side = r in (2, grid.last_row - 1) or c in (2, grid.last_column - 1)
            if not partition_side and not facade_side:
                core_count += 1
            if air_level == 'full':
                key = (r, c)
            elif air_level == 'room':
                key = 'room'
            elif grid.applied[grid.index[r, c], room] > 0:
                key = 'diffuser'
            else:
                key = (partition_side, facade_side)
            parts.setdefault(key, []).append((r, c))
        core_counts.append(core_count)
        for coords in parts.values():
            groups.append([grid.index[p] for p in coords])
            groups.append([grid.mass_index[p] for p in coords])

    wall_parts = {}
    for (r, c), i in grid.index.items():
        if (r, c) in grid.mass_index:
            continue
        if wall_level == 'full':
            key = (r, c)
        elif r in (1, grid.last_row):
            # Top/bottom facade split at the shared partition, including a
            # separate junction cell. Don't merge the two rooms' whole facade.
            side = 'left' if c < grid.partition_column else 'right' if c > grid.partition_column else 'junction'
            key = ('top' if r == 1 else 'bottom', side)
        elif c == grid.partition_column:
            key = ('partition',)
        else:
            key = ('left' if c == 1 else 'right',)
        wall_parts.setdefault(key, []).append(i)
    for members in wall_parts.values():
        if wall_level == 'segments':
            groups.extend([part.tolist() for part in np.array_split(members, min(2, len(members)))])
        else:
            groups.append(members)
    basis = np.zeros((len(grid.capacity), len(groups)))
    for column, members in enumerate(groups):
        basis[members, column] = 1.
    return basis, core_counts


class CompiledResponse:
    """Exactly the same projected equations, cached by air conductance tuple."""

    def __init__(self, grid, basis):
        self.basis = basis
        # Basis columns are disjoint indicator groups, so projected C is diagonal.
        self.capacity = basis.T @ grid.capacity
        self.rate = self.capacity / grid.seconds
        self.conduction = basis.T @ (grid.laplacian + np.diag(grid.exposure)) @ basis
        self.applied = basis.T @ grid.applied
        self.sensed = grid.sensed.T @ basis
        self.uniform = basis.T @ grid.uniform
        self.exposure = basis.T @ grid.exposure
        self.factors = {}

    def initial(self, grid, state):
        return (self.basis.T @ (grid.capacity * state)) / self.capacity

    def advance(self, state, inputs):
        key = tuple(float(x) for x in inputs['conductance'])
        if key not in self.factors:
            matrix = (np.diag(self.rate) + self.conduction
                      + (self.applied * key) @ self.sensed)
            self.factors[key] = lu_factor(matrix)
        rhs = (self.rate * state + self.exposure * inputs['ambient']
               + self.uniform @ inputs['fixed']
               + self.applied @ (np.asarray(key) * inputs['supply']))
        return lu_solve(self.factors[key], rhs)


def check(*, steps=288):
    if not isinstance(steps, int) or not 1 <= steps <= 288:
        raise ValueError('Require an integer budget in 1..288')
    grid = fixture(room_rows=6, room_columns=5)
    reference = CompiledResponse(grid, np.eye(len(grid.capacity)))
    models, counts = {}, {}
    for air in AIR_LEVELS:
        for wall in WALL_LEVELS:
            basis, core_counts = partitions(grid, air_level=air, wall_level=wall)
            name = f'{air}__{wall}'
            models[name] = CompiledResponse(grid, basis)
            counts[name] = dict(air_partition=air, wall_partition=wall,
                               states=basis.shape[1], core_cells_per_room=core_counts)
    walls = [i for p, i in grid.index.items() if p not in grid.mass_index]
    cases = []
    maximum_reference_residual = 0.
    maximum_source_disagreement = 0.
    for protocol in PROTOCOLS:
        initial = np.full(len(grid.capacity), 295.)
        if protocol == 'mixed':
            initial += np.linspace(-2., 2., len(initial))
        full = initial.copy()
        states = {name: model.initial(grid, initial) for name, model in models.items()}
        errors = {name: [] for name in models}
        wall_errors = {name: [] for name in models}
        balances = {name: [0., 0.] for name in models}
        # Validate the generalized fixture against the actual source solver
        # for each distinct protocol's first step, including latent history.
        source = solve_source(grid, initial, **forcing(protocol, 0))
        for step in range(steps):
            inputs = forcing(protocol, step)
            old_full = full
            full = reference.advance(full, inputs)
            if step == 0:
                maximum_source_disagreement = max(maximum_source_disagreement,
                                                  float(np.max(np.abs(source - full))))
            reference_residual = balance_residual(grid, old_full, full, inputs)
            maximum_reference_residual = max(maximum_reference_residual,
                                            float(np.max(np.abs(reference_residual))))
            for name, model in models.items():
                previous = model.basis @ states[name]
                states[name] = model.advance(states[name], inputs)
                updated = model.basis @ states[name]
                residual = balance_residual(grid, previous, updated, inputs)
                balances[name][0] = max(balances[name][0], abs(float(residual.sum())))
                balances[name][1] = max(balances[name][1], float(np.max(np.abs(model.basis.T @ residual))))
                errors[name].append(grid.sensed.T @ (updated - full))
                wall_errors[name].append(updated[walls] - full[walls])
        metrics = {}
        for name, values in errors.items():
            absolute = np.abs(values)
            metrics[name] = dict(**counts[name], zone_mae_K=float(absolute.mean()),
                maximum_zone_error_K=float(absolute.max()),
                first_12_steps_zone_mae_K=float(absolute[:12].mean()),
                after_12_steps_zone_mae_K=float(absolute[12:].mean()) if steps > 12 else None,
                maximum_wall_error_K=float(np.max(np.abs(wall_errors[name]))),
                final_zone_error_K=np.asarray(values[-1]).tolist(),
                maximum_total_balance_residual_W=balances[name][0],
                maximum_group_balance_residual_W=balances[name][1])
        cases.append(dict(protocol=protocol, reductions=metrics))
    files = ['tools/thermal_spatial_check.py', 'tools/thermal_grid_reference.py',
             'tools/thermal_reduction_check.py',
             'smart_buildings/smart_control/simulator/finite_difference_solver.py']
    return dict(protocol='Synthetic known-system independent free responses, no fitting, '
                'no environment runs or convection shuffling; same synthetic coefficients for all bases',
        full_states=len(grid.capacity), room_cells=[int(mask.sum()) for mask in grid.rooms],
        steps_per_case=steps, seconds_per_step=grid.seconds,
        maximum_first_step_source_reference_error_K=maximum_source_disagreement,
        maximum_reference_node_balance_residual_W=maximum_reference_residual,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}, cases=cases)


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
    print('states', result['full_states'], 'source agreement K', result['maximum_first_step_source_reference_error_K'])
    for case in result['cases']:
        print(case['protocol'], {name: dict(states=m['states'], mae_K=round(m['zone_mae_K'], 6),
            maximum_K=round(m['maximum_zone_error_K'], 6)) for name, m in case['reductions'].items()})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
