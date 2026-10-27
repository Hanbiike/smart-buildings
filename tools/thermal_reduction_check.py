"""Bounded synthetic input-response tests of nested spatial reductions.

No environment, scenario data, fitting, hidden state recovery or controller.
The known dense algebra fixture is the reference, not an online forecast.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture, solve_reference

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ('room', 'diffuser', 'boundary_side', 'full')
PROTOCOLS = ('passive', 'fixed_pulse', 'air_pulse', 'mixed')


def nested_basis(grid, level):
    """Nested binary partitions, including paired slab subdivisions.

    All wall cells stay explicit. This tiny fixture has no deep room core;
    'boundary_side' means partition-facing versus facade-facing air, with the
    diffuser isolated first. It does not claim to test large-room core modes.
    """
    if level not in LEVELS:
        raise ValueError('Unknown reduction level')
    if level == 'full':
        return np.eye(len(grid.capacity))
    groups = []
    for room, mask in enumerate(grid.rooms):
        parts = {}
        for coordinate in zip(*np.where(mask)):
            i = grid.index[coordinate]
            if level == 'room':
                key = 'room'
            elif grid.applied[i, room] > 0:
                key = 'diffuser'
            elif level == 'diffuser':
                key = 'rest'
            else:
                # Shared partition is col 4, rows 2..4 in this fixture.
                key = 'partition_side' if coordinate[1] in (3, 5) else 'facade_side'
            parts.setdefault(key, []).append(coordinate)
        for coordinates in parts.values():
            groups.append([grid.index[p] for p in coordinates])
            groups.append([grid.mass_index[p] for p in coordinates])
    groups.extend([[i] for p, i in grid.index.items() if p not in grid.mass_index])
    basis = np.zeros((len(grid.capacity), len(groups)))
    for column, members in enumerate(groups):
        basis[members, column] = 1.
    return basis


def project_initial(grid, initial, basis):
    """Capacity-weighted projection, preserving each partition's initial energy."""
    return basis @ ((basis.T @ (grid.capacity * initial)) / (basis.T @ grid.capacity))


def forcing(protocol, step):
    if protocol == 'passive':
        return dict(ambient=275., fixed=(0., 0.), conductance=(0., 0.), supply=(295., 295.))
    if protocol == 'fixed_pulse':
        return dict(ambient=295., fixed=(40. if step < 12 else 0., 0.),
                    conductance=(0., 0.), supply=(295., 295.))
    if protocol == 'air_pulse':
        return dict(ambient=295., fixed=(0., 0.),
                    conductance=(4. if step < 12 else 0., 0.), supply=(280., 295.))
    if protocol == 'mixed':
        return dict(ambient=280. + 3 * np.sin(2 * np.pi * step / 288),
                    fixed=(8. if step % 72 < 36 else 2., 4.),
                    conductance=(4. if step % 48 < 24 else 0., 3.),
                    supply=(282. if step % 48 < 24 else 300., 289.))
    raise ValueError('Unknown input protocol')


def balance_residual(grid, previous, updated, inputs):
    air_power = np.asarray(inputs['conductance']) * (inputs['supply'] - grid.sensed.T @ updated)
    return (grid.capacity * (updated - previous) / grid.seconds
            + grid.laplacian @ updated - grid.exposure * (inputs['ambient'] - updated)
            - grid.uniform @ inputs['fixed'] - grid.applied @ air_power)


def check(*, steps=288):
    if not isinstance(steps, int) or not 1 <= steps <= 288:
        raise ValueError('Synthetic response is limited to 1..288 steps')
    grid = fixture()
    bases = {level: nested_basis(grid, level) for level in LEVELS}
    cases = []
    for protocol in PROTOCOLS:
        initial = np.full(len(grid.capacity), 295.)
        if protocol == 'mixed':
            # Known nonuniform history, not a fitted or observed latent state.
            initial += np.linspace(-2., 2., len(initial))
        full = initial.copy()
        states = {level: project_initial(grid, initial, basis) for level, basis in bases.items()}
        errors = {level: [] for level in LEVELS}
        balances = {level: dict(maximum_total_balance_residual_W=0.,
                                maximum_group_balance_residual_W=0.,
                                maximum_cell_balance_residual_W=0.) for level in LEVELS}
        for step in range(steps):
            inputs = forcing(protocol, step)
            full = solve_reference(grid, full, **inputs)
            for level, basis in bases.items():
                previous = states[level]
                updated = solve_reference(grid, previous, basis=basis, **inputs)
                states[level] = updated
                residual = balance_residual(grid, previous, updated, inputs)
                values = dict(maximum_total_balance_residual_W=abs(float(residual.sum())),
                    maximum_group_balance_residual_W=float(np.max(np.abs(basis.T @ residual))),
                    maximum_cell_balance_residual_W=float(np.max(np.abs(residual))))
                for key, value in values.items():
                    balances[level][key] = max(balances[level][key], value)
                errors[level].append(grid.sensed.T @ (updated - full))
        cases.append(dict(protocol=protocol, steps=steps, initial_state='nonuniform' if protocol == 'mixed' else 'uniform',
            reductions={level: dict(states=bases[level].shape[1],
                zone_mae_K=float(np.mean(np.abs(values))),
                maximum_zone_error_K=float(np.max(np.abs(values))),
                final_zone_error_K=np.asarray(values[-1]).tolist(),
                checkpoints={str(h): np.asarray(values[h - 1]).tolist()
                             for h in (1, 2, 6, 12, 72, 288) if h <= steps},
                **balances[level]) for level, values in errors.items()}))
    paths = ['tools/thermal_reduction_check.py', 'tools/thermal_grid_reference.py',
             'smart_buildings/smart_control/simulator/finite_difference_solver.py']
    return dict(protocol='Known synthetic discrete system, independent free response per model; '
                'no observation corrections, convection shuffling, scenario fitting or environment runs',
        seconds_per_step=grid.seconds, steps_per_case=steps,
        source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        cases=cases)


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
            maximum_K=round(m['maximum_zone_error_K'], 7)) for name, m in case['reductions'].items()})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
