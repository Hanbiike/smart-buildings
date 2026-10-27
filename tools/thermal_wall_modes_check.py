"""Static wall constraint modes plus slow dynamic modes on a known fixture.

Synthetic diagnostics only: oracle materials and full port states are known.
No controller, environment, archived targets, fitting or hidden warm-up.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_reduction_check import PROTOCOLS, balance_residual, forcing
from tools.thermal_spatial_check import partitions

ROOT = Path(__file__).resolve().parents[1]
MODE_COUNTS = (0, 2, 4, 8, 16, 32, 43)


def orthonormalize_capacity(grid, basis):
    """Keep the span, with E^T C E = I for numerical conditioning."""
    root = np.sqrt(grid.capacity)
    q, r = np.linalg.qr(root[:, None] * basis, mode='reduced')
    if np.min(np.abs(np.diag(r))) < 1e-10:
        raise ValueError('Dependent reduction columns')
    return q / root[:, None]


def wall_basis(grid, additional_modes):
    """Exact air/slab ports, static wall lift, ambient lift and slow wall modes.

    The ambient response is a FREE dynamic coordinate, not clamped to weather.
    Retaining it puts the uniform-temperature vector in the trial space and
    preserves total energy in the Galerkin equations. No affine weather-state
    substitution, missing weather derivative or static-wall clock reset.
    """
    walls = [i for p, i in grid.index.items() if p not in grid.mass_index]
    ports = [i for i in range(len(grid.capacity)) if i not in set(walls)]
    if not isinstance(additional_modes, int) or not 0 <= additional_modes < len(walls):
        raise ValueError('Invalid wall-mode budget')
    k = grid.laplacian + np.diag(grid.exposure)
    kw = k[np.ix_(walls, walls)]
    lift = np.linalg.solve(kw, -k[np.ix_(walls, ports)])
    ambient = np.linalg.solve(kw, grid.exposure[walls])
    eigenvalues, eigenvectors = eigh(kw, np.diag(grid.capacity[walls]))
    root = np.sqrt(grid.capacity[walls])
    # Weighted Gram-Schmidt of ambient first, then slow generalized eigenmodes.
    # A fixed full enrichment sequence makes all requested subspaces nested.
    wall_vectors = []
    for candidate in [ambient, *eigenvectors.T]:
        vector = root * candidate
        for _ in range(2):
            for accepted in wall_vectors:
                vector -= accepted * (accepted @ vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            wall_vectors.append(vector / norm)
        if len(wall_vectors) == additional_modes + 1:
            break
    if len(wall_vectors) != additional_modes + 1:
        raise ValueError('Insufficient independent wall modes')
    basis = np.zeros((len(grid.capacity), len(ports) + len(wall_vectors)))
    basis[ports, :len(ports)] = np.eye(len(ports))
    basis[walls, :len(ports)] = lift
    basis[np.ix_(walls, range(len(ports), basis.shape[1]))] = np.array(wall_vectors).T / root[:, None]
    return orthonormalize_capacity(grid, basis), dict(
        air_slab_port_states=len(ports), wall_states=len(walls),
        ambient_dynamic_mode=True, additional_slow_modes=additional_modes,
        unconstrained_wall_time_constants_seconds=(1 / eigenvalues).tolist())


class ModalResponse:
    """General Galerkin response: projected capacity is not assumed diagonal."""

    def __init__(self, grid, basis):
        self.basis = basis
        self.capacity = basis.T @ (grid.capacity[:, None] * basis)
        self.rate = self.capacity / grid.seconds
        self.k = basis.T @ (grid.laplacian + np.diag(grid.exposure)) @ basis
        self.p = basis.T @ grid.applied
        self.s = grid.sensed.T @ basis
        self.u = basis.T @ grid.uniform
        self.h = basis.T @ grid.exposure
        self.factors = {}

    def initial(self, grid, temperature):
        return np.linalg.solve(self.capacity, self.basis.T @ (grid.capacity * temperature))

    def advance(self, state, inputs):
        key = tuple(float(g) for g in inputs['conductance'])
        if key not in self.factors:
            self.factors[key] = lu_factor(self.rate + self.k + (self.p * key) @ self.s)
        rhs = (self.rate @ state + self.h * inputs['ambient'] + self.u @ inputs['fixed']
               + self.p @ (np.asarray(key) * inputs['supply']))
        return lu_solve(self.factors[key], rhs)


def transfer_audit(grid, bases, *, periods=(None, 600., 3600., 21600., 86400.),
                   feedback=None):
    """Discrete BE transfer at DC and selected periods; no time rollout.

    Inputs are two fixed-power ports, two air-power ports and ambient. Relative
    errors are normalized per input column so different input units aren't
    silently pooled in one matrix norm. Both outputs are zone mean temperature.
    """
    results = {name: {} for name in bases}
    b = np.column_stack((grid.uniform, grid.applied, grid.exposure))
    base_k = grid.laplacian + np.diag(grid.exposure)
    if feedback is None:
        feedback = [('no_air_feedback', (0., 0.)), ('air_feedback', (4., 3.))]
    for running, g in feedback:
        k = base_k + (grid.applied * g) @ grid.sensed.T
        for period in periods:
            omega = 0. if period is None else 2 * np.pi / period
            dynamic = (1 - np.exp(-1j * omega * grid.seconds)) / grid.seconds
            matrix = k + np.diag(grid.capacity * dynamic)
            full = grid.sensed.T @ np.linalg.solve(matrix, b)
            denominator = np.linalg.norm(full, axis=0)
            key = running + ('__dc' if period is None else f'__period_{int(period)}s')
            for name, basis in bases.items():
                reduced = grid.sensed.T @ basis @ np.linalg.solve(basis.T @ matrix @ basis, basis.T @ b)
                relative = np.linalg.norm(reduced - full, axis=0) / np.maximum(denominator, 1e-12)
                results[name][key] = dict(maximum_relative_column_error=float(relative.max()),
                    relative_input_column_errors=relative.tolist())
    return results


def check(*, steps=288):
    if not isinstance(steps, int) or not 1 <= steps <= 288:
        raise ValueError('Require an integer budget in 1..288')
    grid = fixture(room_rows=6, room_columns=5)
    bases, metadata = {}, {}
    for count in MODE_COUNTS:
        basis, details = wall_basis(grid, count)
        bases[f'wall_modes_{count}'] = basis
        metadata[f'wall_modes_{count}'] = details
    for level in ('coarse', 'segments'):
        bases[f'isothermal_{level}'] = partitions(grid, air_level='full', wall_level=level)[0]
    models = {name: ModalResponse(grid, basis) for name, basis in bases.items()}
    reference = ModalResponse(grid, np.eye(len(grid.capacity)))
    transfer = transfer_audit(grid, bases)
    cases = []
    for protocol in PROTOCOLS:
        initial = np.full(len(grid.capacity), 295.)
        if protocol == 'mixed':
            initial += np.linspace(-2., 2., len(initial))
        full = initial.copy()
        states = {name: model.initial(grid, initial) for name, model in models.items()}
        errors = {name: [] for name in models}
        energy = {name: 0. for name in models}
        initial_error = {name: float(np.max(np.abs(grid.sensed.T @ (model.basis @ states[name] - initial))))
                         for name, model in models.items()}
        for step in range(steps):
            inputs = forcing(protocol, step)
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
            initial_projected_zone_error_K=initial_error[name], maximum_total_energy_residual_W=energy[name])
            for name, values in errors.items()}))
    files = ['tools/thermal_wall_modes_check.py', 'tools/thermal_grid_reference.py',
             'tools/thermal_spatial_check.py', 'tools/thermal_reduction_check.py']
    return dict(protocol='Known synthetic coefficients; full air/slab ports; capacity projection; '
                'independent 1..288-step response, no fitting/convection/environment',
        steps_per_case=steps, full_states=len(grid.capacity), mode_metadata=metadata,
        transfer_input_columns=['fixed_room_1_W', 'fixed_room_2_W', 'air_room_1_W', 'air_room_2_W', 'ambient_K'],
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
    print('DC maximum relative errors', {name: max(m['maximum_relative_column_error'] for key, m in groups.items()
        if key.endswith('__dc')) for name, groups in result['transfer'].items()})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
