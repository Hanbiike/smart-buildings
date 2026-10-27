"""Causal delayed-temperature observer on known synthetic reduced dynamics.

Filter inputs are two received temperatures and completed command/forcing
intervals only. Full synthetic states are restricted to data generation and
evaluation, never initialization or updates. No simulator environment runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_grid_reference import fixture
from tools.thermal_reduction_check import PROTOCOLS
from tools.thermal_system_modes_check import initial_state, input_at, system_basis
from tools.thermal_wall_modes_check import ModalResponse

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 2, 6, 12)


class DelayedObserver:
    """State refers to x[t-1] after obs[t], except the reset duplicate.

    Kalman coefficients/noise are frozen assumptions, not fitted on evaluation
    targets. The first observation initializes from a uniform temperature based
    ONLY on the two sensors; no true hidden initial history is passed here.
    """

    def __init__(self, grid, model, *, correction=True, initial_std=2.,
                 process_std=.01, measurement_std=1e-4):
        self.model = model
        self.h = model.s
        self.grid = grid
        if (not np.isfinite([initial_std, process_std, measurement_std]).all()
                or initial_std <= 0 or process_std < 0 or measurement_std <= 0):
            raise ValueError('Invalid fixed noise assumptions')
        # Independent physical-cell temperature prior, projected to coordinates.
        transform = np.linalg.solve(model.capacity, model.basis.T * grid.capacity)
        self.unit_covariance = transform @ transform.T
        self.p = initial_std ** 2 * self.unit_covariance
        self.q = process_std ** 2 * self.unit_covariance
        self.r = np.eye(self.h.shape[0]) * measurement_std ** 2
        self.correction = correction
        self.index = -1
        self.state = None
        self.transitions = {}

    def transition(self, inputs):
        key = tuple(float(g) for g in inputs['conductance'])
        if key not in self.transitions:
            matrix = self.model.rate + self.model.k + (self.model.p * key) @ self.model.s
            self.transitions[key] = np.linalg.solve(matrix, self.model.rate)
        return self.transitions[key]

    def receive(self, temperatures, *, completed_input=None):
        temperatures = np.asarray(temperatures, dtype=float)
        if temperatures.shape != (self.h.shape[0],) or not np.isfinite(temperatures).all():
            raise ValueError('Require finite received zone temperatures')
        next_index = self.index + 1
        if next_index >= 2 and completed_input is None:
            raise ValueError('A completed delayed command interval is required')
        if next_index < 2 and completed_input is not None:
            raise ValueError('Reset/duplicate observation has no completed interval')
        self.index = next_index
        if self.index == 0:
            self.state = self.model.initial(self.grid,
                np.full(len(self.grid.capacity), float(temperatures.mean())))
        elif self.index == 1:
            # Same reset field, not a second independent measurement.
            return self.state.copy()
        else:
            f = self.transition(completed_input)
            self.state = self.model.advance(self.state, completed_input)
            self.p = f @ self.p @ f.T + self.q
        # Both arms get the same two-sensor initialization. The open-loop arm
        # disables only subsequent corrections, not information at reset.
        if self.correction or self.index == 0:
            innovation = temperatures - self.h @ self.state
            variance = self.h @ self.p @ self.h.T + self.r
            gain = np.linalg.solve(variance, self.h @ self.p).T
            self.state += gain @ innovation
            residual = np.eye(len(self.state)) - gain @ self.h
            self.p = residual @ self.p @ residual.T + gain @ self.r @ gain.T
            self.p = (self.p + self.p.T) / 2
        return self.state.copy()

    def forecast(self, input_path):
        """Path starts with pending u[t-1]; new u[t] only affects step two."""
        if self.index < 1 or not input_path:
            raise ValueError('Need a pending interval after reset')
        state = self.state.copy()
        predictions = []
        for inputs in input_path:
            state = self.model.advance(state, inputs)
            predictions.append(self.h @ state)
        return np.array(predictions)


def command_path(protocol, steps):
    """Known synthetic commands, independent of any hidden temperature state."""
    if not isinstance(steps, int) or not 1 <= steps <= 288:
        raise ValueError('Require 1..288 synthetic transitions')
    inputs = []
    for step in range(steps):
        if protocol in ('late_fixed_pulse', 'late_air_pulse'):
            active = 96 <= step < 108
            inputs.append(dict(ambient=295.,
                fixed=(40. if active and protocol == 'late_fixed_pulse' else 0., 0.),
                conductance=(4. if active and protocol == 'late_air_pulse' else 0., 0.),
                supply=(280., 295.)))
        else:
            inputs.append(input_at(protocol, step))
    return inputs


def generate_data(grid, protocol, steps):
    """Bounded synthetic physical steps; the filtering interface gets no field."""
    inputs = command_path(protocol, steps)
    reference = ModalResponse(grid, np.eye(len(grid.capacity)))
    state = initial_state(grid, protocol)
    physical_temperatures = [grid.sensed.T @ state]
    for command in inputs:
        state = reference.advance(state, command)
        physical_temperatures.append(grid.sensed.T @ state)
    # obs[0]=Cx0, obs[1]=Cx0, obs[t>=2]=Cx[t-1]. Last x[steps]
    # is not exposed by the last observation, matching sensor latching.
    observations = np.array([physical_temperatures[0], *physical_temperatures[:-1]])
    return observations, inputs


def observability_audit(grid, model, *, observations=72):
    """Prior-scaled fixed-coefficient observability, not an online rank test."""
    observer = DelayedObserver(grid, model)
    values, vectors = np.linalg.eigh(observer.p)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0)))
    result = {}
    for g in ((0., 0.), (4., 3.)):
        f = observer.transition(dict(conductance=g))
        power = np.eye(model.capacity.shape[0])
        rows = []
        for _ in range(observations):
            rows.append(model.s @ power @ root)
            power = power @ f
        singular = np.linalg.svd(np.vstack(rows), compute_uv=False)
        result[str(g)] = dict(singular_values_K=singular.tolist(),
            smallest_to_largest=float(singular[-1] / singular[0]),
            relative_rank_1e_8=int(np.count_nonzero(singular > singular[0] * 1e-8)),
            states=len(singular), observation_count=observations,
            interpretation='Local fixed-input prior-scaled observability; not a switching-system proof')
    return result


def _metrics(values):
    return dict(samples=len(values), mae_K=float(np.mean(values)),
                maximum_K=float(np.max(values)), p95_K=float(np.percentile(values, 95)))


def check(*, steps=288, prefix=72):
    if (not isinstance(steps, int) or not isinstance(prefix, int)
            or not 1 <= prefix < steps <= 288 or steps - prefix < max(HORIZONS)):
        raise ValueError('Require bounded response with a complete validation horizon')
    grid = fixture(room_rows=6, room_columns=5)
    models = {str(count + 5): ModalResponse(grid, system_basis(grid, count, adjoint=True)[0])
              for count in (8, 16, 24)}
    observability = {name: observability_audit(grid, model) for name, model in models.items()}
    cases = []
    for protocol in (*PROTOCOLS, 'initial_only', 'late_fixed_pulse', 'late_air_pulse'):
        obs, commands = generate_data(grid, protocol, steps)
        for name, model in models.items():
            filters = {'kalman': DelayedObserver(grid, model),
                       'open_loop': DelayedObserver(grid, model, correction=False)}
            errors = {h: {key: [] for key in (*filters, 'persistence')} for h in HORIZONS}
            pulse_errors = {h: {key: [] for key in (*filters, 'persistence')} for h in HORIZONS}
            sensor_residuals = []
            minimum, maximum = float('inf'), float('-inf')
            for t, received in enumerate(obs):
                for kind, observer in filters.items():
                    observer.receive(received, completed_input=commands[t - 2] if t >= 2 else None)
                    if kind == 'kalman':
                        field = model.basis @ observer.state
                        minimum, maximum = min(minimum, float(field.min())), max(maximum, float(field.max()))
                        sensor_residuals.append(float(np.max(np.abs(model.s @ observer.state - received))))
                # All horizons/methods use the same complete-horizon origins.
                if prefix <= t <= steps - max(HORIZONS):
                    path = commands[t - 1:t - 1 + max(HORIZONS)]
                    predictions = {key: observer.forecast(path) for key, observer in filters.items()}
                    for h in HORIZONS:
                        target = obs[t + h]
                        for kind in filters:
                            values = np.abs(predictions[kind][h - 1] - target).tolist()
                            errors[h][kind].extend(values)
                            if protocol.startswith('late_') and 84 <= t < 120:
                                pulse_errors[h][kind].extend(values)
                        values = np.abs(received - target).tolist()
                        errors[h]['persistence'].extend(values)
                        if protocol.startswith('late_') and 84 <= t < 120:
                            pulse_errors[h]['persistence'].extend(values)
            cases.append(dict(protocol=protocol, states=int(name),
                evaluation_prefix_observations=prefix,
                maximum_received_sensor_residual_K=max(sensor_residuals),
                estimated_physical_temperature_range_K=[minimum, maximum],
                pulse_origin_window=[84, 120] if protocol.startswith('late_') else None,
                pulse_window_metrics={f'h{h}': {kind: _metrics(values) if values else None
                    for kind, values in groups.items()} for h, groups in pulse_errors.items()},
                metrics={f'h{h}': {kind: _metrics(values) for kind, values in groups.items()}
                         for h, groups in errors.items()}))
    files = ['tools/thermal_state_observer_check.py', 'tools/thermal_system_modes_check.py',
             'tools/thermal_wall_modes_check.py', 'tools/thermal_grid_reference.py',
             'tools/thermal_reduction_check.py']
    return dict(protocol='Known synthetic coefficients; causal two-sensor updates and past '
                'forcing only; forecasts conditional on supplied future synthetic inputs; no policy/environment',
        total_physical_steps=steps, observations_including_reset=steps + 1,
        initial_physical_cell_prior_std_K=2., physical_cell_process_std_K_per_step=.01,
        assumed_measurement_std_K=1e-4, actual_synthetic_measurement_noise_K=0.,
        thermal_parameters_frozen=True, duplicate_reset_measurement_updated_twice=False,
        delay='obs[t] is C x[t-1]; pending u[t-1] then new u[t]; h1 does not evaluate a new action',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        observability=observability, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--prefix', type=int, default=72)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.prefix < args.steps <= 288 or args.steps - args.prefix < max(HORIZONS):
        parser.error('Invalid bounded response/prefix/horizon')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report')
    result = check(steps=args.steps, prefix=args.prefix)
    for case in result['cases']:
        print(case['protocol'], case['states'], {h: {key: round(value['mae_K'], 7)
            for key, value in metrics.items()} for h, metrics in case['metrics'].items()})
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
