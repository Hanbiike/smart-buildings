"""Causal pre-update forecast adequacy monitor; diagnostic, not control trust."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.thermal_family_check import select_family
from tools.thermal_grid_reference import fixture
from tools.thermal_mismatch_check import Mismatch, generate_mismatch
from tools.thermal_prefix_identification_check import candidate, excitation_support
from tools.thermal_state_observer_check import DelayedObserver
from tools.thermal_system_modes_check import system_basis


class AdequacyMonitor:
    """Calibrate once; score arriving sensors against a saved prior envelope.

    HEALTHY means no rejection detected, not a guarantee or control permission.
    First version latches rejection: no automatic recovery/threshold widening.
    """

    def __init__(self):
        self.calibration = []
        self.limit = None
        self.consecutive = 0
        self.first_alarm = None

    def calibrate(self, received, lower, upper):
        if self.limit is not None:
            raise ValueError('Calibration is frozen')
        y, low, high = [np.asarray(values, dtype=float) for values in (received, lower, upper)]
        if (y.ndim != 1 or low.shape != y.shape or high.shape != y.shape
                or not np.isfinite([y, low, high]).all() or np.any(low > high)
                or (self.calibration and y.shape != self.calibration[0][0].shape)):
            raise ValueError('Invalid calibration sensor/envelope')
        self.calibration.append((y.copy(), low.copy(), high.copy()))

    def freeze(self):
        if self.limit is not None or len(self.calibration) < 12:
            raise ValueError('Need >=12 calibration observations, exactly once')
        y, low, high = [np.array(values) for values in zip(*self.calibration)]
        errors = y - (low + high) / 2
        mad = 1.4826 * np.median(np.abs(errors - np.median(errors, axis=0)), axis=0)
        gaps = np.maximum(np.maximum(low - y, y - high), 0)
        self.limit = np.maximum(.001, np.quantile(gaps, .95, axis=0) + 3 * mad)

    def observe(self, index, received, lower, upper):
        if self.limit is None:
            raise ValueError('Freeze calibration before monitoring')
        y, low, high = [np.asarray(values) for values in (received, lower, upper)]
        if (y.shape != self.limit.shape or low.shape != y.shape or high.shape != y.shape
                or not np.isfinite([y, low, high]).all() or np.any(low > high)):
            raise ValueError('Invalid sensor/envelope shapes or values')
        gap = np.maximum(np.maximum(low - y, y - high), 0)
        ratio = float(np.max(gap / self.limit))
        self.consecutive = self.consecutive + 1 if ratio > 1 else 0
        if self.first_alarm is None and (ratio > 3 or self.consecutive >= 2):
            self.first_alarm = index
        return dict(status='UNTRUSTED' if self.first_alarm is not None else 'HEALTHY',
                    gap_ratio=ratio, first_alarm=self.first_alarm)


def run_monitor(nominal, basis, received, commands, *, selection_end=48, calibration_end=72):
    steps = len(commands)
    if (not 30 <= selection_end <= calibration_end - 12 < calibration_end < steps <= 288
            or np.shape(received) != (steps + 1, 2) or not np.isfinite(received).all()):
        raise ValueError('Invalid disjoint bounded selection/calibration windows')
    vectors, _, selection = select_family(nominal, basis, received[:selection_end], commands[:selection_end])
    members = [candidate(nominal, basis, vector, commands) for vector in vectors]
    observers = [DelayedObserver(grid, model) for grid, model, _ in members]
    monitor = AdequacyMonitor()
    trace = []
    for t, temperatures in enumerate(received):
        if t >= 2:
            # x[t-1] was forecast from obs[t-1] with completed u[t-2].
            # Must execute BEFORE receive(), never use post-update residuals.
            predictions = np.array([observer.forecast([path[t - 2]])[0]
                                    for observer, (_, _, path) in zip(observers, members)])
            low, high = predictions.min(axis=0), predictions.max(axis=0)
            if selection_end <= t < calibration_end:
                monitor.calibrate(temperatures, low, high)
            elif t >= calibration_end:
                if monitor.limit is None:
                    monitor.freeze()
                trace.append(dict(observation=t, **monitor.observe(t, temperatures, low, high)))
        for observer, (_, _, path) in zip(observers, members):
            observer.receive(temperatures, completed_input=path[t - 2] if t >= 2 else None)
    supported, _ = excitation_support(received[:selection_end], commands[:selection_end])
    return dict(retained_models=selection['retained'], calibration_limit_K=monitor.limit.tolist(),
        selection_observations=selection_end, calibration_observations=calibration_end-selection_end,
        prefix_excitation_support=supported.tolist(), first_alarm=monitor.first_alarm,
        rejected_fraction=float(np.mean([row['status'] == 'UNTRUSTED' for row in trace])),
        control_permission=False, trace=trace)


def check(*, steps=288):
    if not isinstance(steps, int) or not 100 <= steps <= 288:
        raise ValueError('Require 100..288 physical steps for onset checks')
    grid = fixture(room_rows=6, room_columns=5)
    basis = system_basis(grid, 8, adjoint=True)[0]
    cases = [('nominal_mixed', 'mixed', Mismatch('nominal'), None),
             ('nominal_late_air', 'late_air_pulse', Mismatch('nominal'), None),
             ('hidden_heat', 'mixed', Mismatch('hidden', hidden_room_1_heat_W=5.), 98),
             ('outside_airflow', 'late_air_pulse', Mismatch('outside', airflow_scale=1.5), 98),
             ('noise_onset', 'late_air_pulse', Mismatch('nominal'), 98),
             ('outside_airflow_from_reset', 'mixed', Mismatch('outside', airflow_scale=1.5), 2),
             ('noise_from_reset', 'late_air_pulse', Mismatch('noise', sensor_noise_std_K=.05), 0)]
    rows = []
    for name, protocol, spec, onset in cases:
        received, _, commands = generate_mismatch(grid, protocol, steps, spec)
        if name == 'noise_onset':
            received[98:] += np.random.default_rng(202611).normal(0, .05, received[98:].shape)
        result = run_monitor(grid, basis, received, commands)
        # Onset labels are evaluator-only: never an input to run_monitor.
        alarm = result['first_alarm']
        result.update(case=name, first_observable_change=onset,
            false_alarm_before_change=alarm is not None and (onset is None or alarm < onset),
            detection_delay_observations=alarm-onset if onset is not None and alarm is not None and alarm >= onset else None)
        rows.append(result)
        print(name, 'alarm', alarm, 'onset', onset, 'rejected fraction', round(result['rejected_fraction'], 4), flush=True)
    return dict(protocol='Known synthetic family; selection obs0..47, independent calibration obs48..71; '
                'pre-update forecasts only; frozen thresholds; no automatic recovery, environment or control',
        physical_steps_per_case=steps, empirical_not_confidence_guarantee=True, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 100 <= args.steps <= 288:
        parser.error('Require 100..288 steps')
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report')
    report = check(steps=args.steps)
    if args.output:
        with args.output.open('x') as out:
            json.dump(report, out, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
