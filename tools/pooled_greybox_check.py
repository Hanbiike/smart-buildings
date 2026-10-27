"""Leave-one-building-out validation of a small shared thermal prior.

Replays at most 288 observations per saved case. No simulator, training-scenario
config, environment reset or preceding dump observations are consumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.baselines.pooled_greybox import (  # noqa: E402
    PooledGreyBoxModel, fit_population_prior,
)
from smart_buildings.baselines.thermal_greybox import ThermalGreyBoxModel  # noqa: E402
from tools.greybox_check import RESULTS, evaluate  # noqa: E402


@dataclass
class CaseExamples:
    scenario: str
    x: np.ndarray
    y: np.ndarray
    occupied_observations: int | None

    @property
    def building(self):
        return self.scenario.split('/')[0]


def collect_case(scenario, policy, offset, steps):
    if not 2 <= steps <= 288 or offset < 0:
        raise ValueError('Require 2 <= steps <= 288 and offset >= 0')
    base = RESULTS / scenario
    meta = json.loads((base / 'scenario_meta.json').read_text())
    with np.load(base / f'nn_data_{policy}' / 'obs_dump.npz') as dump:
        window = slice(offset, offset + steps)
        obs = dump['obs'][window]
        continuous = dump['action_continuous'][window]
        discrete = dump['action_discrete'][window]
    if any(len(array) != steps for array in (obs, continuous, discrete)):
        raise ValueError(f'Incomplete saved window: {scenario}')
    model = ThermalGreyBoxModel(meta, fit_window=steps + 1,
                                physical_water=False, thermal_mass=True)
    for t in range(steps):
        model.update(obs[t])
        model.set_action(continuous[t], discrete[t])
    rows = [row for room in model.rooms for row in model.history[room]]
    occupied_idx = model.field_pos.get('num_occupants')
    return CaseExamples(scenario, np.array([x for x, _ in rows]),
                        np.array([y for _, y in rows]),
                        int(np.count_nonzero(obs[:, occupied_idx] > 0))
                        if occupied_idx is not None else None)


def fit_fold(examples, heldout_building):
    if not any(case.building == heldout_building for case in examples):
        raise ValueError('Held-out building is absent from the dataset')
    training = [case for case in examples if case.building != heldout_building]
    if not training:
        raise ValueError('No other buildings are available for training')
    coefficients = fit_population_prior((case.x, case.y) for case in training)
    manifest = dict(heldout_building=heldout_building,
                    training_buildings=sorted({case.building for case in training}),
                    training_scenarios=[case.scenario for case in training],
                    training_rows=sum(len(case.y) for case in training),
                    coefficients=coefficients.tolist())
    assert heldout_building not in manifest['training_buildings']
    return coefficients, manifest


def summarize(rows, horizons):
    result = {}
    for variant in sorted({row['model'] for row in rows}):
        selected = [row for row in rows if row['model'] == variant]
        table = {}
        for h in horizons:
            model_mae = float(np.mean([row[f'h{h}'] for row in selected]))
            persist = float(np.mean([row[f'h{h}_persist'] for row in selected]))
            entry = dict(model_mae_K=model_mae, persistence_mae_K=persist,
                         relative_reduction=1 - model_mae / persist if persist > 0 else None,
                         cases=len(selected), cases_beating_persistence=sum(
                             row[f'h{h}_skill'] is not None and row[f'h{h}_skill'] > 0
                             for row in selected))
            for label, key in [('active', f'h{h}_regimes'),
                               ('occupied', f'h{h}_occupancy_regimes')]:
                conditional = [row[key][label] for row in selected if label in row[key]]
                entry[label] = (dict(
                    model_mae_K=float(np.mean([row['model'] for row in conditional])),
                    persistence_mae_K=float(np.mean([row['persist'] for row in conditional])),
                    cases=len(conditional)) if conditional else None)
            table[f'h{h}'] = entry
        result[variant] = table
    return result


def forecast_gate(rows, horizons, minimum_reduction=0.20):
    """Conservative forecast prerequisite, NOT permission to enable MPC.

    Overall, active and occupied errors must each improve by the prespecified
    fraction. No held-out building may regress in any of those regimes.
    """
    if not rows or not horizons or not 0 <= minimum_reduction < 1:
        raise ValueError('Nonempty rows/horizons and a reduction in [0,1) are required')
    results = {}
    for variant in sorted({row['model'] for row in rows}):
        selected = [row for row in rows if row['model'] == variant]
        failures = []
        for h in horizons:
            for regime in ('all', 'active', 'occupied'):
                metrics = []
                for row in selected:
                    entry = (dict(model=row[f'h{h}'], persist=row[f'h{h}_persist'])
                             if regime == 'all' else row[
                                 f'h{h}_regimes' if regime == 'active'
                                 else f'h{h}_occupancy_regimes'].get(regime))
                    if entry is not None:
                        metrics.append((row['scenario'].split('/')[0], entry))
                if len(metrics) != len(selected):
                    failures.append(f'h{h}/{regime}: incomplete regime coverage')
                if not metrics:
                    continue
                mean = float(np.mean([entry['model'] for _, entry in metrics]))
                baseline = float(np.mean([entry['persist'] for _, entry in metrics]))
                if baseline <= 0 or not np.isfinite(mean) or mean > (1 - minimum_reduction) * baseline:
                    failures.append(f'h{h}/{regime}: improvement below {minimum_reduction:.0%}')
                for building in sorted({group for group, _ in metrics}):
                    group = [entry for key, entry in metrics if key == building]
                    if np.mean([entry['model'] for entry in group]) > np.mean(
                            [entry['persist'] for entry in group]):
                        failures.append(f'h{h}/{regime}: regression on {building}')
        results[variant] = dict(passed=not failures, failures=failures)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default='dsat_gen31')
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--offset', type=int, default=576)
    parser.add_argument('--fit-window', type=int, default=72)
    parser.add_argument('--horizons', nargs='+', type=int, default=[6, 12])
    parser.add_argument('--test-buildings', nargs='+',
                        help='Limit test folds, not the other-building training pool')
    parser.add_argument('--local-ridge-strength', type=float, default=16.0)
    parser.add_argument('--output', type=Path, help='New JSON report path; never overwrite an existing file')
    args = parser.parse_args()
    if not 1 <= args.fit_window < args.steps <= 288 or args.offset < 0:
        parser.error('Require 1 <= fit-window < steps <= 288 and offset >= 0')
    if min(args.horizons) < 1 or max(args.horizons) >= args.steps - args.fit_window:
        parser.error('Horizons do not fit in the validation window')
    if not np.isfinite(args.local_ridge_strength) or args.local_ridge_strength <= 0:
        parser.error('--local-ridge-strength must be finite and positive')
    if args.output and args.output.exists():
        parser.error(f'Refusing to overwrite {args.output}')
    scenarios = sorted(str(path.parent.parent.relative_to(RESULTS))
                       for path in RESULTS.glob(f'*/*/nn_data_{args.policy}/obs_dump.npz'))
    if not scenarios:
        parser.error('No saved cases found')
    buildings = sorted({scenario.split('/')[0] for scenario in scenarios})
    if args.test_buildings and set(args.test_buildings) - set(buildings):
        parser.error('Unknown held-out building')
    print('Collecting bounded-window features; no simulation.', flush=True)
    examples = []
    for scenario in scenarios:
        example = collect_case(scenario, args.policy, args.offset, args.steps)
        examples.append(example)
        print('collected', scenario, 'rows', len(example.y),
              'occupied observations', example.occupied_observations, flush=True)
    rows, folds = [], []
    for building in args.test_buildings or buildings:
        prior, manifest = fit_fold(examples, building)
        folds.append(manifest)
        print('held out', building, 'trained on', len(manifest['training_buildings']),
              'other buildings;', manifest['training_rows'], 'rows', flush=True)
        for case in [case for case in examples if case.building == building]:
            for variant, adapt in [('mass', None), ('pooled_frozen', False), ('pooled_adapted', True)]:
                factory = (None if adapt is None else partial(
                    PooledGreyBoxModel, population_prior=prior, adapt=adapt,
                    local_ridge_strength=args.local_ridge_strength))
                row = evaluate(case.scenario, args.policy, horizons=tuple(args.horizons),
                               fit_window=args.fit_window, steps=args.steps,
                               model_kind=variant, offset=args.offset, model_factory=factory)
                rows.append(row)
                print(variant, case.scenario,
                      {f'h{h}': round(row[f'h{h}'], 4) for h in args.horizons}, flush=True)
    report = dict(protocol='leave-one-building-out; frozen population prior or causal adaptation',
                  steps=args.steps, offset=args.offset, fit_window=args.fit_window,
                  local_ridge_strength=args.local_ridge_strength, horizons=args.horizons,
                  policy=args.policy, oracle_actions=False,
                  population_ridge_strength=0.01, folds=folds, rows=rows,
                  summary=summarize(rows, args.horizons),
                  forecast_gate=forecast_gate(rows, args.horizons))
    print(json.dumps(report['summary'], indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            json.dump(report, file, indent=2, allow_nan=False)
        print('Report:', args.output)


if __name__ == '__main__':
    main()
