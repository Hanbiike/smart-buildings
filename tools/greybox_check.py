"""Offline validation of the grey-box zone model on the banked NN dumps.

Replays a recorded observation sequence through
``smart_buildings.baselines.greybox.GreyBoxModel`` and measures how well the
identified model predicts the zone temperature L steps ahead, compared with
the persistence baseline (T[t+1] = T[t]).

    python tools/greybox_check.py --scenarios building_train_1/fairbanks_1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.baselines.greybox import GreyBoxModel  # noqa: E402
from smart_buildings.baselines.causal_greybox import CausalGreyBoxModel  # noqa: E402
from smart_buildings.baselines.thermal_greybox import ThermalGreyBoxModel  # noqa: E402

MODEL_KINDS = ('legacy', 'causal', 'sensors', 'delayed', 'calibrated', 'water', 'mass', 'thermal')

RESULTS = Path(__file__).resolve().parents[1] / 'smart_buildings' / 'results'


def make_model(meta, model_kind, mode, fit_window):
    if model_kind not in MODEL_KINDS:
        raise ValueError(f'Unknown model: {model_kind}')
    if model_kind == 'legacy':
        return GreyBoxModel(meta, mode=mode, fit_window=fit_window)
    if model_kind in ('water', 'mass', 'thermal'):
        return ThermalGreyBoxModel(meta, fit_window=fit_window,
                                  physical_water=model_kind != 'mass',
                                  thermal_mass=model_kind != 'water')
    return CausalGreyBoxModel(meta, fit_window=fit_window,
                             use_actions=model_kind != 'sensors',
                             sensor_delay=int(model_kind in ('delayed', 'calibrated')),
                             calibrate_plant=model_kind == 'calibrated')


def load_case(scenario: str, policy: str) -> tuple[GreyBoxModel, np.ndarray, dict]:
    base = RESULTS / scenario
    meta = json.loads((base / 'scenario_meta.json').read_text())
    dump = np.load(base / f'nn_data_{policy}' / 'obs_dump.npz')
    obs = dump['obs']
    model = GreyBoxModel(meta, mode='batch_rls', fit_window=144)
    return model, obs, meta


def evaluate(
    scenario: str,
    policy: str = 'dsat_gen31',
    horizons: tuple[int, ...] = (1, 3, 6, 12),
    mode: str = 'batch_rls',
    fit_window: int = 144,
    rooms: int = 0,
    steps: int = 288,
    model_kind: str = 'legacy',
    oracle_actions: bool = False,
    offset: int = 0,
    model_factory=None,
) -> dict:
    if not 1 <= fit_window < steps <= 288:
        raise ValueError('require 1 <= fit_window < steps <= 288')
    if not horizons or min(horizons) < 1 or max(horizons) >= steps - fit_window:
        raise ValueError('Horizons must fit within the validation window')
    if oracle_actions and model_kind in ('legacy', 'sensors'):
        raise ValueError('Oracle commands require an action-conditioned model')
    if offset < 0:
        raise ValueError('offset must be non-negative')
    base = RESULTS / scenario
    meta = json.loads((base / 'scenario_meta.json').read_text())
    with np.load(base / f'nn_data_{policy}' / 'obs_dump.npz') as dump:
        window = slice(offset, offset + steps)
        obs = dump['obs'][window]
        continuous = dump['action_continuous'][window]
        discrete = dump['action_discrete'][window]
    if any(len(array) != steps for array in (obs, continuous, discrete)):
        raise ValueError('Saved dump does not contain the complete requested window')
    model = (model_factory(meta, fit_window=fit_window) if model_factory is not None
             else make_model(meta, model_kind, mode, fit_window))
    room_ids = sorted(model.room_temp_obs)
    if rooms:
        room_ids = room_ids[:rooms]

    n_steps = obs.shape[0]
    err = {h: [] for h in horizons}
    base_err = {h: [] for h in horizons}
    drift_err = {h: [] for h in horizons}
    warm_err = {h: [] for h in horizons}
    regime_errors = {h: {} for h in horizons}
    occupancy_errors = {h: {} for h in horizons}
    occupancy_idx = model.field_pos.get('num_occupants')
    for t in range(n_steps):
        model.update(obs[t])
        if isinstance(model, CausalGreyBoxModel):
            model.set_action(continuous[t], discrete[t])
        if t < fit_window:
            continue
        path = None
        if oracle_actions:
            max_h = min(max(horizons), n_steps - t - 1)
            commands = [(continuous[i], discrete[i]) for i in range(t, t + max_h)]
            path = model.predict_action_path(commands, max_h)
        for room in room_ids:
            t_now = model.zone_temp(obs[t], room)
            ahu = model.room_ahu.get(room)
            active = ahu is not None and model._ahu_running(obs[t], ahu) > 0
            regime = 'active' if active else 'passive'
            for h in horizons:
                if t + h >= n_steps:
                    continue
                actual = model.zone_temp(obs[t + h], room)
                pred = (path[h - 1][room] if path is not None
                        else model.predict(room, h, temp0=t_now))
                err[h].append(abs(pred - actual))
                base_err[h].append(abs(t_now - actual))
                bucket = regime_errors[h].setdefault(regime, [0.0, 0.0, 0])
                bucket[0] += abs(pred - actual)
                bucket[1] += abs(t_now - actual)
                bucket[2] += 1
                if occupancy_idx is not None:
                    occupancy_regime = 'occupied' if obs[t, occupancy_idx] > 0 else 'empty'
                    bucket = occupancy_errors[h].setdefault(occupancy_regime, [0.0, 0.0, 0])
                    bucket[0] += abs(pred - actual)
                    bucket[1] += abs(t_now - actual)
                    bucket[2] += 1
                delta = t_now - model.zone_temp(obs[t - 1], room)
                drift_err[h].append(abs(t_now + h * delta - actual))
                if t < fit_window + 24:
                    warm_err[h].append(abs(pred - actual))

    occupied_observations = (int(np.count_nonzero(obs[:, occupancy_idx] > 0))
                             if occupancy_idx is not None else None)
    out = {'scenario': scenario, 'policy': policy, 'model': model_kind,
           'oracle_actions': oracle_actions, 'observations': n_steps,
           'offset': offset,
           'occupied_observations': occupied_observations,
           'mode': mode if model_kind == 'legacy' else 'bounded_arx',
           'fit_window': fit_window, 'rooms': len(room_ids)}
    if isinstance(model, CausalGreyBoxModel):
        # These counters diagnose whether the first half-day ever exercised
        # the gains later used for counterfactual active-control predictions.
        out['prefix_excitation'] = {
            'air_excited_rooms': int(sum(sum(abs(x[2]) > 1e-6
                                        for x, _ in model.history[r][:fit_window]) >= 8
                                     for r in room_ids)),
            'reheat_excited_rooms': int(sum(sum(abs(x[3]) > 1e-6
                                           for x, _ in model.history[r][:fit_window]) >= 8
                                        for r in room_ids)),
            'required_transitions_per_room': 8,
        }
    for h in horizons:
        e = float(np.mean(err[h])) if err[h] else float('nan')
        b = float(np.mean(base_err[h])) if base_err[h] else float('nan')
        out[f'h{h}'] = e
        out[f'h{h}_persist'] = b
        out[f'h{h}_drift'] = float(np.mean(drift_err[h]))
        out[f'h{h}_skill'] = (b - e) / b if b > 1e-9 else None
        out[f'h{h}_warm'] = float(np.mean(warm_err[h])) if warm_err[h] else float('nan')
        out[f'h{h}_regimes'] = {key: dict(model=v[0] / v[2], persist=v[1] / v[2],
                                          samples=v[2])
                                 for key, v in regime_errors[h].items()}
        out[f'h{h}_occupancy_regimes'] = {
            key: dict(model=v[0] / v[2], persist=v[1] / v[2], samples=v[2])
            for key, v in occupancy_errors[h].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', nargs='*', default=[
        'building_train_1/fairbanks_1',
        'building_train_2/fairbanks_2',
        'building_train_8/fairbanks_8',
        'building_train_1/atlanta_2',
        'building_train_8/singapore_7',
    ])
    ap.add_argument('--policy', default='dsat_gen31')
    ap.add_argument('--mode', default='batch_rls', choices=['batch_rls', 'rls'])
    ap.add_argument('--fit-window', type=int, default=144)
    ap.add_argument('--horizons', nargs='*', type=int, default=[1, 3, 6, 12])
    ap.add_argument('--steps', type=int, default=288)
    ap.add_argument('--model', choices=MODEL_KINDS, default='legacy')
    ap.add_argument('--regimes', action='store_true', help='Split errors by AHU state at prediction origin')
    ap.add_argument('--oracle-actions', action='store_true',
                    help='OFFLINE ONLY: recorded future commands; no future observations')
    ap.add_argument('--all-scenarios', action='store_true',
                    help='Use every saved dump for the requested policy')
    ap.add_argument('--json', action='store_true', help='Print structured results instead of tables')
    ap.add_argument('--offset', type=int, default=0,
                    help='Start within an existing dump; fresh model, no preceding observations consumed')
    args = ap.parse_args()
    if args.oracle_actions and args.model in ('legacy', 'sensors'):
        ap.error('--oracle-actions requires an action-conditioned model')
    if not 1 <= args.fit_window < args.steps <= 288:
        ap.error('require 1 <= fit-window < steps <= 288')
    if args.offset < 0:
        ap.error('--offset must be non-negative')
    if not args.horizons or min(args.horizons) < 1 or max(args.horizons) >= args.steps - args.fit_window:
        ap.error('horizons must be positive and fit within the validation window')

    horizons = tuple(args.horizons)
    rows = []
    if args.all_scenarios:
        args.scenarios = sorted(str(p.parent.parent.relative_to(RESULTS))
                                for p in RESULTS.glob(f'*/*/nn_data_{args.policy}/obs_dump.npz'))
        if not args.scenarios:
            ap.error('No saved dumps found')
    for scen in args.scenarios:
        rows.append(evaluate(scen, args.policy, horizons, args.mode, args.fit_window,
                              steps=args.steps, model_kind=args.model,
                              oracle_actions=args.oracle_actions, offset=args.offset))

    if args.json:
        print(json.dumps(rows, indent=2, allow_nan=False))
        return

    print(f'Model: {args.model}; window: {args.steps} observations from offset {args.offset}; '
          f'warm-up: {args.fit_window} transitions; '
          f'{"recorded command path" if args.oracle_actions else "held observed inputs" if args.model in ("legacy", "sensors") else "held commands"} '
          + ('(OFFLINE ORACLE future commands; NOT an online forecast).'
           if args.oracle_actions else '(no future inputs).'))
    head = f"{'scenario':34s} {'mode':10s}"
    for h in horizons:
        head += f" | h{h}: model/persist/skill"
    print(head)
    print('-' * len(head))
    for r in rows:
        line = f"{r['scenario']:34s} {r['mode']:10s}"
        for h in horizons:
            skill = r[f'h{h}_skill']
            shown_skill = f'{skill * 100:+.0f}%' if skill is not None else 'n/a'
            line += (f" | {r[f'h{h}']:.3f}/{r[f'h{h}_persist']:.3f}"
                     f"/{shown_skill}")
        print(line)
    print('Linear-drift baseline MAE (K), same origins:')
    for r in rows:
        print(r['scenario'], ' '.join(f"h{h}={r[f'h{h}_drift']:.3f}" for h in horizons))
    if args.regimes:
        print('AHU state at origin (not an oracle future-regime split):')
        for r in rows:
            if 'prefix_excitation' in r:
                print(r['scenario'], 'warm-up excitation:', r['prefix_excitation'],
                      'of', r['rooms'], 'rooms')
            for h in horizons:
                print(r['scenario'], f'h{h}', r[f'h{h}_regimes'])


if __name__ == '__main__':
    main()
