# MPC gate 2: held-out multistep forecasts vs persistence/drift.
# Fits the causal greybox on the first FIT steps of a saved dump, then
# evaluates rolling origins over the rest (every EVAL_EVERY steps).
# Horizons include 2 (first causal response, see timing_audit) through 144.
# Sim-free: works on nn_data_* obs dumps.
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smart_buildings.baselines.causal_greybox import CausalGreyBoxModel  # noqa: E402

RESULTS = ROOT / 'smart_buildings' / 'results'
HORIZONS = (2, 6, 12, 48, 144)
FIT = 288
EVAL_EVERY = 24


def run_case(scenario, policy='occ_gated4'):
    base = RESULTS / scenario
    meta = json.loads((base / 'scenario_meta.json').read_text())
    with np.load(base / f'nn_data_{policy}' / 'obs_dump.npz') as dump:
        obs = dump['obs'][:]
        continuous = dump['action_continuous'][:]
        discrete = dump['action_discrete'][:]
    model = CausalGreyBoxModel(meta, fit_window=FIT)
    room_ids = sorted(model.room_temp_obs)
    err = {h: [] for h in HORIZONS}
    per = {h: [] for h in HORIZONS}
    dri = {h: [] for h in HORIZONS}
    n = obs.shape[0]
    for t in range(n):
        model.update(obs[t])
        model.set_action(continuous[t], discrete[t])
        if t < FIT or (t - FIT) % EVAL_EVERY:
            continue
        for room in room_ids:
            t_now = model.zone_temp(obs[t], room)
            prev = model.zone_temp(obs[t - 1], room)
            for h in HORIZONS:
                if t + h >= n:
                    continue
                actual = model.zone_temp(obs[t + h], room)
                pred = model.predict(room, h, temp0=t_now)
                err[h].append(abs(pred - actual))
                per[h].append(abs(t_now - actual))
                dri[h].append(abs(t_now + h * (t_now - prev) - actual))
    out = {'scenario': scenario, 'policy': policy, 'rooms': len(room_ids)}
    for h in HORIZONS:
        e, b, d = (float(np.mean(err[h])), float(np.mean(per[h])),
                   float(np.mean(dri[h])))
        out[f'h{h}'] = dict(model=e, persist=b, drift=d,
                            skill=(b - e) / b if b > 1e-9 else None,
                            origins=len(err[h]))
    return out


def main():
    scenarios = sys.argv[1:] or ['building_train_1/atlanta_2']
    for sc in scenarios:
        r = run_case(sc)
        print(f"== {r['scenario']} rooms={r['rooms']}")
        for h in HORIZONS:
            d = r[f'h{h}']
            print(f"  h{h:>3}: model={d['model']:.4f} persist={d['persist']:.4f} "
                  f"drift={d['drift']:.4f} skill={d['skill']:+.3f} n={d['origins']}")


if __name__ == '__main__':
    main()
