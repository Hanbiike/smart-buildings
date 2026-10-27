# -*- coding: utf-8 -*-
"""Tokenize dumped trajectories into cached feature tensors for BC training.

Per (scenario, teacher) run recorded by --dump-obs, builds:

    zone_feats  [T, Z, 8]   per-room features (band deltas, commands)
    ahu_feats   [T, A, 10]  per-AHU features (OAT, SAT, pressure, flow, cmds)
    glob_feats  [T, Fg]     occupant, schedule harmonics, hot-water system
    cont        [T, C]      continuous action target (as recorded)
    disc        [T, Nd]     discrete run targets (as recorded)
    rewards     [T]
    mapping.json            room/AHU -> action index maps (training targets)

All feature math and the room/AHU wiring come from the single shared module
`smart_buildings.nn_model.FeatureContext` — the identical code path the
deployed NN policy runs — so training and deployment cannot drift apart.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.nn_model import FeatureContext  # noqa: E402

RESULTS = Path(PROJECT_ROOT) / "smart_buildings" / "results"


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", default="building_train_1,building_train_2,"
                                         "building_train_4,building_train_5,"
                                         "building_train_7,building_train_8")
    ap.add_argument("--weathers", default="all")
    ap.add_argument("--policies", default="dsat_gen31,dsat_gen2")
    return ap.parse_args()


def tokenize_scenario(building: str, weather: str, policy_name: str) -> str:
    run_dir = RESULTS / building / weather / f"nn_data_{policy_name}"
    npz_path = run_dir / "obs_dump.npz"
    meta_path = RESULTS / building / weather / "scenario_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return f"skip {building}/{weather}/{policy_name}"

    meta = json.load(open(meta_path))
    ctx = FeatureContext(meta)

    z = np.load(npz_path)
    obs = z["obs"].astype(np.float64)          # [T, D] as the policy saw it
    cont = z["action_continuous"].astype(np.float32)
    disc = z["action_discrete"].astype(np.int64)
    rewards = z["rewards"].astype(np.float32)
    T = obs.shape[0]

    zone_feats = ctx.room_feats(obs)           # [T, Z, 8]
    ahu_feats = ctx.ahu_feats(obs)             # [T, A, 10]
    glob_feats = ctx.glob_feats(obs)           # [T, Fg]

    if "state" in z:
        state = z["state"].astype(np.float32)
        nz, na = ctx.Z * 2, ctx.A * 3
        expect = nz + na + 2
        if state.shape[1] == expect and state.shape[0] == T:
            z_ex = state[:, :nz].reshape(T, ctx.Z, 2)      # u, integral
            z_ex = z_ex.copy()
            z_ex[..., 1] /= 100.0                          # clip(±200) -> O(1)
            a_ex = state[:, nz:nz + na].reshape(T, ctx.A, 3)  # emas, hold
            g_ex = state[:, nz + na:]                      # pi_active, occ
            zone_feats = np.concatenate([zone_feats, z_ex], axis=2)
            ahu_feats = np.concatenate([ahu_feats, a_ex], axis=2)
            glob_feats = np.concatenate([glob_feats, g_ex], axis=1)

    mapping = {
        "rooms": [{"damp_act": r["damp_act"], "reheat_act": r["reheat_act"],
                   "ahu": r["ahu"]} for r in ctx.rooms],
        "ahus": ctx.ahu_actions,
        "water": ctx.water,
    }
    out = run_dir / "tokens.npz"
    np.savez_compressed(
        out,
        zone_feats=zone_feats, ahu_feats=ahu_feats, glob_feats=glob_feats,
        cont=cont, disc=disc, rewards=rewards,
    )
    with open(run_dir / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=1)
    return (f"ok {building}/{weather}/{policy_name} T={T} Z={ctx.Z} "
            f"A={ctx.A} Fg={glob_feats.shape[1]}")


def main():
    args = parse_arguments()
    buildings = args.buildings.split(",")
    policies = args.policies.split(",")
    for building in buildings:
        weather_dir = RESULTS / building
        if not weather_dir.exists():
            continue
        weathers = (args.weathers.split(",") if args.weathers != "all"
                    else sorted(p.name for p in weather_dir.iterdir() if p.is_dir()))
        for weather in weathers:
            if not (weather_dir / weather).is_dir():
                continue
            for policy_name in policies:
                print(tokenize_scenario(building, weather, policy_name),
                      flush=True)


if __name__ == "__main__":
    main()
