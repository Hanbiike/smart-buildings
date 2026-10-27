# -*- coding: utf-8 -*-
"""Fit slot-linear heads from teacher trajectories (ridge least squares).

For every (bank, slot): accumulate X'X / X'Y sufficient statistics over the
teacher's (obs, action) pairs, then solve. Zone/AHU/global heads are fit
separately with the shared-across-devices structure the policy uses at
inference. Run bits are fit as regression to {0,1} (thresholded at 0.5).

MASKING: only teacher steps where the equipment runs train the heads
(zone/AHU rows gated on their AHU's run bit, water on the loop run bit).
Shutdown steps teach a reversed need response (cold zones -> minimums)
that the hybrid then replays as a cold-start trap; the hybrid never
queries heads in shutdown anyway. Memory advances only on delegating
steps (feat.step update=False otherwise) to match inference dynamics.

Slot assignment is hard (each step trains its own slot's heads); the
inference-time neighbor blend smooths across them.

Outputs results/slot_runs/<out>.npz with Wz/bz/Wa/ba/Wg/bg.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.baselines.slot_linear import (  # noqa: E402
    BANKS,
    N_SLOTS,
    SlotFeatures,
)

RESULTS = PROJECT_ROOT / "smart_buildings" / "results"
RUNS_DIR = RESULTS / "slot_runs"


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="occ_gated4",
                    help="comma-separated dump tags to aggregate (DAgger)")
    ap.add_argument("--hold-building", default="building_train_5")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out", default="slot_bc_occ4")
    ap.add_argument("--scenarios", default="all",
                    help="'all' or comma list building/weather")
    return ap.parse_args()


def find_runs(teachers, only=None):
    if isinstance(teachers, str):
        teachers = [teachers]
    runs = []
    for building_dir in sorted(RESULTS.glob("building_train_*")):
        if not building_dir.is_dir():
            continue
        for weather_dir in sorted(building_dir.iterdir()):
            if not weather_dir.is_dir():
                continue
            key = f"{building_dir.name}/{weather_dir.name}"
            if only is not None and key not in only:
                continue
            meta = weather_dir / "scenario_meta.json"
            for teacher in teachers:
                dump = weather_dir / f"nn_data_{teacher}" / "obs_dump.npz"
                if dump.exists() and meta.exists():
                    runs.append((f"{key} [{teacher}]", dump, meta))
    return runs


class Accum:
    """Sufficient statistics for ridge regression: X'X, X'Y, n."""

    def __init__(self, f, o):
        self.A = np.zeros((f + 1, f + 1))
        self.B = np.zeros((f + 1, o))
        self.n = 0

    def add(self, X, Y):
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.A += X1.T @ X1
        self.B += X1.T @ Y
        self.n += X.shape[0]

    def solve(self, ridge):
        if self.n == 0:
            return None, None
        A = self.A + ridge * np.eye(self.A.shape[0])
        A[-1, -1] -= ridge  # do not penalize the bias
        W1 = np.linalg.solve(A, self.B)
        return W1[:-1], W1[-1]


def main():
    args = parse_arguments()
    only = None if args.scenarios == "all" else set(args.scenarios.split(","))
    runs = find_runs(args.teacher.split(","), only)
    print(f"teacher runs: {len(runs)}", flush=True)
    if not runs:
        raise SystemExit("no teacher dumps found")

    # dims from the first run's meta
    meta0 = json.load(open(runs[0][2]))
    feat0 = SlotFeatures(meta0)
    FZ, FA, FG = feat0.FZ, feat0.FA, feat0.FG
    print(f"dims: FZ={FZ} FA={FA} FG={FG}", flush=True)

    acc_z = [[Accum(FZ, 2) for _ in range(N_SLOTS)] for _ in range(2)]
    acc_a = [[Accum(FA, 4) for _ in range(N_SLOTS)] for _ in range(2)]
    acc_g = [[Accum(FG, 3) for _ in range(N_SLOTS)] for _ in range(2)]
    on_a = np.zeros((2, N_SLOTS))
    n_a = np.zeros((2, N_SLOTS))
    on_w = np.zeros((2, N_SLOTS))
    n_w = np.zeros((2, N_SLOTS))

    for key, dump_path, meta_path in runs:
        meta = json.load(open(meta_path))
        feat = SlotFeatures(meta)
        z = np.load(dump_path)
        obs, cont = z["obs"], z["action_continuous"].astype(np.float64)
        disc = z["action_discrete"].astype(np.float64)
        rooms = feat.ctx.rooms
        damp = [r["damp_act"] for r in rooms]
        reh = [r["reheat_act"] for r in rooms]
        ahus = feat.ctx.ahu_actions
        w = feat.ctx.water
        ahu_of_room = np.array([r["ahu"] for r in rooms])
        run_disc = [a["run_disc_idx"] for a in ahus]
        T = obs.shape[0]
        n_deleg = 0
        for t in range(T):
            ahu_on = disc[t, run_disc] > 0.5
            deleg = bool(ahu_on.any())
            bank, slot, (zf, af, gf) = feat.step(obs[t], update=deleg)
            ya_run = disc[t, run_disc]
            on_a[bank, slot] += ya_run.sum()
            n_a[bank, slot] += len(ahus)
            w_run = (disc[t, w["water_discrete_idx"]]
                     if 0 <= w["water_discrete_idx"] < disc.shape[1] else 0.0)
            on_w[bank, slot] += w_run
            n_w[bank, slot] += 1
            if not deleg:
                continue
            n_deleg += 1
            zkeep = ahu_on[ahu_of_room]
            if zkeep.any():
                yz = np.stack([cont[t, damp], cont[t, reh]], axis=1)
                acc_z[bank][slot].add(zf[zkeep].astype(np.float64),
                                      yz[zkeep])
            ya = np.zeros((ahu_on.sum(), 4))
            kept_af = af[ahu_on].astype(np.float64)
            for j, i in enumerate(np.where(ahu_on)[0]):
                a = ahus[i]
                ya[j, 0] = cont[t, a["temp_idx"]]
                ya[j, 1] = (cont[t, a["pressure_idx"]]
                            if a["pressure_idx"] is not None else -1.0)
                ya[j, 2] = (cont[t, a["recirc_idx"]]
                            if a["recirc_idx"] is not None else 1.0)
                ya[j, 3] = disc[t, a["run_disc_idx"]]
            acc_a[bank][slot].add(kept_af, ya)
            yg = np.zeros(3)
            yg[0] = (cont[t, w["water_temp_idx"]]
                     if w["water_temp_idx"] is not None else -1.0)
            yg[1] = (cont[t, w["water_pressure_idx"]]
                     if w["water_pressure_idx"] is not None else -1.0)
            yg[2] = w_run
            if yg[2] > 0.5:
                acc_g[bank][slot].add(gf.astype(np.float64)[None, :],
                                      yg[None, :])
        print(f"  {key}: T={T} deleg={n_deleg}", flush=True)

    Wz = np.zeros((2, N_SLOTS, FZ, 2))
    bz = np.zeros((2, N_SLOTS, 2))
    Wa = np.zeros((2, N_SLOTS, FA, 4))
    ba = np.zeros((2, N_SLOTS, 4))
    Wg = np.zeros((2, N_SLOTS, FG, 3))
    bg = np.zeros((2, N_SLOTS, 3))
    def fill(W, b, acc):
        for bank in range(2):
            have = [s for s in range(N_SLOTS) if acc[bank][s].n > 0]
            for s in have:
                Ws, bs = acc[bank][s].solve(args.ridge)
                W[bank, s], b[bank, s] = Ws, bs
            # empty slots inherit the bank mean head (two passes: the mean
            # must see all solved heads)
            Wm = np.mean([W[bank, h] for h in have], axis=0)
            bm = np.mean([b[bank, h] for h in have], axis=0)
            for s in range(N_SLOTS):
                if s not in have:
                    W[bank, s], b[bank, s] = Wm, bm

    fill(Wz, bz, acc_z)
    fill(Wa, ba, acc_a)
    fill(Wg, bg, acc_g)
    nz = np.array([[a.n for a in row] for row in acc_z])
    na = np.array([[a.n for a in row] for row in acc_a])
    ng = np.array([[a.n for a in row] for row in acc_g])
    for name, n in (("zone", nz), ("ahu", na), ("glob", ng)):
        print(f"{name}: samples/slot min={n.min()} med={np.median(n):.0f} "
              f"zeros={(n == 0).sum()}", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.out}.npz"
    rate_a = np.divide(on_a, np.maximum(n_a, 1),
                       out=np.full_like(on_a, 0.5), where=n_a > 0)
    rate_w = np.divide(on_w, np.maximum(n_w, 1),
                       out=np.full_like(on_w, 0.5), where=n_w > 0)
    np.savez_compressed(out, Wz=Wz, bz=bz, Wa=Wa, ba=ba, Wg=Wg, bg=bg,
                        rate_a=rate_a, rate_w=rate_w,
                        ridge=args.ridge, teacher=args.teacher)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
