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
inference-time neighbor blend smooths across them. With --soft k3/k5,
each step's statistics spread to neighbor slots with binomial weights
(circular mod 288, bank-hard): train-time soft assignment (MoE T1/T2).

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
    ap.add_argument("--soft", default="none", choices=("none", "k3", "k5"),
                    help="train-time soft slot assignment kernel "
                         "(k3=(1,2,1)/4, k5=(1,4,6,4,1)/16, circular, "
                         "bank-hard); none = hard assignment")
    ap.add_argument("--no-holdout", action="store_true",
                    help="train on all buildings including --hold-building "
                         "(default excludes it for held-out comparison)")
    ap.add_argument("--residual-base", default=None,
                    help="P22-6: fit delta=T-B instead of T, where B is the "
                         "base replay in B_<tag>.npy next to each dump")
    ap.add_argument("--modes", type=int, default=1,
                    help="P23-4: climate modes (1 = legacy bank/slot only)")
    ap.add_argument("--mode-bounds", default="283.15",
                    help="P23-4: comma OAT bounds in K (K-1 values)")
    return ap.parse_args()


SOFT_KERNELS = {
    "none": ((0,), (1.0,)),
    "k3": ((-1, 0, 1), (0.25, 0.5, 0.25)),
    "k5": ((-2, -1, 0, 1, 2),
           (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)),
}


def find_runs(teachers, only=None, exclude_building=None):
    if isinstance(teachers, str):
        teachers = [teachers]
    runs = []
    for building_dir in sorted(RESULTS.glob("building_train_*")):
        if not building_dir.is_dir():
            continue
        if exclude_building and building_dir.name == exclude_building:
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

    def add(self, X, Y, w=1.0):
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.A += w * (X1.T @ X1)
        self.B += w * (X1.T @ Y)
        self.n += w * X.shape[0]

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
    excl = None if args.no_holdout else args.hold_building
    runs = find_runs(args.teacher.split(","), only, excl)
    print(f"teacher runs: {len(runs)} (soft={args.soft}, "
          f"holdout={excl})", flush=True)
    if not runs:
        raise SystemExit("no teacher dumps found")
    offsets, weights = SOFT_KERNELS[args.soft]

    # dims from the first run's meta
    meta0 = json.load(open(runs[0][2]))
    feat0 = SlotFeatures(meta0)
    FZ, FA, FG = feat0.FZ, feat0.FA, feat0.FG
    print(f"dims: FZ={FZ} FA={FA} FG={FG}", flush=True)

    K = args.modes
    if K == 1:
        bounds: tuple = ()
    else:
        bounds = tuple(float(x) for x in args.mode_bounds.split(","))
        assert len(bounds) == K - 1, (bounds, K)
    print(f"modes: K={K} bounds={bounds}", flush=True)

    acc_z = [[[Accum(FZ, 2) for _ in range(N_SLOTS)] for _ in range(2)]
             for _ in range(K)]
    acc_a = [[[Accum(FA, 4) for _ in range(N_SLOTS)] for _ in range(2)]
             for _ in range(K)]
    acc_g = [[[Accum(FG, 3) for _ in range(N_SLOTS)] for _ in range(2)]
             for _ in range(K)]
    on_a = np.zeros((K, 2, N_SLOTS))
    n_a = np.zeros((K, 2, N_SLOTS))
    on_w = np.zeros((K, 2, N_SLOTS))
    n_w = np.zeros((K, 2, N_SLOTS))

    for key, dump_path, meta_path in runs:
        meta = json.load(open(meta_path))
        feat = SlotFeatures(meta)
        feat.mode_bounds = bounds
        z = np.load(dump_path)
        obs, cont = z["obs"], z["action_continuous"].astype(np.float64)
        disc = z["action_discrete"].astype(np.float64)
        base_c = None
        if args.residual_base:
            bp = Path(dump_path).parent / f"B_{args.residual_base}.npy"
            base_c = np.load(bp).astype(np.float64)
            assert base_c.shape == cont.shape, (key, base_c.shape)
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
            mode = feat.mode
            ya_run = disc[t, run_disc]
            w_run = (disc[t, w["water_discrete_idx"]]
                     if 0 <= w["water_discrete_idx"] < disc.shape[1] else 0.0)
            for off, wt in zip(offsets, weights):
                s2 = (slot + off) % N_SLOTS  # circular, bank-hard
                on_a[mode, bank, s2] += wt * ya_run.sum()
                n_a[mode, bank, s2] += wt * len(ahus)
                on_w[mode, bank, s2] += wt * w_run
                n_w[mode, bank, s2] += wt
            if not deleg:
                continue
            n_deleg += 1
            bt = base_c[t] if base_c is not None else None
            zkeep = ahu_on[ahu_of_room]
            if zkeep.any():
                yz = np.stack([cont[t, damp], cont[t, reh]], axis=1)
                if bt is not None:
                    yz = yz - np.stack([bt[damp], bt[reh]], axis=1)
                Xz = zf[zkeep].astype(np.float64)
                Yz = yz[zkeep]
            else:
                Xz = Yz = None
            ya = np.zeros((ahu_on.sum(), 4))
            kept_af = af[ahu_on].astype(np.float64)
            for j, i in enumerate(np.where(ahu_on)[0]):
                a = ahus[i]
                ya[j, 0] = cont[t, a["temp_idx"]]
                ya[j, 1] = (cont[t, a["pressure_idx"]]
                            if a["pressure_idx"] is not None else -1.0)
                ya[j, 2] = (cont[t, a["recirc_idx"]]
                            if a["recirc_idx"] is not None else 1.0)
                if bt is not None:
                    ya[j, 0] -= bt[a["temp_idx"]]
                    if a["pressure_idx"] is not None:
                        ya[j, 1] -= bt[a["pressure_idx"]]
                    if a["recirc_idx"] is not None:
                        ya[j, 2] -= bt[a["recirc_idx"]]
                ya[j, 3] = disc[t, a["run_disc_idx"]]
            yg = np.zeros(3)
            yg[0] = (cont[t, w["water_temp_idx"]]
                     if w["water_temp_idx"] is not None else -1.0)
            yg[1] = (cont[t, w["water_pressure_idx"]]
                     if w["water_pressure_idx"] is not None else -1.0)
            if bt is not None:
                if w["water_temp_idx"] is not None:
                    yg[0] -= bt[w["water_temp_idx"]]
                if w["water_pressure_idx"] is not None:
                    yg[1] -= bt[w["water_pressure_idx"]]
            yg[2] = w_run
            Xg = gf.astype(np.float64)[None, :]
            for off, wt in zip(offsets, weights):
                s2 = (slot + off) % N_SLOTS  # circular, bank-hard
                if Xz is not None:
                    acc_z[mode][bank][s2].add(Xz, Yz, w=wt)
                acc_a[mode][bank][s2].add(kept_af, ya, w=wt)
                if yg[2] > 0.5:
                    acc_g[mode][bank][s2].add(Xg, yg[None, :], w=wt)
        print(f"  {key}: T={T} deleg={n_deleg}", flush=True)

    Wz = np.zeros((K, 2, N_SLOTS, FZ, 2))
    bz = np.zeros((K, 2, N_SLOTS, 2))
    Wa = np.zeros((K, 2, N_SLOTS, FA, 4))
    ba = np.zeros((K, 2, N_SLOTS, 4))
    Wg = np.zeros((K, 2, N_SLOTS, FG, 3))
    bg = np.zeros((K, 2, N_SLOTS, 3))
    def fill(W, b, acc):
        for m in range(K):
            for bank in range(2):
                have = [s for s in range(N_SLOTS) if acc[m][bank][s].n > 0]
                for s in have:
                    Ws, bs = acc[m][bank][s].solve(args.ridge)
                    W[m, bank, s], b[m, bank, s] = Ws, bs
                # Empty slots inherit the mode-bank mean head (two passes:
                # the mean must see all solved heads). A fully starved
                # mode-bank falls back to the bank mean over all modes.
                if have:
                    Wm = np.mean([W[m, bank, h] for h in have], axis=0)
                    bm = np.mean([b[m, bank, h] for h in have], axis=0)
                else:
                    Wm = np.mean([W[mm, bank] for mm in range(K)], axis=(0, 1))
                    bm = np.mean([b[mm, bank] for mm in range(K)], axis=(0, 1))
                for s in range(N_SLOTS):
                    if s not in have:
                        W[m, bank, s], b[m, bank, s] = Wm, bm

    fill(Wz, bz, acc_z)
    fill(Wa, ba, acc_a)
    fill(Wg, bg, acc_g)
    nz = np.array([[[a.n for a in row] for row in modes] for modes in acc_z])
    na = np.array([[[a.n for a in row] for row in modes] for modes in acc_a])
    ng = np.array([[[a.n for a in row] for row in modes] for modes in acc_g])
    for name, n in (("zone", nz), ("ahu", na), ("glob", ng)):
        print(f"{name}: samples/slot min={n.min()} med={np.median(n):.0f} "
              f"zeros={(n == 0).sum()}", flush=True)
        for m in range(K):
            print(f"  mode{m}: min={n[m].min()} med={np.median(n[m]):.0f} "
                  f"zeros={(n[m] == 0).sum()}", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.out}.npz"
    rate_a = np.divide(on_a, np.maximum(n_a, 1),
                       out=np.full_like(on_a, 0.5), where=n_a > 0)
    rate_w = np.divide(on_w, np.maximum(n_w, 1),
                       out=np.full_like(on_w, 0.5), where=n_w > 0)
    if K == 1:
        # legacy 4-D format (bit-exact with the pre-P23-4 trainer)
        payload = dict(Wz=Wz[0], bz=bz[0], Wa=Wa[0], ba=ba[0],
                       Wg=Wg[0], bg=bg[0], rate_a=rate_a[0],
                       rate_w=rate_w[0], nz=nz[0], na=na[0], ng=ng[0],
                       on_a=on_a[0], n_a=n_a[0], on_w=on_w[0], n_w=n_w[0])
    else:
        payload = dict(Wz=Wz, bz=bz, Wa=Wa, ba=ba, Wg=Wg, bg=bg,
                       rate_a=rate_a, rate_w=rate_w,
                       nz=nz, na=na, ng=ng, on_a=on_a, n_a=n_a,
                       on_w=on_w, n_w=n_w,
                       modes=np.asarray(bounds, dtype=np.float64))
    np.savez_compressed(out, ridge=args.ridge, teacher=args.teacher,
                        soft=args.soft, holdout=(excl or ""), **payload)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
