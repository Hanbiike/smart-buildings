# -*- coding: utf-8 -*-
"""Held-out comparison: hard-fit + deploy-blend vs soft-spread BC (MoE T1).

Loads two slot_train.py artifacts, replays the held-out building's teacher
dumps through SlotFeatures (update=deleg, same dynamics as training),
predicts continuous commands with each variant's k3 deploy-blended heads,
and reports action-MSE overall and on deleg-starved slots (hard nz == 0).

T1 gate: soft MSE < hard MSE on held-out by >= 0.0002 (the note's bar, in
the offline metric's units — billing cannot move it since teacher actions
are billing-invariant), with night-slot predictions moving from the bank
mean toward neighbor interpolation; else CLOSE the line.
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
    N_SLOTS,
    SlotFeatures,
    blend_weights,
)
from tools.slot_train import find_runs  # noqa: E402

BLEND = blend_weights((1, 2, 1))
HALF = len(BLEND) // 2
GATE = 0.0002


def blended(W, b, bank, slot):
    Wm = np.zeros_like(W[bank, slot], dtype=np.float64)
    bm = np.zeros_like(b[bank, slot], dtype=np.float64)
    for j, w in enumerate(BLEND):
        s = (slot + j - HALF) % N_SLOTS
        Wm += w * W[bank, s]
        bm += w * b[bank, s]
    return Wm, bm


def load(path):
    z = np.load(path)
    return {k: z[k] for k in
            ("Wz", "bz", "Wa", "ba", "Wg", "bg", "nz", "na", "ng")}


def predict(H, bank, slot, zf, af, gf):
    Wz, bz = blended(H["Wz"], H["bz"], bank, slot)
    Wa, ba = blended(H["Wa"], H["ba"], bank, slot)
    Wg, bg = blended(H["Wg"], H["bg"], bank, slot)
    return (zf.astype(np.float64) @ Wz + bz,
            af.astype(np.float64) @ Wa + ba,
            gf.astype(np.float64) @ Wg + bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", required=True)
    ap.add_argument("--soft", required=True)
    ap.add_argument("--teacher", default="occ_econdbhhot9")
    ap.add_argument("--hold-building", default="building_train_5")
    args = ap.parse_args()

    H = load(args.hard)
    S = load(args.soft)
    runs = [r for r in find_runs(args.teacher)
            if r[0].split(" ")[0].split("/")[0] == args.hold_building]
    print(f"held-out runs: {len(runs)}", flush=True)
    if not runs:
        raise SystemExit("no held-out dumps found")

    se = {k: np.zeros(2) for k in ("z", "a", "g")}      # [hard, soft]
    n = {k: np.zeros(2) for k in ("z", "a", "g")}
    se_starved = np.zeros(2)
    n_starved = np.zeros(2)
    se_night = np.zeros(2)   # slot in 0..72 (00:00-06:00), delegating
    n_night = np.zeros(2)
    se_edge = np.zeros(2)    # slots within +-6 of 72/228 (06:00/19:00 steps)
    n_edge = np.zeros(2)

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
        for t in range(obs.shape[0]):
            ahu_on = disc[t, run_disc] > 0.5
            deleg = bool(ahu_on.any())
            bank, slot, (zf, af, gf) = feat.step(obs[t], update=deleg)
            if not deleg:
                continue
            w_run = (disc[t, w["water_discrete_idx"]]
                     if 0 <= w["water_discrete_idx"] < disc.shape[1]
                     else 0.0)
            zkeep = ahu_on[ahu_of_room]
            yz = (np.stack([cont[t, damp], cont[t, reh]], axis=1)[zkeep]
                  if zkeep.any() else None)
            ya = np.zeros((ahu_on.sum(), 4))
            for j, i in enumerate(np.where(ahu_on)[0]):
                a = ahus[i]
                ya[j, 0] = cont[t, a["temp_idx"]]
                ya[j, 1] = (cont[t, a["pressure_idx"]]
                            if a["pressure_idx"] is not None else -1.0)
                ya[j, 2] = (cont[t, a["recirc_idx"]]
                            if a["recirc_idx"] is not None else 1.0)
                ya[j, 3] = disc[t, a["run_disc_idx"]]
            yg = np.array([
                cont[t, w["water_temp_idx"]]
                if w["water_temp_idx"] is not None else -1.0,
                cont[t, w["water_pressure_idx"]]
                if w["water_pressure_idx"] is not None else -1.0,
                w_run])
            for vi, HH in enumerate((H, S)):
                pz, pa, pg = predict(HH, bank, slot, zf, af, gf)
                if yz is not None:
                    e = pz[zkeep] - yz
                    se["z"][vi] += float((e ** 2).sum())
                    n["z"][vi] += e.size
                e = pa[ahu_on] - ya
                se["a"][vi] += float((e ** 2).sum())
                n["a"][vi] += e.size
                if w_run > 0.5:
                    e = pg - yg
                    se["g"][vi] += float((e ** 2).sum())
                    n["g"][vi] += e.size
                # starved-slot diagnostic: hard saw ~nothing here
                if H["nz"][bank, slot] == 0 and yz is not None:
                    e = (pz[zkeep] - yz)
                    se_starved[vi] += float((e ** 2).sum())
                    n_starved[vi] += e.size
                if slot <= 72:
                    tot, cnt = 0.0, 0
                    if yz is not None:
                        e = pz[zkeep] - yz
                        tot += float((e ** 2).sum())
                        cnt += e.size
                    e = pa[ahu_on] - ya
                    tot += float((e ** 2).sum())
                    cnt += e.size
                    se_night[vi] += tot
                    n_night[vi] += cnt
                if min(abs(slot - 72), abs(slot - 228)) <= 6:
                    tot, cnt = 0.0, 0
                    if yz is not None:
                        e = pz[zkeep] - yz
                        tot += float((e ** 2).sum())
                        cnt += e.size
                    e = pa[ahu_on] - ya
                    tot += float((e ** 2).sum())
                    cnt += e.size
                    se_edge[vi] += tot
                    n_edge[vi] += cnt

    print("head-group MSE (hard vs soft):", flush=True)
    tot = np.zeros(2)
    cnt = np.zeros(2)
    for k in ("z", "a", "g"):
        mh, ms = se[k][0] / max(n[k][0], 1), se[k][1] / max(n[k][1], 1)
        print(f"  {k}: hard {mh:.6f} soft {ms:.6f} "
              f"delta {ms - mh:+.6f} (n={int(n[k][0])})", flush=True)
        tot += se[k]
        cnt += n[k]
    mh, ms = tot[0] / max(cnt[0], 1), tot[1] / max(cnt[1], 1)
    print(f"OVERALL: hard {mh:.6f} soft {ms:.6f} delta {ms - mh:+.6f}",
          flush=True)
    if n_starved[0] > 0:
        print(f"STARVED-SLOT zone MSE (hard-nz==0, n={int(n_starved[0])}): "
              f"hard {se_starved[0] / n_starved[0]:.6f} "
              f"soft {se_starved[1] / n_starved[1]:.6f}",
              flush=True)
    else:
        print("STARVED-SLOT zone MSE: no delegating steps on hard-empty "
              "slots", flush=True)
    if n_night[0] > 0:
        print(f"NIGHT (slot<=72) MSE: hard {se_night[0] / n_night[0]:.6f} "
              f"soft {se_night[1] / n_night[1]:.6f}", flush=True)
    if n_edge[0] > 0:
        print(f"EDGE (72/228 +-6) MSE: hard {se_edge[0] / n_edge[0]:.6f} "
              f"soft {se_edge[1] / n_edge[1]:.6f}", flush=True)
    win = mh - ms
    print(f"T1 GATE (soft wins by >= {GATE}): "
          f"{'PASS' if win >= GATE else 'FAIL'} (win {win:+.6f})",
          flush=True)


if __name__ == "__main__":
    main()
