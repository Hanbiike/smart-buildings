# -*- coding: utf-8 -*-
"""P23-5: offline per-channel R2 of slot heads on teacher demos.

Replays teacher obs through SlotFeatures (train-identical deleg),
predicts with deploy-blended heads from an .npz, and reports
R2 = 1 - SS_res/SS_tot per used channel (zone damp/reheat, AHU
SAT/press/recirc, water T/P) per pair. Baseline-vs-candidate
deltas gate feature augmentation (P23-5).
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
    N_SLOTS, SlotFeatures, blend_weights,
)

RESULTS = PROJECT_ROOT / "smart_buildings" / "results"
BLEND = blend_weights((1, 2, 1))
HALF = len(BLEND) // 2

CHANNELS = ("z.damp", "z.reheat", "a.sat", "a.press", "a.recirc",
            "w.temp", "w.press")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--teacher", default="occ_econdbhhot9")
    ap.add_argument("--residual-base", default="occ_gated7")
    ap.add_argument("--pairs", default="building_train_8/fairbanks_8,"
                    "building_train_8/singapore_7,building_train_5/phoenix_1")
    return ap.parse_args()


def blended(W, b, bank, slot):
    Wm = np.zeros_like(W[bank, slot], dtype=np.float64)
    bm = np.zeros_like(b[bank, slot], dtype=np.float64)
    for j, w in enumerate(BLEND):
        s = (slot + j - HALF) % N_SLOTS
        Wm += w * W[bank, s]
        bm += w * b[bank, s]
    return Wm, bm


def main():
    args = parse_args()
    z = np.load(args.npz)
    assert z["Wz"].ndim == 4, "R2 checker supports K=1 heads only"
    Wz, bz, Wa, ba, Wg, bg = (z[k].astype(np.float64) for k in
                              ("Wz", "bz", "Wa", "ba", "Wg", "bg"))
    print(f"heads: FZ={Wz.shape[2]} FA={Wa.shape[2]} FG={Wg.shape[2]}")
    for pair in args.pairs.split(","):
        b, wthr = pair.split("/")
        wdir = RESULTS / b / wthr
        meta = json.load(open(wdir / "scenario_meta.json"))
        feat = SlotFeatures(meta)
        assert feat.FZ >= Wz.shape[2], (pair, feat.FZ, Wz.shape)
        assert feat.FA >= Wa.shape[2], (pair, feat.FA, Wa.shape)
        assert feat.FG >= Wg.shape[2], (pair, feat.FG, Wg.shape)
        d = np.load(wdir / f"nn_data_{args.teacher}" / "obs_dump.npz")
        obs = d["obs"]
        cont = d["action_continuous"].astype(np.float64)
        disc = d["action_discrete"].astype(np.float64)
        base = np.load(wdir / f"nn_data_{args.teacher}"
                       / f"B_{args.residual_base}.npy").astype(np.float64)
        rooms = feat.ctx.rooms
        damp = [r["damp_act"] for r in rooms]
        reh = [r["reheat_act"] for r in rooms]
        ahus = feat.ctx.ahu_actions
        w = feat.ctx.water
        ahu_of_room = np.array([r["ahu"] for r in rooms])
        run_disc = [a["run_disc_idx"] for a in ahus]
        ss_res = {c: 0.0 for c in CHANNELS}
        ss_tot = {c: 0.0 for c in CHANNELS}
        mean = {c: 0.0 for c in CHANNELS}
        cnt = {c: 0 for c in CHANNELS}

        def acc(ch, y, p):
            y = np.asarray(y, float).ravel()
            p = np.asarray(p, float).ravel()
            for yi, pi in zip(y, p):
                cnt[ch] += 1
                d = yi - mean[ch]
                mean[ch] += d / cnt[ch]
                ss_tot[ch] += d * (yi - mean[ch])
                ss_res[ch] += (yi - pi) ** 2

        n_deleg = 0
        for t in range(obs.shape[0]):
            ahu_on = disc[t, run_disc] > 0.5
            deleg = bool(ahu_on.any())
            bank, slot, (zf, af, gf) = feat.step(obs[t], update=deleg)
            if not deleg:
                continue
            n_deleg += 1
            bt = base[t]
            Wzb, bzb = blended(Wz, bz, bank, slot)
            Wab, bab = blended(Wa, ba, bank, slot)
            Wgb, bgb = blended(Wg, bg, bank, slot)
            zkeep = ahu_on[ahu_of_room]
            if zkeep.any():
                Xz = zf[zkeep].astype(np.float64)[:, :Wzb.shape[0]]
                Pz = Xz @ Wzb + bzb
                Yz = (np.stack([cont[t, damp], cont[t, reh]], 1)
                      - np.stack([bt[damp], bt[reh]], 1))[zkeep]
                acc("z.damp", Yz[:, 0], Pz[:, 0])
                acc("z.reheat", Yz[:, 1], Pz[:, 1])
            if ahu_on.any():
                Xa = af[ahu_on].astype(np.float64)[:, :Wab.shape[0]]
                Pa = Xa @ Wab + bab
                Ya = np.zeros((ahu_on.sum(), 3))
                for j, i in enumerate(np.where(ahu_on)[0]):
                    a = ahus[i]
                    Ya[j, 0] = cont[t, a["temp_idx"]] - bt[a["temp_idx"]]
                    Ya[j, 1] = ((cont[t, a["pressure_idx"]]
                                 - bt[a["pressure_idx"]])
                                if a["pressure_idx"] is not None else -1.0)
                    Ya[j, 2] = ((cont[t, a["recirc_idx"]]
                                 - bt[a["recirc_idx"]])
                                if a["recirc_idx"] is not None else 1.0)
                acc("a.sat", Ya[:, 0], Pa[:, 0])
                acc("a.press", Ya[:, 1], Pa[:, 1])
                acc("a.recirc", Ya[:, 2], Pa[:, 2])
            w_run = (disc[t, w["water_discrete_idx"]]
                     if 0 <= w["water_discrete_idx"] < disc.shape[1]
                     else 0.0)
            if w_run > 0.5:
                Xg = gf.astype(np.float64)[None, :][:, :Wgb.shape[0]]
                Pg = (Xg @ Wgb + bgb)[0]
                Yg = np.zeros(2)
                Yg[0] = ((cont[t, w["water_temp_idx"]]
                          - bt[w["water_temp_idx"]])
                         if w["water_temp_idx"] is not None else -1.0)
                Yg[1] = ((cont[t, w["water_pressure_idx"]]
                          - bt[w["water_pressure_idx"]])
                         if w["water_pressure_idx"] is not None else -1.0)
                acc("w.temp", Yg[0], Pg[0])
                acc("w.press", Yg[1], Pg[1])
        line = f"{pair} deleg={n_deleg}"
        for c in CHANNELS:
            r2 = (1 - ss_res[c] / ss_tot[c] if ss_tot[c] > 0
                  else (1.0 if ss_res[c] == 0 else float("-inf")))
            line += f" {c}={r2:.3f}(n={cnt[c]})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
