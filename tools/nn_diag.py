# -*- coding: utf-8 -*-
"""Open-loop diagnostics: run the BC model on the TEACHER's dumped features
and compare predictions with the teacher's recorded actions per head.

Separates model error from closed-loop compounding: if open-loop is tight but
online diverges, iterate with DAgger rather than more epochs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.nn_model import ControllerNet, N_HIST  # noqa: E402
from tools.bc_train import RESULTS  # noqa: E402

CHECKPOINT = RESULTS / "nn_runs" / "bc_v0.pt"
SCENARIOS = [
    "building_train_1/atlanta_2", "building_train_1/denver_7",
    "building_train_1/fairbanks_1", "building_train_1/phoenix_5",
    "building_train_5/fairbanks_5", "building_train_8/singapore_7",
    "building_train_7/atlanta_8", "building_train_4/phoenix_8",
]


def main():
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    cfg = ckpt["args"]
    model = ControllerNet(ckpt["fz"], ckpt["fa"], ckpt["fg"],
                          d=cfg["d_model"], layers=cfg["layers"],
                          heads=cfg["heads"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    agg = {}
    for scen in SCENARIOS:
        run_dir = RESULTS / scen / "nn_data_dsat_gen31"
        z = np.load(run_dir / "tokens.npz")
        mg = json.load(open(run_dir / "mapping.json"))
        zone, ahu, glob = z["zone_feats"], z["ahu_feats"], z["glob_feats"]
        T = zone.shape[0]
        zm, am = zone.mean(axis=1), ahu.mean(axis=1)
        pool = np.concatenate([zm, am, glob], axis=1)
        hp = np.zeros((T, N_HIST, pool.shape[1]), dtype=np.float32)
        tt = np.arange(T)
        for k in range(N_HIST):
            s = np.maximum(tt - (N_HIST - k) * 6, 0)
            hp[:, k] = pool[s]
        B = 128
        out = {}
        off = {k: [] for k in ("zone", "sat", "press", "recirc", "run",
                               "water", "wrun")}
        tgt = {"damper": [], "reheat": [], "sat": [], "press": [],
               "recirc": [], "run": [], "water": [], "wrun": []}
        with torch.no_grad():
            for s0 in range(N_HIST, T, B):
                s1 = min(s0 + B, T)
                sel = np.arange(s0, s1)
                o = model(torch.from_numpy(zone[sel]),
                          torch.from_numpy(ahu[sel]),
                          torch.from_numpy(glob[sel]),
                          torch.from_numpy(hp[sel]))
                cont = z["cont"][sel].astype(np.float32)
                disc = z["disc"][sel].astype(np.float32)
                tgt["damper"].append(cont[:, [r["damp_act"] for r in mg["rooms"]]])
                tgt["reheat"].append(cont[:, [r["reheat_act"] for r in mg["rooms"]]])
                tgt["sat"].append(cont[:, [a["temp_idx"] for a in mg["ahus"]]])
                pi = [a["pressure_idx"] or -1 for a in mg["ahus"]]
                tgt["press"].append(cont[:, [i if i >= 0 else -1 for i in pi]])
                ri = [a["recirc_idx"] or -1 for a in mg["ahus"]]
                tgt["recirc"].append(cont[:, [i if i >= 0 else -1 for i in ri]])
                tgt["run"].append(disc[:, [a["run_disc_idx"] for a in mg["ahus"]]])
                tgt["water"].append(cont[:, mg["water"]["water_temp_idx"]])
                tgt["wrun"].append(disc[:, mg["water"]["water_discrete_idx"]])
                off["zone"].append(o["zone"].numpy())
                off["sat"].append(o["ahu_sat"].numpy())
                off["press"].append(o["ahu_pressure"].numpy())
                off["recirc"].append(o["ahu_recirc"].numpy())
                off["run"].append(o["ahu_run_logit"].numpy())
                off["water"].append(o["water"].numpy())
                off["wrun"].append(o["water_run_logit"].numpy())
        print(f"== {scen}  T={T}")
        rows = []
        for name in ("damper", "reheat", "sat", "press", "recirc", "water"):
            pred_map = {"damper": ("zone", 0), "reheat": ("zone", 1),
                        "sat": ("sat", None), "press": ("press", None),
                        "recirc": ("recirc", None), "water": ("water", None)}
            key, col = pred_map[name]
            pred = np.concatenate(off[key])
            if col is not None:
                pred = pred[..., col]
            true = np.concatenate(tgt[name])
            mask = true > -1.5
            if mask.sum() == 0:
                rows.append((name, np.nan, np.nan, np.nan))
                continue
            p, t = pred[mask], true[mask]
            mse = float(np.mean((p - t) ** 2))
            corr = float(np.corrcoef(p, t)[0, 1])
            rows.append((name, mse, corr, float(np.mean(np.abs(p - t)))))
        for name, mse, corr, mae in rows:
            print(f"   {name:7s} mse {mse:.4f}  mae {mae:.4f}  corr {corr:.3f}")
        for name, logit_key in (("run", "run"), ("wrun", "wrun")):
            logits = np.concatenate(off[logit_key])
            true = np.concatenate(tgt[name])
            mask = true > -0.5
            p = (logits[mask] > 0)
            t = (true[mask] > 0.5)
            acc = float((p == t).mean())
            flip = float((p[:-1] != p[1:]).mean())
            churn_t = float((t[:-1] != t[1:]).mean())
            print(f"   {name:7s} bit-acc {acc:.3f}  pred-flip {flip:.3f} "
                  f"(teacher {churn_t:.3f})")
        print(f"   teacher run-bits: on={float((np.concatenate(tgt['run']) > .5).mean()):.2f}, "
              f"wr on={float((np.concatenate(tgt['wrun']) > .5).mean()):.2f}")


if __name__ == "__main__":
    main()
