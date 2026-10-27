# -*- coding: utf-8 -*-
"""Behavior-clone a windowed controller from dumped gen-3.1 trajectories.

Architecture v0 (Zipformer-lite, CPU-friendly, no recurrence):

  - history: 12 pooled tokens sampled at stride 6 over the last 72 steps
    (mean of room features + mean of AHU features + global vector), projected
    by a shared linear with learned positional embeddings;
  - current step: one token per room (shared linear), one per AHU, one global;
  - encoder: attention blocks over [history; rooms; ahus; global] tokens —
    fully batched over the sample axis (all windows in a batch come from one
    scenario, so token counts match);
  - heads: per-room (damper, reheat) via shared MLP, per-AHU (SAT, pressure,
    recirculation, run bit), global water (temp, pressure, run bit). A
    context vector (mean history + mean current outputs) feeds every head, so
    zone decisions are permutation-equivariant and building sizes vary freely.

Loss: masked MSE on continuous commands + BCE on run bits against the
teacher's recorded actions. Validation: a held-out building (cross-building
BC loss) plus the in-training loss curve.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.nn_model import (  # noqa: E402
    ControllerNet,
    N_HIST,
    STRIDE,
    WINDOW,
)

RESULTS = Path(PROJECT_ROOT) / "smart_buildings" / "results"
RUNS_DIR = RESULTS / "nn_runs"


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="dsat_gen31")
    ap.add_argument("--extra-teachers", default="")
    ap.add_argument("--extra-bags", default="",
                    help="DAgger bag names (nn_data_<name>) with labels "
                         "overridden by dagger_labels.npz")
    ap.add_argument("--hold-building", default="building_train_5")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="bc_v0")
    return ap.parse_args()


def collect_runs(teacher: str, extra_teachers, extra_bags, hold_building):
    """Tokenized scenario runs split by hold-building.

    `extra_teachers` are plain bag names (`nn_data_<name>`) whose recorded
    actions are imitation targets in their own right. `extra_bags` are DAgger
    bags: features come from the executed (NN-driven) obs, but the labels are
    overridden by dagger_labels.npz (teacher actions on those states).
    """
    train, val = [], []
    for building_dir in sorted(RESULTS.glob("building_train_*")):
        if not building_dir.is_dir():
            continue
        for weather_dir in sorted(building_dir.iterdir()):
            run_dir = weather_dir / f"nn_data_{teacher}"
            if not (run_dir / "tokens.npz").exists():
                continue
            mapping = json.load(open(run_dir / "mapping.json"))
            z = np.load(run_dir / "tokens.npz")
            bags = [{"tokens": dict(z.items()), "mapping": mapping}]
            mixed = ([(n, False) for n in extra_teachers]
                     + [(n, True) for n in extra_bags])
            for name, is_dagger in mixed:
                extra_dir = weather_dir / f"nn_data_{name}"
                if not (extra_dir / "tokens.npz").exists():
                    continue
                tokens = dict(np.load(extra_dir / "tokens.npz"))
                labels_path = extra_dir / "dagger_labels.npz"
                if is_dagger and labels_path.exists():
                    lab = np.load(labels_path)["labels"]
                    n_cont = tokens["cont"].shape[1]
                    tokens["cont"] = np.ascontiguousarray(
                        lab[:, :n_cont], dtype=np.float32)
                    tokens["disc"] = np.ascontiguousarray(
                        lab[:, n_cont:], dtype=np.int64)
                bags.append({"tokens": tokens, "mapping": mapping})
            dest = val if building_dir.name == hold_building else train
            dest.append({"scen": f"{building_dir.name}/{weather_dir.name}",
                         "bags": bags})
    return train, val


def build_hist_window(zone, ahu, glob, t, n_hist=N_HIST, stride=STRIDE):
    """Pooled history features before t (oldest first), one row per slot."""
    feats = np.zeros((n_hist, zone.shape[2] + ahu.shape[2] + glob.shape[1]),
                     dtype=np.float32)
    for k in range(n_hist):
        s = t - (n_hist - k) * stride
        if s < 0:
            s = 0
        feats[k] = np.concatenate([zone[s].mean(axis=0),
                                   ahu[s].mean(axis=0), glob[s]])
    return feats


def hist_cache(run):
    """Precompute pooled history for every t in [0, T). [T, H, Fh]."""
    zone, ahu, glob = run["zone"], run["ahu"], run["glob"]
    T = zone.shape[0]
    pooled = np.zeros((T, N_HIST, zone.shape[2] + ahu.shape[2] + glob.shape[1]),
                      dtype=np.float32)
    zm = zone.mean(axis=1)          # [T, Fz]
    am = ahu.mean(axis=1)           # [T, Fa]
    pool = np.concatenate([zm, am, glob], axis=1)   # [T, Fh]
    for k in range(N_HIST):
        s = np.maximum(np.arange(T) - (N_HIST - k) * STRIDE, 0)
        pooled[:, k] = pool[s]
    return pooled


def head_loss(model_out, mg, cont, disc, dst):
    """Masked BC loss for one scenario batch (cont/disc already batched)."""
    cont_d = torch.from_numpy(cont).to(dst)
    disc_d = torch.from_numpy(disc.astype(np.float32)).to(dst)
    damper = cont_d[:, [r["damp_act"] for r in mg["rooms"]]]
    reheat = cont_d[:, [r["reheat_act"] for r in mg["rooms"]]]
    loss = F.mse_loss(model_out["zone"], torch.stack([damper, reheat], -1))
    sat = cont_d[:, [a["temp_idx"] for a in mg["ahus"]]]
    loss = loss + F.mse_loss(model_out["ahu_sat"], sat)
    press_idx = [a["pressure_idx"] if a["pressure_idx"] is not None else -1
                 for a in mg["ahus"]]
    pm = [i for i, v in enumerate(press_idx) if v >= 0]
    if pm:
        loss = loss + F.mse_loss(model_out["ahu_pressure"][:, pm],
                                 cont_d[:, [press_idx[i] for i in pm]])
    recirc_idx = [a["recirc_idx"] if a["recirc_idx"] is not None else -1
                  for a in mg["ahus"]]
    rm = [i for i, v in enumerate(recirc_idx) if v >= 0]
    if rm:
        loss = loss + 0.5 * F.mse_loss(
            model_out["ahu_recirc"][:, rm],
            cont_d[:, [recirc_idx[i] for i in rm]])
    run_bits = disc_d[:, [a["run_disc_idx"] for a in mg["ahus"]]]
    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
        model_out["ahu_run_logit"], run_bits)
    w = mg["water"]
    if w["water_temp_idx"] is not None and w["water_temp_idx"] >= 0:
        loss = loss + 0.5 * F.mse_loss(
            model_out["water"], cont_d[:, w["water_temp_idx"]])
    if w["water_pressure_idx"] is not None and w["water_pressure_idx"] >= 0:
        loss = loss + 0.5 * F.mse_loss(
            model_out["water_pressure"], cont_d[:, w["water_pressure_idx"]])
    if w["water_discrete_idx"] is not None and w["water_discrete_idx"] >= 0:
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            model_out["water_run_logit"],
            disc_d[:, w["water_discrete_idx"]])
    return loss


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("mps" if args.device == "auto" and torch.backends.mps.is_available()
              else args.device if args.device != "auto" else "cpu")

    train_list, val_list = collect_runs(
        args.teacher, [t for t in args.extra_teachers.split(",") if t],
        [b for b in args.extra_bags.split(",") if b],
        args.hold_building)
    print(f"train scenarios {len(train_list)}, val scenarios {len(val_list)}",
          flush=True)

    def unpack(slist):
        rows = []
        for s in slist:
            for bag in s["bags"]:
                z = bag["tokens"]
                rows.append({"scen": s["scen"],
                             "zone": z["zone_feats"], "ahu": z["ahu_feats"],
                             "glob": z["glob_feats"], "cont": z["cont"],
                             "disc": z["disc"], "rewards": z["rewards"],
                             "mapping": bag["mapping"]})
        return rows

    train_runs, val_runs = unpack(train_list), unpack(val_list)
    fz = train_runs[0]["zone"].shape[2]
    fa = train_runs[0]["ahu"].shape[2]
    fg = train_runs[0]["glob"].shape[1]
    print(f"features: fz={fz} fa={fa} fg={fg}; scenarios with Z/A: "
          f"{[(r['zone'].shape[1], r['ahu'].shape[1]) for r in train_runs[:3]]}",
          flush=True)

    for r in train_runs + val_runs:
        r["hist"] = hist_cache(r)

    model = ControllerNet(fz, fa, fg, args.d_model, args.layers, args.heads)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def n_batches(T):
        return max(1, (T - WINDOW) // args.batch)

    total_steps = args.epochs * sum(n_batches(r["zone"].shape[0])
                                    for r in train_runs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for run in train_runs:
            Z, A = run["zone"].shape[1], run["ahu"].shape[1]
            T = run["zone"].shape[0]
            starts = WINDOW + np.random.permutation(
                (T - WINDOW) // args.batch) * args.batch
            for s0 in starts:
                s1 = s0 + min(args.batch, T - s0)
                sel = np.arange(s0, s1)
                zf = torch.from_numpy(run["zone"][sel]).to(device)
                af = torch.from_numpy(run["ahu"][sel]).to(device)
                gf = torch.from_numpy(run["glob"][sel]).to(device)
                hp = torch.from_numpy(run["hist"][sel]).to(device)
                out = model(zf, af, gf, hp)
                loss = head_loss(out, run["mapping"], run["cont"][sel],
                                 run["disc"][sel], device)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                ep_loss += float(loss.detach()) * len(sel)
                ep_n += len(sel)
                step += 1
                if step % 200 == 0:
                    print(f"  ep {epoch} step {step}: train "
                          f"{ep_loss / max(ep_n, 1):.5f}", flush=True)
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for run in val_runs:
                T = run["zone"].shape[0]
                for s0 in range(WINDOW, T, args.batch):
                    sel = np.arange(s0, min(s0 + args.batch, T))
                    zf = torch.from_numpy(run["zone"][sel]).to(device)
                    af = torch.from_numpy(run["ahu"][sel]).to(device)
                    gf = torch.from_numpy(run["glob"][sel]).to(device)
                    hp = torch.from_numpy(run["hist"][sel]).to(device)
                    out = model(zf, af, gf, hp)
                    L = float(head_loss(out, run["mapping"], run["cont"][sel],
                                        run["disc"][sel], device))
                    vl += L * len(sel)
                    vn += len(sel)
        print(f"epoch {epoch}: train {ep_loss / max(ep_n, 1):.5f}  "
              f"val {vl / max(vn, 1):.5f}", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{args.out}.pt"
    torch.save({"model": model.state_dict(),
                "args": vars(args), "fz": fz, "fa": fa, "fg": fg}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
