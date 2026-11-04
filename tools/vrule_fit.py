# -*- coding: utf-8 -*-
"""Frozen rule-value baseline for the H2 residual critic (plan-19 P3).

Fits a small MLP V_rule on pooled step features from a rule-only
batch (MC gamma-returns as targets), evaluated on held-out trajs.
Compares against (a) the shipped critic at init (uncalibrated --
the H1/H4 clip-crowding regime) and (b) a 100-step v-pretrained
single critic (status quo). P3 gate: split-critic value-MSE down
>= 2x vs single critic, else KILL H2.

Usage: vrule_fit.py /tmp/p1/rule10.npz [--steps N] [--seed S]
"""
import sys

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import torch
import torch.nn as nn

import skynet_ppo as ppo


def mc_returns(r, bounds, gamma=ppo.GAMMA):
    out, off = [], 0
    for T in bounds:
        sl = slice(off, off + int(T))
        out.append(ppo.compute_gae(r[sl], np.zeros(int(T)), gamma=gamma,
                                   lam=1.0)[1])
        off += int(T)
    return np.concatenate(out)


def pooled(zone, zmask, ahu, amask, plant):
    zm = zmask.astype(np.float64)[..., None]
    am = amask.astype(np.float64)[..., None]
    return np.concatenate([
        (zone * zm).sum(axis=1) / np.maximum(zm.sum(axis=1), 1.0),
        (ahu * am).sum(axis=1) / np.maximum(am.sum(axis=1), 1.0),
        plant,
    ], axis=1).astype(np.float64)


class VRule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(d, 64), nn.Tanh(),
                              nn.Linear(64, 64), nn.Tanh(),
                              nn.Linear(64, 1))

    def forward(self, x):
        return self.m(x).reshape(-1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="/tmp/p1/rule10.pt.behavior")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    z = np.load(a.batch, allow_pickle=True)
    bounds = [int(t) for t in z["bounds"]]
    n = sum(bounds)
    print(f"batch: S={len(bounds)} flat={n} "
          f"sha={str(z['pool_sha'])[:12]}", flush=True)
    X = pooled(np.asarray(z["zone"], dtype=np.float64),
               np.asarray(z["zmask"], dtype=np.float64),
               np.asarray(z["ahu"], dtype=np.float64),
               np.asarray(z["amask"], dtype=np.float64),
               np.asarray(z["plant"], dtype=np.float64))
    mc = mc_returns(np.asarray(z["r"], dtype=np.float64), bounds)
    # Train on trajs 0..S-3, test on last 2 trajs.
    cut = sum(bounds[:-2])
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = mc[:cut], mc[cut:]
    print(f"train {Xtr.shape[0]} / test {Xte.shape[0]} steps, "
          f"feat {Xtr.shape[1]}", flush=True)

    # Baselines: constant mean + shipped critic at init.
    const = float(ytr.mean())
    mse_const = float(((yte - const) ** 2).mean())
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    import skynet_surgery as surg
    if "zone.0.adapter.weight" in ck.get("model", {}):
        net, _ = surg.expand_net_from_ckpt(ck)
    else:
        from skynet.policy import load as _load
        net0, _, _, _ = _load("/tmp/opencode/sb-rl/skynet.pt")
        net, _ = surg.expand_net(net0)
        net.load_state_dict(ck["model"])
    net.eval()
    t = torch.from_numpy
    with torch.no_grad():
        vte = net(t(z["zone"][cut:].astype(np.float32)),
                  t(z["zmask"][cut:].astype(np.float32)),
                  t(z["ahu"][cut:].astype(np.float32)),
                  t(z["amask"][cut:].astype(np.float32)),
                  t(z["plant"][cut:].astype(np.float32)))[1].numpy()
    mse_init = float(((yte - vte) ** 2).mean())
    print(f"baseline const-mean test MSE: {mse_const:.4f}", flush=True)
    print(f"baseline shipped-critic-init test MSE: {mse_init:.4f}",
          flush=True)

    # V_rule fit.
    vr = VRule(Xtr.shape[1])
    opt = torch.optim.Adam(vr.parameters(), lr=1e-3)
    Xt = t(Xtr.astype(np.float32))
    yt = t(ytr.astype(np.float32))
    bs = 4096
    for s in range(a.steps):
        ii = np.random.randint(0, Xt.shape[0], bs)
        loss = ((vr(Xt[ii]) - yt[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"step {s + 1}: train MSE {float(loss):.4f}", flush=True)
    vr.eval()
    with torch.no_grad():
        pred = vr(t(Xte.astype(np.float32))).numpy()
    mse_rule = float(((yte - pred) ** 2).mean())
    print(f"V_rule test MSE: {mse_rule:.4f}", flush=True)
    print(f"ratio vs init-critic: {mse_rule / max(mse_init, 1e-12):.4f} "
          f"(gate: <= 0.5)", flush=True)
    print(f"ratio vs const-mean: {mse_rule / max(mse_const, 1e-12):.4f}",
          flush=True)
    torch.save(vr.state_dict(), "/tmp/p1/vrule.pt")
    print("saved /tmp/p1/vrule.pt", flush=True)


if __name__ == "__main__":
    main()
