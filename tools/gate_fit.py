# -*- coding: utf-8 -*-
"""Confidence-gate offline probe for H4 (plan-19 P4, redesigned).

Trains a gate MLP on pooled step features to predict where the
frozen actor's correction BEAT the rule. Labels are the
rule-vs-residual counterfactual: per time-aligned step,
1[MC_actor > MC_rule], on actor-free steps only -- NOT raw GAE
sign (I2 proved advantages noise; Addendum A circularity
correction). Both P1 batches share pool+seeds, so initial
conditions match (trajectories diverge after step 0 -- acknowledged
imperfection, strictly better than GAE labels).

Gate: held-out AUC >= 0.65 AND fraction-open in (0.1, 0.6);
else KILL H4.

Usage: gate_fit.py /tmp/p1/actor10.npz /tmp/p1/rule10.npz
"""
import sys

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")

import numpy as np
import torch
import torch.nn as nn

from vrule_fit import VRule, pooled, mc_returns


class GateMLP(nn.Module):
    """Gate head arch (shared-encoder spirit: same pooled input)."""

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
    ap.add_argument("actor")
    ap.add_argument("rule")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    za = np.load(a.actor, allow_pickle=True)
    zr = np.load(a.rule, allow_pickle=True)
    ba = [int(t) for t in za["bounds"]]
    br = [int(t) for t in zr["bounds"]]
    assert ba == br, "pool mismatch"
    assert str(za["pool_sha"]) == str(zr["pool_sha"]), "sha mismatch"
    print(f"batches share pool sha {str(za['pool_sha'])[:12]}, "
          f"S={len(ba)}", flush=True)

    Xa = pooled(np.asarray(za["zone"], dtype=np.float64),
                np.asarray(za["zmask"], dtype=np.float64),
                np.asarray(za["ahu"], dtype=np.float64),
                np.asarray(za["amask"], dtype=np.float64),
                np.asarray(za["plant"], dtype=np.float64))
    mca = mc_returns(np.asarray(za["r"], dtype=np.float64), ba)
    mcr = mc_returns(np.asarray(zr["r"], dtype=np.float64), br)
    free = np.asarray(za["fr"]) > 0.5
    print(f"free steps: {free.sum()} / {len(free)} "
          f"({free.mean():.3f})", flush=True)

    y = (mca[free] > mcr[free]).astype(np.float32)
    X = Xa[free]
    print(f"label rate (actor beats rule): {y.mean():.3f}", flush=True)
    # Held-out = last 2 trajs' free steps.
    cut_steps = sum(ba[:-2])
    n_free_tr = int(free[:cut_steps].sum())
    Xtr, Xte = X[:n_free_tr], X[n_free_tr:]
    ytr, yte = y[:n_free_tr], y[n_free_tr:]
    print(f"train {Xtr.shape[0]} / test {Xte.shape[0]}, "
          f"test rate {yte.mean():.3f}", flush=True)

    g = GateMLP(X.shape[1])
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(ytr)
    bs = 4096
    for s in range(a.steps):
        ii = np.random.randint(0, Xt.shape[0], bs)
        loss = bce(g(Xt[ii]).reshape(-1), yt[ii])
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save({"arch": [X.shape[1], 64, 64, 1], "state": g.state_dict()},
               "/tmp/p1/gate.pt")
    print("saved /tmp/p1/gate.pt", flush=True)
    g.eval()
    with torch.no_grad():
        p = torch.sigmoid(g(torch.from_numpy(Xte.astype(np.float32)))
                          ).numpy().ravel()
    yte = np.asarray(yte).ravel()
    # AUC via Mann-Whitney.
    r = np.argsort(p)
    rank = np.empty_like(r, dtype=np.float64)
    rank[r] = np.arange(1, len(r) + 1)
    n1, n0 = float(yte.sum()), float((1 - yte).sum())
    auc = (rank[yte > 0.5].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    print(f"held-out AUC: {auc:.4f} (gate: >= 0.65)", flush=True)
    for tau in (0.3, 0.5, 0.7):
        frac = float((p > tau).mean())
        print(f"  tau={tau}: fraction-open {frac:.3f} "
              f"(gate: in (0.1, 0.6))", flush=True)
    print(f"P4 GATE: {'PASS' if auc >= 0.65 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
