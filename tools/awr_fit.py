# -*- coding: utf-8 -*-
"""BCOOLER-style offline iteration, minimal form: advantage-weighted
regression on a frozen-actor batch (plan-19 P10.1).

V(s) MLP on pooled feats with MC gamma=1.0 targets (BCOOLER: TD
rejected offline; undiscounted like our LB), then advantage-weighted
BC fine-tune of the surgery-expanded net on free steps only:
  w = exp(A / beta), clipped at 20; loss = -mean(w * norm_logprob).
Frozen by construction (deploy-consistency + "no new effector"):
surgery adapters (exact-zero; the P1 batch predates the 4 newest
cols, zero-padded here), the target-head row, and all logstds
(deploy is deterministic mean, std unused).

Offline metrics are SANITY ONLY (T3: offline wins evaporate
online); the P10 verdict comes from the P10.2 full-episode
2-climate screen, never from this script.

Usage: awr_fit.py /tmp/p1/actor10.npz [--steps N] [--seed S]
  [--out /tmp/p10/awr.pt] [--beta B] (0 = auto std(A)) [--lr LR]
"""
import sys

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import torch

import skynet_ppo as ppo
import vrule_fit as vf


def masked_lp2(net, zone, zmask, ahu, amask, plant, za, aa, pa):
    """Per-step normalized logprob WITHOUT the target head (P10: no new
    effector). Mirrors ppo.masked_lp at the pre-P7 :2 zone slice."""
    import torch.distributions as _D
    (mz, ma, mp), _v = net(zone, zmask, ahu, amask, plant)
    stds = [torch.clamp(s, min=ppo.MIN_LOGSTD) for s in net.stds()]
    dz = _D.Normal(mz[..., :2],
                   stds[0][:2].exp().expand_as(mz[..., :2]))
    lpz = (dz.log_prob(za[..., :2]).sum(-1) * zmask).sum(-1)
    n = 2.0 * zmask.sum(-1)
    da = _D.Normal(ma[..., :3],
                   stds[1][:3].exp().expand_as(ma[..., :3]))
    lpa = (da.log_prob(aa[..., :3]).sum(-1) * amask).sum(-1)
    n = n + 3.0 * amask.sum(-1)
    dp = _D.Normal(mp[..., :2],
                   stds[2][:2].exp().expand_as(mp[..., :2]))
    lpp = dp.log_prob(pa[..., :2]).sum(-1)
    n = n + 2.0
    return (lpz + lpa + lpp) / n.clamp_min(1.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/p10/awr.pt")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    z = np.load(a.batch, allow_pickle=True)
    bounds = [int(t) for t in z["bounds"]]
    n = sum(bounds)
    print(f"batch: S={len(bounds)} flat={n} "
          f"sha={str(z['pool_sha'])[:12]}", flush=True)
    # Zero-pad the pre-surgery 23-col zone feats to 27 (cols 0-22
    # coincide: base 21 + plan-18 pair; the 4 newest cols are absent).
    z23 = np.asarray(z["zone"], dtype=np.float32)
    assert z23.shape[2] == 23, z23.shape
    zone = np.zeros(z23.shape[:2] + (27,), dtype=np.float32)
    zone[..., :23] = z23
    zmask = np.asarray(z["zmask"], dtype=np.float32)
    ahu = np.asarray(z["ahu"], dtype=np.float32)
    amask = np.asarray(z["amask"], dtype=np.float32)
    plant = np.asarray(z["plant"], dtype=np.float32)
    za = np.asarray(z["za"], dtype=np.float32)
    aa = np.asarray(z["aa"], dtype=np.float32)
    pa = np.asarray(z["pa"], dtype=np.float32)
    fr = np.asarray(z["fr"], dtype=np.float32)
    r = np.asarray(z["r"], dtype=np.float64)

    # V on MC gamma=1.0 (BCOOLER-style undiscounted).
    X = vf.pooled(zone.astype(np.float64), zmask.astype(np.float64),
                  ahu.astype(np.float64), amask.astype(np.float64),
                  plant.astype(np.float64))
    mc = vf.mc_returns(r, bounds, gamma=1.0)
    cut = sum(bounds[:-2])
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = mc[:cut], mc[cut:]
    const = float(ytr.mean())
    print(f"V train {Xtr.shape[0]} / test {Xte.shape[0]}, "
          f"const-mean test MSE {float(((yte - const) ** 2).mean()):.4f}",
          flush=True)
    vr = vf.VRule(Xtr.shape[1])
    opt = torch.optim.Adam(vr.parameters(), lr=1e-3)
    t = torch.from_numpy
    Xt, yt = t(Xtr.astype(np.float32)), t(ytr.astype(np.float32))
    bs = 4096
    for s in range(a.steps):
        ii = np.random.randint(0, Xt.shape[0], bs)
        loss = ((vr(Xt[ii]) - yt[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    vr.eval()
    with torch.no_grad():
        pred = vr(t(Xte.astype(np.float32))).numpy()
        vtr = vr(t(Xtr.astype(np.float32))).numpy()
    print(f"V_rule(g1) test MSE: {float(((yte - pred) ** 2).mean()):.4f}",
          flush=True)
    adv = (ytr - vtr).astype(np.float64)
    beta = a.beta if a.beta > 0 else float(adv.std())
    beta = max(beta, 1e-6)
    w = np.exp(np.clip(adv / beta, -20.0, np.log(20.0)))
    print(f"AWR: beta={beta:.4f} w_mean={w.mean():.3f} "
          f"w_max={w.max():.3f}", flush=True)

    # Expanded net, frozen adapters/target/logstds.
    import skynet_surgery as surg
    from skynet.policy import load as _load
    net0, _, _, _ = _load("/tmp/opencode/sb-rl/skynet.pt")
    net, _ = surg.expand_net(net0)
    net.train()
    frozen = []
    for n_, p in net.named_parameters():
        if (".adapter." in n_ or "log_std" in n_
                or n_ in ("zone_mu.weight", "zone_mu.bias")):
            frozen.append(n_)
    # zone_mu rows 0-1 stay live; only row 2 (target) is frozen --
    # handled by zeroing its grad below, since rows share a tensor.
    live = [p for n_, p in net.named_parameters()
            if n_ not in frozen or n_.startswith("zone_mu")]
    for n_, p in net.named_parameters():
        if n_ in frozen and not n_.startswith("zone_mu"):
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in live if p.requires_grad], lr=a.lr)

    # Free train steps only (gated steps carry no residual signal).
    free = (fr[:cut] > 0.5)
    F = dict(zone=t(zone[:cut]), zmask=t(zmask[:cut]), ahu=t(ahu[:cut]),
             amask=t(amask[:cut]), plant=t(plant[:cut]), za=t(za[:cut]),
             aa=t(aa[:cut]), pa=t(pa[:cut]))
    W = t(w.astype(np.float32))
    idx = np.nonzero(free)[0]
    print(f"policy train steps: {len(idx)} free / {cut}", flush=True)
    with torch.no_grad():
        l0 = float(-(W[idx] * masked_lp2(
            net, F["zone"][idx], F["zmask"][idx], F["ahu"][idx],
            F["amask"][idx], F["plant"][idx], F["za"][idx],
            F["aa"][idx], F["pa"][idx])).mean())
    print(f"weighted NLL before: {l0:.4f}", flush=True)
    bsp = 1024
    for s in range(a.steps):
        ii = idx[np.random.randint(0, len(idx), bsp)]
        lp = masked_lp2(net, F["zone"][ii], F["zmask"][ii],
                        F["ahu"][ii], F["amask"][ii], F["plant"][ii],
                        F["za"][ii], F["aa"][ii], F["pa"][ii])
        loss = -(W[ii] * lp).mean()
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            # Keep the target-head row exactly zero (no new effector).
            if net.zone_mu.weight.grad is not None:
                net.zone_mu.weight.grad[2].zero_()
                net.zone_mu.bias.grad[2].zero_()
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"step {s + 1}: {float(loss):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        l1 = float(-(W[idx] * masked_lp2(
            net, F["zone"][idx], F["zmask"][idx], F["ahu"][idx],
            F["amask"][idx], F["plant"][idx], F["za"][idx],
            F["aa"][idx], F["pa"][idx])).mean())
    print(f"weighted NLL after: {l1:.4f}", flush=True)
    # Frozen-part audit.
    sd = net.state_dict()
    ad = [k for k in sd if ".adapter." in k]
    assert all(bool((sd[k] == 0.0).all()) for k in ad), "adapter moved"
    assert bool((sd["zone_mu.weight"][2] == 0.0).all()), "target row moved"
    assert bool((sd["zone_mu.bias"][2] == 0.0).all()), "target bias moved"
    print(f"frozen audit PASS ({len(ad)} adapter tensors + target row "
          f"exact-zero)", flush=True)
    ppo.save_ckpt(net, opt, a.out, extra={"kind": "awr", "beta": beta})
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()