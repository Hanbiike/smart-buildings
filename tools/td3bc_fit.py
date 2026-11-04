# -*- coding: utf-8 -*-
"""Constrained TD3 small probe (plan-19 P10): TD3+BC reading of
"CQL-style" on the actor10 batch.

Twin pooled Q(s,a) MLPs (pooled feats + pooled continuous actions),
offline TD (gamma 0.995, target smoothing, tau 0.005), terminal
bootstrap 0.0 (full-episode trajs); actor = surgery-expanded FlatNets
fine-tune (adapters/target-head/logstds frozen, as in AWR) with
  loss = -alpha * Q1(s,pi(s)) / |Q1|.detach + BC_MSE   (alpha = 2.5)
on free steps only, actor updated every 2nd Q step. Simplifications
vs full TD3+BC, logged: pooled (not per-entity) Q; no target actor
(the live actor feeds the TD target with smoothing noise).

Offline metrics are SANITY ONLY (T3); a P10 screen runs only if the
actor moves (|dmu| above noise). Verdict per the P10 gate.

Usage: td3bc_fit.py /tmp/p1/actor10.npz [--steps N] [--seed S]
  [--out /tmp/p10/td3bc.pt]
"""
import sys

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import torch
import torch.nn as nn

import skynet_ppo as ppo
import vrule_fit as vf

GAMMA = 0.995
TAU = 0.005
ALPHA = 2.5
TGT_NOISE = 0.2
TGT_CLIP = 0.5


class QNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(d, 64), nn.Tanh(),
                              nn.Linear(64, 64), nn.Tanh(),
                              nn.Linear(64, 1))

    def forward(self, x):
        return self.m(x).reshape(-1)


def pooled_a(za, zmask, aa, amask, pa):
    """[N,7] masked-mean continuous actions (zone :2, ahu :3, plant :2)."""
    zm = zmask[..., None]
    am = amask[..., None]
    z = (za[..., :2] * zm).sum(1) / np.maximum(zm.sum(1), 1.0)
    a = (aa[..., :3] * am).sum(1) / np.maximum(am.sum(1), 1.0)
    p = pa[..., :2]
    return np.concatenate([z, a, p], axis=1).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/p10/td3bc.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)
    t = torch.from_numpy

    z = np.load(a.batch, allow_pickle=True)
    bounds = [int(t_) for t_ in z["bounds"]]
    print(f"batch: S={len(bounds)} flat={sum(bounds)} "
          f"sha={str(z['pool_sha'])[:12]}", flush=True)
    z23 = np.asarray(z["zone"], dtype=np.float32)
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
    r = np.asarray(z["r"], dtype=np.float32)

    X = vf.pooled(zone.astype(np.float64), zmask.astype(np.float64),
                  ahu.astype(np.float64), amask.astype(np.float64),
                  plant.astype(np.float64)).astype(np.float32)
    A = pooled_a(za, zmask, aa, amask, pa)
    cut = sum(bounds[:-2])
    # Next-step index inside each traj (-1 = terminal).
    nxt = np.full(cut, -1)
    off = 0
    for T in bounds[:-2]:
        nxt[off:off + T - 1] = np.arange(off + 1, off + T)
        off += T

    q1, q2 = QNet(X.shape[1] + 7), QNet(X.shape[1] + 7)
    qt1, qt2 = QNet(X.shape[1] + 7), QNet(X.shape[1] + 7)
    qt1.load_state_dict(q1.state_dict())
    qt2.load_state_dict(q2.state_dict())
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                          lr=3e-4)

    import skynet_surgery as surg
    from skynet.policy import load as _load
    net0, _, _, _ = _load("/tmp/opencode/sb-rl/skynet.pt")
    net, _ = surg.expand_net(net0)
    net.train()
    for n_, p in net.named_parameters():
        if ".adapter." in n_ or "log_std" in n_:
            p.requires_grad_(False)
    live = [p for n_, p in net.named_parameters()
            if p.requires_grad or n_.startswith("zone_mu")]
    oa = torch.optim.Adam([p for p in live if p.requires_grad], lr=3e-4)

    Xt, At, rt = t(X[:cut]), t(A[:cut]), t(r[:cut])
    F = dict(zone=t(zone[:cut]), zmask=t(zmask[:cut]), ahu=t(ahu[:cut]),
             amask=t(amask[:cut]), plant=t(plant[:cut]))
    Zt, AAt, Pt = t(za[:cut]), t(aa[:cut]), t(pa[:cut])
    free = np.nonzero(fr[:cut] > 0.5)[0]
    al = np.nonzero(np.ones(cut, bool))[0]
    print(f"Q train steps: {cut} (all); actor train steps: "
          f"{len(free)} free", flush=True)
    bs = 1024
    for s in range(a.steps):
        ii = al[np.random.randint(0, len(al), bs)]
        # TD target with smoothing noise on the live actor's s' action.
        with torch.no_grad():
            nx = nxt[ii]
            ok = nx >= 0
            qn = torch.zeros(bs)
            if ok.any():
                (mz, ma, mp), _ = net(
                    F["zone"][nx[ok]], F["zmask"][nx[ok]],
                    F["ahu"][nx[ok]], F["amask"][nx[ok]],
                    F["plant"][nx[ok]])
                zm = F["zmask"][nx[ok]][..., None]
                am = F["amask"][nx[ok]][..., None]
                an = torch.cat([
                    (mz[..., :2] * zm).sum(1)
                    / zm.sum(1).clamp_min(1.0),
                    (ma[..., :3] * am).sum(1)
                    / am.sum(1).clamp_min(1.0),
                    mp[..., :2]], dim=1)
                an = an + (torch.randn_like(an) * TGT_NOISE
                           ).clamp(-TGT_CLIP, TGT_CLIP)
                xn = torch.cat([Xt[nx[ok]], an], dim=1)
                qn[ok] = torch.min(qt1(xn), qt2(xn))
            y = rt[ii] + GAMMA * qn
        q = torch.cat([Xt[ii], At[ii]], dim=1)
        loss_q = ((q1(q) - y) ** 2).mean() + ((q2(q) - y) ** 2).mean()
        oq.zero_grad()
        loss_q.backward()
        oq.step()
        with torch.no_grad():
            for q_, qt_ in ((q1, qt1), (q2, qt2)):
                for p, pt in zip(q_.parameters(), qt_.parameters()):
                    pt.mul_(1.0 - TAU).add_(p, alpha=TAU)
        # Delayed actor step (every 2nd): -alpha*Q/|Q| + BC.
        if s % 2 == 1:
            jj = free[np.random.randint(0, len(free), bs)]
            (mz, ma, mp), _ = net(
                F["zone"][jj], F["zmask"][jj], F["ahu"][jj],
                F["amask"][jj], F["plant"][jj])
            zm = F["zmask"][jj][..., None]
            am = F["amask"][jj][..., None]
            ap = torch.cat([
                (mz[..., :2] * zm).sum(1) / zm.sum(1).clamp_min(1.0),
                (ma[..., :3] * am).sum(1) / am.sum(1).clamp_min(1.0),
                mp[..., :2]], dim=1)
            qv = q1(torch.cat([Xt[jj], ap], dim=1))
            lam = ALPHA / qv.abs().mean().clamp_min(1.0).detach()
            bc = (((mz[..., :2] - Zt[jj][..., :2]) ** 2 * zm).sum()
                  / zm.sum().clamp_min(1.0)
                  + ((ma[..., :3] - AAt[jj][..., :3]) ** 2 * am).sum()
                  / am.sum().clamp_min(1.0)
                  + ((mp[..., :2] - Pt[jj][..., :2]) ** 2).mean())
            loss_a = -lam * qv.mean() + bc
            oa.zero_grad()
            loss_a.backward()
            with torch.no_grad():
                if net.zone_mu.weight.grad is not None:
                    net.zone_mu.weight.grad[2].zero_()
                    net.zone_mu.bias.grad[2].zero_()
            oa.step()
        if (s + 1) % 1000 == 0:
            with torch.no_grad():
                qs = float(q1(torch.cat([Xt[:4096], At[:4096]],
                                        dim=1)).mean())
            print(f"step {s + 1}: td {float(loss_q):.4f} "
                  f"qmean {qs:.3f}", flush=True)
    net.eval()
    sd = net.state_dict()
    ad = [k for k in sd if ".adapter." in k]
    assert all(bool((sd[k] == 0.0).all()) for k in ad), "adapter moved"
    assert bool((sd["zone_mu.weight"][2] == 0.0).all()), "target row moved"
    print("frozen audit PASS", flush=True)
    # Movement audit vs init.
    from skynet.policy import load as _load2
    net00, _, _, _ = _load2("/tmp/opencode/sb-rl/skynet.pt")
    si = surg.expand_net(net00)[0].state_dict()
    for k in ("zone_mu.weight", "ahu_mu.weight", "plant_mu.weight"):
        d = (sd[k][:2] if k == "zone_mu.weight" else sd[k])
        d0 = (si[k][:2] if k == "zone_mu.weight" else si[k])
        dd = (d - d0).abs()
        print(f"{k}: max|d|={float(dd.max()):.2e} "
              f"mean|d|={float(dd.mean()):.2e}", flush=True)
    ppo.save_ckpt(net, oa, a.out, extra={"kind": "td3bc"})
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
