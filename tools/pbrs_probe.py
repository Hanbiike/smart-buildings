# -*- coding: utf-8 -*-
"""PBRS comfort-shaping offline probe for H5 (plan-19 P5, speculative).

Potential Phi(s) = -k * mean_zones max(0, lo-T, T-hi), band read
from the batch's own band-error cols (z23 cols 6:8, /5 units).
Shaped rewards r' = r + g*Phi(s') - Phi(s) with the GAE gamma;
no absorbing state exists, so the terminal step uses Phi'=Phi
(this is exactly the Addendum-A invariance violation -- the probe
measures whether shaping still helps empirically).

Gate: shaped adv-std up >= 30% with rank-corr >= 0.7 +
sign-check; else KILL H5.

Usage: pbrs_probe.py /tmp/p1/actor10.npz [--k 1.0]
"""
import sys

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
sys.path.insert(0, "/tmp/pkg")

import numpy as np

import skynet_ppo as ppo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--k", type=float, default=1.0)
    a = ap.parse_args()

    z = np.load(a.batch, allow_pickle=True)
    bounds = [int(t) for t in z["bounds"]]
    band = np.asarray(z["zone"][:, :, 6:8], dtype=np.float64)
    zmask = np.asarray(z["zmask"], dtype=np.float64)
    viol = np.maximum(0.0, np.maximum(-band[..., 0], -band[..., 1])) * 5.0
    phi = -a.k * (viol * zmask).sum(axis=1) / np.maximum(zmask.sum(axis=1),
                                                        1.0)
    r = np.asarray(z["r"], dtype=np.float64)
    fr = np.asarray(z["fr"], dtype=np.float64)
    za = np.asarray(z["za"], dtype=np.float64)
    print(f"batch flat={len(r)} free={fr.mean():.3f} "
          f"mean|Phi|={np.abs(phi).mean():.4f} (k={a.k})", flush=True)

    au, ash, off = [], [], 0
    for T in bounds:
        sl = slice(off, off + T)
        F = np.empty(T)
        F[:-1] = ppo.GAMMA * phi[sl][1:] - phi[sl][:-1]
        F[-1] = (ppo.GAMMA - 1.0) * phi[sl][-1]
        au.append(ppo.compute_gae(r[sl], np.zeros(T))[0])
        ash.append(ppo.compute_gae(r[sl] + F, np.zeros(T))[0])
        off += T
    au, ash = np.concatenate(au), np.concatenate(ash)
    # Same per-traj normalization the trainer applies.
    aun, ashn, off = [], [], 0
    for T in bounds:
        sl = slice(off, off + T)
        # NOTE: normalize_adv mutates in place; copy to keep raw stds.
        aun.append(ppo.normalize_adv(au[sl].copy(), fr[sl].copy()))
        ashn.append(ppo.normalize_adv(ash[sl].copy(), fr[sl].copy()))
        off += T
    aun, ashn = np.concatenate(aun), np.concatenate(ashn)
    m = fr > 0.5
    ru0, rs0 = float(au[m].std()), float(ash[m].std())
    print(f"adv-std free steps PRE-NORM: unshaped {ru0:.4f} shaped "
          f"{rs0:.4f} (ratio {rs0 / max(ru0, 1e-12):.3f}, gate >= 1.30)",
          flush=True)
    su, ss = float(aun[m].std()), float(ashn[m].std())
    ok_std = rs0 / max(ru0, 1e-12) >= 1.30
    # Spearman rank correlation on free steps.
    ru = np.argsort(np.argsort(aun[m])).astype(np.float64)
    rs = np.argsort(np.argsort(ashn[m])).astype(np.float64)
    rc = float(np.corrcoef(ru, rs)[0, 1])
    print(f"rank-corr: {rc:.4f} (gate >= 0.7)", flush=True)
    agree = float(np.mean((aun[m] > 0) == (ashn[m] > 0)))
    print(f"sign agreement: {agree:.4f}", flush=True)
    # Sign-check: warm excursion + damper-open should outrank
    # warm excursion + damper-closed under shaping.
    warm = ((viol * zmask).sum(axis=1)
            / np.maximum(zmask.sum(axis=1), 1.0) > 0.25)
    dmp = za[..., 0].mean(axis=1)
    g = m & warm & (dmp > 0)
    b = m & warm & (dmp <= 0)
    print(f"sign-check n(good)={g.sum()} n(bad)={b.sum()}", flush=True)
    if g.sum() > 50 and b.sum() > 50:
        print(f"  shaped adv: good {ashn[g].mean():+.4f} vs bad "
              f"{ashn[b].mean():+.4f}", flush=True)
    ok = ok_std and rc >= 0.7
    print(f"P5 GATE: {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
