"""PPO update with V-pretrain for the surgery-expanded FlatNets.

Mirrors the mechanics proven in tools/slot_ppo.py: GAE advantages,
a critic-only V-pretrain phase with a fast optimizer (the shipped critic
is miscalibrated under a new base), then clipped PPO with per-minibatch
KL early stop. Logprobs are mean-field per-dim normalized so buildings
with different zone/AHU counts share one clip/KL scale; padded entities
contribute exactly 0 through the masks.

Continuous heads only (zone damper/reheat/target, AHU
temp/pressure/recirc, plant water/pressure; switch votes are frozen):
the target head drives the PI loop via the delay-1 offset dict (P7).

Self-test (no sim): ./.venv/bin/python -B tools/skynet_ppo.py
"""

import sys

sys.path.insert(0, "/tmp/pkg")

import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.995
LAM = 0.95
REWARD_SCALE = 10.0
LR = 3e-5
V_LR = 1e-3
CLIP = 0.2
V_COEF = 0.5
KL_STOP = 0.01
V_PRETRAIN_STEPS = 100
# Logstd floor (H5, slot_ppo-style): shipped logstds sit at -1.1..-1.5,
# so -2.0 (std 0.135) is init-minus-~0.5 and a no-op at start.
MIN_LOGSTD = -2.0


def clamp_logstd(net, floor=MIN_LOGSTD):
    """In-place floor on the three head logstds (no-op above floor)."""
    with torch.no_grad():
        for s in net.stds():
            s.clamp_(min=floor)


def critic_params(net):
    return (list(net.c_zone.parameters()) + list(net.c_ahu.parameters())
            + list(net.c_plant.parameters()) + list(net.c_head.parameters()))


def _grad_norm(params):
    """L2 grad norm over params (read-only, no clipping)."""
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().norm(2).item()) ** 2
    return tot ** 0.5


def policy_params(net):
    """All params except the critic heads (actor trunk + heads)."""
    cids = {id(p) for p in critic_params(net)}
    return [p for p in net.parameters() if id(p) not in cids]


# Item-40 guard (P7 step 3): L2 on the surgery adapter weights only, so
# the PI loop-state features cannot grow unbounded authority. Value is a
# small standard decay; the plan fixes no number.
ADAPTER_L2 = 1e-4


def l2_groups(named_params):
    """Adam param groups: plain rest + ADAPTER_L2-decayed adapter."""
    rest, adapt = [], []
    for n, p in named_params:
        (adapt if ".adapter." in n else rest).append(p)
    groups = []
    if rest:
        groups.append({"params": rest})
    if adapt:
        groups.append({"params": adapt, "weight_decay": ADAPTER_L2})
    return groups


def masked_lp(net, zone, zmask, ahu, amask, plant, za, aa, pa):
    """Summed Gaussian logprob per timestep + entity-dim count.

    zone [T,V,27], ahu [T,K,NA], plant [T,NP]; masks [T,V]/[T,K] float.
    All three zone heads enter the ratio (P7: the target head drives the
    PI loop via the delay-1 _zone_target_offset dict). Switch votes stay
    frozen.
    Returns (lp_sum [T], ndim [T]).
    """
    import torch.distributions as _D
    (mz, ma, mp), _v = net(zone, zmask, ahu, amask, plant)
    stds = [torch.clamp(s, min=MIN_LOGSTD) for s in net.stds()]
    outs = []
    total = torch.zeros(zone.shape[0])
    # Zone damper/reheat/target heads.
    dz = _D.Normal(mz[..., :3],
                   stds[0][:3].exp().expand_as(mz[..., :3]))
    lpz = dz.log_prob(za[..., :3]).sum(-1) * zmask
    outs.append(lpz.sum(-1))
    n = 3.0 * zmask.sum(-1)
    # AHU heads (temp, pressure, recirc; run vote frozen).
    da = _D.Normal(ma[..., :3],
                   stds[1][:3].exp().expand_as(ma[..., :3]))
    lpa = da.log_prob(aa[..., :3]).sum(-1) * amask
    outs.append(lpa.sum(-1))
    n = n + 3.0 * amask.sum(-1)
    # Plant water/pressure; pump vote frozen.
    dp = _D.Normal(mp[..., :2],
                   stds[2][:2].exp().expand_as(mp[..., :2]))
    lpp = dp.log_prob(pa[..., :2]).sum(-1)
    outs.append(lpp)
    n = n + 2.0
    return torch.stack(outs).sum(0), n


def predict_v(net, zone, zmask, ahu, amask, plant):
    with torch.no_grad():
        return net(zone, zmask, ahu, amask, plant)[1]


def compute_gae(rewards, values, gamma=GAMMA, lam=LAM, bootstrap=0.0):
    """rewards/values [T] numpy; returns (advantages, returns).

    bootstrap is V beyond the traj end: 0.0 is correct for our
    full-episode terminal rollouts (H7); pass the last value for
    truncated rollouts (slot_ppo-style).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < T else bootstrap
        delta = rewards[t] + gamma * nxt - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + np.asarray(values, dtype=np.float64)


def normalize_adv(adv, fr):
    """Per-traj advantage norm over free steps (their 5.3, plan F/E2).

    adv is modified in place on free entries only; gated entries keep
    raw values (the fr mask zeroes their PG contribution anyway).
    With no free steps the array is returned untouched. Cold-climate
    rewards can never set the scale for other trajs: each traj is
    normalized against itself.
    """
    m = np.asarray(fr) > 0
    if m.any():
        a = np.asarray(adv, dtype=np.float64)
        adv[m] = (a[m] - a[m].mean()) / (a[m].std() + 1e-8)
    return adv


def clip_frac(ratio, fr, clip=CLIP):
    """Fr-masked fraction of ratios outside the [1-c, 1+c] band (E5)."""
    fr = torch.as_tensor(fr, dtype=ratio.dtype, device=ratio.device)
    out = (ratio - 1.0).abs() > clip
    return float((out.float() * fr).sum() / fr.sum().clamp(min=1.0))


def pg_objective(ratio, adv, fr, clip=CLIP):
    """Masked-mean clipped policy objective (their skynet_train.py:199).

    fr=0 steps (rule alone, plan F/D3) contribute exactly 0; the mean
    runs over free steps only. All-fr-0 returns 0.0, never NaN.
    """
    pol = -torch.minimum(ratio * adv,
                         torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    return (pol * fr).sum() / fr.sum().clamp(min=1.0)


def mean_kl(old_lp, old_nd, new_lp, new_nd, fr):
    """Masked-mean per-dim KL for the early-stop probe (fr as above)."""
    kl = old_lp / old_nd - new_lp / new_nd
    return float((kl * fr).sum() / fr.sum().clamp(min=1.0))


def ppo_epoch(net, opt, batch, clip=CLIP, kl_stop=KL_STOP, mb=32,
              trace=False):
    """One shuffled epoch; returns (ran_minibatches, stopped_early,
    mean clip_frac over ran minibatches). trace=True prints a per-mb
    H1 diagnostic line (frsum, grad norm, k1/k3 KL, clip_frac, |dmu|,
    dlogstd)."""
    (zone, zmask, ahu, amask, plant, za, aa, pa, old_lp, old_nd,
     adv, ret, fr) = batch
    T = zone.shape[0]
    idx = torch.randperm(T)
    ran = 0
    cf_sum = 0.0
    for s in range(0, T, mb):
        ii = idx[s:s + mb]
        sl = lambda t: t[ii]
        new_lp, new_nd = masked_lp(net, sl(zone), sl(zmask), sl(ahu),
                                   sl(amask), sl(plant), sl(za), sl(aa),
                                   sl(pa))
        # Mean-field per-dim ratio (slot_ppo convention).
        r = torch.exp(new_lp / new_nd - old_lp[ii] / old_nd[ii])
        a = adv[ii]
        pol = pg_objective(r, a, fr[ii], clip)
        mus, v = net(sl(zone), sl(zmask), sl(ahu), sl(amask),
                       sl(plant))
        if trace:
            mus0 = tuple(m.detach().clone() for m in mus)
            stds0 = [t.detach().clone() for t in net.stds()]
        loss = pol + V_COEF * ((v - ret[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if trace:
            gpol = _grad_norm(policy_params(net))
            gval = _grad_norm(critic_params(net))
        gnorm = float(nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        clamp_logstd(net)
        opt.step()
        ran += 1
        cf = clip_frac(r.detach(), fr[ii], clip)
        cf_sum += cf
        kl = mean_kl(old_lp[ii], old_nd[ii], new_lp.detach(), new_nd,
                     fr[ii])
        if trace:
            with torch.no_grad():
                mus1, _ = net(sl(zone), sl(zmask), sl(ahu), sl(amask),
                              sl(plant))
                stds1 = net.stds()
                dh = [float((b - a).abs().max())
                      for a, b in zip(mus0, mus1)]
                dmu = max(dh)
                dls = max(float((b - a).abs().max())
                          for a, b in zip(stds0, stds1))
            lr = torch.log(r.detach())
            k3 = float((((r.detach() - 1.0) - lr) * fr[ii]).sum()
                       / fr[ii].sum().clamp(min=1.0))
            print(f"mb {ran}: frsum={float(fr[ii].sum()):.0f} "
                  f"gnorm={gnorm:.3f} gpol={gpol:.3f} gval={gval:.3f} "
                  f"k1={kl:.6f} k3={k3:.6f} "
                  f"cf={cf:.4f} dmu={dmu:.2e} "
                  f"dmuz={dh[0]:.2e} dmua={dh[1]:.2e} dmup={dh[2]:.2e} "
                  f"dlogstd={dls:.2e}",
                  flush=True)
        if kl > kl_stop:
            return ran, True, cf_sum / ran
    return ran, False, cf_sum / max(ran, 1)


def v_pretrain(net, batch, steps=V_PRETRAIN_STEPS, lr=V_LR):
    (zone, zmask, ahu, amask, plant, _za, _aa, _pa, _ol, _on, _ad,
     ret, _fr) = batch
    cids = {id(p) for p in critic_params(net)}
    opt = torch.optim.Adam(l2_groups(
        [(n, p) for n, p in net.named_parameters() if id(p) in cids]),
        lr=lr)
    before = float(((predict_v(net, zone, zmask, ahu, amask, plant)
                     - ret) ** 2).mean())
    for _ in range(steps):
        v = net(zone, zmask, ahu, amask, plant)[1]
        loss = ((v - ret) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = float(((predict_v(net, zone, zmask, ahu, amask, plant)
                    - ret) ** 2).mean())
    return before, after


def save_ckpt(net, opt, path, extra=None):
    torch.save({"model": net.state_dict(),
                "opt": opt.state_dict() if opt is not None else None,
                "cfg": dict(net.cfg),
                "extra": extra or {}}, path)


def load_ckpt(path, map_location="cpu"):
    """Rebuild the expanded structure via surgery, then load weights."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
    import skynet_surgery as _surg
    from skynet.nets import FlatNets
    base = FlatNets(d=ck["cfg"]["d"], depth=ck["cfg"]["depth"], zone_out=2)
    net, _ = _surg.expand_net(base)
    net.load_state_dict(ck["model"])
    return net, ck


def self_test():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS/tools")
    import skynet_surgery as surg
    from skynet.policy import load
    from skynet.features import N_AHU, N_PLANT
    net, _, _, _ = load("/tmp/opencode/sb-rl/skynet.pt")
    new, _ = surg.expand_net(net)
    new.train()
    T, V, K = 64, 5, 2
    NA = N_AHU
    NP = N_PLANT
    zone = torch.randn(T, V + 1, 21 + surg.NEW_ZONE_INPUTS)
    zmask = torch.ones(T, V + 1)
    zmask[:, V] = 0.0  # padded row, never a real zone
    ahu = torch.randn(T, K, NA)
    amask = torch.ones(T, K)
    plant = torch.randn(T, NP)
    za = torch.randn(T, V + 1, 3)
    aa = torch.randn(T, K, 4)
    pa = torch.randn(T, 3)
    with torch.no_grad():
        old_lp, old_nd = masked_lp(new, zone, zmask, ahu, amask, plant,
                                   za, aa, pa)
        v0 = new(zone, zmask, ahu, amask, plant)[1]
    rew = np.random.randn(T) * 0.01
    adv_np, ret_np = compute_gae(rew, v0.numpy(), GAMMA, LAM)
    adv = (torch.from_numpy(adv_np).float()
           - float(adv_np.mean())) / (float(adv_np.std()) + 1e-8)
    ret = torch.from_numpy(ret_np).float()
    fr = torch.ones(T)
    fr[::2] = 0.0  # half the steps gated: rule alone, no PG signal
    batch = (zone, zmask, ahu, amask, plant, za, aa, pa,
             old_lp.detach(), old_nd, adv, ret, fr)
    # 0. fr masking: gated rows contribute exactly 0 to pg + KL.
    r = torch.exp(torch.randn(T))
    a = torch.randn(T)
    one = torch.zeros(T)
    one[7] = 1.0
    pol = -torch.minimum(r * a, torch.clamp(r, 1 - CLIP, 1 + CLIP) * a)
    assert torch.equal(pg_objective(r, a, one), pol[7]), "one-hot inexact"
    assert pg_objective(r, a, torch.zeros(T)) == 0.0, "all-gated not 0"
    got = pg_objective(r, a, fr)
    want = pg_objective(r[1::2], a[1::2], fr[1::2])
    assert abs(float(got - want)) < 1e-6, "fr=0 rows leak into pg"
    k_all = mean_kl(old_lp, old_nd, old_lp + 1.0, old_nd, fr)
    k_free = mean_kl(old_lp[1::2], old_nd[1::2], old_lp[1::2] + 1.0,
                     old_nd[1::2], fr[1::2])
    assert abs(k_all - k_free) < 1e-6, "fr=0 rows leak into KL"
    print("fr-mask check PASS (gated rows contribute exactly 0.0)")
    # 1. Mask correctness: padded row contributes exactly 0.
    with torch.no_grad():
        l_full, _ = masked_lp(new, zone, zmask, ahu, amask, plant,
                              za, aa, pa)
        l_cut, _ = masked_lp(new, zone[:, :V], zmask[:, :V], ahu, amask,
                             plant, za[:, :V], aa, pa)
    assert torch.equal(l_full, l_cut), "mask leak from padded row"
    print("mask check PASS (padded row contributes exactly 0.0)")
    # 2. V-pretrain reduces MSE.
    before, after = v_pretrain(new, batch)
    print(f"v-pretrain: {before:.6f} -> {after:.6f}")
    assert after < before, "v-pretrain did not reduce loss"
    # 3. KL-stop fires under an absurd lr (1e-1 moves past 0.01 in an epoch).
    opt = torch.optim.Adam(new.parameters(), lr=1e-1)
    ran, stopped, cf = ppo_epoch(new, opt, batch)
    print(f"kl-stop check: ran {ran} minibatches, stopped={stopped} "
          f"clip_frac={cf:.3f}")
    assert stopped, "KL-stop did not fire at lr=1e-1"
    # 4. Save/load roundtrip bit-identical (via load_ckpt).
    opt2 = torch.optim.Adam(new.parameters(), lr=LR)
    save_ckpt(new, opt2, "/tmp/opencode/sb-rl/ppo_test.pt")
    new2, ck = load_ckpt("/tmp/opencode/sb-rl/ppo_test.pt")
    assert set(ck.keys()) == {"model", "opt", "cfg", "extra"}, ck.keys()
    for (k1, p1), (k2, p2) in zip(new.state_dict().items(),
                                 new2.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), f"roundtrip diff: {k1}"
    print("ckpt roundtrip PASS (all tensors bit-identical)")
    print("SELF-TEST PASS")


if __name__ == "__main__":
    self_test()
