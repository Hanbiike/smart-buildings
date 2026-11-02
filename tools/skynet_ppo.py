"""PPO update with V-pretrain for the surgery-expanded FlatNets.

Mirrors the mechanics proven in tools/slot_ppo.py: GAE advantages,
a critic-only V-pretrain phase with a fast optimizer (the shipped critic
is miscalibrated under a new base), then clipped PPO with per-minibatch
KL early stop. Logprobs are mean-field per-dim normalized so buildings
with different zone/AHU counts share one clip/KL scale; padded entities
contribute exactly 0 through the masks.

Continuous heads only (zone damper/reheat, AHU temp/pressure/recirc,
plant water/pressure/pump votes are frozen): the target head and switch
votes are excluded from the ratio until their effectors exist.

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


def critic_params(net):
    return (list(net.c_zone.parameters()) + list(net.c_ahu.parameters())
            + list(net.c_plant.parameters()) + list(net.c_head.parameters()))


def masked_lp(net, zone, zmask, ahu, amask, plant, za, aa, pa):
    """Summed Gaussian logprob per timestep + entity-dim count.

    zone [T,V,23], ahu [T,K,NA], plant [T,NP]; masks [T,V]/[T,K] float.
    Only the two trained zone heads enter the ratio (target head excluded).
    Returns (lp_sum [T], ndim [T]).
    """
    import torch.distributions as _D
    (mz, ma, mp), _v = net(zone, zmask, ahu, amask, plant)
    stds = net.stds()
    outs = []
    total = torch.zeros(zone.shape[0])
    # Zone damper/reheat heads.
    dz = _D.Normal(mz[..., :2],
                   stds[0][:2].exp().expand_as(mz[..., :2]))
    lpz = dz.log_prob(za[..., :2]).sum(-1) * zmask
    outs.append(lpz.sum(-1))
    n = 2.0 * zmask.sum(-1)
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


def compute_gae(rewards, values, gamma=GAMMA, lam=LAM):
    """rewards/values [T] numpy; returns (advantages, returns)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + np.asarray(values, dtype=np.float64)


def ppo_epoch(net, opt, batch, clip=CLIP, kl_stop=KL_STOP, mb=32):
    """One shuffled epoch; returns (ran_minibatches, stopped_early)."""
    (zone, zmask, ahu, amask, plant, za, aa, pa, old_lp, old_nd,
     adv, ret) = batch
    T = zone.shape[0]
    idx = torch.randperm(T)
    ran = 0
    for s in range(0, T, mb):
        ii = idx[s:s + mb]
        sl = lambda t: t[ii]
        new_lp, new_nd = masked_lp(net, sl(zone), sl(zmask), sl(ahu),
                                   sl(amask), sl(plant), sl(za), sl(aa),
                                   sl(pa))
        # Mean-field per-dim ratio (slot_ppo convention).
        r = torch.exp(new_lp / new_nd - old_lp[ii] / old_nd[ii])
        a = adv[ii]
        pol = -torch.minimum(r * a, torch.clamp(r, 1 - clip, 1 + clip) * a)
        v = net(sl(zone), sl(zmask), sl(ahu), sl(amask), sl(plant))[1]
        loss = pol.mean() + V_COEF * ((v - ret[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ran += 1
        kl = float((old_lp[ii] / old_nd[ii]
                    - new_lp.detach() / new_nd).mean())
        if kl > kl_stop:
            return ran, True
    return ran, False


def v_pretrain(net, batch, steps=V_PRETRAIN_STEPS, lr=V_LR):
    (zone, zmask, ahu, amask, plant, _za, _aa, _pa, _ol, _on, _ad,
     ret) = batch
    opt = torch.optim.Adam(critic_params(net), lr=lr)
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
    zone = torch.randn(T, V + 1, 23)
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
    batch = (zone, zmask, ahu, amask, plant, za, aa, pa,
             old_lp.detach(), old_nd, adv, ret)
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
    ran, stopped = ppo_epoch(new, opt, batch)
    print(f"kl-stop check: ran {ran} minibatches, stopped={stopped}")
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
