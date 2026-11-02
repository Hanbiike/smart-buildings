"""Weight surgery: expand shipped FlatNets with stateful inputs + zone-target head.

Adds 2 zone inputs (PI-target error, PI integral) via a zero-init adapter
alongside the unchanged 21-wide first GEMM (same summation order, exact-0.0
addition), plus a third zone output (zone-target offset, TARGET_K scale
already in skynet/features.py) with a zeroed row. Old heads are therefore
bit-identical and the new head outputs exactly 0.0 (PI target and composed
action unchanged).

Run: ./.venv/bin/python -B tools/skynet_surgery.py
"""

import sys

sys.path.insert(0, "/tmp/pkg")

import torch
import torch.nn as nn

NEW_ZONE_INPUTS = 2
NEW_ZONE_OUT = 3
OLD_ZONE_OUT = 2


class AdapterFirst(nn.Module):
    """Bit-identical trunk extension: old inputs run through the unchanged
    21-wide GEMM (same summation order), new inputs through a zero-init
    adapter whose exact-0.0 output is added before Tanh."""

    def __init__(self, old_first, d):
        super().__init__()
        self.old = nn.Linear(old_first.in_features, d)
        with torch.no_grad():
            self.old.weight.copy_(old_first.weight)
            self.old.bias.copy_(old_first.bias)
        self.adapter = nn.Linear(NEW_ZONE_INPUTS, d)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.bias.zero_()

    def forward(self, x):
        return self.old(x[..., : -NEW_ZONE_INPUTS]) + self.adapter(x[..., -NEW_ZONE_INPUTS:])


def expand_net(net):
    """Return (new_net, report) with copied weights and zeroed additions."""
    from skynet.nets import FlatNets

    cfg = dict(net.cfg)
    new = FlatNets(d=cfg["d"], depth=cfg["depth"], zone_out=NEW_ZONE_OUT)
    new.zone = nn.Sequential(AdapterFirst(net.zone[0], cfg["d"]), nn.Tanh(),
                             net.zone[2], nn.Tanh())
    new.c_zone = nn.Sequential(AdapterFirst(net.c_zone[0], cfg["d"]), nn.Tanh(),
                               net.c_zone[2], nn.Tanh())
    with torch.no_grad():
        # Trunks already carry copied weights via AdapterFirst + shared
        # layer-2 modules; heads below get row copies with zeroed additions.
        # Zone mu head: copy old rows, zero the target row.
        new.zone_mu.weight.zero_()
        new.zone_mu.weight[:OLD_ZONE_OUT].copy_(net.zone_mu.weight)
        new.zone_mu.bias.zero_()
        new.zone_mu.bias[:OLD_ZONE_OUT].copy_(net.zone_mu.bias)
        # Critic zone trunk already carries copied weights (see above).
        # Rest identical shapes: direct copy.
        for name in ("ahu", "plant", "c_ahu", "c_plant"):
            new._modules[name].load_state_dict(net._modules[name].state_dict())
        new.ahu_mu.load_state_dict(net.ahu_mu.state_dict())
        new.plant_mu.load_state_dict(net.plant_mu.state_dict())
        new.c_head.load_state_dict(net.c_head.state_dict())
        # Log-stds: keep old values, init target head like a fresh net.
        new.zone_log_std[:OLD_ZONE_OUT].copy_(net.zone_log_std)
        new.ahu_log_std.copy_(net.ahu_log_std)
        new.plant_log_std.copy_(net.plant_log_std)
    n_new = sum(1 for _ in new.parameters())
    return new, {"params_total": sum(p.numel() for p in new.parameters()),
                 "tensors": n_new}


def main():
    # Single-threaded GEMMs: multithreaded reductions sum in a different
    # order call-to-call (1-ulp noise), which would fail exact equality.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    from skynet.nets import FlatNets
    from skynet.policy import load

    net, params, scale, latch = load("/tmp/opencode/sb-rl/skynet.pt")
    new, rep = expand_net(net)
    print("expanded:", rep, "params:", params, "scale:", scale, "latch:", latch)

    from skynet.features import N_AHU, N_PLANT
    z = torch.randn(1, 7, 21)
    a = torch.randn(1, 2, N_AHU)
    p = torch.randn(1, N_PLANT)
    zm = torch.ones(1, 7)
    am = torch.ones(1, 2)
    (mz, ma, mp), v = net(z, zm, a, am, p)

    zp = torch.zeros(1, 7, 23)
    zp[:, :, :21].copy_(z)
    (mz2, ma2, mp2), v2 = new(zp, zm, a, am, p)

    assert mz2.shape == (1, 7, 3), mz2.shape
    assert torch.equal(mz2[:, :, :2], mz), "old zone heads differ"
    assert torch.equal(ma2, ma), "ahu heads differ"
    assert torch.equal(mp2, mp), "plant heads differ"
    assert torch.equal(v2, v), "critic differs"
    assert bool((mz2[:, :, 2] == 0.0).all()), "target head not exactly zero"
    print("PASS: old heads bit-identical, target head exactly 0.0, "
          "critic bit-identical")


def expand_net_from_ckpt(ck):
    """Rebuild the expanded structure for a ckpt saved from it.

    ck has {"model", "cfg", ...} with adapter keys present; reconstructs
    via expand_net on a fresh base then loads the state dict (shapes match
    by construction).
    """
    from skynet.nets import FlatNets
    base = FlatNets(d=ck["cfg"]["d"], depth=ck["cfg"]["depth"], zone_out=2)
    net, rep = expand_net(base)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, rep


def expanded_zone_cols(floor, ix, obs):
    """[V,2] float32 stateful inputs, shared by probe and rollout.

    Column 0: the loop's true zone error (deadband-aware via
    floor._zone_err, same lo/hi logic as _command_zones) over 5 K.
    Column 1: PI integral saturation fraction (integral * KI, in [-1,1]
    by the conditional-integration clip at +/-1/KI).
    Read-only with respect to the floor policy: band/target/err/integral
    accessors are pure reads, so base-policy trajectories are untouched.
    Exact in normal/prestart conditioning; approximate in floor/shutdown
    (the residual is gated to conditioning modes at runtime).
    """
    import numpy as _np
    zr = _np.asarray(obs, dtype=_np.float64)[ix.zone_obs]
    temp_k = zr[:, 0] * ix.t_std + ix.t_mean
    lo_c, hi_c = (float(x) for x in floor.comfort_temp_window)
    cols = _np.zeros((ix.V, 2), dtype=_np.float32)
    ki = float(floor.KI)
    for i, vav in enumerate(ix.zone_vav):
        lo, hi = floor._zone_band(obs, vav, lo_c, hi_c)
        if hi - lo > (hi_c - lo_c) + 1e-6:
            lo, hi = lo_c, hi_c
        err = float(floor._zone_err(obs, float(temp_k[i]), lo, hi)) / 5.0
        sat = float(floor._zone_integral.get(vav, 0.0)) * ki
        cols[i, 0] = _np.float32(_np.clip(err, -10.0, 10.0))
        cols[i, 1] = _np.float32(_np.clip(sat, -10.0, 10.0))
    return cols


if __name__ == "__main__":
    main()
