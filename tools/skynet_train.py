"""Online residual-PPO train driver, cycle 0: short loop, full diagnostics.

Rollouts (sequential sims) -> pad/collate across buildings -> V-pretrain
-> GAE recompute -> PPO epochs with KL-stop -> save ckpt. Deterministic
seeds; all diagnostics printed (batch reward sums, value loss, per-epoch
policy/value loss, clip fraction, KL, early-stop flag).

  ./.venv/bin/python -B tools/skynet_train.py --help

Cycle-0 smoke (fast, proves the loop end to end):
  ./.venv/bin/python -B tools/skynet_train.py --rollout-steps 512 \
      --scenarios building_train_8/atlanta_1,building_train_7/denver_5 \
      --epochs 1 --v-steps 20 --out /tmp/opencode/sb-rl/cycle0.pt
"""

import argparse
import sys

sys.path.insert(0, "/tmp/pkg")
sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS")

import numpy as np
import torch


def _load_mod(name, path):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(name, path)
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[name] = _mod
    _spec.loader.exec_module(_mod)
    return _mod


def collate(trajs):
    """Pad trajs to common (V,K); return flat batch dict + per-traj index."""
    nV = max(t["zone"][0].shape[0] for t in trajs)
    nK = max(t["ahu"][0].shape[0] for t in trajs)
    F = {k: [] for k in ("zone", "zmask", "ahu", "amask", "plant",
                         "za", "aa", "pa", "lpz", "lpa", "lpp", "r", "v")}
    bounds = []
    for t in trajs:
        T = len(t["r"])
        V = t["zone"][0].shape[0]
        K = t["ahu"][0].shape[0]
        F["zone"].append(np.stack([np.pad(z, ((0, nV - V), (0, 0)))
                                   for z in t["zone"]]))
        m = np.zeros((T, nV), dtype=np.float32)
        m[:, :V] = 1.0
        F["zmask"].append(m)
        F["ahu"].append(np.stack([np.pad(a, ((0, nK - K), (0, 0)))
                                  for a in t["ahu"]]))
        m = np.zeros((T, nK), dtype=np.float32)
        m[:, :K] = 1.0
        F["amask"].append(m)
        F["plant"].append(np.stack(t["plant"]))
        F["za"].append(np.stack([np.pad(a, ((0, nV - V), (0, 0)))
                                 for a in t["za"]]))
        F["aa"].append(np.stack([np.pad(a, ((0, nK - K), (0, 0)))
                                 for a in t["aa"]]))
        F["pa"].append(np.stack(t["pa"]))
        F["lpz"].append(np.stack([np.pad(a, (0, nV - V)) for a in t["lpz"]]))
        F["lpa"].append(np.stack([np.pad(a, (0, nK - K)) for a in t["lpa"]]))
        F["lpp"].append(np.stack(t["lpp"]))
        F["r"].append(np.stack(t["r"]).astype(np.float64) * 10.0)
        F["v"].append(np.stack(t["v"]).astype(np.float64))
        bounds.append(T)
    return {k: np.concatenate(v, axis=0) for k, v in F.items()}, bounds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-steps", type=int, default=512)
    ap.add_argument("--scenarios", default="building_train_8/atlanta_1,"
                    "building_train_8/fairbanks_8,"
                    "building_xfer_1/minneapolis_1,"
                    "building_xfer_3/singapore_2")
    ap.add_argument("--transfer-dir", default="smart_buildings/"
                    "transfer_scenarios")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-stop", type=float, default=0.01)
    ap.add_argument("--v-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/opencode/sb-rl/cycle0.pt")
    ap.add_argument("--init", default="/tmp/opencode/sb-rl/skynet.pt")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    surg = _load_mod("skynet_surgery",
                     "/home/a_rakhymbekov/BUILDINGS/tools/skynet_surgery.py")
    roll = _load_mod("skynet_rollout",
                     "/home/a_rakhymbekov/BUILDINGS/tools/skynet_rollout.py")
    ppo = _load_mod("skynet_ppo",
                    "/home/a_rakhymbekov/BUILDINGS/tools/skynet_ppo.py")
    from skynet.policy import load as _load
    _init_ck = torch.load(a.init, map_location="cpu", weights_only=False)
    if "zone.0.adapter.weight" in _init_ck.get("model", {}):
        net, _rep = surg.expand_net_from_ckpt(_init_ck)
    else:
        net0, _p, _s, _l = _load(a.init)
        net, _rep = surg.expand_net(net0)
    net.train()

    from pathlib import Path
    from smart_buildings import evaluate as E
    from smart_buildings.evaluate import POLICY_MAP
    trajs = []
    for i, pair in enumerate(a.scenarios.split(",")):
        b, w = pair.split("/")
        root = Path("smart_buildings/transfer_scenarios" if b.startswith(
            "building_xfer") else "smart_buildings/training_scenarios")
        if b.startswith("building_xfer"):
            root = Path(a.transfer_dir)
        d = root / b / w
        env = E.load_hybrid_environment(d / "config.gin")
        env.metrics_path = None
        meta = E.create_building_meta_data(env, d)
        spec = env.action_spec()
        base = E.initialize_policy(POLICY_MAP["occ_gated7"], meta, spec)
        tr = roll.rollout_env(env, meta, spec, base, a.rollout_steps,
                              seed=a.seed + i)
        trajs.append(tr)
        print(f"rollout {pair}: steps={len(tr['r'])} "
              f"reward_sum={sum(tr['r']):.4f} "
              f"{tr['steps_per_sec']:.1f} steps/sec", flush=True)

    F, bounds = collate(trajs)
    print(f"batch: S={len(trajs)} T={bounds} "
          f"flat={F['zone'].shape[0]} "
          f"nV={F['zone'].shape[1]} nK={F['ahu'].shape[1]}", flush=True)
    t = torch.from_numpy
    B = {k: t(v) for k, v in F.items() if k not in ("r", "v")}
    B["zone"] = B["zone"].float()
    B["ahu"] = B["ahu"].float()
    B["plant"] = B["plant"].float()
    B["za"] = B["za"].float()
    B["aa"] = B["aa"].float()
    B["pa"] = B["pa"].float()
    B["zmask"] = B["zmask"].float()
    B["amask"] = B["amask"].float()
    B["lpz"] = B["lpz"].float()
    B["lpa"] = B["lpa"].float()
    B["lpp"] = B["lpp"].float()
    # Old logprobs MUST come from the same masked_lp the update uses
    # (rollout stores all 3 zone heads; the ratio uses 2 + frozen votes).
    # Recompute deterministically from the init weights instead.
    with torch.no_grad():
        old_lp, old_nd = ppo.masked_lp(
            net, B["zone"], B["zmask"], B["ahu"], B["amask"],
            B["plant"], B["za"], B["aa"], B["pa"])
    old_lp = old_lp.detach()
    old_nd = old_nd.detach()
    batch0 = (B["zone"], B["zmask"], B["ahu"], B["amask"], B["plant"],
              B["za"], B["aa"], B["pa"], old_lp, old_nd,
              torch.zeros(F["zone"].shape[0]),
              torch.zeros(F["zone"].shape[0]))
    vb, va = ppo.v_pretrain(net, batch0, steps=a.v_steps)
    print(f"v-pretrain: {vb:.4f} -> {va:.4f}", flush=True)

    # GAE per scenario on scaled rewards with fresh values.
    with torch.no_grad():
        v_all = net(B["zone"], B["zmask"], B["ahu"], B["amask"],
                    B["plant"])[1].numpy()
    adv_all, ret_all = [], []
    off = 0
    for T in bounds:
        sl = slice(off, off + T)
        an, rn = ppo.compute_gae(F["r"][sl], v_all[sl])
        adv_all.append(an)
        ret_all.append(rn)
        off += T
    adv = np.concatenate(adv_all)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    batch = (B["zone"], B["zmask"], B["ahu"], B["amask"], B["plant"],
             B["za"], B["aa"], B["pa"], old_lp.detach(), old_nd,
             t(adv).float(), t(np.concatenate(ret_all)).float())
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    for e in range(a.epochs):
        ran, stopped = ppo.ppo_epoch(net, opt, batch, clip=ppo.CLIP,
                                     kl_stop=a.kl_stop, mb=a.minibatch)
        with torch.no_grad():
            nl, _ = ppo.masked_lp(net, B["zone"], B["zmask"], B["ahu"],
                                  B["amask"], B["plant"], B["za"], B["aa"],
                                  B["pa"])
            kl = float((old_lp - nl / old_nd).mean())
        print(f"epoch {e}: ran_mb={ran} stopped={stopped} "
              f"mean_kl={kl:.6f}", flush=True)
        if stopped:
            break
    extra = {"scenarios": a.scenarios, "bounds": bounds,
             "v_before": vb, "v_after": va}
    ppo.save_ckpt(net, opt, a.out, extra=extra)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
