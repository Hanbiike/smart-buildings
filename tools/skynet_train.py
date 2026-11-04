"""Online residual-PPO train driver: rollout cycles with D4 recipe.

Per cycle: rollouts (sequential sims) -> pad/collate -> V-pretrain
(cycle 0 only) -> GAE recompute -> PPO epochs with KL-stop -> save
per-cycle ckpt. Deterministic seeds; all diagnostics printed.
D4 recipe (their 5.1-5.3): lr linear decay to --lr-end over the run,
per-traj advantage norm over free steps, honest bill by default (B3).

  ./.venv/bin/python -B tools/skynet_train.py --help

Smoke (fast, proves the loop end to end):
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
                         "za", "aa", "pa", "lpz", "lpa", "lpp", "r", "v",
                         "fr")}
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
        F["fr"].append(np.stack(t["fr"]).astype(np.float32))
        bounds.append(T)
    return {k: np.concatenate(v, axis=0) for k, v in F.items()}, bounds


def pool_checksum(pairs):
    """SHA-256 over the sorted scenario list (E4 pool checksum)."""
    import hashlib
    body = "\n".join(sorted(pairs))
    return hashlib.sha256(body.encode()).hexdigest()


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
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=0.0)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bill", default="honest", choices=("honest", "shipped"))
    ap.add_argument("--kl-stop", type=float, default=0.01)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--v-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule-only", action="store_true",
                    help="force-gate every step (zero corrections): rule-only "
                         "batch, no update (plan-19 P1)")
    ap.add_argument("--dump-batch", default=None,
                    help="npz path for the cycle-0 collated batch + meta "
                         "(plan-19 P1)")
    ap.add_argument("--gate", default=None,
                    help="gate weights (.pt from gate_fit.py) for "
                         "rollout-side confidence gating (plan-19 P4)")
    ap.add_argument("--gate-tau", type=float, default=0.5)
    ap.add_argument("--out", default="/tmp/opencode/sb-rl/cycle0.pt")
    ap.add_argument("--init", default="/tmp/opencode/sb-rl/skynet.pt")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    surg = _load_mod("skynet_surgery",
                     "/home/a_rakhymbekov/BUILDINGS/tools/skynet_surgery.py")
    # Plain import (not _load_mod): spawn workers unpickle _rollout_task
    # by module reference, so the module must be importable in children.
    import tools.skynet_rollout as roll
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

    if a.bill == "honest":
        from tools.honest_bill import apply_honest_bill
        apply_honest_bill()
    print(f"bill: {a.bill} (B3: train on honest)", flush=True)

    from pathlib import Path
    from smart_buildings import evaluate as E
    from smart_buildings.evaluate import POLICY_MAP
    pairs = a.scenarios.split(",")
    # E4: pool checksum before update 0 (their 175-of-720 incident).
    print(f"pool: n={len(pairs)} sha256={pool_checksum(pairs)}", flush=True)

    def _gin(pair):
        b, w = pair.split("/")
        root = Path("smart_buildings/transfer_scenarios"
                    if b.startswith("building_xfer")
                    else "smart_buildings/training_scenarios")
        if b.startswith("building_xfer"):
            root = Path(a.transfer_dir)
        return str(root / b / w / "config.gin")

    gin_paths = [_gin(p) for p in pairs]
    total = len(pairs) * a.rollout_steps * a.cycles
    opt = torch.optim.Adam(ppo.l2_groups(net.named_parameters()),
                             lr=a.lr)
    behavior = f"{a.out}.behavior"
    done_steps = 0
    best_r, best_c = None, None  # best-ckpt retention (E3)
    t = torch.from_numpy
    import time as _time
    for c in range(a.cycles):
        lr_c = a.lr + (a.lr_end - a.lr) * (done_steps / total)
        for g in opt.param_groups:
            g["lr"] = lr_c
        print(f"--- cycle {c} lr={lr_c:.2e} ---", flush=True)
        # Behavior policy for this cycle's rollouts (fixes the stale
        # shipped-weights behavior of single-cycle drivers).
        ppo.save_ckpt(net, opt, behavior, extra={"cycle": c})
        seeds = [a.seed + c * 1000 + i for i in range(len(pairs))]
        t0 = _time.time()
        gate_spec = ((a.gate, a.gate_tau) if a.gate else None)
        if a.workers > 1:
            trajs = roll.rollout_parallel(gin_paths, a.rollout_steps,
                                          seeds, behavior, a.bill,
                                          a.workers,
                                          force_rule=a.rule_only,
                                          gate_spec=gate_spec)
        else:
            gate = None
            if gate_spec is not None:
                from tools.gate_fit import GateMLP
                gck = torch.load(gate_spec[0], map_location="cpu",
                                 weights_only=False)
                gate = (GateMLP(gck["arch"][0]), gate_spec[1])
                gate[0].load_state_dict(gck["state"])
                gate[0].eval()
            trajs = []
            for i, pair in enumerate(pairs):
                d = Path(gin_paths[i]).parent
                env = E.load_hybrid_environment(d / "config.gin")
                env.metrics_path = None
                meta = E.create_building_meta_data(env, d)
                spec = env.action_spec()
                base = E.initialize_policy(POLICY_MAP["occ_gated7"], meta,
                                           spec)
                tr = roll.rollout_env(env, meta, spec, base,
                                      a.rollout_steps, seed=seeds[i],
                                      weights=behavior,
                                      force_rule=a.rule_only,
                                      gate=gate)
                trajs.append(tr)
        wall = max(_time.time() - t0, 1e-6)
        flat = sum(len(tr["r"]) for tr in trajs)
        for pair, tr in zip(pairs, trajs):
            print(f"rollout {pair}: steps={len(tr['r'])} "
                  f"reward_sum={float(np.sum(tr['r'])):.4f} "
                  f"free_frac={np.mean(tr['fr']):.3f}", flush=True)
        print(f"rollout wall: {flat} steps in {wall:.0f}s "
              f"({flat / wall:.1f} steps/sec, {a.workers} workers)",
              flush=True)

        F, bounds = collate(trajs)
        print(f"batch: S={len(trajs)} T={bounds} "
              f"flat={F['zone'].shape[0]} "
              f"nV={F['zone'].shape[1]} nK={F['ahu'].shape[1]}", flush=True)
        if a.dump_batch and c == 0:
            np.savez_compressed(
                a.dump_batch, bounds=np.array(bounds),
                scenarios=np.array(pairs),
                pool_sha=pool_checksum(pairs), seeds=np.array(seeds),
                bill=a.bill, weights=behavior, base="occ_gated7",
                rollout_steps=a.rollout_steps, rule_only=a.rule_only,
                **{k: np.asarray(v) for k, v in F.items()})
            print(f"dumped batch {a.dump_batch}", flush=True)
        if a.rule_only:
            print("rule-only: no update, done", flush=True)
            break
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
        B["fr"] = B["fr"].float()
        # Old logprobs MUST come from the same masked_lp the update uses
        # (rollout stores all 3 zone heads; the ratio uses 2 + frozen
        # votes). Recomputed from the current weights each cycle.
        with torch.no_grad():
            old_lp, old_nd = ppo.masked_lp(
                net, B["zone"], B["zmask"], B["ahu"], B["amask"],
                B["plant"], B["za"], B["aa"], B["pa"])
        old_lp = old_lp.detach()
        old_nd = old_nd.detach()
        if c == 0:
            # H2: fit the critic on MC gamma-returns, not zeros
            # (slot_ppo fits on MC returns; zeros gave v~0 + huge
            # value MSE that saturated the global grad-norm clip).
            mc_all, off0 = [], 0
            for T in bounds:
                sl = slice(off0, off0 + T)
                mc_all.append(ppo.compute_gae(F["r"][sl], np.zeros(T),
                                              lam=1.0)[1])
                off0 += T
            batch0 = (B["zone"], B["zmask"], B["ahu"], B["amask"],
                      B["plant"], B["za"], B["aa"], B["pa"], old_lp,
                      old_nd, torch.zeros(F["zone"].shape[0]),
                      t(np.concatenate(mc_all)).float(), B["fr"])
            vb, va = ppo.v_pretrain(net, batch0, steps=a.v_steps)
            print(f"v-pretrain: {vb:.4f} -> {va:.4f}", flush=True)

        # GAE per scenario on scaled rewards with fresh values, then
        # per-traj advantage norm over free steps (their per-worker
        # norm over acted steps, 5.3). Returns stay unnormalized.
        with torch.no_grad():
            v_all = net(B["zone"], B["zmask"], B["ahu"], B["amask"],
                        B["plant"])[1].numpy()
        adv_all, ret_all = [], []
        off = 0
        for T in bounds:
            sl = slice(off, off + T)
            an, rn = ppo.compute_gae(F["r"][sl], v_all[sl])
            adv_all.append(ppo.normalize_adv(an, F["fr"][sl]))
            ret_all.append(rn)
            off += T
        adv = np.concatenate(adv_all)
        batch = (B["zone"], B["zmask"], B["ahu"], B["amask"], B["plant"],
                 B["za"], B["aa"], B["pa"], old_lp.detach(), old_nd,
                 t(adv).float(), t(np.concatenate(ret_all)).float(),
                 B["fr"])
        for e in range(a.epochs):
            ran, stopped, cf = ppo.ppo_epoch(
                net, opt, batch, clip=ppo.CLIP, kl_stop=a.kl_stop,
                mb=a.minibatch, trace=a.trace)
            with torch.no_grad():
                nl, _ = ppo.masked_lp(net, B["zone"], B["zmask"],
                                      B["ahu"], B["amask"], B["plant"],
                                      B["za"], B["aa"], B["pa"])
                kl = ppo.mean_kl(old_lp, old_nd, nl, old_nd, B["fr"])
            upd = c * a.epochs + e
            print(f"epoch {e}: ran_mb={ran} stopped={stopped} "
                  f"mean_kl={kl:.6f} clip_frac={cf:.4f}", flush=True)
            # E5: abort if clip_frac > 0.15 in the first 10 updates
            # (their 0.19-0.34 abort case, 9.5).
            if upd < 10 and cf > 0.15:
                raise RuntimeError(
                    f"E5 abort: clip_frac={cf:.4f} > 0.15 at update {upd}")
            if stopped:
                break
        done_steps += sum(bounds)
        ckpt = f"{a.out}.c{c}"
        extra = {"scenarios": a.scenarios, "bounds": bounds, "cycle": c,
                 "lr": lr_c, "v_before": vb, "v_after": va}
        ppo.save_ckpt(net, opt, ckpt, extra=extra)
        print(f"saved {ckpt}", flush=True)
        # Best-ckpt retention (E3): best mean reward/step over cycles.
        mean_r = float(np.sum(F["r"])) / max(len(F["r"]), 1)
        if best_r is None or mean_r > best_r:
            best_r, best_c = mean_r, c
            ppo.save_ckpt(net, opt, f"{a.out}.best",
                          extra=dict(extra, best_r=mean_r))
            print(f"new best {a.out}.best cycle={c} mean_r={mean_r:.6f}",
                  flush=True)
    ppo.save_ckpt(net, opt, a.out, extra={"scenarios": a.scenarios,
                                          "cycles": a.cycles})
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
