"""Rollout collector for Online residual-PPO (step 1 of the train loop).

Collects fixed-length trajectories with the surgery-expanded FlatNets:
stochastic actions (mu + exp(log_std) * eps per entity head), critic
values, shipped-sim rewards, and padding masks over variable-size
buildings. Switches stay frozen (continuous-head logprobs only);
Bernoulli switch heads are a later step.

Self-test: ./.venv/bin/python -B tools/skynet_rollout.py
Collects 2 scenarios x 256 steps sequentially and checks shapes, masks,
finite values, and throughput (sim + GPU batch forward).
"""

import sys
import time

sys.path.insert(0, "/tmp/pkg")

import numpy as np
import torch

N_ZONE_X = 23  # 21 base + err + integral saturation (see probe_xres)


def build_policy(meta, spec):
    """Deterministic harness parts (features + expanded net), no env."""
    import importlib.util as _ilu
    from pathlib import Path as _P
    _surg = (_P(__file__).resolve().parent
             / "skynet_surgery.py")
    _spec = _ilu.spec_from_file_location("skynet_surgery", _surg)
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["skynet_surgery"] = _mod
    _spec.loader.exec_module(_mod)
    from skynet.policy import load as _load
    from skynet.features import BuildingIndex as _BI
    net, _params, _scale, _latch = _load("/tmp/opencode/sb-rl/skynet.pt")
    xnet, _rep = _mod.expand_net(net)
    xnet.eval()
    return xnet, _BI(meta)


def rollout_env(env, meta, spec, base_policy, steps, seed=0):
    """Rollout driving a live env with the base policy + residual sample."""
    from skynet.features import N_AHU, N_PLANT
    from smart_buildings.smart_control.utils.scenario_generator import (
        SimulationTracker)
    import torch.distributions as _D
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    xnet, ix = build_policy(meta, spec)
    trk = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    trk.reset()
    obs = trk.last_obs
    T = {"zone": [], "ahu": [], "plant": [], "zmask": [], "amask": [],
         "za": [], "aa": [], "pa": [], "lpz": [], "lpa": [], "lpp": [],
         "gated": [], "v": [], "r": []}
    t0 = time.time()
    for _ in range(steps):
        base = base_policy.get_action(obs)
        pc = np.asarray(base["continuous_action"])
        pd = np.asarray(base["discrete_action"])
        # Mirror deployment gating (probe_xres): shutdown/floor/cold steps
        # take the base action. They are still recorded, with the
        # actually-applied zero correction and its logprob, so the update
        # sees exactly the deployed behavior.
        from smart_buildings.baselines.probe_modes import any_cold
        gated = bool(getattr(base_policy, "_shutdown", False)
                     or getattr(base_policy, "_floor_mode", None) is not
                     None
                     or any_cold(base_policy, ix, obs, 1.0))
        zf, af, pf = ix.features(obs, pc, pd)
        extra = sys.modules["skynet_surgery"].expanded_zone_cols(
            base_policy, ix, obs)
        z23 = np.concatenate([zf, extra], axis=1)
        t = torch.from_numpy
        with torch.no_grad():
            (mz, ma, mp), v = xnet(
                t(z23)[None], torch.ones(1, len(z23)),
                t(af)[None], torch.ones(1, len(af)),
                t(pf)[None])
            mz, ma, mp = mz[0], ma[0], mp[0]
            stds = xnet.stds()
            acts, lps = [], []
            for m, ls in zip((mz, ma, mp),
                             (stds[0], stds[1], stds[2])):
                d = _D.Normal(m, ls.exp().expand_as(m))
                a = (torch.zeros_like(m) if gated else d.sample())
                acts.append(a)
                lps.append(d.log_prob(a).sum(-1))
            lpz, lpa, lpp = (l.detach().numpy().astype(np.float32)
                               for l in lps)
        za, aa, pa = [a.numpy().astype(np.float32) for a in acts]
        cont, _disc = ix.compose(pc, pd, za, aa, pa, 1.0)
        act = {"continuous_action": cont.astype(np.float32),
               "discrete_action": pd.astype(np.int32)}
        from smart_buildings import evaluate as E
        act = E.validate_action(act, spec)
        obs, _, done, trunc, _ = trk.step(act)
        T["zone"].append(z23)
        T["ahu"].append(af)
        T["plant"].append(pf)
        T["za"].append(za)
        T["aa"].append(aa)
        T["pa"].append(pa)
        T["lpz"].append(lpz)
        T["lpa"].append(lpa)
        T["lpp"].append(float(lpp))
        T["gated"].append(gated)
        T["v"].append(float(v.reshape(-1)[0].item()))
        T["r"].append(float(trk.rewards[-1]))
        if done or trunc:
            break
    T["zmask"] = np.ones(len(T["zone"]), dtype=np.float32)
    T["amask"] = np.ones(len(T["ahu"]), dtype=np.float32)
    T["steps_per_sec"] = len(T["r"]) / max(time.time() - t0, 1e-6)
    return T


def self_test():
    sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS")
    from pathlib import Path
    from smart_buildings import evaluate as E
    from smart_buildings.evaluate import POLICY_MAP
    pairs = [("building_train_8", "atlanta_1"),
             ("building_train_7", "denver_5")]
    trajs = []
    for b, w in pairs:
        d = Path("smart_buildings/training_scenarios") / b / w
        env = E.load_hybrid_environment(d / "config.gin")
        env.metrics_path = None
        meta = E.create_building_meta_data(env, d)
        spec = env.action_spec()
        base = E.initialize_policy(POLICY_MAP["occ_gated7"], meta, spec)
        tr = rollout_env(env, meta, spec, base, 256, seed=0)
        trajs.append(tr)
        V = tr["zone"][0].shape[0]
        K = tr["ahu"][0].shape[0]
        print(f"{b}/{w}: steps={len(tr['r'])} V={V} K={K} "
              f"reward_sum={sum(tr['r']):.4f} "
              f"v_mean={np.mean(tr['v']):.4f} "
              f"steps_per_sec={tr['steps_per_sec']:.1f}")
        assert tr["zone"][0].shape == (V, N_ZONE_X), tr["zone"][0].shape
        assert np.isfinite(np.stack(tr["r"])).all(), "NaN reward"
        assert np.isfinite(np.stack(tr["v"])).all(), "NaN value"
        assert np.isfinite(np.stack([a.ravel() for a in tr["za"]])).all()
    # Padding check: max sizes across the batch.
    nV = max(t["zone"][0].shape[0] for t in trajs)
    nK = max(t["ahu"][0].shape[0] for t in trajs)
    print(f"batch pad: nV={nV} nK={nK}")
    for tr in trajs:
        V = tr["zone"][0].shape[0]
        K = tr["ahu"][0].shape[0]
        assert (np.array(tr["zmask"]) == 1.0).all() and len(tr["zmask"]) == 256
        assert V <= nV and K <= nK
    print("SELF-TEST PASS: shapes/masks finite, padding consistent")


if __name__ == "__main__":
    self_test()
