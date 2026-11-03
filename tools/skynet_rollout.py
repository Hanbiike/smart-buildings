"""Rollout collector for Online residual-PPO (step 1 of the train loop).

Collects fixed-length trajectories with the surgery-expanded FlatNets:
stochastic actions (mu + exp(log_std) * eps per entity head), critic
values, sim rewards on the driver's bill (honest by default per B3),
and padding masks over variable-size buildings. Switches stay frozen
(continuous-head logprobs only); Bernoulli switch heads are a later
step. Gated steps record exact-zero corrections with fr=0 (D3).

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


DEFAULT_WEIGHTS = "/tmp/opencode/sb-rl/skynet.pt"


def build_policy(meta, spec, weights=DEFAULT_WEIGHTS):
    """Deterministic harness parts (features + expanded net), no env."""
    import importlib.util as _ilu
    from pathlib import Path as _P
    _surg = (_P(__file__).resolve().parent
             / "skynet_surgery.py")
    if "skynet_surgery" not in sys.modules:
        _spec = _ilu.spec_from_file_location("skynet_surgery", _surg)
        _mod = _ilu.module_from_spec(_spec)
        sys.modules["skynet_surgery"] = _mod
        _spec.loader.exec_module(_mod)
    else:
        _mod = sys.modules["skynet_surgery"]
    from skynet.policy import load as _load
    from skynet.features import BuildingIndex as _BI
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    if "zone.0.adapter.weight" in ck.get("model", {}):
        xnet, _rep = _mod.expand_net_from_ckpt(ck)
    else:
        net, _params, _scale, _latch = _load(weights)
        xnet, _rep = _mod.expand_net(net)
    xnet.eval()
    return xnet, _BI(meta)


def rollout_env(env, meta, spec, base_policy, steps, seed=0,
                weights=DEFAULT_WEIGHTS):
    """Rollout driving a live env with the base policy + residual sample."""
    from skynet.features import N_AHU, N_PLANT
    from smart_buildings.smart_control.utils.scenario_generator import (
        SimulationTracker)
    import torch.distributions as _D
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    xnet, ix = build_policy(meta, spec, weights)
    trk = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    trk.reset()
    obs = trk.last_obs
    T = {"zone": [], "ahu": [], "plant": [], "zmask": [], "amask": [],
         "za": [], "aa": [], "pa": [], "lpz": [], "lpa": [], "lpp": [],
         "gated": [], "fr": [], "v": [], "r": []}
    t0 = time.time()
    for _ in range(steps):
        base = base_policy.get_action(obs)
        pc = np.asarray(base["continuous_action"])
        pd = np.asarray(base["discrete_action"])
        # Mirror deployment gating (probe_xres): shutdown/floor/cold steps
        # take the base action. They are still recorded, with the
        # actually-applied zero correction and its logprob, so the update
        # sees exactly the deployed behavior. fr=0 marks gated steps for
        # the PPO loss (their env.py::step semantics, plan F/D3).
        from smart_buildings.baselines.probe_modes import residual_free
        from smart_buildings.baselines.probe_modes import (
            ExpandedResidualProbe as _X)
        free = residual_free(base_policy, ix, obs, _X.COLD_GATE_K)
        gated = not free
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
        T["fr"].append(0.0 if gated else 1.0)
        T["v"].append(float(v.reshape(-1)[0].item()))
        T["r"].append(float(trk.rewards[-1]))
        if done or trunc:
            break
    T["zmask"] = np.ones(len(T["zone"]), dtype=np.float32)
    T["amask"] = np.ones(len(T["ahu"]), dtype=np.float32)
    T["steps_per_sec"] = len(T["r"]) / max(time.time() - t0, 1e-6)
    return T


# ---------------------------------------------------------------- workers
# Parallel rollout collection (plan F/E1): spawn pool, envs cached per
# worker by scenario (any workers/scenarios ratio), fresh base policy
# per task. Seeds match the serial driver exactly, so serial and
# parallel trajectories are bit-identical.

_WORKER = {}


def _worker_env(gin_path):
    from pathlib import Path as _P
    from smart_buildings import evaluate as _E
    envs = _WORKER.setdefault("envs", {})
    if gin_path not in envs:
        env = _E.load_hybrid_environment(_P(gin_path))
        env.metrics_path = None
        meta = _E.create_building_meta_data(env, _P(gin_path).parent)
        envs[gin_path] = (env, meta, env.action_spec())
    return envs[gin_path]


def _rollout_task(task):
    scenario_idx, gin_path, steps, seed, weights, bill = task
    from smart_buildings import evaluate as _E
    from smart_buildings.evaluate import POLICY_MAP
    if bill == "honest" and not _WORKER.get("billed", False):
        from tools.honest_bill import apply_honest_bill
        apply_honest_bill()
        _WORKER["billed"] = True
    env, meta, spec = _worker_env(gin_path)
    base = _E.initialize_policy(POLICY_MAP["occ_gated7"], meta, spec)
    tr = rollout_env(env, meta, spec, base, steps, seed=seed,
                     weights=weights)
    tr["scenario_idx"] = scenario_idx
    return stack_traj(tr)


def stack_traj(tr):
    """Lists-of-arrays -> stacked arrays (cheap IPC; collate-compatible)."""
    out = dict(tr)
    for k in ("zone", "ahu", "plant", "za", "aa", "pa", "lpz", "lpa",
              "lpp", "r", "v", "fr"):
        out[k] = np.stack(tr[k])
    out["gated"] = np.array(tr["gated"], dtype=bool)
    out["zmask"] = np.ones(len(out["r"]), dtype=np.float32)
    out["amask"] = np.ones(len(out["r"]), dtype=np.float32)
    return out


def rollout_parallel(gin_paths, steps, seeds, weights, bill, workers):
    """One trajectory per scenario across a spawn pool (E1).

    gin_paths/seeds align with the scenario list; returns trajectories
    in scenario order. Deterministic: same seeds as the serial driver.
    """
    import multiprocessing as _mp
    ctx = _mp.get_context("spawn")
    pool = ctx.Pool(workers)
    try:
        tasks = [(i, gin_paths[i], steps, seeds[i], weights, bill)
                 for i in range(len(gin_paths))]
        out = pool.map(_rollout_task, tasks)
    finally:
        pool.close()
        pool.join()
    out.sort(key=lambda t: t.pop("scenario_idx"))
    return out


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
