"""Oracle action search from step N over horizon H (teacher-seeded shooting).

Usage:
  oracle_search.py <pair> <teacher_key> <N> <H> <K> [--noise X] [--seed S]
                   [--out path.json]

Method: roll the teacher 0..N, snapshot (tools/ckpt.py), then evaluate K
candidates over N..N+H. Candidate 0 is pure teacher (sanity: must exactly
match a plain rollout); candidates 1..K-1 add Gaussian noise (scale --noise)
to the teacher's CONTINUOUS action each step; discrete actions stay at the
teacher's (v1 limitation). Score = mean(|reward|) over H, same convention
as the leaderboard loss.

Outputs JSON: teacher_H_loss, best_H_loss, headroom_pct, per-candidate
losses, and the best candidate's first action (oracle action at step N);
plus an .npz with the oracle cont/disc actions for BC/DAgger training.

Cost per query: prefix N steps + K * (restore ~1s + H steps). Deterministic
given --seed (candidate k uses rng fork seed+k).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.ckpt import reseed_convection, restore, snapshot  # noqa: E402
from smart_buildings.evaluate import (  # noqa: E402
    POLICY_MAP,
    create_building_meta_data,
    get_policy_action,
    initialize_policy,
    load_hybrid_environment,
    validate_action,
)
from smart_buildings.smart_control.utils.scenario_generator import (  # noqa: E402
    SimulationTracker,
)


def make_run(pair: str, policy_key: str, scenarios_dir: str = "training_scenarios"):
    b, w = pair.split("/")
    input_dir = ROOT / "smart_buildings" / scenarios_dir / b / w
    env = load_hybrid_environment(input_dir / "config.gin")
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, input_dir)
    policy = initialize_policy(POLICY_MAP[policy_key], meta, env.action_spec())
    tracker.reset()
    return env, tracker, policy


def rollout(env, tracker, policy, steps, noise_fn=None):
    """Roll `steps`; noise_fn(cont, t) -> perturbed cont or None for teacher."""
    obs = tracker.last_obs
    spec = env.action_spec()
    losses, first_action = [], None
    for t in range(steps):
        raw = get_policy_action(policy, obs)
        cont = np.asarray(raw["continuous_action"], dtype=np.float64)
        if noise_fn is not None:
            cont = np.clip(noise_fn(cont, t), -1.0, 1.0)
        action = validate_action(
            {"continuous_action": cont,
             "discrete_action": raw["discrete_action"]}, spec)
        if t == 0:
            first_action = (np.asarray(action["continuous_action"]),
                            np.asarray(action["discrete_action"]))
        obs, _, term, trunc, _ = tracker.step(action)
        losses.append(abs(float(tracker.rewards[-1])))
        if term or trunc:
            break
    return float(np.mean(losses)), first_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("teacher_key")
    ap.add_argument("N", type=int)
    ap.add_argument("H", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenarios-dir", default="training_scenarios")
    a = ap.parse_args()
    if a.N + a.H > 8640:
        ap.error("N + H must be <= 8640 (episode length)")

    t0 = time.time()
    env, tracker, policy = make_run(a.pair, a.teacher_key, a.scenarios_dir)
    rollout(env, tracker, policy, a.N)  # prefix to step N
    master = snapshot(env, tracker, policy)
    t_prefix = time.time() - t0

    # Sanity: uninterrupted teacher over H (no restore in the middle).
    teacher_loss, _ = rollout(env, tracker, policy, a.H)

    cand_losses, first_actions = [], []
    stream = (a.seed * 1000003) ^ (a.N * 9176 + 11)
    for k in range(a.K):
        e, tr, p = restore(master)
        reseed_convection(stream)  # same shuffle stream for every candidate
        if k == 0:
            loss, first = rollout(e, tr, p, a.H)
        else:
            rng = np.random.default_rng(a.seed + k)
            loss, first = rollout(
                e, tr, p, a.H,
                noise_fn=lambda c, t: c + rng.normal(0.0, a.noise, size=c.shape))
        cand_losses.append(loss)
        first_actions.append(first)

    # In-harness restore validation: candidate 0 rides a reseeded shuffle
    # stream, so it matches the plain teacher up to ~1e-12 relative (proven
    # symmetric under restore: two restores with the same reseed are exact).
    denom = max(abs(teacher_loss), 1e-12)
    rel = abs(cand_losses[0] - teacher_loss) / denom
    assert rel < 1e-4, (
        f"restore mismatch: {cand_losses[0]} vs {teacher_loss} (rel {rel})")

    best = int(np.argmin(cand_losses))
    headroom = (teacher_loss - cand_losses[best]) / teacher_loss * 100.0
    result = {
        "pair": a.pair, "scenarios": a.scenarios_dir,
        "teacher": a.teacher_key, "N": a.N, "H": a.H,
        "K": a.K, "noise": a.noise, "seed": a.seed,
        "teacher_H_loss": teacher_loss,
        "best_H_loss": cand_losses[best],
        "headroom_pct": headroom,
        "best_k": best,
        "cand_losses": cand_losses,
        "prefix_seconds": t_prefix,
        "total_seconds": time.time() - t0,
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k != "cand_losses"}, indent=2))
    if a.out:
        out = Path(a.out)
        out.write_text(json.dumps(result, indent=2))
        all_cont = np.array([f[0] for f in first_actions])
        all_disc = np.array([f[1] for f in first_actions])
        np.savez(out.with_suffix(".npz"),
                 oracle_cont=first_actions[best][0],
                 oracle_disc=first_actions[best][1],
                 teacher_cont=first_actions[0][0],
                 teacher_disc=first_actions[0][1],
                 all_first_cont=all_cont,
                 all_first_disc=all_disc,
                 cand_losses=np.array(cand_losses))
        print(f"wrote {out} + {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
