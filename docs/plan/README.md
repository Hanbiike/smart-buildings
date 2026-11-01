# Plan: Online RL and rule-transfer work plan

Updated: **2026-11-01**. Status: draft. Companion research: `docs/research/11-17`.

Goal: beat the `occ_gated7` train-30 mean with a shipped-safe policy
(rule-first, residual-RL second), selected on an honest bill and a validated
second judge — never on a single building alone. Loss convention:
`abs(df.agent_reward.sum()) / len(df)` per run CSV over full 8640-step
episodes (30 days x 5 min). Coverage tags: train-30 (6 buildings x 5
climates), subset-N (named scenarios), transfer-12 (3 generated buildings x
4 unseen weathers), public (Kaggle leaderboard over 30 hidden pairs).

## Checklist index

| # | Checklist | Research | Status |
|---|---|---|---|
| A | [Algorithm selection](11-algorithms-checks.md) (residual SAC primary, TD3 backup, gated PPO + STOP rule) | [11 Online RL algorithms](../research/11-online-rl-algorithms.md) | Draft 2026-11-01 |
| B | [Architecture checks ARCH-01..24](12-architectures-checks.md) | [12 Online RL architectures](../research/12-online-rl-architectures.md) | Draft 2026-11-01 |
| C | [Training-logic checks](13-training-logic-checks.md) (adv norm, masking, DR pools, judge, LR schedule, billing) | [13 Online RL training logic](../research/13-online-rl-training-logic.md) | Draft 2026-11-01 |
| D | [Failure-mode and exploit checks](14-failures-checks.md) (cycling, collapse, authority, billing audit) | [14 Online RL failures and exploits](../research/14-online-rl-failures-exploits.md) | Draft 2026-11-01 |
| E | [Model-based checks](15-model-based-checks.md) (short rollouts, synth:real >= 10:1, physics-regularized dynamics) | [15 Online RL model-based](../research/15-online-rl-model-based.md) | Draft 2026-11-01 |
| F | [Skynet replication and transfer checks](16-skynet-replication-checks.md) | [16 Skynet residual-PPO notebook](../research/16-skynet-residual-ppo-notebook.md) | Draft 2026-11-01 |
| G | [Repo-grounding checks](17-grounding-checks.md) (invariants RL work must not break) | [17 Repo grounding for Online RL](../research/17-repo-grounding-online-rl.md) | Draft 2026-11-01 |

## Recommended execution order

1. F groups A+B (offline reconstruction + honest-bill decision) — no-sim,
   unblocks every training decision.
2. F group C (occ_gated7 baseline + cost map) — establishes the ladder entry
   all RL must beat.
3. A (algorithm selection + STOP rule) — picks SAC vs TD3 vs PPO before
   burning 5M-step runs.
4. F groups D+E (gated residual spike + trainer recipe) interleaved with B
   (architecture) — parity and feature/gating audits first, then first
   250k smoke, then full runs.
5. F group F (second judge) before promoting ANY checkpoint — no promotion
   on one building alone.
6. E (model-based / synthetic pools) once the model-free residual curve turns.
7. C, D, G guard throughout (training-logic, failure-mode, and
   repo-invariant checks apply to every run).

## Standing constraints (from track G)

- Full episodes only (8640 steps); short runs are not comparable.
- Keep total concurrent sims <= 10 with 1500s+ timeouts
  (`tools/fullrun.py`, `tools/fullrun_t.py`).
- Never edit policy code while a validation run using it is in flight.
- Assemble notebooks only via `tools/assemble_notebook.py`; local exec
  check must run 300+ steps.
- Record active days in `docs/YYYY/MM/DD/README.md` (rules R1-R9);
  committed docs are English only.

## Definition of done (whole plan)

- Every P0 item across tracks is green with command outputs pasted in day
  notes; `docs/ladder.md` records each promoted policy (R5); negative
  results are closed with numbers (R4).
- Final promotion rests on train-30 mean AND worst-climate screen AND the
  validated second judge — never on one building or the shipped bill alone.
