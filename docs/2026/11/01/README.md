# 2026-11-01

## Summary

Online-RL research day, no simulator runs. Two deep-research workflow runs
collected discovery-grade web evidence on residual/model-free/model-based
HVAC RL (both verification stages returned `no_evidence`, so all web
numbers are read pages/PDFs, not verifier-confirmed claims).
Reverse-engineered the public residual-PPO notebook and dataset added to
the repo root (`yes-rl-can-run-a-building-residual-ppo.ipynb`, `rl.zip`,
untracked): shipped `m3r_ramp_cmagen_seed1@752640`, public 0.01128,
honest-bill gain -10.6% vs shipped -16.9% (tank billing bug is ~39% of
the headline gain). A 9-agent documentation workflow (scout + 7 writers +
synthesis) wrote `docs/research/11-17` (2090 lines) and `docs/plan/`
(7 checklists + README, 678 lines); all 14 files merged from isolated
worktrees (track F wrote shared directly), verified English-only with
resolving links. No scores, no submissions, no policy changes.

## Status

| Whose | Score | State |
|---|---|---|
| leader (Joe Penn) | 0.00521 | LB |
| us (occ_gated4) | 0.01323 | COMPLETE (best, tied) |
| us (occ_gated7) | 0.01323 | COMPLETE (tied 10-27) |
| us (occ_gated6) | 0.01337 | COMPLETE (+1.1% regression) |

No new scores. Table carried from 2026-10-28; occ8/occ9 submitted,
scores pending (journal-reported, no fresh query). External reference
(not ours): skynet residual-PPO public 0.01128 (notebook-reported).

## Done

### 23:30-23:45 — deep-research v1 + repo inventory

- Launched `deep-research` (args: SOTA architecture/training for Online
  RL HVAC) and inventoried the repo in parallel: `POLICY_MAP` in
  `smart_buildings/evaluate.py:218`, `occ_gated7.py`, `nn_model.py`
  (`FeatureContext` + `ControllerNet`), `tools/slot_ppo.py`,
  `tools/bc_train.py`, `docs/ladder.md`, `docs/research/04,05,06,09,10`.
  Why: v1 workflow is web-only, so repo grounding was done inline.
- Synthesized interim recommendation (residual over occ_gated7, SAC
  default) from run-1 discovery + inventory. Superseded by the
  two-track verdict below.

### 23:32-23:55 — skynet notebook + dataset study

- Extracted `rl.zip` to `/tmp/rl_zip` (22 files) and read all 28
  notebook cells: residual PPO over `zt_off`, FlatNets d64x2 (~48k
  params), zone-target head +-1K, PPO 512x6/lr 3e-4/gamma 0.995,
  per-worker adv norm, free-masking, honest bill, 20-generated
  checkpoint judge. Why: first proven residual-RL precedent at
  public 0.01128, -14.8% vs our 0.01323.
- Corrected the interim synthesis: PPO is viable as a residual
  fine-tuner of a strong rule (not from scratch); keep residual-PPO
  and residual-SAC as two tracks.

### 23:45-00:15 — deep-research v2 (new prompt)

- Wrote a 5-question English prompt (algorithms / architectures /
  training logic / failures / model-based, numbers+settings+limits
  required) and launched `deep-research` v2. Discovery returned
  Sinergym/BOPTEST/CityLearn/B2B/NOMAD-RL/HVAC-DPT/ADAPT/LDLP/
  Counter-Dyna numbers with limits; verification again `no_evidence`.
  Full per-question synthesis delivered in chat.

### 23:55-00:40 — multi-agent documentation + merge

- Ran `document-online-rl-research-and-plan` (scout + tracks A-G +
  synthesis; 6 writers isolated, track F shared): produced
  `docs/research/11-online-rl-algorithms.md` through
  `docs/research/17-repo-grounding-online-rl.md` and `docs/plan/11-17`
  checklists. Script-side assembly under-reported (large payloads
  spilled; D/G notes null) but all files landed in retained
  worktrees and were verified on disk.
- Merged 12 files from worktrees, wrote `docs/plan/README.md`,
  added the Online RL series section to `docs/research/README.md`,
  verified Cyrillic-clean + resolving links. Why: durable,
  checkable record of everything the day established.

## Measured

No new simulator measurements (no runs launched). Notebook-reported
numbers (third-party, not reproduced here):

- skynet residual-PPO — public — 0.01128 — -14.8% vs our occ4 0.01323.
- skynet 5.0M — held-out-b8-shipped-bill — 0.01040 vs rule 0.01251 (-16.9%).
- skynet 5.0M — held-out-b8-honest-bill — 0.01079 vs rule 0.01207 (-10.6%).
- skynet 5.0M — held-out-b8-PR#2-sim — 0.01072 vs rule 0.01178 (-9.0%).
- skynet ship — held-out-b8 — 0.01026; board/holdout ratio 1.05-1.16.
- Documentation: 14 files, 2768 lines total (research 2090 + plan 678).

## Decisions

- Residual-over-strong-rule is the sanctioned RL shape. Proof: skynet
  0.01128 public with zero-init residual + gating; RL-on-weak-base
  fails (their pid_water 0.03184). Consequence: RL work starts from
  `occ_gated7`, never from scratch; see `docs/plan/README.md` order.
- Two algorithm tracks, not one. Proof: PPO precedent (0.01128) vs
  SAC boundary-robustness literature. Consequence: track A runs
  residual-PPO (primary, CPU throughput) and residual-SAC
  (challenger); STOP rule if no residual beats occ_gated7 train mean.
- Switches stay with the rules. Proof: learned on/off killed
  Fairbanks (0.026 to 0.56); BOPTEST PPO cycles 2-4x over baseline.
  Consequence: plan D keeps discrete authority rule-side; votes at
  most, with latch.
- Tank-billing audit is P0 before any training decision. Proof: ~39%
  of skynet's headline gain is the setpoint-vs-actual bug; no public
  documentation of this exploit exists. Consequence: plan F groups
  A+B run before trainer work; train bill decided explicitly.
- No checkpoint promoted on one building. Proof: building-8-only
  selection picked overfit checkpoints (m4 beats m3r on b8, loses
  on board 0.01150 vs 0.01132). Consequence: generated-buildings
  judge (transfer-12 to 60) gates every promotion.
- Diffusion inference and MCTS planning are deployment-excluded.
  Proof: 150ms/step on A100 (ADAPT), 100 sims/decision (LDLP).
  Consequence: phase-2 model-based is Dyna-style short rollouts
  (synth:real >= 10:1) or offline teacher only.

## Open

- From 10/28 (research update reconciled): DONE for the October
  series; the new 11-17 series extends it with Online-RL-only scope
  and does not close the collection-plan items.
- From 10/28 (collection plan: run identity, full-suite/transfer
  outcomes, night tails, counterfactual accuracy): MOVED, now
  itemized as executable checks in `docs/plan/` (tracks C, D, G).
- From 10/28 (live status, unread bodies, field constraints as gaps):
  OPEN; today's web evidence is discovery-grade and inherits the
  same exclusion rule (report 09 standard).
- Execute plan P0s in `docs/plan/README.md` order (F-A/B first).
- Resolve the train-bill decision (honest vs shipped) after the
  billing audit; our policies' tank-setpoint behavior is unmeasured.
- Close the loop on `tools/rl_train.py` skeleton (track C checks)
  only after the audit + judge exist.
- occ8/occ9 public scores still pending; reconcile on arrival.

## Appendix

File registry (all English-only, links resolve; line counts via `wc -l`):

| File | Lines | Contents |
|---|---|---|
| `docs/research/11-online-rl-algorithms.md` | 305 | PPO/SAC/TD3/recurrent per-benchmark evidence |
| `docs/research/12-online-rl-architectures.md` | 325 | 23 findings A1-E7 + 8 open questions |
| `docs/research/13-online-rl-training-logic.md` | 134 | Adv norm, masking, DR, judges, LR, billing |
| `docs/research/14-online-rl-failures-exploits.md` | 289 | Cycling, collapse, authority, exploits |
| `docs/research/15-online-rl-model-based.md` | 163 | ADAPT/LDLP/Counter-Dyna/hypernet/DreamerV3 |
| `docs/research/16-skynet-residual-ppo-notebook.md` | 507 | Notebook+dataset reconstruction |
| `docs/research/17-repo-grounding-online-rl.md` | 367 | Repo constraints bounding RL design |
| `docs/plan/11-algorithms-checks.md` | 105 | Algorithm selection checks + STOP rule |
| `docs/plan/12-architectures-checks.md` | 70 | ARCH-01..24 |
| `docs/plan/13-training-logic-checks.md` | 101 | Trainer-recipe checks |
| `docs/plan/14-failures-checks.md` | 69 | Safety/anti-exploit checks |
| `docs/plan/15-model-based-checks.md` | 84 | Phase-2 model-based checks |
| `docs/plan/16-skynet-replication-checks.md` | 105 | Replication groups A-G, P0/P1/P2 |
| `docs/plan/17-grounding-checks.md` | 85 | Repo-invariant checks |
| `docs/plan/README.md` | 59 | Index, execution order, whole-plan DoD |
| `docs/research/README.md` | +21 | Online RL series section |

Workflow notes: `deep-research` v1 (Russian prompt) and v2 (English
5-question prompt) both completed with discovery payloads and
`no_evidence` verification — the synthesis answers in chat carry the
numbers with per-claim sources. Doc workflow
`document-online-rl-research-and-plan`: 9 agents, 6 isolated worktrees
retained under `.muse/worktrees/` (dirty, not auto-applied; merged
manually via `cp` after on-disk verification). Script-side
`complete=false` for all tracks was a spill artifact (payloads over
the inline limit), contradicted by the landed files; synthesis
correctly flagged the gap before the merge and its README was
adapted to VERIFIED state after the merge.

R1-R9 tick: R1 deltas on all comparisons; R2 coverages labeled
(notebook-reported marked third-party); R3 all paths/commands exist
(worktree paths under `.muse/` are session-local); R4 no new
negatives (no runs); R5 no new policy (no ladder entry due); R6 no
submissions sent; R7 session times are same-day local, no
world-facing events; R8 unverified numbers labeled (discovery-grade,
notebook-reported); R9 English-only (Cyrillic check clean).
