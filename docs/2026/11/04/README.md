# 2026-11-04

## Summary

Public-notebook distillation day: pulled all 31 competition
kernels + alex's dataset (fresh weights), verified the full
leaderboard (547 teams: Daniel Wu 0.01061 #149, alex 0.01107
#183, us 0.01236 #273; top-150 sub-0.0106 = exploit wave).
Mined alex-latest (§4d generated-judge, 720-pool, B1, ramp =
already ported J1) and the malyshevdanil rule line (shipped
zt-best constants); ported 3 probes (median-SAT, per-zone
u-shift, weekend setback) -- all train-30 CLOSED below the
0.0002 floor. Deep-read danielwuu's 0.01061 MPC-lite (decoded
archive, sha ok) and facilities v42-v46 (byte-identical).
Second deep-research workflow (arXiv/OpenReview/PwC/forum)
merged as plan-19 Addendum A + 2 draft corrections. Simulator
update check: our tree is pre-PR#1; PR #2 open; hosts aim final
verification at the corrected sim; no local update (LB still
shipped).

## Status

| Whose | Score | State |
|---|---|---|
| Daniel Wu (analytic) | 0.01061 | LB VERIFIED |
| alex RL (V23) | 0.01107 | LB |
| us (occ9) | 0.01236 | LB (Oct 28, stale) |
| exploit wave top | 0.00470 | LB (expect evaporation) |

## Done

### Distillation D1-D3: zt-best mechanisms on occ9 (all CLOSED)

- 3 probes on occ9 (`probe_modes.py`): `probe_medsat05`
  (median-demand SAT via new `DEMAND_QUANTILE` hook, None
  default), `probe_ushift06` (per-zone u-EMA economic target
  shift 0.6K/EMA144 via new `_cur_vav` loop context = the D5
  hook plan-19 P7 needs), `probe_wknd6` (weekend lo-edge -6K,
  pure probe-level). Base edits parity-checked (occ9
  bit-identical: fb8 0.021008, sg8 0.007944), 72/72 unit
  tests; liveness proven per probe (median hook via extreme
  quantile 0.0 on mixed climates after fb8/sg8 homogeneous
  identical).
- 120/120 train-30 + occ9 control 0.00878189 (== J4/J6):
  medsat05 -0.00002398 (-0.27%, 13W/14 ties -- median rarely
  flips the binary SAT), ushift06 +0.00006532 (+0.74%,
  denver +0.000248 kills it, b7/denver_5 +0.001535 worst),
  wknd6 -0.00013008 (-1.48%, 7W/23 ties/0 losses,
  fairbanks-driven: b8/fb8 -0.001440). ALL below the 0.0002
  floor -- no judge screen (F2, kp069/zssc05 precedent).
- VERDICT: no promotion. Details in
  `docs/research/21-notebook-distillation.md` §3/§5.

### Notebook inventory (31 kernels + dataset)

- `/tmp/kaggle_nb/` (31/31 OK) + `/tmp/sb_alex/` (dataset,
  TAG `m2_gated_from_l1w12@8011776`, 8.01M steps).
- alex-latest: §4d generated-judge (20->60, LB-validated gap
  0.000164 vs 0.01128), 720-pool trains better than 2700,
  B1 removes drift without gain, FA1 thermal storage NO NET
  GAIN, ramp already ported (C3/J1). Listing 0.01107 likely
  post-pull B1 score.
- sb-alex-rl CLOSED (28-cell §4c-era copy, nothing new);
  ship-a CLOSED (1-pair vs PID-water, ramp family);
  host notebook trivial; facilities v42-v46 byte-identical
  (= danielwuu + RLS-ID + error reserve, no LB score).
- danielwuu 0.01061 decoded: per-zone 6-dim thermal ID +
  online RLS + joint 11x2 AHU search (fan+chiller/3.5+0.3
  gas+65000 comfort) + OFF candidate; "obs delay" = standard
  MDP causality (verified `environment.py:1651+`), not a bug.
  Full port = rewrite; pieces queued (OFF-candidate,
  pressure~fraction^2, water hysteresis).
- Incident: `kernels get` overwrote
  `notebooks/smart-buildings-submission-gen31.ipynb` with
  alex-latest; restored from git, tree verified clean.

### Deep research round 2 (sources: arXiv/PwC/OpenReview/forum)

- 9 arXiv full texts + benchmark/challenge sweep + forum
  mining, merged as plan-19 Addendum A: H1/H3/H4 supported,
  H2 scope-weakened (Q-Prop CV is action-dependent), H5
  weakened (Ng1999 primary: undiscounted needs absorbing
  state), H6 schedule-only, H7 contested + numbering adopted;
  new N1 (planning beats PG), N2 (comfort+generalization,
  seed-17321 ban), N3 (honest-bill + gateway parity: ~3%
  local-vs-LB artifact EXCEEDS the floor). Fixed 2 draft
  errors (TRANSIC shared encoder + two-stage; reversed
  Sinergym citation). Gap-4: Q-Prop + Ng1999 closed; ARC,
  Biemann/Mahbod/Zhang, OpenReview wall, PR bodies open.

### Plan 19/P0: trace instrumentation (GREEN)

- `tools/skynet_ppo.py`: `ppo_epoch(trace=True)` now prints
  per-head `dmuz/dmua/dmup` + pre-clip `gpol/gval`
  (policy-vs-value grad shares via new `policy_params` /
  `_grad_norm` helpers; old `dmu` max kept). 1-mb unit check
  on a synthetic batch prints all fields (gval 3555 vs gpol
  2.45 -- value-drowning directionally confirmed on
  uncalibrated critic); PPO self-test PASS, 72/72 unit tests.
- VERDICT: P0 GREEN, 0 sim. Logged in plan-19.

### Plan 19/P2: H-gate re-pass rig (GREEN)

- All I1 boxes re-verified present in the current tree:
  KL-stop, E5 clip-abort, E3 best-ckpt + LR-decay, E4 pool
  sha (also persisted in P1 batch meta), `zone_count.py`
  (b8: 3 AHUs/30 zones/max 10, b7: 4/36/10 -- valid),
  train-only P1 pool (xfer holdout holds), conc 10,
  `--trace` with P0 fields, no policy edits in flight.
- No code changes needed; rig ready for any 750k at a new
  setting after its H-gate re-passes.
- VERDICT: P2 GREEN, 0 sim. Logged in plan-19.

### Plan 19/P1: persisted batches (GREEN)

- `tools/skynet_rollout.py`: `force_rule` threads through
  `rollout_env` / `_rollout_task` / `rollout_parallel`;
  `tools/skynet_train.py`: `--rule-only` (gate all steps,
  no update) + `--dump-batch` (collated F + bounds + meta:
  scenarios, pool sha, seeds, bill, weights, base).
- Smoke (2 pairs x 512, seed 7): fr/za/aa/pa exact zeros,
  meta ok; rollout self-test PASS, 72/72 unit tests.
- Real batches: `/tmp/p1/rule10.npz` (86 MB, free 0.000)
  + `/tmp/p1/actor10.npz` (103 MB, free 0.363, za_std
  0.185): b8x5+b7x5 full episodes, honest bill, seed 11,
  pool sha 058d815f0357. 172,800 sim steps at 10 workers.
- Regen: same two `skynet_train.py` commands with
  `--dump-batch` (rule: `--rule-only`; actor: `--cycles 1
  --epochs 1 --v-steps 10 --minibatch 1024`).
- VERDICT: P1 GREEN. Logged in plan-19.

### Plan 19/P3: H2 residual critic (KILLED)

- `tools/vrule_fit.py` (new): pooled-feature MLP (masked
  zone/ahu means + plant, 79 dims) on rule10 MC returns,
  held-out trajs: test MSE 21.05 vs shipped-critic-init
  26.7 (ratio 0.79, gate <= 0.5 MISSED) vs const-mean
  123.8. Saved `/tmp/p1/vrule.pt`.
- Decisive probe (P0 instrumentation on real actor10,
  post-v-pretrain-100: 79.67 -> 28.89): gpol=0.007,
  gval=4.570 -- 650x value-drowning CONFIRMED real.
- But the split fails its own gate: frozen V_rule scores
  47.21 on actor10 states (shift destroys the rule-fit
  baseline: 21.05 -> 47.21), and V_rule + V_res-MLP
  (100 steps, v-pretrain budget) combined = 38.60 vs
  single critic 28.89. No MSE drop.
- Mechanism kicker: Adam absorbs the global-clip scale
  (bias-corrected m-hat/sqrt(v-hat) is scale-invariant),
  so clip-crowding was never causal; gpol ~= nil absolute
  with dmu 3.2e-3 going nowhere = SIGNAL problem (fail-9
  noise diagnosis), not step size. H2's fix targets a
  symptom.
- VERDICT: H2 KILLED per kill-gate (no MSE drop). 0 sim.
  Logged in plan-19.

### Plan 19/P4: H4 confidence gate (SURVIVES)

- `tools/gate_fit.py` (new) + `GateMLP`: labels =
  1[MC_actor > MC_rule] on actor-free steps (same
  pool+seeds/trajs, time-aligned; non-circular by design).
  Label rate 0.294 (test 0.259); held-out AUC 0.7197
  (gate >= 0.65 PASS, deterministic re-run); open-frac
  0.599/0.476/0.353 at tau 0.3/0.5/0.7 (all in range).
  Weights `/tmp/p1/gate.pt`.
- Rollout wiring: `--gate/--gate-tau` through
  `rollout_parallel` (worker-cached load) + serial path;
  features-before-free reorder (pure reads moved, RNG
  order untouched). Smoke (2x512): free 0.314 -> 0.131,
  end-to-end ok.
- Online confirm (fresh seed 31, honest, full episodes,
  frozen actor, tau 0.5): fb8 gated -0.030149 vs ungated
  -0.031893 (+5.5%; gate opened 0.000 -- pure rule, the
  correct shutoff of harmful heating noise), phx3
  -0.007516 vs -0.007669 (+2.0%; opened 0.154).
  Gated >= ungated on BOTH pairs: PASS.
- VERDICT: H4 SURVIVES to P7. 34,560 sim steps.
  Logged in plan-19.

### Plan 19/P5: H5 PBRS shaping (KILLED)

- `tools/pbrs_probe.py` (new): Phi(s) = -k x mean zone
  violation from the batch's own band cols (z23 6:8),
  terminal Phi'=Phi (documents the no-absorbing-state
  violation), r' = r + g Phi' - Phi, per-traj norm like
  training. k=1.0: pre-norm std 1.7249 -> 1.7229 (ratio
  0.999, gate >= 1.30 MISSED), rank-corr 0.9974,
  sign-check backwards (good -0.43 vs bad -0.15, n
  114/65). k=5.0: ratio 1.020, corr 0.9657. Shaping is
  pure redundancy (cosine ~= 1 regime).
- Two probe bugs caught and fixed during the run:
  post-norm std is vacuous by construction (always 1.0);
  `normalize_adv` mutates its input in place through
  slice views -- copies required.
- VERDICT: H5 KILLED per kill-gate. 0 sim. Logged in
  plan-19.

### Simulator update check

- Our tree is pre-PR#1 (tracker occupancy RNG stream);
  starter-kit PR #1 merged Oct 19, PR #2 (tank) OPEN,
  PR #3 (return-air) referenced. Hosts (Oct 29): aim final
  verification at corrected sim, no firm plan; LB stays
  shipped. Decision: NO local update (keep local == LB);
  watch threads 749531/751998. occ9 exploits nothing.

## Measured

- medsat05 — train-30 — 0.00875791 — -0.00002398 vs occ9.
- ushift06 — train-30 — 0.00884721 — +0.00006532 vs occ9.
- wknd6 — train-30 — 0.00865181 — -0.00013008 vs occ9.
- alex v11 dataset weights — b8 shipped — 0.01026 (their
  table; board 0.01128).
- LB 2026-11-04 — 547 teams — danielwuu 0.01061 (#149),
  alex 0.01107 (#183), us 0.01236 (#273), p10 0.00746,
  p50 0.01244.

## Decisions

- Claim. Proof: numbers/file. Consequence: ship/close/probe.
- Promotion rule (plan F/F3): no checkpoint promoted on b8 or
  train alone — judge-first (F1 pool, 24 pairs, sha256 in
  `docs/2026/11/02/`), b8-second; gaps below the 0.0002 floor
  do not resolve (F2).
- D1-D3: all below floor. Proof: §Done numbers. Consequence:
  close, no judge screen; wknd6's one-sidedness (7W/0L)
  noted for future combined probes, not acted on.
- Local sim stays pre-PR#1. Proof: no host update
  announcement; LB on shipped. Consequence: no re-download
  until hosts announce; re-validate occ9 then.

## Open

- D4 error-reserve analogue probe (facilities v42 idea on
  occ9) — queued, not started.
- danielwuu pieces (OFF-candidate, pressure~fraction^2,
  water hysteresis) — queued, not started.
- chaitanyajamble (0.01853) + royalacecat smartpolicy reads
  — weak, optional.
- plan-19 P0-P2 execution (H1-750k rails) — not started.
- occ9 LB submission is Oct-28 stale; no new submission
  (nothing beat occ9 locally).

## Appendix

- D1-D3 per-climate gaps: medsat05
  atl -0.000032 / den -0.000045 / fb -0.000010 /
  phx -0.000032 / sg +0.000000;
  ushift06 atl +0.000002 / den +0.000248 / fb -0.000035 /
  phx +0.000108 / sg +0.000004;
  wknd6 atl -0.000017 / den -0.000006 / fb -0.000627 /
  phx +0.000000 / sg +0.000000.
- Suite: `tools/fullrun.py /tmp/dist1 8640 8 1500
  probe_medsat05,probe_ushift06,probe_wknd6,occ_econdbhhot9
  <train-30>`, 120/120.
- Leaderboard CSV: `/tmp/lb/*.csv` (547 rows, Nov 4).
- Kernels: `/tmp/kaggle_nb/` (31 dirs); alex dataset:
  `/tmp/sb_alex/`; danielwuu policy: `/tmp/dw_policy.py`.
