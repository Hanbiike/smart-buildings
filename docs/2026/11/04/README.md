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
shipped). Plan-19 CLOSED (P0-P10, RL line closed); plan-20 ran
P20-1..P20-11 (frozen-det vs occ_gated7 confirmed, seed rig,
PR#3, R1/R2/N1 levers, D4 closed). Kernel count: 31 valid
dirs + 1 empty (alexchilton_v46) + 1 via-get file = 33 entries.

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

### Plan 19/P6: H3 PI loop-state features (KILLED)

- 6-col surgery (`tools/skynet_surgery.py`, NEW_ZONE_INPUTS=6):
  plan-18 pair (err/5K, integral saturation) + P6 cols
  (PI-target offset from band centre /5K, freeze flag,
  pi_active bit, prestart-or-normal bit) + zeroed
  third zone head; `_cur_vav` loop context in
  `zone_pi.py:159,176` = D5 hook. Parity PASS
  (bit-identical, target exactly 0.0).
- Wiring check caught stale `N_ZONE_X=23` in
  `tools/skynet_rollout.py` (now 27; `z23`->`zx` + width
  assert); `masked_lp` docstring to [T,V,27]; collate
  width-agnostic confirmed; target head excluded from
  the ratio (`mz[..., :2]`). Self-tests PASS: rollout
  2x256, PPO, 72/72 units.
- 1-cycle trace (protocol-only): 2 P1 pairs x 2016,
  epochs 1, seed 11, honest -- free 0.390/0.314,
  v-pre 76.7->20.9, mean_kl 0.001002, clip 0.0000,
  dmu ~3e-2, per-head dmuz/dmua/dmup print. Trace
  cost 4,032 steps + self-tests 512.
- 3-cycle full-episode screen LANDED (exit 0):
  P1 pool (10 pairs, sha 058d815f0357) x 8640 x 3 =
  259,200 steps, 10 workers, seed 11, honest,
  out `/tmp/p6/screen.pt`. Mechanics healthy (KL-stop
  each cycle, clip 0.0, v-pre 79.9->29.3). Honest
  means: c0 -0.0109494, c1 -0.0109118 (+0.0000375,
  below floor, 6/10 better), c2 -0.0110379
  (-0.0000886 vs c0, 1/10 better). No monotone gain;
  c2 below behavior = I2 flat/drift signature.
- VERDICT: H3 KILLED per kill-gate at ~250k. Total P6
  sim 263,744 (screen + trace 4,032 + self-tests 512).
  Logged in plan-19.

### Plan 19/P7: H1 zone-target head (steps 0-4 done, H-GREEN, 750k awaits user go)

- Step 0 (D5 hook, read not assumed): `_cur_vav` at
  `zone_pi.py:159,176`; plan-18 "per-step offset dict"
  did NOT exist (added in step 1); `TARGET_K=1.0` +
  `BuildingIndex.target_offsets()` already in
  `features.py:58,140-142`; occ9 chain roots at
  `ZonePIPolicy`, but 3 dsat `_zone_target` overrides
  skip super() (found the hard way, see below).
- Step 1: `_zone_target_offset` dict + `_zone_target_shift`
  helper (in-band clip vs the effective MRO target)
  applied as an err delta in the `_command_zones` PI
  branch -- the single choke point covering the whole
  occ path. FIRST attempt hooked base `_zone_target`
  and was DEAD (dsat overrides bypass it); caught by
  a clean two-policy test (0.0) after a naive
  repeated-get_action probe faked liveness via PI
  integral drift (quantized 0.0074/call artifact).
  Repair: +1K -> max|dc| 0.147, empty-vs-empty 0.0.
- Step 2: third head into `masked_lp` (`mz[..., :3]`,
  ndim 3.0/zone) + delay-1 offset dict in rollout
  (two-pass get_action would double-advance the PI
  integral; policy sees the live offset via P6 col 2,
  keeping the extended state Markov). Rollout A/B
  DIVERGED: live -7.7521 vs lobotomized -7.6424.
  Gated/force_rule steps still record exact zeros.
- Step 3: ADAPTER_L2=1e-4 on the 896 surgery-adapter
  params via `l2_groups()` in both Adam sites
  (train + v-pretrain); plan fixes no number.
- Step 4: H-gate re-pass GREEN at the new setting
  (2 P1 pairs x 2016, seed 11): mean_kl 0.000630,
  clip 0.0000, v-pre 75.0->20.7, free 0.314+,
  per-head dmu printed, pool sha/E3/E5 train-only/
  monster-check/conc all green. 72/72 units + PPO/
  rollout/surgery self-tests PASS throughout.
- Cost ~21k probe sim steps (no verdicts). STOPPED
  per protocol: the ONE 750k (step 5) is a user
  branch choice. Logged in plan-19.
- USER GO received: ONE 750k RAN to completion
  (exit 0, 9 cycles x 86,400 = 777,600 steps, P1 pool
  sha 058d815f0357, seed 11, 10 workers, honest,
  `/tmp/p7/run750k.pt`). Mechanics healthy (KL-stop
  8/9, clip 0.0, v-pre 80.7->29.3). Train-pool
  honest: c0 -0.0110471 = BEST, c1..c8 all worse
  (c6 -0.0118862 worst, c8 -0.0114187); best-c0 =
  +0.0000000. Climates c8-c0: atl -0.00078, den
  -0.00031, fb -0.00080 (blowup), phx +0.00000, sg
  +0.00003 -- zero improve. All 3 gate clauses fail;
  no ckpt to judge (judging c0 ~= judging base).
- VERDICT: H1 KILLED (flat like I2; lever ceiling
  binds). Total P7 sim ~800k. Logged in plan-19.

### Plan 19/P8: H6 KL-anchor (SKIPPED, condition unmet)

- Opens ONLY if H2/H5 show signal but drift eats
  it. Both were killed signal-less (P3/P5), and P7
  shows drift is not the problem (policy moves dmu
  ~3e-2/cycle, reward does not). H6 closed. 0 sim.
  Logged in plan-19.

### Plan 19/P9: honest bill (GREEN)

- Sim ledger: P0 0; P1 172,800 (honest, sha
  058d815f0357, seed 11) + ~1k smoke; P2 0; P3 0;
  P4 34,560 (honest, seed 31) + ~1k smoke; P5 0;
  P6 263,744 (honest P1); P7 ~800,000 (777,600
  honest P1 + ~23k probes); P8 0. Total ~1.271M.
  Judge/b8 steps: 0 (no ckpt qualified).
- No promotion claim anywhere (all killed/
  skipped/unopened except offline-only H4), so the
  >1M-honest bar is vacuous. Logged in plan-19.

### Plan 19/P10: long-shots (KILLED 1/2, RL line closed)

- P10.1 BCOOLER-style AWR (`tools/awr_fit.py`, MC
  gamma=1, beta=std(A), w clip 20): weighted NLL
  0.0330->0.0330 FLAT, |dmu| ~1e-2 noise -- no
  offline movement, no screen qualifier (35k saved).
  Fixed `mc_returns` dropping gamma + `test_awr.py`
  (suite 74/74).
- HRL precondition fails: warm-up vs intraday
  Cohen's d = -0.077 (negligible) -- nothing to
  split. 0 sim.
- TD3+BC (`tools/td3bc_fit.py`, twin pooled Q,
  gamma 0.995, alpha 2.5): Q unconverged (TD
  wandering, qmean -1.2 vs true ~-30) but actor
  MOVED (zone max|d|=0.127). Screen (deterministic
  mean via new `deterministic` rollout flag, fb8+
  phx3, honest, seed 31, 34,560 steps): phx3
  +0.0007337 (above floor), fb8 +0.0001769 (below
  floor). Gate 2/2 fails -- P10 KILLED. First
  same-sign-positive RL signal noted for future
  research (not a promotion). Logged in plan-19.

### Deep research round 3 (post-RL) -> plan-20

- Workflow (5 angles + critic + gap + synthesis,
  status partial): distillation remainder (D4 + 3
  danielwuu probe specs with hooks), N1-N3 grounding
  (optimizer/seed/bill inventory, all bodies opened),
  TD3BC post-mortem ((a) determinism favored prior,
  (b) collapses into (a), (c) movement yes/help
  unproven + 2-step discriminating experiment),
  rule-side R1/R2/R3 (How/Gate/Kill/cost on occ9),
  LB/submission state. Critic caught overclaims
  (sign conventions, 17321-all-pools, distillation
  completeness, cross-climate transfer) -- plan-20
  corrects each. Named gaps kept in plan-20 §gaps.
- `docs/plan/20-rule-side-and-closures.md` (new):
  T post-mortem (P20-1 frozen-det, P20-11 BC0
  conditional), E enablers (P20-2 seed rig, P20-3
  PR#3 patch), D distillation (P20-4 D4), R levers
  (P20-5 R1 OFF, P20-6 R2 T&R, P20-8 R3a oracle,
  P20-13 R3b conditional), N finals (P20-7 N1.1,
  P20-9 N1.2, P20-10 N2.2, P20-12 N3.2).

### Plan 20/P20-1: frozen-det control (done, (a) confirmed)

- NEW `tools/frozen_det_screen.py` committed tool
  (serial deterministic-mean vs force_rule, fb8+
  phx3, --weights/--tag/--seed/--steps); --help +
  74/74 green. Screen landed (34,560 steps, honest,
  seed 31): frozen-det gaps fb8 +0.0016810, phx3
  +0.0007738. phx3 within floor of TD3BC; fb8 beats
  TD3BC 9.5x (training hurt there). (a)
  CONFIRMED stronger than predicted; (b)(c) dead;
  STOP RL training. Base arms bit-reproduce P10.
  P20-11 skipped (conditional). Logged in plan-20.

### Plan 20/P20-2: OCC_SEED rig (GREEN)

- Claim verified by reading: 17321 = ctor default
  of the only live occupancy class; gin binds only
  the timestamp; 511211 class dead; building-gen
  seeds are a separate axis.
- Rig: `OCC_SEED` env rebind in the loader (after
  parse, before construction) + banner in
  fullrun/fullrun_t + `tools/test_occ_seed.py` (3
  tests, suite 77/77). No regen needed.
- Liveness 3x1500: unset == 17321 bit-identical,
  99999 differs. (256 steps insufficient: weekend
  nights.) 4,500 probe sim. Enabler for P20-10;
  17321-only finals banned. Logged in plan-20.

### Plan 20/P20-3: PR#3-equivalent patch (GREEN)

- `_recirc_weight` helper (area default; airflow
  under PR3_AIRFLOW=1) wired at both recirc sites
  (4 lines). In-sim edit over monkeypatch (would
  duplicate 2 method bodies). Shipped default
  untouched.
- `tools/test_pr3_patch.py` (3 tests, suite 80/80).
  Liveness 2x1500: unset matches P20-2 baseline,
  flag differs. Semantics unverified (gap 1).
  3,000 probe sim. Enabler for P20-12. Logged in
  plan-20.

### Plan 20/P20-4: D4 error-reserve (CLOSED below floor)

- NEW `probe_errres025` (ErrResProbe on occ9) +
  POLICY_MAP key; 80/80 green; liveness PASS.
- Suite 60/60 clean (shipped, 518,400 sim): mean
  gap +0.00001827 (11x below floor); climates all
  sub-floor (fb best +0.0000635); 9W/15L/6 ties;
  worst b8/fb8 -0.000133. Gate fails all clauses
  -- CLOSE, no judge. occ9 control 0.00878193 ==
  D1-D3 (P7 zone_pi parity on train-30). Logged in
  plan-20.

### Plan 20/P20-5: R1 OFF levers (both CLOSED)

- NEW `occ_off1.py` (offa1 spike-override + offb1
  integral-reset) + POLICY_MAP keys; 80/80 green.
- R1b CLOSED pre-screen (0 sim): HWS reset
  integrator gain-0.0-disabled in occ9 (traced
  max 0.0 over 1500 fb8 steps; shortfalls to
  1.98K) -- nothing to reset.
- R1a CLOSED (no-op screen 4/4, 34,560 sim): both
  pairs bit-identical (fb8 0.027586, atl1
  0.006261). Mechanism live but never triggers:
  seed-17321 occupancy has no night spikes.
  Re-screen under fresh P20-2 seeds noted for
  P20-10. Logged in plan-20.

### Plan 20/P20-6: R2 self-tuning T&R (CLOSED gain-0.0)

- NEW `occ_ptr1.py` (PtrLoopMixin between
  DeadbandHeat and Gated7 in occ9 base list) +
  keys occ_ptr0/a1/b1/c1; 248/248 green.
- Base parity: occ_ptr0 == occ9 BIT-IDENTICAL
  (1500 fb8 steps).
- fb8 4-cell prescreen x 8640 (34,560 sim): all
  bit-identical (0.027586 x4).
- Fire-count scan 30/30: fire == 0 on EVERY
  train pair x 1500 steps. The T&R loop never
  fires under occ9 on train (dampers open but
  temps stay inside db_eff); b1/fb1+b2/fb2
  pressure lifts are heatopt's. All subs ==
  occ9 by construction; gain 0.0, no survivors.
  139,560 sim. Re-screen under fresh seeds /
  transfer tails noted for P20-10. Logged in
  plan-20.

### Plan 20/P20-8: R3a oracle (R3 KILLED UNBUILT)

- Menu v1+v2 died in liveness (MixedSat
  overwrites demand/forced SAT; raw recirc
  metric-dead at sim 0.70 clamp). Menu v3.1
  all 4 LIVE: MILD_SAT -1.0, pressure floor
  -0.6, econ clip-hi 0.7, WATER_SCALE 0.7.
- NEW `tools/oracle_bound.py` (96-step bins,
  best-of-5 incl. D4-occ9 control); 256/256
  green. 20/20 suite (172,800 sim): bound
  +0.000199 < 0.0005 bar (below floor too).
  P20-13 SKIPPED (condition unmet). 223,800
  sim. Logged in plan-20.

### Plan 20/P20-7: N1.1 hybrid build (done)

- 3 danielwuu pieces ported (OFF-candidate,
  pressure frac^2, water hysteresis) + stack;
  all liveness green (off on atl pairs).
- PG arm: probe_skynet frozen (liveness
  green). BO 20/20 (86.4k honest): best
  it04 sum 0.116293; tuned stack fixed
  consts verify EXACT. Ledgers: hybrid
  90,720 honest + 13,500 shipped, PG 0
  train (+1,500 live), equal budget. 263/263
  green. Finals (P20-9): tuned vs skynet.
  Logged in plan-20.

### Plan 20/P20-9: N1.2 finals (N1 KILLED)

- Judge-24 48/48 honest: tuned 0.020333 vs
  skynet 0.017712 (PG wins +0.002621, 13x
  floor, wrong sign). Comfort split fails
  too (all 3 components worse); skynet wins
  4/6 buildings. Holdout/b8 not run (gate
  decided). Compute stays rule-side.
  414,720 honest sims. Logged in plan-20.

### Plan 20/P20-10: N2.2 finals (VACUOUS, no survivors)

- Audit: D4/R1/R2 closed, R3/N1 killed --
  nothing to final. 0 sim. Frozen-det
  follow-up (+0.00168/+0.00077, above floor)
  queued as round-4 backlog (train-30 +
  judge + G8 + notebook feasibility), not a
  P20-10 rider. Logged in plan-20.

### Plan 20/P20-12: N3.2 bill/parity (done)

- B8 3-bill 40/40: honest delta occ9
  -0.000074 (Mima 0.00003 DROPPED as
  constant, policy-dependent +-0.0004);
  PR#3 delta +0.00003..0.00007 (below
  floor). Tuning re-KILL on corrected sim
  (+0.003791). Gateway parity holds by
  construction (gap 0).
- Judge 3-bill 168/168: honest delta occ9
  +0.000078 (pool sign flips, Mima stays
  dropped); PR#3 all below floor. Tuning
  re-KILL +0.005314; skynet transfer gap
  (b8 -0.000477, judge +0.002578). G8 all
  PASS (worst 1.896, moot). 1,797,120
  sims. PLAN-20 COMPLETE, no survivors.
  Logged in plan-20.

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

- plan-20 COMPLETE 11-05 (P20-1..13 done/closed/killed/
  skipped, no survivors; see `docs/2026/11/05/`).
- FOLLOW-UP closed as P21-1 (11-05): occ9-base screen
  KILLED (fb8 -0.003512 wrong sign, phx3 +0.000015
  floor) — P20-1 signal was a weak-base artifact.
- chaitanyajamble (0.01853) + royalacecat smartpolicy reads
  — weak, optional.
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
