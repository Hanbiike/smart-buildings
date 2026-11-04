# Residual-RL unlock: learning on top of a near-optimal rule base

Date: 2026-11-04. Status: draft.
Scope: make residual policy-gradient learn on top of occ9 (public
0.01236) after I2 ran 3.02M steps flat (best = cycle 0) with healthy
PPO mechanics. Source: deep-research workflow 2026-11-04 (5 angles +
critic + synthesis; report in session tool-outputs), grounded in
`docs/research/16-skynet-residual-ppo-notebook.md`,
`docs/2026/11/03/README.md` (I2/I3/J1-J6, MoE T1-T3), and the skynet
stack (`tools/skynet_ppo.py`, `tools/skynet_train.py`,
`tools/skynet_rollout.py`, `tools/skynet_surgery.py`).
Coverage tags: offline (no sim), probe (1-2 pairs x 512-2016 steps,
epochs=1, single cycle: KL/clip protocol ONLY, never promotion),
full-episode (8640 steps; short runs are not comparable),
train-pool (logged-sha pair set), judge-24 (xfer_5..10 x 4 weathers,
sha e71ea1a7...). Loss convention:
`abs(df.agent_reward.sum()) / len(df)` per run CSV.
Rules: no 750k run until the H-gate re-passes AT THE NEW SETTING (new
head + new features invalidate H-reuse); no promotion except
judge-first/b8-second with the 0.0002 floor (F3) and the G8 3x
screen; conc <= 10, timeouts 1500/5400; pool sha logged per run.

## Group P. Instrumentation and persisted batches (P0, before everything)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P0 | Per-head \|dmu\| + per-group grad-norm instrumentation | extend the `ppo_epoch` trace in `tools/skynet_ppo.py:147-200` (today `dmu`/`dls` are max-only at :190-193; loss is shared at :174 with global clip 1.0 at :177): print per-head dmu (zone/AHU/plant/target) + policy-vs-value grad-norm shares; unit-check on 1 minibatch | per-head dmu + pol/vf grad shares printable; H1/H2 gates become measurable (today they are not) | P0 |
| P1 | Fresh-batch rollouts exist and persist | run rule-only + frozen-actor full-episode dumps with logged seeds/bill into a committed dump dir; record pool sha | persisted batches for H2/H4/H5 offline tests; ~10-20 x 8640 steps. No I2 batch survives (never persisted), so every "offline on a logged batch" test starts here | P0 |
| P2 | H-gate re-pass rig ready | KL/clip/E3-E5/xfer-exclusion/monster-check/conc/pool-sha checklist per plan-18 I1; H-green required before any 750k at the new setting | rig green on probes before P7 opens | P0 |

## Group Q. Cheap offline kills (parallel on the P1 batch)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P3 | H2 residual-critic offline: frozen V_rule control variate | fit a small V_rule MLP on rule-only rollouts; values into `compute_gae` become `V_rule.detach()+V_res` (`tools/skynet_ppo.py:89-104`); value loss on V_res only; `v_pretrain` fits V_res on (MCret - V_rule); 1-cycle `--trace` probe. NOTE: differs from closed paired-reward (fail-4/G7: coupled trajectories, 2x sim) by state-matched frozen baseline; distribution-shift + bootstrap caveats apply | value-MSE down >= 2x AND policy-grad share up (P0-measured, not assumed) AND full-episode screen >= behavior; else KILL H2 (critic was never the bottleneck) | P1 |
| P4 | H4 confidence gate offline (redesigned, non-circular) | gate SHARES the residual feature encoder with a separate head + TWO-STAGE training (freeze encoder+action head, then train gate; joint is suboptimal -- TRANSIC full text, Addendum A correction); labels NOT raw GAE sign (I2 proved advantages noise) -- use held-out positive-return free steps from a FRESH frozen-actor rollout or rule-vs-residual counterfactual; deploy as rollout-side `residual_free` extension fixed BEFORE PPO (policy-dependent masks break the PPO ratio) | held-out AUC >= 0.65 AND fraction-open in (0.1, 0.6); online confirm: gated >= ungated full-episode on 2 pairs; else KILL H4, keep fixed `residual_free` gating | P1 |
| P5 | H5 PBRS comfort shaping offline (speculative) | recompute advantages on the FRESH P1 batch with/without shaping: `r' = r + g*Phi(s') - Phi(s)`, `Phi(s) = -k*mean_zones max(0, lo-T, T-hi)` with DECLARED band; report rank-corr, shaped adv-std on free steps, sign-check vs a known-good action. CAUTION: reward is already dense (comfort slope dominates per-step); LB is undiscounted mean but GAE uses gamma 0.995, which breaks the PBRS invariance theorem | adv-std up >= 30% with rank-corr >= 0.7 + sign-check + full-episode >= behavior; else KILL H5 (rank-corr < 0.5, or cosine ~= 1 yet still flat = pure noise) | P2 |

## Group R. Online screens (cheapest surviving first)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P6 | H3 PI loop-state features screen | AdapterFirst zero-init surgery (bit-identical start, scale-0 parity): 2 zone cols (PI-target offset from band centre /5K; freeze flag 0/1 from the `zone_pi.py:220-226` predicate) + minimal global mode cols; 1-cycle trace (KL/clip) then full-episode 3-cycle screen. Live scope is down-shift <= 1.0K only: night-drop/floor-window steps are fr=0 masked, not PG. CAUTION: competitor item-40 (b8-win/gen-20-loss) is a direct overfit warning for these exact signals | full-episode >= behavior + monotone over 3 full-episode cycles (NOT 2016-step mean_r); else KILL H3 by ~250k full-episode steps, or on the item-40 signature (b8-win + judge-24 loss) -> redirect to the lever side | P1 |
| P7 | H1 zone-target head: build, H-green, then ONE 750k | (0) confirm the D5 hook location in code (plan-18 asserts `_cur_vav` + per-step offset dict; NO file:line inspected yet -- find it first); (1) add `_zone_target_offset` + in-band clip on the occ path; (2) wire the third zone head in `masked_lp` (`mz[...,:3]`, `tools/skynet_ppo.py:51-81`) + rollout compose + zero-init; (3) PI target+integral features with L2/dropout (item-40 guard); (4) H-gate re-pass at the new setting; (5) ONE 750k on the logged-sha pool with holdout. Budget: H-probes + 750k train + judge-24 (207k) + b8 (43k) per ckpt; ~1M+ honest. Mechanism: the only channel the PI loop TRACKS instead of cancelling (competitor Fairbanks 0.02563 vs 0.02633, Phoenix ~0.0101 vs 0.01050; our J6 rule analogue 29/30 same-sign, gap -0.00011144 sub-floor). Cap acknowledged: item-20 best-3avg null + item-35 at-ceiling + J6 sub-floor -- this unlocks learning, not a jackpot | H-green + full-episode train-pool >= cycle-0 by >= 0.0002 AND >= 2 climates improve + worst-climate screen (no Fairbanks blowup); else KILL H1 (flat like I2, or judge-24+G8 unresolved-or-worse = lever ceiling binds) | P1 (after P0-P2) |
| P8 | H6 Kickstarting KL-anchor probe (conditional) | ONLY if H2/H5 show signal but drift eats it: 3-cycle full-episode probe with/without anchor at fixed LR; proper after-T0 schedule (NOT linear-to-0 alongside LR) + masked KL; residual teacher = zero correction, closed-form KL. CAUTION: I2 policy was essentially frozen (\|dc\| <= 1.8e-2, logstd drift <= 0.007), so the -4..-5% rollout-mean drift is not large-\|mu\| growth | drift removed (c2 >= c0 full-episode) with final >= behavior; else KILL H6 (KL -> 0 with \|dmu\| -> 0 pinned at behavior = base is the ceiling for these levers) | P3 |

## Group S. Accounting and long-shots

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P9 | Honest bill + pool sha + holdout on every run | log every rollout/judge/b8 step count, bill mode, pool sha | go/no-go stated as >1M honest steps; no sub-1M promotion claim | P0 |
| P10 | Literature long-shots (only if H1+H2 both killed) | BCOOLER-style offline iteration on rule data first (no new effector); then a small SAC/TD3 probe -- but constrained (CQL-style: plain SAC/TD3 collapse on narrow expert data, F6) and WITHOUT the reversed Sinergym citation (2412.08293 actually shows PPO best; needs Biemann21/Mahbod22/Zhang23 primaries -- see Addendum A); HRL timescale split only if warm-up vs intraday separable in our cadence | full-episode >= 0.0002 edge on 2 climates; else KILL | P3 |

## Priority summary

P0 (rails): P0, P1, P2, P9.
P1 (mechanism bets): P3, P4, P6, P7.
P2 (speculative cheap): P5.
P3 (conditional): P8, P10.
Execution order (cheap first): P0+P1+P2 -> P3+P4+P5 offline in
parallel -> cheapest surviving online screen (P6 or P4-online) ->
P7 (H-green then ONE 750k) -> judge-24 + b8 + G8 per ckpt. If P7 is
flat, H1 is falsified; compute goes rule-side, not to longer runs.

## DO-NOT (closed lines -- spend zero PPO sims)

- Branch-a as-is: I2 3.02M flat (best = c0) + I3
  unresolved-vs-occ9 (+0.00019989 < 0.0002) / loses b8 (+3.26%).
  P7 reopens branch-a with a new effector, so it must re-pass I1
  + fresh judge+G8 per ckpt.
- MoE hybrids: T3 CLOSE (offline MSE win evaporates online),
  T4/T5 never open; ladder slot/PPO diverged-closed.
- Per-day schedule / warm-up-timing (G5): competitor oracle
  below bar + warm-up levers worse every climate; our J5
  +0.000186 < 0.0005 + J1 dead tie -0.00000140; fixed horizon
  dead via the prestart latch. A NEW-menu oracle
  (PRESTART_MARGIN x SAFETY x GRACE) is ungrounded
  (SAFETY/GRACE read-sites unverified) -- do not spend 20-60
  sims to lock an already-closed door. Reopen bar only:
  grounded code refs + >= 0.0005 same-sign on >= 2 climates,
  offline first.
- Also closed: full-authority (0.011175 vs 0.010868 gated), +-2K
  outside-band (worse), paired-reward shadow (fail-4 flat
  750k-1.5M at 2x sim; G7 closed), per-zone credit (fail-33
  null), PPO-recipe tweaks (fail-44 null).

## Transfer classes (where each idea comes from)

- Directly transferable (no sim change): RPL zero-init residual
  head + critic warmup/burn-in; Guo reward-shaping discipline +
  small physics-informed basis + baseline-mixed exploration;
  Sinergym/Johannink diagnostics (explain the I2 flat curve:
  strong RBC + on-policy + comfort-dominated reward +
  cancellable authority).
- Needs adaptation: zone-target head + PI-target interface (P7);
  PI loop-state features, regularized (P6); confidence gate with
  separate net + non-circular labels (P4); BCOOLER-style offline
  iteration; frozen V_rule residual critic (P3); HRL timescale
  split (P10).
- Speculative: PBRS comfort potential (P5: dense-reward +
  gamma mismatch); Kickstarting KL-anchor (P8); TTDM T-symmetry
  GNN; AH-RL blending (uninspected full text); any MoE hybrid
  (closed T3).

## Open gaps (carried from the deep research)

1. D5 GO hook code location (`_cur_vav` / per-step offset
   dict): plan-18 asserts existence, no file:line inspected --
   blocks the zone-target cost estimate (P7 step 0).
2. PRESTART_SAFETY / EMPTY_GRACE_STEPS exact defs and
   read-sites -- blocks any warm-up oracle menu (stays closed
   until grounded).
3. Raw pre-norm advantage std + explained variance per I2
   cycle; per-cycle free_frac on the 25-pair pool -- batches
   were not persisted, needs a fresh rollout (P1).
4. TRANSIC/ARC/Johannink/ResidualPPO full texts not
   re-inspected by the critic; Ng 1999 primary + Q-Prop full
   text not inspected (BSRS restatement + abstract-only).
5. Per-worker throughput curve at conc <= 10 for 750k
   wall-clock -- 120 vs 153 steps/s figures unreconciled.

## Addendum A (2026-11-04): expanded-source deep research

Second workflow (arXiv full-text x9, OpenReview/PwC sweep, Kaggle
forum mining; all claims from inspected bodies). Per-hypothesis
delta vs the original draft:

- H1 (setpoint superposition): SUPPORTED. BCOOLER (2211.07357):
  12-dim supervisory setpoints over low-level loops, 9%/13% vs
  tuned SOO, MC targets gamma=1 (TD rejected offline), feature
  engineering "critical". Johannink (1812.03201): residual
  recovers from biased controllers, sim-init solves real task
  in <1000 steps. RPL (1812.06298): zero-init + critic burn-in
  (loss<1.0) explains early-training dips; ~10x samples vs
  scratch.
- H2 (residual critic): MECHANISM SUPPORTED, SCOPE WEAKENED.
  Q-Prop (1611.02247, full text -- gap-4 CLOSED): off-policy
  Taylor critic as control variate, ~10x vs TRPO, critic need
  only correlate. But Q-Prop's CV is action-dependent; H2's
  frozen V_rule is state-only (strictly weaker). Consider an
  action-dependent CV if state-only stalls.
- H3 (history/loop-state): STRONGLY SUPPORTED + overfit
  caution. Offline-HVAC (2408.07986): no-history RL LOSES to
  rules; +Transformer history: violations 30.9%->2.36%,
  -12.1% power; history len 20/30 enough; 1e6->1e5 steps no
  degradation; suboptimal mix (eps 0.1-0.5) beats near-expert
  data (stitching). Merino (2401.05737): curriculum
  cool->hot WORSE (catastrophic forgetting), cross-climate
  drops. BEAVER (2507.07769): naive context mixing marginal.
- H4 (confidence gate): STRONGLY SUPPORTED + CORRECTION.
  TRANSIC (2405.10315, full text): gated 85% vs 55%, trained
  ONLY on correction data, fine-tuning base collapses, robust
  to noisy corrections (-6%). CORRECTION to plan-19 P4: the
  gate SHARES the residual feature encoder; separation is a
  separate head + TWO-STAGE training (freeze encoder+action
  head, then train gate; joint is suboptimal). Fraction-open
  (0.1, 0.6) and non-circular labels stand. BEMS review + BCOOLER
  masking/Maskable-PPO concur.
- H5 (PBRS shaping): WEAKENED as specified; P5 kill-gate
  affirmed. Ng1999 PRIMARY (gap-4 CLOSED): Thm1 sufficiency +
  NECESSITY; undiscounted case REQUIRES an absorbing state +
  proper policies. Our undiscounted LB + gamma=0.995 GAE +
  dense reward + no absorbing state violates the conditions,
  so shaping changes the optimum. Keep rank-corr/sign-check
  posture; expect the kill.
- H6 (KL anchor): schedule SUPPORTED, unlock WEAKENED.
  Kickstarting (1803.03835): lambda_k->0 AFTER T0, constant
  plateaus, PBT auto-tunes, 9.58x +42%. But their student
  surpasses a WEAK teacher; ours is near-optimal + frozen, so
  the anchor prevents drift but cannot create signal. P8
  stands as conditional-only with the after-T0 (not
  linear-to-0-with-LR) schedule.
- H7 (off-policy/offline >> PPO): CONTESTED. FOR: CQL best on
  narrow expert data, SAC/TD3 collapse there (F6 -- predicts
  plain-SAC-on-P1-batch FAILS, needs CQL constraint);
  BCOOLER offline iteration wins real world; CityLearn21 SAC
  1.092 vs PPO 1.199 (worst); BEMS review (off-policy
  dominates continuous). AGAINST: Sinergym (2412.08293)
  from-scratch SB3-default datacenter shows PPO BEST and most
  stable, SAC~=RBC -- the draft's "Sinergym: off-policy>>PPO"
  citation is REVERSED (caveats: from-scratch, datacenter,
  default hypers). Merino 5Zone: RBC beats ALL DRL.
  NUMBERING GAP: H7 was never defined in plan-19 (H1-H6
  only); children assumed H7 = P10 off-policy. Adopt: H7 =
  off-policy/offline methods beat on-policy PPO on HVAC
  rule-data. Still needs Biemann21/Mahbod22/Zhang23 primaries
  to replace the reversed citation.

New candidates (plan-19 style):

- N1 planning/optimization+forecast ensemble beats PG
  (CityLearn21 winner ZO-iRL 0.962 vs PPO 1.199 worst;
  CityLearn22: 1st stoch-opt+MAPPO, 2nd DP+CMA-ES, 3rd LP;
  CityLearn23 CHESCA: PID + A* + XGBoost forecasts wins on
  comfort 0.132 vs 0.304/0.203 + public->private
  consistency). TEST: same-budget split, optimizer-hybrid vs
  pure PG on generated+held-out pools, gateway-scale MAE.
  GATE: hybrid wins total + comfort with LB consistency.
  KILL: hybrid <= rule or loses comfort to residual-PG.
- N2 comfort-term + generalization decides; RL overfits small
  pools (CHESCA, BEAVER, Merino + forum judge evidence).
  TEST: judge-first with OCCUPANCY-SEED VARIATION (seed 17321
  replay BANNED for finals, thread 752678), generated
  buildings + held-out weather (cf. K07 tuning: LB 0.01500 vs
  0.01788, -16.1%, zero shipped scenarios in tuning),
  comfort-vs-energy split reported. GATE: improve ALL pools
  incl. unseen-climate comfort, no seed dependence. KILL:
  seed-dependent gain or new-pool comfort regression.
- N3 honest-bill + gateway-parity training transfers across
  the corrected sim (threads 749531/752142/753804/749896).
  TEST: train/eval on energy-balanced-tank + airflow-weighted
  return-air billing; compare pre/post fix + gateway vs
  evaluate.py scale. GATE: honest delta <= ~0.00003 (+0.3%,
  Mima AI) and cycling-policy cost rises (~2x); adopt a
  gateway-parity harness + G8 screen because local-vs-LB
  scale artifacts (~3% phantom-occupant tracker gap, -215.2
  vs -208.2) EXCEED the 0.0002 floor. KILL: honest tuning
  regresses >floor on corrected sim or relies on unfixed
  billing.

Forum facts adopted into protocol:

- Final verification AIMS at the corrected sim; LB stays
  shipped meanwhile (ayushrai1114, ~Oct 29). Train honest,
  score shipped, expect a re-score.
- SECOND billing bug: area-weighted (not airflow-weighted)
  return-air temp (thread 753804, fix = PR #3): ~117 kW avg
  artificial heat vs ~1 kW billed boiler on a scripted
  policy; corrected sim needs +15% boiler heat +18% fan.
  Do NOT exploit; DO verify our policies don't accidentally
  depend on it (N3).
- Metric = MAE vs zero over 30 held-out scenarios, equal
  weights, all 8640 steps incl. first. Public/private are
  SUBSETS of the same 30, not separate runs. 9h limit covers
  the whole submission incl. final re-run.
- Residual-PPO notebook now V23, board 0.01107; new §4e "Give
  the RL the night" (all-hours residual revisited). Full
  v17-23 cell text not fetchable (client-rendered).

Open gaps carried (subset of prior gap-4 + new):

1. ARC exact citation (2107.04487 unrelated); ResidualPPO
   distinct-paper id if any (RPL family covered).
2. Biemann21/Mahbod22/Zhang23 primaries for the SAC>>PPO
   premise (P10/H7).
3. OpenReview 2024-26 forums blocked by challenge wall
   (TRANSIC _8DoIe8G3t, MimicTouch EXQ0eXtX3OW for an
   authenticated pass).
4. Starter-kit PR #1/#2/#3 GitHub bodies uninspected (pin
   exact corrected-sim semantics).
5. Host-open questions: fresh Policy instance per episode,
   corrected-sim commit/date/LB-rescore, humidity/freeze
   scope, wall-clock limits, heat-pump/unseen climates.

## Execution log

- 2026-11-04 P0 GREEN (day note `docs/2026/11/04/`):
  per-head dmuz/dmua/dmup + pre-clip gpol/gval in the
  `ppo_epoch` trace (`policy_params`/`_grad_norm` helpers);
  1-mb check prints all fields (synthetic batch:
  gval 3555 >> gpol 2.45, value-drowning visible);
  self-test PASS, 72/72 unit tests. 0 sim.
- 2026-11-04 P2 GREEN (same note): H-gate re-pass rig
  verified present, no code changes: KL-stop
  (`ppo_epoch`), E5 clip-abort (`train.py:285-289`), E3
  best-ckpt + LR-decay (`:156-160,300-301`), E4 pool sha
  (`:74-78`, printed pre-cycle-0 + in batch meta),
  monster check (`zone_count.py`: b8/b7 max 10/AHU valid),
  P1 pool train-only (xfer_5..10 exclusion holds; P7 pool
  must re-verify), conc 10, trace with P0 fields, no
  policy edits in flight. 0 sim (2 loads ~1s).
- 2026-11-04 P1 GREEN (same note): `--rule-only` +
  `--dump-batch` in the skynet stack (smoke: fr/za exact
  zeros, meta ok, rollout self-test PASS, 72/72); two
  persisted batches `/tmp/p1/rule10.npz` (86 MB,
  free_frac 0.000) + `/tmp/p1/actor10.npz` (103 MB,
  free_frac 0.363, za_std 0.185): 10 pairs (b8x5+b7x5) x
  8640, honest, seed 11, pool sha 058d815f0357.
  172,800 sim steps. Regen commands in the day note.
- 2026-11-04 P3 done: H2 KILLED (same note). V_rule MLP
  (pooled feats, rule10): test MSE 21.05 vs init-critic
  26.7 (0.79, gate <= 0.5 MISSED) and const-mean 123.8.
  Decisive 1-mb trace on actor10 post-v-pretrain (100
  steps, 79.67 -> 28.89): gpol=0.007 vs gval=4.570
  (650x drowning CONFIRMED real). But split combined
  (frozen V_rule + V_res-MLP/100 steps): 38.60, WORSE
  than single critic 28.89 -- frozen baseline destroyed
  by actor-state shift (47.21 on actor10 vs 21.05 on
  rule10). Mechanism kicker: under Adam the global clip
  scale is absorbed (m-hat/sqrt(v-hat) scale-invariant),
  and gpol ~= nil absolute while dmu 3.2e-3 moves go
  nowhere = SIGNAL problem (fail-9 noise), not step-size.
  H2 closed per kill-gate (no MSE drop). 0 sim.
- 2026-11-04 P4 done: H4 SURVIVES (same note).
  `tools/gate_fit.py` (new): GateMLP on pooled feats,
  rule-vs-residual counterfactual labels (actor10 vs
  rule10, same pool+seeds, actor-free steps; NOT GAE):
  held-out AUC 0.7197 >= 0.65, open-frac 0.599/0.476/
  0.353 at tau 0.3/0.5/0.7. Rollout-side wiring
  (`--gate/--gate-tau`, worker-cached, features-before-
  free reorder): smoke free 0.314 -> 0.131 at tau 0.5.
  Online confirm (seed 31, honest, full 8640, epochs 0):
  gated >= ungated BOTH pairs -- fb8 +0.001744 (+5.5%,
  gate opened 0.000 = pure rule, correctly shut off
  harmful heating noise), phx3 +0.000153 (+2.0%, opened
  0.154). H4 stays alive for P7. 4x8640 = 34,560 steps.
- 2026-11-04 P5 done: H5 KILLED (same note).
  `tools/pbrs_probe.py` (new): Phi from batch band cols,
  terminal Phi'=Phi; k=1: pre-norm std ratio 0.999,
  rank-corr 0.9974, sign-check BACKWARDS (good -0.43 vs
  bad -0.15); k=5: ratio 1.020, corr 0.9657. Shaping adds
  no signal (reward already dense, comfort dominates
  per-step) -- exactly the Addendum-A prediction. Two
  probe bugs caught+fixed en route (post-norm std vacuous;
  normalize_adv in-place aliasing). 0 sim.
