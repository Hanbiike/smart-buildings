# 2026-11-05

## Summary

Plan-20 closed COMPLETE (P20-12 done, pushed 4276292): all 13 items
done/closed/killed/skipped, no survivors, no promotions. Judge 3-bill
chain landed 168/168: Mima 0.00003 stays dropped (pool-dependent sign),
PR#3 deltas below floor, tuning re-KILL confirmed (+0.005314 judge),
G8 all PASS (moot). Deep research round 4 ran as a workflow (5 angles
+ critic + synthesis, 7/7 results read): VLM-RL killed, frozen-det
over occ9 queued as plan-21 head, transfer-gap F-series specified,
seed rescreens + offline probes + 0-sim backlog scoped. Critic caught
a plan-20 wording slip (P20-10:344 "beats occ9-rule" vs occ_gated7
screen base) — P21-1 re-screens vs occ9 first. Wrote
`docs/plan/21-frozen-determinism.md`, starting action.
Plan-21 ran P21-1..P21-14 same day: frozen line KILLED
(weak-base artifact), F-series all killed/vacuous
(F2 22.7x, F4 trigger 1e-44, F3 1/3, F6/F7 closed),
rescreens killed (R1a counter 0, R2 fire 0, N1 singles
lose to occ9), E2 collapse, R3a no-sweep, MoE DAgger
2/2 failed (BLOCKED), docs closed. COMPLETE, no survivors.
Round-5 research workflow (5 angles + critic + followup +
synthesis, 8/8 read): ranked 10 plan-22 candidates; parent
refuted two-teacher theory (dag1 labels ARE occ9),
confirmed NO-RLS (covariance alloc-only) and pockets
(h67 76.7% exact). Wrote docs/plan/22-loss-pockets.md.
Plan-22 ran P22-1..11: UGAS KILL, PRE CLOSED (+0.000071),
DWU KILL (-0.00107), DMAX KILL, F2B stays KILLED, RESID
PASS (P21-13 unblocked), OFFM/P2M/DAGX closed/killed,
docs closed. 1 PASS. Round-6 (3/8 results, angles 1-4
lost to schema) ranked 8 revival candidates; parent
verified bank1-zero + memory skew. Wrote plan-23.

## Status

| Whose | Score | State |
|---|---|---|
| leader (boys) | 0.00470 | LB Nov-03 |
| us (occ9) | 0.01236 | LB Oct-28 (#273, stale) |
| us (occ9) | 0.00878 | train-30 shipped control |

## Done

### Plan-20 completion (carried from 11-04)

- P20-12 done: judge 168/168 clean, verdicts logged (Mima dropped,
  re-KILL +0.005314/+0.003791, skynet transfer gap b8 -0.000477 vs
  judge +0.002578, G8 all PASS worst 1.896). 1,797,120 sims.
  Pushed `4276292`. Plan-20 COMPLETE, no survivors.

### Deep research round 4 (workflow, 7 agents)

- 5 angles (frozen-det anatomy, transfer gap, killed lines,
  offline-RL/VLM, gaps) + critic (complete=false: angles 0/1/2
  unvetted by critic) + synthesis. All 7 results read inline.
- Verdicts: VLM-RL KILL (tabular obs, CPU notebook); frozen-det
  distillation sanctioned (P0); F1-F7 frozen fixes cheapest-first;
  R1a/R2/N1-singles rescreens scoped; E2 RvS + R3a mine + MoE
  (blocked) scoped; K07 cf + ladder + README + honest_bill diff
  closable 0-sim; PR/forum/budget needs-external.
- Critic corrections applied to plan-21: occ9 IS occ_econdbhhot9
  (dumps current-best, staleness struck); tokens.npz absent (0 on
  disk); forecast-KILL scoped to R3 menu; E1/E2 bars must pin.

### Plan-21 written

- `docs/plan/21-frozen-determinism.md`: Group F (P21-1 occ9-base
  2-pair screen kill-first, P21-2 mechanism dump, P21-3
  distillation), Group X (P21-4..7 F-series), Group S (P21-8..10
  rescreens), Group O (P21-11..13 offline), Group B (P21-14
  0-sim), KILLED list, needs-external list.

## Measured

`occ9det — fb8+phx3 x8640 honest — fb8 gap -0.003512, phx3 gap +0.000015 — KILL (wrong sign / below floor)`.
`clip-rate — train 0/60M vs judge 0/46M — F6 KILL (judge not greater)`.
`F3 scale — 1/3 probes (xfer_10/msp1 only) — KILL (needs 2/3)`; `F7 — no target head — CLOSED pre-screen`.
`F2 cold — Monday regret 22.7x occ9 (gate 3x) — KILL`; `F4 occ-gate — trigger P~1e-44/day — CLOSED vacuous`; P21-7 DEAD.
`R1a rescreen — OCC_SEED=99999 counter 0 (bit-identical) — KILL`.
`R2 rescreen — fire 0 (xfer_3 + 2 full C-parity) — KILL`.
`N1 singles — p2 explodes train-fb, off/hyst lose to occ9 — all KILLED`.
`E2 RvS — collapse (shuffled RTG beats true) + teacher ceiling — KILLED, 0 sim`.
`R3a mine — sat wins 0/5, no MILD_SAT sweep — CLOSED, 0 sim`.
`MoE DAgger — idea-1 fb8 -91% but sg8 12x worse, idea-2 worse than soft7 — BLOCKED, 192k`.
`P21-14 — 6 doc closes (K07, ladder, r01, 11-04, bill bound ≤0.0003, 31/33) — PLAN-21 COMPLETE`.
`P22-1 UGAS — floor-cap lever mixed below floor (fb8 -0.00015, fb1 +0.00013) — KILL`.
`P22-2 PRE — screen PASS (denver_3 -0.000515) but train-30 +0.000071 below floor — CLOSED`.
`P22-3 DWU — phase-1 train-30 -0.00107 (hot catastrophe, SAT +5K w/o joint dampers) — KILL`.
`P22-4 DMAX — fb1 -0.010, fb8 -0.004 (min-airflow under-delivers in deep cold) — KILL`.
`P22-5 F2B — ablation bit-identical to F2 (stale-offset refuted, F7-consistent) — F2 stays KILLED`.
`P22-6 RESID — PASS: fb8 gap cut 102% (beats occ7 by 0.00088), elsewhere ties — P21-13 UNBLOCKED`.
`P22-7 OFFM — mild wins 7x below floor, sg7 bit-identical (gate blocks hot) — CLOSED`.
`P22-8 P2M — OAT-gate kills explosions (fb ties) but atl2/sg7 lose — KILL`.
`P22-9 DAGX — cold-transfer confirmed (fb1 93%, fb2 50.5%) but sg8 poisons — CLOSED`.
`P22-10 — doc closes (RLS correction, pocket_stats.py, weight cite) — PLAN-22 COMPLETE (1 PASS, P21-13 unblocked)`.
`P23-1 SCALE — monotone scale-up, res30 wins (fb8 cut 52%), sg8 holds — PASS, res30 new baseline`.
`P23-2 SKEW — mismatch is divergence not a bug (9.6% deleg), unfixable by update rules — REVERTED, 0 sim`.
`P23-3 BREAK — self-limit holds to 0.5 (res50 fb8 best) but s=1.0 explodes everywhere — KILL`.
`P23-4 K2 — OAT-split hurts fb8 (-0.000305): sample-split cost beats specialization — KILL, K=1 holds`.
`P23-5 FEAT — zero R2 lift on phx1 SAT (.176→.176), err-channels exactly-0 weights (degenerate) — KILL, 0 sim`.
`P23-6 DAG6 — cold DAgger-on-resid: no win anywhere below floor (fb8 -0.000140), R2 warned sg8 .399→.137 — KILL, dag2-pattern`.
`P23-7 STACK — 6×6 OAT table: all arms below floor (best stackb +161e-6), additive, no blowups — CLOSED`.
`P23-8 SHIP — self-contained res30 + 179KB f32 cell, parity bit-exact, bars pass, 2×500 exec-check PASS — PASS (infra, not promotion)`.
`P23-9 DOCS — round-6 loss record + findings to research — PLAN-23 COMPLETE (1 PASS baseline, 1 PASS infra, 4 KILL, 1 CLOSED, 1 REVERTED)`.

### Plan 21/P21-1: occ9-base screen (FROZEN LINE KILLED)

- `--base` added to frozen_det_screen.py (default occ_gated7).
  Screens seed 31 + 99999 (69,120 honest): fb8 det loses
  -0.003512 (17x floor, wrong sign), phx3 +0.000015 (floor).
  P20-1 signal was a weak-base artifact; P21-2/P21-3 SKIPPED.
  Seed replica degenerate (rollout seed != occupancy).
  Logged in plan-21.

### Plan 21/P21-4: F6 clip audit (KILLED)

- Train 0/60M vs judge 0/46M clip hits (24 fresh judge dumps).
  Input scale healthy; transfer gap elsewhere. 36,000 sims.
  Logged in plan-21.

### Plan 21/P21-5: F3 scale + F7 offsets (F3 KILLED, F7 pre-screen)

- F3 12/12 7-day: scale<1 wins only xfer_10/msp1 (1/3, needs
  2/3). Residual hurts in deep cold, helps elsewhere. No
  scale=0 redirect. F7: no target head in ckpt — CLOSED
  pre-screen. 31,692 sims. Logged in plan-21.

### Plan 21/P21-6: F2 recovery + F4 occ-gate (F2 KILLED, F4 vacuous)

- SkynetColdProbe (k=1.0K) + SkynetOccProbe (OCC_GATE) + 5
  tests green. Liveness xfer_10/msp1 x1500: F2 0.0137 vs
  skynet 0.0167 (live, better); F4 == base bit-identical
  (base _pi_hours already day<5; weekday zero-occ is nights
  only). F4 mechanism live synthetically, trigger
  P(empty weekday) ~= 1e-44/day (N~=53) + 0 triggers in
  ~40 episodes -> CLOSED gain-0.0 by construction (P20-6
  precedent). F2 full x8640: Monday regret -0.055 vs occ9
  -0.0024 = 22.7x (gate 3x); first Monday good (-0.0038)
  then explodes (-0.15/-0.07/-0.05); full +0.009 worse.
  21,780 sims. P21-7 DEAD (conditional, both killed).

### Plan 21/P21-8: R1a fresh-seed rescreen (KILLED)

- Trigger counter OCC_SEED=99999 x1500: fb8 + atl1
  bit-identical all 6 CSVs (offa1 == occ9). Counter = 0,
  gate needs > 0: no screen. R1a dead under 2 seeds;
  forced-departure model makes night spikes
  near-impossible. 6,000 sims. No code changes.

### Plan 21/P21-9: R2 T&R fire rescreen (KILLED)

- C-parity method (any fire diverges C via EMA): xfer_3/msp1
  x1500 C == occ9 bit-identical; full-8640 C-parity b1/fb1
  + b2/fb2 bit-identical all 6 CSVs. Fire == 0 everywhere
  incl. pressure-lift pairs (lifts are heatopt's). R2 dead
  on train-30 + transfer + 3 full episodes. 37,560 sims.
  No code changes.

### Plan 21/P21-10: N1 singles scored liveness (all KILLED)

- Screen 5 arms x 2 xfer x1500: all mean-win vs skynet,
  p2 only same-sign both. Harness 3 singles x 5 SCEN
  x864: p2 explodes train-fb (+0.027/+0.025, mean
  +0.011); off == occ9 bit-identical all 3 fb (lever
  vacuous, win was occ9-inside), +0.00025 vs occ9 mean;
  hyst +0.00023 vs occ9 mean. No single beats occ9.
  36,600 sims. No code changes.

### Plan 21/P21-11: E2 RvS collapse probe (KILLED, 0 sim)

- Bars on 30 occ9 dumps (259,200 steps): spread healthy
  (0.17-0.33). BC 0.022307 vs RvS-true 0.021316 vs
  RvS-shuffled 0.020966 vs BC+clock 0.021100 (fb8
  held-out, GPU): shuffled beats true — no return
  signal, collapse confirmed. Teacher = occ9 = current
  best: ceiling = occ9. No escalation. 0 sim.

### Plan 21/P21-12: R3a 0-sim mine (CLOSED, no sweep)

- Located 20x b8x5 CSVs (/tmp/r3a20). Sat vs hws
  per-pair: worse atl1/den6/fb8, tie phx4/sg7.
  0/5 wins: no MILD_SAT sweep, forecast->menu dead,
  bound +0.000199 stands. 0 sim.

### Plan 21/P21-13: MoE revival via DAgger (BLOCKED, 2/2 failed)

- SlotDaggerProbe (parity IDENTICAL, corrective 0.23) +
  dag1/dag2 probes + 6 tests green. Diagnosis: 1K cold
  drift by day 3 -> OOD -> comfort spiral. Idea-1:
  fb8 gap -91% but sg8 12.6x worse. Idea-2 (3-pair):
  mean 0.047 vs soft7 0.029 — worse everywhere but
  fb8. Shared linear heads cannot hold conflicting
  correctives; no convergence. T4 shut. Residual-ize
  + per-mode routing queued for research. 191,940 sims.

### Plan 21/P21-14: doc/gap closes (done, PLAN-21 COMPLETE)

- K07 clause dropped (plan-20 gaps renumbered); ladder
  leader boys 0.00470; research/01 -> research/21
  pointer; 11/04 Summary/Open refreshed; honest_bill
  bound diff <= 0.0003 (bound, not closed); 31-vs-33
  reconciled (31 valid + empty + file). 0 sim.
  Plan-21: no survivors; P21-13 BLOCKED pending research.

### Plan 22/P22-1: UGAS floor-cap trim (KILLED)

- UgasCap20Policy (eco_lo - 2K) + 3 tests green.
  FLOOR_CAPABILITY proven vacuous (gated3 cap binds).
  Replay-diff liveness (791 modes). Screen: fb8
  -0.000152 win, fb1 +0.000126 loss — mixed below
  floor. fb1 comfort -46.8 vs -38.3 (recovery breaks).
  48,060 sims.

### Plan 22/P22-2: PRE margin doubling (CLOSED below floor)

- PreMargin12Policy + 3 tests green. Screen PASS:
  denver_3 -0.000515 (comfort -44%), fb8 -0.000045.
  Train-30 30/30: +0.000071 mean, 22/30 wins, 4
  above-floor wins, 1 above-floor loss (denver_5).
  Below floor: no judge, no promotion. 555,960 sims.

### Plan 22/P22-3: DWU sat-search port (phase-1 KILLED)

- DwusatProbe (drift + observer + 11x2 search,
  SAT/recirc/pressure only) + 3 tests green, parity
  0/1235. Train-30: -0.00107 (5x floor wrong sign),
  10/30 wins. Hot catastrophe (phx5 -0.0061): SAT
  +5K warmer w/o joint dampers = comfort collapse.
  Full-joint + climate-gate queued. 524,400 sims.

### Plan 22/P22-4: DMAX dual-max staging (KILLED)

- OccDmax9Policy (split 0.5/min 0.1 fixed) + 3 tests
  green. Screen: fb1 -0.0101, fb8 -0.0037 — same-sign
  wrong, 18-50x floor. Min-airflow heating under the
  discharge cap under-delivers in deep cold. 37,560 sims.

### Plan 22/P22-5: F2b ablation (F2 stays KILLED)

- SkynetColdBProbe + 6 tests green. 1-run ablation:
  F2b == F2 bit-identical (0.026814 x2). Stale-offset
  refuted (F7-consistent: offset never set). Cause is
  residual-debt. 8,640 sims.

### Plan 22/P22-6: residual slot (PASS, P21-13 UNBLOCKED)

- --residual-base refit (delta=T-B, self-replay
  bit-exact) + ResidualSlotProbe (scale 0.15) + 7
  tests green. 9-run: fb8 0.0296 vs soft7 0.0697
  vs occ7 0.0305 (cut 102%, beats occ7 0.00088);
  phx1/sg8 tie occ7 (2e-6). Mean-3 beats occ7 by
  0.000295 (T4 bar). 82,260 sims.

### Plan 22/P22-7: mild-only OFF (CLOSED below floor)

- N1OffMildProbe (OAT-gate) + 9 tests green. Screen:
  atlanta_2 +0.000026, denver_7 +0.000028 (same-sign,
  7x below floor); sg7 bit-identical (gate blocks hot
  fire; ungated lost -1.18). Mechanism confirmed, no
  escalation. 54,840 sims.

### Plan 22/P22-8: heat-interlocked P2 (KILLED)

- N1P2MildProbe (OAT >= 10C) + 11 tests green.
  SCEN-5 x864: fb ties (explosions gone) but atl2
  -0.000545, sg7 -0.000168 lose. All-5-win gate
  fails. P2 dead everywhere. 8,640 sims.

### Plan 22/P22-9: dag1 transfer falsification (CLOSED)

- Stage-1 fb1+fb2 x8640: dag1 wins both, gap cut
  93.4% + 50.5% — cold-to-cold transfer works.
  Stage-2 skipped (sg8 poisoning already proven).
  Revival gate failed; resid dominates. 51,840 sims.

### Plan 22/P22-10+11: doc closes + deferred (PLAN-22 COMPLETE)

- RLS->observer correction in research/21;
  tools/pocket_stats.py repro (exact); weight source
  scenario_generator.py:372-374. P22-11 stays DEFERRED
  (no torch candidate for smoke). 0 sim. Plan-22:
  1 PASS (P22-6), 3 CLOSED, 5 KILLED, 1 DEFERRED.

### Plan 23/P23-1: scale ablation (PASS, res30 baseline)

- ResidS05/S30/PCProbes + 8 tests green, refactor
  parity exact. 18-run: fb8 monotone (res30 0.028963,
  cut 52%); respc ~= resid (killed); sg8 ties at 0.3;
  phx1 +0.00009 below floor. res30 = revival baseline.
  161,520 sims.

### Plan 23/P23-2: memory-skew fix (REVERTED by analysis)

- Deploy already matches train given same modes
  (chain runs only non-shutdown). 9.6% deleg mismatch
  is closed-loop divergence (DAgger domain), not an
  update-rule bug. No screen, kept skew. 0 sim.

### Plan 23/P23-3: noise-head break-test (KILLED)

- ResidS50/S100Probes + 8 tests green. 9-run: fb8
  res50 0.028820 (best) / res100 explosion; sg8 tie
  at 0.5, 10x break at 1.0; phx4 res50 2.7x worse.
  Self-limit holds to 0.5 only. res50 hot-risk, not
  promoted; baseline res30. 80,760 sims.

### Plan 23/P23-4: per-mode K=2 resid (KILLED)

- mode_of_oat + hysteresis in SlotFeatures, per-mode
  accums in slot_train (K=1 bit-exact legacy),
  mode-aware _blended, ResidK2Probe + 4 tests
  green (301 total). Refit slot_resid_k2: zone/ahu
  1 zero-slot each, water-mode1 370/576 fallback
  (no heat in warm mode). 6-run vs res30: fb8
  0.029268 vs 0.028963 (-0.000305, gate fail);
  sg8/phx1 below floor; mean-3 -0.000107.
  Split cost > gain (mode0 med 212 vs 1730).
  K=1 holds; baseline res30. 52,140 sims.

### Plan 23/P23-5: feature augment (KILLED by R2 gate)

- slot_r2.py (per-channel offline R2) + 7 new
  channels (zone unclipped tw/oat, AHU satsp/
  sat-err/dp-err/fan, water sp-gap) with prefix
  slices (old weights bit-identical, parity
  proven) + 4 tests green (305 total). Refit
  f5: phx1 SAT .176→.176 (zero lift), water
  still negative; sg8 no inflation. sat-err/
  dp-err weights exactly 0 (teacher holds
  setpoints; degenerate in demos). No screen
  (ordered gate). Baseline res30. 0 sims.

### Plan 23/P23-6: DAgger-on-resid cold-only (KILLED)

- replay_base.py (bit-exact vs shipped B) +
  dagger_collect POLICY_MAP keys + 4 tests
  green (309 total). Collect 6 fb pairs
  (res30/occ9) + B-replay + refit dag6 +
  ResidDag6Probe. R2 warned (sg8 a.sat
  .399→.137). Screen vs res30: fb8
  -0.000140 (gate fail), sg8 -0.000025,
  phx1 +0.000116; all below floor,
  mean-3 -0.000016. Bound held poison
  inside floor, bought no win:
  non-convergent (dag2-pattern). Infra:
  /tmp quota-capped (1GB write fails at
  6G avail) — tests moved to slot_runs,
  TMPDIR=~/sim_scratch/tmp for suite.
  Baseline res30. 121,260 sims.

### Plan 23/P23-7: OAT-gated stack (CLOSED below floor)

- OatStackA/BProbes (offmild+pre12 / +ugas)
  + 4 tests green (313 total). Stage-0
  6x6x1500 PASS (36/36 live); 6x6x8640
  table: stackb best mean-6 +161e-6 <
  floor; prem12 phx4 +422e-6 single-pair;
  ugas fb8-only +152e-6. Stacks additive
  (fb8 exact) to superadditive (atl2);
  worst sg7 -21e-6. No promotion;
  catastrophe-avoidance value only.
  365,040 sims (home-disk results).

### Plan 23/P23-8: ship hardening (PASS, infra only)

- occ_resid30.py (self-contained ship,
  lazy stdlib, no heavy imports) +
  make_ship_weights.py (179KB f32 cell)
  + nb_exec_check.py + 6 tests green
  (319 total). Ship-vs-f32 bit-exact
  (8640 obs); P21-3 bars pass; 2x500
  exec-check PASS (shutdown+night+
  prestart covered). res30 still below
  occ9: no promotion. 1,000 live sims.

### Plan 23/P23-9: doc closes (PLAN-23 COMPLETE)

- research/22-round6-angle-loss.md: gap
  record (5/8 children lost) + 3 kept
  findings (bank1-zero, skew-real,
  p99 consts). Final tally: P23-1 PASS
  (res30 revival baseline), P23-8 PASS
  (ship infra), P23-3/4/5/6 KILLED,
  P23-7 CLOSED, P23-2 REVERTED. No
  promotion; revival gap to occ9 stands.
  0 sims.

## Decisions

- Plan-21 head is frozen-det over occ9 (only above-floor signal),
  but vs-occ9 is UNTESTED — P21-1 kill-first screen before any
  dumps or distillation. Proof: critic + angle-0 evidence
  (frozen_det_screen.py base occ_gated7). Consequence: P0-2/P0-3
  gated behind P21-1.
- Promotion rule (plan F/F3): no checkpoint promoted on b8 or
  train alone — judge-first, b8-second; gaps below the 0.0002
  floor do not resolve (F2); G8 3x guard.

## Open

- Plan-23 COMPLETE. Next: fresh deep research
  (revival gap to occ9 unresolved; res30
  -0.0014 on fb8). P21-14 backlog closes
  (0-sim, anytime).
- Needs-external: PR diffs, forum bodies, budget/rescore timing.

## Appendix

- Round-4 workflow: 5 angles + critic + synthesis, refs in session
  tool-outputs/readable-text (index 273b45c5..., synthesis
  caf2dbe3...). Terminal-value aggregate unread (duplicate of
  inline results, no effect).
- Round-4 backlog numbers: F6/F7/R3a-mine/K07/ladder/README/diff
  0 sim; E2 0 sim; R1a probe ~6k; R2 counter ~1.5k; N1 ~12k+13k;
  MoE ~78k/idea; P0-2 ~34.5k; R1a full 69k; R2 full ~160k; P0-1
  dumps ~259k + eval screens; judge-24/survivor 207k + finals
  829k honest.
