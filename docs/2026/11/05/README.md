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

- Plan-22 action (next): P22-6 RESID residual slot.
  P21-14 backlog closes (0-sim, anytime).
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
