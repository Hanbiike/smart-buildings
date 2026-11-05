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

- P21-11 E2 bars (next), P21-12 R3a mine, P21-13 MoE blocked,
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
