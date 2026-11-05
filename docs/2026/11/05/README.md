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

No new sims today yet (research + planning only).

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

- P21-1 occ9-base 2-pair screen (seed 31 + 99999 replica).
- P21-14 backlog closes (0-sim, anytime).
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
