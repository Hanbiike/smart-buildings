# 2026-11-03

## Summary

I2 (D4-full, 3.02M steps, 14 cycles) finished overnight: clean run,
all guards held (cf=0, no E5 aborts, KL-stop annealing healthy),
but the curve is flat-to-down — best is cycle 0, no learning.
Branch (a) does not promote on this curve; I3 screens the c0-best
on judge-24 + b8 to close the loop, then Group J (rule levers)
and the MoE track per goal order.

## Status

| Whose | Score | State |
|---|---|---|
| leader (Pardheev Krishna, from 10-28, no fresh query) | 0.00514 | LB |
| us (occ9) | 0.01236 | COMPLETE (current best) |
| us (occ8) | 0.01260 | COMPLETE |
| us (occ_gated7) | 0.01323 | COMPLETE |

## Done

### Plan 18/I2: D4-full 3.02M curve — flat, best is cycle 0 (GREEN on execution)

- Run (launched 11-02, detached, log `/tmp/i2.log`): 25 pairs x
  8640, 14 cycles, (lr 3e-5 -> 0, mb 1024, epochs 4, v-steps 100,
  24w, seed 1, honest bill), out `/tmp/i2.pt{,.cN,.best}`. All 14
  cycles completed, final ckpt saved, no errors, no E5 aborts,
  clip_frac 0.0000 on every epoch of every cycle.
- Curve (rollout mean_r/step; x10 in `best_r`):

  ```
  cyc  mean_r     epochs (ran/stop, KL)
  0    -0.009502 BEST | e0 211, e1 stop@134 kl .0094
  1    -0.009516      | e0 211 kl .0048, e1 stop@50 kl .0107
  2    -0.009621      | e0 211 kl .0087, e1 stop@5 kl .0100
  3    -0.009756      | e0 stop@162 kl .0098
  4    -0.009892      | e0 211, e1 stop@185 kl .0103
  5    -0.009686      | e0-e3 full (kl .0001-.0084)
  6    -0.009745      | e0-e2 full, e3 stop@13 kl .0102
  7    -0.009890      | e0-e1 full, e2 stop@179 kl .0105
  8    -0.010040      | e0-e2 full, e3 stop@19 kl .0099
  9    -0.009855      | e0-e2 full, e3 stop@129 kl .0094
  10   -0.009947      | e0-e3 full (kl ≤ .0006)
  11   -0.009943      | e0-e3 full (kl ≤ .0002)
  12   -0.009969      | e0-e3 full (kl ≤ .0004)
  13   -0.009926      | e0-e3 full (kl ≤ .00003)
  ```

- Reading: KL-stop engagement is textbook (early stops, annealing
  to full epochs as lr decays) — the H-repair did its job. But
  rollout mean_r never beats cycle 0 (-4 to -5% drift-down, then
  plateau): the updates do not accumulate into better rollouts.
  Best ckpt = c0 (`best_r` -0.095019 x10 scale).
- Verdict: I2 GREEN on execution (clean 3M run, guards held, best
  retained per E3), RED on learning — branch (a) does not promote
  on this curve. I3 still runs (judge+G8 on the c0-best) to close
  honestly; then Group J per DoD(I) branch-closure rule.
- Marked I2 GREEN-execution in the plan-18 execution log.

### Plan 18/I3: judge+G8 on i2-best (shipped: PASS, honest running)

- Shipped bill, `/tmp/i2.pt.best` (c0) vs g7 + occ9, 87/87 jobs:
  judge probe 0.01441214 vs g7 0.01609473 (-10.45%) vs occ9
  0.01503697 (-4.16%); b8 probe 0.01100393 vs g7 0.01272170
  (-13.50%) vs occ9 0.01138791 (-3.37%). All gaps >> 0.0002
  floor. G8 PASS vs g7 (worst 0.975, wins every pair) and vs
  occ9 (worst 1.065).
- Note: best=c0 ≈ shipped-weights transfer + 1 cycle, and this
  is the shipped bill — D4-smoke says ~half is billing exploit.
  Honest-bill I3 (`/tmp/i3h_*`, same 87 jobs) launched to close
  honestly per B3 (screen promotions on both bills). Verdict
  after honest lands.
- Honest I3 landed (87/87; honest verified applied — CSV sums
  differ from shipped): probe vs g7 -4.95% judge / -7.35% b8
  (gaps 0.00080/0.00093, resolved); probe vs occ9 +1.32% judge
  (+0.00019989 — 1.1e-7 BELOW the 0.0002 floor, unresolved) /
  +3.26% b8 (+0.000369, resolved for occ9). G8 PASS everywhere
  (worst 0.997 vs g7, 1.105/1.091 vs occ9).
- VERDICT: branch (a) CLOSED for this cycle. Transfer is real
  (beats g7 on both bills, both coverages) but does not beat
  the current best: judge-first is unresolved-against occ9,
  b8-second loses resolved, and I2 showed no learning (best=c0).
  No promotion. Closure recorded here + ladder failed-rung note.
- Marked I3 CLOSED in the plan-18 execution log. Group I
  complete. Next: Group J (rule-side levers, branch B).

### Plan 18/J1: heating-gated ramp (policy done, suite running)

- New `smart_buildings/baselines/occ_rampgated9.py` (KEY
  `occ_rampgated9`, POLICY_MAP-registered): occ9 + C3 ramp
  applied only when OAT < band center (heating regime);
  cooling pre-mornings keep the base target. Unknown/missing
  OAT fails open (= C3 blanket) by design.
- 3 committed tests (heat applies / cooling skips / weekend+
  morning skips), 7/7 with ramp tests, 57/57 full set.
- Train-30 verdict suite (`/tmp/j1_train30`, 60 jobs,
  occ_rampgated9 vs occ_econdbhhot9) launched after I3 idled.
- Process note: J1 files were written while honest-I3 ran —
  additive-only (new files + 2-line MAP entry, running
  policies untouched, import verified); verdict data
  re-confirmed identical after landing, no corruption.
- Train-30 verdict (60/60): gated9 0.00878049 vs occ9 0.00878189,
  gap -0.00000140 (-0.02%) — dead tie, 140x below the 0.0002
  floor. Wins 7/30, ties 9/30 (bit-identical). The gate works
  exactly as designed (ALL cooling pairs tied to +0.000000 —
  C3's phx/sg damage eliminated) but heating wins are ~1e-6:
  lone suggestive point b8/denver_6 -0.000086 (-2%, still below
  floor). No regressions anywhere (worst +0.000027 phx3).
- VERDICT: J1 CLOSED-negative — gate correct, gain absent on the
  occ9 base. Likely mechanism: occ-shutdown base + occ9 stack
  leave the pre-morning target no room (the research-16 caveat
  cut the other way: their ramp evidence ran on a no-shutdown
  base). No promotion; file stays as the gated-ramp reference.
- Marked J1 CLOSED-negative in the plan-18 execution log.

### Plan 18/J2: PTR re-tune on occ9 (CLOSED, no promotion)

- 5 probes in `probe_modes.py` + POLICY_MAP (PTR_UP 0.15/0.35,
  PTR_DB 2.0, PTR_PERSIST 8, HOPT_SAT_U -0.7), 57/57. Sweep
  150/150 jobs vs J1's occ9 reference 0.00878189:
- 4 PTR probes: gaps ±1e-7 (dead ties) — PTR consts do not move
  occ9 at all on train-30; the mechanism is saturated (or
  identically-triggered) under heatopt.
- probe_hopt9_satu07: -0.00003063 (-0.35%), below the 0.0002
  floor → unresolved. Per-pair: the whole gap is one pair
  (b1/fairbanks_1 -0.000950); everything else ≤0.000032, worst
  regression denver_2 +0.000032.
- VERDICT: no promotion (mean unresolved, single-pair-driven).
  Probes stay in `probe_modes.py` as throwaways per convention.
- Marked J2 CLOSED in the plan-18 execution log.

### Plan 18/J3: dbh tails (CLOSED-negative)

- 3 probes (DB 0.10, leak 1.00, leak-only-00-06), 90/90 jobs vs
  occ9 reference 0.00878189. ALL REGRESS: db10 +0.29% (below
  floor, wrong direction), leak100 +5.23%, leaknight +5.17%.
- Finding: the 0.97 leak is LOAD-BEARING — persisting the
  integral hurts badly, and leaknight ≈ leak100 proves the
  damage comes from daytime persistence (midnight-only leak
  behaves like no leak at all). The tail-fix hypothesis was
  inverted; tails need a different mechanism, not DB/leak.
- VERDICT: no promotion. Marked J3 CLOSED-negative in the
  plan-18 execution log.

### Plan 18/J4: gentle-centre + coolshift stack (CLOSED, no promotion)

- 3 probes (csh9 / kp069 / csh9_kp06), 90/90 vs occ9 0.00878189:
  csh9 +0.17%, kp069 -0.78% (gap -0.00006832), stack -0.20% —
  ALL below the 0.0002 floor. kp069 is broad-based (17/30 wins,
  best b5/denver_3 -0.000765, worst b1/fb1 +0.000138) but
  unresolved; coolshift adds nothing (stack worse than KP alone).
- VERDICT: no promotion. Marked J4 CLOSED in the plan-18 log.

### Plan 18/J5: G5 oracle bound on occ9 (CLOSED-negative)

- First menu (PRESTART_H 0..3, b8 x5) came back bit-identical:
  the fixed horizon is dead on the occ path — `_decide`
  (occ_gated3) latches prestart adaptively via `_needed_prestart`
  and `_prestart_active` forces the PI window open, so the PI
  window start is never read. Repaired the menu onto the live
  knob PRESTART_MARGIN_STEPS {0, 6, 12, 24} (shipped 6).
- 20/20 full episodes, per-day (288-step) slicing. Fixed-best
  per pair: atlanta h1 0.006261, denver h3 0.004205, fairbanks
  h3 0.027516, phoenix h3 0.011848, singapore h0 0.006343.
  Hindsight oracle (best margin per day): mean 0.011048 vs
  fixed-best mean 0.011234 — BOUND +0.000186 < 0.0005 bar.
- Side note: single fixed MARGIN=24 beats shipped b8 mean by
  +0.000136 (phoenix 0.012424 -> 0.011848 drives it), below the
  0.0002 floor — recorded, not screened on judge.
- VERDICT: G5 stays closed, no per-day schedule head. Marked J5
  CLOSED-negative in the plan-18 log.

### Plan 18/J6: rule zone-target analogue (CLOSED, no promotion)

- 3 probes on occ9 (`_ZoneSideShiftMixin` in probe_modes.py:
  effective PI target slides toward zone temp by up to
  ZSS_SHIFT, in-band only, never flips err sign): sym 0.5K,
  sym 1.0K (their magnitude), cooling-only 0.5K. Mixin math
  unit-checked (7/7), sim-liveness confirmed, 57/57 unit tests.
- 120/120 train-30 + occ9 control 0.00878189 (exactly == J4
  control): zss05 +0.00002058 (+0.23%, heating half hurts —
  conflicts dbh), zss10 -0.00006467 (-0.74%), zssc05
  -0.00011144 (-1.27%, 29/30 wins, same-sign on all 5
  climates: atl -0.000129 / den -0.000075 / fb -0.000080 /
  phx -0.000047 / sg -0.000225; best b7/sg6 -0.000383,
  only loss b1/phx5 +0.000529). Beats J4 kp069 (-0.78%) but
  below the 0.0002 floor — no judge screen per F2, no promotion.
- Mid-analysis scare: per-pair printout columns were mislabeled
  (printed zss05/zss10/zssc05/occ9 under occ9/zss05/zss10/zssc05
  header), faking a J5-vs-J6 control mismatch. Means were
  keyed-correct throughout; direct CSV compare (0 differing
  steps) + fresh repro (h1 == occ9: fb8 0.027586, phx4
  0.012424) confirm the sim deterministic and the J5 control
  valid. Per-pair table above uses the corrected mapping.
- VERDICT: no promotion. Marked J6 CLOSED in the plan-18 log.
  Group J done (J1-J6 all closed, no promotions).

### MoE/T1: soft-spread BC vs hard+post-smooth (PASS, +0.000322)

- Harness: `slot_train.py --soft {none,k3,k5}` (k-weighted stats,
  circular mod 288, bank-hard), `--hold-building` now actually
  excluded (was parsed-but-ignored; `--no-holdout` restores),
  per-slot counts persisted in `.npz` (MoE-note gap #1 closed).
  New `tools/slot_compare.py` (held-out MSE + starved/night/edge
  diagnostics + 0.0002 gate) and `tools/test_slot_train.py` (6/6).
- Teacher dumps did not exist anywhere: generated 30/30 fresh
  (occ9 teacher — best rule, modes shared with occ7; shipped
  physics = LB physics; billing cannot move action-MSE).
  Train 25 pairs (b5 held out), ridge 1.0.
- Held-out (b5, 5 pairs) MSE hard -> soft-k3: zone 0.001863 ->
  0.001580 (-15%), ahu 0.020051 -> 0.019522 (-2.6%), glob
  0.000963 -> 0.001015 (+5% on tiny n), OVERALL 0.004904 ->
  0.004582, win +0.000322 >= 0.0002 GATE PASS. Night (slot<=72)
  0.010021 -> 0.008897 (-11%) carries the interpolation
  evidence; starved-slot diagnostic vacuous (hard zone min 10 —
  occ9 delegates at night, no empty slots). Self-check
  hard-vs-hard delta exactly 0.
- VERDICT: T1 PASS, line stays open. Logged in MoE-note §8.

### MoE/T2: k5 train-time ablation (DROP k5, keep k3)

- Same harness (k5 soft fit, k3 deploy blend): k5 vs hard win
  +0.000301 (also passes, weaker than k3's +0.000322); k5 vs k3
  directly -0.000021 overall, worse on night (0.009050 vs
  0.008897) and edge 72/228+-6 (0.005870 vs 0.005831). k5 helps
  nowhere, mild edge damage as predicted.
- VERDICT: drop k5, uniform k3. Logged in MoE-note §8.

### MoE/T3: probe-scale online screen (CLOSE the line)

- Probes: `_SlotHybrid7Mixin` + MI (SlotHybrid, OccGated7) =
  occ7 modes/run bits + slot heads on continuous;
  `probe_slot_soft7` (T1-winner k3) vs `probe_slot_hard7`
  (hard+deploy-blend) vs occ_gated7. 72/72 unit tests, MRO
  verified, sim-liveness OK. 9/9 full-8640 (b5/phx1 held-out
  bldg, b8/fb8 heating, b1/sg8 cooling).
- soft vs hard: phx1 0.007494 vs 0.007239 (1.035x, LOSS on the
  held-out pair), fb8 0.069742 vs 0.072765 (0.958x), sg8
  0.008549 vs 0.008655 (0.988x); mean -3.2% but wrong side
  where it counts. soft vs occ7: 1.584x / 2.289x / 1.161x —
  no 3x trip, but beaten everywhere; vs its occ9 teacher on
  fb8: 0.0697 vs 0.0276 = 2.5x teacher. BC compounding blowup
  persists (10-27 pattern: hybrid +243%); the T1 MSE win does
  not transfer online.
- VERDICT: CLOSE per §4 kill (offline gain evaporates; no path
  to T4's beat-occ7-by-0.0002 — fb8 probe gap is +0.039).
  T4/T5 never open (gated on T3). MoE line closed, logged §8.

## Measured

- i2.c0-best — rollout curve (25 pairs) — mean_r -0.009502 (best, cycle 0); cycles 1-13 all worse (-0.0095..-0.0100).

## Decisions

- Claim. Proof: numbers/file. Consequence: ship/close/probe.
- Promotion rule (plan F/F3): no checkpoint promoted on b8 or
  train alone — judge-first (F1 pool, 24 pairs, sha256 in
  `docs/2026/11/02/`), b8-second; gaps below the 0.0002 floor
  do not resolve (F2).

## Open

- I3: judge-24 + b8 + G8 on `/tmp/i2.pt.best` (c0), then J1 (heating-gated ramp).
- From 11-02 (plan-18 H1-H8, I1, K1-K2): DONE, all GREEN + journaled.

## Appendix

- I2 launch args: `--rollout-steps 8640 --scenarios <25 pairs,
  sha 6cf1ee90> --epochs 4 --cycles 14 --workers 24 --lr 3e-5
  --lr-end 0 --minibatch 1024 --v-steps 100 --seed 1 --bill honest
  --out /tmp/i2.pt`. Full trace: `/tmp/i2.log`.
