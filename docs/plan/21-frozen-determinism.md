# Plan 21 — frozen determinism (round-4 research)

Follows plan-20 (COMPLETE, no survivors, no promotions) and deep
research round 4 (5 angles + critic + synthesis; critic `complete=false`
— angles 0/1/2 unvetted by critic, re-vet on intake via gates below).

## Headline correction (critic-caught)

P20-10:344 says frozen-det "beats occ9-rule". The P20-1 screen base was
`POLICY_MAP[occ_gated7]` (`tools/frozen_det_screen.py:24-51`), NOT occ9.
Vs-occ9 is UNTESTED. P21-1 re-screens vs occ9 first (kill-before-build);
everything downstream dies if the gap does not hold there.

## Standing gates (law)

0.0002 floor; judge-first/b8-second; G8 3x (`tools/promote_screen.py:17`);
short runs protocol-only; full episode 8640; STOP RL (no training —
frozen/BC-distillation only; P20-1 fb8 9.5x training-hurt).

## Group F. Frozen-det over occ9 (P0, head)

| ID | Check | How | Gate / kill |
|---|---|---|---|
| P21-1 | F0 occ9-base 2-pair screen | `tools/frozen_det_screen.py` on fb8+phx3 x 8640 vs occ9-rule (== occ_econdbhhot9; no separate occ9 key), seed 31 + seed 99999 replica (34,560 x 2 = 69,120 honest sims); base-vs-base parity first | Gap >= 0.0002 same-sign both pairs AND both seeds; else KILL the frozen line here (no dumps, no distillation) |
| P21-2 | F0b mechanism dump (diagnostic) | dump-obs both-arm run frozen-det vs rule, per-channel SAT/pressure/recirc/HWS/switch-votes compare (~34.5k) | Names carrier channel(s); if no channel differs, P0-1 premise breaks (KILL) |
| P21-3 | F1 distillation screen | Build occ9-based deterministic-mean residual Policy with embedded weights (probe_xres is occ7/disk/env — not submittable); dump frozen-det train-30 (~259k); BC-train ControllerNet via bc_train/nn_tokenizer/nn_policy d64 CPU path (0 training sims) | train-30 >= 0.0002, >= 2 climates same-sign, fb-pair delta > -0.0002 (pinned no-blowup bar); then judge-24 confirms + G8 3x + notebook 300+ steps incl night/shutdown/prestart + step-time bar + kernels-push smoke for torch+200KB base64; any fail = KILL |

## Group X. Transfer-gap frozen fixes (P1, cheapest-first)

| ID | Check | How | Gate / kill |
|---|---|---|---|
| P21-4 | F6 input-repair audit, 0 sim | Frozen forward harness; judge-vs-train clip-+-10 hit-rate | Build only if judge rate > train rate; predicted KILL |
| P21-5 | F3 scale grid + F7 offset clamp | Per-building scale {0,0.25,0.5} on 7-day screens (xfer_5/sg2, xfer_10/msp1, xfer_7/msp1); scale=0 doubles as zt_off-base-only ablation; F7 clamp/zero target_offsets A/B on xfer_5/sg2 7-day | scale<1 beats scale=1 by >= 0.0002 on >= 2/3 probes; F7 >= 0.0002; if scale=0 wins everywhere kill residual, redirect to base |
| P21-6 | F2 recovery suppression + F4 occ-gate | any_cold return-base/boost (probe_modes.py:48-65); occupancy-gate PI hours | F2: xfer_10/msp1 Monday prod within 3x of occ9; F4: zero-occ weekday energy <= occ9; else KILL each |
| P21-7 | F1 floor-heat + F5 guard (CONDITIONAL: only if P21-5 or P21-6 passes) | Off-hours floor-heat wrapper; GuardProbe fallback to occ9 | F1: Mon-arrival min zone >= 12C at weekend energy <= occ9, then judge-24 gap cut >= 50% no sg regression > 0.0002; F5: 0 triggers b8/fb8 AND >= 50% Monday-regret cut xfer_10/msp1; else KILL |

## Group S. Seed rescreens (P1)

Seeds 17321 + 99999 (+424242 proposal); OCC_SEED runtime rebind, no regen.

| ID | Check | How | Gate / kill |
|---|---|---|---|
| P21-8 | R1a fresh-seed rescreen | 1500-step trigger-counter probe (~6k), then OCC_SEED=99999 2-pair screen (34,560); full 69,120 shipped | counter > 0 AND gain >= 0.0002 same-sign both pairs; else KILL |
| P21-9 | R2 T&R fire-scan | Fire-counter xfer_3/msp1 x1500 (~1.5k) + full-8640 C-parity on 2 train pairs; full ~160k | Any fire + scored gain >= 0.0002; fire == 0 everywhere = KILL |
| P21-10 | N1 singles scored liveness | 2-pair x1500 per single (~12k), then n1bo-harness 5x864 honest per single (~13k) | Single wins screen >= 0.0002; all <= floor = KILL |

## Group O. Offline probes (P2)

| ID | Check | How | Gate / kill |
|---|---|---|---|
| P21-11 | E2 RvS collapse probe, 0 sim | Return-conditioned MLP on existing 30 dumps (259,200 steps); PIN numeric bars first (action spread + held-out loss); correct premise: dumps are CURRENT-best teacher | Escalate to 2-pair screen ONLY if passes; expect KILL |
| P21-12 | R3a 0-sim mine (menu-scoped) | Mine R3a 20x b8x5 CSVs (outside repo — locate first) for per-pair sat-arm wins | Sweep MILD_SAT only if a pair won; forecast->R3-menu stays dead; non-menu forecast needs a NEW menu clearing 0.0005 |
| P21-13 | MoE revival (BLOCKED until compounding fix) | Harness intact; revival must lead with compounding fix (slot_hybrid +243%), not better fit | Compounding fix first; ~78k/idea 9-run screens after |

## Group B. Backlog 0-sim closes (P2, anytime)

| ID | Check | How |
|---|---|---|
| P21-14 | Doc/gap closes | Drop K07 cf clause (one-line plan edit); ladder leader row 0.00470 boys; research/01 header pointer to research/21; 11/04 README Summary/Open/Measured refresh; honest_bill bound diff (bound-not-close); 31-vs-33 count reconciliation |

## KILLED — do not run

VLM-RL (tabular obs, images metrics-only, CPU-only notebook);
offline actor-critic (P10 2/2 fail); end-to-end foundation control;
R3 menu v1/v2/v3.1 as-is; full-scale residual on judge without F-gates;
Mima 0.00003 (P20-12 dropped).

## Needs-external (not in this plan)

PR #1/#2/#3 diff bodies; forum thread bodies (6 IDs); submission
budget + corrected-sim/rescore timing (Kaggle API/host watch).

## Execution log

- 2026-11-05 plan-21 written from round-4 synthesis (7/7 results read;
  terminal-value aggregate unread duplicate, no effect).
- 2026-11-05 P21-1 running. Added --base to
  tools/frozen_det_screen.py (default
  occ_gated7, backward-compat). Smoke 1500
  occ9-base OK (fb8 base -30.2186 vs det
  -35.8223, phx3 -9.1100 vs -9.1809 --
  protocol only). Full screens seed 31 +
  99999 (69,120 honest) running serial in
  background. No code edits until they land.
- 2026-11-05 P21-1 done: FROZEN LINE KILLED
  (same note). occ9-base screens landed:
  fb8 base 0.027419 vs det 0.030931 (gap
  -0.003512, wrong sign, 17x floor); phx3
  base 0.006412 vs det 0.006397 (gap
  +0.000015, below floor). Frozen-det beats
  occ7-rule (+0.00168 P20-1) but LOSES to
  occ9-rule -- the round-4 headline signal
  was a weak-base artifact (P20-10:344 slip
  confirmed critical). Seed-99999 replica
  bit-identical to seed 31 (rollout seed
  does not touch occupancy -- degenerate
  replica; true second seed needs OCC_SEED,
  moot now). P21-2 + P21-3 SKIPPED (gated
  behind P21-1). 69,120 honest sims.
- 2026-11-05 P21-4 done: F6 KILLED (same
  note). Clip-+-10 hit-rate: train 0/60M
  (30 dumps) vs judge 0/46M (24 fresh
  dumps x1500) -- judge NOT greater than
  train. Input scale is healthy; transfer
  gap is elsewhere. 36,000 sims (dumps;
  plan said 0-sim -- judge obs did not
  exist, corrected).
- 2026-11-05 P21-5 running. F3/F7 probes
  + keys + 4 tests green. Liveness xfer_5
  sg2 x1500: s0/s025/s05 DIFFERS (mono:
  0.0338/0.0279/0.0247 vs s1 0.0230 --
  smaller scale worse here, protocol only),
  noto IDENTICAL. F7 CLOSED pre-screen
  (0 gate sims): ckpt zone_mu is (2,64) --
  NO target head, mz.shape[1]>2 never true,
  target_offset never set (mechanism
  absent, R1b-style). F3 7-day screens
  (3 pairs x 4 scales x 2016 = 24,192)
  running in background. No code edits
  until they land.
- 2026-11-05 P21-5 done: F3 KILLED, F7
  CLOSED pre-screen (same note). F3 12/12:
  scale<1 beats scale=1 by >=0.0002 on
  1/3 probes only -- xfer_10/msp1 all win
  (s0 -0.001999, s025 -0.001747, s05
  -0.001455); xfer_5/sg2 + xfer_7/msp1 all
  lose (s0 worst both). Pattern: residual
  hurts in deep cold (msp1), helps
  elsewhere -- but gate needs 2/3. Scale=0
  is NOT a universal base win (no redirect).
  31,692 sims (24,192 screens + 7,500
  liveness).
- 2026-11-05 P21-6 done: F2 KILLED, F4
  CLOSED vacuous (same note). NEW
  SkynetColdProbe (any-cold k=1.0K base
  fallback) + SkynetOccProbe (OCC_GATE)
  + keys + 5 tests green. Liveness
  xfer_10/msp1 x1500: F2 0.013745 vs
  skynet 0.016705 (DIFFERS, better),
  F4 == base BIT-IDENTICAL -- base
  _pi_hours already excludes weekends
  (day<5), weekday zero-occ is nights
  only: gate vacuous on-window. F4
  mechanism proven live (synthetic Wed
  noon: occ=0 -> False, occ=5 -> True)
  but trigger P(empty weekday) ~=
  (e^-2)^N ~= 1e-44/day (N~=53,
  work_occupancy=3/zone) + 0 triggers
  in ~40 scanned episodes -> CLOSED
  gain-0.0 by construction (P20-6
  precedent). F2 full-episode screen
  xfer_10/msp1 x8640: Monday regret
  -0.055 vs occ9 -0.0024 = 22.7x
  (gate 3x) -- first Monday good
  (-0.0038) then explodes (-0.15,
  -0.07, -0.05); full +0.009 worse
  than occ9. 21,780 sims (4,500
  liveness + 17,280 full). P21-7 DEAD:
  conditional on P21-5/6 pass, both
  killed.
