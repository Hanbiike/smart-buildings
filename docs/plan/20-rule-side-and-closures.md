# Plan 20: rule-side and closures (post-RL)

Date: 2026-11-04. Status: draft.
Source: deep-research round 3 (5 angles + critic + gap follow-up +
synthesis; workflow completed partial with named gaps below),
grounded in plan-19 (RL line CLOSED: H1/H2/H3/H5/P10 killed,
H4 offline-only, H6 unopened), `docs/research/21` (D1-D3 closed,
D4 + danielwuu queued), and plan-18 group J (all levers closed).
Conventions (plan-19 carried): loss = abs(sum)/len per run CSV
(lower-better); gaps reported as loss_base - loss_new (+ = gain);
0.0002 floor (F3); judge-first/b8-second; G8 3x guard
(`tools/promote_screen.py`); short runs are protocol-only, never
verdicts; conc <= 10, timeouts 1500/5400; pool sha + bill + seeds
logged per run (P9 ledger continues).
Execution order is numeric. Conditional items open only on their
stated trigger. Distillation D4 rejoins here (P20-4); danielwuu
pieces ride inside N1.1 (P20-7) per synthesis.

## Group T. RL post-mortem (close the books)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P20-1 | T1 frozen-det control (TD3BC Step 1) | NEW `tools/frozen_det_screen.py` serial driver (commit as tool): `rollout_env(deterministic=True)` (`tools/skynet_rollout.py:54-66,127-131`) with weights `/tmp/opencode/sb-rl/skynet.pt` vs `force_rule` arms; `building_train_8/fairbanks_8` + `building_train_7/phoenix_3` config.gin; seed 31, 8640 steps, honest bill. P10 base arms (fb8 -260.4863, phx3 -62.9343) re-run, not reused (determinism cross-check) | gap_frozen_det vs TD3BC gaps (phx3 +0.0007337, fb8 +0.0001769) within 0.0002 on BOTH = (a) determinism CONFIRMED, (b)(c) dead, STOP RL training. frozen ~= rule both + TD3BC-phx3 gap reproduces > 0.0002 = (a) DEAD -> P20-11. Mixed = report magnitudes | P0 |
| P20-11 | T2 BC0 ablation (CONDITIONAL: only if P20-1 kills (a)) | `tools/td3bc_fit.py` ALPHA=0 one-line edit, same fit seed 0 / 5000 steps, out `/tmp/p10/bc0.pt`, same deterministic screen as P20-1 | bc0-det within 0.0002 of frozen-det while td3bc-det leads > 0.0002 on phx3 = (c) garbage-Q luck WINS by elimination (needs 2nd-seed replication before any claim); bc0 ~= td3bc = (b) BC WINS (surprising, revisit) | P3 |

## Group E. Enablers (P0, ~0 sim)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P20-2 | N2.1 occupancy-seed variation rig | Verify the 17321-replay claim first (critic: single-file check is insufficient): gin-bind `randomized_occupancy` seed per scenario (`smart_buildings/smart_control/simulator/randomized_arrival_departure_occupancy.py:190` default; template `scenario_generator.py:1049-1056`; `gin.bind_parameter` per `evaluate.py:620`); >= 3 seeds (17321 + 2 fresh); regen via `tools/gen_transfer.py`, `tools/gen_holdout.py`, `tools/regen_judge.sh`; seeds logged per run | >= 3 seed-realizations per pool; seed-17321-only finals BANNED without it | P0 |
| P20-3 | N3.1 PR#3-equivalent local patch | Local-only airflow-weighted recirculation-temp patch at `simulator_flexible_floor_plan.py:318-334 + :740-759` (only variant in repo is area-weighted); mirror `tools/honest_bill.py:24-49` pattern + unit tests mirroring `tools/test_honest_bill.py` (idempotent, dynamics-untouched); shipped default unchanged | patch + tests green; semantics-unverified until PR bodies read (gap-4 stands) | P0 |

## Group D. Distillation remainder

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P20-4 | D4 error-reserve probe (facilities v42) | NEW `probe_errres025` on occ9 (`probe_modes.py`, Wknd6Probe `_zone_band` pattern): per-zone errEMA init 0.08, update 0.94/0.06 on \|temp-target\| clipped 2.0K; extra = min(clip(0.5*(err-0.08),0,0.25), 0.25*max(hi-lo,0)); apply lo += extra, hi -= extra in comfort mode only (`_comfort_now`/`_pi_active`); verdict `tools/fullrun.py` train-30 vs occ9 (D1-D3 pattern, 120/120) | train-30 mean gain >= 0.0002 AND >= 2 climates same-sign AND no Fairbanks blowup; else CLOSE like D1-D3, no judge below floor | P1 |

## Group R. Rule-side levers on occ9 (all NEW variants; J1/J2/J5/coast/staging/DCV doors stay closed)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P20-5 | R1 occupancy-conditional OFF (best gain/sim) | NEW `smart_buildings/baselines/occ_offX.py` on occ9 + POLICY_MAP: (a) occupancy-EMA-conditional evening early-shutdown + arrival-grace tighten reusing `_occupied` (`occ_gated.py:92-103`, OCC_EPS) + 24h-EMA skip pattern (`zone_pi.py:753-768`) on the latch (`occ_gated3.py:78-83) -- NOT closed wall-clock coast (`occ_coast.py:31-41`); (b) unoccupied HWS full-off via `_shutdown_action` (`occ_gated.py:226-246`) + pump idle-off (`zone_pi.py:659-669`) + reset integrator (`pid_plus.py:526-535`); verdict `tools/fullrun.py` train-30 (plan-18 J1 command verbatim) | 2-pair screen first (~17k); train-30 mean gain >= 0.0002 AND >= 2 climates same-sign AND no Fairbanks blowup AND no comfort tail (fb tail cap +0.0005); holds across P20-2 seeds; G8 | P1 |
| P20-6 | R2 self-tuning T&R (NOT fixed re-tune -- J2 closed) | NEW `smart_buildings/baselines/occ_ptrX.py` on occ9, each sub-lever separately gated: (a) normalize PTR_PERSIST/UP by zones-per-AHU in the live loop (`occ_gated7.py:39-85`); (b) G36 rogue-zone exclusion (single-zone starve hold does not advance `_ptr_nstarve`); (c) self-tune db_eff from trailing request-rate EMA vs fixed PTR_DB/SLOPE/REF | 4-cell prescreen (~35k); surviving sub-levers to train-30 (>= 0.0002 same-sign, >= 2 buildings, worst-climate screen, b8-second/judge-24 hold -- the clause fixed consts failed) | P1 |
| P20-8 | R3a hindsight oracle bound (kill-before-build gate for R3) | Per-day hindsight oracle for the R3 menu (SAT depth x pressure cap x recirc bias x HWS frac) on occ9, 8h bins (96 steps), J5-style 20x b8x5 (~172.8k) | bound >= 0.0005 (J5 bar) before building the head; else kill R3 UNBUILT | P2 |
| P20-13 | R3b receding-horizon build (CONDITIONAL: only if P20-8 >= 0.0005) | NEW `smart_buildings/baselines/occ_planX.py` on occ9: every 96 steps re-solve tiny per-AHU menu against the adaptive rate predictor (`occ_gated.py:138-161`), fed by forecast inputs; PI tracking untouched; writes via demand-SAT/PTR/econ-recirc/HWS-reset; arithmetic enumeration only (10x bang-bang cap, wall-clock unmeasured -- verify) | train-30 >= 0.0002 + >= 2 climates + worst-climate screen; judge hold | P2 |

## Group N. Addendum-A finals (N1-N3)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| P20-7 | N1.1 hybrid-arm build at fixed budget | Port research/21 danielwuu pieces (OFF-candidate, pressure=fraction^2, water hysteresis) as probe policies + POLICY_MAP keys (probe specs in round-3 index result 3); tune ONLY via `tools/g4bo.py`-style BO on 5-scenario screens; PG arm = frozen residual at EQUAL honest-step budget (P7 777.6k precedent); sim ledgers logged, no PG sims in hybrid arm | both arms built, ledgers logged; budget-mixing invalidates P20-9 | P1 |
| P20-9 | N1.2 same-budget hybrid-vs-PG finals (needs P20-7 + P20-2) | `tools/fullrun_t.py` 8640 conc <= 10 on judge-24 + holdout-16 + b8, HONEST_BILL=1, loss abs(sum)/len per run CSV | hybrid wins total + comfort split on judge-24 AND b8, gaps >= 0.0002, consistent across pools; else KILL N1, compute stays rule-side | P1 |
| P20-10 | N2.2 promotion finals (applies to any survivor) | `tools/fullrun_t.py` 8640 on judge-24 x seeds + holdout-16, honest bill; splits via `tools/xfer_analyze.py` + summary.json | improve ALL pools incl unseen-climate comfort; cross-seed |delta| < 0.0002; no climate comfort regression vs occ9; else KILL | P1 |
| P20-12 | N3.2 pre/post-fix + gateway-parity (needs P20-3) | occ9 + top probes on b8 + judge-24 under shipped vs honest vs honest+PR#3 (HONEST_BILL=1 hook); gateway-parity replicates `kaggle_evaluation/gateway.py` MAE/scenario semantics; G8 each candidate | honest delta ~0.00003-scale (Mima, UNVERIFIED -- confirm or drop); local-vs-gateway gap < 0.0002 or harnessed; tuning regressing > 0.0002 on corrected sim = KILL | P1 |

## Open gaps (carried + new from round 3)

1. PR #1/#2/#3 bodies uninspected (plan-19 gap-4 stands): N3.1 patch is site-located but semantics-unverified; honest_bill is PR#2-approximate (rule honest 0.01207 vs PR#2 0.01178 split documented).
2. Forum thread bodies (749531/752142/753804/749896/752678/751998) absent: Mima/cycling/phantom numbers are digest-text only, not bars.
3. N2 cf claim (K07 tuning -16.1%) ungrounded in repo: not a bar.
4. TD3BC fit seed 0 (mine, logged here -- closes the research gap); P10 screen driver was `/tmp/p10_screen.py` scratch (P20-1 commits a tool version).
5. Second-seed replication of phx3 +0.0007337 absent (single seed 31).
6. 8h-enumeration wall-clock vs 10x cap unmeasured (P20-13 must verify).

## Execution log

- 2026-11-04 P20-1 running (day note): NEW
  `tools/frozen_det_screen.py` committed tool (serial
  deterministic-mean vs force_rule screen, --weights/
  --tag/--seed/--steps); --help + 74/74 green.
  Frozen-det screen (fb8+phx3 x 8640 x 2 arms =
  34,560 steps, honest, seed 31) running in
  background; verdict vs TD3BC gaps (+0.0007337/
  +0.0001769) when it lands. No code edits until
  then.
- 2026-11-04 P20-1 done: (a) determinism CONFIRMED,
  stronger than predicted (same note). Frozen-det
  gaps vs rule: fb8 +0.0016810, phx3 +0.0007738.
  phx3 within floor of TD3BC (+0.0000401 diff);
  fb8 BEATS TD3BC by 0.0015041 (9.5x) -- training
  HURT vs frozen-det there (garbage-Q direction),
  determinism is the whole signal and then some.
  (b)(c) dead; STOP RL training stands. Base arms
  reproduce P10 bit-identically (fb8 -260.4863,
  phx3 -62.9343) -- determinism cross-checked.
  NOTE (not a promotion): frozen-det beats
  occ_gated7-based rule on both climates above
  floor -- first live gain signal; needs occ9-base
  train-30 + judge + G8 as a rule-side artifact in
  a future round (logged in day-note Open).
  P20-11 SKIPPED (conditional on killing (a);
  (a) confirmed). 34,560 sim, honest, seed 31.
