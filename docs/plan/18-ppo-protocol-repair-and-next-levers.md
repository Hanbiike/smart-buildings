# PPO-protocol repair and next levers

Date: 2026-11-02. Status: draft.
Scope: repair the PPO update protocol that KL-stopped after 19/844
minibatches, then either run D4-full (branch A) or pivot to rule-side
levers on occ9 (branch B). Source for every Expected value is the
19/844 diagnosis (prior result 3), its critic (prior result 4), the
gating/judge inventory (prior result 1), and the occ9/closed-doors
inventory (prior result 2). Coverage tags: offline (no sim), probe
(1-2 pairs x 512-2016 steps, epochs=1, single cycle, minutes), train-30
(6 buildings x 5 climates x 8640 steps), judge-24 (xfer_5..10 x 4
weathers, sha e71ea1a7...). Loss convention:
`abs(df.agent_reward.sum()) / len(df)` per run CSV.
Rule: no 3-5M-step run until the H-group gate passes at one (lr, mb)
setting; no promotion except judge-first/b8-second with the 0.0002
floor (F3) and the G8 3x screen.

## Group H. PPO-protocol repair: cheap probes first (all probe-scale)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| H1 | Observability-only probe: per-mb KL/k3/clip_frac/fr.sum()/grad-norm/|dmu|/dlogstd trace | add per-mb prints to `tools/skynet_ppo.py::ppo_epoch` (fr.sum, global grad norm before clip at :155, |dmu|, dlogstd, k1 + k3 KL, clip_frac from :108-112); run `tools/skynet_train.py` with `--lr 3e-6 --epochs 1 --cycles 1` on 1-2 pairs x 512-2016 steps; console already prints ran_mb/stopped/mean_kl/clip_frac at `tools/skynet_train.py:253-254`, capture it | stop no longer fires at 3e-6; trace establishes the noise floor and decides: instantaneous-vs-running-mean noise, whether the Adam-step mechanism holds or the global grad-norm clip binds (critic included, :155), and whether E5 (clip_frac > 0.15 in first 10 updates, `tools/skynet_train.py:257-259`) would also have tripped | P0 |
| H2 | Fix v-pretrain-to-0 bug, then re-probe at fixed lr | `tools/skynet_train.py:217-221` builds batch0 with `ret=torch.zeros`; fix to pass real GAE/MC returns into `tools/skynet_ppo.py:166-180` (slot_ppo fits on MC returns, `tools/slot_ppo.py:341-349`); rerun the H1/H3 probe at fixed lr | value MSE drops vs zeros-fit; early-mb grad-norm trace calms; GAE runs from a real baseline instead of v~0. MUST-FIX before any long run. Note: old_lp recomputed at `tools/skynet_train.py:210-215` is NOT stale (actor/critic trunks fully separate, `nets.py:30,37-38,46-49`; critic_params critic-only, `tools/skynet_ppo.py:36-38`; surgery adapters disjoint, `skynet_surgery.py:51-54`) | P0 |
| H3 | LR sweep at mb=256: find the largest lr surviving a full epoch | `tools/skynet_train.py --lr {3e-5,1e-5,3e-6} --mb 256 --epochs 1 --cycles 1` on the H1 probe pool; log ran_mb/844-equivalent, clip_frac, mean KL | largest lr with full-epoch survival and clip_frac < 0.05 wins. CAUTION (critic): 3e-5 is a hypothesis, NOT proven -- the slot_ppo run at (3e-5, 1024) DIVERGED 0.040 -> 0.076 (`docs/ladder.md:162-165`, line CLOSED), and `tools/skynet_ppo.py:28` LR=3e-5 is DEAD CODE (driver builds Adam from `--lr`, `tools/skynet_train.py:146`, never reads ppo.LR except self-test :273). Expect ~10x more mbs than 3e-4, but let the trace decide | P0 |
| H4 | Minibatch 1024 vs 256 at the surviving lr | same probe pool, `--mb 1024` vs `--mb 256` at H3-winning lr; compare per-mb KL trace variance + fr.sum() distribution (`tools/skynet_ppo.py:121-123` pg mean runs over free steps only) | 1024 shows lower per-mb KL variance; low-fr.sum() minibatches identified as high-variance steps. Current default 256 (`tools/skynet_train.py:91`) vs slot_ppo 1024 (`tools/slot_ppo.py:73`) | P0 |
| H5 | Logstd floor (slot_ppo-style clamp) | logstds are learned Params (`nets.py:34-36`) entering `masked_lp` unclamped (`tools/skynet_ppo.py:54-67`); add clamp at init-minus-~0.5 mirroring `tools/slot_ppo.py:398,417-418`; rerun H3 probe | dlogstd channel closed on long runs. CAUTION (critic): confirmed real longer-run hazard, but CANNOT explain mb-19 (19 steps x 3e-4 move logstd <= ~0.006 vs shipped stds 0.21-0.33: zone [-1.29,-1.47], ahu [-1.54,-1.28,-1.12,-1.40], plant [-1.31,-1.10,-1.34] in `/tmp/opencode/sb-rl/skynet.pt`) | P0 |
| H6 | KL-stop rule: keep instantaneous or switch to running-mean | read `tools/skynet_ppo.py:159-162` (instantaneous fr-masked per-dim k1 KL vs kl-stop 0.01 at :32 / driver :97, aborts epoch; driver `tools/skynet_train.py:260-261` breaks the CURRENT cycle's epoch loop only -- "all epochs" true only for `--cycles 1`) vs `tools/slot_ppo.py:424-433` running-mean check inside the mb loop (own comment :424 "Per-minibatch trust region", NOT epoch-end as first implied) | switch to running-mean form only if the H1 trace shows single-mb noise trips; otherwise keep instantaneous. CAUTION (critic): mean(old_lp-new_lp) over behavior-policy samples is an UNBIASED estimator of KL(old\|\|new) -- "k1 biased low, true KL even larger" does NOT follow; the clip_frac co-movement trace (H1) is the real arbiter of genuine-overshoot-vs-false-stop | P1 |
| H7 | GAE bootstrap fix (minor, cheap) | `tools/skynet_ppo.py:85` bootstraps 0.0 at traj end; switch to last-value bootstrap like `tools/slot_ppo.py:358`; unit-test on a synthetic traj | direction right, magnitude SMALL (critic: GAE horizon 1/(1-0.995*0.95) ~= 18 steps, error negligible beyond ~100 steps/traj, not ~600; slot_ppo's own bootstrap is also biased, just less so). Fix while touching the file, not a gate | P1 |
| H8 | Behavior-vs-updated screen on the probe pool | short rollout with SAME seeds: behavior ckpt vs post-epoch ckpt from H3/H4 winner on the same 1-2 pairs; compare mean reward/step + KL/clip_frac budget | ship the update only if updated >= behavior on mean reward/step AND KL/clip_frac in budget. Record the exact 25-pair geometry + CLI flags of any 844-mb-scale rerun: 844x256 => T in (215808,216064], 25x8640=216000 fits exactly and E2 records an independent "25 per-traj" smoke (`docs/2026/11/02/README.md:507-508`), but WHICH 25 pairs/flags is unconfirmed and the default CLI (4x512=2048 steps = 8 mbs/epoch) CANNOT reproduce the event | P0 |

Definition of done (H): H1 trace captured and pasted in the day note
(KL/k3/clip_frac/fr.sum/grad-norm/|dmu|); H2 fix committed with a test;
one (lr, mb) setting passes H3+H4+H8; H5 clamp committed; H6 decided
with trace evidence; H7 fixed or explicitly deferred. No Group I run
starts until this gate is green.

## Group I. Branch (a): D4-full gate and long run (expensive only)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| I1 | D4-full launch gate | verify H-gate green + E3 best-ckpt+LR-decay (`tools/skynet_train.py:149,153-156,263-275`) + E4 pool checksum (`:74-78,:132-133`) + E5 clip-abort (`tools/skynet_ppo.py:108-112,132-134,158,161-163`; driver `:243-259`) active; `--scenarios` keeps xfer_5..10 OUT (judge holdout; default pool is judge-clean but trains on 2 SEEN transfer-12 pairs, declared SEEN `docs/2026/11/02/README.md:560-562`); zone-count monster check on every pool member (xfer_4 125-zone precedent, `docs/2026/11/02/README.md:560-565`); conc <= 10, 1500s+ timeouts; no policy edits in flight (AGENTS.md) | all boxes ticked in the day note; pool sha + n logged before cycle 0; resume/ckpt paths named | P0 |
| I2 | D4-full: 3-5M-step gated-residual PPO on the H-winning recipe | `tools/skynet_train.py` with H-winning (lr, mb) + H2/H5/H7 fixes, honest bill (B3), keep-best-not-last, ckpt every ~250k | curve logged per cycle (mean reward/step, clip_frac/epoch, KL stops); no post-peak drift; best ckpt retained as `{out}.best` with {cycle,lr,best_r} | P1 |
| I3 | Promote only via judge-24 + G8 | regenerate git-ignored judge pool if missing (`tools/gen_transfer.py 4 6`, seeds 50004-50009, xfer_5..10, sha e71ea1a7...); rebuild `/tmp/f1_judge.txt`; `./.venv/bin/python -B tools/fullrun_t.py ...` judge-24 + b8 for candidate vs occ9/g7; `tools/promote_screen.py` (exit 1 if any pair > 3x rule, exit 2 on missing; 4 tests in `tools/test_promote_screen.py`) invoked manually post-suite | judge-first/b8-second, 0.0002 floor (F3: `docs/TEMPLATE.md:25-28`, `docs/ladder.md:17-19`); F2 precedent (g7 < ramp < g6, b8 tie, floor 0.0002); G8 PASS required (F2 real-data worst 1.175/1.106). Fresh judge+G8 per new RL ckpt | P1 |

Definition of done (I): I1 gate recorded; I2 curve + best ckpt logged;
I3 judge/G8 verdict recorded; promotion or closure written to
`docs/ladder.md` (R4/R5). If I2 fails the H-gate twice at all probed
(lr, mb), close branch (a) for this cycle and work Group J.

## Group J. Branch (b): rule-side alt levers on occ9 (if PPO fails)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| J1 | Heating-gated ramp (new variant, not the C3 port) | new `smart_buildings/baselines/occ_*rampgated*.py` + `POLICY_MAP`: ramp only where the plant heats (cooling-side damage was the whole C3 regression: blanket ramp train-30 +0.60% overall, heating wins atl -0.26%/den -1.12%/fb -0.52%, cooling loses phx +3.31%/sg +2.48%, worst b1/phx5 +7.67%, `docs/2026/11/02/README.md:239-263`); `./.venv/bin/python -B tools/fullrun.py results/... 8640 8 1500 <variant>,occ_econdbhhot9 train30` | must beat occ9 on train-30 mean AND worst climate (C3 precedent). C3 counter-evidence on THEIR base (`docs/research/16-skynet-residual-ppo-notebook.md:174-176`) does not close this: their base lacks occ shutdown, so the gate is still untested here. Base: occ9 = occ_econdbhhot9 (occ7+econ+dbh+heatopt), public 0.01236 (-6.6% vs occ7), occ8 0.01260, local train-30 -8.8%, transfer ~75% of local (`docs/2026/10/28/README.md:210-215,231-232`); ablations econ -1.67%/dbh -5.39%/heatopt -2.21% (`:235-239`); fresh-holdout 16/16, -11.47% (`:240-244`) | P1 |
| J2 | PTR re-tune jointly with heatopt-POPT on occ9 | PTR (occ_gated7) constants PTR_UP=0.25/DOWN=0.03/MAX=0.0/DSAT=0.85/DB=1.0+slope 0.1/K, ref 10, persist 4 (`smart_buildings/baselines/occ_gated7.py:25-32,61-84`) were tuned on occ4/occ7 and never re-tuned under dbh/heatopt; heatopt POPT raises pressure by a second path (`smart_buildings/baselines/occ_heatopt.py:108-119`); small grid on occ9, train-30 verdict | any adopted change needs same-sign gain on >=2 buildings; trim=0-on-hidden precedent (occ7 public 0.01323 exact tie, `docs/2026/10/28/README.md:296-297`) | P2 |
| J3 | DBH midnight-tail fix (evening pre-charge / knob sweep) | dbh raises midnight tails 0.0021 -> 0.0030 (`docs/2026/10/28/README.md:235-239`); sweep DBH_DB/DBH_LEAK (`smart_buildings/baselines/occ_dbh.py:24-26`, never swept on occ9); probe evening pre-charge shape; train-30 verdict | fb1 tail (1.5 -> 2.6%) is the target; prestart-shape precedent is CLOSED-negative in blanket form (`docs/2026/10/28/README.md:118-120`; RL warm-up levers worse every climate by 500k, research/16:416-420), so shape changes need per-climate sign checks | P2 |
| J4 | Banked stack: gentle-centre + coolshift7 on occ9 (prop-v3 EXCLUDED) | gentle-centre -1.0%, coolshift7 ~-0.9% on 4 hot cells (phx6 -2.9%, tails better), prop-v3 ~-0.2% (`docs/2026/10/28/README.md:118,129-154`); stack gentle+coolshift on occ9, train-30 verdict | coolshift may duplicate econ on the cooling side -- verify, do not assume; prop+coolshift combo stays REJECTED (double-shift +0.3..+2% hot). Gap-forensics suspects behind these: occupied efficiency 15-25%, prestart shape 8-15%, excursions 5-15%, soft-start+coast 3-8% (`docs/research/05-gap-forensics.md:331-346`) | P1 |
| J5 | G5 oracle bound recomputed on occ9 day-type structure | cheap bound first: per-day oracle upper bound on occ9 (no head built); build the per-day schedule head ONLY if bound >= 0.0005 | current G5 closure rests on THEIR variants (-0.00013 b8 / -0.00007 b7, research/16:412-415); recompute before re-closing. Stays closed if bound < 0.0005 | P1 |
| J6 | Zone-target rule analogue (their one working structural lever) | their zone-target head shifts PI target +-1K in-band (Fairbanks 0.02563 vs rule 0.02633, Phoenix ~0.0101 vs 0.01050; residual could not move damper/reheat through PI-integral cancellation, research/16:289-303,328-337); rule-side analogue partially present (gen31 down-shift centre-up-to-1.0K, `docs/ladder.md:74-75`); D5 GO hook exists (`_cur_vav` context + per-step offset dict + in-band clip); design a rule-side in-band target shift on occ9, train-30 verdict | needs its own gain loop (D5 finding); keep in-band +-1K; compare against the J1/J4 winners, not in isolation | P2 |

Definition of done (J): each attempted lever has a train-30 mean +
worst-climate verdict in the day note; adopted changes get a
`docs/ladder.md` rung, rejected ones close with numbers (R4/R5).
Doors that stay closed without new evidence: evening coast, AHU
staging, satcap, DCV, hot-port dup, C4 set (`docs/2026/10/28/README.md`
items 4/5/9/10b, :216-218; `docs/2026/11/02/README.md:265-279`); G1
hourly hold, G2 all-hours/full-authority/+2K, G3 nets>d64x2 until D4
peaks, G4 learned switches, G6 weight averaging, G7
credit/graph-att/cold-pools/paired-reward (research/16 9.x +
`docs/2026/11/02/README.md:646-662`); CMA 30-point grid null (+0.00007
b8, research/16:373; warm-start variant b8 0.01063, :159-161).

## Group K. Ledger and hygiene (unblocks honest scoring)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| K1 | Repair ladder + journal drift | `docs/ladder.md` submission log shows occ6/occ7 PENDING and no occ8/occ9 rungs vs COMPLETE scores (occ6 0.01337, occ7 0.01323, `docs/2026/10/28/README.md:296-297`; occ8 0.01260, occ9 0.01236, `:231-232`; `docs/2026/11/02/README.md:20-21` agrees); `docs/2026/11/02/README.md:23` says occ8/occ9 "submitted, scores pending" (stale side); E3 journal lines 143/146 drifted vs current 153-156 | ladder gains occ8/occ9 rungs + 4 COMPLETE scores; 11-02:23 corrected; rung-11 "current best" reflects the econ/dbh/heatopt stack (-6.6% public); E3 line refs fixed | P1 |
| K2 | Harden the ephemeral chain (judge list + zone checker) | judge pool recipe + sha committed (F1), but pool dir is git-ignored and `/tmp/f1_judge.txt`, `/tmp/f2_judge`, `/tmp/f2_b8` are ephemeral (documented /tmp wipe, `docs/2026/11/02/README.md:29-30`; all present on this machine today, critic re-verified, incl xfer_1..10 with xfer_4 dir retained-but-excluded); zone-count checker is ephemeral `/tmp/zone_count.py` | commit a `tools/zone_count.py` (or equivalent) + a one-command judge-regen recipe (`tools/gen_transfer.py 4 6` + list build) so K-group + I3 survive a /tmp wipe; re-verify F2 ranking agreement after any regen | P1 |

Definition of done (K): ladder/journal agree on all COMPLETE scores;
ephemeral chain reproducible from committed tools in one command each.

## Priority summary

P0 (protocol gate, days 1-3, all probe-scale): H1-H5, H8, I1.
P1 (branch decision, days 3-10): H6-H7, I2-I3, J1, J4-J5, K1-K2.
P2 (only if P1 branch stalls or is cheap): J2-J3, J6, plus any G-door
reopen that meets its stated bar with fresh numbers.

## Execution log

- 2026-11-02 H1 GREEN (day note `docs/2026/11/02/README.md`):
  per-mb trace (frsum/gnorm/k1/k3/cf/dmu/dlogstd) via `--trace`;
  8/8 mbs at 3e-6, no stop, cf=0; DECISIVE: pre-clip gnorm
  32-43 binds the max-norm-1.0 clip (H2 bug), Adam-step
  mechanism falsified as stated.
- 2026-11-02 H2 GREEN with caveat (same day note): batch0 ret =
  MC gamma-returns (lam=1/zeros) + `MCReturnsTest`; 52/52;
  re-probe gnorm 32-43 -> 4.5-9.4, v-pretrain 89.3 -> 6.9 on
  real returns; clip STILL binds at v-steps 20.
- 2026-11-02 K1 GREEN (same day note): ladder rungs 10/11
  COMPLETE (0.01337/0.01323), rungs 12/13 added (occ8 0.01260,
  occ9 0.01236 current best), 4 log rows COMPLETE; 11-02 status
  + E3 line refs fixed; no PENDING remains.
- 2026-11-02 K2 GREEN (same day note): `tools/zone_count.py`
  committed (48/18 zones match F1) + `tools/regen_judge.sh`
  one-command regen (syntax-checked, sha path re-verified
  e71ea1a7...). Group K complete.
- 2026-11-02 H3 GREEN (same day note): 8-mb probe inconclusive
  (8/8 all lrs); 64-mb rerun: 3e-4 stops at 23/64 (reproduces
  19/844 family, cf=0 so E5 silent), 3e-5/1e-5/3e-6 run 64/64.
  Winner 3e-5 (mean_kl 0.00028, cf=0).
- 2026-11-02 H4 GREEN (same day note): mb=1024 k1 std 3.2x
  lower than 256 (0.000048 vs 0.000155), max 3.9x lower; both
  full epochs, cf=0. PICK 1024. frsum-starvation unconfirmed
  (corr ~0). Clip still binds (gnorm to 268) — carried to H6/H8.
- 2026-11-02 H5 GREEN (same day note): MIN_LOGSTD=-2.0 +
  `clamp_logstd()` before step + clamp-at-use in `masked_lp`;
  `ClampLogstdTest` (float32 test bug fixed); 53/53 + self-test;
  no-op probe clean, ckpt logstds all above floor.
- 2026-11-02 H6 GREEN (same day note): 3e-4 trace k1 climbs
  0.0000->0.0110 over 23 mbs (sustained, mb22 already 0.0089),
  cf=0 throughout — genuine overshoot, keep instantaneous stop.
  No code change.
- 2026-11-02 H7 GREEN with deviation (same day note): terminal
  rollouts make 0-bootstrap CORRECT, so added `bootstrap=0.0`
  param (default) instead of switching; tail-decay unit test;
  54/54 + self-test. No behavior change.
- 2026-11-02 H8 GREEN (same day note): 3 cycles x 16 mbs all
  survive, cf=0, rollout mean_r +0.18%; scale-0 screen bug
  caught+fixed; movement real but noise-scale (1e-5, mixed).
  H-GATE GREEN: recipe (3e-5, mb 1024, honest, best-ckpt).
  Residual risks: clip still binds; weak-positive learning.
- 2026-11-02 I1 GREEN (same day note): E3/E4/E5/H5 verified;
  pool 25 pairs (b8 holdout, no xfer) sha 6cf1ee90; all 5
  buildings valid (36-85 zones); args (3e-5, mb 1024, ep 4,
  14 cycles = 3.02M, 24w with deviation note); ETA ~17.5h.
- 2026-11-03 I2 GREEN-execution, RED-learning (day note
  `docs/2026/11/03/README.md`): clean 3.02M/14 cycles, cf=0, no
  aborts, textbook KL annealing; curve flat-to-down, best = c0
  (-0.009502). Branch (a) does not promote; I3 closes the loop.
- 2026-11-03 I3 CLOSED (day note `docs/2026/11/03/README.md`):
  shipped PASS (probe -10.5%/-13.5% vs g7, beats occ9 too);
  honest: beats g7 (-5.0%/-7.4%) but unresolved-against occ9
  on judge (+0.00019989, below floor) and loses b8 (+3.26%).
  G8 PASS all. Branch (a) closed for this cycle. Group I done.
- 2026-11-03 J1 CLOSED-negative (day note `docs/2026/11/03/`):
  `occ_rampgated9` (OAT-gated ramp on occ9) train-30 gap
  -0.00000140 (-0.02%, dead tie); cooling damage eliminated
  (all tied) but heating wins ~1e-6. No promotion.
- 2026-11-03 J2 CLOSED (day note `docs/2026/11/03/`): 5 probes,
  150/150; 4 PTR consts dead ties (±1e-7); satu07 -0.35% below
  floor, single-pair-driven (b1/fb1 -0.00095). No promotion.
- 2026-11-03 J3 CLOSED-negative (day note `docs/2026/11/03/`):
  db10 +0.29%, leak100 +5.23%, leaknight +5.17% — leak is
  load-bearing, daytime persistence is the damage. No promotion.
- 2026-11-03 J4 CLOSED (day note `docs/2026/11/03/`): csh9
  +0.17%, kp069 -0.78% (broad 17/30, best -0.000765), stack
  -0.20% — all below floor. No promotion.
- 2026-11-03 J5 CLOSED-negative (day note `docs/2026/11/03/`):
  first menu (PRESTART_H 0..3) bit-identical — fixed horizon is
  dead on the occ path (_decide latches adaptively). Repaired
  menu on PRESTART_MARGIN_STEPS {0,6,12,24}, 20/20 b8x5:
  per-day hindsight oracle bound +0.000186 < 0.0005 bar. G5
  stays closed. Side note: fixed MARGIN=24 beats shipped b8
  mean by +0.000136 (phoenix-driven), below floor — not run
  on judge.
- 2026-11-03 J6 CLOSED (day note `docs/2026/11/03/`): 3 probes
  (zss05/zss10/zssc05) + occ9 control, 120/120 train-30, occ9
  0.00878189 (== J4 control, bit-consistent). zss05 +0.23%,
  zss10 -0.74%, zssc05 -1.27% (gap -0.00011144, 29/30 wins,
  same-sign all 5 climates) — bests J4 kp069 but below the
  0.0002 floor. No promotion. Mid-analysis control scare
  resolved: mislabeled print columns, means were keyed-correct;
  repro (h1==occ9 bit-equal) confirms J5 control valid.
