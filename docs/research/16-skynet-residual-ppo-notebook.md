# Skynet residual PPO: notebook and dataset reconstruction

Date: 2026-11-01. Status: draft.
Scope: exhaustive reverse-engineering of the participant notebook
`yes-rl-can-run-a-building-residual-ppo.ipynb` (versions 1-16, author alexchilton)
and its dataset (`/tmp/rl_zip/`, tag `m3r_ramp_cmagen_seed1@752640`):
architecture, features, trainer, every reported curve and table, the billing-bug
audit, the failure table, and transfer limits to our Online RL on `occ_gated7`.
No simulator episode, training job, or benchmark was run for this report.
Numbers below are notebook/dataset-reported unless marked Recomputed
(recalculated from `curves.json` values without simulation).

## Sources

| # | Source | Content used |
|---|---|---|
| S1 | `yes-rl-can-run-a-building-residual-ppo.ipynb`, markdown cells 0,1,4,6,8,10,12,14,16,18,20,22,24,26 | all prose, tables of sections 0-7 |
| S2 | same notebook, code cells 2,3,9,11,13,15,17,19,21,23,25,27 + text outputs | PPO settings, per-climate splits, zone-target table, submission cell |
| S3 | `/tmp/rl_zip/README.md` | usage, training recipe, file map |
| S4 | `/tmp/rl_zip/MANIFEST.json` | checkpoint tag, full train args, base constants, SHA-256s |
| S5 | `/tmp/rl_zip/TAG.txt` | `m3r_ramp_cmagen_seed1@752640 SkynetPolicy` |
| S6 | `/tmp/rl_zip/features.py` | field lists, scales, dims, `compose`, `Latch` |
| S7 | `/tmp/rl_zip/nets.py` | `FlatNets`, critic, log-prob masking, padding |
| S8 | `/tmp/rl_zip/policy.py` | `SkynetPolicy`, `load`, gating, hold |
| S9 | `/tmp/rl_zip/base.py` | `Floor`/`ZTFloor`/`ZTOffFloor`/`ZTOffAllFloor`, `make_floor` |
| S10 | `/tmp/rl_zip/zt_controller.py` | full rule stack down to `ZTPolicy` + research knobs |
| S11 | `/tmp/rl_zip/env.py` | worker protocol, `rollout`, latch/vmask, paired reward |
| S12 | `/tmp/rl_zip/skynet_train.py` | PPO loop, GAE, losses, LR schedule, checkpointing |
| S13 | `/tmp/rl_zip/skynet_eval.py` | held-out eval, `AvgNets`, selection sets, billing switch |
| S14 | `/tmp/rl_zip/sbrl.py` | `honest_bill()` patch, `make_base`, legacy `ResidualPID` |
| S15 | `/tmp/rl_zip/cma_search.py` | CMA-ES over `pid_water` constants |
| S16 | `/tmp/rl_zip/curves.json` | every plotted checkpoint score, board history, gas hours |
| S17 | `/tmp/rl_zip/hist_cma*.jsonl` (4 files) | tuned constants for each `--base` |
| S18 | `/tmp/rl_zip/skynet.pt` (metadata via `torch.load`, no training) | cfg, shapes, 48,277 params |
| S19 | `/tmp/rl_zip/LICENSE-NOTICE.txt` | Apache-2.0 provenance chain |

## 1. What it is

One shared MLP per entity type (zone / AHU / plant, d64 x 2 layers, tanh)
outputs a bounded additive correction to a hand rule; the last layer is
zero-initialised so step 0 of training is exactly the rule (S1 cell 4, S7
`FlatNets.__init__`). The rule is malyshevdanil's public per-zone PI
controller `zt` inside its PI hours plus an off-hours rule outside them
(S1 cells 4-8, S9 `ZTOffFloor`). The residual acts only in PI hours; the
network output is ignored at night/weekends (S8 `get_action`, `free` flag).
Shipped board score 0.01128, rank 147 of 461 on 2026-10-31 (S1 cell 0).

## 2. Scoreboard (all versions, held-out building 8 + public board)

Notebook cell 0 table, held-out column on shipped bill (S1):

| weights | held-out b8 | board |
|---|---:|---:|
| public per-zone PI (zt) | 0.01358 | 0.01420 |
| hand rule under RL (zt + off-hours) | 0.01251 | - |
| residual seed 0, 2.0M | 0.01136 | 0.01237 |
| residual seed 0, 2.5M | 0.01100 | 0.01245 |
| residual seed 0, 3.0M (nb v1) | 0.01065 | 0.01224 |
| residual seed 1, 4.75M (nb v2) | 0.01051 | 0.01216 |
| residual seed 1, 5.0M (nb v3-v9) | 0.01040 | 0.01199 |
| zone-target RL on pre-start ramp rule, honest bill, 250k (nb v10) | 0.01017 | 0.01132 |
| same run at 750k, picked on 20 held-out generated buildings (nb v11/v13, shipped) | 0.01026 | 0.01128 |
| v10 weights retrained 750k on shipped + 720 generated (nb v12) | 0.01033 | 0.01133 |

Board-vs-held-out ratios for 10 submissions (S2 cell 19 output):
1.113, 1.153, 1.157, 1.149, 1.132, 1.089, 1.093, 1.077, 1.072, 1.046;
board sits 5-15% above held-out b8 with the same order (S1 cell 18).
Board history rows (`what`, held_out, board) in S16 `board` (10 rows).

Per-climate split of the 5.0M checkpoint vs rule on b8, shipped bill (S2 cell 17):

| climate | rule | RL 5.0M | delta |
|---|---:|---:|---:|
| atlanta_1 | 0.00864 | 0.00579 | -33.0% |
| denver_6 | 0.00519 | 0.00469 | -9.6% |
| fairbanks_8 | 0.02989 | 0.02490 | -16.7% |
| phoenix_4 | 0.01233 | 0.01050 | -14.8% |
| singapore_7 | 0.00651 | 0.00610 | -6.3% |

Second fold (train holds out building 7 instead, seed 0, stopped 2.0M):
0.00758 vs rule 0.00828, -8.6% (S1 cell 26; S16 `rl_b7_seed0` last row
Recomputed mean 0.00758; `rule_b7` Recomputed mean 0.00828).
On training building 7 the 5.0M weights score 0.00726, -12.4% vs rule
(S1 cell 26). On training building 1 (7 days, 5 climates): 0.01411 vs
0.01609 for the rule (S1 cell 26).

## 3. Network and features

3.1. Topology (S7, S18). `FlatNets(d=64, depth=2, zone_out=3)`:
actor = 3 independent MLPs (Linear+ tanh x2, heads zero-init);
`zone_mu`(64->3), `ahu_mu`(64->4), `plant_mu`(64->3).
Exploration: per-output learned `log_std` (init -1.0, i.e. std 0.37),
Gaussian sampling; no entropy bonus in the loss (S12 `loss = pg + vf_coef*vf`).
Critic: separate encoders `c_zone/c_ahu/c_plant` (same shape as actor),
masked-mean pooling over zones and AHUs, head `Linear(192->64)->tanh->Linear(->1)`.
Total parameters 48,277 (S2 cell 9; S18 Recomputed 48,277).
`from_checkpoint` accepts 2-output zone checkpoints (pre-zone-target head) (S7).

3.2. Feature widths (S6): zone 21, AHU 34, plant 22.
Zone = 6 own-sensor fields + 2 band errors + 2 PID damper/reheat proposal
+ 2 AHU context (supply temp, run flag) + 9 global.
AHU = 15 own + 6 pooled zone stats (mean err, cold frac, warm frac,
mean PID damper, mean PID reheat, max err) + 4 PID proposal
(temp/pressure/recirc/run) + 9 global.
Plant = 7 own + 3 pooled (mean err, cold frac, mean AHU run)
+ 3 PID water temp/pressure/pump + 9 global.
Global 9 = hod_sin/cos, dow_sin/cos, comfort_mode_now/soon, occupants/125,
(OAT-295)/10, in-comfort-hours flag (S2 cell 9, S6 `features`).
Fixed physical scales per field (no running stats; zone temp denormalised
with metadata constants) (S6 header comment).

3.3. Outputs (S6 `compose`, S1 cell 8). Per zone: damper, reheat,
target offset (x1.0 K per unit, PI clips to band, S6 `TARGET_K`).
Per AHU: supply temp, static pressure, recirculation, run vote.
Plant: water temp, water pressure, pump vote.
`action = clip(rule + scale * correction, -1, 1)`, shipped scale 1.0
(S4 train_args, S6 `compose`). Switch votes pass a +-0.5 dead zone;
inside it the vote defers to the rule (S6 `compose`).

3.4. Latch and hold (S6 `Latch`, S8, S11). A vote past +-0.5 sets a switch
for `hold` steps (shipped latch=1, i.e. plain dead-zone vote, S4/S5);
while held the vote is masked out of log-prob and entropy via `vmask`
(S7 `log_prob_entropy`). Separately, `--hold` (shipped 1) holds a sampled
residual N steps; only the sampling step carries a log-prob (S11 `rollout`,
`fr` mask; S12 loss masks by `fr_t`).

3.5. Gating (S8/S9/S11). The residual is computed and applied only when
`floor.free` (zt `_pi_active`); outside PI hours the off-hours rule acts
alone (all continuous -1, recirculation +1, all switches off), latch
untouched, `vmask` zeros, held residual cleared (S11 `step`).

## 4. Base rule under the residual

4.1. Stack (S10): `PIDWaterResetPolicy` -> `BaselinePolicy`
(sensor band, decoded OAT) -> `CoolingHoldPolicy` (HOLD_STEPS=288)
-> `ZonePIPolicy` (per-zone PI on band centre in PI hours, forced AHUs on)
-> `ZonePIEconFixPolicy` (KP 0.10, KI 0.005, kelvin OAT fix)
-> `DemandSatPolicy` (supply temp follows zone-PI demand quantile)
-> `TunedPIPolicy`/`TunedDemandSatPolicy` -> `AdaptiveColdPolicy`
(OAT-EMA regime: prestart 2h/4h, water-max, damper gain, mild shift)
-> `APResearchPolicy` (zone-shift EMA, ramp, target-offset interface,
idle/water/weekend knobs) -> `ZTPolicy` (submitted constants:
KP 0.1, KI 0.005, HOLD_STEPS 288, NIGHT_COOL_CEILING False,
ZONE_SHIFT_K 0.6, DEMAND_QUANTILE 0.5, ZONE_EMA_STEPS 144,
WEEKEND_EXTRA_DROP_K 6.0, PRESSURE_MIN_ALL True).

4.2. PI window (S1 cell 6, S10 `_pi_hours`): weekdays from
morning_start - PRESTART_H (2h, 4h in deep cold via EXTREME_BELOW_C)
to evening_start - EVENING_STOP_H; network ignored outside.

4.3. Shipped tuned constants (`zt_off_cmagen_ramp`, S4/S17 `hist_cma_zt_gen`,
gen 12, fitness 0.01203084): KP 0.020318, KI 0.013228,
ZONE_SHIFT_K 0.976552, DEMAND_QUANTILE 0.301305, MILD_SHIFT_K 0.272104,
EXTREME_HEAT_DAMPER 0.996690, ZONE_SHIFT_FULL_U 0.050184,
FORCED_SAT -0.830619, plus RAMP_K 1.0 added by the base alias (S12
`zt_off_cmagen_ramp`). CMA-tuned rule alone: b8 0.01109 honest
(S12 docstring; S16 `honest.cma_rule` Recomputed mean 0.01109),
b8 0.01077 / b7 0.00734 with ramp (S12 docstring).
Warm-start CMA variant (`hist_cma_zt_warm`, 13 levers incl. PRESTART_H
2.502, EXTREME_PRESTART_H 5.994, MIN_DAMPER 0.0375): b8 0.01063 /
b7 0.00723 with ramp (S12 docstring). Plain CMA-zt (`hist_cma_zt`,
gen 12, best 0.00492361 on its train window) is a different
(short-window) fitness scale, not comparable to full-episode scores.

4.4. Pre-start ramp (S10 `_zone_target`, RAMP_K=1.0): on weekdays before
`morning_start_hour`, zone target sits K/h x hours-to-morning below the
band, floored at RAMP_FLOOR_K=289.0. Residual target offsets clip
0.3 K inside the band by default (`TARGET_OFF_MARGIN_K`, S10).

4.5. Rule-only ablations that failed (S1 cell 26, b8/b7 honest):
AHU idle-off (IDLE_U 0.02-0.2): 0.01109 -> 0.01109-0.01134 (b8),
0.00767 -> 0.00767-0.00796 (b7), null/worse; reheat-first sequences:
best -0.00008/-0.00002, strict +0.00049 on b7, null; Monday early
warm-up 1h: -0.00006/+0.00007, worse at 2-3h, null; ramp+cool-zone
raise at 1 K/h: +0.00083/+0.00085, worse, reverted; ramp-only-where-heats:
-0.00020..-0.00034 (b8) vs -0.00032/-0.00033 plain ramp, reverted;
softer PI (KP/KI halved): -0.00015 b8 but +0.00012 b7, sign flips;
1.5h pre-start: +0.00006/-0.00010, sign flips.

## 5. Trainer

5.1. PPO settings of the shipped run (S2 cell 11, S4 train_args):
n_steps 512, workers 6, epochs 4, minibatch 256, lr 3e-4 (Adam eps 1e-5),
gamma 0.995, GAE lam 0.95, clip 0.2, vf_coef 0.5, reward_scale 10.0,
scale 1.0, d 64, depth 2, seed 1, max_steps 8640, ckpt_every 10,
eval_every 250000, latch 1, hold 1, reward raw, holdout building_train_8.
Rollout per update = 3072 steps; one 30-day episode = 8640 steps (S2).
Warm-started from `m2_zonetarget_cmagen_seed1/ckpt_4002816.pt` (S4 `init`);
`--init` grows narrower checkpoints with zero rows/cols so the wider net
computes exactly the checkpoint until trained (S12).

5.2. Rollout scheme (S11 `rollout`, S12): each worker samples with its own
net copy and steps until a shared counter reaches `n_steps x workers`;
fast buildings contribute more steps; nobody waits per step.
Scenarios: 5 train buildings x 5 climates = 25, full 30-day episodes back
to back, shuffled per worker with seed (S12 `scenario_lists`).
Generated buildings optional via `--gen-cold/--gen-other` from
`data/gen/manifest.jsonl` excluding `skynet/selection_set.json` (S12
`gen_scenarios`; neither ships in the dataset, see 9.6).
Throughput 110-135 steps/s on laptop CPU, 6 workers (S3);
5M steps therefore ~10-13 h wall clock. Async variant 140-200 steps/s
vs ~120 lockstep cited as in-progress (S1 cell 26, section 6).

5.3. Learning details (S12): GAE per worker bootstrapped at rollout cut;
advantages normalised per worker over free (residual-acted) steps only;
policy loss masked by `fr_t` and normalised by its sum; value loss
unclipped MSE; grad-norm clip 0.5; LR decays linearly to 0 at `--steps`
(S12 line 168; B1 experiment in S1 cell 26 confirms the schedule removes
post-peak drift without adding gain). No entropy bonus, no KL term.
History line per update to `logs/skynet/<run>/history.jsonl` with
sps, reward_mean, explained_var, pg/vf loss, entropy, clip_frac,
log_std per head, overrides/1k, finished-episode scores (S12).
Checkpoints `latest.pt` every 10 updates, snapshot every ~250k steps
(actual: multiples of 3072, e.g. 251904) plus background `skynet_eval.py`
on the holdout (S12).

5.4. Failure-mode telemetry quoted (S1 cell 26, section 5): cutting noise
to 1/4 without lowering LR gives clip fraction 0.19-0.34 after 5 updates;
seed-0 entropy falls 15.6 -> 8.4 over the overtraining drift.

## 6. Evaluator and checkpoint selection

6.1. `skynet_eval.py` (S13): mean actions (no sampling), up to 20 sim
processes at once, `--floor` mode for the rule alone, `--bill
honest|shipped` (rows to `eval_honest.jsonl` vs `eval.jsonl`),
`--set selection|selection60` for 20/60 held-out generated scenarios
(7 days = 2016 steps), per-step rewards saved to `.npz`, `AvgNets`
averages mean actions (not weights) across checkpoints.

6.2. Selection-set methodology (S1 cell 26, 4d): 20 generated buildings
(4 per board climate, 7 days each), never trained on; validates by
ranking m3r above m4 on both bills (shipped: 0.010841 vs 0.011005,
gap 0.000164 vs leaderboard gap 0.00018) while b8 ranked them reversed.
Checkpoint table, held-out generated honest bill (S1):

| checkpoint | gen-20 honest | Fairbanks | b8 honest |
|---|---:|---:|---:|
| m3r + 720 gen, 750k (v12, board 0.01133) | 0.010744 | 0.02832 | 0.01031 |
| m3r @ 750k (v11/v13, board 0.01128) | 0.010958 | 0.02938 | 0.01050 |
| m3r @ 250k (v10, board 0.01132) | 0.011044 | 0.02968 | 0.01032 |
| m3r @ 500k | 0.011170 | 0.03033 | 0.01036 |
| m4 @ 250k (board 0.01150) | 0.011109 | 0.02984 | 0.01019 |
| m4 + 1M | 0.011411 | 0.03101 | 0.01039 |
| m4 + 1M + rule-state inputs | 0.011727 | 0.03045 | 0.01021 |

Re-scoring: m3r@750k best on both bills (shipped 0.010709 vs 0.010841
for 250k, better in all 5 climates) while b8 ranks it last of the three;
zone-target run re-scored 0.011195/0.011258/0.012095 at 3M/4M/5M (S1).
Training-pool experiment from m3r@250k, same seed/settings (S1):

| pool | 250k | 500k | 750k | 1M |
|---|---:|---:|---:|---:|
| 7 shipped only | 0.011117 | - | - | - |
| + 200 generated | 0.011009 | - | - | - |
| + 720 generated | 0.010868 | 0.010859 | 0.010744 | 0.010870 |

Fairbanks at 250k: 0.03018/0.02940/0.02880. Shipped bill: 720-run@750k
0.010639 vs v11 0.010709, yet board 0.01133 vs 0.01128: a 0.00007 judge
gap is below the 20-building resolution; gaps >= 0.0002 rank correctly (S1).
B1 schedule run (linear LR decay, 2048 steps/worker): 0.011027 / 0.010997 /
0.010956 / 0.010939 / 0.011004 / 0.010979 / 0.010973 at
258k/504k/762k/1.01M/1.25M/1.51M/1.76M; flat 1M-2M within 0.00007 (S1).
Lesson: hash the training pool (two machines drew pools sharing only
175 of 720 buildings; same command+seed diverged) (S1).

## 7. Billing bug and honest bill (section 4b)

7.1. Mechanism (S1 cell 20, S14 `honest_bill`): shipped boiler bills flow x
(Tsetpoint - Treturn) while coils draw at tank actual Ts; with Tsp < Ts
heat leaves unbilled (forum 752142, organisers' fix PR #2). The 5.0M
weights drew 914 kWh more than billed over 30 Fairbanks days, unprompted (S1).
Honest re-bill: flow billed at max(Ts, Treturn) + dissipation + tank term;
dynamics unchanged, observations byte-identical for 1500 steps (S1, S14).

7.2. Numbers, held-out b8 (S1 cell 20):

| bill | rule | RL 5.0M | gain |
|---|---:|---:|---:|
| shipped (board today) | 0.01251 | 0.01040 | -16.9% |
| honest (tank temp) | 0.01207 | 0.01079 | -10.6% |
| PR #2 (energy-balanced tank) | 0.01178 | 0.01072 | -9.0% |

0.00211 shipped gain vs 0.00128 honest => ~39% of shipped gain is the bug
(S1). Delivered Fairbanks heat falls 7.2%: physics, not accounting (S1).
All later training/comparison on honest bill; board still shipped sim (S1).
Second forum bug (area-weighted return-air temp, discussion 753804) not
used on purpose; damper pickup unmeasured (S1 cell 26).

## 8. What the RL cannot reach, and the zone-target fix (section 4c)

8.1. Gas audit (S1 cells 22-24, S16 `gas_hours`, Fairbanks b8 30 days):
99.8% of gas burns in residual-eligible hours; 42% at 04:00-06:59 warm-up
(S2 cell 23). Yet RL saves ~nothing daytime; Fairbanks 0.02633 ties the
tuned rule. Causes: warm-up start is a rule constant; AHUs forced on
(always heating >=30% outside air, recirc capped 0.7); PI integral cancels
damper/reheat pushes; net sees neither PI target nor integral (S1).
Cost map (S1 section 6): weekday 01:00-08:00 = 61% of Fairbanks,
73% of Atlanta; 19:00-01:00 + weekends ~free; Fairbanks cost is 2/3 gas
carbon; boiler losses 0.3-0.5%.

8.2. Fix: third zone output shifts the PI target +-1 K inside the band so
the loop helps (S1, S6 `target_offsets`, S10 `target_offset`).
Zone-target run from 5.0M on CMA rule, honest bill, full table (S2 cell 25):

| steps | mean | atlanta | denver | fairbanks | phoenix | singapore |
|---:|---:|---:|---:|---:|---:|---:|
| 251,904 | 0.01148 | 0.00686 | 0.00561 | 0.02582 | 0.01253 | 0.00660 |
| 500,736 | 0.01111 | 0.00662 | 0.00508 | 0.02606 | 0.01163 | 0.00615 |
| 752,640 | 0.01088 | 0.00677 | 0.00519 | 0.02579 | 0.01057 | 0.00609 |
| 1,001,472 | 0.01091 | 0.00683 | 0.00528 | 0.02578 | 0.01045 | 0.00621 |
| 1,250,304 | 0.01087 | 0.00683 | 0.00539 | 0.02575 | 0.01025 | 0.00616 |
| 1,502,208 | 0.01078 | 0.00654 | 0.00500 | 0.02585 | 0.01036 | 0.00616 |
| 1,751,040 | 0.01081 | 0.00654 | 0.00511 | 0.02595 | 0.01029 | 0.00615 |
| 2,002,944 | 0.01099 | 0.00656 | 0.00552 | 0.02607 | 0.01040 | 0.00639 |
| 2,251,776 | 0.01108 | 0.00662 | 0.00535 | 0.02688 | 0.01026 | 0.00629 |
| 2,500,608 | 0.01132 | 0.00673 | 0.00571 | 0.02795 | 0.01007 | 0.00612 |
| 2,752,512 | 0.01151 | 0.00679 | 0.00566 | 0.02881 | 0.01019 | 0.00611 |
| 3,001,344 | 0.01110 | 0.00665 | 0.00540 | 0.02729 | 0.01008 | 0.00609 |
| 3,250,176 | 0.01093 | 0.00666 | 0.00524 | 0.02657 | 0.01015 | 0.00601 |
| 3,502,080 | 0.01094 | 0.00670 | 0.00490 | 0.02707 | 0.01020 | 0.00583 |
| 3,750,912 | 0.01087 | 0.00668 | 0.00482 | 0.02693 | 0.01014 | 0.00578 |
| 4,002,816 | 0.01062 | 0.00663 | 0.00492 | 0.02563 | 0.01012 | 0.00578 |
| 4,251,648 | 0.01092 | 0.00670 | 0.00501 | 0.02699 | 0.01010 | 0.00579 |
| 4,500,480 | 0.01107 | 0.00676 | 0.00495 | 0.02782 | 0.01006 | 0.00576 |
| 4,752,384 | 0.01128 | 0.00686 | 0.00514 | 0.02841 | 0.01014 | 0.00587 |
| 5,001,216 | 0.01116 | 0.00673 | 0.00542 | 0.02775 | 0.01011 | 0.00580 |

Reference: RL 5.0M mean 0.01079; tuned rule alone 0.01062-claim context
0.01109 (S2 cell 25). Readings (S1 cell 26): Fairbanks best 0.02563 at
4.0M vs 0.02633 rule; Phoenix -> ~0.0101 vs 0.01050; mean best single
0.01062 (4.0M) but checkpoint sd ~0.00025, best 3-avg 0.01080 vs 0.01079;
drift 1.5M-2.75M to 0.01151 then recovery. Building 7 check (trained-on):
5.0M 0.00749 vs zone-target@4.0M 0.00755; Fairbanks better (0.00949 vs
0.01015), Denver worse (0.00745 vs 0.00581). Caveat: 4.0M picked as best
of 20 b8 evals; its 19 evals from 750k average 0.01098 (sd 0.00023) vs
rule 0.01109 (S1). Successor m3r (4.0M + 250k on ramp rule): honest b8
0.01032, b7 0.00712, board 0.01132 (S1).

## 9. What failed (section 5, full table, held-out b8 shipped bill unless stated)

Verbatim numbers from S1 cell 26; bill/scope notes preserved:

1. RL on CMA-tuned pid_water: 0.03184 held-out, board 0.03414; residual
   cannot move far from a bad base.
2. RL on CMA-tuned zt: 0.01196 at 500k, worse every checkpoint to 0.01275
   at 1.25M; tuned base starts better (0.01187) but RL never beats it.
3. 1-hour action hold: 0.01265-0.01345, worse than every-step; gain is in
   fast reaction (S16 `rl_hour_hold` rows Recomputed means 0.01341 /
   0.01265 / 0.01345).
4. Paired reward (minus zt shadow) + low noise + 1h switch latch on plain
   zt: 0.01298-0.01307 flat 750k-1.5M; small gain over zt 0.01358 at 2x
   sim cost; off-hours base far better.
5. Noise/4 without LR cut: clip fraction 0.19-0.34 after 5 updates; LR
   must fall with noise.
6. Stopping at the 1M dip would keep 0.01203 instead of 0.01065;
   held-out bounces ~+-0.0005 between checkpoints.
7. Seed 0 past best (3.0M -> 6.75M): 0.01065 -> 0.0110-0.0115;
   HYPOTHESIS: overfit of 25 train scenarios; entropy 15.6 -> 8.4.
8. Trusting the mean: seed 1 at 3.0M mean 0.198, Fairbanks 0.96 alone
   (S16: 0.96343 / 0.56061 / 0.55288 at 3.00M/3.25M/3.50M, recovered
   0.02699 at 3.75M); screen worst climate.
9. Single 5-min switch vote ~ 1/100 of reward noise; barely learnable.
10. Seed 1 past 5.0M: 0.01040 -> 0.01090 (5.25M) -> 0.01111 (5.5M);
    keep best, not last.
11. Weight averaging 4.0M-5.0M: 0.01054 vs 0.01040; not one flat basin.
12. Residual at every hour (not only PI): 0.01079/0.01073/0.01078 at
    250k/500k/750k from 5.0M; off-hours rule near its floor.
13. Residual on CMA-tuned rule from 5.0M: 0.01126 at 250k; weights fit
    the untuned rule.
14. Bigger nets on Kaggle CPU (d128x3, d256x3, d512x3 vs d64x2): best
    0.01229 / 0.01237 / 0.01340 by 0.75-1.25M; d256x3 rose to 0.01426.
15. 30-point grid over rule constants: best +0.00007 vs shipped on b8; null.
16. Lower noise + matching LR from 5.0M: 0.01038/0.01042/0.01044 at
    250k/500k/1M vs 0.01040; null, closed.
17. +1000 generated buildings from 5.0M (honest): 0.01079 at 250k = start;
    null at first check, stopped.
18. Zone-target head on untuned rule from 5.0M (honest): 0.01090/0.01094
    at 250k/500k vs 0.01079; Fairbanks gain at 250k (0.02613) gone by
    500k (0.02642). Works better on tuned rule (section 4c).
19. Face-value gain: 39% was the tank bug (section 4b).
20. Zone-target head to 5M (honest): best 3-avg 0.01080 vs 0.01079;
    0.01151 at 2.75M; Fairbanks gain, no mean gain.
21. Cold-weighted generated pool (20 cold : 2 other) from 5.0M, shipped
    bill: 0.01057/0.01094 at 250k/500k vs 0.01040; stopped.
22. Warm-up-start CMA (8 of 12 gens): best 0.011372 vs 0.011715 start
    (-2.9%) on 17 train scenarios x 7 days; b8 check pending.
23. Learned on/off switches (Bernoulli per AHU + pump, from 4.0M):
    0.01079 at 250k, then Fairbanks 0.02604 -> 0.07143 -> 0.56009 at
    750k; learned to switch heating off in deep cold; stopped.
24. Graph attention over one AHU's zones (Kaggle, scratch, shipped):
    0.01313/0.01375/0.01398/0.01398/0.01773 at 0.25-1.25M; stopped.
25-31. Rule ablations: see 4.5 (idle-off, reheat-first, Monday prestart,
    ramp variants, softer PI / shorter prestart).
32. Residual retrained on ramp rule from 4.0M (1M planned, stopped 750k):
    0.01032/0.01036/0.01050 at 250k/500k/750k vs 0.01062 start; keeps
    rule gain (-0.00030), adds nothing; 250k was v10 (board 0.01132),
    750k better on gen-20 so v11 ships it (4d).
33. Per-zone credit (own comfort + shared energy, critic per zone, from
    4.0M): 0.01071/0.01072/0.01067 at 250k/500k/750k vs 0.01062; null
    (comfort zero most steps; unsplittable energy dominates).
34. Hourly-held switches at 5%-equivalent flip rate: reward/step -0.10
    to -0.14 first updates (normal -0.006); restarted at 0.4%.
35. CMA-ES headroom fit of zone-target schedule on one 7-day window
    (Denver, Singapore): 0.00466 vs RL 0.00475; 0.00607 vs 0.00592;
    RL already at the lever's ceiling.
36. Hourly-held switches at 0.4% (from 4.0M, 1.75M): 0.01067/0.01065/
    0.01060/0.01047/0.01070/0.01056/0.01058 at 250k-1.75M vs 0.01062;
    null (no control run, older rule: not a clean test).
37. OAT/zone-temp history inputs (1h/3h/24h deltas, 24h OAT mean, from
    4.0M): 0.01082/0.01053/0.01059 at 250k/500k/750k vs 0.01062; null
    (no lever to act on).
38. Per-day schedule head (11-menu prestart/ramp/gain/stop; oracle upper
    bound best-variant-per-day after the fact): -0.00013 (b8) / -0.00007
    (b7) vs best fixed variant per climate; below the 0.0005 bar, not
    built (first 06:00-cut version said -0.00060: wrong).
39. Warm-up levers (evening choice of start +-2h, damper cap, 5-16 C
    floor, from rule-state run): gen-20 honest 0.011781/0.013193 at
    250k/500k vs control 0.011856/0.011425; worse every climate by 500k
    (Phoenix damper capped 0.24 in pre-cooling; Fairbanks floor/cap
    switching off).
40. Rule-state inputs (PI target/integral/mode): b8 0.01021 vs control
    0.01039 (ahead 2 of 3 seeds) but gen-20 0.011727 vs 0.011411;
    better on b8, worse unseen.
41. Fairbanks specialist (cold-only training): 0.02399-0.02422 vs
    0.02388-0.02401 general net; no better in its own climate.
42. Full authority (every hour, scale 1.0, votes override all switches;
    rule only the start): gen-20 0.011175 at 250k vs 0.010868 gated
    residual; the working-hours gate helps.
43. Zone target +-2 K outside band (all-hours 0.011051; gated
    0.011038-0.011315 over 250k-1M) vs 0.010868; band clip stays.
44. PPO recipe change (batch-wide adv norm, separate actor/critic clips,
    linear LR decay): 0.01020/0.01040/0.01036 at 250k/500k/1M vs control
    0.01044/0.01022/0.01042; null.

9.5. Non-failure overfit check (S1 cell 26): no overfit seen on shipped
pool (train b7 -12.4% vs unseen b8 -16.9%); later overturned for
post-m3r checkpoints by the gen-20 judge (section 4d HYPOTHESIS: PPO
overfits the 7 shipped buildings; b8 looks like them).

9.6. Honest bill of materials and data quirks.
Params 48,277 (S18); checkpoint file 204,745 bytes, sha256
61bc2d62... (S4). Shipped-bill b8 of shipped weights 0.01026 (S1);
WHAT the notebook evaluates but the dataset omits: `selection_set.json`,
`selection_extra.json`, `data/gen/*` (pool + judge), training logs
(`logs/skynet/*/history.jsonl`), intermediate checkpoints, `cma_zt`
tuner (`tools/cma_zt.py`), `gen_buildings.py`, gas-audit tools.
Consequence: the section-3b smoke command fails with
`FileNotFoundError: .../skynet/selection_set.json` (S2 cell 13 output)
even as printed; `--gen-*`=0 path still calls `gen_scenarios`, which
reads the missing file unconditionally (S12 lines 49-54).
`curves.json` quirk: `rl_b8_seed1` repeats the 5,001,216 row twice
(S16 lines 466-506). Cell 9 print text `(damper, reheat)` omits the
target head though the value 3 is correct (S2). Gateway parity claim:
policy reproduces training-env scores within 1e-8 on 1500 steps per
held-out climate (S1 cell 26); gateway run in S2 cell 27 took 523 s
for 8641 steps (building_train_1/atlanta_2, mean reward -0.0038).
Reward weights used by the sim (comfort/energy/carbon) are not stated
in the notebook; our verified values are (0.6, 0.2, 0.2) per project
notes, not from this source.

## 10. Transfer limits to our Online RL on occ_gated7

10.1. Directly portable: residual-on-rule with zero-init head (S7);
per-entity shared nets with fixed physical scales and no running stats
(S6); PI-hours-only gating with `free`/`vmask`/`fr` masking (S8/S11/S12);
per-worker advantage normalisation over acted steps (S12); linear LR
decay to 0 at run end + keep-best-not-last (S12, S1-5 items 7/10);
worst-climate screening per checkpoint (S1-5 item 8); gen-20-style
second judge validated against the board before trusting b8 (S1-4d);
pool checksum at run start (S1-4d); honest-bill training (S14).
10.2. Not portable as-is: the zt rule stack and its CMA constants
(our base is `occ_gated7`, different levers and PI structure);
zone-target head (needs a PI-target interface our rule may not expose);
their exact feature list (built around zt proposal fields and AHU
pooling; our obs/policy split differs); Fairbanks-heavy conclusions
(cost map, warm-up share) are zt-off-specific.
10.3. Closed doors (do not retry without a new angle): hourly action
hold; all-hours residual; bigger nets (d128x3+ worse at equal steps);
weight averaging; per-zone credit; cold-weighted pools; full authority;
+-2 K band violation; graph attention; per-day schedule head below
0.0005 oracle bound; learned on/off without long hold + tiny flip rate
(S1-5 items 2,9-14,21,24,32-34,36,42-44).
10.4. Open doors worth our compute: working-hours-gated residual on
`occ_gated7` with d64x2; 720-pool-style generated-building training
once `tools/gen_transfer.py` pools exist; critic with look-ahead
weather (actor never sees it); wider d128-from-m3r-style growth;
60-building judge; warm-up lever search with a same-pool control run.
10.5. Scale warning: their full result cost ~5M steps x2 seeds + zone-target
5M + m3r/m4/B1 follow-ups at ~120 steps/s with 6 workers; a single 5M
run is ~11-12 h of 6-worker CPU. Our `tools/fullrun.py` cap (<=10
concurrent sims) and 1500 s+ timeouts apply to any replication.

## Open questions

1. What are the exact `selection_set.json` / `selection_extra.json`
   scenarios and the `data/gen` generator settings? Neither ships (9.6);
   the gen-20 judge cannot be reproduced bit-identically without them.
2. What are the m4 / B1 / rule-state / warm-up-lever code diffs vs the
   shipped trainer? Only their scores ship, not their sources.
3. What is the per-climate breakdown of the shipped m3r@750k weights on
   b8 honest bill (only means 0.01050 honest / 0.01026 shipped given)?
4. Does their residual pick up the return-air-temp bug (discussion
   753804) through dampers? Stated unmeasured (S1 cell 26).
5. What are the pending v14/v15 (B1@1M/2M) board scores? Pending at
   write time (S1 cell 26).
6. Exact `tools/cma_zt.py` search space and scenario list behind
   `hist_cma_zt*.jsonl`? Only the final lines ship.
