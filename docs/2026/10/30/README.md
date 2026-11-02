# 2026-10-30

## Summary

Downloaded the external residual-RL dataset (`alexchilton/smart-buildings-residual-rl`, Apache-2.0) linked to `yes-rl-can-run-a-building-residual-ppo.ipynb` and inspected all 7 files. Rebuilt `.venv` on Python 3.14 (simulator needs PEP 695 syntax) with `torch 2.15.0+cu132` on RTX 5090. Reconstructed the `/tmp/pkg/skynet` package and verified the shipped checkpoint `m1c_ztoff_s10_seed1@5001216` reproduces `curves.json` on a full 8640-step episode. Decided the next line is Online RL as a finetune of these weights (not from scratch), with a better architecture: stateful residual inputs plus Bernoulli switches first, graph attention second. No submissions or pushes today.

## Status

| Whose | Score | State |
|---|---|---|
| leader (Pardheev Krishna) | 0.00514 | LB (from 10-28 journal) |
| us (occ9) | 0.01236 | COMPLETE (from 10-28 journal) |
| us (occ8) | 0.01260 | COMPLETE (from 10-28 journal) |
| us (occ7) | 0.01323 | COMPLETE (from 10-28 journal) |
| us (occ4) | 0.01323 | COMPLETE (best tie) |
| external residual-RL 5.0M (dataset, not ours) | 0.01199 | public LB reported in notebook (held-out b8 0.01040) |

No new public scores today.

## Done

### 08:00-08:30 — dataset

- Downloaded `alexchilton/smart-buildings-residual-rl` via `uvx kaggle datasets download` to `/tmp/opencode/sb-rl/` (zip 203K), unpacked 7 files: `skynet.pt`, `curves.json`, `TAG.txt`, `base.py`, `features.py`, `nets.py`, `policy.py`, `zt_controller.py` (plus empty `__init__.py`).
- Read checkpoint args: `n_steps 512, epochs 4, minibatch 256, lr 0.0003, gamma 0.995, lam 0.95, clip 0.2, vf_coef 0.5, reward_scale 10.0, scale 1.0, d 64, depth 2, workers 6, base zt_off, holdout building_train_8, hold 1, latch 1`; model `d64x2`, inputs `21/34/22`, outputs `2/4/3` (zone_out 2, no target head), `48211` params, `params {"controller": "zt_off"}`.

### 08:30-09:30 — environment

- Recreated `.venv` on Python 3.14.4 (3.11 failed on `building.py:369` PEP 695 syntax), installed `numpy, pandas, matplotlib, gin-config, absl-py, scipy, holidays, protobuf, pytz, pyyaml, seaborn, numba, gymnasium, scikit-learn, plotly, opencv-python-headless, tzdata` plus `torch 2.15.0+cu132` (CUDA True, `NVIDIA GeForce RTX 5090 D v2`).
- Rebuilt `/tmp/pkg/skynet/` from the 6 dataset source files; loaded `smart_buildings/training_scenarios/building_train_8/atlanta_1` (30 zones, 3 AHUs).

### 09:30-10:00 — verification

- Ran 300-step smoke of `SkynetPolicy` on `building_train_8/atlanta_1`: 300 steps, `mean_reward -0.00586`.
- Ran full 8640-step episode on `building_train_8/atlanta_1`: 8640 steps, `loss 0.005785826410752398` vs `curves.json 0.0057927768980073745` (delta `-0.00000695`), reproduce confirmed.

### 10:00-11:00 — probe + train-30

- Added `SkynetResidualProbe` (`KEY probe_skynet`) to `smart_buildings/baselines/probe_modes.py`, wired import plus `POLICY_MAP` entry in `smart_buildings/evaluate.py`; verify printed `True SkynetResidualProbe`.
- Head-to-head full-8640 on `building_train_8/atlanta_1` via `tools/fullrun.py /tmp/opencode/r-skynet-occ7 8640 2 1500 probe_skynet,occ_gated7 building_train_8/atlanta_1`: `probe_skynet 0.005786 (187s)`, `occ_gated7 0.007700 (184s)`.
- Ran full `train-30` via `tools/fullrun.py /tmp/opencode/r30 8640 8 1500 probe_skynet,occ_gated7 <30 pairs>`: 60/60 jobs complete (see Appendix table).
- Regenerated transfer suite via `tools/gen_transfer.py`: `TRANSFER_SUITE_OK 16 scenarios`; launched `transfer-12` via `tools/fullrun_t.py /tmp/opencode/x12 8640 6 5400 probe_skynet,occ_gated7 <12 pairs> smart_buildings/transfer_scenarios` (drops `xfer_4` monster). Result: skynet +61.5% (1/12) — frozen weights CLOSED (see Decisions).

### 11:00-12:00 — surgery for TODO-1

- Wrote `tools/skynet_surgery.py`: expands shipped `FlatNets` with 2 adapter inputs (PI-target error, PI integral) plus a zeroed zone-target head row (`48661` params vs `48211`). First attempt (widened GEMM copy) failed exact equality at `8.9e-08` (different summation order); fixed with `AdapterFirst` (unchanged 21-wide GEMM plus exact-0.0 adapter add). Multithreaded GEMM adds 1-ulp call-to-call noise, so the test pins single-threading.
- `tools/skynet_surgery.py` prints `PASS: old heads bit-identical, target head exactly 0.0, critic bit-identical` (verified twice).

### 12:00-13:00 — expanded probe + identity smoke

- Added `ExpandedResidualProbe` (`KEY probe_xres`) to `smart_buildings/baselines/probe_modes.py`, wired in `smart_buildings/evaluate.py` import plus `POLICY_MAP`. occ_gated7 base, surgery-expanded net, 23-wide zone features (21 base + true loop err via `floor._zone_err` + integral saturation via `integral * KI`), switches frozen, target head dead by construction, `RESIDUAL_SCALE = 0.0`.
- First smoke harness was broken (two `SimulationTracker` on one `env` interleaved steps; the step-120 `2.0` diff and the `_last_mean` diff were harness artifacts, not probe state). Corrected harness uses one env per policy.
- Corrected 300-step smoke on `building_train_8/atlanta_1` (separate envs): totals exactly equal (`-2.987396905416972` both), all 300 actions bit-equal.

### 13:00-14:00 — rollout collector for Online PPO

- Wrote `tools/skynet_rollout.py`: sequential rollout driver (base `occ_gated7` proposal + sampled residual at scale 1.0, frozen switches), per-entity Gaussian logprobs (`lpz/lpa/lpp`), critic values, shipped-sim rewards, padding masks. Two bugs fixed inline (`_D` import scope, per-head logprob shapes, scalar value cast).
- Self-test `tools/skynet_rollout.py` PASS: `building_train_8/atlanta_1` 256 steps (V=30 K=3, reward_sum=-1.2344, 48.9 steps/sec), `building_train_7/denver_5` 256 steps (V=36 K=4, reward_sum=-0.5198, 44.7 steps/sec); batch pad nV=36 nK=4; all finite.
- Known limits (not bugs): new feature cols rollout-side are zeros (true err/integral wiring is probe-side only so far — must be unified before training); shipped critic is miscalibrated under the occ base (v_mean -24.1/-9.1 vs trajectory scale — needs the V-pretrain phase like `tools/slot_ppo.py`); training reward is the shipped bill (board metric; bug-gaming watched via transfer + worst-climate screens).

### 14:00-15:00 — unified expanded features

- Added `expanded_zone_cols(floor, ix, obs)` to `tools/skynet_surgery.py` (true loop err via `floor._zone_err` + integral saturation via `integral * KI`, read-only); `probe_xres._expanded_zone_cols` now delegates to it; `tools/skynet_rollout.py` uses it instead of zero cols. Fixed a self-inflicted line-merge (`"""` + import on line 46) from an earlier no-op edit.
- Verified: surgery PASS unchanged; rollout SELF-TEST PASS unchanged (identical sums expected — zero-init adapter contributes exactly 0.0); new cols live on real trajectories (300 steps b8/atlanta_1: |err|>0.01 on 99.1% zone-steps, |sat|>0.01 on 20.0%, all finite); `probe_xres` identity re-verified (totals exactly equal, all 300 actions bit-equal).

### 15:00-16:00 — PPO update with V-pretrain

- Wrote `tools/skynet_ppo.py`: GAE (`GAMMA 0.995, LAM 0.95`, `REWARD_SCALE 10.0`), critic-only V-pretrain (`V_LR 1e-3`, 100 steps), clipped PPO (`LR 3e-5, CLIP 0.2, V_COEF 0.5`, grad-clip 1.0) with per-minibatch KL-stop (`KL_STOP 0.01`), mean-field per-dim ratio (slot_ppo convention), mask-exact padded entities, continuous heads only, `save_ckpt`/`load_ckpt` roundtrip via surgery rebuild. Two test bugs fixed inline (KL probe lr `1e-2 -> 1e-1` after measuring no-fire at the lower lr; `load_ckpt` implemented properly and covered by the roundtrip).
- Self-test PASS (4/4): mask check (padded row exactly 0.0); v-pretrain `3627.600586 -> 4.685040` (confirms shipped critic miscalibration, pretrain closes 3 orders); kl-stop fires (`ran 2, stopped=True` at lr `1e-1`); ckpt roundtrip bit-identical.

### 16:00-17:00 — first train cycle (cycle0) + eval fix

- Wrote `tools/skynet_train.py`: rollout collation with cross-building padding, V-pretrain, per-scenario GAE on x10 rewards, PPO epochs with KL-stop, ckpt save with diagnostics. Added `SKYNET_WEIGHTS`/`SKYNET_SCALE` env overrides to `probe_xres` (probe-only knobs; class default scale stays 0.0) and `expand_net_from_ckpt` to `tools/skynet_surgery.py`.
- Cycle-0 smoke run (2 scenarios x 512 steps, 1 epoch, 20 v-steps): rollouts b8/atlanta_1 reward_sum -3.9581 (47.9/s), b7/denver_5 -2.9777 (44.6/s); batch flat=1024 nV=36 nK=4; v-pretrain 463.5159 -> 2.7084; epoch 0 ran_mb=1 stopped=True mean_kl=-0.176254 (KL-stop fired on the first minibatch — epoch barely trained); saved `/tmp/opencode/sb-rl/cycle0.pt` (48661 params, loads via `load_ckpt`).
- Eval harness bug: `fullrun.py` ThreadPool workers crash on `set_num_interop_threads` after parallel work starts (both jobs exit 1, no scores). Fixed as best-effort try/except in `probe_xres`; identity re-verified after the fix (totals exactly equal, 300/300 actions equal). Relaunched with thread caps: `SKYNET_WEIGHTS=cycle0.pt SKYNET_SCALE=1.0 tools/fullrun.py /tmp/opencode/r-cycle0 8640 2 1500 probe_xres building_train_8/atlanta_1,building_train_8/fairbanks_8`.

### 17:00-18:00 — cycle0 scored, cycle-1 launched, logprob bug fixed, cycle-1 scored, cycle-2 launched

- Cycle0 eval landed (`/tmp/opencode/r-cycle0`, 2 jobs): atlanta_1 0.006204, fairbanks_8 0.026802 (185s each).

- Cycle0 eval landed (`/tmp/opencode/r-cycle0`, 2 jobs): atlanta_1 0.006204, fairbanks_8 0.026802 (185s each).
- Added `--lr/--kl-stop/--minibatch/--init` knobs (`ppo_epoch` takes `mb`; ckpt-format-aware init). Launched cycle-1 in background (init cycle0.pt, 4 scenarios incl. transfer xfer_1/minneapolis_1 + xfer_3/singapore_2, 1024 steps, 4 epochs, v-steps 100, lr 1e-5, mb 256, seed 1, out cycle1.pt) to test whether gentler updates train past the 1-minibatch KL-stop.
- Root-caused the instant KL-stop: rollout stored all-3-heads zone logprobs while the update ratio used 2 heads — systematic ratio shift, not a too-big lr. Fixed by recomputing old logprobs with the same `masked_lp` from init weights. Verification smoke (2x512, 2 epochs, lr 1e-5): both epochs ran all 4 minibatches, no stop. Side observation: post-epoch mean_kl -4.14 (policy moved hard toward sampled actions; likely logstd collapse — watch entropy/stds; tune next if eval regresses). Cycle-1 relaunched fixed in background (shell sh_18ceaed2e001pZjZ8TtRHmSS2E).
- Cycle-1 finished (shell sh_18ceaed2e001pZjZ8TtRHmSS2E): rollouts -6.4055/-17.9597/-5.5430/-13.6965; batch flat=4096 nV=30 nK=5; v-pretrain 21.0873 -> 0.0641; epochs 0-3 all ran_mb=16 stopped=False, mean_kl -3.710165 -> -3.710083 (epoch 0 did the whole move, epochs 1-3 inert — clipped saturation). Eval next.
- Cycle-1 holdout eval landed (`/tmp/opencode/r-cycle1`, train driver): atlanta_1 0.006221 (+0.27% vs cycle0 0.006204), fairbanks_8 0.026806 (+0.01% vs 0.026802) — flat, no gain no regression. Transfer pairs errored in the wrong driver (`fullrun.py` sees only training_scenarios: "No scenario config directories matched"); relaunched via `fullrun_t.py /tmp/opencode/r-cycle1x ... smart_buildings/transfer_scenarios` (shell sh_18cf08ae3001LqxErDVlCXDYGC). Cycle-1 transfer landed: xfer_1/minneapolis_1 0.014766, xfer_3/singapore_2 0.010334.
- Cycle-2 launched in background (shell sh_18cf768ff001qXVlmIEDdhDe5S): init cycle1.pt, same 4 scenarios, FULL-8640 rollouts, 4 epochs, v-steps 100, lr 1e-5, seed 3, out cycle2.pt.
- Cycle-2 finished: full-episode rollouts -60.9338/-224.7454/-133.0626/-112.4295 (means -0.00705/-0.02601/-0.01540/-0.01301, exploration-stochastic so worse than deterministic evals); batch flat=34560 nV=30 nK=5; v-pretrain 6.0399 -> 0.0313; epoch 0 ran_mb=135 stopped=False mean_kl=-3.641270; epoch 1 ran_mb=82 stopped=True mean_kl=-3.634614 (KL-stop fired mid-epoch, epochs 2-3 skipped). Eval next.
- Cycle-2 eval launched (2 drivers, conc 2+2=4 total sims): holdout `/tmp/opencode/r-cycle2` (b8/atlanta_1, b8/fairbanks_8, shells sh_18d0c6ae8001CM16obR7lDZmvH) + transfer `/tmp/opencode/r-cycle2x` (xfer_1/minneapolis_1, xfer_3/singapore_2, shell sh_18d0c6ae8002dCXOG6pX5wEPIT).
- Cycle-3 launched in background (shell sh_18d13a11c001CNFT55tSYWoUNP): init cycle1.pt (best), cold-heavy 8-scenario pool (b8/atlanta_1, b8/fairbanks_8, b7/denver_5, b7/fairbanks_7, xfer_1/minneapolis_1, xfer_1/minneapolis_5, xfer_3/minneapolis_5, xfer_3/singapore_2), full-8640 rollouts, 4 epochs, v-steps 100, lr 3e-6, mb 256, seed 4, out cycle3.pt.
- Cycle-3 holdout landed (`/tmp/opencode/r-cycle3`): atlanta_1 0.006477 (+4.11% vs cycle1, -15.88% vs occ7), fairbanks_8 0.027122 (+1.18% vs cycle1, -11.0% vs occ7) — still drifting from cycle-1 despite the cold-heavy pool. Transfer (3 pairs) pending for the final call.
- Cycle-3 transfer landed (`/tmp/opencode/r-cycle3x`): xfer_1/minneapolis_1 0.016738 (+13.35% vs cycle1, +21.81% vs occ7), xfer_1/minneapolis_5 0.007995 (-6.25% vs occ7), xfer_3/singapore_2 0.010227 (-1.04% vs cycle1, -9.89% vs occ7). Cycle-3 mean 0.01371 vs occ7 0.01436 (-4.5%). NOT promoted (cold regression persists and worsens).
- Decomposition on xfer_1/minneapolis_1 (reward_components.csv): occ7 regret -12.47 / gas 949.62 / carbon 389.05 vs cycle3 regret -65.42 (5.2x) / gas 873.24 (-8%) / carbon 360.27 — the residual under-heats cold buildings to save gas, exploding the quadratic comfort tail. Forbidden trade, structural fix next: cold-gate the residual (base action while any zone >1K cold).
- Implemented cold-gate in `probe_xres` (`COLD_GATE_K = 1.0`, `SKYNET_SCALE` env already wired): `_any_cold` read-only check, gate returns base action. Verified on xfer_1/minneapolis_1 January 300 steps at scale 1.0: gate fired 178/300, gated==base on 178/178 fired steps. Scale-0 identity re-verified after the edit (totals exactly equal, 300/300 actions equal).
- Gate eval launched (gated cycle-1 weights, scale 1.0): holdout `/tmp/opencode/r-gate1` (2 pairs, shell sh_18d491549001C3jqwveNJgBcGa) + transfer `/tmp/opencode/r-gate1x` (3 pairs incl. minneapolis_5, shell sh_18d49154a0010301nChZOUq1dE).
- Gate holdout landed: atlanta_1 0.005935 (-4.60% vs ungated cycle1 0.006221, -22.93% vs occ7), fairbanks_8 0.025595 (-4.52% vs 0.026806, -16.00% vs occ7) — the gate improves even train-family pairs, confirming the under-heating pathology was active there too. Transfer pending.
- Gate transfer landed (`/tmp/opencode/r-gate1x`): xfer_1/minneapolis_1 0.011657 (-21.05% vs ungated cycle1 0.014766, -15.17% vs occ7 — cold regression FIXED and beaten), xfer_1/minneapolis_5 0.008146 (-4.48% vs occ7), xfer_3/singapore_2 0.010334 (gate never fires on cooling, same as ungated, -8.94% vs occ7). Gated cycle-1: 5/5 pairs negative vs occ7, mean 0.01233 vs 0.01436 (-14.10%). PROMOTION CANDIDATE — full train-30 + transfer-12 next (vs occ_gated7 + occ_econdbhhot9) before any submit talk.
- Launched full train-30 for the candidate (shell sh_18d505a95001Jg4OSsNALchu6v): 90 jobs, conc 8. Transfer-12 follows after (conc rule: one suite at a time).
- Train-30 landed (`/tmp/opencode/r-gate30`, 90/90): xres mean 0.00847 median 0.00647 vs occ7 0.00963/0.00717 (-12.08%, wins 27/30, 3 losses <=1.36%) vs occ9 0.00878/0.00638 (-3.59%, wins only 14/30, worst loss +9.76% b4/phoenix_8). Beats occ7 decisively, edges occ9 on mean but not uniform — transfer-12 decides.
- Transfer-12 landed (`/tmp/opencode/r-gate12`, 36/36): xres mean 0.01639 median 0.01155 vs occ7 0.01802/0.01308 (-9.05%, wins 12/12 UNIFORM) vs occ9 0.01665/0.01181 (-1.53%, wins 10/12, losses x1/minneapolis_5 +3.17% and x3/minneapolis_1 +13.95%). Verdict: NO SUBMIT — consistent direction over occ9 on both coverages (-3.6%/-1.5%) but not uniform (16/30 + 2/12 losses); margin too thin for the submission budget.
- Train/eval mismatch found: `rollout_env` applies the residual everywhere (no shutdown/floor/cold gating) while `probe_xres` deployment gates all three — PPO learns from states/actions deployment never takes. Next lever: mirror the probe gating in the rollout (gated steps recorded with the actually-applied zero correction and its logprob).
- Mirrored gating in `tools/skynet_rollout.py` (module-level `any_cold` in `probe_modes.py`, shared by probe method and rollout; gated steps record exact-zero corrections). Verified: xfer_1/minneapolis_1 January 300 steps — gated 291/300, gated rows exactly zero, all finite; self-test PASS unchanged; probe identity re-verified after the refactor (totals exactly equal, 300/300).
- Cycle-2 holdout landed: atlanta_1 0.006427 (+3.31% vs cycle1 0.006221, -16.53% vs occ7), fairbanks_8 0.027222 (+1.55% vs cycle1 0.026806, -10.66% vs occ7) — REGRESSION vs cycle-1 on both pairs. Transfer pending for the final call.
- Cycle-2 transfer landed: xfer_1/minneapolis_1 0.016860 (+14.18% vs cycle1, +22.70% vs occ7), xfer_3/singapore_2 0.010299 (-0.34% vs cycle1, -9.25% vs occ7). Cycle-2 mean 0.01520 vs cycle1 0.01453 (+4.6%) vs occ7 0.01582 (-3.88%). CLOSED as regression: two cycles degrade the cold pair (+7.5% then +22.7% vs occ7 on minneapolis_1) while the shipped-bill rollout batch rewards it — truncated-batch overfit pattern, same family as the author's 5.0M->5.5M +7% drift.
- Logstd-collapse hypothesis REJECTED by measurement: cycle0 vs cycle1 std mean 0.2803 both (min 0.2134/0.2133, max 0.3679 both) — the -3.71 KL move came from mus, not noise shrink. Eval launched: `SKYNET_WEIGHTS=cycle1.pt SKYNET_SCALE=1.0 tools/fullrun.py /tmp/opencode/r-cycle1 8640 4 1500 probe_xres` on b8/atlanta_1, b8/fairbanks_8, xfer_1/minneapolis_1, xfer_3/singapore_2 (shell sh_18ced8e7f001UQsIVND7skTR5T).

## Measured

- skynet 5.0M — single-pair (building_train_8/atlanta_1, 8640) — loss 0.00578583 — delta -0.00000695 vs curves.json 0.00579278 for the same pair.
- skynet 5.0M — single-pair (building_train_8/atlanta_1, 300) — mean_reward -0.00586504 — smoke only, not comparable to full episodes.
- external residual-RL — held-out-b8 (5 climates, from curves.json) — mean 0.01040 — delta -16.9% vs shipped-bill rule, -10.6% on honest bill, -9.0% under PR #2 (author's attribution, not re-measured).
- external residual-RL — public (30 hidden pairs, from notebook) — 0.01199 — delta -0.00124 vs our occ_gated4 0.01323.
- probe_skynet — train-30 (30 pairs, full-8640 each) — mean 0.00867747, median 0.00619050 — delta -0.00095240 (-9.89%) vs occ_gated7 on the same coverage.
- occ_gated7 — train-30 (30 pairs, full-8640 each) — mean 0.00962987, median 0.00716700 — reference for the delta above.
- probe_skynet vs occ_gated7 — train-30 — skynet wins 27/30 pairs; 3 losses marginal: building_train_1/fairbanks_1 +1.98%, building_train_1/singapore_8 +0.91%, building_train_4/singapore_3 +0.72%.
- probe_skynet — transfer-12 (xfer_1..3 x minneapolis_1/minneapolis_5/atlanta_4/singapore_2, full-8640 each) — mean 0.02911, median 0.01622 — delta +0.01108 (+61.5%) vs occ_gated7 on the same coverage.
- occ_gated7 — transfer-12 (same 12) — mean 0.01802, median 0.01374 — reference for the delta above.
- probe_skynet vs occ_gated7 — transfer-12 — skynet wins 1/12 (only building_xfer_1/minneapolis_1 -7.5%); worst loss building_xfer_3/minneapolis_1 +175.0% (0.154304 vs 0.056106); excluding the worst pair skynet still +21.7%.
- skynet_surgery — torch unit test (single-threaded, seeded) — old zone/AHU/plant heads bit-identical, target head exactly 0.0, critic bit-identical — PASS twice.
- probe_xres — single-pair (building_train_8/atlanta_1, 300, separate envs) — totals exactly equal (-2.987396905416972 both), all 300 actions bit-equal vs occ_gated7 — identity harness proven; smoke only, not comparable to full episodes.
- skynet_rollout self-test — 2 scenarios x 256 steps — b8/atlanta_1 reward_sum -1.2344 at 48.9 steps/sec, b7/denver_5 reward_sum -0.5198 at 44.7 steps/sec, pad nV=36 nK=4, all finite — collector mechanics proven; not a score.
- expanded features live — b8/atlanta_1 300 steps — |err|>0.01 on 99.1% zone-steps, |sat|>0.01 on 20.0%, all finite; probe_xres identity holds after delegation (totals exactly equal, 300/300 actions equal).
- skynet_ppo self-test — synthetic batch — mask exactly 0.0 on pads; v-pretrain 3627.600586 -> 4.685040; kl-stop fires (ran 2, stopped=True); ckpt roundtrip bit-identical — update mechanics proven; not a score.
- cycle0 train smoke — 2 scenarios x 512 steps — rollouts -3.9581/-2.9777; v-pretrain 463.5159 -> 2.7084; epoch 0 ran_mb=1 stopped=True — loop runs end to end; not a score (eval pending in /tmp/opencode/r-cycle0).
- cycle0 eval — holdout-b8 full-8640 — atlanta_1 0.006204 (185s), fairbanks_8 0.026802 (185s) — vs occ_gated7 -19.43%/-12.04%, vs shipped skynet +7.22%/+6.02% — one barely-trained epoch already beats the occ base on both; shipped-zt corrections partially carry to the occ base.
- cycle1 eval — 4 pairs full-8640 (holdout 2 + transfer 2): b8/atlanta_1 0.006221, b8/fairbanks_8 0.026806, xfer_1/minneapolis_1 0.014766, xfer_3/singapore_2 0.010334 — mean 0.01453 vs occ_gated7 0.01582 (-8.12%); per-pair -19.21%/-12.03%/+7.46%/-8.94% — cold transfer regresses, NOT promoted.

## Decisions

- Artifact reproduces locally. Proof: full-episode delta `7e-06` on `building_train_8/atlanta_1`. Consequence: proceed to register a `skynet` probe and run the declared `train-30` / `transfer-12` suites before any training.
- Online RL starts from `5001216`, not from scratch. Proof: from-scratch RL closed negative (`pid_water` base 0.03184 held-out, `slot_ppo` 0.040 -> 0.076 diverged). Consequence: finetune residual on the stronger base.
- Base switches from `zt_off` to `occ_gated7` / `occ_econdbhhot9`. Proof: author states residual never leaves a bad base; our `occ9` public 0.01236 already beats `zt` 0.01420. Consequence: port `skynet/policy.py` composition onto the occ stack.
- Architecture order is stateful-inputs plus Bernoulli switches first, graph attention second. Proof: `slot` line root cause is stateful teacher vs memoryless student; switches learn at `~1/100` SNR through the `+-0.5` dead zone; `d128x3/d256x3` wider nets already failed (0.01229/0.01237/0.01340). Consequence: TODO-1 (inputs + switches + zone-target head), TODO-2 (attention) only if TODO-1 plateaus.
- Shipped 5.0M weights do NOT transfer and are not a finetune base. Proof: transfer-12 mean +61.5% vs occ_gated7, wins 1/12, worst pair +175.0% (xfer_3/minneapolis_1 cold blowup, same family as the author's seed-1 Fairbanks spike to 0.96). Consequence: TODO-0 closed negative for the frozen weights; TODO-1 trains Online with transfer coverage in the loop from the start (warm start optional, never trusted blind).
- Cycle-1 NOT promoted (earlier call, kept for provenance). Proof at the time: mean -8.12% over 4 eval pairs but xfer_1/minneapolis_1 regressed +7.46% vs occ_gated7. Diagnosis with numbers: 1024-step rollouts starting at episode start saw rollout-mean 0.0054 on x1/minneapolis_1 while the full episode scored 0.0148 — truncated rollouts are blind to the late cold tail (author trained on full 30-day episodes). Consequence applied: cycle-2 used full-8640 rollouts.
- Cycle-2 NOT promoted. Proof: cold transfer +22.7% vs occ_gated7 (xfer_1/minneapolis_1), holdout also regressed vs cycle-1 (+3.31%/+1.55%); mean gain over occ7 shrank -8.12% -> -3.88%. Consequence: stop same-setup cycles; best checkpoint stays cycle-1 (also unpromotable: +7.5% cold). Next lever is protocol, not more epochs: cold-heavy 8-scenario pool + lower lr + init from best (cycle-1).

## Open

- Reconciled with 2026-10-28 Open: items 1-10 closures stand (dual-max, shape/timing, heating, stop-side, cooling tail banked, prop-v3 banked, start/gsat/stage/dcv closed); collection plan and causal-gate STOP (MPC gate 2 FAILED) unchanged; no new hidden scores to update them. 2026-10-29 has no journal file, skipped honestly.
- TODO-0 (verify): register `skynet` probe in `POLICY_MAP` in `smart_buildings/evaluate.py`; run `tools/fullrun.py /tmp/r-skynet 8640 8 1500 skynet <train-30 pairs>` and `tools/fullrun_t.py` on `transfer-12` via `tools/gen_transfer.py`; record per-scenario tables. DONE 2026-10-30: train-30 skynet -9.89% (27/30), transfer-12 skynet +61.5% (1/12) — frozen weights CLOSED as non-transferable.
- TODO-1 (online baseline): write residual-PPO train loop from `curves.json` args on the honest bill, warm-started at `5001216`, base `occ_gated7`; add PI target plus integral inputs, zone-target head `+-1K` (`TARGET_K=1.0`), Bernoulli switch heads with existing `Latch`; holdout `building_train_8` plus fold `building_train_7`, worst-climate screening, keep best checkpoint. AMENDED: train with transfer buildings in the pool from the start; treat `5001216` as initialization only, re-screen on transfer-12 every checkpoint. PROGRESS 2026-10-30: `tools/skynet_surgery.py` expands shipped weights (adapter inputs + target head, 48661 params) with old heads bit-identical, target head exactly 0.0, critic bit-identical.
- TODO-2 (attention): one graph-attention layer over `ahu_members` zones per AHU; promote only on a measured transfer-12 gain with clean misfire pairs.
- TODO-3 (submit): new rung in `docs/ladder.md` before any submit (R5); assemble only via `tools/assemble_notebook.py` with `'occ_gated:<consts>'`; 300+ step exec check; log kernel id and timestamps (R6/R7).

## Appendix

### A. Provenance and working tree

- Dataset: `https://www.kaggle.com/datasets/alexchilton/smart-buildings-residual-rl`, license Apache-2.0, local copy `/tmp/opencode/sb-rl/smart-buildings-residual-rl.zip`.
- `TAG.txt`: `m1c_ztoff_s10_seed1@5001216`.
- Git: commits `a777cd0` (head), `4cb8700`, `19db659`, `2d686eb`, `71f23e4`; status has `D sb-t-combo-d.ipynb` (working-tree deletion, not committed) and untracked `yes-rl-can-run-a-building-residual-ppo.ipynb`; no pushes or submits today.
- `.venv` is git-ignored per `.gitignore` (`.venv/` entry) and was recreated; it is not a committed artifact.

### B. Verification checklist (all checked)

- `tools/fullrun.py`, `tools/fullrun_t.py`, `tools/assemble_notebook.py`, `tools/gen_transfer.py` exist.
- `smart_buildings/evaluate.py`, `smart_buildings/baselines/occ_gated7.py`, `notebooks/smart-buildings-submission-occ9.ipynb`, `yes-rl-can-run-a-building-residual-ppo.ipynb` exist.
- `smart_buildings/training_scenarios/building_train_8/atlanta_1/config.gin` loads (30 zones, 3 AHUs, 71 continuous + 4 discrete actions).
- `/tmp/opencode/sb-rl/skynet.pt`, `curves.json`, `base.py`, `features.py`, `nets.py`, `policy.py`, `zt_controller.py` exist and load; `/tmp/pkg/skynet/` package imports.
- Commands used: `uvx kaggle datasets download alexchilton/smart-buildings-residual-rl`, `uv venv .venv --python 3.14`, `uv pip install <deps>`, `uv pip install torch`, `./.venv/bin/python -B -c <smoke>`, full-episode run logging to `/tmp/opencode/sb-rl/eval_atl1.log`.
- Coverage labels: single-pair runs are labeled as such and never mixed with `train-30`, `subset-8`, `transfer-12`, or `public` aggregates.

### C. TODO-1 design pin (read-only, no code written while transfer-12 in flight)

- Base for finetune: `OccEconDbhHot9Policy` (`smart_buildings/baselines/occ_econdbhhot9.py`, KEY `occ_econdbhhot9`) = `HeatOptMixin` + `OccEconMixin` + `DeadbandHeatMixin` + `OccGated7Policy`. Residual composition (`skynet/features.py:compose`) calls `floor.propose()` then adds corrections, so swapping the floor only changes the starting action; weights must be refit, not reused frozen.
- PPO skeleton to reuse: `tools/slot_ppo.py` (590 lines: `PPOPolicy`, `_rollout` spawn workers, `compute_gae`, `SlotPPO.update` with per-minibatch KL stop, `collate`, `resolve_scenarios`). New script needed for `FlatNets` residual heads; `slot_ppo.py` stays untouched as reference.
- Heads to add per plan: PI target plus integral inputs (teacher state the student cannot see), zone-target head `+-1K` (`TARGET_K=1.0` constant already in `features.py`, checkpoint has `zone_out 2`), Bernoulli switch heads through existing `Latch`.
