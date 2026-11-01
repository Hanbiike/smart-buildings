# Skynet replication and transfer checks

Date: 2026-11-01. Status: draft.
Scope: replicate what the Skynet residual-PPO notebook measured, then transfer
the portable parts onto our Online RL over `occ_gated7`. Source for every
Expected value is `docs/research/16-skynet-residual-ppo-notebook.md`
(sections in parentheses). Coverage tags: offline (no sim), train-30
(6 buildings x 5 climates x 8640 steps), subset-N (named scenarios).
Loss convention: `abs(df.agent_reward.sum()) / len(df)` per run CSV.

## Group A. Offline reconstruction (no simulator)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| A1 | Recompute every notebook mean from raw per-climate values | `./.venv/bin/python -B -c "import json; C=json.load(open('/tmp/rl_zip/curves.json')); ..."` (means over `rl_b8_seed0/seed1`, `rl_b7_seed0`, `honest.*`) | rule_b8 0.01251, zt_b8 0.01358, seed1@5.0M 0.01040 (-16.9%), honest rule 0.01207 / rl5m 0.01079 (-10.6%), b7 fold 0.00758 vs 0.00828 (-8.6%); all within 1e-5 of research doc 2/7.2 | P0 |
| A2 | Verify seed-1 Fairbanks blow-up rows exist as reported | same `curves.json`: `rl_b8_seed1[11..13]` fairbanks_8 | 0.96343 / 0.56061 / 0.55288, recovered 0.02699 at 3.75M (9.8) | P0 |
| A3 | Verify zone-target run table transcription | diff research doc table 8.2 against `/tmp/rl_zip/curves.json` `honest.zone_target` (20 rows + ref) | all 20 means within 1e-5; best 0.01062 at 4,002,816; ref rl5m 0.01079, cma_rule 0.01109 | P0 |
| A4 | Verify checkpoint metadata matches MANIFEST | `./.venv/bin/python -B -c "import torch; ck=torch.load('/tmp/rl_zip/skynet.pt',map_location='cpu',weights_only=False); ..."` (shapes/counts only, no training) | cfg d64/depth2/zone_out3, 48,277 params, steps 752640, base `zt_off_cmagen_ramp`, scale 1.0 (research 3.1, 9.6) | P0 |
| A5 | Confirm dataset gaps that block bit-identical rerun | `ls /tmp/rl_zip/` vs imports in `/tmp/rl_zip/skynet_train.py` (`selection_set.json`, `data/gen/manifest.jsonl`) and `skynet_eval.py` (`selection_extra.json`) | all three missing; smoke command in notebook cell 13 fails FileNotFoundError as logged (9.6) | P0 |
| A6 | Confirm duplicate `rl_b8_seed1` tail row | `./.venv/bin/python -B -c` count of steps==5001216 in `rl_b8_seed1` | 2 identical rows (9.6); dedupe before any curve fit | P1 |

Definition of done (A): A1-A5 green with pasted command outputs in the day
note; A6 recorded as a data quirk wherever the curve is reused.

## Group B. Billing-bug replication (honest bill)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| B1 | Reproduce the honest-bill patch semantics offline | read `/tmp/rl_zip/sbrl.py::honest_bill` and our vendored boiler source; write the equivalent patch as a local-only probe (never shipped) | patch bills flow at max(tank T, return T) + dissipation + tank term; dynamics untouched (7.1) | P0 |
| B2 | Measure shipped-vs-honest gap for `occ_gated7` on train-30 | `./.venv/bin/python -B tools/fullrun.py results/skynet_bill 8640 8 1500 occ_gated7 train30` with and without the B1 probe patch | gap direction matches Skynet (honest < shipped for a heating policy); record both means + per-climate deltas; Skynet reference: rule 0.01251 -> 0.01207, RL 0.01040 -> 0.01079 (7.2) | P0 |
| B3 | Decide the training bill for our Online RL | day-note decision citing B2 + research 7.2 | train and compare on honest bill; board stays shipped sim | P0 |

Definition of done (B): B2 numbers logged with coverage train-30; B3
decision written; probe patch kept out of `smart_buildings/baselines/`.

## Group C. Base-rule parity (what their rule scores here)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| C1 | Confirm our `occ_gated7` train-30 baseline before any RL work | `./.venv/bin/python -B tools/fullrun.py results/skynet_base 8640 8 1500 occ_gated7 train30` (fresh dir, no policy edits in flight) | single mean + per-scenario CSVs; entry point for `docs/ladder.md` | P0 |
| C2 | Check Fairbanks share of our loss (their cost map says ~half the score, 2/3 gas carbon, warm-up dominated) | `./.venv/bin/python -B` reduce of C1 CSVs by climate + hour-of-day gas | Fairbanks fraction and 01:00-08:00 weekday share recorded; compare research 8.1 (61% Fairbanks / 73% Atlanta in 01:00-08:00) | P1 |
| C3 | Port their pre-start ramp idea as a rule variant (not RL): lower zone targets K/h before morning, floored | new `smart_buildings/baselines/occ_ramp.py` + `POLICY_MAP` key in `smart_buildings/evaluate.py`; `./.venv/bin/python -B tools/fullrun.py results/skynet_ramp 8640 8 1500 occ_ramp,occ_gated7 train30` | their ramp gain reference: b8 0.01109 -> 0.01077, b7 0.00749-rule-family -> 0.00734 (4.3); our variant must beat `occ_gated7` on train-30 mean AND worst climate to proceed | P1 |
| C4 | Re-test their rejected rule ablations only if cheap (idle-off, reheat-first, Monday prestart, softer PI) | same harness as C3, one variant per run | expected null/worse per research 4.5; any adopted change needs same-sign gain on >=2 buildings (their sign-flip lesson) | P2 |

Definition of done (C): C1 logged; C2 cost map in day note; C3 either
promoted with a `docs/ladder.md` entry or closed with numbers (R4/R5).

## Group D. Residual architecture transfer (onto occ_gated7)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| D1 | Spike: zero-init residual wrapper that is exactly the rule at init | new throwaway probe in `probe_modes.py` + `POLICY_MAP` (per AGENTS.md), wrapping `occ_gated7` with `action = clip(rule + scale * net_out)` and zeroed head | 8640-step parity run vs `occ_gated7` bit-identical (their 1e-8 parity claim, 9.6) | P0 |
| D2 | Feature audit: map their 21/34/22 features onto our obs + `occ_gated7` proposal | table in day note: each Skynet field -> our obs field / proposal / missing | missing list explicit (esp. PI-target interface for a zone-target head, research 10.2); no running stats by design (3.2) | P0 |
| D3 | Gating audit: define our PI-hours-equivalent `free` flag + vmask/fr masking | design note + unit test next to the probe: off-hours => rule alone, latch untouched, masks zero (3.5) | matches `features.py::compose`, `Latch`, `env.py::step` semantics | P0 |
| D4 | First gated-residual PPO run, d64x2, their hyperparams (5.1-5.3) | trainer run with n_steps 512 x 6 workers, 4 epochs, mb 256, lr 3e-4 linear-decay-to-0, gamma 0.995, lam 0.95, clip 0.2, reward_scale 10, scale 1.0, per-worker adv norm over acted steps, holdout building_train_8 | smoke 250k then full 3-5M; keep-best-not-last; checkpoint every ~250k with worst-climate screen (their items 7/8/10) | P1 |
| D5 | Zone-target-head feasibility | check whether `occ_gated7` exposes a settable zone-target path (read `smart_buildings/baselines/occ_gated7.py` + parent `occ_gated4.py`) | go/no-go: their head needs +-1 K in-band target shifts (8.2); without the interface, skip (their item 18: head on wrong base is flat) | P1 |

Definition of done (D): D1 parity proven; D2/D3 tables committed to the day
note; D4 or D5 produces the first learned delta vs C1 (either sign, recorded).

## Group E. Trainer-recipe adoption (steal the method, not the weights)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| E1 | Async shared-counter rollout (fast buildings contribute more) | implement in our trainer per `/tmp/rl_zip/env.py::rollout` + `skynet_train.py` main loop | throughput >= their 110-135 steps/s at 6 workers on comparable CPU (5.2); correctness: update boundaries at multiples of n_steps x workers | P1 |
| E2 | Per-worker advantage normalisation over acted steps + fr-masked losses | same files; unit-test the mask algebra on a synthetic batch | cold-climate rewards do not set the scale for other climates (5.3) | P1 |
| E3 | Linear LR decay to `--steps` + best-checkpoint retention | same files; log LR per update | no post-peak drift past ~750k-1M without decay; their B1 flat-with-decay reference (6.2) | P1 |
| E4 | Pool checksum at run start | log SHA-256 over the sorted training scenario list before update 0 | prevents their 175-of-720 pool mismatch incident (6.2) | P0 |
| E5 | Noise/LR coupling guardrail | if exploration std is cut by k, cut LR by ~k; abort if clip_frac > 0.15 in the first 10 updates | their clip-frac 0.19-0.34 abort case (9.5) | P1 |

Definition of done (E): E4 in every run log; E1-E3, E5 covered by trainer
unit tests plus one 250k smoke run each where applicable.

## Group F. Selection methodology (do not trust building 8 alone)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| F1 | Build a second judge from `tools/gen_transfer.py` pools (their gen-20 analogue) | `tools/gen_transfer.py` with fixed seeds per AGENTS.md; hold out 20+ scenarios no training/tuning sees | judge scenarios listed + checksummed; each builds in the sim (their 60-judge build check, section 6) | P0 |
| F2 | Validate the judge against known-ordered checkpoints before trusting it | score >=3 ordered policies (e.g. `occ_gated7`, `occ_gated6`, C3 variant) on judge + b8 | judge must rank them the same way b8/board history does; resolution floor ~0.0002 (gaps of 0.00007 do NOT resolve, 6.2) | P0 |
| F3 | Pick all future checkpoints on judge-first, b8-second | update the day-note template + `docs/ladder.md` entry rule | no checkpoint promoted on b8 alone (their m3r/m4 reversal, 6.2) | P1 |

Definition of done (F): F1 pool committed (git-ignored, regenerable);
F2 ranking agreement recorded; F3 rule written into the journal template.

## Group G. Failure-table guards (do not reopen closed doors)

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| G1 | Hourly action hold | do NOT implement; if proposed, cite research 9.3 (0.01265-0.01345, worse than every-step) | closed | P2 |
| G2 | All-hours residual / full authority / +-2 K band violation | keep the working-hours gate + in-band clip; cite 9.12/9.42/9.43 | closed | P2 |
| G3 | Nets bigger than d64x2 before d64x2 peaks | cap at d64x2 until its curve turns; cite 9.14 (d128x3+ worse at equal steps) | closed until D4 peaks | P2 |
| G4 | Learned on/off switches | only with hour hold + <=0.4% flip rate + a same-pool control run; cite 9.23/9.34/9.36 (0.56009 Fairbanks blow-up) | closed by default | P2 |
| G5 | Per-day schedule head | require an oracle upper bound >= 0.0005 before building; cite 9.38 (-0.00013/-0.00007, not built) | closed until bound proven | P2 |
| G6 | Weight averaging across checkpoints | prefer `AvgNets`-style action averaging if anything; cite 9.11 (0.01054 vs 0.01040) | closed | P2 |
| G7 | Per-zone credit / graph attention / cold-weighted pools / paired reward without need | cite 9.4/9.21/9.24/9.33 | closed | P2 |
| G8 | Worst-climate screen on every checkpoint | add to eval wrapper: fail promotion if any single climate exceeds 3x its rule value | their 0.96-Fairbanks near-miss (9.8) | P0 |

Definition of done (G): G8 implemented in the eval path; G1-G7 cited (not
re-run) in any proposal touching those levers.

## Priority summary

P0 (blocks Online-RL start): A1-A5, B1-B3, C1, D1-D3, E4, F1-F2, G8.
P1 (first learning runs): A6, C2-C3, D4-D5, E1-E3, E5, F3.
P2 (only if cheap): C4, G1-G7.
