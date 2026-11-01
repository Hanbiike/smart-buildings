# Training-Logic Checks for tools/rl_train.py

Date: 2026-11-01
Status: draft
Scope: Exhaustive build-and-verify checklist for the online residual-PPO trainer over the rule-based controller: per-worker advantage normalization, free-step masking, shared-counter rollout, honest billing, building-count ladder, generated-building judge, BC warm-start, LR schedule, and update math. Each row names the exact in-repo file or command that proves the check. `tools/rl_train.py` is TO-CREATE; rows citing it list the out-of-repo spec lines it must port plus the in-repo command that verifies the port.

How to read: Priority P0 = must pass before the first training run; P1 = must pass before runs longer than 250k steps; P2 = required before shipment. Coverage labels: b8-held-out = building_train_8 x 5 climates; judge-20/60 = generated selection sets; board-30 = hidden leaderboard pairs. Research backing is in `docs/research/13-online-rl-training-logic.md` (claims C1-C40).

## G1. Advantage normalization

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-01 | Per-worker advantage normalization over free steps only | Port spec `/tmp/rl_zip/skynet_train.py:180-186` into `tools/rl_train.py` (TO-CREATE); verify with `grep -n "frs > 0" tools/rl_train.py` and `grep -n "1e-8" tools/rl_train.py` | Normalized group has mean 0 and std 1 within 1e-6 on every worker with more than 1 free sample; `fr == 0` samples keep advantage exactly 0 (C2) | P0 |
| TL-02 | Global-norm ablation switch | `grep -n "adv-norm" tools/rl_train.py` shows a `--adv-norm {per-worker,global,none}` flag (TO-CREATE); spec rationale `/tmp/rl_zip/skynet_train.py:180-182` | Flag exists; default `per-worker`; a 250k-step A/B on judge-20 decides the default (OQ1) | P1 |
| TL-03 | Minibatch composition logging | `grep -n "minibatch" tools/rl_train.py` (TO-CREATE); compare SB3 per-minibatch semantics (C1) | History rows log minibatch size (256) and per-worker free counts; no silent fallback to per-minibatch normalization | P1 |

## G2. Free-step and vote masking

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-04 | Off-hours rule action with untouched latch | Port spec `/tmp/rl_zip/env.py:56-70` into the worker module used by `tools/rl_train.py` (TO-CREATE); verify with `grep -n "latch" tools/rl_train.py` | When free is false: action equals rule `(pc, pd)`, vote mask is all zeros, latch state unchanged (C5) | P0 |
| TL-05 | Held-step log-prob masking | Port spec `/tmp/rl_zip/env.py:100-108` and `/tmp/rl_zip/skynet_train.py:196-199`; verify with `grep -n "fr_t" tools/rl_train.py` | Only sampling steps carry log-prob; policy loss divides by `fr.sum().clamp(min=1)`; off-hours and episode end reset the held action (C6) | P0 |
| TL-06 | Post-step log-prob with latch mask | `grep -n "log_prob_entropy" tools/rl_train.py` (TO-CREATE); spec `/tmp/rl_zip/env.py:104-105` | Log-prob call happens after the environment step with the post-step mask; no pre-step mask reuse | P0 |
| TL-07 | Guard-off attribution metric | `grep -n "free" tools/rl_train.py` shows a logged free-step fraction and free-only reward split (TO-CREATE); precedent C8 | Every eval row reports the free-step fraction and the free-only mean reward alongside the closed-loop score | P1 |
| TL-08 | Shutdown-step gradient masking | Follow in-repo precedent `tools/slot_ppo.py:14-16` (`Shutdown steps carry no gradient ... but count for GAE/value`); `grep -n -i "shutdown" tools/rl_train.py` | Shutdown steps excluded from the policy gradient, included in GAE/value; unit probe passes (C10) | P1 |

## G3. Shared-counter rollout and episode chaining

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-09 | Shared-counter rollout accounting | Port spec `/tmp/rl_zip/skynet_train.py:152-159` and `/tmp/rl_zip/env.py:132-154`; verify with `grep -n "counter" tools/rl_train.py` | Each update consumes exactly `n_steps x workers` (default 512 x 6 = 3072) environment steps; lock-guarded spawn-context counter (C34) | P0 |
| TL-10 | Full 30-day back-to-back chaining | `grep -n "8640\|max_steps" tools/rl_train.py` (TO-CREATE); spec `/tmp/rl_zip/env.py:12,76-80` | `done` ends a scenario and the worker starts the next list entry by itself; default episode length 8640 steps (C40) | P0 |
| TL-11 | Resume semantics documented and enforced | Port spec `/tmp/rl_zip/skynet_train.py:105-116`; verify with `grep -n "NOT restored" tools/rl_train.py` | Resume restores weights/optimizer/RNGs/counters/cursors; in-flight rollout and in-scenario position restart at day 0 with a loud log line; config-mismatch resume refuses (C36) | P0 |
| TL-12 | Unbuildable-scenario skip | Port spec `/tmp/rl_zip/env.py:36-45`; verify with `grep -n "SKIP scenario" tools/rl_train.py` | Unbuildable generated plans are skipped with a loud log; the worker raises only if no scenario builds (C17) | P0 |

## G4. Honest billing and reward scale

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-13 | Honest billing in training and judging | Port spec `/tmp/rl_zip/sbrl.py:31-53` behind a `--bill {honest,shipped}` flag defaulting to `honest`; verify with `grep -n "honest_bill\|_honest_bill" tools/rl_train.py` | Flow billed at `max(tank, return)` plus dissipation plus tank change, floored at 0; idempotent; dynamics untouched (C30) | P0 |
| TL-14 | Reward scale 10 round-trip | Port spec `/tmp/rl_zip/skynet_train.py:79,162,208`; verify with `grep -n "reward_scale\|reward-scale" tools/rl_train.py` | Buffer stores `r * 10.0`; logged reward mean divides back by 10; default `--reward-scale 10.0` (C29) | P0 |
| TL-15 | Shipped-bill audit path kept | `grep -rn "honest_bill" tools/ smart_buildings/ --include=*.py` shows references only inside `tools/rl_train.py` (TO-CREATE); no other in-repo module imports the patch | Audits never import the honest patch; judge rows record which bill produced them (C30, C21) | P1 |
| TL-16 | Billing offset measured on b8 | After the first judge run compare bills on identical checkpoints; reference offset honest-minus-shipped -0.000443 on b8 5-climate means (C31) | Offset reproduced within +-0.0002 before trusting honest-bill checkpoint deltas | P1 |
| TL-17 | Paired-reward option costed | Port spec `/tmp/rl_zip/skynet_train.py:89` and `/tmp/rl_zip/env.py:50-52,73-75` as `--reward {raw,paired}`; verify with `grep -n "paired" tools/rl_train.py` | Default `raw`; `paired` subtracts a zt shadow stepped in lockstep at documented 2x simulation cost; no measured gain claimed until OQ3 closes (C32) | P2 |

## G5. Building pool and building-count ladder

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-18 | Pool manifest plus hash per run | `tools/rl_train.py` (TO-CREATE) writes the pool (building/weather pairs, draw seed, manifest hash) to the run log; spec draw `/tmp/rl_zip/skynet_train.py:49-67` | Re-running with the same seed reproduces the identical pool hash; different machines compared by hash (C17) | P0 |
| TL-19 | Selection-set exclusion guards | Port exclusion of `selection_set.json` and the `selection_extra.json` overlap abort; verify with `grep -n "selection" tools/rl_train.py` | Training refuses to start if any pooled scenario is in a held-out selection file (C17) | P0 |
| TL-20 | Building-count ladder 0/200/720 | Run `--gen` ladder per spec flags `--gen-cold/--gen-other/--gen-seed` (`/tmp/rl_zip/skynet_train.py:91-92`); judge on judge-20 honest | Ladder reproduces ordering shipped-only > +200gen > +720gen within judge resolution 0.0001; reference deltas -0.000108 and -0.000373 (C15, C19) | P1 |
| TL-21 | Cold-climate oversampling preserved | `grep -n "gen_cold\|gen-cold" tools/rl_train.py` (TO-CREATE); spec `/tmp/rl_zip/skynet_train.py:57-62` | Separate cold (fairbanks/minneapolis) and other-climate counts; pool fails loudly if fewer scenarios exist than requested | P1 |
| TL-22 | In-repo transfer suite as second judge | `./.venv/bin/python -B tools/gen_transfer.py` then `ls smart_buildings/transfer_scenarios/` | 16 scenarios (4 buildings x minneapolis_1/minneapolis_5/atlanta_4/singapore_2) build; suite used as the cross-check judge (C39) | P1 |

## G6. Generated-building judge and checkpoint selection

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-23 | Judge protocol (mean actions, capped parallelism) | Port spec `/tmp/rl_zip/skynet_eval.py:1-7,39-54,73-99` into the eval path (TO-CREATE, same module or `tools/rl_eval.py`); verify with `grep -n "no sampling\|mean action\|20" tools/rl_train.py` | Mean actions, no sampling; 8640 steps on b8-held-out, 2016 steps per selection scenario; at most 20 simulator processes; per-scenario \|mean reward\| plus mean; per-step npz archive (C21) | P0 |
| TL-24 | Snapshot cadence 250k with background eval | Port spec `/tmp/rl_zip/skynet_train.py:216-224`; verify with `grep -n "eval_every\|eval-every" tools/rl_train.py` | Snapshot plus background held-out eval every 250,000 steps; `latest.pt` every 10 updates (C36) | P0 |
| TL-25 | Selection rule uses the judge, never b8 alone | Judge-first rule written in the run README; reference reversal m4-vs-m3r (b8 delta -0.00013 flips to board +0.00018) `/tmp/rl_zip/curves.json:1058-1109` and C18 | Ship checkpoint is the judge-20/60 argmin; b8-only argmin is reported but never ships alone | P0 |
| TL-26 | Resolution floor 0.0001 enforced | Selection script treats judge deltas under 0.0001 as ties (C19: judge blind to 0.00007) | Ties broken by earlier step count, then by shipped-bill judge mean; no shipment on a sub-floor delta | P1 |
| TL-27 | Multi-seed fairbanks check | At least 2 seeds past 750k before shipment; reference seed-1 fairbanks spikes 0.96/0.56/0.55 at 3.0-3.5M versus seed-0 clean (C26) | No seed shows fairbanks_8 above 0.05 at any snapshot; seeds agree within 0.001 on judge mean or the run is rejected | P1 |

## G7. Warm start and exploration init

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-28 | Warm-start init with grow semantics | Port `--init` block semantics `/tmp/rl_zip/skynet_train.py:117-131` (copy overlapping blocks, zero new rows/columns, special-case embedding concat); verify with `grep -n "\-\-init" tools/rl_train.py` | Init from a smaller checkpoint reproduces its outputs exactly before training; grown rows start at zero (C36, OQ9) | P1 |
| TL-29 | P-only stable start option | Follow `tools/slot_ppo.py:54-57` (zero memory-feature columns at init); `grep -n -i "zero.*mem\|mem.*zero\|P-only" tools/rl_train.py` | A documented flag reproduces the stable P-only start; default start is stable (no first-update explosion in clip fraction) (C10) | P1 |
| TL-30 | Exploration std init and reporting | `grep -n "log_std\|log-std\|init-std" tools/rl_train.py` (TO-CREATE); spec `--log-std` `/tmp/rl_zip/skynet_train.py:93,138-141`; slot precedent init-std 0.02 `tools/slot_ppo.py:79-80` | Initial log-std is explicit, set only on fresh start/init (never on resume), and logged per head (zone/ahu/plant) every update (C36) | P1 |

## G8. LR schedule and overtraining guards

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-31 | Linear LR to 0 with horizon equal to --steps | Port spec `/tmp/rl_zip/skynet_train.py:168`; verify with `grep -n "1.0 - steps" tools/rl_train.py` | LR reaches exactly 0.0 at `--steps`; resume with changed `--steps` logs the reshaped schedule loudly (C24) | P0 |
| TL-32 | 750k turn-down guard | Early-stop/plateau rule keyed to judge-20: stop or decay if no improvement over the last 500k steps after 750k; reference C25 turn-down without decay and C26 +0.000546 4M-to-5M regression | No shipped checkpoint comes from past a documented judge regression; the 750k snapshot is always scored (C25, C26) | P1 |
| TL-33 | KL/clip/entropy monitoring with zero-weight entropy | `grep -n "clip_frac\|entropy\|kl" tools/rl_train.py` (TO-CREATE); update spec `/tmp/rl_zip/skynet_train.py:201-214` | Every update logs clip fraction, mean entropy, and per-head log-std; an entropy-collapse alert fires if std collapses while the judge regresses (C35, C27) | P1 |

## G9. Update math and run logging

| ID | Check | How (exact command/file in this repo) | Expected/threshold | Priority |
|----|-------|----------------------------------------|--------------------|----------|
| TL-34 | Exact PPO loss form | Port spec `/tmp/rl_zip/skynet_train.py:187-203` (epochs 4, minibatch 256, clip 0.2, `loss = pg + 0.5 * vf`, no entropy term, grad-norm clip 0.5); verify with `grep -n "clip_grad_norm\|vf_coef\|vf-coef" tools/rl_train.py` | Loss matches the spec term-for-term; any entropy coefficient defaults to 0.0 and is logged (C35, C40) | P0 |
| TL-35 | GAE per worker with bootstrapped cut | Port spec `/tmp/rl_zip/skynet_train.py:169-179` (gamma 0.995, lambda 0.95, `explained_var` log); verify with `grep -n "explained_var\|gae\|lam" tools/rl_train.py` | GAE runs per worker with bootstrap at the rollout cut; explained variance logged every update (C35) | P0 |
| TL-36 | history.jsonl and eval jsonl schemas | `grep -n "history.jsonl\|eval_honest" tools/rl_train.py` (TO-CREATE); spec `/tmp/rl_zip/skynet_train.py:206-215` and `/tmp/rl_zip/skynet_eval.py:100-110` | One JSON object per update (update, steps, sps, losses, entropy, clip_frac, log_std, overrides/1k, episodes) and one per eval (tag, ckpt, steps, per-scenario scores, mean, bill, set) (C36, C21) | P1 |
| TL-37 |Concurrency and runtime caps | `grep -n "set_num_threads\|workers" tools/rl_train.py`; spec torch 2 main / 1 worker threads `/tmp/rl_zip/skynet_train.py:98-99` and `/tmp/rl_zip/env.py:87` | Thread caps match spec; total concurrent simulators stay within the repo cap (see AGENTS.md evaluate section) | P1 |

## Definition of done

G1 (advantage normalization): TL-01 passes with a numeric probe (per-worker mean 0 / std 1 on free steps, exact 0 elsewhere); TL-02 flag exists with default `per-worker`.
G2 (masking): TL-04/05/06 pass with step-level probes (off-hours action equality, held-step log-prob zeros, post-step mask usage); TL-07 free-only metric appears in the first eval row.
G3 (rollout): TL-09/10/11 pass (exact 3072-step updates, self-chaining scenarios, loud day-0 resume); TL-12 skip path covered by a forced-unbuildable probe.
G4 (billing): TL-13/14 pass (honest default with idempotence probe, x10 round-trip); TL-16 offset reproduced within +-0.0002 before any honest-bill shipment decision.
G5 (pool): TL-18/19 pass (pool hash logged and reproducible, selection overlap refuses); TL-20 ladder ordering reproduced within judge resolution.
G6 (judge): TL-23/24/25 pass (judge protocol, 250k cadence, judge-first selection); TL-26 ties handled; TL-27 two-seed fairbanks gate green.
G7 (warm start): TL-28 init reproduces source outputs exactly; TL-29/30 start stable with logged per-head exploration.
G8 (LR): TL-31 LR hits 0.0 at `--steps`; TL-32 guard blocks shipment past a documented regression.
G9 (update/logging): TL-34/35 match the spec term-for-term; TL-36 schemas validated by parsing the first history and eval rows as JSON.
