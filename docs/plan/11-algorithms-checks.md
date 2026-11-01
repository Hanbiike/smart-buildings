# Track A Plan: Algorithmic Checks for Online Residual RL on occ_gated7

Date: 2026-11-01
Status: draft
Scope: Exhaustive checkable list for model-free algorithm decisions (residual
PPO vs residual SAC vs TD3, recurrent variants, seeds, action bounds) before
and during online RL on top of the frozen rule-based controller occ_gated7.
Every check names the exact file or command in this repo. Research basis:
`docs/research/11-online-rl-algorithms.md` (claims R1-R19 below refer to its
numbered claims). Forbidden during these checks: editing policy code while a
validation run using it is in flight; committing/pushing from this checkout.

Conventions: `PY = ./.venv/bin/python -B`; full episode = 8640 steps;
loss from CSV = `abs(df.agent_reward.sum()) / len(df)`; keep total concurrent
sims <= 10 with 1500s+ timeouts. Result coverage labels: train-30 / subset-8 /
transfer-12 / public (never mixed unlabeled).

## Group A — Residual algorithm selection (PPO vs SAC vs TD3)

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|---|---|---|---|---|
| A-01 | Residual PPO baseline exists and runs 8640-step episodes without crash | Register probe in `smart_buildings/baselines/probe_modes.py` + `POLICY_MAP` in `smart_buildings/evaluate.py`; run `PY tools/fullrun.py results/a01 8640 4 1500 res_ppo_s0 subset8` (define `subset8` pair list in the driver call) | Completes 8/8 episodes; loss within 3x of occ_gated7 subset-8 loss (sanity, not win) | P0 |
| A-02 | Residual SAC (auto-tuned entropy) matches A-01 protocol exactly | Same probe + driver pattern as A-01 with SAC residual head; `PY tools/fullrun.py results/a02 8640 4 1500 res_sac_s0 subset8` | Completes 8/8; comfort component <= occ_gated7 comfort (no cliff falls) | P0 |
| A-03 | TD3 residual on same protocol | Same pattern; `PY tools/fullrun.py results/a03 8640 4 1500 res_td3_s0 subset8` | Completes 8/8; report comfort/energy/carbon split + violation counts | P0 |
| A-04 | Head-to-head A-01/A-02/A-03 on identical seeds and scenarios decides the default algorithm | Compare `results/a01`, `results/a02`, `results/a03` CSVs: loss + comfort/energy/carbon components, per-scenario table | Winner = lowest mean loss AND comfort <= occ_gated7 comfort; if none beats occ_gated7 train mean, STOP per R19/Sinergym warning (R1) | P0 |
| A-05 | Entropy ablation: PPO with fixed entropy bonus vs ent-coef 0.0 (our `tools/slot_ppo.py` default) | Fork `tools/slot_ppo.py --ent-coef` values `0.0 / 1e-4 / 1e-3`; short 2016-step rollout check first: `PY tools/slot_ppo.py --ent-coef 1e-3 --rollout-steps 2016 --iters 2 --workers 4 --scenarios subset8` | Entropy > 0 runs show action std > 0.05 on all channels (no boundary pinning per R8); ent 0.0 reproduces saturation or it is exonerated | P1 |
| A-06 | SAC auto-tune vs fixed temperature | Same harness as A-02 with `--auto-entropy on/off` (name TBD at implementation; record exact flag in day log) | Auto-tune keeps policy stochastic (action std > 0.05) through final 25% of training | P1 |
| A-07 | TD3 exploration-noise and target-smoothing sweep (narrow, 2 values each) | Same harness as A-03; vary noise `0.1/0.2`, smoothing `0.005/0.01` | Pick setting with best subset-8 loss subject to comfort <= baseline; R2 warns TD3 buys wins with violations — violations veto | P1 |
| A-08 | Winner re-run on full train-30 before any transfer claim | `PY tools/fullrun.py results/a08 8640 8 1500 <winner>_s0 train30` | Report train-30 mean loss + delta vs occ_gated7 with coverage label; entry in `docs/ladder.md` before any submission talk | P0 |

Definition of done (Group A): A-04 decision recorded with per-scenario
numbers; winner (or documented STOP) on train-30 via A-08; all runs use
frozen occ_gated7 center (`smart_buildings/baselines/occ_gated7.py`
untouched, verified by `git status --short`); negative result recorded with
numbers if no residual beats the center.

## Group B — Action bounds, residual scale, and smoothing

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|---|---|---|---|---|
| B-01 | Hard projection layer clamps every channel (dampers >= 10%, discharge <= 308.15 K, recirc <= 0.70) OUTSIDE the policy | Implement in probe policy file under `smart_buildings/baselines/`; unit-check with `PY -c "from smart_buildings.baselines.<probe> import project; project(...)"` on out-of-range vectors | 100% of random out-of-range actions project inside bounds; policy network never emits the clamp (verify by logging pre-projection values) | P0 |
| B-02 | Residual output scale sweep (small delta on top of occ_gated7 center) | Same harness as A-01; residual scale `0.05 / 0.1 / 0.2` in normalized action units | Smallest scale that still improves train loss; largest scale must not increase comfort loss vs center (R8 collapse geometry) | P0 |
| B-03 | Slew-rate limit |du| inside the projection (per-step max change) | Add to projection from B-01; sweep `0.05 / 0.1 / inf` per step | Finite limit cuts oscillation metric (sign changes of du per 100 steps) >= 30% vs inf at equal-or-better loss | P1 |
| B-04 | Action-rate (jerk) penalty in training objective (R14: -60% switching in literature) | Add `\|du\|` penalty term; sweep weight `0.0 / 0.01 / 0.1` | Nonzero weight reduces switching >= 30% with loss delta within +1% of unpenalized (else drop) | P1 |
| B-05 | Boundary-pinning monitor on every training run | Log per-channel action mean/std/min/max over each 8640-step eval episode to run dir CSV | PASS = no channel with std < 0.01 AND mean within 2% of a bound (R8 detector); FAIL triggers A-05/A-06 revisit | P0 |
| B-06 | Absolute-setpoint vs residual-delta action interface comparison | One run with absolute action head vs A-01 residual head, same algorithm/seed/scenarios | Residual wins or ties on loss with strictly fewer B-05 violations; else record negative with numbers | P1 |

Definition of done (Group B): B-01 projection merged into the probe policy
with the unit check passing; B-02 scale fixed and recorded; B-05 monitor
present in every subsequent run; B-06 interface decision recorded.

## Group C — Normalization, reward scale, and compute budget

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|---|---|---|---|---|
| C-01 | Measure OUR simulator throughput (steps/s, 1 worker and 8 workers) | `PY tools/slot_ppo.py --rollout-steps 864 --iters 1 --workers 1 --scenarios subset8` then `--workers 8`; time with `time` | Record steps/s; compute wall-clock for 1M steps; R16 analog is 770 (PPO) / 70 (SAC) steps/s on EnergyPlus — ours must be measured, not assumed | P0 |
| C-02 | SAC-vs-PPO wall-clock ratio on our sim for equal env steps | Same 1-iter timing runs for both harnesses (A-01 vs A-02 code paths) | Ratio recorded; if SAC > 5x slower per step, budget training iters accordingly (R16 mechanism: per-step updates vs batched) | P1 |
| C-03 | Observation normalization audit (VecNormalize-style or fixed scales) | Log per-obs-dim mean/std over one 8640-step occ_gated7 episode per train building; check file `smart_buildings/evaluate.py` obs pipeline | Every dim within ~2 orders of magnitude post-norm; outliers get fixed scales before any RL run | P0 |
| C-04 | Reward term-scale audit: comfort vs energy vs carbon magnitudes | From occ_gated7 subset-8 CSVs: decompose mean loss into comfort/energy/carbon shares | Post-scale terms within ~1 order of magnitude (R15 recipe); comfort slope ~0.18/K must not dominate batches 100:1 | P0 |
| C-05 | Advantage whitening on/off ablation | Same harness as A-01; toggle per-batch `(A-mu)/sigma` | Whitening on: no gradient-norm explosions (log grad norm; PASS = p99 < 10x median); off must justify itself | P1 |
| C-06 | Occupancy-gated comfort shaping kept exact (no penalty reshape without A/B) | Diff training reward vs competition reward term-by-term in probe file | Any shaping beyond occ_gated7 gating gets its own A/B run and never silently replaces the selection metric (full-episode competition loss decides) | P0 |

Definition of done (Group C): C-01/C-02 wall-clock numbers recorded;
C-03/C-04 audits pass with logged scales; C-06 guarantees the selection
metric is always the true competition loss on full episodes.

## Group D — Recurrent / memory variants (exploratory, evidence-free)

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|---|---|---|---|---|
| D-01 | Feedforward residual baseline frozen as the memory ablation control | Use A-04 winner config, memory features zeroed (same convention as `tools/slot_ppo.py` ZMEM/AMEM/WMEM zeroing, lines 54-57) | Baseline subset-8 loss recorded; all D checks compare against THIS, not against occ_gated7 directly | P0 |
| D-02 | Short observation-history stack (last K obs, K = 4 / 12) | Extend probe obs pipeline; same algorithm/seed as D-01 | Either K beats D-01 by >= 1% subset-8 loss with comfort <= baseline, or record negative and stop stacking | P2 |
| D-03 | Recurrent head (LSTM/GRU over the episode) vs D-01 | Same algorithm/seed; truncated BPTT length recorded in day log | Must beat D-01 by >= 2% AND pass B-05 AND train within 3x D-01 wall-clock, else close as negative (R8 open question 2: no literature prior) | P2 |
| D-04 | Memory-feature-only variant (slot memory columns enabled, no recurrence) | Enable ZMEM/AMEM/WMEM columns in `tools/slot_ppo.py` convention (`--init` without zeroing; exact flag recorded) | Compare vs D-01; if unstable (B-05 FAIL or loss explosion), re-zero and close | P2 |
| D-05 | Night/shutdown/prestart behavior spot-check for any D-variant that passes | 300+ step local exec check covering night/shutdown/prestart (per repo rule) on 2 scenarios | No NameError/no-op pathologies; actions differ sensibly across modes (manual trace review) | P0 |

Definition of done (Group D): D-01 control recorded; each attempted variant
ends in a promoted setting or a closed negative with numbers; any survivor
passes D-05 and B-05.

## Group E — Seeds, generalization, and transfer protocol

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|---|---|---|---|---|
| E-01 | Minimum 3 training seeds for the A-04 winner (single-seed results are a known caveat, R8) | `... res_<winner>_s0/s1/s2 subset8` via `PY tools/fullrun.py` (one call per seed, conc <= 10 total) | Report mean +- range across seeds; PASS = all 3 seeds beat occ_gated7 subset-8 mean (else the "win" is seed luck) | P0 |
| E-02 | Same-climate vs cross-climate split test (R10: specialists win home climate, p < 0.05) | Train on warm-half scenarios, eval subset-8 mixed; then reversed; scenario lists recorded | Gap quantified; if cross-climate collapses (> 20% worse than home), add climate-mixing to training before transfer | P1 |
| E-03 | Transfer-12 eval of the E-01 survivor (regenerate scenarios with fixed seeds first) | `PY tools/gen_transfer.py` (fixed seeds) then `PY tools/fullrun_t.py results/e03 8640 4 1500 <winner>_s0 <scenarios_dir>` | transfer-12 mean loss + per-pair table; any pair with comfort blowup vetoes promotion regardless of mean | P0 |
| E-04 | Catastrophic-forgetting check if training is staged multi-climate (R10) | Eval on stage-1 scenarios after stage-2 training | PASS = stage-1 loss within +2% of pre-stage-2 checkpoint; FAIL = joint training required, no sequential curriculum | P1 |
| E-05 | Public-score snapshot discipline: local wins are NOT scored wins | Record in day log + `docs/ladder.md`: coverage, loss, delta, commit SHA | No "beats occ_gated7" claim without the coverage label and SHA; submissions follow the kaggle push flow only after E-01+E-03 pass | P0 |

Definition of done (Group E): E-01 3-seed table + E-03 transfer-12 table both
recorded with coverage labels; E-05 logging complete; promotion decision
(ship / iterate / kill) written in the day log.

## Group F — Stop rules and negative-result capture

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|---|---|---|---|---|
| F-01 | Fixed step budget per algorithm with a kill rule | Set budget (e.g. 1M env steps) BEFORE the run in the day log; enforce via `--iters` math in `tools/slot_ppo.py` or harness equiv | If no residual beats the occ_gated7 train mean within budget, kill model-free and record (R1/R19 Sinergym warning) | P0 |
| F-02 | Comfort veto: any residual that grows comfort loss vs center is rejected even if total loss improves | Per-run comfort/energy/carbon decomposition from eval CSVs | Veto is absolute (R2 TD3 warning: cheap-violation wins do not transfer to our 0.18/K slope) | P0 |
| F-03 | Every closed lever gets a one-line negative with numbers | Day log Measured line + research-doc Open-question update | Format: `<variant> — <coverage> — <loss> — <delta vs center>`; no silent drops | P0 |

Definition of done (Group F): budget set before runs; every started check
ends in PASS numbers or a recorded negative; no check left "pending" at day
close without a named blocker.
