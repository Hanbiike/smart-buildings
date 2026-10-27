# 2026-10-22 — BC weights overnight, teacher freeze, greybox start

## Summary

Overnight (00:10–01:39) the full behavior-cloning lineage was trained and saved:
eight checkpoints in `smart_buildings/results/nn_runs/` (`smoke`, `smoke2`,
`bc_v0`, `bc_v1`, `bc_v2`, `bc_v2b`, `bc_v3`, `bc_v3b`), spanning the base BC
run, two DAgger rounds, the shadow-teacher state-channel upgrade, a d=96
wide variant, and round-3 data mixes.
At 13:36 `smart_buildings/baselines/zone_pi.py` was frozen as the gen-3.1
reference teacher (`DsatGen31Policy`, KEY `dsat_gen31`); every BC checkpoint
was trained against its trajectories and the V2 line replays its internal
state exactly through a shadow instance.
At 15:58 the greybox line started with `smart_buildings/baselines/greybox.py`:
a per-zone lumped RC balance with online recursive-least-squares
identification, predictor-only, never wired to a controller.
No new competition scores appeared; gen-3.1 holds at 0.01350.
Honestly recorded: no episodic closed-loop BC evaluations were saved on this
day — checkpoint selection rested on held-out building BC loss only.

## Status

| policy | coverage | loss | delta |
| --- | --- | --- | --- |
| dsat_gen31 (teacher, frozen 13:36) | full train | 0.01350 | holds, reference |
| BcV0 (bc_v0.pt) | hold building_train_5 BC loss | not saved | no episodic eval |
| BcV1 (bc_v1.pt, DAgger r2) | hold building_train_5 BC loss | not saved | no episodic eval |
| BcV1Snap (SAT snap, no weights) | none | n/a | inference-only |
| BcV2 (bc_v2.pt, shadow state) | hold building_train_5 BC loss | not saved | no episodic eval |
| BcV2Snap (SAT snap, no weights) | none | n/a | inference-only |
| BcV2B (bc_v2b.pt, d=96) | hold building_train_5 BC loss | not saved | no episodic eval |
| BcV2Ens (d64+d96 head mean) | none | n/a | inference-only |
| BcV3Ens (N-way head mean) | none | n/a | inference-only |
| BcV3/BcV3B (round-3 data) | hold building_train_5 BC loss | not saved | no episodic eval |
| BcV4Ens (mean of 4 heads) | none | n/a | inference-only |
| greybox RLS predictor | offline replay only | n/a | tool, not controller |

## Done

- Trained the BC lineage overnight — `tools/bc_train.py` plus
  `smart_buildings/baselines/nn_policy.py`. Why: build a CPU-only
  transformer controller imitating the frozen gen-3.1 teacher across
  DAgger rounds and architecture variants.
- Froze the teacher — `smart_buildings/baselines/zone_pi.py` (13:36,
  42160 bytes). Why: `DsatGen31Policy` is the imitation reference; its
  `_dump_state` channels are the exact contract the V2 shadow replays.
- Added SAT-snapped inference variants — `BcV1SnapPolicy`,
  `BcV2SnapPolicy` in `nn_policy.py`. Why: the teacher dump histogram is
  strongly multimodal (deep-cool/mid/neutral/max-heat), and plain
  regression smears branch choices; snapping restores discrete choice.
- Added the shadow-teacher V2 line — `BcV2Policy` in `nn_policy.py`.
  Why: fold the teacher's exact internal state channels into model
  features to diagnose statefulness vs compounding error.
- Added capacity and ensemble variants — `BcV2BPolicy` (d=96),
  `BcV2EnsemblePolicy`, `BcV3EnsemblePolicy`, `BcV4EnsemblePolicy` in
  `nn_policy.py`. Why: big buildings (85 zones / 9 AHUs) needed headroom;
  head-averaging cancels distinct closed-loop excursions of underfit
  members.
- Started the greybox line — `smart_buildings/baselines/greybox.py`
  (15:58, 17764 bytes). Why: an online-identified thermal predictor as a
  future plan-B input, explicitly a tool and not a controller.
- Shared the feature contract — `smart_buildings/nn_model.py`
  (`FeatureContext`, `ControllerNet`) and the older copy
  `tools/nn_features.py`. Why: identical decode paths keep training
  tokenization and deployment-time inference in sync.

## Measured

- dsat_gen31 — full train — 0.01350 — holds, frozen reference.
- bc_v0 — hold building_train_5 BC loss — value not saved — no episodic eval.
- bc_v1 — hold building_train_5 BC loss — value not saved — no episodic eval.
- bc_v2 — hold building_train_5 BC loss — value not saved — no episodic eval.
- bc_v2b — hold building_train_5 BC loss — value not saved — no episodic eval.
- bc_v3 — hold building_train_5 BC loss — value not saved — no episodic eval.
- bc_v3b — hold building_train_5 BC loss — value not saved — no episodic eval.
- greybox RLS — offline replay, no score — n/a — predictor-only.

## Decisions

- Claim: gen-3.1 is the frozen teacher. Proof: `zone_pi.py` mtime 13:36
  preserved; `DsatGen31Policy.KEY = "dsat_gen31"`; every checkpoint args
  record `teacher=dsat_gen31`. Consequence: all BC work imitates this
  exact code, and V2 shadows its `_dump_state` layout bit-for-bit.
- Claim: exact teacher state channels via a shadow instance.
  Proof: `BcV2Policy.get_action` advances a shadow `DsatGen31Policy` on
  every obs, folds `st[:Z*2]` (PI signal, integral/100) into room feats,
  `st[Z*2:Z*2+A*3]` into AHU feats, remainder into global feats — the
  same layout as the tokenizer concat. Consequence: the statefulness
  diagnosis is exact, not an approximation.
- Claim: SAT branch snapping is inference-only. Proof: `SAT_MODES =
  (-1.0, -0.4, -0.05, 0.95)` is a class attribute consumed in `_compose`;
  no snapped checkpoint was trained. Consequence: snap variants cost no
  training and can be toggled per policy.
- Claim: ensembles average heads, never logits-to-actions post hoc.
  Proof: `BcV2EnsemblePolicy._infer` and `BcV3EnsemblePolicy._infer`
  average every head dict entry over members, then call the shared
  `_compose`. Consequence: run bits follow the mean-logit sign and
  continuous heads cancel member-specific excursions.
- Claim: no closed-loop BC claim may be made from this day. Proof: no
  episodic result file for any `bc_*` policy exists in results; only
  `*.pt` weights plus hold-building BC loss (values not saved).
  Consequence: BC validation moved to the slot line and closed 10-27
  with all variants below the teacher.
- Claim: greybox stays a predictor. Proof: module docstring "The model
  is a *tool*, not a controller", and `CausalGreyBoxModel.trusted`
  unconditionally returns False. Consequence: the line can only feed a
  future control layer, never act.

## Open

- Closed-loop BC validation moved to the slot line; closed 10-27 with
  every variant below the teacher (compounding beats the shadow-teacher
  diagnosis).
- Greybox online ID + LOBO validation — done 10-23.
- Equipment observers — done 10-24.
- MPC loop closure on greybox predictions — open, never done (plan B).

## Appendix

### A. Checkpoint registry (mined from the .pt files themselves)

All eight files live in `smart_buildings/results/nn_runs/` (git-ignored).
Sizes/mtimes from disk; hyperparameters read from each checkpoint's
stored `args` dict (verbatim):

- `smoke.pt` — 355550 bytes, 00:10. d=64, L=2, H=4, teacher
  dsat_gen31, hold building_train_5, epochs 2, batch 64, lr 3e-4,
  no extra data, fz/fa/fg = 8/10/58.
- `smoke2.pt` — 355657 bytes, 00:22. Same, epochs 1, batch 128.
- `bc_v0.pt` — 355550 bytes, 00:23. Base BC: d=64, epochs 20,
  batch 128, lr 3e-4, teacher-only data, fz/fa/fg = 8/10/58.
- `bc_v1.pt` — 355614 bytes, 00:40. DAgger round 2: epochs 25,
  extra_bags = dagger_v1, fz/fa/fg = 8/10/58.
- `bc_v2.pt` — 359198 bytes, 01:06. Shadow-teacher state channels:
  epochs 25, extra_bags = dagger_v2, fz/fa/fg = 10/13/60 (the +2/+3/+2
  are the folded teacher state channels).
- `bc_v2b.pt` — 754761 bytes, 01:17. V2 at d=96: epochs 30,
  extra_bags = dagger_v2, fz/fa/fg = 10/13/60.
- `bc_v3.pt` — 359198 bytes, 01:33. Round-3 DAgger at d=64: epochs 25,
  extra_bags = dagger_v2,dagger_v3, fz/fa/fg = 10/13/60.
- `bc_v3b.pt` — 754761 bytes, 01:39. Round-3 DAgger at d=96: epochs 30,
  extra_bags = dagger_v2,dagger_v3, fz/fa/fg = 10/13/60.

Draft shorthand preserved verbatim: `results/nn_runs/bc_v1..v3b.pt`
(00:40–01:39, 355–755KB). Note the draft omits `bc_v0.pt`, `smoke.pt`,
`smoke2.pt`, which also exist; byte counts above refine the KB range.
On epochs: the CLI default in `tools/bc_train.py` is 30 (`--epochs`,
line 55) and the draft records "epochs 30", but the stored per-run
args show epochs 20 (v0), 25 (v1/v2/v3), 30 (v2b/v3b) — the default was
overridden per run. Batch 128, lr 3e-4, teacher dsat_gen31, hold
building_train_5 are identical in all six BC checkpoints.

### B. Full BC lineage: what changed in each step and why

- V0 base (`BcV0Policy`, CHECKPOINT `bc_v0.pt`). Plain behavior cloning
  on teacher trajectories: `FeatureContext` decode → 72-step ring
  buffer → 12 pooled history slots at stride 6 → one `ControllerNet`
  forward → `_compose` heads through room/AHU/water index wiring.
  Warmup clamps slots to step 0, exactly like the training tokenizer.
  Why: establish the imitation baseline before any distribution fix.
- V1 DAgger round 2 (`BcV1Policy`, `bc_v1.pt`; collectors
  `DaggerPolicy`/`DaggerV2Policy`). The NN executes, the teacher labels
  the visited obs (`teacher_labels` list of cont‖disc vectors feeding
  `dagger_labels.npz`), and training adds the bag with labels
  overridden (`collect_runs`: features from NN obs, `cont`/`disc`
  replaced from `dagger_labels.npz`). Why: train on the NN's own state
  distribution instead of the teacher's, attacking compounding error.
- V1Snap (`BcV1SnapPolicy`, no weights). Inherits V1, sets `SAT_MODES =
  (-1.0, -0.4, -0.05, 0.95)` so each AHU SAT command snaps to the
  nearest teacher branch mode. Why: regression smears the teacher's
  multimodal SAT histogram; snapping restores branch choice. Also
  `MIN_RUN_HOLD` exists (default 0) to latch run bits against fan
  chatter — left off.
- V2 exact teacher state channels (`BcV2Policy`, `bc_v2.pt`;
  collector `DaggerV2Policy` with `NN_CLASS = BcV1Policy`). A shadow
  `DsatGen31Policy` advances on every observation (action ignored);
  `_dump_state` channels fold into features: per-zone PI signal +
  integral/100 (Z×2 → room feats 8→10), per-AHU mixedness/cooling EMAs
  + hold counter (A×3 → AHU feats 10→13), global PI-hours flag +
  occupant EMA (→ global feats 58→60). All channels are deterministic
  functions of own obs history, so train and deploy share the path.
  Why: the correct diagnosis of statefulness — give the student the
  teacher's memory and see whether compounding survives.
- V2Snap (`BcV2SnapPolicy`, no weights). V2 + the same SAT_MODES tuple.
  Why: branch smearing applies to V2 heads too.
- V2B d=96 (`BcV2BPolicy`, `bc_v2b.pt`). Identical V2 code path at
  d_model=96 (754761 vs 359198 bytes). Why: headroom for the biggest
  buildings (85 zones / 9 AHUs in the draft's words).
- V2Ens (`BcV2EnsemblePolicy`, no new weights). Loads `bc_v2.pt` as
  `model` and `bc_v2b.pt` as `model_b` (SECOND_CHECKPOINT), raises
  unless `proj_hist.in_features` match, averages all heads 0.5/0.5.
  Why: d64 and d96 underfit different buildings; averaging cancels
  distinct closed-loop excursions.
- V3Ens (`BcV3EnsemblePolicy`, CHECKPOINTS generalizer). N-member
  mean-of-heads over an explicit CHECKPOINTS tuple; dedupes the member
  already loaded as `self.model`. Note: as saved, `BcV3EnsemblePolicy`
  ships with `CHECKPOINTS = ()` (single member) — it is the mechanism,
  not a populated ensemble. Why: generalize the two-member average.
- V3/V3B round 3 (`BcV3Policy` → `bc_v3.pt` d64; `BcV3BPolicy` →
  `bc_v3b.pt` d96; collector `DaggerV3Policy` with
  `NN_CLASS = BcV2EnsemblePolicy`). Training data = teacher +
  dagger_v2 + dagger_v3 bags. Why: second DAgger iteration, this time
  collecting off the V2 ensemble's state distribution.
- V4Ens (`BcV4EnsemblePolicy`, no new weights). 4-way head mean over
  bc_v3 (as `model`) + bc_v2 + bc_v2b + bc_v3b (CHECKPOINTS). Why: pool
  both capacities and both DAgger generations in one inference.

### C. Architecture numbers (verified in `smart_buildings/nn_model.py`)

- `ControllerNet(fz, fa, fg, d=64, layers=2, heads=4)` — transformer
  d64/L2/H4 as the draft records. `EncoderBlock`: LayerNorm →
  batch-first MultiheadAttention → residual → MLP (Linear d→2d, GELU,
  Linear 2d→d) with a second residual on the re-normalized path.
- History: WINDOW=72, STRIDE=6, N_HIST=12 slots. Tokens per forward:
  12 history + Z rooms + A AHUs + 1 global. Learned `hist_pos`
  (N_HIST × d) positional embeddings on projected history only.
- Feature widths: room [Z,8] (cool delta, heat delta, band centre,
  band width, zone–OAT delta, damper cmd, reheat cmd, absolute temp),
  AHU [A,10] (OAT, SAT, SAT−setpoint, setpoint, pressure err, flow
  ratio, recirc, fan, cooling requests, run), global [Fg] (occupants,
  comfort flags, hod/dow harmonics, HWS). Scales: SCALE_K=10, DIFF_K=8.
- Heads: shared per-room Linear(d→2) damper/reheat; per-AHU
  Linear(d→4) SAT/pressure/recirc/run-logit; global Linear(d→3)
  water temp/pressure/run-logit. Context vector (mean history + mean
  zone) feeds every head, keeping zone decisions
  permutation-equivariant across building sizes.
- Runtime ring buffer keeps the last 80 pooled rows; the 12 slots read
  back at stride 6 with clamp-to-0 warmup (`nn_policy.py` lines 95–102,
  identical math in `hist_pool_features` / `hist_cache`).
- `tools/nn_features.py` is the older sibling copy (00:13): same [Z,8] /
  [A,10] layout but lexicographic (`sorted()`) AHU order instead of the
  numeric-suffix order, and no AHU-action/water wiring. The shipped
  policy and all checkpoints use `smart_buildings/nn_model.py`.

### D. Training CLI (verified in `tools/bc_train.py`)

- Defaults: `--teacher dsat_gen31`, `--hold-building building_train_5`,
  `--epochs 30`, `--batch 128`, `--lr 3e-4`, `--d-model 64`,
  `--layers 2`, `--heads 4`, `--seed 0`, `--out bc_v0`
  (argparse lines 47–64). Optimizer AdamW + OneCycleLR (pct_start
  0.05, total steps from epochs × per-run batch count).
- Split: every `building_train_*` scenario with an
  `nn_data_<teacher>/tokens.npz` bag trains, except hold-building
  `building_train_5`, which validates. Extra teachers contribute own
  labels; `--extra-bags` (DAgger) override labels from
  `dagger_labels.npz` with `n_cont` split into cont/disc.
- Batches: windows start at WINDOW=72 (`starts = WINDOW +
  permutation((T-WINDOW)//batch)*batch`); validation sweeps
  `range(WINDOW, T, batch)`.
- Loss (`head_loss`): full-weight MSE on zone damper/reheat + AHU SAT;
  half-weight MSE on AHU pressure, recirc, water temp, water pressure;
  half-weight BCE on AHU and water run bits. Masked to wired indices
  (pressure/recirc/water heads skip unwired positions).
- Saved dict per checkpoint: `model` state, `args` (the full CLI
  namespace — the source of the §A table), `fz/fa/fg`.

### E. Greybox equation and RLS update (verified in `greybox.py`)

Model (per VAV zone, one 300 s step):

dT = a·(Tout−T) + b·flow·(Tsup−T) + c·flow·reheat·(Twater−T) + d·occ + e

- a (`P_ENV=0`): envelope coupling to outside air; prior 0.004.
- b (`P_SUP=1`): supply-air coupling, flow = damper command × handler
  run state; prior 0.02.
- c (`P_RHT=2`): reheat (water) coupling, flow × reheat opening ×
  (Twater − T); prior 0.02.
- d (`P_OCC=3`): internal gains from building occupancy; prior 0.0.
- e (`P_CONST=4`): constant bias; prior 0.0.
- Only observable quantities feed regressors (`_regressors`,
  `_snapshot`); zone temps decoded with metadata normalizers, damper /
  reheat clipped to [0,1], AHU run bit thresholded at 0.5.
- Modes: `rls` (pure RLS from prior, forgetting 0.999, P0 = 25·I, no
  warmup) and `batch_rls` (default; ridge least squares over the first
  `fit_window=144` steps with ridge scaled by trace(XᵀX), then RLS from
  that solution). RLS step: k = P·x/(λ + xᵀPx), θ += k·(y−xᵀθ),
  P = (P − k·(Px)ᵀ)/λ, λ = 0.999; residual/bias EMAs at 0.05.
- `trusted(room, 0.5)`: n_obs ≥ 24, plausibility box (a ∈ [−0.05,0.2],
  b,c ∈ [−0.5,2], d ∈ [−1,1], |e| ≤ 1), residual < 0.5 K.
- Prediction: `predict` rolls held inputs; `steps_to_reach` counts
  steps to target (returns max_steps on failure).
- Teacher freeze cross-check: `zone_pi.py` docstring lineage (public LB
  0.01853 k10 → transfer 0.06439 → … → `DsatGen31Policy` standing
  freeze candidate); `_dump_state` contract quoted in §B is lines
  785–808, matching the V2 fold layout exactly.

### F. Honest gaps

- No episodic closed-loop BC evals were saved on 10-22 (no rollout
  JSON/CSV for any bc_* policy; verified by absence in
  `smart_buildings/results/`). BC loss values themselves were printed
  to stdout during training, not persisted.
- `BcV3EnsemblePolicy.CHECKPOINTS = ()` as shipped: the N-way
  mechanism exists but no populated V3Ens checkpoint list is stored.
- `tools/nn_features.py` vs `smart_buildings/nn_model.py` AHU ordering
  differs (lexicographic vs numeric-suffix); checkpoints record
  fz/fa/fg only, so which tokenizer produced each bag is from memory,
  not verifiable from the .pt files.

