# Architecture Checks for Online RL over occ_gated7

Date: 2026-11-01
Status: draft
Scope: Exhaustive verification checklist for policy-architecture decisions
for online RL over the rule-based controller occ_gated7. Companion to
`docs/research/12-online-rl-architectures.md` (finding IDs A1-E7, open
questions OQ1-OQ8 there map to groups below). Every How entry names an
exact command or file in this repo. Thresholds are initial proposals;
tighten after the first measurement round.

Conventions: loss = `abs(df.agent_reward.sum()) / len(df)` over a full
8640-step episode (30 days x 5 min); coverage labels train-30 / subset-8 /
transfer-12 per repo rules; keep total concurrent sims <= 10 with 1500s+
timeouts (`tools/fullrun.py`, `tools/fullrun_t.py`); never edit policy
code while a validation run using it is in flight.

## Check table

| ID | Check | How (exact command/file) | Expected / threshold | Priority |
|----|-------|--------------------------|----------------------|----------|
| ARCH-01 | M1 shared-MLP baseline loss on train-30 | `tools/fullrun.py results 8640 <conc> 1500 <m1_policy> <train_pairs>` with M1 policy registered in `POLICY_MAP` in `smart_buildings/evaluate.py` (nets from `/tmp/rl_zip/nets.py`, features from `/tmp/rl_zip/features.py`) | Loss recorded per pair; baseline for all deltas below | P0 |
| ARCH-02 | Shared MLP (M1, d=64 depth=2) vs GRU-latent policy, same features | Add GRU-latent variant to `probe_modes.py` + `POLICY_MAP` (AGENTS.md probe convention); run `tools/fullrun.py results 8640 <conc> 1500 m1_gru <train_pairs>` on identical pairs as ARCH-01 | Decide by delta: GRU wins only if train-30 mean loss improves by >= 2% relative, else keep MLP (OQ1) | P0 |
| ARCH-03 | Shared MLP vs small transformer policy (d=64, window tokens) | Add transformer variant reusing `ControllerNet`/`FeatureContext` from `smart_buildings/nn_model.py` (WINDOW=72, STRIDE=6, N_HIST=12) via `probe_modes.py` + `POLICY_MAP`; run `tools/fullrun.py results 8640 <conc> 1500 m1_tr <train_pairs>` | Transformer wins only if >= 2% relative over ARCH-01 at equal CPU (see ARCH-11); else keep MLP | P0 |
| ARCH-04 | Zone-target offset head on/off (TARGET_K=1.0 K/unit) | Toggle third zone output in `BuildingIndex.target_offsets` (`/tmp/rl_zip/features.py`, `TARGET_K`); 2-output checkpoint compat via `FlatNets.from_checkpoint` (`/tmp/rl_zip/nets.py`); run `tools/fullrun.py` on subset-8 both ways | Keep head only if subset-8 mean loss improves by >= 1% relative or comfort violations drop; else ship 2-output (OQ8) | P0 |
| ARCH-05 | Pooled AHU features on/off (mean err, cold/warm frac, worst-zone err) | Ablate `pooled` block in `BuildingIndex.features` (`/tmp/rl_zip/features.py` lines ~127-133); run `tools/fullrun.py` on subset-8 both ways | Keep pooling if >= 1% relative gain; log per-pair deltas (aliasing hypothesis E1) | P0 |
| ARCH-06 | Zone-net AHU context on/off (supply-temp + run state) | Ablate `ahu_ctx` in `BuildingIndex.features` (`/tmp/rl_zip/features.py` line ~124); run `tools/fullrun.py` on subset-8 both ways | Keep context if >= 1% relative gain; this is the explicit anti-aliasing input (E1) | P0 |
| ARCH-07 | Fixed physical scales vs running-norm stats | Compare current fixed scales (`ZONE_FIELDS`/`AHU_FIELDS`/`PLANT_FIELDS` in `/tmp/rl_zip/features.py`) vs a running-stats variant; run `tools/fullrun_t.py results 8640 <conc> 1500 <policy> <pairs> smart_buildings/transfer_scenarios/` | Ship fixed scales unless running stats win transfer-12 by >= 3% relative AND a shipping story for per-building stats exists (E5; default: fixed) | P1 |
| ARCH-08 | Clip-rate audit on transfer ([-10, 10] rails) | Instrument `clip` in `BuildingIndex.features` (`/tmp/rl_zip/features.py` line ~138) to log per-field saturation counts; run `tools/fullrun_t.py` on transfer-12 | Report per-field clip %; any field > 5% clipped triggers rescale of that field (OQ3) | P0 |
| ARCH-09 | Scale coverage: which transfer field saturates first | Same run as ARCH-08; rank fields by clip rate; cross-check against fit range note (building_train_8, atlanta_1 night, phoenix_4 day) in `/tmp/rl_zip/features.py` docstring | Rescale top saturating field(s) and re-run subset-8 to confirm no train regression | P1 |
| ARCH-10 | CPU budget per step: MLP vs GRU vs transformer | Time single-step forward (B=1, torch CPU) for each variant: ` ./.venv/bin/python -B tools/time_policy.py <policy>` (create under `tools/`; forward-only, no sim); measure mean/p99 ms over 8640 calls | All variants must fit budget (initial: p99 < 5 ms/step CPU); any variant over budget is disqualified regardless of loss (OQ7) | P0 |
| ARCH-11 | Forward-pass parity with BcV0Policy plumbing | Reuse ring-buffer + `ControllerNet` path from `smart_buildings/baselines/nn_policy.py` for the transformer candidate; verify warmup clamps to step 0 for first 72 steps | Bit-identical warmup behavior vs `BcV0Policy` for the same checkpoint; no NaN over 8640 steps | P1 |
| ARCH-12 | Time-feature ablation (hod/dow harmonics + comfort flags) | Ablate `GLOBAL_FIELDS` time entries (`/tmp/rl_zip/features.py` line ~50; `glob_cols` in `smart_buildings/nn_model.py`) ; run `tools/fullrun.py` on subset-8, including night/shutdown/prestart coverage (>= 300 steps local exec check per AGENTS.md) | Keep time features if night/prestart comfort violations increase without them; aliasing hypothesis E2 | P1 |
| ARCH-13 | Cross-morphology: train on mixed zone counts, eval per AHU size | Train M1 on mixed-morphology pairs; evaluate with `tools/fullrun.py` grouped by zone count (<= 10 zones/AHU valid; check zone count before suite per AGENTS.md) | Report loss per morphology bucket; flag any bucket > 10% worse than its specialist (D2 long-tail check) | P1 |
| ARCH-14 | Specialist-vs-generalist gap on train set | Train per-building specialist heads (same M1 trunk) vs one shared policy; compare train-30 mean loss via `tools/fullrun.py` | Record gap %; if gap > 15%, revisit shared capacity (d/depth) or per-type heads (OQ5, E6) | P1 |
| ARCH-15 | Mean pooling vs attention over entities | Implement per-AHU attention pooling variant over `ahu_members` (`/tmp/rl_zip/features.py` line ~103) in `probe_modes.py`; run `tools/fullrun.py` on subset-8 | Attention wins only if >= 2% relative over mean pooling at equal CPU (OQ2, D4) | P2 |
| ARCH-16 | Per-node-type encoder/decoder (Amorpheus-lite) probe | Probe per-type linear encoders + shared trunk + per-type decoders (embed_dim=64 per S4/S5) in `probe_modes.py` + `POLICY_MAP`; run on subset-8 | Promote to `occ_gatedN.py` only if >= 3% relative over M1 on subset-8 with no CPU-budget breach (D1) | P2 |
| ARCH-17 | Width/depth sweep: d in {32, 64, 128} x depth in {1, 2, 3} | Sweep `FlatNets(d, depth)` (`/tmp/rl_zip/nets.py` line ~28) on subset-8 via `tools/fullrun.py`; one config per run, no concurrent edits | Ship smallest config within 1% of best loss (robustness-tax control, E6); record full sweep table | P1 |
| ARCH-18 | Zero-init residual check (untrained == base controller) | Fresh `FlatNets()` (zero heads) vs base rule controller over 8640 steps on one pair: `./.venv/bin/python -B tools/fullrun.py results 8640 1 1500 <both_policies> <one_pair>` | Bit-identical actions (deterministic sim per AGENTS.md); any diff is a wiring bug | P0 |
| ARCH-19 | Mask/latch correctness (padded entities + held votes) | Unit probe of `pad`, `pad_vmask`, `log_prob_entropy` (`/tmp/rl_zip/nets.py`), `Latch.apply` + `compose` dead zone +-0.5 (`/tmp/rl_zip/features.py`); plus 300+ step local exec covering night/shutdown/prestart | Padded entities contribute exactly 0 to loss; held votes excluded; no fan chatter below teacher rate | P0 |
| ARCH-20 | Weather-ordering: wide-variance-first vs reverse | Train M1 twice with reversed climate order (wide-variance climates first vs last); eval both on `tools/fullrun_t.py` transfer-12 | Wide-first wins expected (E4); record delta; adopt winning order for final training (OQ6) | P1 |
| ARCH-21 | Recurrence warmup: hidden-state init over first 72 steps | For GRU variant (ARCH-02): zero-init vs learned-init hidden state; local 300+ step exec + subset-8 `tools/fullrun.py` | No comfort-violation spike in first 72 steps; pick init with lower subset-8 loss | P1 |
| ARCH-22 | In-context / history-length probe (12 vs 24 slots) | Vary pooled history slots (N_HIST=12 default from WINDOW=72/STRIDE=6 in `smart_buildings/nn_model.py`) for transformer candidate; subset-8 runs | Longer history wins only if >= 1% relative at equal CPU; else keep 12 (C1/C4) | P2 |
| ARCH-23 | Comfort-vs-energy slope guard on every architecture run | For each ARCH run above, decompose reward into comfort/energy/carbon (weights 0.6/0.2/0.2 per AGENTS.md); comfort slope (~0.18/K) dominates | Reject any architecture that trades comfort for energy (comfort violations must not increase vs ARCH-01) | P0 |
| ARCH-24 | Submission plumbing for winning architecture | Assemble via `tools/assemble_notebook.py` (must include `occ_gated:<consts>` + `STEPS_PER_HOUR`); local exec check 300+ steps; `kaggle kernels push` -> COMPLETE -> submit | Kernel COMPLETE with no ERROR; submission logged per R6 | P0 |

## Definition of done

- Core-architecture group (ARCH-01..06): M1 baseline loss recorded on
train-30; MLP vs GRU vs transformer decision made with >= 2% relative
thresholds; zone-target head, pooled AHU features, and AHU-context
decisions made with per-pair deltas; comfort guard (ARCH-23) green on all.
- Scale/robustness group (ARCH-07..09, ARCH-20): clip-rate audit complete
on transfer-12 with per-field percentages; saturating fields rescaled with
no train regression; fixed-vs-running decision recorded; weather ordering
adopted.
- Budget/correctness group (ARCH-10, ARCH-11, ARCH-18, ARCH-19): p99
per-step CPU measured for every candidate, over-budget variants
disqualified; zero-init bit-identity holds; mask/latch unit probes pass;
warmup behavior verified.
- Capacity/morphology group (ARCH-13..17, ARCH-21, ARCH-22): loss recorded
per morphology bucket with no silent long tail; specialist gap measured;
width/depth sweep table recorded and smallest-near-best shipped; attention
and Amorpheus-lite probes either promoted with numbers or closed as
negative results (R4).
- Time/context group (ARCH-12): time-feature ablation complete with
night/prestart coverage; decision recorded.
- Submission group (ARCH-24): winning architecture assembled, locally
executed 300+ steps, pushed, and logged; no policy edits were made during
any in-flight validation run.
