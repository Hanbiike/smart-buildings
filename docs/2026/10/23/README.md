# 2026-10-23 — Greybox validation

## Summary

No new competition scores appeared on this day; the work was validation
of the greybox thermal line started 10-22. In the evening the model
family landed: `thermal_greybox.py` (water/mass ablations over the
causal base), `pooled_greybox.py` (transferable population prior),
`causal_greybox.py` (bounded action-conditioned ARX predictor),
`equipment_balance.py` (pure equipment energy balances), plus the two
offline harnesses `tools/greybox_check.py` and
`tools/pooled_greybox_check.py` with RLS online ID and
adequacy/selection protocols.
Coverage is synthetic + workday protocols under
`smart_buildings/results/thermal_*.json` (git-ignored): adequacy v2
with 7 cases (nominal rejected 0.0% with no alarm; fault rejected 88%,
first alarm at obs 98), and the workday LOBO v1 with 6 folds, 407540
training rows, fit window 72, horizons 6/12.
Honest record: the adequacy gates pass on synthetic cases, but the
workday LOBO forecast errors do NOT beat the persistence baseline
(negative skill at h6/h12 in all three variants) — predictors are valid
as gated diagnostics, and the control loop was never closed, so the
line stands as plan B.

## Status

| policy | coverage | loss | delta |
| --- | --- | --- | --- |
| adequacy gate (synthetic, nominal) | 7-case v2 (+5-case v1) | 0.0% rejected, no alarm | gate passes |
| adequacy gate (synthetic, fault) | 7-case v2 | 88% rejected, alarm @ obs 98 | gate fires |
| LOBO mass / pooled_frozen / pooled_adapted | 30 workday cases × 3 | h6 MAE 0.182–0.243 K vs persist 0.147 K | below persistence |
| thermal family / grid / mismatch / modes | synthetic fixtures | algebra-level residuals | valid, offline |
| state observer v1/v2 | synthetic 288-step | observability spectra saved | valid, offline |
| greybox MPC loop closure | none | n/a | never done |

## Done

- Built the causal predictor — `smart_buildings/baselines/causal_greybox.py`
  (14432 bytes). Why: an action-conditioned bounded ARX that reconstructs
  discharge temperature from setpoints (sensors report setpoints, not
  discharge) with sensor-delay and plant-calibration options.
- Built the ablation layer — `smart_buildings/baselines/thermal_greybox.py`
  (6210 bytes). Why: switch physical-water and thermal-mass changes
  independently over the causal base for controlled experiments.
- Built the transfer prior — `smart_buildings/baselines/pooled_greybox.py`
  (3352 bytes). Why: a population coefficient vector shared across
  buildings, adapted from the observed test prefix only, for
  leave-one-building-out validation.
- Built the equipment physics — `smart_buildings/baselines/equipment_balance.py`
  (6210 bytes). Why: pure energy balances (air flows, handler, water
  flows, reheat coil, zone derivatives) with no controller, fitting,
  prices, or hidden config.
- Built the validation harnesses — `tools/greybox_check.py` (model
  kinds, horizon/skill evaluation vs persistence/drift) and
  `tools/pooled_greybox_check.py` (LOBO folds, population fit,
  forecast gate). Why: offline replay on banked NN dumps with frozen
  protocols, never touching the simulator.
- Banked the artifact registry — `smart_buildings/results/thermal_*.json`
  plus `greybox_lobo_workday_v1.json` (git-ignored). Why: every protocol
  string and metric below is read from these files, not from memory.

## Measured

- adequacy nominal — 7-case v2 gate — 0.0% rejected, no alarm — passes.
- adequacy fault — 7-case v2 gate — 88% rejected, first alarm @ obs 98 — fires.
- LOBO pooled_adapted h6 — 30 workday cases — 0.182 K vs persist 0.147 K — below baseline.
- LOBO pooled_adapted h12 — 30 workday cases — 0.307 K vs persist 0.265 K — below baseline.
- LOBO mass h6/h12 — 30 workday cases — 0.188/0.292 K vs 0.147/0.265 K — below baseline.
- LOBO pooled_frozen h6/h12 — 30 workday cases — 0.243/0.398 K — worst arm.
- grid_reference fixtures — 2 rows each — source err 6.3e-13 K — algebra exact.
- reduction_nested room — 27 states — zone MAE 0.0176 K — valid.
- spatial_factorial room__coarse — 13 states — zone MAE 0.2715 K — valid.
- greybox control loop — no coverage — n/a — never closed.

## Decisions

- Claim: predictors are valid as gated diagnostics. Proof: adequacy v2
  nominal rows reject 0.0% with `first_alarm: null`; fault rows reject
  0.880184331797235 (~88%) with `first_alarm: 98`; `control_permission`
  is false in every row. Consequence: the gate discriminates cleanly
  but grants no control authority by itself.
- Claim: the workday LOBO does not support forecasting superiority
  claims. Proof: `greybox_lobo_workday_v1.json` summary shows negative
  `relative_reduction` at h6/h12 for all three variants (mass −28%/−10%,
  pooled_adapted −24%/−16%, pooled_frozen −66%/−50%), with only 1–6 of
  30 cases beating persistence. Consequence: recorded as plan B; no MPC
  loop was closed on these predictors.
- Claim: control loop never closed. Proof: `CausalGreyBoxModel.trusted`
  returns False unconditionally ("Forecast improvements do not
  establish counterfactual control accuracy"); `forecast_gate` in the
  harness is documented as "NOT permission to enable MPC"; no MPC
  artifact exists in results. Consequence: the line parks as plan B.
- Claim: all validation is offline replay. Proof: every JSON protocol
  string leads with synthetic/offline scoping ("no environment runs",
  "no simulator", "held at origin", "OFFLINE ORACLE"); harnesses read
  only banked `obs_dump.npz` windows. Consequence: numbers describe
  predictor diagnostics, not policy scores.
- Claim: manifest pins the synthetic toolchain. Proof:
  `thermal_adequacy_v2_manifest.json` records sha256 of the report and
  nine tool sources. Consequence: adequacy v2 is reproducible and
  tamper-evident.

## Open

- Equipment observers — done 10-24.
- MPC loop closure on greybox predictions — open, never done.
- Workday forecast skill (beating persistence at h6/h12) — open; the
  blocker for promoting plan B.

## Appendix

### A. File roles (all bodies opened in full)

- `smart_buildings/baselines/causal_greybox.py` — `CausalGreyBoxModel
  (GreyBoxModel)`: 7-feature bounded delta-T ARX. Features per
  (temp, delta, inputs): (ambient−temp)/40, (mean−temp)/5,
  flow·(supply−temp)/10, flow·reheat_gap/10, comfort, 1.0, delta, with
  reheat_gap = min(reheat·max(0, water−supply), max(0, 308.15−supply)).
  Fit: `lsq_linear` with ridge 0.01 prior pull toward `feature_prior =
  [0,0,0,0,0,0,0.5]`, bounds lower [0,0,0,0,−1,−1,0], upper
  [2,1,2,2,1,1,0.98]. `sensor_delay` ∈ {0,1}; `calibrate_plant`
  learns damper/pressure floors, water heat/cool steps from observed
  exceedances. `inputs()` reconstructs supply as
  min(recirc·mean_zone + (1−recirc)·ambient, SAT setpoint) and flows as
  damper·design·sqrt(pressure/2000)·run with a handler-capacity ratio
  cap. `predict_action_path` is an explicitly labeled OFFLINE ORACLE
  diagnostic (recorded future commands, held weather/water).
  `steps_to_reach` raises NotImplementedError; `trusted` returns False.
- `smart_buildings/baselines/thermal_greybox.py` —
  `ThermalGreyBoxModel(CausalGreyBoxModel)`: adds `physical_water`
  (Kv·sqrt(bar) water flows in m³/s, i.e. m³/h ÷ 3600, with
  `water_kv=0.108`, `header_resistance=0.0`; `coil_power` bounded by
  air/water capacities and the 308.15 K discharge headroom) and
  `thermal_mass` (filtered air-temp proxy mass with time constant
  `mass_time_steps=24`, 8th coefficient in [0,1]). Constructor rejects
  mass_time_steps < 1, negative kv, negative header resistance.
- `smart_buildings/baselines/pooled_greybox.py` —
  `fit_population_prior`: 8 shared coefficients, equal total weight per
  case (gram += XᵀX/n), ridge 0.01 toward DEFAULT_PRIOR
  [0,0,0,0,0,0,0.5,0], Cholesky + `lsq_linear` in FEATURE bounds
  (lower [0,0,0,0,−1,−1,0,0], upper [2,1,2,2,1,1,0.98,1]).
  `PooledGreyBoxModel`: population prior, `fit_window=72`,
  `local_ridge_strength=16.0`, `adapt` flag (frozen vs prefix-adapted).
- `smart_buildings/baselines/equipment_balance.py` — pure balances:
  AIR_CAPACITY = 1.2·1006 J/(m³K), WATER_CAPACITY = 1000·4180.
  `air_flows` (pressure-driven, sqrt(p/design_p), handler-capacity cap,
  zeros when off); `air_handler` (mixed = recirc·return +
  (1−recirc)·outdoor, supply = min(mixed, cooling setpoint), fan =
  flow·pressure/efficiency); `water_flows` (Kv m³/h → m³/s ÷ 3600,
  header coupling); `reheat_coil` (min(air,water)·gap capped by
  max-discharge headroom); `zone_derivatives` (two-node K/s, PRE-reheat
  supply convention — post-coil + reheat_power would double-count).
- `tools/greybox_check.py` — MODEL_KINDS = legacy, causal, sensors,
  delayed, calibrated, water, mass, thermal. `evaluate`: windows of
  `steps=288` from `offset`, `fit_window=144` transitions warmup,
  horizons (1,3,6,12), per-room MAE vs persistence and linear-drift
  baselines, skill = (persist − model)/persist, regime splits
  (active/passive, occupied/empty), prefix-excitation counters
  (≥8 transitions per room). Oracle-actions mode requires
  action-conditioned kinds.
- `tools/pooled_greybox_check.py` — LOBO driver: `--steps 288`,
  `--offset 576`, `--fit-window 72`, `--horizons 6 12`,
  `--local-ridge-strength 16.0`, `--policy dsat_gen31`. Per fold:
  population prior on other buildings → evaluate held-out scenarios
  under three variants (mass baseline, pooled_frozen, pooled_adapted).
  `forecast_gate` (20% minimum reduction in all/active/occupied with no
  per-building regression) is labeled "NOT permission to enable MPC".
  Saved report has no `forecast_gate` key — the shipped
  `greybox_lobo_workday_v1.json` predates it (keys end at `summary`).

### B. Full artifact registry (every file opened; protocol verbatim)

- `thermal_adequacy_v1.json` (158013 bytes, 6641 lines): protocol
  "Known synthetic family; selection obs0..47, independent calibration
  obs48..71; pre-update forecasts only; frozen thresholds; no automatic
  recovery, environment or control". 5 rows, 288 physical steps/case,
  selection 48 + calibration 24 obs. Nominal rows (retained 19 / 135):
  rejected 0.0, first_alarm null. Fault rows: rejected
  0.880184331797235 (~88%), first_alarm 98. control_permission false
  everywhere; `empirical_not_confidence_guarantee: true`.
- `thermal_adequacy_v2.json` (224035 bytes, 9295 lines): same protocol,
  7 cases. Rows: nominal 19 → 0.0/null; nominal 135 → 0.0/null; fault
  19 → 0.8802/98; fault 135 → 0.8802/98 (twice, two configs); plus a
  retained-5 row → 0.0968/268 and a retained-135 row → 0.0/null.
  Draft shorthand preserved verbatim: nominal rejected 0.0% no alarm /
  fault rejected 88% first alarm @ obs 98.
- `thermal_adequacy_v2_manifest.json` (1250 bytes, 15 lines): report
  path + report_sha256 c10561d8…f958a51 + source_sha256 of nine tools
  + protocol "Synthetic 288-step responses; no environment runs or
  policy changes".
- `thermal_family_v1.json` (555078 bytes, 20899 lines): protocol
  "Finite parameter/state family selected on prefix only; membership
  and weights frozen thereafter; causal state updates; nominal
  conditional future inputs; no environment or policy". 16 rows, 288
  steps/response, prefix 72, states_per_member 13, prior_scales
  [0.8,1.0,1.2], five prior load pairs (±5 W), prefix_score_floor 0.001
  K, envelope sigma multiplier 2. Coverage claim: "Empirical
  finite-family coverage only, NOT a calibrated confidence interval".
- `thermal_grid_reference_v1.json` (2685 bytes, 88 lines) and
  `thermal_grid_reference_v2.json` (8445 bytes, 352 lines): protocol
  "Synthetic single-step algebra fixtures only; not a building rollout
  or score". Full 47 → reduced 27 states ("Piecewise-constant room
  air/slab; all wall CVs retained; no fitting"). Uniform-init row:
  source_reference_maximum_error 6.252776074688882e-13 K; reduced zone
  errors 0.012327377640427056 / 0.005930811699395615 K; full balance
  residuals ~1e-11 W, reduced max node residual 31.29 W.
- `thermal_mismatch_v1.json` (896599 bytes, 28091 lines) and
  `thermal_mismatch_v2.json` (915978 bytes, 28631 lines): protocol
  "Synthetic plant perturbed; nominal bases, observer, priors and
  noise frozen; no refitting or truth-parameter/state updates;
  forecasts use nominal conditional future inputs". 108 rows each, 288
  steps, prefix 72, noise_seed 202610, target = noiseless latched
  physical zone temperature, observer sensor std 0.0001 K, process std
  0.01 K. Perturbations include nominal + capacity ±20% (0.8/1.2);
  origin windows pulse [84,120], hidden heat [84,192].
- `thermal_port_modes_v1.json` (101296 bytes, 3121 lines): protocol
  "Known synthetic input-response bases; independent bounded
  responses; no trajectory fitting, convection, environment or
  policy". 4 cases, 288 steps, full 164 states; design periods
  600/1800/3600 s, feedback conductances [0,0]/[4,3]; DC column error
  ~4.87e-14.
- `thermal_system_modes_v1.json` (214531 bytes, 6968 lines): protocol
  "Whole-system reduction of known synthetic matrices; independent
  bounded responses with oracle initial projection, no fitting,
  convection or environment". 5 cases, 288 steps, 164 states,
  snapshot rank 25, spectral radius ~0.988–0.989 (stable).
- `thermal_wall_modes_v1.json` (52818 bytes, 1613 lines): protocol
  "Known synthetic coefficients; full air/slab ports; capacity
  projection; independent 1..288-step response, no
  fitting/convection/environment". 4 cases, 288 steps, 164 states
  (120 air/slab ports + 44 wall states), wall time constants
  ~5500–10050 s, DC column error ~4.87e-14.
- `thermal_state_observer_v1.json` (49376 bytes, 1588 lines, 15 cases)
  and `thermal_state_observer_v2.json` (89658 bytes, 3001 lines, 21
  cases; both have empty `rows`): protocol "Known synthetic
  coefficients; causal two-sensor updates and past forcing only;
  forecasts conditional on supplied future synthetic inputs; no
  policy/environment". 288 physical steps, 289 observations incl.
  reset, prior cell std 2.0 K, process std 0.01 K/step, assumed meas
  std 0.0001 K, actual synthetic noise 0.0 K, parameters frozen,
  duplicate-reset-updated-twice false, delay "obs[t] is C x[t-1];
  pending u[t-1] then new u[t]; h1 does not evaluate a new action".
- `thermal_reduction_nested_v1.json` (18685 bytes, 647 lines):
  protocol "Known synthetic discrete system, independent free response
  per model; no observation corrections, convection shuffling,
  scenario fitting or environment runs". 4 cases, 300 s/step; room
  reduction 27 states: zone MAE 0.017579275276727466 K, max
  0.02369921329034241 K.
- `thermal_spatial_factorial_v1.json` (19118 bytes, 519 lines) and
  `thermal_spatial_factorial_v2.json` (28086 bytes, 759 lines):
  protocol "Synthetic known-system independent free responses, no
  fitting, no environment runs or convection shuffling; same synthetic
  coefficients for all bases". 4 cases, 164 states, room_cells
  [30,30], 300 s/step, first-step source-reference error 1.08e-12 K,
  node balance residual 1.29e-10 W; room__coarse (13 states, 12 core
  cells/room): zone MAE 0.27154649299256267 K.
- `thermal_prefix_identification_v1.json` (238740 bytes, 7014 lines)
  and `thermal_prefix_identification_v2.json` (246279 bytes, 7217
  lines): protocol "Prefix-only conditional local scalar fit; nominal
  basis; no true labels/states or validation targets in fitting;
  frozen coefficients after prefix; causal latent updates only". 12
  rows each, 288 steps, prefix 72. Gates: sensitivity RMS 0.001
  K/coordinate, singular ratio 0.001, min prefix improvement 0.1,
  regularization 0.001 K, optimizer max_nfev 40.
- `greybox_lobo_workday_v1.json` (199320 bytes, 7105 lines): protocol
  "leave-one-building-out; frozen population prior or causal
  adaptation". steps 288, offset 576, fit_window 72,
  local_ridge_strength 16.0, horizons [6,12], policy dsat_gen31,
  oracle_actions false, population_ridge_strength 0.01. 6 folds (held
  out 1/2/4/5/7/8; training rows 407540/341530/386015/350140/411845/
  420455 — draft figure 407540 preserved verbatim as the fold-1
  count). 90 rows = 30 scenarios × 3 variants. Summary (model/persist
  MAE, K): mass h6 0.1883/0.1466 (skill −28%, 1/30 beat), h12
  0.2924/0.2650 (−10%, 6/30); pooled_adapted h6 0.1817/0.1466 (−24%,
  2/30), h12 0.3070/0.2650 (−16%, 3/30); pooled_frozen h6 0.2431
  (−66%, 0/30), h12 0.3985 (−50%, 0/30). Active/occupied splits track
  the same ordering. Fold-1 coefficients start [0.1279, ~0, 0.5761,
  0.8158, 0.0436, 0.0122, 0.2713, 0.3698].

### C. Honest gaps

- The draft's "predictors valid" covers the synthetic adequacy gates;
  on workday LOBO data every variant trails persistence — both facts
  are kept side by side above, not averaged away.
- `thermal_state_observer_v1/v2` carry cases but empty `rows`: spectra
  and observability data only, no forecast-error rows.
- The saved LOBO report has no `forecast_gate` verdict (harness added
  the gate later); with the 20%-improvement/no-regression rule and the
  negative skills above, the gate would fail on this report — from
  memory of the rule text, not a computed re-run.
- Timestamps: result JSON mtimes span 10-23 (LOBO 22:54) into 10-25
  (thermal_* 12:04–13:16); the synthetic thermal family was banked
  after this day. Day attribution follows the work (validation
  protocols written 10-23), artifact bytes landed later.

