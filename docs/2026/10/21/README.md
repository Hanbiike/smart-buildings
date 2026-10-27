# 2026-10-21 — Gen-3.1 dsat stack (public 0.01350); BC scaffold at night

## Summary

The gen-3.1 rule stack (`dsat_gen31`) was frozen, pushed through the
submission notebook, submitted, and scored 0.01350 public COMPLETE.
The stack layers seven independent economies onto the leaderboard-verified
zone-PI skeleton: occupant-gated PI hours, down-only cheap-side zone targets,
minimum static pressure on every running AHU, per-AHU supply-air depth from
load mixedness (mild -0.4) and cooling dominance (deep -1.0), a 2 K night
heating drop, a hot-water reset with pump idle-off, and a prestart window.
The decisive structural call was down-only shifting: symmetric targets that
push zones up in hot weather cost more comfort in tight bands than the chiller
ever repays, with singapore up +25% as the exhibit.
At night the supervised-learning line was scaffolded: shared observation
decoding (`tools/nn_features.py`), a windowed ControllerNet
(`smart_buildings/nn_model.py`), and a behavior-cloning trainer
(`tools/bc_train.py`) aimed at imitating gen-3.1 before any RL.
Bayesian optimization over the seven live knobs later confirmed the hand
values almost exactly (kp 0.025 vs 0.07 hand — the one axis BO pushed further;
all others within noise), an independent check documented in the Appendix.

## Status

| Whose | Score | State |
|---|---|---|
| us (gen-3.1, `dsat_gen31`) | 0.01350 | public, COMPLETE |
| others (`transfer`, organizer PID port) | 0.06439 | public, from docstring |
| others (`k10` zone-PI probe) | 0.01853 | public, from docstring |

Gen-3.1 is the standing freeze candidate: the gen-2 skeleton (pressmin,
down-shift, occupant gate) plus per-AHU SAT depth from load mixedness and
cooling dominance (`DsatGen31Policy`, `zone_pi.py:779-783`).

## Done

- Froze the gen-3.1 stack — file `smart_buildings/baselines/zone_pi.py`,
  class `DsatGen31Policy` (KEY `dsat_gen31`) — why: the mixedness-SAT +
  pressmin + down-shift skeleton was the best measured assembly and needed a
  stable submission target.
- Layered down-only target shifting — classes `DsatShiftDownPolicy` and
  `DsatPMShiftDownPolicy` (`zone_pi.py:413-431, 497-512`) — why: heating
  economy without the hot-climate comfort loss the symmetric shift caused.
- Added minimum pressure on all running AHUs — class
  `DsatPressureMinPolicy` (`zone_pi.py:447-463`) — why: fan power is the
  biggest electricity term in hot climates and the organizer baseline ran
  mid-scale pressure on cooling units.
- Added per-AHU SAT depth from load mixedness — classes `DsatMixedSatPolicy`
  and `DsatSatMixedDeepPolicy` (`zone_pi.py:724-776`) — why: mixed-load
  handlers pay twice for deep cooling (chill then reheat), while pure-cooling
  handlers want the deepest SAT; 24 h EMAs classify each handler with no
  climate identities.
- Added occupant gating, night drop, and water/pump economy — classes
  `DsatGen1OccGatePolicy`, `ZonePiDsatPolicy` (drop 2 K, prestart 2 h),
  `DsatWaterPumpFracPolicy`, `DsatWaterPumpIdleOffPolicy`
  (`zone_pi.py:380-387, 580-623`) — why: skip heating empty buildings, cut
  night gas, and stop pinning the pump on at zero heating demand.
- Shipped the submission notebook — file
  `notebooks/smart-buildings-submission-gen31.ipynb` — why: kernel push and
  submit run the hidden-scenario rerun; the notebook carries the full
  controller chain plus the inference-server call.
- Submitted and scored 0.01350 public COMPLETE — why: the number locks the
  rule-track ceiling that the later gen-4 screen and the NN line had to beat.
- Scaffolded behavior cloning at night — files `tools/nn_features.py`,
  `smart_buildings/nn_model.py`, `tools/bc_train.py` — why: the plan is
  supervised BC first, then RL, with gen-3.1 trajectories as teacher labels.

## Measured

- `dsat_gen31` — public LB, this account — 0.01350 — freeze candidate,
  COMPLETE submission.
- symmetric up-shift — hot tight bands (singapore_8 in docstring) — +25%
  regret — rejected, down-only fixed instead (`zone_pi.py:413-417`).
- ramp035 SAT glide — atlanta-style loads — -23% — shipped as the
  mixedness-gate predecessor; singapore-style sustained loads — -4% deep
  (`zone_pi.py:530-536`).
- instant demand ramps — mid-range cycling — +8..12% worse — rejected in
  favor of the slow EMA in `DsatAdaptiveSatPolicy` (`zone_pi.py:546-556`).
- gen-4 BO screen — 5-scenario exact screen — best -2.0% vs -5% gate —
  rule track parked; exact optimum in
  `smart_buildings/results/g4bo_params.json`, bounds and workflow in
  `tools/g4bo.py`.
- BC scaffold — code only, no trained weights this date — trainer defaults 30
  epochs, batch 128, lr 3e-4 (`bc_train.py:47-64`) — weights arrived 10-22.

## Decisions

- Claim: zone targets shift down only, never up. Proof: the upward shift in
  tight hot bands regressed singapore by +25% (`DsatShiftDownPolicy`
  docstring). Consequence: every shipped shift class clips the up side away,
  capped at 0.4 x band width.
- Claim: pressure goes to minimum on all running AHUs, not just forced ones.
  Proof: fan power dominates hot-climate electricity and the organizer ran
  mid-scale pressure. Consequence: `DsatPressureMinPolicy` pins -1.0
  normalized on every running unit.
- Claim: SAT depth follows per-AHU load mixedness, not one global schedule.
  Proof: mixed handlers reheat what deep cooling just chilled (double gas
  payment) while pure-cooling handlers gain from depth; the 24 h heat/cool
  EMAs separate them without climate labels. Consequence: mild -0.4 under
  mixed load, deep -1.0 under sustained cooling dominance, reference depth
  otherwise.
- Claim: supervised BC first, RL second. Proof: gen-3.1 trajectories give
  exact per-step labels for every actuator, so imitation bounds the search
  before exploration. Consequence: the night scaffold (features, model,
  trainer) with `dsat_gen31` as default teacher.

## Open

- Train the BC weights (done 10-22; carried from the earlier draft, run logs
  not re-opened this session).
- Start greybox world models for MPC (started 10-22; carried from the earlier
  draft).
- Run the gen-4 single-knob screen on the 30-table gate (later failed at
  -2.0% vs the -5% gate; parked as the future teacher config — see Appendix
  section D).

## Appendix

### A. Full dsat registry in file order (`zone_pi.py`, 1136 lines)

Base chain, all shipped underneath everything:

- `BaselinePolicy` (`zone_pi.py:42-76`): organizer `PIDWaterResetPolicy`
  plus sensor comfort bands, mean-zone aggregation, raw-kelvin outside-air
  decode. Submitted as `transfer`, public 0.06439. `USE_SENSOR_BAND = True`,
  `USE_WORST_ZONE = False` (worst-zone raised loss in both pilot scenarios).
- `CoolingHoldPolicy` (`:79-124`): cooling runs held `HOLD_STEPS = 288` steps
  at last commands (organizer loop chattered 50-107 on/off per day);
  `NIGHT_COOL_CEILING = True` swaps the eco high for the comfort high outside
  work hours so hot buildings do not drift overnight.
- `ZonePIPolicy` (`:127-237`, KEY `zone_pi`): per-zone PI to the band centre
  in work hours, one signal driving damper by |u| and reheat by -u, AHUs
  forced on (`FORCED_SAT = -0.7` ~ 15 C, `FORCED_PRESSURE = -1.0`). Class
  defaults are the k10 probe (public 0.01853): `KP = 0.1`, `KI = 0.005`,
  `PRESTART_H = 1`, `MIN_DAMPER = 0.1`, `HEAT_DAMPER_GAIN = 1.0`.
- `ZonePIEconFixPolicy` (`:240-271`, KEY `econ_fix`): the economizer fix —
  the organizer transfer policy decoded the AHU outside-air sensor as
  normalized when it arrives in kelvin, so the economizer saw ~3000 K and
  never opened. `WARM_SAT = None`, `WARM_BELOW_K = 283.15`.
- `DemandSatPolicy` (`:274-302`, KEY `dsat_bin`): supply temperature follows
  mean zone PI demand; net heating need pins max supply (no chiller), cooling
  demand ramps to `COLD_SAT = -0.7`, reached at mean demand `RAMP = 0.0`
  (binary switch at zero).
- `TunedPIPolicy` (`:305-363`, KEY `tuned`): switchable economy measures, all
  off by default: `WARM_SAT = -0.4`, `NIGHT_HEAT_DROP_K = 0.0`,
  `WATER_SCALE = 1.0`, `TARGET_SHIFT_K = 0.0`, `OCC_GATE = False`. The
  occupant gate only bites after arrival hour + 2 and below 0.5 occupants
  (`:330-336`); the symmetric shift caps at 0.4 x band (`:343`).
- `TunedDemandSatPolicy` (`:366-370`, KEY `tuned_dsat`): dsat_bin plus the
  Tuned switches; `WARM_SAT = None`.
- `ZonePiDsatPolicy` (`:373-387`, KEY `zone_pi_dsat`): the ds-k07 submission
  config, newer iteration, LB unconfirmed: `KP = 0.07`, `KI = 0.0035`,
  `HOLD_STEPS = 288`, `NIGHT_COOL_CEILING = True`,
  `FORCED_PRESSURE = -1.0`, `NIGHT_HEAT_DROP_K = 2.0`, `PRESTART_H = 2`.

Shipped layers (each in gen-3.1 through its descendants):

- `DsatTargetShiftPolicy` (`:390-396`, `dsat_shift10`): symmetric cheap-side
  targets, `TARGET_SHIFT_K = 1.0`, capped 0.4 x band; cold outside aims low,
  hot outside aims high.
- `DsatWaterScalePolicy` (`:399-403`, `dsat_water07`): hot-water command at
  70% of range, gas saving.
- `DsatOccGatePolicy` (`:406-410`, `dsat_occgate`): `OCC_GATE = True`, skips
  heating empty buildings past arrival + 2 h.
- `DsatShiftDownPolicy` (`:413-430`, `dsat_shiftdown`): down-only 1.0 K with
  the 0.4-band cap; the up-shift is clipped because tight hot bands lost more
  comfort than the chiller saved (singapore_8 +25%).
- `DsatShiftHalfPolicy` (`:433-437`, `dsat_shift05`): small symmetric 0.5 K.
- `DsatShiftDownOccGatePolicy` (`:440-445`): down-shift plus occupant gate.
- `DsatPressureMinPolicy` (`:447-463`, `dsat_pressmin`): -1.0 normalized
  pressure on every running AHU, 200 Pa floor.
- `DsatRampPolicy` (`:466-471`, `dsat_ramp05`): `RAMP = 0.5` glide instead of
  the binary deep-cool switch.
- `DsatPressureMinRampPolicy` (`:474-478`): pressmin plus ramp05.
- `DsatNightDrop3Policy` (`:481-487`, `dsat_nightdrop3`): `3.0` K night
  heating setback.
- `DsatPMShift05Policy` (`:490-494`, `dsat_pm_shift05`): pressmin plus
  symmetric 0.5 K (independent mechanisms).
- `DsatPMShiftDownPolicy` (`:497-512`, `dsat_pm_shiftdown`): pressmin plus
  down-only 1.0 K; the gen-2-era parent of the freeze chain.
- `DsatSatMildPolicy` (`:515-520`, `dsat_sat_mild`): `COLD_SAT = -0.4` gentle
  cooling supply for hot climates with deep dampers.
- `DsatSatDeepPolicy` (`:523-528`, `dsat_sat_deep`): `COLD_SAT = -1.0` max
  cooling per unit air.
- `DsatRamp035Policy` (`:531-536`, `dsat_ramp035`): `RAMP = 0.35` early
  saturation; atlanta wanted mild (-23%), singapore wanted deep (-4%).
- `DsatRamp035DeepPolicy` (`:539-544`, `dsat_ramp035_deep`): ramp035 at the
  deepest SAT.
- `DsatAdaptiveSatPolicy` (`:546-577`, `dsat_adaptive`): SAT depth from a
  slow EMA of cooling demand (`EMA_KEEP = 0.97`, `EMA_SATURATION = 0.35`,
  `COLD_SAT = -1.0`); the slow EMA stops the mid-range cycling that made
  instant ramps +8..12% worse.
- `DsatGen1OccGatePolicy` (`:580-585`, `dsat_gen1_occ`): gen-1 with occupant
  gating.
- `DsatWaterPumpFracPolicy` (`:587-602`, `dsat_wpump_frac`): pump head follows
  the water-reset fraction above 0.02 instead of pinning full scale.
- `DsatWaterPumpIdleOffPolicy` (`:605-623`, `dsat_wpump_idleoff`): pump head
  plus discrete fully off when the reset fraction is under 0.05 and max zone
  heat under 0.05.
- `DsatGen2Policy` (`:625-629`, `dsat_gen2`): the Stage-2 assembly (pressmin,
  down-shift, occupant gate) renamed as the standing configuration.
- `DsatMixedSatPolicy` (`:724-766`, `dsat_sat_mixed`): per-AHU SAT from load
  mixedness — `AHU_EMA_ALPHA = 1.0 - 1.0/288` (~24 h), `HEAT_SHARE_GATE =
  0.05`, `MILD_SAT = -0.4`, `COOL_ACTIVE_GATE = 0.02`,
  `COOL_SHARE_DEEP_GATE = None` (base never goes deep).
- `DsatSatMixedDeepPolicy` (`:769-776`): sustained cooling-dominant handlers
  (`COOL_SHARE_DEEP_GATE = 0.7`) saturate `DEEP_SAT = -1.0`.
- `DsatGen31Policy` (`:779-808`, `dsat_gen31`): the freeze; adds `_dump_state`
  with replayable per-step channels (zone PI signals and integrals, per-AHU
  mixedness/cooling EMAs, hold counters, PI-hours flag, occupant EMA), all
  deterministic functions of the obs history.
- `DsatGen3Policy` (`:825-831`, `dsat_gen3`): Stage-3 assembly standing on
  gen-2 with mixedness-driven SAT depth; freeze candidate for validation.

Rejected variants (tested, not shipped):

- `DsatResetFull25Policy` (`dsat_reset_full25`): water reset saturates at 2.5
  K mean shortfall instead of 4.0 (`RESET_FULL_K = 2.5`).
- `DsatResetEffort2Policy` (`dsat_reset_effort2`): double air-side saturation
  weight (`RESET_EFFORT_WEIGHT = 2.0`).
- `DsatResetTightPolicy` (`dsat_reset_tight`): tighter reset deadband
  (`RESET_DEADBAND_K = 0.1` vs 0.25).
- `DsatForcedPressureUpPolicy` (`dsat_fp_06`): forced AHUs at mid-scale
  pressure (`FORCED_PRESSURE = -0.6`); the docstring already foresaw pressure
  starvation in deep-cold escalation.
- `DsatForcedPressureHighPolicy` (`dsat_fp_02`): forced AHUs at
  `FORCED_PRESSURE = -0.2`.
- `DsatShiftDeepPolicy` (`dsat_shift_deep`): `TARGET_SHIFT_K = 2.0` with cap
  raised to 0.6 x band (1.8 K); `DsatShiftDeepestPolicy` pushes the cap to
  0.8 x band (2.4 K).
- `DsatPrestartGatePolicy` (`dsat_prestart_gate`): 24 h occupant EMA
  (`OCC_24H_ALPHA = 1.0 - 1.0/288.0`) skipping prestart burn after empty days.
- `DsatPrestart1Policy` / `DsatPrestart3Policy`: 1 h vs 3 h prestart against
  the standing 2 h.
- G4 screen series (`DsatG4*`, `zone_pi.py:840-1136`): single-knob re-parents
  onto the gen-3.1 chain (reset, pressure, prestart-gate, mild SAT, heat
  gate, pump frac/idle-off, prestart 2 h, shift 1.5/2.0, water 0.8/0.7/0.6,
  night drop 3.0, KP 0.04-0.08 plus 0.03/0.035/0.045 floor probes, damper
  0.85, combos, KI rebalance and OAT-split KI). Screened after this date;
  outcome from `tools/g4bo.py:4-8`: rounds 4-6 could not reach the -5% gate
  (-2.0% exact at best), so the rule track parked and the best combo stayed
  a future teacher config.

### B. POLICY_MAP cross-check (`smart_buildings/evaluate.py:198-339`)

Every dsat KEY above is registered in POLICY_MAP under its own string, from
`zone_pi` and `zone_pi_dsat` through `dsat_shift10`, `dsat_water07`,
`dsat_occgate`, `dsat_shiftdown`, `dsat_shift05`, `dsat_shiftdown_occgate`,
`dsat_pressmin`, `dsat_ramp05`, `dsat_pressmin_ramp05`, `dsat_nightdrop3`,
`dsat_pm_shift05`, `dsat_pm_shiftdown`, `dsat_sat_mild`, `dsat_sat_deep`,
`dsat_ramp035`, `dsat_ramp035_deep`, `dsat_adaptive`, `dsat_gen1_occ`,
`dsat_wpump_frac`, `dsat_wpump_idleoff`, `dsat_gen2`, the six reset/pressure/
shift/prestart rejects, `dsat_sat_mixed`, `dsat_prestart1`, `dsat_prestart3`,
`dsat_gen3`, `dsat_sat_mixed_deep`, `dsat_gen31`, and the full `g4_*` screen
(`evaluate.py:215-293`). The map also registers the later lines that grew
out of this day: `occ_gated*`/`occ_sat*`, slot policies, the `probe_*`
diagnostics, and the BC checkpoints `bc_v0` through `bc_v3b` with the
`dagger_*` collectors (`:294-339`).

### C. Submission notebook

`notebooks/smart-buildings-submission-gen31.ipynb` (10 cells: imports, EDA
markdown plus EDA code, inference-guard code, Controllers markdown, full
controller-chain code, gen-3.1 controller markdown, kaggle server import,
Inference markdown, server run call). Cell 5 ports the whole LB-verified
chain (Baseline, CoolingHold, ZonePI, EconFix, DemandSat, TunedPI, gen-2/3
economics) on top of the organizer `PIDWaterResetPolicy`; the head comment
names minimum running pressure, demand-shifted targets, occupant-gated PI
hours, and per-AHU mixedness SAT. Cell 8 notes the server runs the public
example on a normal commit and hidden scenarios writing `submission.parquet`
on rerun, with internet off; cell 9 calls
`SmartBuildingsInferenceServer(SubmissionPolicy).run(...)`.

### D. Bayes optimum (`smart_buildings/results/g4bo_params.json`)

Exact file content: `{"kp": 0.025, "ki": 0.003523, "water": 0.7637205,
"mild": -0.384852, "hgate": 0.0500838, "shift": 0.9992882, "ndrop":
2.0005794}`. Search bounds and standing defaults from `tools/g4bo.py:34-48`:
kp in (0.025, 0.095) default 0.07; ki in (0.002, 0.0065) default 0.0035;
water in (0.45, 1.0) default 1.0; mild in (-0.55, -0.2) default -0.4; hgate in
(0.04, 0.12) default 0.05; shift in (0.0, 1.5) default 1.0; ndrop in (0.0,
3.0) default 2.0. The optimum sits at the kp lower bound (0.025 vs hand
0.07 — the one axis BO pushed past the manual stack), ki 0.0035 exact,
water 0.76 vs hand 0.7-0.8 winners, mild -0.38 vs hand -0.4, hgate 0.05
exact, shift 1.0 exact, ndrop 2.0 exact. Screen: 5 scenarios (fb1, atl2,
fb2, fb8, sg7), 864 steps, target = sum of |reward|/steps from on-disk CSVs
(`g4bo.py:73-90`); a fresh champion needs CONFIRM_MARGIN 1e-4 over the best
warm config to trigger the full 30-table gate check (`:159-185`). The
earlier draft's rounded values (0.025, 0.0035, 0.76, -0.38, 0.05, 1.0, 2.0)
match the file to the quoted precision.

### E. Night BC scaffold (three files, read in full)

- `tools/nn_features.py` (211 lines): `FeatureContext` built once per
  building from metadata; decodes room features [Z, 8] (band deltas, zone/OAT
  delta, absolute temp, damper/reheat commands), AHU features [A, 10]
  (OAT/SAT kelvin, SAT delta/setpoint, pressure and flow ratios, fan/recirc
  commands, request/run), and globals (occupant, comfort flags, hod/dow
  harmonics, HWS). Same code path for single rows and [T, D] matrices;
  kelvin-vs-normalized guard at 150.0; room scales `SCALE_K = 10.0`,
  `DIFF_K = 8.0`; defaults 291.0/297.0 K when setpoint sensors are absent.
- `smart_buildings/nn_model.py`: shared decoding plus the windowed
  ControllerNet — history of 12 pooled tokens at stride 6 over the last 72
  steps (`WINDOW = 72`, `STRIDE = 6`, `N_HIST = 12`), one token per room, per
  AHU, and global; per-room (damper, reheat), per-AHU (SAT, pressure,
  recirculation, run bit), and global water heads, all permutation
  equivariant across building sizes. Correction to the earlier draft: the
  model lives at `smart_buildings/nn_model.py` (imported by the trainer as
  `from smart_buildings.nn_model import ControllerNet, N_HIST, STRIDE,
  WINDOW`), not at `tools/nn_model.py`, which does not exist.
- `tools/bc_train.py` (280 lines): default teacher `dsat_gen31`, hold
  building `building_train_5`, 30 epochs, batch 128, lr 3e-4, d-model 64, 2
  layers, 4 heads, seed 0; OneCycleLR schedule; masked MSE on continuous
  heads (recirc at half weight, water terms at half weight) plus BCE on run
  bits; checkpoints to `smart_buildings/results/nn_runs/bc_v0.pt`; DAgger
  bags supported via label override (`dagger_labels.npz`).

### F. Unresolved items and provenance notes

- `results/g4bo_params.json` (repo-root `results/`) does not exist; the live
  file is `smart_buildings/results/g4bo_params.json`. The earlier draft's
  path is kept above only through this note.
- `tools/nn_model.py` does not exist; the model is
  `smart_buildings/nn_model.py` (see section E).
- The 10-27 probe closures (SetpointProbe, occ5, TodShiftProbe) and the
  10-22 BC-weight and greybox follow-ups are carried from the earlier drafts
  from memory; their code and logs were not re-opened this session.
- No G4 full-30-table champion ever passed the -5% gate per `tools/g4bo.py`;
  the frozen rule-track ceiling stays gen-3.1 at 0.01350 public.
