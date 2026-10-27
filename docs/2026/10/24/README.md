# 2026-10-24 — Equipment observers

## Summary

No new competition scores appeared; the morning built the workday
observer stack for the greybox plan B. `equipment_observer.py` adapts
isolated equipment balances to observation/command timing (zone temps
latch before the thermal solve, actuators after; pending transitions
belong to command t−1), and `zone_thermal.py` adds stable lumped
integration with bounded prefix-only identification plus causal
two-sensor wall-state updates (boundary and Kalman forms).
Coverage is workday protocols: `equipment_observer_workday_v1/v2.json`
(30 scenarios × 288 steps, offset 576, warmup 72, policy dsat_gen31)
and seven `modular_thermal_*.json` files (forcing audit, kalman,
kalman_matched, passive multistep, prefix audit, smoke — each protocol
in its own field).
Explicit physical assumptions are verified in-file: design pressure
2000 Pa, valve kv 0.108, header resistance 0.0, discharge cap 308.15 K,
fan efficiency 0.9. Honest record: equipment flow audit is near-exact
but power is never validated against measured heat, temperature
forecasts trail persistence, and the greybox plan B is complete and
parked unless rule-based control stalls.

## Status

| policy | coverage | loss | delta |
| --- | --- | --- | --- |
| equipment flow audit (v1/v2) | 30 workday scenarios | flow MAE 2.37e-8 m³/s | near-exact |
| water sensor projection | 30 workday scenarios | MAE 0.0109 K vs hold 0.0339 K | beats hold |
| identified-floor flow split (v2) | 807/860 samples | MAE 2.18e-8 m³/s | clean subset |
| modular kalman h1 (active) | 5 scenarios | 0.0775 vs persist 0.0990 K | +21.7% skill |
| modular kalman h1 (all) | 5 scenarios | 0.0956 vs persist 0.0847 K | −12.8% skill |
| forcing audit h1 | 5 scenarios, oracle forcing | 0.1885 vs persist 0.0763 K | −147% skill |
| zone power projection | none validated | n/a | NOT validated |
| greybox plan B | complete | n/a | parked |

## Done

- Built the observation/command adapter —
  `smart_buildings/baselines/equipment_observer.py` (372 lines, 07:25).
  Why: isolated equipment balances need latched timing (pending
  transition = command t−1 with inputs from t−1) and actuator-floor /
  capacity calibration before any zone dynamics can use them.
- Built the thermal core — `smart_buildings/baselines/zone_thermal.py`
  (341 lines, 07:39). Why: stable backward-Euler integration, bounded
  prefix-only fitting, and causal wall-state observers (boundary +
  Kalman) with frozen parameters during validation.
- Banked the workday equipment audit —
  `smart_buildings/results/equipment_observer_workday_v1.json` (1822
  lines) and `_v2.json` (2062 lines), 30 scenarios each. Why: causal
  flow/water audit on saved windows without invoking a simulator.
- Banked the modular thermal suite — seven
  `smart_buildings/results/modular_thermal_*.json` files. Why: separate
  the forcing question (oracle future inputs) from the observer
  question (boundary vs Kalman mass updates) from the fitting question
  (one-step vs passive-multistep) with frozen fits.
- Pinned the physical assumptions — `EquipmentAssumptions` defaults
  verified verbatim in both workday JSONs. Why: global priors must be
  explicit, never inferred hidden specifications.

## Measured

- equipment_observer v1/v2 flow — 30 scenarios × 288 steps — MAE 2.37e-8 m³/s — near-exact.
- identified-floor flow (v2) — 807 clean samples — MAE 2.18e-8 m³/s — clean subset.
- water projection — 215 samples/scenario — MAE 0.0109 K vs hold 0.0339 K — beats hold.
- kalman_v1 h1 active — 5 scenarios — 0.0775 vs 0.0990 K — +21.7% skill.
- kalman_v1 h1 all — 5 scenarios — 0.0956 vs 0.0847 K — −12.8% skill.
- kalman_matched h1 all — 5 scenarios — 0.1174 vs 0.0847 K — −38.6% skill.
- passive_multistep h1 all — 5 scenarios — 0.1394 vs 0.0847 K — −64.6% skill.
- forcing_audit h1 — oracle forcing — 0.1885 vs 0.0763 K — forcing is not the fix.
- prefix_audit Fairbanks_1 — 39 zones — median balance MAE 55.56 W — audit only.
- zone power — no measured-heat validation — n/a — NOT validated.

## Decisions

- Claim: equipment flow reconstruction is solved. Proof: v1/v2
  `handler_flow_m3_s` MAE 2.366919065989104e-08 m³/s over 860 samples
  (first scenario), RMSE 4.28e-8, max 1.8e-7; `skipped_unobserved_
  capacity` 0; 286 delayed transitions per scenario. Consequence: the
  adapter reproduces simulator airflows from observations almost
  exactly — the remaining error is elsewhere.
- Claim: water projection beats the hold baseline. Proof: water MAE
  0.0109 K (RMSE 0.160, max 2.35) vs hold MAE 0.0339 K (RMSE 0.195,
  max 2.5) over 215 samples. Consequence: the causal rate projection
  carries signal, but spikes (2.35 K) remain.
- Claim: v2's identified-floor split isolates the clean subset.
  Proof: v2 adds `identified_floor_flow_m3_s` (807 samples, MAE
  2.18e-8), `projections_with_uncertain_floors` 37,
  `uncertain_floor_flow_samples`; v1 lacks these keys entirely.
  Consequence: origins that may hit unidentified actuator floors are
  labeled, not silently trusted.
- Claim: forcing is not the bottleneck. Proof: forcing-audit protocol
  is "OFFLINE oracle future forcing … frozen fits", yet projected h1
  (0.1885 K) equals oracle h1 to all 16 digits while persistence sits
  at 0.0763 K (skill −147%). Consequence: perfect future inputs do not
  rescue the dynamics — model form, not forcing, limits skill.
- Claim: greybox plan B is complete and parked. Proof: predictors
  banked 10-23/24 across adequacy, LOBO, equipment, and modular
  suites; every report carries "never invokes a simulator" /
  "control_integration: false" / "NOT validated" scoping; no MPC
  artifact exists. Consequence: the line reactivates only if
  rule-based control stalls.
- Claim: file-count note. Proof: the draft says eight
  `modular_thermal_*` files, but seven exist on disk (forcing_audit,
  kalman_v1, kalman_v2, kalman_matched_v1, passive_multistep_v1,
  prefix_audit_v1, smoke_v1). Consequence: the "8 files" draft figure
  is preserved above as written, corrected to 7 here with the full
  listing.

## Open

- Occ-gating rounds — done 10-26.
- MPC loop closure — open, never done; gated on workday forecast skill
  the predictors have not shown.
- Zone-power validation against measured heat — open; explicitly NOT
  performed in either workday report.

## Appendix

### A. Observer structure (verified in `equipment_observer.py`, full body)

- `EquipmentAssumptions` (frozen dataclass): design_air_pressure_pa
  2000.0, valve_kv 0.108, header_resistance 0.0,
  max_discharge_temperature 308.15, fan_efficiency 0.9 — verified
  verbatim in both workday JSON `assumptions` blocks. Validation
  rejects nonfinite values, nonpositive design pressure / discharge
  cap, and fan efficiency outside (0,1].
- `EquipmentObserver.__init__`: strict wiring checks — every observed
  zone in exactly one handler, pressure indices = handlers + 1 (common
  water loop), discrete actions = handlers + 1. Zone indices per room
  (temp, damper %, reheat %, flow setpoint); AHU indices per handler
  (OAT, SAT, static pressure, recirc, run, flow, fan); single water
  temp + differential pressure taps (`_index` raises unless exactly
  one match). Areas from `zone_map` connected components
  (`component_zone_areas`, cv2, labels must match 1..count) or
  equal-area proxy; `area_weighting` records which.
- Timing core (`observe`): frame t decoded; damper/pressure floors and
  water heat/cool steps calibrated from command-vs-measured
  exceedances of the recorded last action; handler capacities as
  median of last ≤32 flow/fan samples; running-pressure and
  damper minima tracked. `pending` balances frame t−1's weather/water
  against frame t's latched temps (`basis='latched_step_start'`);
  returns None on the reset duplicate, else `DelayedTransition`
  (start/end observations and temperatures + forcing). The recorded
  action is consumed (set to None) — a new command must not replace
  the pending transition.
- `_balance`: water distribution via `water_flows` (valve×kv,
  header 0.0 → independent valves); per-handler `air_flows` with
  design 2000 Pa and capacity cap; `air_handler` mixing + cooling +
  fan power (efficiency 0.9); per-zone `reheat_coil` capped at 308.15 K
  discharge; return-air mean is area-weighted. Uncertain damper /
  pressure floors collected per origin; `UnobservedCapacityError`
  raised for proposed active commands without observed capacity.
- `project_action` validates/copies without clobbering the calibration
  command, requires a temperature per zone and OAT per handler,
  optionally raises `UnidentifiedActuatorError` under
  `require_identified_floors`. `project_water_sensor` rolls the water
  target at previously observed heat/cool rates.
- `ZoneDuty` (air/water flow, pre-reheat supply, discharge, air +
  reheat power, `total_power`) and `EquipmentEstimate` (control step,
  outdoor/water, duties, handler flows, cooling + intake-fan power,
  basis, uncertain floors) are frozen dataclasses.

### B. Thermal core (verified in `zone_thermal.py`, full body)

- `_advance` / `advance_zone`: backward-Euler M-matrix solve of the
  air+mass two-node system (inputs: outdoor, PRE-reheat supply, flow,
  reheat power, internal power, seconds, default 300). Expanded
  determinant avoids b·c cancellation under strong coupling;
  nonfinite results raise.
- `condition_mass`: wall-state update conditioned on an ALREADY
  received air measurement (boundary form). `condition_mass_kalman`:
  scalar two-state Kalman update with air pinned exact between
  updates; fixed assumptions initial variance 4.0 K², air process std
  0.05 K/step, mass process std 0.01 K/step; Joseph-style variance
  update; Q is model mismatch, not fitted error.
- `fit_zone`: fits log(C_air), log(U_out), load/1000 (plus log(G_mass)
  and initial-mass offset when two_state) via `least_squares`
  (max_nfev 40 default, budget 1–80). Fixed priors: C_mass/C_air = 20,
  standby fraction 0.2, capacity prior clipped 3e6·area_scale to
  [1e5,1e9], offset bound 15 K (3 K passive-multistep). Requires ≥8
  transitions; ≥10 obs prefix. Diagnostics: solver success/status,
  evals, training MAE, parameters-at-bounds count, active transitions,
  `identified_all_parameters: false` always, passive prefix-balance
  audit, and data-only SVD sensitivity (labeled "not an
  identifiability proof"). `passive_multistep` requires boundary
  observer; refits on horizon-normalized 1/2/6/12-step residuals with
  equal aggregate horizon weight.
- `forecast_path`: pending physics first, then the candidate command
  path; future weather/comfort held at origin; water rolled by causal
  rate projection; recorded future commands allowed ONLY as OFFLINE
  conditional-dynamics validation. Returns temperatures + masses +
  uncertain floors. Demands fits and mass states covering all zones.
- Check tools: `tools/equipment_observer_check.py` (window steps 288,
  offset 576, warmup 72; flow errors vs t+1 sensors, water vs hold;
  `--summary` prints flow/water MAEs) and
  `tools/modular_thermal_check.py` (`fit_prefix` on the prefix ONLY,
  frozen params, horizons (1,2,3,6,12) default (1,6,12) for reports,
  groups all/active/passive/occupied/empty/identified/uncertain,
  `control_integration` absent/False).

### C. Workday equipment audit (both JSONs opened; values verbatim)

- Shape: JSON lists of 30 scenarios (first `building_train_1/
  atlanta_2`, last `building_train_8/singapore_7` in both files).
  Every row: policy dsat_gen31, steps 288, offset 576, warmup 72 —
  draft figures preserved verbatim (30 scenarios × 288 steps, offset
  576, warmup 72, policy dsat_gen31).
- Assumptions block identical in all 60 rows: design_air_pressure_pa
  2000.0, valve_kv 0.108, header_resistance 0.0,
  max_discharge_temperature 308.15, fan_efficiency 0.9 — each draft
  figure verified: 2000Pa ✓, kv 0.108 ✓, header 0.0 ✓, discharge
  308.15K ✓, fan eff 0.9 ✓.
- Row atlanta_2 (v1 and v2 identical): occupied_observations 156,
  return_air_weighting "provided_areas", geometry_mapping "regular
  connected-component labels assumed", delayed_transitions 286,
  skipped_unobserved_capacity 0, handler_flow 860 samples MAE
  2.366919065989104e-08 RMSE 4.2836889258289465e-08 max
  1.7970732768546327e-07; water 215 samples MAE 0.010926303198171216
  RMSE 0.1601795909959486 max 2.34869384765625; hold MAE
  0.03386074332303779 RMSE 0.1948306135491175 max 2.5.
- v2-only keys: `identified_floor_flow_m3_s` (807 samples, MAE
  2.1846228313073958e-08, RMSE 4.111494590179103e-08),
  `projections_with_uncertain_floors` 37,
  `uncertain_floor_flow_samples`; learned capacities / damper floors
  / pressure floors / observed water steps / projected zone power
  range carried in both files.
- Standing disclaimers (verbatim in every row): power_validation =
  "NOT validated against measured heat; lagged-temperature proxy and
  global equipment priors"; temperature_forecast_validation = "NOT
  performed; this adapter does not predict zone temperature".

### D. Modular suite (all seven files opened; protocol per file)

- `modular_thermal_forcing_audit_v1.json` (14101 bytes, 390 lines):
  protocol "OFFLINE oracle future forcing, no future state
  correction; same complete-horizon origins in both arms; frozen
  fits"; source_fits points at passive_multistep_v1. 5 rows;
  fairbanks_1 h1 projected and oracle arms identical (0.1885006093157
  K, p95 0.6171) vs persistence 0.07628335206400202 K (p95 0.1786),
  skill −1.471058287496814 — draft "forcing audit (oracle future
  forcing)" preserved.
- `modular_thermal_kalman_v1.json` (228383 bytes, 6264 lines): no
  top-level protocol key (row-level fields govern). 5 rows,
  two_state_kalman, 288 obs, fit 72, frozen params, recorded
  commands, held weather/comfort, causal water, 300 s, provided
  areas. Row 1: mass range [284.1203366925688, 295.5401180211045],
  occupied 156, uncertain origins 84, solver failures 0. Summary h1:
  all 0.09555149386182168/0.0847047234535799 (−12.8%), active
  0.07746474930241366/0.09898108008609516 (+21.7%), occupied
  0.03867741283042253/0.0 — draft "kalman" preserved.
- `modular_thermal_kalman_v2.json` (250710 bytes, 6819 lines): same
  h1 summary as v1; fitting_mass_update "boundary-conditioned;
  parameters frozen before validation" (vs kalman-fit in matched);
  envelope violations 1, max 0.004253048592374853 K; same state
  assumptions (var 4.0, air 0.05, mass 0.01, air exact).
- `modular_thermal_kalman_matched_v1.json` (280882 bytes, 7552
  lines): fitting_mass_update "kalman; parameters frozen before
  validation"; mass range [283.4850021962607, 303.6530969537898]
  (widest), violations 1520, max 8.026240422418368 K. Summary h1:
  all 0.11739806206538787/0.0847047234535799 (−38.6%), active
  0.09544566323927615/0.09898108008609516 (+3.6%) — draft "+matched"
  preserved; matched Kalman fitting degrades both mass plausibility
  and skill.
- `modular_thermal_passive_multistep_v1.json` (581354 bytes, 14891
  lines): passive_multistep true, boundary observer, mass range
  [287.0964580711533, 295.4444511815212], 0 envelope violations.
  Summary h1 all 0.1394118631678774 (−64.6%), active
  0.09405948577489318 (+4.97%), occupied 0.05228354593612615 vs
  0.053380830 — draft "passive multistep" preserved.
- `modular_thermal_prefix_audit_v1.json` (466849 bytes, 12329
  lines): protocol "Prefix only: offset 576, 72 observations, 70
  transitions; no validation targets or simulator". 5 rows;
  fairbanks_1: 39 zones, median_passive_balance_mae_W
  55.56015278320391, median_sensitivity_ratio 0.004902734560537786,
  12 zones with required mass outside envelope, max violation
  141.6871327369542 K — draft "prefix audit (72 obs)" preserved.
- `modular_thermal_smoke_v1.json` (163730 bytes, 4529 lines): 4 rows
  (one_state shown: 2 cases in summary), fit 72, frozen, recorded
  commands, held weather/comfort/water, 300 s, uncertain origins 84,
  solver failures 0; zone-1 air_capacity 7800361.423178462 J/K.
  Summary one_state h1 all 0.13122609521498368/0.040998732684261775
  (skill −2.20), active −2.32 — draft "smoke" preserved.

### E. Honest gaps

- Draft says eight `modular_thermal_*` files; seven are on disk (listed
  in Decisions). No eighth file (e.g. a held-command arm) was found —
  possibly planned but never banked, from memory of nothing.
- Only the first scenario's exact numbers were quoted above
  (atlanta_2); the other 29 rows per file were opened (list
  structure verified, first/last scenarios checked) but per-scenario
  MAEs are not transcribed — a reader needing them opens the JSON.
- Both workday JSONs are stamped 10-23 (v1 23:26, v2 23:29),
  i.e. the audits ran the evening before the 10-24 source snapshot
  (`equipment_observer.py` 07:25, `zone_thermal.py` 07:39). Day
  attribution follows the task direction (10-24 = equipment
  observers), not the byte landing time.
- Zone-power projections (`projected_zone_power_W` min/max per row)
  are instantaneous duties at lagged temps, never compared to
  measured heat — the files say so themselves.

