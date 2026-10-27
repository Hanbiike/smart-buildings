# 2026-10-20 — Simulator and reward teardown (no submissions)

## Summary

No submissions went out from this account on 2026-10-20.
The day was a full read-through of the simulator and the reward stack, done so
that every later policy decision would rest on the exact formulas, constants,
and action/observation limits in the code rather than on guesses.
The headline result is the asymmetry the reward creates: comfort moves the
score roughly two orders of magnitude faster per kelvin than energy does, so
occupant gating dominates every other lever.
The reward weights are 0.6 comfort against 0.2 energy plus 0.2 carbon, with
productivity linear for the first 5 K outside the band and unbounded quadratic
after that.
Energy and carbon are normalized by fixed per-AHU max rates, never clipped, so
marginal electricity is never free and the gradient never dies at high draw.
The plant audit found a unity-efficiency boiler, a COP 3.5 chiller, an
ambient-only ASHP curve, a cooling-only air handler, and hard actuator floors
that bound every policy: recirculation at most 0.70, dampers at least 10%, VAV
discharge at most 308.15 K.

## Status

| Whose | Score | State |
|---|---|---|
| this account | — | no submissions sent |
| others (`transfer`, organizer PID port) | 0.06439 | public LB, from docstring |
| others (`k10` zone-PI probe) | 0.01853 | public LB, from docstring |

The two scores are other accounts' public-leaderboard numbers quoted in
`smart_buildings/baselines/zone_pi.py` docstrings (lines 46 and 135).
They are carried here as reference points only; nothing was submitted or run
locally on this date.

## Done

- Read the regret reward end to end — file
  `smart_buildings/smart_control/reward/setpoint_energy_carbon_regret.py` —
  why: it is the scoring contract, and its normalization choices (fixed
  denominators, no clipping) decide whether energy-saving tricks have any
  gradient at all.
- Read the electricity cost and carbon tables — file
  `smart_buildings/smart_control/reward/electricity_energy_cost.py` — why: the
  full 24-value weekday/weekend TOU schedules and the 24-value carbon curve
  are needed to price time-of-use arbitrage against thermal-mass losses.
- Read the gas cost model — file
  `smart_buildings/smart_control/reward/natural_gas_energy_cost.py` — why: the
  12-value monthly gas table in $/1000 ft3 plus unit conversions sets the
  heating side of every winter tradeoff.
- Read the productivity and rate-summation base — file
  `smart_buildings/smart_control/reward/base_setpoint_energy_carbon_reward.py`
  — why: it holds the linear-then-quadratic shortfall formula and the exact
  list of which plant terms count as electricity versus gas.
- Read the energy-scale helper — file
  `smart_buildings/smart_control/reward/energy_scale.py` — why: it proves the
  max rates are per-AHU constants (80000 W electric, 205000 W gas) and that
  identical draw would otherwise score differently per building size.
- Read one shipped scenario config — file
  `smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin`
  (lines 294-312) — why: it pins the concrete numbers (300.0, 0.0, 5.0,
  320000.0, 820000.0, 0.6/0.2/0.2) that the gin actually binds.
- Audited the plant files — files
  `smart_buildings/smart_control/simulator/boiler.py`,
  `smart_buildings/smart_control/simulator/air_handler.py`,
  `smart_buildings/smart_control/simulator/air_source_heat_pump.py`,
  `smart_buildings/smart_control/simulator/vav.py` — why: efficiency and limit
  constants (boiler unity, chiller COP 3.5, ASHP ambient curve, recirc 0.70,
  damper floor 0.10, discharge cap 308.15 K) bound every feasible policy.
- Read the schedule and scenario plumbing — files
  `smart_buildings/smart_control/simulator/setpoint_schedule.py` and
  `smart_buildings/smart_control/utils/scenario_generator.py` (lines 342-375)
  — why: comfort/eco windows, morning/evening hours, holidays, and time_zone
  are the recovery hooks every occupancy-gated policy depends on.
- Read the early geometry attempt — file
  `smart_buildings/baselines/geometry_aware_baseline.py` — why: to record what
  it tried (area-weighted zone control from the zone map) and that it went no
  further, so nobody re-attempts it blind.
- Derived the comfort and energy slopes per kelvin — why: the 0.18/K versus
  ~0.002/K asymmetry is the quantitative case for gating as the main lever,
  and the derivation exposed the min-productivity discrepancy documented in
  the Appendix.

## Measured

- `transfer` — public LB, other accounts — 0.06439 — organizer PID port
  baseline, quoted in `zone_pi.py:46`, no local run.
- `k10` (`ZonePIPolicy` defaults) — public LB, other accounts — 0.01853 —
  per-zone PI probe, quoted in `zone_pi.py:135`, no local run.
- reward-config audit — code coverage, all shipped scenario gins —
  min_productivity 0.0 everywhere — differs from the unit test's 100.0, see
  Appendix.
- energy-scale audit — code coverage, `energy_scale.py:43-44` — 80000 W/AHU
  electric, 205000 W/AHU gas — flat values would skew 2.33x across buildings
  per `scenario_generator.py:362-366`.
- TOU-arbitrage check — arithmetic on the copied tables — peak/off-peak ratio
  20/16 = 1.25x — carried from the earlier draft; closed 10-27 as not covering
  thermal-mass losses (from memory, run artifacts not re-opened).
- comfort-slope check — derivation from the opened formula — 0.18/K under the
  test config, 0.12/K under shipped gins — closed 10-27 via SetpointProbe plus
  occ5 (from memory, probe code not re-opened).

## Decisions

- Claim: occupant gating is the main structural lever. Proof: more than 60%
  of steps are unoccupied while energy is paid every step the plant runs
  (occupancy schedules in the simulator scenarios). Consequence: the occ-gated
  policy line was built on 10-26.
- Claim: only down-shift zone targets, never up-shift. Proof: the upward shift
  regressed hot tight-band buildings (singapore +25%, quoted in
  `DsatShiftDownPolicy` docstring, `zone_pi.py:413-417`). Consequence: fixed
  in the gen-3.1 stack on 10-21.
- Claim: energy denominators must stay unclipped. Proof: measured draw runs 2x
  to 10x the 160 kW-scale setting, so clipping pins the term constant and
  kills its gradient (`setpoint_energy_carbon_regret.py:223-228`).
  Consequence: keep the fixed-max-rate normalization and never re-add a clip.
- Claim: the geometry-aware baseline goes no further. Proof: it is an early
  self-contained attempt with no measured score attached anywhere in the repo.
  Consequence: recorded here with its tunables so the idea (area-weighted
  small-zone authority) survives without a rewrite.

## Open

- Verify the comfort-slope asymmetry empirically (done, closed 10-27 via
  SetpointProbe and occ5; carried from the earlier draft, probe artifacts not
  re-opened this session).
- Verify TOU arbitrage cannot cover thermal-mass losses (done, closed 10-27
  via TodShiftProbe plus arithmetic; carried from the earlier draft).
- Build the occupant-gated policy line (moved to 10-21 and 10-26 work).
- Start supervised behavior cloning, then RL (moved to 10-21 night scaffold).

## Appendix

### A. Reward weights and productivity constants (exact)

From `fairbanks_1/config.gin:294-301` and
`scenario_generator.py:356-374` (identical values):

| Constant | Value |
|---|---|
| `max_productivity_personhour_usd` | 300.0 |
| `min_productivity_personhour_usd` | 0.0 |
| `productivity_deviation_ref` (w) | 5.0 K |
| `max_electricity_rate` (fairbanks_1, K=4) | 320000.0 W = 80000 x 4 |
| `max_natural_gas_rate` (fairbanks_1, K=4) | 820000.0 W = 205000 x 4 |
| `productivity_weight` | 0.6 |
| `energy_cost_weight` | 0.2 |
| `carbon_emission_weight` | 0.2 |

Discrepancy, kept verbatim from both sides: the reward unit test
(`setpoint_energy_carbon_regret_test.py:37-38`) constructs the function with
`max_productivity_personhour_usd=300.0, min_productivity_personhour_usd=100.0`,
while every shipped scenario gin audited (all `training_scenarios` and
`evaluation_scenarios` configs) binds `min_productivity_personhour_usd = 0.0`.
The earlier draft's comfort math ($300/$100, -0.15 normalized at 0.5 K) follows
the test config; the shipped-config math is derived in section E below. Both
variants are preserved here so the numbers stay checkable.

### B. Electricity TOU tables, full 24 values (exact copy)

From `electricity_energy_cost.py:50-101`, units cents/kWh, index = hour 0-23.

Weekday (`WEEKDAY_PRICE_BY_HOUR`):

| h | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| c | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 18.0 | 18.0 |

| h | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| c | 18.0 | 18.0 | 18.0 | 18.0 | 20.0 | 20.0 | 20.0 | 20.0 |

| h | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|
| c | 20.0 | 20.0 | 20.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 |

Shorthand kept from the earlier draft: 16x6 (h0-5), 18x6 (h6-11), 20x7
(h12-18), 16x5 (h19-23); 6+6+7+5 = 24 entries, verified against the tuple.
Weekend (`WEEKEND_PRICE_BY_HOUR`): flat 16.0 for all 24 hours.
Unit handling in code: prices convert cents/kWh to USD/W/s by dividing by
100, 1000, and 3600 (`electricity_energy_cost.py:128-134`); cost multiplies
price by |energy_rate| by seconds (`:161`), so heating and cooling both bill
by magnitude. Queries over one hour log a warning but still price at the
start hour (`:150-155`).

### C. Carbon table, full 24 values (exact copy)

From `electricity_energy_cost.py:17-42`, units kg carbon/MWh, index hour 0-23:

| h | rate | h | rate |
|---|---|---|---|
| 0 | 88.19666493 | 12 | 102.5919168 |
| 1 | 87.79190866 | 13 | 103.4403736 |
| 2 | 87.87607686 | 14 | 104.1380294 |
| 3 | 87.83054163 | 15 | 104.7359292 |
| 4 | 88.00279618 | 16 | 102.0714466 |
| 5 | 88.19648183 | 17 | 97.04226176 |
| 6 | 89.70663283 | 18 | 93.57895651 |
| 7 | 93.97947901 | 19 | 92.46355045 |
| 8 | 98.85868291 | 20 | 91.72914657 |
| 9 | 100.7853521 | 21 | 90.69209747 |
| 10 | 101.3866866 | 22 | 89.76552213 |
| 11 | 101.7795612 | 23 | 88.99950995 |

Night low 87.79 (h1), peak 104.7359292 at h15, matching the earlier draft's
"88.0 at night, peak 104.7 at 15h". Rates convert kg/MWh to kg/Ws by dividing by
1e6 and 3600 (`:123-126`); carbon bills |energy_rate| by seconds at the start
hour (`:186`), same magnitude convention as cost.

### D. Gas monthly table (exact copy)

From `natural_gas_energy_cost.py:15-28`, units $/1000 cubic feet, index month
Jan-Dec: 9.02, 8.35, 7.77, 7.26, 6.69, 6.86, 6.77, 6.76, 6.99, 7.19, 7.96,
8.98. June is the cheapest month (6.69), January the priciest (9.02).
Conversion: $/1000 ft3 to $/J via `KWH_PER_KFT3_GAS` and `JOULES_PER_KWH`
(`:46-50`); carbon uses a flat `GAS_CO2` rate per 1000 ft3 (`:52-57`), so gas
carbon has no time-of-day shape at all. Negative gas rates are clamped to zero
with a warning (`:76-80`), i.e. gas only ever heats.

### E. Comfort-slope derivation (both normalizations)

Productivity per zone-hour (`base_setpoint_energy_carbon_reward.py:106-118`):
deviation d <= w=5 K gives shortfall d/w; beyond w it is
1 + ((d-w)/w)^2, unbounded, so productivity goes negative past 5 K.
Inside the knee the branch is linear, not squared: at w = 5 K, 1 K outside
costs 20% of productivity where a squared form charged 4% (`:101-103`).
Normalized regret (`setpoint_energy_carbon_regret.py:216-221`):
(actual - min) / (max - min) - 1.0, or 0.0 when nobody is home.
Under the test config (max 300, min 100): 0.5 K shortfall gives productivity
300 x (1 - 0.5/5) = 270 $/person-h; normalized (270-100)/200 - 1 = -0.15;
times weight 0.6 divided by weight sum 1.0 gives -0.09 loss per step, i.e.
slope 0.18/K. This is the earlier draft's number, preserved verbatim.
Under the shipped gins (max 300, min 0.0): the same 0.5 K gives 270/300 - 1 =
-0.10 normalized, x 0.6 = -0.06 loss per step, i.e. slope 0.12/K.
Either way the comfort slope is tens of times the energy slope below, which
is the quantitative case for gating. Empty buildings contribute exactly 0.0
comfort regret, so unoccupied steps are pure energy cost.

### F. Energy-slope structure and the ~0.002/K figure

Normalized energy cost (`setpoint_energy_carbon_regret.py:294-302`) is
combined actual cost over combined max-rate cost; the gradient per watt is
therefore 1/max_rate by construction, and likewise for carbon
(`:304-313`). Raw reward is the weighted sum over the weight total
(`:333-344`), and r_i targets [-1, 0] with excursions below -1 allowed when
draw exceeds the maxima (`:36-40`). The ~0.002/K energy-slope figure from
the earlier draft is carried as stated: its per-K step needs a plant
watts-per-K factor that was not re-derived from opened files this session,
so it stays labeled as the draft's number, not a fresh derivation.

### G. Plant findings per file

- `boiler.py`: no efficiency attribute anywhere in the constructor or the gas
  math; gas burn is the floored sum of flow, dissipation, and tank terms
  (`boiler.py:241-252`), i.e. effectively constant efficiency 1.0. Tank
  geometry defaults: convection 5.6, length 2.0 m, radius 0.5 m, capacity 1.5
  m3, insulation 0.067 W/m/K over 0.06 m.
- `air_handler.py`: chiller COP default 3.5, "mid range for a commercial
  chiller" (`:63-67`); recirculation cap 0.70 (`:60-62`); minimum static
  pressure 200 Pa, about 30% design flow for the ventilation minimum
  (`:55-59`); "The handler only ever cools, so there is no heating branch"
  (`:354-355`).
- `air_source_heat_pump.py`: ambient-only efficiency, nominal COP 3.2 rated
  at 7 C outside air, linear degradation 0.05 per C, clamped to [1.0, 5.0]
  (`:86, :164-169`); electrical power = thermal / COP (`:203-205`).
- `vav.py`: damper floor 0.10, without which the cheapest control is to seal
  zones shut (`:61-65`); discharge cap 308.15 K = 35 C, the reheat-box design
  limit, binding hardest at the damper floor where coil effectiveness
  saturates (`:66-79`); design static 2000 Pa (`:58`); default band seeds
  294.0/297.0 K (`:59-60`).
- Electricity summation (`base_setpoint_energy_carbon_reward.py:132-166`):
  blower plus |AC| per AHU, boiler pumps, and |heat-pump| pump plus heating
  terms. Gas summation (`:168-179`): boiler gas rates only.

### H. Action and observation inventory

Continuous actions are 3K + 2 + 2V for K air handlers and V VAV boxes
(`energy_scale.py:52-53`); discrete actions are K AHU run bits plus one water
loop bit (`geometry_aware_baseline.py:199` per-AHU run index, `:215` water
index at n_discrete - 1). Observations carry per-zone temperatures and
setpoints, AHU outside-air/supply-air sensors, pressure and flow fields, run
bits, plus exactly one building-level `num_occupants` field — there is no
per-zone occupancy (`tools/nn_features.py:95-98`, which also confirms the
global `comfort_mode_now/soon` flags, `hod_`/`dow_` harmonics, and `hws_`
hot-water-supply fields). Missing from obs (from memory, not re-verified in
code): per-zone occupancy, weather forecasts, humidity. Recovery hooks that
do exist in opened files: `time_zone` (default US/Pacific,
`scenario_generator.py:352`, threaded into gin at `:949, :956, :1056`);
holidays forced to eco windows (`setpoint_schedule.py:17-20, :32, :67-70`);
comfort window default (294.0, 297.0) and eco window (289.0, 298.0) with
morning hour 6 and evening hour 19 (`scenario_generator.py:346-352`, mirrored
as policy fallbacks in `geometry_aware_baseline.py:139-148`). Policy runtime
limit of 10x bang-bang comes from the competition paper and was not
re-verified in opened code (from memory).

### I. Geometry-aware baseline teardown

`geometry_aware_baseline.py` (547 lines) is a self-contained deterministic
policy using both the flat obs vector and the static metadata geometry
(`zone_map` cell counts, `ahu_zone_assignments`, action index tables).
Control shape follows the shipped PID skeleton: every AHU idles (fan off,
minimum pressure) while all served zones sit inside the active comfort/eco
band, else a PI loop (kp 0.5, ki 0.02, kd 0.1, integral cap 10.0, activation
threshold 0.02 K) drives supply temperature with fan pressure tracking
effort; zone dampers open proportionally (full at 2.0 K violation) with
reheat as a hot/cold switch; the hot-water loop is binary hot/cold with the
pump tracking activity (`:476-511`). Geometry additions: per-room areas from
zone-map labels, small zones (under 0.6x median area) get a 0.5 K pre-trigger
margin and up to 2.5x damper gain via sqrt(median/area) (`:85-89, :227-244`),
and AHU aggregates use area-weighted means (`:435-442`). Time comes from
hod/dow cyclic features with work-hour gating on the comfort window
(`:369-381`). Registered as `geometry_zonal` in POLICY_MAP. No score is
attached to it anywhere; it went no further, and this record is its
obituary.
