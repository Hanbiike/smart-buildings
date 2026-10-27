# Action-conditioned grey-box diagnostic

Experimental module: `smart_buildings/baselines/causal_greybox.py`.
Not connected to any submission policy; `trusted()` always returns false.

## Corrected input interpretation

- Dumps follow `obs[t] -> action[t] -> obs[t+1]`.
- The AHU supply temperature sensor returns its **setpoint**, not actual
  discharge temperature. Actual discharge is the minimum of the setpoint and
  mixed outdoor/return air while running. There is no AHU heating-to-setpoint.
- Airflow depends on damper, square-root pressure and shared handler capacity.
- New model uses same-step proposed commands, bounded physical coefficients,
  a thermal-inertia regressor, and a capped reheat proxy.

Remaining approximations: return air is a plain zone mean rather than an
area-weighted mean; minimum pressure is estimated; water distribution and
future water temperature are not reconstructed exactly. This is not an exact
simulator or a validated counterfactual model.

## Reproduction (no simulator invocation)

```sh
./.venv/bin/python -B -m unittest tools.test_causal_greybox
./.venv/bin/python -B tools/greybox_check.py --model causal --steps 288
./.venv/bin/python -B tools/greybox_check.py --model sensors --steps 288
./.venv/bin/python -B tools/greybox_check.py --model causal --policy dsat_gen2 --steps 288
```

Each case uses the first 288 recorded observations. Fit starts after 144
transitions and is updated causally thereafter. Predictions hold current
commands/inputs, never the recorded future commands. This is online forecast
validation, not a frozen-parameter or unseen-building test.

Initial action-conditioned model, Gen-3.1 dump MAE skill relative to persistence:

| Scenario | 30 minutes | 60 minutes |
|---|---:|---:|
| building_train_1/fairbanks_1 | +23% | +29% |
| building_train_2/fairbanks_2 | -14% | -8% |
| building_train_8/fairbanks_8 | +6% | +21% |
| building_train_1/atlanta_2 | +18% | +11% |
| building_train_8/singapore_7 | -20% | -11% |

Persistence remains stronger in two cases. Linear drift (also reported by the
checker) is competitive or better at some shorter horizons. The 20% improvement
gate across regimes is **not met**. First-day windows may have no occupants;
these results say nothing about leaderboard improvement.

Next: separate passive and active errors, verify action acceptance/actuator
delays, and improve the water/air input map before any control integration.

## Follow-up: observation delay and plant calibration

The VAV zone-temperature reading is latched **before** the thermal solve in
`SimulatorFlexibleGeometries.execute_step_sim`. Consequently action[t] first
affects the zone sensor at obs[t+2]. Actuator readings at obs[t+1], by contrast,
already reflect action[t]. The `delayed` variant therefore uses the pending
previous command for its first forecast step and the proposed command thereafter.
This was traced in source, not inferred by trying future temperature labels.

Reproduce the follow-up:

```sh
./.venv/bin/python -B tools/greybox_audit.py --steps 288
./.venv/bin/python -B tools/greybox_check.py --model delayed --steps 288 --regimes
./.venv/bin/python -B tools/greybox_check.py --model calibrated --steps 288
```

The audit uses only past transitions to estimate actuator floors and water
rates; targets are consulted only for scoring subsequent predictions:

- Reheat commands match the next actuator observation exactly in fb2 and sg7.
- Damper mismatch disappears after learning the 0.1 ventilation floor.
- Both cases reveal a 200 Pa fan-pressure floor (the initial model assumed at
  most 100 Pa). The calibrated variant learns this, rather than hardcoding 200.
- In fb2 the water hold baseline MAE is 0.8442 K; the causal rate model is below
  0.000001 K over the last 143 transitions. Learned rates: +2.5/-0.5 K per step.
- In sg7 water MAE falls from 0.03279 to 0.01033 K. Heating rate remains poorly
  excited there; its final observed maximum is not a proven physical limit.

The `calibrated` variant learns these quantities online and rolls the water
temperature toward the candidate setpoint. It still does not model shared
water flow or detailed wall state.

| Scenario | Delayed 60-min skill | Calibrated 60-min skill |
|---|---:|---:|
| building_train_1/fairbanks_1 | +29% | +29% |
| building_train_2/fairbanks_2 | -8% | -17% |
| building_train_8/fairbanks_8 | +21% | +19% |
| building_train_1/atlanta_2 | +11% | +11% |
| building_train_8/singapore_7 | -10% | +6% |

Better water predictions do **not** imply better zone predictions: fb2 worsens.
The origin-regime split also reveals that passive-state forecasts drive much
of the aggregate improvement. For delayed fb1 at 30 min, active-origin MAE is
0.758 K vs persistence 0.623 K; passive-origin MAE is 0.093 vs 0.174 K. Regime
membership is based on observable AHU state at origin, not future labels.

### Oracle command-path diagnostic (not deployable forecast evidence)

`--oracle-actions` uses recorded future actions but **never future observations**.
It isolates part of the error caused by comparing a held-command forecast to a
closed-loop trajectory whose commands change. With calibrated inputs, fb2
60-min skill changes from -17% to -2%, sg7 from +6% to +12%. This is an offline
diagnostic only, not evidence of an achievable online score or control benefit.

No policy integration or simulator changes were made. The control trust gate
remains closed. Remaining priorities: active-regime thermal state/shared-water
flow and validation on independently excited command trajectories.

## Coil physics and slow-state ablations

`smart_buildings/baselines/thermal_greybox.py` adds two independent switches:

- `water`: coil power uses water Kv, square-root pressure, cubic-metres/hour
  conversion and the minimum of air/water capacity rates, plus discharge cap.
  Pump-off removes water flow. Kv=0.108 is a common public-training prior,
  never selected by scenario/building identity. It is not an identified unknown
  test-building parameter.
- `mass`: an exponential-filtered temperature proxy with a fixed two-hour time
  constant supplies one additional non-negative heat-exchange regressor. The
  proxy advances during rollout without reading future measurements; it is
  not an observed wall temperature or a fully identified 2R2C model.
- `thermal`: both changes together. Neither turns off the control trust gate.

All local training configs omit header resistance, whose default is zero.
For that case a zone's water flow is `Kv_zone * sqrt(dP) / 3600`: the aggregate
valve demand cancels when allocating total flow. A fixed shared-flow budget
would invent competition. Nonzero resistance is supported as a global prior
but is not assumed present in these dumps.

Five-case online MAE, held commands, 288 observations, 144-transition warm-up:

| Scenario | Calibrated h6/h12 | Water h6/h12 | Mass h6/h12 | Both h6/h12 |
|---|---:|---:|---:|---:|
| fb1 | .179/.321 | .174/.321 | .158/.321 | .153/.321 |
| fb2 | 1.022/1.215 | 1.149/1.410 | .926/1.095 | .993/1.157 |
| fb8 | .470/.646 | .492/.671 | .418/.665 | .414/.664 |
| atl2 | .024/.050 | .028/.050 | .025/.049 | .027/.049 |
| sg7 | .083/.145 | .083/.145 | .077/.157 | .077/.157 |

The coil calculation alone is not a win; mass improves several 30-minute
forecasts but does not reliably improve the one-hour forecast. For active
fb2 origins, even mass MAE is 1.790 K at 30 min vs persistence 1.155 K.

The warm-up excitation check explains an important limitation: in fb1, fb8
and atl2, **zero zones** have at least eight nonzero air/reheat-feature
transitions during the initial 144 transitions. In fb2 only 26/85 zones have
air excitation and 20/85 reheat excitation; sg7 has air excitation in 30/30
zones but reheat in 0/30. Good passive forecasts cannot validate active-control
gains that the prefix never exercised.

Reproduce:

```sh
./.venv/bin/python -B -m unittest tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check
./.venv/bin/python -B tools/greybox_check.py --model water --steps 288 --horizons 6 12 --regimes
./.venv/bin/python -B tools/greybox_check.py --model mass --steps 288 --horizons 6 12 --regimes
./.venv/bin/python -B tools/greybox_check.py --model thermal --steps 288 --horizons 6 12 --regimes
./.venv/bin/python -B tools/greybox_check.py --model mass --all-scenarios --steps 288 --horizons 6 12 --json
```

`--all-scenarios` discovers saved dumps, not simulator scenarios to execute.
`--json` exposes metrics, origin-regime splits and prefix excitation counts.
These runs remain forecasts of existing policy trajectories, not new policy
scores. Improving the warm-up prior without depending on building identity
will require pooled training with held-out-building validation, or deliberately
excited short trajectories; the current first-day online fit is insufficient.

### All 30 saved cases, not just the diagnostic five

Every case used its first 288 observations and the same online protocol. No
new simulation was run. Metrics below average per-case MAE equally; they are
temperature-prediction metrics, **not reward/leaderboard scores**.

| Model | Horizon | Mean MAE K | Persistence K | Relative reduction | Cases beating persistence |
|---|---:|---:|---:|---:|---:|
| calibrated | 30 min | .169343 | .172571 | 1.9% | 20/30 |
| mass | 30 min | .150146 | .172571 | 13.0% | 28/30 |
| calibrated | 60 min | .268172 | .304611 | 12.0% | 26/30 |
| mass | 60 min | .254556 | .304611 | 16.4% | 26/30 |

The active-origin subset remains unacceptable. Averaging conditional MAE over
cases that contain active origins, mass gives .401434 K vs persistence .325875 K
at 30 min, and .705703 vs .570766 K at 60 min: approximately **23% worse**.
The aggregate improvement must not be used to open the MPC gate.

### Critical first-day coverage limitation

All 30 saved Gen-3.1 dumps have **zero occupied observations among their first
288 observations**. Their first-day windows begin on Sunday around 06:00 and
end around the Monday opening time. Comparing saved rewards to their CSVs
confirms alignment exactly; this is not a stale-dump or column-order error.

The reward CSV has one occupied reward interval at the very end of 23 cases,
while the policy's preceding observation is still empty. Seven cases have no
occupied reward interval even then. Thus the earlier statement that only seven
windows were entirely empty must not suggest meaningful occupied coverage in
the other 23: those have only that single terminal interval.

Next validation must include an explicitly shifted workday start with a fresh
reset and **at most 288 actual simulator steps**, not an uncounted warm-up or a
longer rollout. The shift is an evaluation setting, not a date signal available
to the policy. Baselines and candidates must share the identical shifted reset.
Existing Sunday-screen thresholds cannot establish workday comfort quality.

### Existing workday-window check (no extra simulator steps)

Added `--offset` to replay a separate window from an **already saved** dump.
The model is constructed fresh and consumes no preceding observations. This is
not a cold-reset environment/policy score: the historical policy that produced
the archive still has its own earlier state. Unit tests compare a shifted
window with the same window stored alone to prevent accidental model warm-up.
Incomplete windows and offsets below zero are rejected.

Tuesday-window check: `--offset 576 --steps 288 --fit-window 72`, using the
same five cases. Six-hour identification leaves afternoon active observations
in the validation portion rather than validating almost entirely overnight.
This protocol change is explicit and shared by all compared models; its
numbers must not be mixed with the first-day/144-transition results.

| Scenario | Persistence h12 | Calibrated h12 | Mass h12 | Both h12 |
|---|---:|---:|---:|---:|
| fb1 | .685 | .734 | .731 | .703 |
| fb2 | .756 | .735 | .726 | .705 |
| fb8 | .665 | .598 | .555 | .528 |
| atl2 | .074 | .078 | .093 | .088 |
| sg7 | .248 | .338 | .270 | .271 |

Even the combined model is worse than persistence for **all five active-origin
subsets at 30 min**, and for four of five at 60 min. The apparently strong fb8
aggregate improvement is mainly passive-origin improvement, not evidence of
accurate active control. Prefix excitation now reaches all zones on the air
side, so missing first-day excitation is not the only problem.

```sh
./.venv/bin/python -B tools/greybox_check.py --model calibrated --offset 576 --steps 288 --fit-window 72 --horizons 6 12 --regimes
./.venv/bin/python -B tools/greybox_check.py --model mass --offset 576 --steps 288 --fit-window 72 --horizons 6 12 --regimes
./.venv/bin/python -B tools/greybox_check.py --model thermal --offset 576 --steps 288 --fit-window 72 --horizons 6 12 --regimes
```

Decision: do not connect these online-only variants to MPC or tune additional
wall/Kv constants on these same diagnostic cases. A next modeling experiment
needs a transferable pooled prior and genuinely held-out buildings, or
independent short excitation data, before any policy integration. All tested
variants retain `trusted() == False`; the champion controller is unchanged.

## Population prior: leave-one-building-out experiment

`smart_buildings/baselines/pooled_greybox.py` accepts one eight-coefficient
population prior, not a building/scenario identity. It uses the same features
and slow state as `mass`. A new offline harness,
`tools/pooled_greybox_check.py`, excludes **all five weather cases** of each test
building from population training. Fold manifests list the excluded building,
the actual training cases and the resulting coefficients for audit purposes;
they are not a deployment lookup table keyed by building identity.

Protocol fixed before viewing the new fold results:

- Thirty existing workday windows: offset 576, length 288 observations each.
  Feature extraction resets its state at each window; no extra simulation or
  preceding observations are used.
- Six held-out-building folds; each population prior uses 25 cases from the
  other five buildings. IDs only select folds and report labels.
- Eight shared coefficients, fixed feature scales, physical coefficient bounds
  and population ridge strength 0.01. Each training case has total weight one,
  independent of its number of zones or rows.
- Sufficient statistics reduce the bounded least-squares solve to 8 by 8.
  This is lightweight CPU regression, not neural-network training.
- Three matched test variants: existing online-only `mass`, `pooled_frozen`
  thermal coefficients, and `pooled_adapted` with local ridge strength 16.
  Frozen thermal coefficients still permit causal sensor/floor calibration.
- All variants absorb the same 72-transition test prefix and score the same
  origins thereafter. Adaptation may use only observations already seen.
  No recorded future commands are supplied.
- Report overall MAE plus separate active-origin and occupied-origin errors.
  First-day aggregate wins cannot substitute for workday comfort evidence.

No per-zone geometry area is fabricated: the saved `zone_map.npy` arrays use
material categories 0/1/2, not a map of room-number labels. This experiment
keeps the common observational features rather than incorrectly dividing
thermal gains by counts of those material labels.

Reproduction:

```sh
./.venv/bin/python -B -m unittest tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
./.venv/bin/python -B tools/pooled_greybox_check.py --steps 288 --offset 576 --fit-window 72 --horizons 6 12 --output smart_buildings/results/greybox_lobo_workday_v1.json
```

The harness refuses to overwrite an existing report. Use a new output filename
for a rerun. Reports contain forecast errors and exclusion manifests, not new
policy rewards or evidence that a submission beats the leaderboard. No pooled
model is registered in the controller/policy map; control trust stays closed.
