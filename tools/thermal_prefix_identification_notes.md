# Prefix-only identification audit: useful correction is not physical identification

Follow-up: `tools/thermal_family_notes.md` retains a finite prefix-compatible
parameter/state family with frozen weights. It improves some perturbed means
but fails noise/out-of-prior coverage and no-regression gates. Not promoted.
Expanded combined suite: 154 tests.

Tool: `tools/thermal_prefix_identification_check.py`.
Latest report: `smart_buildings/results/thermal_prefix_identification_v2.json`.
V1 is retained. V2 adds initial-state nuisance profiling after V1 exposed
nominal-model regression. This is exploratory method development on the same
cases, not a new untouched validation set or a deployable calibration policy.

## Information boundary and fixed protocol

Same 164-state known nominal synthetic geometry/basis, 13-state observer.
Six selected stress scenarios (nominal, capacity +20%, exterior -20%, airflow
-20%, late hidden heat, sensor noise) cross mixed and late-air-pulse inputs.
Each response has **288 physical steps / 24 hours**, no extra warm-up.

The fitter receives **only the first 72 observations and 72 commands**. No
truth labels, hidden fields, validation arrays or later disturbance schedule
are arguments. True coefficients appear only in report diagnostics, after
generation/fitting. Parameters are then frozen; latent state is replayed and
updated causally using each candidate's fixed coefficients. Both baseline and
candidate evaluation use the same noisy data, clean targets and forecast origins.

Future forcing remains conditional nominal input, not online weather/load
prediction. This is synthetic inference with a known nominal material basis,
not participant-only real-building identification. Convection is absent.

## Candidate parameters and loss

Coordinates: log global capacity scale, log exterior conductance scale, log
airflow gain, and two constant load biases measured in units of 5 W. Scale
bounds are 0.5..2; load bounds +/-10 W. Biases describe the observed prefix,
not future occupancy. Nominal basis stays fixed; no true-plant basis update.

Data residuals are h2/h6/h12 conditional prediction errors divided by horizon,
at prefix origins 12,18,...54: **48 scalar residuals**. The initial 12 samples
are assimilated inside the supplied prefix, not extra hidden steps. Maximum
target index is 66. A fixed .001 K prior regularizer favors nominal values.
The observer's Q/R assumptions are unchanged.

Necessary excitation checks require a 0.05 K temperature range, outdoor
gradient >=1 K or nominal thermal driver >=0.1 W; airflow additionally needs
nonzero flow. Finite-difference data sensitivities require RMS >=.001 K per
coordinate. Column-normalized singular ratios require >=.001. Solver limit is
40 SciPy `nfev`; actual residual calls including numerical Jacobians and
sensitivity diagnostics are separately reported. In V2 mixed cases take
83..91 calls, not 40 total forward computations. No convergence budget was
exhausted. Fits at bounds or with <10% prefix-MAE improvement are rejected.

## Initial-state nuisance profiling

A parameter-only Jacobian ignores unknown wall/slab history. V2 adds 11
initial modal offsets invisible to the two reset sensors (H delta=0), scaled
by the same 2 K physical-cell prior. It measures their prefix-residual
sensitivities and removes their local span before applying the parameter
sensitivity/rank checks. Eight nuisance directions survive the declared SVD
cutoff on these mixed prefixes.

This is a conservative **unconstrained local** initial-offset profile: the
2 K prior sets coordinates, not a hard bound. It is not global profile
likelihood, joint state/parameter optimization, or a proof of identifiability.
Other frozen but uncertain parameters may still confound retained parameters.

## What the checks correctly reject

- In every late-air-pulse case, there is no informative prefix forcing or
  temperature variation before step 96. **All parameter candidates remain
  unidentified and nominal**, including airflow. Future pulse observations
  are not used to fill that gap.
- Load-bias sensitivities fail the necessary strength/profile checks. No
  future hidden 5 W load is inferred before it starts.
- Mixed nominal and mixed late-hidden-load cases have identical prefixes and
  therefore exactly identical fits. Tests enforce this.
- The noisy mixed candidate fails the required prefix improvement and is not
  applied. No Q/R sweep is performed.

After state profiling, mixed-case exterior and load directions are unsupported.
Only capacity/airflow remain locally supported conditional on other frozen
nominal coefficients. These are **candidate corrections**, not verified
physical material estimates.

## V2 held-after-prefix prediction result

Mixed-case h12/60-minute MAE (K):

| Truth scenario | Frozen baseline | Prefix candidate |
| --- | ---: | ---: |
| nominal | **0.000908** | 0.006115 |
| capacity x1.2 | 0.073782 | **0.006961** |
| exterior x0.8 | 0.079355 | **0.002605** |
| airflow x0.8 | 0.012246 | **0.006263** |
| hidden heat after prefix | **0.007186** | 0.010360 |
| sensor noise | 0.021476 | 0.021476 (rejected fit) |

Late-pulse candidates are rejected for no prefix excitation and exactly
reproduce baseline errors. There is no claim that the candidate passes a
no-regression forecast gate: it fails that gate on nominal and hidden-load
mixed cases.

The local rank gate is not enough. In the nominal mixed case capacity moves
to **1.0195** and airflow to **1.0031**; prefix normalized MAE falls from
0.0003075 to 0.0000667, yet later forecast degrades about 6.7x at h12.
In the exterior-x0.8 case, exterior correctly remains unsupported, but the
candidate assigns capacity **1.2140** (truth 1.0) and airflow **1.0275**
(truth 1.0), while exterior stays at the wrong nominal 1.0. Forecast improves,
but the causal attribution to physical parameters is wrong. A correction to
one coefficient can compensate another unidentified coefficient even after
initial-state profiling. V1's nominal regression and false exterior correction
were not solved by merely adding another local sensitivity gate.

## Decision

**Do not promote this point-fitting rule to the controller.** Prefix-only
isolation and explicit unidentified flags work, but unique physical parameter
recovery and no-regression prediction do not. Hidden-state history, reduction
error and frozen unidentified coefficients remain coupled. Small training
residual, full parameter-only rank or better validation in selected mismatch
cases must not be labeled successful physical identification.

Next gate should retain coupled parameter/state uncertainty (e.g. profile
families and forecast ranges), rather than treating a single locally supported
correction as truth. If parameter fitting continues, nuisance parameters and
initial history must be handled jointly, and any selection/regularization gate
must be based on prefix data with a separate untouched evaluation. No future
loads or true material values may enter a participant policy. Champion and
simulator unchanged; no policy score or leaderboard improvement established.

## Verification

Seven new tests cover late-flow rejection, identical fits before a future
hidden load, mutation of future observations/commands, candidate nonmutation
and units, prefix target alignment, nonincreasing sensitivity after state
profiling, and budget/prefix rejection before fixture construction.
Combined suite: **148 tests pass**.

```sh
./.venv/bin/python -B tools/thermal_prefix_identification_check.py --steps 288 --prefix 72
./.venv/bin/python -B -m unittest tools.test_thermal_prefix_identification_check tools.test_thermal_mismatch_check tools.test_thermal_state_observer_check tools.test_thermal_system_modes_check tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
