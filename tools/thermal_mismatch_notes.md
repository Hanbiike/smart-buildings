# Frozen-observer parameter, forcing and sensor stress audit

Follow-up: `tools/thermal_prefix_identification_notes.md` tests prefix-only
scalar correction with excitation and initial-state nuisance checks. It improves
some perturbed cases but regresses nominal physics and misattributes parameter
errors, so the point-fitting rule is not promoted. Combined suite: 148 tests.

Tool: `tools/thermal_mismatch_check.py`.
Latest report: `smart_buildings/results/thermal_mismatch_v2.json`.
V1 retained; V2 adds diagnostic truth/reconstructed physical-temperature ranges.

## Fixed protocol and separation

Same known synthetic geometry: 164-state plant, with nominal 13/21/29-state
observers constructed **before** any perturbed data. No fitting, basis changes,
noise adaptation, simulator environment, real-building data or policy runs.
Each of 36 plant/input responses is limited to **288 steps / 24 hours**;
72 initial observations are inside that budget. Three model sizes are evaluated
on each response, for 108 evaluation rows, not 108 independent buildings.
Convection is absent. Forecasts receive nominal future command/forcing paths,
explicitly conditional, not future realized heat or hidden plant parameters.

Twelve predefined scenarios cross three inputs (mixed, late fixed pulse and
late air pulse):

- nominal;
- all air/wall/slab capacities scaled 0.8 / 1.2;
- conduction links touching walls scaled 0.8 / 1.2;
- exterior boundary conductance scaled 0.8 / 1.2;
- realized airflow conductance scaled 0.8 / 1.2;
- hidden room-1 heat **5 W** during steps **96..179**;
- sensor noise std **0.05 K**, deterministic seed 202610;
- combined: C x1.2, wall links x0.8, exterior x1.2, airflow x0.8, hidden heat
  and sensor noise as above.

These are positive passive-network perturbations, not claimed to be the result
of changing a single GIN material constant. Wall conductance and exterior film
terms are deliberately independent. The fixture room is only 1.2 m2, so the
5 W hidden load is a synthetic disturbance, not a real occupant model.

Observer coefficients and priors remain nominal. Assumed sensor std stays
**0.0001 K** and physical-cell process std stays **0.01 K/step**. In the sensor
stress case this deliberately understates actual noise std by 500x. Current
source simulator has no such injected measurement noise; this is robustness
stress, not a discovered property of its sensor stream.

The observer receives only noisy latched temperatures and nominal commands.
Targets are the **noiseless physical latched zone means**. Reset duplicate
reuses the same noisy reading. Generation can record true temperature ranges,
but those diagnostics are attached **after** estimation and never passed to
the filter or used for clipping. Future clean targets are used only to score.

All h1/h2/h6/h12 methods share complete-horizon origins. Pulse origins 84..119
and hidden-load origins 84..191 are also reported to avoid day-average dilution.
The delay remains: h1 is pending old-command response; new command starts h2.

## Forecast results

All three model sizes beat noisy-temperature persistence in all **36/36**
cases at h2/h6/h12. For 13 states the worst model/persistence MAE ratios are
**0.643 / 0.535 / 0.431**, all in combined-error late-fixed-pulse response.
This is a finite stress grid, not a uniform robustness bound. In noisy cases
persistence is itself degraded by sensor noise; beating it is a weak gate.

13-state mixed-input conditional MAE, K:

| Perturbation | h2 / 10 min | h12 / 60 min |
| --- | ---: | ---: |
| nominal | 0.000191 | 0.000908 |
| capacity x0.8 | 0.014478 | 0.069230 |
| capacity x1.2 | 0.015406 | 0.073782 |
| wall links x0.8 | 0.005872 | 0.023969 |
| exterior x0.8 | 0.016583 | 0.079355 |
| airflow x0.8 | 0.003087 | 0.012246 |
| hidden heat | 0.001813 | 0.007186 |
| sensor noise | 0.025459 | 0.021476 |
| combined | 0.026067 | 0.035537 |

Parameter mismatch substantially increases errors despite beating persistence.
The huge nominal-physics gains must not be transferred to unknown buildings.

## Two failures hidden by the persistence comparison

1. **Correction can hurt the stronger open-loop baseline.** For each model
   size, h12 Kalman MAE is worse than open loop in 6/36 cases: both late pulses
   under wall-link +/-20% and sensor noise. For 13 states in noisy late fixed
   pulse, h12 MAEs are **0.021958 K Kalman vs 0.005612 K open loop**, although
   persistence is worse at 0.059091 K. Thus 36/36 persistence wins do not certify
   a robust observer.
2. **Good output forecasts do not mean correct hidden temperatures.** Under
   noiseless exterior x0.8 with mixed inputs, the 13-state reconstructed field
   reaches **310.386 K** while the full true field never exceeds **297 K**.
   With sensor noise in late fixed pulse it falls to **270.986 K**, while true
   temperatures stay within **295.000..295.518 K**. Largest excursions outside
   each response's global true range are **24.01 K / 23.02 K / 17.28 K** for
   13/21/29 states. These are wrong-state diagnostics, not claims that a modal
   amplitude itself must obey a slab-temperature envelope.

True ranges cover the whole generated response, so using them here is an
explicit offline diagnostic, not a causal constraint or participant feature.
Even without sensor noise, reconstructed ranges can be wrong. The filter is
using weakly observed spatial modes to absorb coefficient/forcing errors.

## Decision

The predictor survives this modest parameter stress against persistence, but
**the observer is not validated as a physical hidden-state estimator or a
controller component**. Do not promote it or clip modal coordinates to hide
the failure. No new policy score or leaderboard gain is established.

Next gate: distinguish parameter bias, unknown load and noise from latent state.
Use prefix-only identifiable parameter/load components with explicit excitation
checks; leave parameters unchanged/unidentified when the prefix cannot identify
them. In particular, late-only airflow pulses cannot identify airflow gains
before they occur. Do not recover true perturbation labels or material values
from the generator, use future responses for fitting, or start a noise sweep
against these reported scores. Known nominal geometry/material basis is still
oracle information; real building identification and other geometries remain
separate unresolved gates.

## Verification

Seven new tests cover passive/symmetric perturbed networks, no nominal-array
mutation, command/realized-input separation, independent noisy observations
and clean targets, reset duplicate, truth diagnostics excluded from estimator,
target changes affecting scores only, frozen coefficients, nominal parity and
early budget/scale rejection. Combined suite: **141 tests pass**.
Simulator and champion unchanged.

```sh
./.venv/bin/python -B tools/thermal_mismatch_check.py --steps 288 --prefix 72
./.venv/bin/python -B -m unittest tools.test_thermal_mismatch_check tools.test_thermal_state_observer_check tools.test_thermal_system_modes_check tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
