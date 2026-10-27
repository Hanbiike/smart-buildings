# Causal two-sensor state estimation with delayed observations

Follow-up: `tools/thermal_mismatch_notes.md` audits frozen nominal observers
against parameter/load/noise errors. Persistence wins survive, but hidden-state
compensation and open-loop regressions prevent promotion. Expanded suite: 141.

Tool: `tools/thermal_state_observer_check.py`.
Latest report: `smart_buildings/results/thermal_state_observer_v2.json`.
V1 is retained; V2 adds late pulses so command-response evaluation does not
consist only of recovery after pulses that ended before the scoring prefix.

## Scope and information boundary

Same known 164-state synthetic system, using frozen 13/21/29-state input/output
reductions. Each data case contains **288 physical transitions / 24 hours**,
plus the reset observation. No environment, scenario configuration, archive
fitting, controller or convection. Full hidden history is used **only** by the
synthetic generator, never to initialize or update the observer.

The observer receives two temperatures and the completed interval's inputs.
Its first prior is uniform at the mean of the two received temperatures, then
updated with both measurements. Open-loop and Kalman arms have identical
initialization; open loop disables subsequent corrections. Coefficients,
reduction bases and all noise assumptions remain frozen.

Known exact equipment/heat/weather inputs remain oracle information here.
Future paths supplied for forecasting are explicitly **conditional synthetic
inputs**, not online weather/load predictions or policy scores. This removes
oracle hidden-state initialization, not oracle physics/forcing.

## Sensor timing

`obs[0]=C x0`, `obs[1]=C x0`, `obs[t>=2]=C x[t-1]`.

- Reset duplicate at t=1 is skipped, not counted as a second independent
  measurement, and does not advance state/covariance.
- At t>=2 propagate with completed u[t-2], then condition on obs[t].
- Forecast first propagates pending u[t-1], giving obs[t+1]. Only the second
  forecast step uses new u[t], giving obs[t+2].

Thus h1 predicts an old command, not the choice of a new action. Tests verify
that changing the new command cannot change the pending prediction.

All horizons 1/2/6/12 share origins t=72..276 with a complete 12-step future.
The initial observations are inside the 288-step budget; there is no hidden
warm-up. Received temperatures through the origin can update latent state,
never thermal parameters. The 72-step prefix is not a parameter-training phase.

## Fixed estimation assumptions

Independent physical-cell prior std **2 K**, per-step process std **0.01 K**,
assumed sensor std **0.0001 K** (actual synthetic measurement noise is zero).
Physical covariance is projected into modal coordinates; these are stated
priors, not learned noise levels. Joseph covariance updates maintain numerical
symmetry. There is no validation-based noise/parameter search.

## Mixed-input result

Conditional prediction MAE (K), **13-state model**:

| Sensor horizon | Kalman | Same model without later corrections | Persistence |
| --- | ---: | ---: | ---: |
| 1 / 5 min | 0.0001094 | 0.0549890 | 0.0376745 |
| 2 / 10 min | 0.0001909 | 0.0546135 | 0.0742879 |
| 6 / 30 min | 0.0004907 | 0.0530986 | 0.2166500 |
| 12 / 60 min | 0.0009076 | 0.0508844 | 0.4269227 |

These are output forecasts, not reconstruction-error bounds on every physical
wall/slab temperature. A tiny received-sensor residual is also not proof that
hidden states are physically correct. Reconstructed temperature ranges are
reported as diagnostics, not used to clip the model into apparent validity.

## Late pulses: test commands after the prefix

Original fixed/air pulses occur during steps 0..11, before the scoring prefix;
their later scores measure recovery. V2 adds separate pulses at **steps 96..107**
with no earlier forcing. A dedicated origin window **84..119** includes the
onset/offset and forecast paths crossing them, avoiding whole-day dilution.

13-state h2/10-minute MAEs in that window:

| Case | Kalman | Open loop | Persistence |
| --- | ---: | ---: | ---: |
| Late uniform-heat pulse | 0.00000491 | 0.00001646 | 0.02539834 |
| Late diffuser-air pulse | 0.00000710 | 0.00002436 | 0.03641269 |

At h12, Kalman MAEs are 0.0000344/0.0000498 K versus persistence
0.13755/0.19765 K. These uniform-start pulse cases also show that the known
open-loop model is already very accurate; the huge persistence improvement
must not all be credited to the observer. Conditional future commands are
supplied, so it is not unexpected that persistence cannot anticipate onset.

## More states do not always improve estimation

At 72 received observations and fixed flow, prior-scaled observability has
effective relative rank (threshold 1e-8):

| States | Effective rank | Smallest/largest singular ratio, no flow |
| --- | ---: | ---: |
| 13 | 13 | 4.73e-6 |
| 21 | 18 | 2.57e-10 |
| 29 | 20 | 1.00e-15 |

This is a finite, fixed-coefficient conditioning diagnostic, not a claim of
exact structural unobservability or a proof for switching inputs. The 29-state
mixed-case h12 MAE is 0.002167 K versus 13-state 0.000908 K. Conversely,
29 states improve the passive and initial-only cases. Do not choose model
size using reduction error alone, or silently select the best per case.

## Decision and next gate

Positive causal-state result **with known physics**: two delayed sensors can
support accurate conditional output forecasts without reading hidden initial
temperatures on this fixture. The small model is not yet transferable:
materials, forcing and geometry-derived bases are still known exactly, only
one synthetic geometry is tested, and parameter/state mismatch is absent.

Next gate is deliberate parameter/forcing mismatch with the observer frozen,
followed by prefix-only identification using allowed observations if the
mismatch audit warrants it. Do not copy true material fields from scenarios
into a controller or promote these synthetic errors as building-policy scores.
Champion and simulator unchanged; no leaderboard improvement established.

## Verification

Eight new tests cover reset duplicate, delayed completed input, pending versus
new action, future observation/input mutation, equal two-sensor initialization,
no hidden-initial-state reads, generator timing including late pulses, matched
samples and budget/horizon rejection. Combined suite: **134 pass**.

```sh
./.venv/bin/python -B tools/thermal_state_observer_check.py --steps 288 --prefix 72
./.venv/bin/python -B -m unittest tools.test_thermal_state_observer_check tools.test_thermal_system_modes_check tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
