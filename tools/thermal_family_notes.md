# Finite prefix-compatible parameter/state family

Follow-up: `tools/thermal_adequacy_notes.md` tests a pre-update causal change
monitor with disjoint family-selection/calibration blocks. Abrupt changes are
detected, but existing mismatch/noise can be absorbed during calibration.
It does not authorize control. Expanded suite: 162 tests.

Tool: `tools/thermal_family_check.py`.
Report: `smart_buildings/results/thermal_family_v1.json`.

## Protocol

Same known nominal 164-state synthetic fixture with **13 states per member**.
No environment, real-building fitting, convection or controller. Each physical
response is limited to **288 steps / 24 hours**, with the first 72 observations
inside that budget. Truth labels and hidden fields never enter selection.
Nominal future commands/forcing are supplied conditionally, not realized
future plant heat or unknown airflow gain.

The finite prior has **135 candidates**: all capacity/exterior/airflow scale
triples in {0.8,1.0,1.2}, crossed with load pairs (0,0), (+/-5,0), (0,+/-5) W.
It does not include simultaneous nonzero loads in both rooms, arbitrary load
schedules, all possible materials, or structural model errors. The prior range
was motivated by the preceding stress audit and is not independent building
knowledge. Each member has its own causal latent-state filter and covariance,
initialized from two sensors only. Physics/noise assumptions remain unchanged.

Compatibility uses the same horizon-normalized prefix residuals as the scalar
fit audit. Let s=max(0.001 K, 1.4826*MAD(nominal residuals)). Retain candidates
with prefix MSE <= best MSE+(2s)^2, then assign normalized weights proportional
to exp(-(MSE-best MSE)/s^2). These are **heuristic compatibility weights**, not
Bayesian posterior probabilities. Membership and weights stay frozen after
the prefix; only latent states update with received data. Nominal and the
previous point-fit rule are retained as separate comparison arms.

The first six truth scenarios are reused from earlier exploratory audits.
Additional checks use capacity/exterior/airflow **(1.1,0.9,1.1)** between grid
nodes and an airflow gain **1.5**, outside the declared prior. Each crosses
mixed inputs and a late air pulse: **16 responses**, not 16 buildings.

## Envelopes and scoring

The plain envelope is the minimum/maximum predicted temperature across
retained members. A second envelope takes each member's mean +/-2 predicted
latent-output standard deviations before the min/max. The latter propagates
the fixed assumed Q/R; future measurement variance is excluded because the
target is noiseless physical temperature. It does not add missing-process or
noise uncertainty magically, and **is not a calibrated 95% confidence interval**.

Coverage, width and maximum miss are reported at matched h1/h2/h6/h12 origins,
both over the full evaluation and pulse origins 84..119. No future coverage
result changes selection or weights. Targets are used for evaluation only.
Each member preserves its own model's energy balance; their weighted output
mean is not claimed to be a single physical model with conserved energy.

## Mean prediction results

Mixed-input h12/60-minute MAE (K):

| Case | Frozen nominal | Prefix point fit | Family mean |
| --- | ---: | ---: | ---: |
| nominal | **0.000908** | 0.006115 | 0.004234 |
| capacity x1.2 | 0.073782 | 0.006961 | **0.002192** |
| exterior x0.8 | 0.079355 | 0.002605 | **0.001469** |
| airflow x0.8 | 0.012246 | 0.006263 | **0.004794** |
| late hidden heat | **0.007186** | 0.010360 | 0.009234 |
| sensor noise | **0.021476** | **0.021476** | 0.022711 |
| between-grid parameters | 0.078373 | 0.003191 | **0.001475** |
| airflow outside prior | 0.030718 | **0.005718** | 0.020315 |

Family mean is worse than frozen nominal in **5/16** responses at h12.
This rule does not pass a no-regression gate and is not promoted as a predictor
or policy. Reducing point-fit regression is not the same as eliminating it.

In mixed nominal response, 16 members are retained (effective count about 10);
in unexcited late-pulse prefixes all 135 remain. This explicitly preserves
uncertainty instead of assigning a future airflow gain from later observations.

## Coverage: where the finite family fails

For noiseless tested cases inside the prior, including between-grid parameters,
plain h12 envelopes cover all evaluated zone samples. Their mean widths range
roughly **0.041..0.122 K**, so coverage is not equivalent to a tiny prediction
error. Do not generalize this finite audit into a robustness guarantee.

Key failures:

- **Hidden future heat:** mixed h12 coverage is 94.39%, maximum miss 0.01785 K;
  covariance augmentation raises coverage to 97.32%, still not complete.
- **Noise mismatch:** late-pulse h2 coverage is only **20.24%** (mean width
  0.01655 K, maximum miss 0.09528 K); augmented coverage is 37.07%. Within the
  pulse-origin window, h2 coverage is 29.17% / 47.22% plain/augmented. This
  deliberately injects 0.05 K noise while assumed sensor std stays 0.0001 K;
  current source simulator does not itself inject this noise.
- **Outside prior:** airflow x1.5 mixed h12 coverage is **40.0%**; augmentation
  yields 93.66%, not a guarantee. In late-pulse origins, h2 coverage is only
  72.22% and maximum miss **0.14688 K**, despite all 135 prior members being
  retained. Keeping every member cannot cover a parameter outside the prior.

Hidden-state compensation is not solved by averaging. In the noisy late-air
case the weighted reconstructed field ranges **271.65..305.90 K**, while the
true field stays **293.80..295.00 K**. Truth ranges are offline diagnostics
attached after estimation, never clipping bounds or participant features.

## Decision

The family usefully expresses some parameter ambiguity and improves several
known-system perturbed cases, including between-grid parameters. It does
**not** deliver reliable general uncertainty bounds or safe mean predictions:
unknown future inputs, noise misspecification and out-of-prior physics remain
failure modes. Do not replace the champion or use the envelope as a hard MPC
safety guarantee. No real-building or leaderboard improvement established.

Next gate is causal adequacy monitoring: detect when received innovations are
incompatible with the frozen family and invalidate its forecast confidence,
rather than forcing measurements into hidden-state corrections. A monitor
cannot predict an unseen change before it occurs; report detection delay and
false alarms, not retrospective rejection using future errors. Noise and
unknown-load uncertainty need explicit treatment before any confidence claim.

## Verification

Six tests added: declared finite grid and weights, future mutation invariance,
forecast-covariance nonmutation, explicit interval misses, clean-target score
isolation with frozen members/weights, and early budget/prefix rejection.
Combined suite: **154 tests pass**. Simulator and champion unchanged.

```sh
./.venv/bin/python -B tools/thermal_family_check.py --steps 288 --prefix 72
./.venv/bin/python -B -m unittest tools.test_thermal_family_check tools.test_thermal_prefix_identification_check tools.test_thermal_mismatch_check tools.test_thermal_state_observer_check tools.test_thermal_system_modes_check tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
