# Matched latent-state identification audit

Source-level follow-up: `tools/simulator_source_audit.md`. The active simulator
has separate dynamic facade/partition CVs, per-air-CV slab states and distinct
spatial application maps for air versus fixed heat. The two-state zone model
is therefore not its exact reduction. Passive-envelope claims below apply to
our isolated slab topology, not to arbitrary effective wall/latent states.
Archive/source version identity has not been established.

Report: `smart_buildings/results/modular_thermal_kalman_matched_v1.json`.

Protocol: five existing archived workday windows, offset 576, 288 observations
each, 72-observation fitting prefix (70 aligned transitions), 40 fitting
evaluations per zone. No simulator runs, controller changes or policy scores.
Future recorded commands condition the offline forecast; weather/comfort are
held and water is causally rate-projected. Parameters stay frozen after fitting.

The fitter now uses the same Kalman mass update as validation, including its
fixed initial variance and process noise. Default boundary fitting is retained
for backwards compatibility. Regression tests check protocol propagation and
the partition of prediction samples by actuator uncertainty.

Equal-case mean MAE (K):

| Horizon | Earlier mismatched fit | Matched fit | Persistence |
| --- | ---: | ---: | ---: |
| 2 | 0.20931 | 0.23477 | 0.14972 |
| 6 | 0.40634 | 0.52656 | 0.26195 |
| 12 | 0.59499 | 0.84619 | 0.48573 |

At h6 even the subset without annotated actuator-floor uncertainty fails:
0.57097 K versus persistence 0.34131 K. This is a conservative origin-wide
partition using uncertainty anywhere in the longest forecast path, not proof
that equipment inputs in the clean subset are exact. Subsets differ in regime;
their errors cannot establish a causal effect of actuator uncertainty.

Solver budget exhaustion falls from 17 to 6 zones (all six in Atlanta), but
physical consistency worsens. Fairbanks-2 estimated mass reaches 424.44 K,
128.50 K outside the passive envelope. Fairbanks-8 maximum envelope violation
is 52.90 K. Counts include prefix replay as well as validation, and count
zone-transition events, not independent failures.

Conclusion: matching the state estimator does not fix the failure. Numerical
optimizer convergence does not establish identifiability or physically valid
latent states. Kalman correction can turn the passive node into an effective
unmodelled heat source. Do not integrate this model into MPC or treat clipping
the latent state as identification. Next diagnostic should isolate heat-balance
residuals and latent-state observability on the prefix, with physical constraints
specified before validation; avoid further noise/parameter tuning to these cases.

Verification: 85 unit tests passed across the thermal, equipment and greybox
test modules. Champion controller remains unchanged.

## Prefix-only balance and sensitivity audit

`smart_buildings/results/modular_thermal_prefix_audit_v1.json` refits only the
same 72-observation prefixes, with matched Kalman updates and no validation
targets. It adds a passive mass replay, observed-air backward-Euler power
residual, and the next mass temperature required to close that balance exactly.
No parameters or noise assumptions were changed.

153/223 zones require mass outside the causal passive envelope at least once
even on the fitting prefix. Median passive power MAEs by case are 55.56,
167.33, 66.46, 46.46 and 22.57 W (report order). These are effective fitted
power residuals, not independently measured missing loads.

Data-only finite-difference sensitivities exclude regularization residuals.
Coordinates are log capacities/conductances, load in kW and initial offset in K.
Fairbanks-2 median smallest/largest singular-value ratio is 2.76e-5: strong
local anisotropy in these coordinates, not a coordinate-independent proof of
structural nonidentifiability. Other case medians range 0.00383–0.01835.
Required-mass extremes are also ill-conditioned when fitted mass conductance
is small; they must not be interpreted as actual wall temperatures. Exact
synthetic physics closes the diagnostic power balance in a regression test.

This supports rejecting these fitted latent states before control integration,
but does not uniquely locate the missing physics or distinguish every source
of equipment error. A next experiment should use passive, constrained fitting
and held-out prediction rather than Kalman tuning. Current test suite: 87 pass.

## Passive multistep experiment

Report: `smart_buildings/results/modular_thermal_passive_multistep_v1.json`.
Same five 288-observation windows and 72-observation prefixes; no simulator.
The opt-in `--passive-multistep` variant uses boundary-conditioned passive mass
at training origins and uncorrected joint rollouts within the prefix. Loss uses
horizons 1/2/6/12, errors divided by horizon and equal aggregate weight per
horizon. Training uses reconstructed recorded equipment forcing, including
recorded supply conditions; validation still reconstructs equipment from the
forecast state. This conditional training protocol is not an online forecast.

The initial mass offset is bounded to +/-3 K as a declared experimental prior,
not a measured physical property. Existing positivity bounds, fixed mass/air
capacity ratio 20 and topology are unchanged. No validation-based prior tuning.
The report's `training_mae_K` in this mode describes horizon-normalized
residuals, explicitly labeled; it is not ordinary one-step MAE.

Passive-envelope violations are zero in all five windows. Nevertheless,
equal-case mean prediction MAEs at h2/h6/h12 are 0.28320/0.53111/0.74677 K,
versus persistence 0.14972/0.26195/0.48573 K. The identified-actuator subset
also loses. 29/223 fits exhaust the unchanged 40-evaluation budget.

Physical state consistency is restored, prediction utility is not. This is a
combined objective/prior experiment, not an ablation that identifies which
change caused the error. Do not promote to MPC. Further parameter searches on
these same windows risk overfitting; resolve missing heat pathways/forcing
and conditional-training mismatch before more tuning. Champion unchanged.

## Oracle-forcing ablation

Tool: `tools/thermal_forcing_audit.py`. Report:
`smart_buildings/results/modular_thermal_forcing_audit_v1.json`.
Reuses the frozen passive-multistep fits (no fitting or simulator calls).
Both arms use identical origins with a full 12-step future available, the same
causal origin mass, and no future zone-temperature correction. The oracle arm
reads future reconstructed recorded flow, supply temperature, reheat, weather
and comfort. This deliberately uses future observations, including their zone
temperatures in equipment reconstruction, and is strictly an offline diagnostic.
The causal projection observer is separate from the future-forcing tape.

Equal-case MAEs (K), on matched origins:

| Horizon | Projected inputs | Oracle recorded inputs | Persistence |
| --- | ---: | ---: | ---: |
| 1 | 0.14035 | 0.14035 | 0.08532 |
| 2 | 0.28527 | 0.23318 | 0.15018 |
| 6 | 0.53354 | 0.40929 | 0.26153 |
| 12 | 0.74677 | 0.58260 | 0.48573 |

Pending-step errors coincide, as expected from the delay contract. Future
forcing accounts for a material part of the discrepancy (about 23% MAE
reduction at h6), but oracle forcing still fails persistence (about 56% worse
at h6 and 20% at h12). Thus fixing forecast equipment/weather inputs alone
does not validate these frozen fits. This does not uniquely prove missing
thermal topology: fitted-parameter errors, origin mass, discretization and
equipment reconstruction errors remain confounded. Oracle inputs are not
independently measured heat powers or a mathematical lower error bound.

Do not compare short-horizon means directly to previous reports: the new
ablation drops incomplete 12-step origins at every horizon for exact pairing.
Next diagnostic should check residuals with known physical parameters on a
transparent synthetic fixture and investigate the simulator's actual thermal
update/discretization, without importing hidden building parameters into a
controller. Test suite: 91 pass. Champion unchanged; no submission improvement.
