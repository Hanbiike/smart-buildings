# Modular equipment input validation

Modules:

- `smart_buildings/baselines/equipment_balance.py`: pure physical balances.
- `smart_buildings/baselines/equipment_observer.py`: metadata decoding,
  causal actuator calibration, delayed-transition bookkeeping and candidate
  equipment inputs. No RLS, fitted zone parameters or controller integration.
- `tools/equipment_observer_check.py`: saved-window input audit.

## Explicit timing contract

For non-initial samples the zone sensor in observation t represents physical
temperature x[t-1]. Actuators and flow in that same observation reflect command
u[t-1]. The thermal solve for that command used outdoor and water inputs from
observation t-1. Therefore:

1. `observe(obs[t])` builds `pending` for u[t-1], using the latched zone
   temperatures in obs[t], effective actuators/flow in obs[t], but preceding
   outdoor/water values. A new command must not replace it.
2. The next zone observation resolves that pending transition. Returned
   `DelayedTransition` pairs include the original forcing and its control-step
   index. The reset duplicate/unknown preceding command is skipped.
3. A future thermal estimator must first propagate pending physics to estimate
   current x[t], then evaluate candidate u[t]. `project_action()` accepts that
   explicit temperature state. If it is omitted, output is labeled
   `lagged_observation_proxy`, not presented as a true current state.
4. `project_water_sensor()` forecasts the next reported water temperature. It
   does not retroactively substitute that value into pending thermal forcing.

The `air_power` diagnostic is instantaneous at the specified zone temperature,
not five-minute mean power. Integration must retain the flow-dependent term
`rho*cp*flow*(supply-T(t))`; holding the initial air power throughout the step
would repeat the old explicit heat-delivery approximation.

## Geometry and equipment assumptions

Regular zone areas are reconstructed as four-connected interior components of
the supplied 0/1/2 material map. Those material numbers are not room IDs. A
component-label coverage check is necessary, but cannot certify arbitrary
custom zone-to-VAV mappings. Custom mappings require validated area input.
Area counts are used as relative weights for return air, not converted into
fabricated thermal capacities. Missing geometry is labeled equal-area proxy.

Global assumptions are explicit and included in each report: 2000 Pa design
air pressure, valve Kv 0.108 m3/h, zero header resistance, 308.15 K discharge
cap and 0.9 fan efficiency. These are public-model priors, not newly identified
unknown equipment parameters. The flow/fan observation ratio identifies AHU
capacity causally; an unobserved capacity is never assumed infinite.

Damper/fan floors are identified only after observed clipping of a prior
command. An unseen floor remains uncertain when a new command goes below its
observed upper bound. Estimates expose this uncertainty; strict projection
raises `UnidentifiedActuatorError`. Floors are not filled with the answers
from future samples or copied from unrelated zones.

`intake_fan_power` excludes exhaust fans, pumps and chiller electricity. Neither
that estimate nor coil duty has been validated against measured power here.
No price, carbon coefficient or reward enters the temperature balance.

## Reproduction and scope

```sh
./.venv/bin/python -B -m unittest tools.test_equipment_balance tools.test_equipment_observer tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
./.venv/bin/python -B tools/equipment_observer_check.py --all-scenarios --steps 288 --offset 576 --summary --output smart_buildings/results/equipment_observer_workday_v2.json
```

The output path must not already exist. Each of 30 cases consumes exactly 288
existing observations, starting at archive offset 576, with 72 included samples
for calibration before scoring next-input predictions. No preceding samples,
new simulations, policy initialization or scenario configuration are consumed.

The v1 report records all input errors. V2 retains those same errors and adds
uncertainty diagnostics; it does not remove the outlier from overall metrics.
One first-time closing command in building_train_2/atlanta_3 requests zero
damper for ten zones whose floor had never been observed. Its flow error is
0.63246 m3/s. An uncertainty annotation is more honest than retrospectively
assigning their true 0.1 floor.

Across cases, mean water-sensor MAE is 0.00725 K versus 0.10361 K for holding
water temperature. Other than the single unidentified-floor event, handler
flow errors are at floating-point precision. These are **input-validation
results, not zone-temperature forecasts, comfort results or leaderboard scores**.

The audit contains 36,550 handler-flow predictions. Of those, 4,785 are marked
as possibly encountering an unidentified floor, even though most happen to be
numerically correct on this archive. The `identified_floor_flow_m3_s` subset
means floors are either identified or bounded by past observations so they
cannot bind at the candidate setting; its maximum error is 2.43e-7 m3/s.
Overall v2 flow/water errors match v1 exactly: uncertainty annotations did not
remove difficult samples from the overall metrics. There were no missing-
capacity skips in this particular workday audit, but synthetic tests explicitly
cover that case and require it to be reported, not filled with infinite flow.

Next: fit only thermal capacities/conductances and estimate the latent mass
temperature, retaining this equipment/timing contract. Use a stable integrator
for the dynamic air-exchange term; do not insert prices into the ODE or attach
an unvalidated model to the champion. Existing policies remain unchanged.
