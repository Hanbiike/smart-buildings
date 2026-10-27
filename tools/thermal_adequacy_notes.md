# Causal forecast adequacy monitor: change detection is not initial validation

Tool: `tools/thermal_adequacy_check.py`.
Report: `smart_buildings/results/thermal_adequacy_v2.json`.
Hashes: `smart_buildings/results/thermal_adequacy_v2_manifest.json`.
V1 retained; V2 adds failures already present during prefix selection/calibration.

## Information boundary

Known synthetic 164-state plant, 13-state members of the finite parameter
family. Each physical response is **288 steps / 24 hours**, no additional
warm-up. No environment, controller, real-building inference or convection.

Family selection uses observations **0..47** only. Calibration uses a disjoint
block **48..71**. Selection, family and calibration thresholds remain frozen
thereafter. Each member continues causal latent-state updates, not coefficient
fitting. Monitor receives no true parameters, onset labels or hidden fields.

Before `receive(obs[t])`, predict that sensor using the previous posterior and
completed command u[t-2]. Only after scoring does the Kalman update consume
obs[t]. This is equivalent to preserving the prior pending forecast; new u[t]
cannot substitute into this residual. In particular, post-update near-zero
sensor residuals cannot mask a pre-update error here.

## Rule

The prior sensor envelope is min/max across members. For each sensor, calibrate
an absolute gap tolerance as max(0.001 K, empirical 95th-percentile prefix gap
+ 3 * robust MAD of midpoint innovations). This small 24-observation block
does **not** justify a calibrated 95% statistical claim.

Reject after two consecutive gap/tolerance ratios >1 or one ratio >3.
Rejection is latched; no automatic recovery, widening of thresholds or
reselection. `HEALTHY` means only 'not rejected by this change detector'.
`control_permission` is explicitly false, even for healthy traces. Excitation
support from the selection prefix is reported separately; the unexcited late
pulse is not magically identified just because its monitor is quiet.

## Results

| Case | First observable violation | First alarm | Interpretation |
| --- | ---: | ---: | --- |
| Nominal mixed | none | none | no false rejection on this response |
| Nominal late-air pulse | none | none | ordinary onset not rejected |
| Hidden heat starts at physical step 96 | obs98 | obs98 | detected at first observable effect |
| Airflow x1.5, first airflow at step 96 | obs98 | obs98 | detected at first observable effect |
| Noise std0.05 K starts at obs98 | obs98 | obs98 | immediate data change detected |
| Airflow x1.5 from reset, mixed input | obs2 | obs268 | **far too late**; existing mismatch partly absorbed |
| Noise std0.05 K present from reset | obs0 | none | **missed**; corrupted prefix becomes calibration norm |

Thermal changes at physical step96 cannot be observed until obs98 because of
sensor latching. Onset labels are used after the monitoring trace for scoring,
never as detector input. Zero detection delay in the three abrupt cases is
relative to that first observable sample, not a forecast of an unseen event.

The two initial nominal responses have zero rejected time. This is not an
estimated guarantee of a low false-alarm rate across buildings or seeds. The
latched monitor rejects 88.02% of the monitored samples after an obs98 alarm;
that fraction reflects the intentionally absent recovery, not event severity.

The noise is a deliberately injected stress; source simulator sensors do not
themselves inject this 0.05 K noise. The test illustrates calibration/model
misspecification, not a measured property of competition observations.

## Decision

This is a useful **change detector**, not a completed reliability gate. A
family that was wrong before calibration can normalize its own error. Absence
of an alarm cannot authorize model control, certify hidden temperatures or
turn a finite family envelope into a safety bound. The late/missed cases must
not be removed from the report or fixed using their future onset labels.

Next integration requires a separate initial-prefix adequacy/excitation gate
and a participant-metadata predictor adapter, then shadow evaluation on the
fresh workday data. Do not extend these two-room successes directly to a real
39-zone controller. No further confidence claim should be based on change
detection alone. Simulator and champion remain unchanged; no policy gain.

## Verification

Five tests check nominal behavior and latched rejection, immutable calibration,
two-moderate-miss logic, future mutation preserving earlier traces/thresholds,
and invalid calibration input rejection. Combined suite: **162 tests pass**.

```sh
./.venv/bin/python -B tools/thermal_adequacy_check.py --steps 288
./.venv/bin/python -B -m unittest tools.test_thermal_adequacy_check tools.test_evaluation_start tools.test_parallel_eval tools.test_thermal_family_check tools.test_thermal_prefix_identification_check tools.test_thermal_mismatch_check tools.test_thermal_state_observer_check tools.test_thermal_system_modes_check tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
