# Input-response wall modes: positive known-system reduction result

Follow-up: `tools/thermal_system_modes_notes.md` reduces the whole system,
including air/slab, to 13/21/29 states and separately audits an initial-only
response. This remains a known-system reduction, not causal identification or
control. Expanded combined suite: 126 tests.

Tool: `tools/thermal_port_modes_check.py`.
Report: `smart_buildings/results/thermal_port_modes_v1.json`.

## Design and protocol

Same known 164-state synthetic fixture: two 6x5-cell rooms with true cores,
44 wall states and 120 retained air/slab port coordinates. This remains a
small-scale algebra fixture, not an office or competition scenario. No materials
were identified, and no environment or policy was run.

Retain the static wall constraint lift and free ambient-response coordinate,
preserving DC transfer and uniform temperature. Collect response to two uniform
heat ports, two diffuser-power ports and ambient at **600/1800/3600-second
periods**, for air conductances **(0,0) and (4,3) W/K**.

From each full response subtract its static wall lift, weight by sqrt(C_wall),
remove the retained ambient direction, split real/imaginary parts and normalize
each nonzero input component separately. SVD orders enrichment directions.
This is input-response enrichment, not balanced truncation or output-optimal
reduction. Snapshot numerical rank is 25 on this fixture.

No response trajectory, mixed-case temperature or future input schedule enters
basis construction. Counts 0/2/4/8/12/16 are all reported. Bases are nested and
C-orthonormalized; the general projected-capacity solver is reused.

Frequency checks include intermediate **900/1200/2700-second periods**, long
periods 7200/21600/86400 s, and feedback **(1,5) W/K** absent from design.
Four free responses independently propagate their own state for **288 steps /
24 hours** each, without warm-up or reference corrections. These are the same
previously used synthetic input cases, not untouched trajectories or held-out
buildings. Convection shuffling is absent.

Initial states are capacity-projected. Uniform starts are exact; mixed-case
initial zone error for 8 response modes is 0.02946 K. Subsequent accuracy does
not establish online state initialization. Exact matrices are known here and
must not become participant controller inputs. Harmonic period is not forecast
horizon; both frequency and transient checks are needed.

## Equal-size results

All three variants below have **129 states** (120 air/slab ports, 9 wall
coordinates):

| Metric | Coarse isothermal | 8 slow modes | 8 response modes |
| --- | ---: | ---: | ---: |
| Passive MAE, K | 0.213672 | 0.045918 | 0.000545 |
| Mixed MAE, K | 0.151000 | 0.036152 | 0.000346 |
| Heat-pulse peak error, K | 0.004617 | 0.226929 | 0.000097 |
| Air-pulse peak error, K | 0.006329 | 0.345589 | 0.000160 |
| Mixed peak error, K | 0.178417 | 0.270926 | 0.006930 |

First-12-step pulse MAEs are retained in the report. Refinement still is not
universally monotone: mixed peak error increases from 0.05007 to 0.07027 K
between 2 and 4 response modes. Nesting is not a maximum-error guarantee.

At 8 response modes, maximum relative complex transfer-column error at the
intermediate periods across all audited feedback settings is **0.05425%**.
Across all non-DC frequencies with off-design (1,5) feedback it is **0.05830%**,
versus 78.93% for 8 slow modes. This is a finite frequency grid, not a uniform
bound or building-generalization result. Columns are normalized separately
to avoid pooling different input units. Response-basis DC error is below
1.2e-13. Maximum absolute total-energy residual is below **4.49e-9 W**.

## Decision and remaining gap

Positive result for the wall-reduction **method** on this known system; retain
as a reference candidate, not an HVAC policy. At 8 modes it eliminates only
35/164 states. Keeping every air/slab cell is not a small scalable building
model, and all physical parameters and initial history are oracle information.

Next structural gate: reduce the complete system using actual actuation and
observation ports, rather than retaining every air/slab cell. Retain DC,
energy, fast-transient and off-design checks. Arbitrary initial-state response,
convection, other geometries/materials and causal parameter/state inference
remain unresolved. No hidden building material fields, wall/slab states or
future weather may enter the controller. No policy-score improvement established.

## Verification

Tests check nesting, C-orthonormality, inclusion of uniform temperature, DC
preservation at off-design feedback, budget rejection, energy and independence
of basis construction from the trajectory forcing function. `transfer_audit`
accepts explicit periods/feedback with old defaults unchanged. Combined suite:
**120 pass**. Simulator and champion unchanged.

```sh
./.venv/bin/python -B tools/thermal_port_modes_check.py --steps 288
./.venv/bin/python -B -m unittest tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
