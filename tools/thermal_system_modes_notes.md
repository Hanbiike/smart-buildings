# Whole-system reduction: no retained full air/slab block

Follow-up: `tools/thermal_state_observer_notes.md` removes hidden-state
initialization and tests causal updates from two delayed zone sensors, including
pulses after the scoring prefix. Known physics/forcing remain oracle inputs.
Expanded suite: 134 tests.

Tool: `tools/thermal_system_modes_check.py`.
Report: `smart_buildings/results/thermal_system_modes_v1.json`.

## Protocol and construction

Use the same source-checked **164-state known synthetic system**, now reducing
air, walls and slab together. No physical state block is retained exactly.
There is no environment, archive fitting, controller or building material
identification. Each response is limited to **288 steps / 24 hours**, without
warm-up or subsequent reference-state corrections. Stochastic convection is
absent. The small symmetric fixture is not an office or held-out building.

First retain uniform temperature and the independent DC responses to two
uniform-power ports, two diffuser-power ports and ambient. This gives **5
static directions**, preserves DC transfer, and permits total energy balance.
Next collect input response at periods 600/1800/3600 s and feedback (0,0)/(4,3),
remove the static span in capacity coordinates, normalize real/imaginary input
columns separately, and order directions by SVD. An input/output variant also
includes responses of the conjugate-transposed operator driven by the actual
two zone-average sensors. Trial/test spaces remain identical (Galerkin), not
Petrov-Galerkin or balanced truncation. No trajectory or initial-state data
enters basis construction.

On this fixture both snapshot families have numerical residual rank **25**.
The 32-enrichment option was excluded after discovering this rank, before the
time-response screen; it would exceed meaningful snapshot rank. All valid
predeclared counts 0/4/8/12/16/24 are reported, for both input and input/output
ordering. Total state counts are **5/9/13/17/21/29**.

Frequency validation includes intermediate 900/1200/2700-second periods,
long periods 7200/21600/86400 s, and feedback (1,5) absent from design. These
are off-design checks on one known operator, not new-building validation.

## Forced and initial-condition responses are separate

The four existing passive/pulse/mixed response cases are retained for direct
comparison, not called untouched validation trajectories. A fifth
`initial_only` case starts from a deterministic seeded nonuniform field with
zero net initial heat perturbation and no applied forcing relative to 295 K.
It is **one** independent initial-condition probe, not a worst-case guarantee.
The full and reduced models propagate independently from capacity-projected
known initial states, without assimilation or future-state correction.

Initial projection remains oracle information. At 21 states in the input/output
variant, initial-only zone projection errors are -0.0110/-0.0535 K; at 29 they
are +0.00178/+0.00240 K. Reported response errors below start after the first
step, not at t=0. Correctly reconstructing online state from two delayed sensors
has **not** been solved by this experiment.

## Input/output-ordered results

Whole-response zone MAE (K):

| Total states | Passive | Heat pulse | Air pulse | Mixed | Initial-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.0129211 | 0.0000850 | 0.0001219 | 0.0106260 | 0.0015265 |
| 21 | 0.0016888 | 0.0000103 | 0.0000130 | 0.0015984 | 0.0003734 |
| 29 | 0.0000954 | 0.0000004 | 0.0000005 | 0.0001433 | 0.0000843 |

At 29 states, mixed peak error is **0.000397 K**, air-pulse peak is
**0.0000012 K**, and initial-only peak is **0.000227 K** after the first step.
First-12-step errors and checkpoints 1/2/6/12/72/288 are retained. Pulse MAEs
alone would dilute fast error over a long recovery, so peak/checkpoint results
must remain visible. The input/output family modestly improves several cases,
but does not uniformly dominate input-only: e.g. initial-only MAE at 21 states
is 0.0003734 versus input-only 0.0003534 K. No output-optimality claim follows.

Intermediate-period maximum relative complex transfer-column errors are:

- 13 states: **0.00501%**;
- 21 states: **0.00000523%**;
- 29 states: approximately **2.14e-10%**.

Across all audited non-DC periods at off-design (1,5) feedback the respective
maxima are **0.3595%, 0.03965%, 0.002188%**. DC error is below 1.4e-14.
Harmonic period is not forecast horizon; tiny transfer errors at sampled
frequencies do not guarantee state-initialized predictions at all horizons.

## Conservation and stability

Every basis is capacity-orthonormal and spans the uniform-temperature vector.
Maximum total-energy residual across reported responses is **1.42e-9 W**.
Fixed-step propagation spectral radii are checked at five conductance pairs:
(0,0), (4,3), (1,5), (0,3), (4,0). All are below one; the largest is
**0.991915**. This is a sampled fixed-coefficient stability audit, not a proof
for all flow values or arbitrary switching. The reported switching cases
also remain finite over the bounded response.

Reduced coordinates are signed spatial-mode amplitudes, not individual slab
or wall temperatures. Old passive-envelope bounds cannot be applied to every
coordinate. Physical constraints, if needed, must be checked on reconstructed
physical states and with the correct thermal topology.

## Decision

Positive structural result: a **29-state** response-based model approximates
the known 164-state system closely, with no full air/slab block and no parameter
tuning to trajectories. This is about model reduction, not a transferable
controller. Basis construction still needs exact materials/geometry, and state
initialization still reads the full known synthetic history.

Next gate: **causal state observability/estimation from only the two delayed
zone temperatures**, with frozen known coefficients first. Keep pending
command timing explicit and report parameter/state mismatch separately.
After that, unknown-material identification and other geometries/convection
must be tested; do not transplant this fixture's matrices or private simulator
state into a participant policy. Champion unchanged, no score improvement.

## Verification

Six tests added: basis size/nesting/constant/capacity checks, off-design DC and
sampled stability, construction independence from initial/trajectory functions,
general projected recurrence, initial-only response/energy and early budget
rejection. Combined suite: **126 pass**.

```sh
./.venv/bin/python -B tools/thermal_system_modes_check.py --steps 288
./.venv/bin/python -B -m unittest tools.test_thermal_system_modes_check tools.test_thermal_port_modes_check tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
