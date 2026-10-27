# Tiny spatial reference and conservative reduction

## Scope

`tools/thermal_grid_reference.py` defines a transparent synthetic 7x9 grid:
two rooms of six air CVs each, a shared partition, facade CVs and a separate
slab node for every air CV. There are 35 solved air/wall CVs plus 12 slab
nodes, **47 states** total. All dimensions, materials, powers and airflow
conductances are explicit invented fixture constants. This is not a real
building, not an inferred parameter set, and not a participant controller.

No environment is constructed, no historical window is fitted, no trajectory
is simulated. Each case is one algebraic backward-Euler step; the source solver
is instantiated fresh for each comparison. Private mass state is initialized
only to prescribe known synthetic history in the solver under test.

## Independent reference

The dense reference assembles physical capacities, series half-face
resistances, exposed surface-film conductances, slab links and both application
maps directly from the fixture. It does **not** call the simulator's matrix,
face-conductance, boundary classification, slab elimination or Woodbury helpers.
The spatial geometry is a simple rectangle with half-width boundary CVs.

Its full matrix is

    diag(C/dt + H) + L + P diag(g) S^T,

including explicit slab states. Fixed heat uses a uniform room map; air heat
uses sparse diffusers and full-room mean sensors. The object under test is
`FiniteDifferenceSolver`, which independently eliminates slab nodes and uses
its sparse/low-rank implementation. This checks the rectangular fixture, not
every possible irregular floor-plan boundary or convection permutation.

## Measured results

Latest artifact: `smart_buildings/results/thermal_grid_reference_v2.json`.
V1 is the preceding report without grouped energy-flow diagnostics; retained.

- Maximum source/reference temperature disagreement: **6.83e-13 K** over the
  two reported starts (uniform and nonuniform with equal room means).
- Full-reference cell energy residuals: below **2.64e-11 W** in the report.
- Energy audit partitions air/slab for each room, partition, and facade.
  Internal transfer is antisymmetric and cancels in the total balance.
  The rooms exchange with their shared partition, not via an invented direct
  air-to-air conductance.
- Tests additionally compare an arbitrary nonuniform air/wall/slab history,
  no air streams, and mixed heating/cooling conditions against source.

## Explicit reduction, not parameter fitting

A binary piecewise-constant basis E groups each room's air into one state and
its slab into another, **retaining every wall CV separately**. It reduces
47 states to **27**, using E^T A E and E^T rhs. Initial storage projection is
capacity-weighted. Each original state belongs to exactly one group.

This conserves summed group energy by construction, including boundary and
air-stream power evaluated at the reduced solution. It does not conserve every
original cell equation. Reported group/total energy residuals stay below
1.8e-10 W while the maximum cell residual is about **31 W**. In the uniform
start, room-temperature errors relative to the full reference are **0.01233 K**
and **0.00593 K** after the one step. The nonuniform start is mapped to the
same reduced initial state but has a different full solution.

Thus conservation is necessary, not a certificate of forecast accuracy. Even
retaining all wall cells does not make the room-mean reduction exact. Collapsing
air modes also erases the distinction between a diffuser injection and a
uniform injection inside each room. A dedicated test demonstrates that replacing
the full model's sparse air map with a uniform map changes its transient while
preserving the sum of the application weights.

Do not compare the 0.012 K synthetic error to archive prediction errors: scale,
states, excitation, known parameters and initial-state availability differ.
No leaderboard or building-level control result follows from this test.

## Reproduction and next gate

```sh
./.venv/bin/python -B -m unittest tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
./.venv/bin/python -B tools/thermal_grid_reference.py
```

Combined suite: **100 tests pass**. Simulator and champion are unchanged.

Next gate is to compare a nested set of reduction bases on the same known
system, retaining diffuser/near-wall modes as needed, with exact energy and
multi-step input-response checks. Only after fixing a reduction structure
should unknown building parameters and latent-state initialization be addressed
from allowed observations/metadata. Hidden material fields and private states
in this reference must never become controller inputs.

Follow-up: `tools/thermal_reduction_notes.md` compares four nested bases on
four 288-step synthetic free responses, including both mean and peak errors.
The expanded test suite passes 105 tests. This does not establish building
prediction or policy performance.
