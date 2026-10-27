# Nested spatial reductions: bounded known-system response

Tool: `tools/thermal_reduction_check.py`.
Report: `smart_buildings/results/thermal_reduction_nested_v1.json`.

## Fixed protocol

Four predeclared input cases, each **288 steps at 300 seconds (24 hours)**,
on the previously source-checked 47-state synthetic two-room system. No
environment, scenario configuration, archived data, fitting, policy or score.
This is dense algebraic system response, not a simulator screen; it does not
invoke `parallel_eval.py` or the simulator step loop. No earlier warm-up.

The reference and each reduction propagate their **own** state; no future
temperature or reference-state correction is provided. The known initial state
is capacity-projected into each reduced partition. Uniform starts are shared
exactly; the mixed case's explicitly nonuniform history loses spatial modes
under reduction. All physical coefficients are the same, without tuning.

Inputs: passive cooling to a held ambient; a 12-step fixed-power heating pulse
in room 1; a 12-step diffuser-air cooling pulse in room 1; and a mixed case
with deterministic weather, power and airflow changes. Pulse magnitudes and
switching times are explicit fixture choices, not building calibration. Stochastic
air shuffling is absent in this linear reference. The report records SHA-256
of the check, fixture/reference and local source solver, not old archive provenance.

## Nested partitions

Every wall CV remains explicit in all variants; room air and slab partitions
are refined together. All bases are binary, cover every state exactly once,
and are nested. Galerkin aggregation preserves partition-summed energy.

| Name | States | Air/slab partition per room |
| --- | ---: | --- |
| room | 27 | one air and one slab state |
| diffuser | 31 | diffuser cell versus remaining cells |
| boundary_side | 35 | diffuser, remaining partition-side, remaining facade-side |
| full | 47 | all cells independently |

The tiny rooms have no deep core: this experiment tests partition-facing versus
facade-facing modes, not a large-room near-wall/core split. Keeping every wall
CV means this is not yet a scalable building-level reduction.

## Results

Whole-response mean absolute zone error in K (both rooms, all 288 steps):

| Input | 27 states | 31 states | 35 states |
| --- | ---: | ---: | ---: |
| passive | 0.0175793 | 0.0145550 | 0.0030585 |
| fixed-power pulse | 0.0002334 | 0.0002028 | 0.0000810 |
| air pulse | 0.0014249 | 0.0005576 | 0.0002308 |
| mixed | 0.0115646 | 0.0087812 | 0.0028052 |

Maximum zone errors (K):

| Input | 27 states | 31 states | 35 states |
| --- | ---: | ---: | ---: |
| passive | 0.0236992 | 0.0199100 | 0.0048360 |
| fixed-power pulse | 0.0027077 | 0.0020342 | 0.0016725 |
| air pulse | 0.0203777 | 0.0048744 | 0.0063351 |
| mixed | 0.0252196 | 0.0148562 | 0.0046928 |

The full basis agrees with the unprojected reference to 3.8e-14 K. Maximum
total energy residual across all cases/bases is 3.15e-10 W; maximum partition
energy residual is 7.14e-11 W. Cell residuals need not vanish for a reduction.

## Interpretation and limits

Diffuser and wall-side states materially improve this known-system mean
response without parameter fitting. In the mixed fixture, 35-state MAE is
about 76% below the 27-state variant. **Refinement is not monotonically better
in maximum zone error**: air-pulse peak error increases from 0.00487 to 0.00634 K
between 31 and 35 states even though overall MAE decreases. Nested conservative
projection alone gives no monotone output-error guarantee.

Whole-day pulse MAEs are diluted by the long unforced recovery. Peak errors
and checkpoints 1/2/6/12/72/288 are retained in the report; do not claim a
pulse prediction gate based on whole-day MAE alone. There is no persistence
comparison or real-building prediction/control gate in this experiment.

This supports retaining distinct air injection and boundary-facing modes in
the next model structure, not copying fixture values into a controller.
Remaining gates: test a larger synthetic geometry with genuine core cells and
wall aggregation, include source convection/state ordering where relevant,
and establish which reduced parameters and initial states can be inferred
from participant observations. No claim of leaderboard improvement.

## Verification

```sh
./.venv/bin/python -B -m unittest tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
./.venv/bin/python -B tools/thermal_reduction_check.py --steps 288
```

105 tests passed. The budget is rejected before fixture creation for lengths
outside 1..288. Tests check nested complete partitions, energy-preserving
initial projection, isolated diffuser states, free-response identity at full
resolution, and conservation. Simulator and champion unchanged.

Follow-up: `tools/thermal_spatial_notes.md` separates air/slab and wall
aggregation on a 164-state fixture with true core cells. Coarse walls remain
a major error source even with exact air/slab states. Combined suite: 110 tests.
