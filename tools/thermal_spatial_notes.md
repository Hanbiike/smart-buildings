# Larger synthetic rooms: isolate air reduction from wall reduction

Tool: `tools/thermal_spatial_check.py`.
Latest report: `smart_buildings/results/thermal_spatial_factorial_v2.json`.
V1 contains the initial 2x3 comparison; V2 adds an exact-air/slab control arm
to isolate the effect of wall aggregation. Earlier artifacts are retained.

## Protocol

The existing independently assembled rectangular reference was generalized,
without changing the original default 47-state fixture. The new fixture has
two 6x5-cell rooms, **30 air cells per room**, including **12 genuine core
cells** not immediately adjacent to facade or partition. With 44 wall cells
and 60 slab nodes the full system has **164 states**.

This remains an invented algebra fixture, not a realistic office: cell size
is 0.2 m, so each room is only 1.2 m2. Its results are about reduction error
in this known system, not real-room prediction accuracy or legal simulator
scenario geometry. There is no environment construction or area-floor bypass
on a competition building.

Same four explicit inputs as the prior reduction audit: passive outdoor
cooling, a fixed-heat pulse, an air-cooling pulse, and mixed inputs. Each case
is **288 steps / 24 hours**, no preceding warm-up. Source coefficients and
known initial history are identical across arms; no parameter fitting or
reference-state assimilation. All models evolve their own state. Initial
projection preserves each group's heat storage. Convection shuffling is absent.

The generalized dense reference agrees with the source solver on each case's
first step to **1.081e-12 K**, including a nonuniform latent history. Tests
also check arbitrary nonuniform histories, not just near-equilibrium states.
The source solver is not run as a trajectory: it is a single-step unit oracle.
Dense projected matrices are LU-factorized once per conductance tuple; caching
changes only computational cost, not the equations.

## Factorial structure

Air/slab partitions:

- `room`: one air and one slab temperature per room.
- `spatial`: isolate diffuser; partition other air into core, partition-facing,
  facade-facing and cells touching both. Slab partitions follow air partitions.
- `full`: every air and slab cell retained. Added after the first comparison
  as a diagnostic control, not a fitted candidate.

Wall partitions:

- `coarse`: facade top/bottom split by served-room side, facade/partition
  junctions, left/right sides, and shared partition: 9 groups.
- `segments`: split each nontrivial coarse wall group into two: 16 groups.
- `full`: every wall CV: 44 states.

Both axes are independently nested; every state belongs to one group.
All three variants preserve aggregate capacities/conductances by Galerkin
projection. Merging cells is an isothermal constraint, not an exact replacement
for conduction within a wall.

## Results

Mixed-input whole-response zone MAE, K (state count in parentheses):

| Air/slab | Coarse walls | Segmented walls | Full walls |
| --- | ---: | ---: | ---: |
| Room means | 0.195469 (13) | 0.178732 (20) | 0.053074 (48) |
| Spatial | 0.174972 (29) | 0.157526 (36) | 0.030773 (64) |
| Full | 0.151000 (129) | 0.125530 (136) | ~0 (164) |

Passive-cooling zone MAE, K:

| Air/slab | Coarse walls | Segmented walls | Full walls |
| --- | ---: | ---: | ---: |
| Room means | 0.271546 | 0.247876 | 0.069814 |
| Spatial | 0.245406 | 0.220635 | 0.041371 |
| Full | 0.213672 | 0.177677 | ~0 |

Thus coarse wall aggregation alone introduces substantial passive/mixed error
even with exact air and slab states. More states by itself is not the answer:
48 states with full walls outperform 129 states with coarse walls in these
two cases. This does not establish a universal ranking across geometries and
inputs. For example, during the first 12 air-pulse steps full-air/coarse-wall
MAE is 0.00178 K, below spatial-air/full-wall MAE 0.00647 K.

Pulse whole-day MAEs remain small partly because of long unforced recovery;
the report separately records first-12-step and subsequent MAEs, maximum zone
error, final error, and maximum wall error. Spatial/full-wall peak errors are
0.00366 K for the heat pulse, 0.01261 K for the air pulse and 0.04403 K for
mixed inputs. No persistence/control gate is established by these numbers.

All arms conserve partition/total energy: largest reported total residual is
1.033e-9 W and largest group residual is 6.35e-10 W. The full-reference maximum
cell residual is 1.287e-10 W. Conservation does not protect against wrong
spatial heat paths under an isothermal wall constraint. This experiment does
not assign the old archive model's errors to the same mechanisms quantitatively.

## Next decision

Do not use a single coarse wall temperature per zone or facade as the next
controller model merely because its balance closes. Before further fitting,
check whether wall reduction preserves steady-state heat-transfer gains and
the dynamic response between room/ambient ports; use retained wall modes or
proper condensation instead of arbitrary isothermal merging. Then validate
the structure with source convection/state ordering and address causal
parameter/state identification. A source-aware structural model is not yet a
transferable controller: hidden materials, private slab states and known
synthetic initial history are still unavailable online.

## Reproduction

```sh
./.venv/bin/python -B tools/thermal_spatial_check.py --steps 288
./.venv/bin/python -B -m unittest tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```

**110 tests pass.** Tests cover complete/nested bases on both axes, expected
state/core counts, source agreement, cached versus dense projected solutions,
independent recurrence, initial energy projection, conservation and budget
rejection before fixture construction. Reports include source hashes.
Simulator and champion unchanged; no environment screens or submissions.

Follow-up: `tools/thermal_wall_modes_notes.md` tests DC-exact static wall
constraints with slow dynamic modes. DC is recovered, but short-input response
is not; the method is not promoted to control. Combined suite: 115 tests.
