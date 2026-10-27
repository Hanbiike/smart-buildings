# DC-preserving wall constraints and slow-mode enrichment

Follow-up: `tools/thermal_port_modes_notes.md` replaces slow-mode ordering with
input-response wall enrichment. It preserves DC and substantially improves
fast response on the same known fixture, including off-design frequency checks.
This is not yet a compact transferable building model. Combined suite: 120 tests.

Tool: `tools/thermal_wall_modes_check.py`.
Report: `smart_buildings/results/thermal_wall_modes_v1.json`.

## Fixed known-system experiment

Same 164-state synthetic two-room fixture and four input protocols as the
spatial audit, capped at **288 steps / 24 hours per response**, with no warm-up.
No scenario, environment, archive fitting or controller. All 120 air/slab
states are retained as ports to isolate wall reduction, so this is not yet a
small building model. Convection is absent. Coefficients and initial history
are known oracle fixture data, not participant-observable estimates.

For K=L+diag(H), partition the 44 wall states w from air/slab ports p. The
static wall constraint is

    w = -K_ww^-1 K_wp p + K_ww^-1 H_w T_ambient.

The first term defines a static wall lift for every retained port. The ambient
wall-response vector is included as an additional **free dynamic coordinate**,
not fixed to current weather. With it the basis spans the all-ones temperature
vector, permitting Galerkin conservation of total energy. This avoids silently
dropping a weather derivative when substituting a time-varying affine offset.

Enrichment adds 0/2/4/8/16/32/43 independent directions in slow-to-fast order
from K_ww phi=lambda C_ww phi. The wall ambient direction is inserted first;
weighted reorthogonalization skips dependent directions. Full trial bases are
C-orthonormalized without changing their spans. The resulting capacities are
handled as general matrices, not assumed diagonal due to physical grouping.
All requested spaces are nested; 43 additional directions recover all 44 wall
coordinates and the full 164-state system.

The initial state is capacity-projected. Uniform starts are reproduced to
roundoff; mixed-case nonuniform projection can change initial zone means
(0.01552 K maximum with 8 extra modes). No later reference correction occurs.

## Steady-state and frequency checks

Static transfer gains from two uniform-power ports, two diffuser-power ports
and ambient to both zone means are checked with airflow feedback off and on.
Modal variants preserve DC gains to **1.24e-13 maximum relative column error**.
The prior isothermal coarse/segmented wall bases have DC errors up to **4.43% /
3.62%**, respectively, despite conserving aggregate energy.

This is exact static response preservation for the **known** fixture matrices,
not identification of an unknown building's U-value or a prediction score.

Discrete backward-Euler transfer is also checked at periods 600/3600/21600/
86400 seconds, using C*(1-exp(-i omega dt))/dt+K (including the fixed air
feedback term). Each input column is normalized separately because watts and
ambient kelvin are different units. Errors measure complex response, not just
amplitude. The 600-second period is the discrete Nyquist period at dt=300 s.

With 8 added wall modes, maximum relative column error without air feedback
is **79.0% at 10 minutes, 72.7% at one hour, 35.5% at six hours, and 5.79% at
24 hours**. Perfect DC is therefore not useful-dynamics validation. Even 32
added modes leave 20.3% error at 10 minutes and 11.8% at one hour. Full
enrichment recovers frequency response to roundoff.

## Free-response results

Whole-response zone MAE (K):

| Wall representation | Total states | Passive | Heat pulse | Air pulse | Mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static constraints + ambient only | 121 | 0.386727 | 0.011181 | 0.016999 | 0.186836 |
| +4 slow modes | 125 | 0.053336 | 0.010355 | 0.016099 | 0.044064 |
| +8 slow modes | 129 | 0.045918 | 0.007655 | 0.011616 | 0.036152 |
| +16 slow modes | 137 | 0.021159 | 0.001088 | 0.001425 | 0.013113 |
| +32 slow modes | 153 | 0.006498 | 0.000427 | 0.000658 | 0.004197 |
| All wall directions | 164 | ~0 | ~0 | ~0 | ~0 |
| Isothermal coarse walls | 129 | 0.213672 | 0.001152 | 0.001536 | 0.151000 |
| Isothermal segmented walls | 136 | 0.177677 | 0.000981 | 0.001402 | 0.125530 |

At equal 129-state size, slow modes improve passive/mixed MAE but substantially
**worsen short pulses**. Heat-pulse peak error rises from 0.00462 K (coarse
walls) to 0.22693 K (8 slow modes); air-pulse peak rises from 0.00633 to
0.34559 K. First-12-step pulse MAEs are also reported, so long recovery does
not hide the transient failure. These errors persist from exactly uniform
initial conditions and cannot all be attributed to initial projection.

Maximum absolute total-energy residual across all responses is below
**1.50e-8 W**. The larger residual than the binary-partition audit reflects
floating-point projection/cancellation, not a useful accuracy guarantee.

## Decision

Do not promote this slow-mode variant to control. It fixes the static heat
paths but is not a low-order accurate model at command-relevant time scales.
This is a negative result for this unweighted slow-eigenmode choice, not a
proof that wall reduction is impossible. If reduction continues, target the
measured input/output ports and 10–60-minute response (for example with
moment/transfer-informed directions), and retain DC and conservation checks.
Do not choose a candidate by mixed-case day-average MAE alone or keep nearly
all wall modes and call that a small model. Causal identification of material
parameters and states remains a separate, unsolved gate.

## Verification

`tools/test_thermal_wall_modes_check.py` checks C-orthonormality, inclusion of
constant temperature, nesting, DC response, full-enrichment frequency recovery,
general projected-capacity dynamics against dense Galerkin solves, full-state
recovery, energy residuals and early budget rejection. The combined modular,
spatial and greybox suite passes **115 tests**. Simulator/champion unchanged.

```sh
./.venv/bin/python -B tools/thermal_wall_modes_check.py --steps 288
./.venv/bin/python -B -m unittest tools.test_thermal_wall_modes_check tools.test_thermal_spatial_check tools.test_thermal_reduction_check tools.test_thermal_grid_reference tools.test_simulator_source_contract tools.test_thermal_forcing_audit tools.test_zone_thermal tools.test_modular_thermal_check tools.test_equipment_balance tools.test_equipment_observer tools.test_equipment_observer_check tools.test_causal_greybox tools.test_thermal_greybox tools.test_greybox_check tools.test_pooled_greybox
```
