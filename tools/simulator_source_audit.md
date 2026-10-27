# Simulator source audit: command → field → observation

Scope: the **current local source** used by generated flexible-floor-plan
scenarios, not every alternate simulator class and not a reconstruction of the
code version that produced an old archive. No environment rollouts, parameter
fitting or controller changes were performed for this audit. A sample generated
GIN was read to confirm wiring, not to supply building-specific values to a
policy. The directory has no Git history; archive metadata contains no source
hash. Therefore source/archive version identity remains unverified.

## 1. Active call chain (do not substitute the base simulator implementation)

- `smart_buildings/evaluate.py:469–474`: parse the scenario GIN and construct
  `HybridActionEnvironment`.
- `smart_buildings/smart_control/utils/scenario_generator.py:1028–1058`:
  generated scenarios bind `SimulatorBuilding` to
  `SimulatorFlexibleGeometries`, using `FloorPlanBasedBuilding`.
- `smart_buildings/smart_control/environment/hybrid_action_environment.py:132–194`:
  merge continuous/discrete commands; discrete 0/1 maps to normalized -1/+1.
- `smart_buildings/smart_control/environment/environment.py:1681–1726`:
  normalize to native actions → `request_action` → `wait_time` → observations
  → reward.
- `smart_buildings/smart_control/simulator/simulator_building.py:224–288`:
  set device actions, then `wait_time()` calls `execute_step_sim()`.
- `smart_buildings/smart_control/simulator/simulator_flexible_floor_plan.py:280–547`:
  the actual thermal/HVAC ordering. Its `setup_step_sim` intentionally does
  nothing (`:266–274`). The base `simulator.py:153–216` has a different order
  and is **not** the executed flexible-building algorithm.

## 2. Exact ordering in one flexible-building step

At start timestamp s, let the building field be X_s.

1. Read zone field averages; assign each VAV's area-weighted sensing
   temperature and call `update_settings` (`:285–319`). Commands override the
   local thermostat, independently for damper and valve (`vav.py:454–496`).
2. Form area-weighted return temperature per AHU, get outdoor temperature at s,
   calculate mixed and actual supply air temperatures (`:321–348`).
3. Reset water/air demand, sum all valve factors, determine an AHU-wide airflow
   reduction ratio before evaluating any zone (`:350–380`).
4. Calculate reheat duty from **current water**, and air-side conductance
   rho_air cp_air flow. Accumulate reheat by served-zone area (`:382–436`).
5. Add lighting/equipment internal gains by zone area and comfort flag, then
   actual per-zone occupant sensible gains (`:438–481`).
6. Assign accumulated fixed power to each zone's CVs (`:483–485`), then solve
   all air/wall fields and interior mass implicitly (`:487–493`).
7. Compute the solved air-stream power for diagnostics and overwrite `input_q`
   with total power (`:495–513`). This is not a second thermal solve.
8. Shuffle air-cell temperatures **within rooms** (`:515–516`), compute return
   water, guard the field against divergence and increment the clock (`:518–547`).

The VAV sensor is not refreshed after the solve. Thus for t > 0:
`obs[t].zone_temperature` = the relevant mean of X_(t-1), while the physical
field is X_t and actuators reflect u_(t-1). The u_t thermal response first
appears in `obs[t+2]`. Reset primes the sensors, so the first two temperature
observations duplicate X_0 (`:230–251`). Earlier pending-action bookkeeping is
consistent with this source ordering.

Weather observations query the new observation timestamp
(`air_handler.py:176–185`), not the preceding solve's timestamp. This confirms
using preceding weather for pending forcing. Water reads have side effects:
`hot_water_system.py:200–205` forwards its last observation timestamp;
`boiler.py:124–164` adjusts temperature when read. Device timestamps are set
in `smart_device.py:176–200,212–241`. Do not replace this with a free-running
water ODE or move the next reported water temperature into pending physics.

## 3. Actual thermal state and equations

The model is a **2-D spatial control-volume network**, not one air temperature
and one wall temperature per room. The grid extrudes each cell by floor height.
Air cells, interior partitions and facade cells have their own capacity and
conductivity. Occupied cells may carry an additional slab/furnishings mass
node (`building.py:876–929`). Walls already store heat independently of that
node. Room averages are ordinary means over room CVs (`:1069–1095`).

For neighboring CVs a,b, shared conductance is

    1/G_ab = dx_a/(2 k_a A_face) + dx_b/(2 k_b A_face).

Exterior film conductance is

    H = A_face / (dx/(2k) + 1/h).

The half-cell geometries matter, and the exposed faces are sides of the floor
plan, not roof/floor surfaces. See `finite_difference_solver.py:278–328,
345–442,547–625`. Exterior cells are eliminated/pinned to ambient, while
interior partitions and facade cells remain solved unknowns (`:752–777`).

For each slab node, using footprint A and floor height z:

    C_m = rho_m cp_m A z;    G_m = k_m A/z
    M_next = (C_m/dt M_previous + G_m T_next)/(C_m/dt + G_m).

These are storage-only nodes: no direct outdoor or mass-to-mass path
(`finite_difference_solver.py:445–483,779–813`). They are initialized from
their own air-cell temperatures at an episode reset (`:841–856`). This does
not imply that wall temperatures equal zone air, or that mass at an arbitrary
archived workday starts within a fixed offset from zone air.

With D=C_air_or_wall/dt, L the grid conductance matrix, H exterior terms,
B=G_m(C_m/dt)/(G_m+C_m/dt), P air-power application weights, S sensor weights,
and g=rho_air cp_air flow, the solved field satisfies, after exterior elimination:

    [D + L + diag(H+B) + P diag(g) S^T] T_next
      = D T_previous + q_fixed + boundary_forcing
        + B M_previous + P diag(g) T_supply.

The implementation eliminates slab nodes and uses a sparse factorization plus
a low-rank Woodbury air-side update (`:893–906,968–1025,1027–1103`). Backward
Euler and implicit supply-minus-end-temperature are already correct choices
in our isolated integrator. The major mismatch is state/forcing reduction,
not evidence that we should switch back to explicit Euler. The full coupled
matrix need not be symmetric when application and sensing weights differ;
symmetry claims about the base conduction matrix do not extend automatically
to the air-side update.

## 4. Important spatial asymmetry: air is not applied like reheat

Read executable assignments rather than relying on docstrings:

- `SimulatorFlexibleGeometries._couple_air_side`, `:179–217`: each VAV's P
  column uses `building.diffusers * zone_area / served_area`. The S column
  averages every served CV. Diffusers are sparse (`building.py:310–366`).
- `FloorPlanBasedBuilding.apply_thermal_power_zone`, `building.py:1097–1123`:
  fixed watts are **uniform across all room cells**, NOT diffuser-weighted,
  despite its older docstring. Reheat, internal equipment and occupants all
  pass through this method. It assigns, not adds, hence sources are summed
  before the call.
- `simulator_flexible_floor_plan.py:495–513` rewrites diagnostic `input_q`
  uniformly after solving. That diagnostic array therefore is not the exact
  spatial air-stream RHS that was applied inside the solve.

Our zone ODE places air and reheat in the same well-mixed air state. Equal
total watts do not guarantee equal transients in the spatial system, especially
near walls. Even two fields with identical zone mean and wall temperature can
have different next zone means because boundary-facing air differs. Hence
an exact Markov model on just zone means and one passive mass per zone does
not follow from the simulator equations.

Convection is a permutation of air temperatures within a room, after the
solve (`stochastic_convection_simulator.py:28–83,130–156`). It preserves that
room's mean immediately, but changes which cells touch walls, diffusers and
fixed slab nodes on the next step. Slab temperatures are not shuffled with air.

## 5. Equipment and heat sources

Existing equipment reconstruction has the right broad form:

- Flow: damper × design flow × sqrt(pressure/design pressure), epsilon floor,
  then the common AHU capacity ratio (`vav.py:254–290`; flexible simulator
  `:368–380`). Actual pressure and recirculation are clamped by the AHU
  (`air_handler.py:199–243`). These limits are not all identified at reset.
- Supply: mixed return/outdoor air, cooling-only cap when running; the reported
  AHU supply-air sensor returns the **setpoint**, not discharge temperature
  (`air_handler.py:224–225,257–289`).
- Water: total Kv times sqrt(dP/(1+Kv_total^2 R_header))/3600 m3/s, shared by
  valve factor (`hot_water_system.py:294–326`; `vav.py:308–314`).
- Coil: water/air capacity-rate effectiveness capped at one, plus a hot
  discharge-temperature cap; heat can be negative when water is colder
  (`vav.py:316–395`). Reheat is fixed watts during the solve; the air-stream
  term uses the solved sensor mean.
- Background load: `gain_W_per_m2 * zone_area * comfort_fraction`; people add
  `per_person_W * per_zone_occupancy` separately (flexible simulator `:438–481`).
  Our fitted comfort-only scalar load omits this independent per-zone signal.

No explicit solar-radiation, infiltration, roof/slab-to-ground, humidity or
radiant exchange forcing appears in this executed thermal path. Facade
conductivity can stand for an effective opaque/glazing blend, but there is no
independent window solar term. Do not add those mechanisms merely because
they are common in real buildings. Weather replay supplies interpolated
temperature and a constant film coefficient (`weather_controller.py:273–328`).

## 6. Occupancy and reward timestamps are not the zone sensor timestamp

`SimulatorBuilding` wires occupancy into thermal gains
(`simulator_building.py:69–76`). Thermal gains use [s,s+dt). After incrementing
the clock, the total-occupancy observation sums [s,s+dt), hardcoded as the
preceding five minutes (`:320–329`). It is **building total**, not per-zone
occupancy. Normalization is `(N-c)/(c+1)`
(`environment.py:1348–1351`); `create_building_meta_data` does not export c.
Do not assume a positive normalized value universally means occupied. The
current constructor default c is zero, and setting an unrelated underscored
attribute in `evaluate.py:951` does not change the public attribute read by
the current environment; historical archives may differ.

Reward reads **post-solve physical zone averages**, not the latched VAV sensor
(`simulator_flexible_floor_plan.py:686–705`). Its occupancy and setpoint queries
use the new clock and [new_clock,new_clock+dt) (`:657–684,830–835`). Thus reward
and reported temperature are not interchangeable supervised targets. The
occupancy model is stochastic and caches interval queries
(`randomized_arrival_departure_occupancy.py:254–324`); do not invoke it in a
participant controller or reproduce its RNG to obtain future zone occupancy.

Energy/carbon billing remains outside dT/dt. Cooling COP, fan/pump billing and
boiler/tank losses do not become extra room heat in this thermal RHS.

## 7. Corrections to our interpretation of previous failures

Confirmed differences from `smart_buildings/baselines/zone_thermal.py`:

1. Missing dynamic facade/partition states and neighboring-zone paths.
2. Missing within-zone temperature modes and air/fixed-power spatial asymmetry.
3. One latent node is asked to represent both passive slab and missing walls,
   while having no outdoor or neighboring-zone heat path.
4. A fixed mass/air capacity ratio of 20 and a +/-3 K workday initial offset
   are model priors, not source identities. Material ratios and history vary.
5. Comfort-only fitted load does not represent independent occupant heat.
6. Conditional training feeds recorded supply reconstructed from recorded
   return air; deployment rolls return air from its own predictions.

**Interpretation correction:** leaving the passive envelope rejects the fitted
state as the *isolated slab node in our chosen topology*. It does not prove
the simulator is nonphysical or that every effective latent state must obey
that slab envelope. Actual facade/partition nodes exchange with outdoors and
neighbors and need different constraints. Large inferred temperatures also
become ill-conditioned at small fitted coupling. Prior oracle-forcing
experiments did not remove any of these structural/identification errors.

These are source-confirmed mismatches, not an attribution of a measured
fraction of forecast error to each mechanism. Parameter errors and source
version provenance are still unresolved. Do not tune another ODE before
establishing a matched reference and defining which states are being reduced.

## 8. Access boundary and next gate

Participant-facing metadata is assembled in `evaluate.py:605–654`: field and
action layouts, floor/zone maps, AHU assignment, schedule and normalization.
It does **not** supply conductivity/density/capacity fields, hidden slab states,
diffuser arrays, floor height/cell scale, weather replay or per-zone occupant
trajectories. Public source rules may inform model structure; actual material
values and hidden states must not be read from `env`, GIN or scenario identity
inside a transferable controller. Geometry scale and materials need explicit
priors or causal identification, not silent copying from training configs.

Next gate before further fitting:

1. Match the spatial equations on a tiny transparent synthetic grid with known
   parameters, retaining separate air-application and fixed-power maps.
2. Check exact aggregation of air/slab/wall energies and cross-zone flux; show
   explicitly which approximations a reduced model makes.
3. Keep source hashes with future bounded archive runs so version identity is
   testable. Old unversioned archives remain unverified rather than assumed
   compatible.
4. Only then test a reduced network on held-out data with frozen parameters
   and the correct pending/new-action delay. No MPC integration yet.

Added `tools/test_simulator_source_contract.py`: executable synthetic checks of
the real air-side registration, uniform fixed-power assignment, slab formulas,
and a three-cell counterexample to closure on zone means. These are unit
fixtures, not environment runs. The combined modular/greybox suite passes
95 tests. Existing controllers, simulator source and archive reports unchanged.

Follow-up: `tools/thermal_grid_reference_notes.md` records the independent
47-state synthetic reference/source match and a conservative but inexact
27-state reduction. The expanded combined suite passes 100 tests.
