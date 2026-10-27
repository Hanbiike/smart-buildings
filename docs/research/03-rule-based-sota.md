# Rule-Based HVAC State of the Art (Track 3)

Exhaustive survey of advanced rule-based HVAC sequences and their mapping to
our simulator's observation/action space. All sources below were opened and
read (full bodies, not snippets); sources that could not be opened are listed
in §7 Unresolved and are NOT used as evidence.

Scope notes:

- Commercial binary firmware (Trane Tracer/UC controllers, JCI Metasys/CCT
  application binaries, Siemens Desigo binaries, Daikin MicroTech firmware) is
  out of scope as an inspectable artifact: the shipped binaries are proprietary
  and were not reverse-engineered. What IS covered for each manufacturer is
  their published application sequences, manuals, and help files (open
  documents), which fully describe the control logic.
- Open-source coverage is limited to equipment/BMS/thermostat firmware and
  control stacks whose source or documentation was actually opened
  (Modelica Buildings/CDL, VOLTTRON, Home Assistant, ESPHome, Sinergym,
  CityLearn, BOPTEST).

## 0. Our simulator contract (mapping baseline)

From `smart_buildings/baselines/transfer_policy.py` (interface contract) and
verified simulator facts:

Actions (hybrid, normalized continuous in [-1,1]):

- Per AHU: supply-air temperature (SAT) setpoint, static-pressure setpoint,
  recirculation fraction (capped at 0.70 by the sim); discrete AHU on/off.
- Per VAV box (per zone): damper command, reheat-valve command.
- Hot-water system: supply-water temperature, differential pressure; discrete
  pump on/off.
- Plant facts: AHU has NO heating coil (cooling only) — `supply = min(mixed,
  setpoint)`; SAT action is inert whenever OAT < -6.8 °C (31.5% of scored hours
  in Fairbanks strips); VAV discharge capped at 308.15 K; dampers floored at
  10%; chiller COP constant 3.5; boiler constant efficiency.

Observations (flat vector, indexed by
`building_meta_data["observation_field_names"]`):

- Per zone: air temperature (normalized), heating + cooling setpoints (raw K),
  damper position (as-taken, i.e. post-floor), reheat-valve position,
  airflow setpoint.
- Per AHU: SAT + setpoint, static pressure + setpoint, airflow + setpoint,
  outside-air temperature (raw K), recirculation, cooling-request count,
  fan status.
- Global: hot-water system state, BUILDING-level occupancy (no per-zone
  occupancy), comfort-schedule flags, cyclic hod/dow encodings.
- NOT present: per-zone occupancy, CO2, humidity/enthalpy, return-air
  temperature, per-zone discharge-air temperature (post-reheat), per-zone
  measured airflow (only setpoints + damper/valve positions).

Reward: (comfort, energy, carbon) = (0.6, 0.2, 0.2); comfort slope ~0.18/K
dominates; TOU peak/night 1.25x (load shifting dead); penalty unbounded
quadratic beyond 5 K.

Mapping verdicts used below: CAN (fully implementable), PARTIAL (implementable
with listed caveats/substitutions), CANNOT (blocked by missing sensor,
actuator, or plant physics, with reason).

## 1. ASHRAE Guideline 36 in depth

Primary evidence: the open-source reference implementation in the Modelica
Buildings Library (`Buildings.Controls.OBC.ASHRAE.G36`, "May 2020" =
G36-2018+addenda, pre-2021; file-level sources listed in §6), plus the
generated LBL documentation pages for the Generic (trim-and-respond,
economizer high limits), TerminalUnits.Reheat, and AHUs.MultiZone.SetPoints
packages, plus the Trane 2023 Engineer Summit G36 deck (concept-level
confirmation) and the JCI/Siemens documents where noted.

### 1.1 Trim-and-respond (T&R) — the core reset engine

Mechanism (G36 §5.1.14.3/5.1.14.4 per `Generic/TrimAndRespond.mo`):

- One T&R loop per reset setpoint (duct static pressure, SAT TMax, hot-water
  supply temp, chilled-water supply temp, plant staging delays).
- Every sample period T (default 120 s), AFTER a startup delay Td
  (default 600 s) with the device proven on (before that / when off, output =
  SP0/init):
  - always trim: `SP += SPtrim` (toward the energy-efficient direction);
  - if requests R > ignored I: additionally respond
    `SP += resAmo*(R-I)`, capped at `|SPres_max|`.
- Optional hold input `uHol` + `dtHol` freezes the loop during plant staging
  and resumes from the held value (no re-init).
- Request suppression after setpoint changes (§5.1.20, `TimeSuppression.mo`):
  cooling/heating requests suppressed for 9 min/°C of (controlled-variable
  vs new-setpoint) difference, capped at 30 min; alarms 18 min/°C capped at
  120 min. Prevents setpoint-step-induced request storms.
- Importance Multiplier (IM): a Guideline-level concept (Trane deck: "Multiply
  requests by Importance Multiplier"; IM=0 excludes a zone, 1 default, >1
  critical). NOTE: the Modelica reference implementation does NOT implement
  IM — `TrimAndRespond` sums integer `numOfReq` directly (verified by grep:
  no "importance"/"rogue" in the G36 sources). Rogue-zone handling in the
  reference code is limited to the "ignore I requests" mechanism. IM/rogue
  exclusion is done at the BAS operator level in practice (cf. JCI advice to
  exclude rogue zones, §2.2).

Mapping: CAN. T&R is pure software on top of zone feedback we have
(temperatures, setpoints, damper/valve positions). Our `occ_gated7` already
implements a T&R variant for static pressure (trim down when all dampers
<50%, respond up on persistent starved zones). Deltas vs G36 worth noting:
G36 timesteps are 2 min (our steps are 5 min — rescale trim/respond amounts);
G36 pressure requests use measured zone airflow vs setpoint (we lack per-zone
measured airflow — must keep the damper+temperature proxy); G36 SAT requests
use a cooling-loop signal (we can compute our own PI loop — CAN).

### 1.2 Supply-air temperature (SAT) reset

Mechanism (`AHUs/MultiZone/VAV/SetPoints/SupplyTemperature.mo`, G36 §5.16.2;
PR1 doc §PART5.N.2 agrees):

- Enabled only with supply fan proven on; else output = deadband (no heating,
  minimum economizer).
- Occupied + Setup modes: `TSupSet` reset linearly by OAT from `TSupCoo_min`
  (default 12 °C / 285.15 K) at `TOut_max` (21 °C) and above, up to `TMax` at
  `TOut_min` (16 °C) and below. `TMax` itself is T&R-reset between
  `TSupCoo_min` and `TSupCoo_max` (default 18 °C / 291.15 K; ≤16 °C in humid
  climates; never >18 °C or fan energy offsets cooling savings).
  - T&R params (defaults): SP0 = SPmax, Td = 10 min, T = 2 min, I = 2,
    SPtrim = +0.1 K, SPres = -0.2 K, SPres_max = -0.6 K. Requests = zone
    cooling requests (§1.6).
  - OAT range 16–21 °C chosen to maximize economizer hours; 13–18 °C variant
    minimizes fan energy instead.
- Cooldown mode: `TSupSet = TSupCoo_min`. Warmup + Setback: `TSupSet = 35 °C`
  (308.15 K default `TSupWarUpSetBac`).
- SAT controlled to setpoint by sequencing heating coil (if any) then chilled
  water valves.

Mapping: PARTIAL.

- CAN: OAT-scheduled reset + T&R on zone cooling requests during occupied
  hours; our `zone_pi` lineage already does demand-driven SAT (`DemandSat`
  chain) but NOT the G36 two-stage form (OAT schedule + T&R-bounded TMax) —
  portable as-is.
- CANNOT (cold): SAT action inert below OAT ≈ -6.8 °C (mixed air already below
  min setpoint); G36 assumes a cooling coil that can always meet setpoint.
  In Fairbanks-like strips the whole SAT loop goes dead for ~1/3 of hours —
  any port must gate on OAT (as our code does with `FORCED_SAT` floors).
- CANNOT (heat): no AHU heating coil, so warmup SAT 35 °C is unachievable;
  warmup in our sim = keep fan low + let reheat do the work (our prestart
  already does this).

### 1.3 Static-pressure reset + supply-fan control

Mechanism (`SetPoints/SupplyFan.mo`, G36 §5.16.1):

- Fan runs in Cooldown/Setup/Occupied; also in Setback/Warmup ONLY if perimeter
  VAV-reheat boxes exist (`have_perZonRehBox`).
- Pressure setpoint T&R defaults: SP0 = 120 Pa, SPmin = 25 Pa, SPmax = design
  max, Td = 10 min, T = 2 min, I = 2, SPtrim = -12 Pa, SPres = +15 Pa,
  SPres_max = +32 Pa. Requests = zone static-pressure requests (§1.6).
- Fan speed: PI on duct static pressure (setpoint and measurement normalized
  by design max); gain-schedule for small zone groups by load indicator.
- Older Siemens-published variant (same Taylor source): range 35–325 Pa,
  SP0 = 125 Pa, trim -10 Pa / respond +15 Pa per 2 min, or weighted respond
  +7.5 Pa × requests capped +30 Pa.

Mapping: CAN. Our `occ_gated6/7` already implements occupied-gated T&R on
pressure (trim -0.03, respond +0.25 normalized, cap 0.0, persist 4 steps,
OAT-widened deadband). Remaining G36 deltas: (a) G36 trims ALWAYS and responds
additionally (ours trims only when all dampers closed — close enough);
(b) G36 SPmin is a fixed floor (25 Pa), ours effectively similar via sim's
200 Pa running floor; (c) G36 request generation uses airflow shortfall
(CANNOT per-zone — no sensor) + damper >95% (CAN).

### 1.4 Demand-controlled ventilation (DCV) + ventilation reset

Mechanism (two layers):

1. Zone layer: per-zone DCV from occupancy sensors (occupied-standby pop
   counting) and/or CO2 loops reset zone minimums between area-only and
   area+population rates (G36 §5.6.8/PR1 ventilation logic; JCI VAV app: "CO2
   Sensor interface that automatically resets the flow settings").
2. System layer — ASHRAE 62.1 dynamic reset (`OutsideAirFlow`, G36
   §PART5.N.3.a, Stanke 2010 "Dynamic Reset for Multiple-Zone Systems"):
   11-step computation: per-zone breathing-zone OA (area + population ×
   occupancy) → distribution effectiveness (cool 1.0 / heat 0.8 depending on
   Tdis vs Tzone) → required zone OA (zero if unoccupied mode or window open;
   area-only if occ-sensor says empty) → zone OA fraction (÷ measured box
   flow) → system uncorrected OA → system ventilation efficiency
   `SysVe = 1 + X + max(Z)` → `effMinOA = min(uncorrected/SysVe, design)`.
   The AHU then tracks this minimum OA through damper limits (separate min-OA
   damper with flow station, or with DP, or common OA/econ damper).

Mapping: PARTIAL at system level, CANNOT at zone level.

- CANNOT: no CO2 sensors, no per-zone occupancy, no per-zone measured airflow,
  no discharge-air temp (needed for cool/heat effectiveness branch).
- CAN: system-level OA modulation through the recirculation action using
  building occupancy + schedule + OAT (pseudo-DCV). Our sim pins minimum OA
  via the 0.70 recirc cap + 10% damper floor, so the "close everything"
  extreme is already fenced. Our `pid_plus` economizer already modulates
  recirc on economics, not ventilation — the G36 addition would be raising
  recirculation (cutting OA) during unoccupied/low-building-occupancy hours,
  which is portable.

### 1.5 Optimal start / optimal stop

G36 proper: `ZoneGroups/OperationMode.mo` (G36 §5.4.3–5.4.6) starts
warmup/cooldown from "the zone with the longest calculated warm-up/cool-down
time", capped at `preWarCooTim` = 3 h before scheduled occupancy — but the
Optimal Start computation itself is explicitly "computed in a separate block"
NOT in the reference implementation. So G36 standardizes the interface
(max warm-up/cool-down time per zone), not the algorithm. Two real algorithms:

1. JCI Optimal Start object (SCT Help LIT-12011964 R13.3, pp. 430–435,
   opened as PDF): `OST_minutes = (Constant/100) × ΔT² + Deadtime`, separate
   heating/cooling constants + deadtimes (deadtime = min per 1 °F). Daily
   adaptation: if zone temp at occupancy missed the (setpoint ± Temp
   Differential) window, increase constant (start earlier toward max
   prestart); if made early, decrease (start later). Optimal stop uses the
   SAME equation with ΔT := Temp Differential and Deadtime := 0, gated by
   Maximum Prestop Time. OAT outside design limits → max prestart / no
   prestop, and adaptation skipped.
2. SES practitioner summary (opened): all vendors' variants are learning
   algorithms over (space temp history, setpoints, OAT); the dominant saving
   is avoided OA conditioning + some fan; OA damper MUST stay closed during
   prestart unless economizing; reported example: 20 kCFM AHU, 30% OA,
   2-h window → ~$2k/yr.

Mapping: CAN (both). We have hod/dow, OAT, zone temps, setpoints, schedule
flags, and full actuator control — the JCI quadratic + adaptation ports
directly (also: our steps are 5 min; JCI works in minutes). Our lineage
already has fixed prestart (`PRESTART_H`, prestart-only rates, latch); the
unexploited part is adaptive per-condition timing + optimal stop coasting
(release heating/cooling early while still occupied). Comfort-slope caution:
with 0.6 weight and unbounded quadratic tails, optimal-stop coasting must be
conservative — but the mechanism is implementable.

### 1.6 Zone requests, states, and prioritization

Request rules (`SystemRequests.mo`, G36 §5.6.8 — exact):

- Cooling SAT requests: ΔT = Tzone − TcoolSet > 3 K for 2 min AND
  post-suppression → 3; > 2 K same → 2; cooling loop > 95% → 1 (until < 85%);
  else 0.
- Static-pressure requests: measured flow < 50% of setpoint (setpoint > 0)
  AND damper > 95% for 1 min → 3; < 70% same → 2; damper > 95% → 1
  (until < 85%); else 0.
- Hot-water reset requests (needs DAT sensor): Tdis < TdisSet − 17 K for
  5 min → 3; − 8 K → 2; valve > 95% → 1 (until < 85%); else 0.
- Hot-water/chilled-water PLANT requests: valve > 95% → 1 until < 10%.
- AHU-level plant requests (`PlantRequests.mo`, §5.16.16) mirror the same
  3/2/1 pattern on SAT-vs-setpoint error (3 K/2 K, 2 min) and coil-valve
  position.

Zone states (`ThermalZones/ZoneStates.mo`, §5.3.4 loops in `ControlLoops.mo`):
separate heating/cooling PI loops (P+I mandatory, I small vs P; P-only
forbidden), each enabled when temp crosses its setpoint and disabled when
temp recovers AND output is 0 for 30 s (anti-windup provision required).
State = heating if uHea > uHigh AND uHea − uCoo > uLow (avoids dual-active
confusion); else cooling if uCoo > uHigh; else deadband.

Prioritization: G36 prioritizes via (requests × IM) + ignore-I + suppression
— NOT via worst-zone tracking. There is no max-zone-error term in G36; a
single hot zone generates at most 3 requests, routinely outvoted by I=2
ignore + many comfortable zones. (Our codebase's `PIDWorstZone` is explicitly
non-G36 — justified by our unbounded-worst-zone reward, a real divergence
between the Guideline's comfort model and this competition's.)

Mapping: CAN for the request/state/prioritization software (IM weighting is
trivially portable and unexploited in our lineage — e.g. down-weighting
chronically deviant zones); PARTIAL for the request inputs (flow-based
3/2-request rules CANNOT per-zone; damper/valve/ΔT-based 1-request rules
CAN; DAT-based hot-water rules CANNOT).

### 1.7 AHU / zone-group operating modes

Mechanism (`ZoneGroups/OperationMode.mo`, G36 §5.4.3–5.4.6; per-mode actuator
tables in `SupplyTemperature`, `SupplyFan`, `ActiveAirFlow`):

- Zone-group modes (all zones in a group share one mode; group = zones on one
  system): Occupied (schedule/override/OT timer), Warm-up (optimal-start
  driven, ≤3 h), Cooldown (same), Setback (unoccupied + ≥5 zones — or all if
  fewer — below unocc heat setpoint, or group-average below average unocc
  setpoint; exit when ALL zones 1 K above; freeze variant: any zone < 4 °C →
  setback until all > 7 °C + L3 alarm), Setup (mirror on cooling side),
  Unoccupied (default).
- Per-mode setpoints: SAT = T&R/OAT-reset (occupied/setup), TSupCoo_min
  (cooldown), 35 °C (warmup/setback); fan on in occupied/setup/cooldown
  (+warmup/setback only with perimeter reheat); VAV active airflows per the
  mode table (§1.9); ventilation OA = 0 except occupied (window-open zones
  get 0 always); zone setpoints = occupied triplet in occupied/warmup/
  cooldown, unoccupied triplet otherwise.

Mapping: CAN. Pure mode machine on schedule + zone temps; every output has a
sim actuator except warmup SAT 35 °C (no heating coil — substitute: fan
minimum + reheat-driven warmup, which our prestart already does) and OA = 0
(recirc cap 0.70 forbids full close while running — closest is max recirc;
alternatively AHU off, which our off-hours logic already does).

### 1.8 Economizer + high-limit logic

Mechanism (`Economizers/Controller.mo` + `Subsequences/Enable.mo`, G36
§5.16.2.3/5.16.4–5.16.7; high limits `Generic/AirEconomizerHighLimits.mo`,
G36 §5.1.17):

- Enable/disable: disabled whenever OA conditions exceed the high-limit
  cutoff; disabled IMMEDIATELY (no delay) on supply-fan-off or any
  freeze-protection stage > 0. On disable: return damper forced full open
  for `retDamFulOpeTim`, then released to min-OA position; OA damper driven
  to min-OA limit after `disDel` — sequencing avoids pressure shocks.
- High-limit cutoffs (ASHRAE 90.1-2016 default table): fixed dry-bulb 24/21/
  18 °C by climate-zone group; differential dry-bulb vs return air;
  fixed enthalpy + fixed dry-bulb (24 °C or 66 kJ/kg); differential enthalpy
  + fixed dry-bulb; fixed+differential dry-bulb combos; Title 24 variants
  shift thresholds by CA zone (21–24 °C fixed; RAT−1/−2/−3 °C differential).
- Modulation: OA/return dampers modulated to track SAT setpoint subject to
  min-OA damper-limit floor (three hardware options: separate min-OA damper
  with airflow station / with DP / common OA+econ damper) and building-
  pressure control variant (relief damper/fan or return-fan tracking/direct).
- Freeze protection stages (§5.16.12, in controller): progressive
  OA-shutoff stages on low mixed/frost temps.

Mapping: PARTIAL.

- CAN: fixed-dry-bulb high limit on OAT (we have OAT); disable-on-fan-off;
  modulate recirc to track SAT target; min-OA floor. Our `pid_plus`
  economizer already does a superset on economics (fan-cost-aware
  proportional mixing, ECON_K_PER_PA = 0.00322 K/Pa from COP 3.5).
- CANNOT: differential-vs-return-air and any enthalpy variant (no return-air
  temp, no humidity sensors); freeze-protection staging beyond fan-off
  (no frost coil temps); pressure-shock damper sequencing (no damper
  dynamics in sim — recirc acts instantly).

### 1.9 VAV terminal dual-maximum logic + active airflows

Mechanism (`DamperValves.mo` §5.6.5 + `ActiveAirFlow.mo` §5.6.4):

- Cooling (uCoo > 0): airflow setpoint mapped from cool-min to cool-max by
  loop; heating coil OFF (unless Tdis < 10 °C floor); if AHU SAT > room temp,
  cooling airflow capped at minimum.
- Deadband: airflow = minimum; coil off (same 10 °C floor).
- Heating (uHea > 0): 0–50% loop resets DAT setpoint from current AHU SAT up
  to (TheatSet + dTDisZonSetMax, default 11 K) at heating-min airflow;
  50–100% (only if Tdis > Troom + 3 K) resets airflow from heat-min to
  heat-max; coil modulated to DAT setpoint; occupied DAT floor 10 °C.
- Damper: inner loop tracks airflow setpoint (pressure-independent) or direct
  mapping. Commissioning overrides force flow to 0/cool-max/min/heat-max.
- Active airflow min/max table by zone-group mode (occupied/cooldown/setup/
  warmup/setback/unoccupied): cool-max = design except 0 in warmup/setback/
  unoccupied; minimums = occupied-min in occupied else 0; heat-min =
  max(design, occ-min) in occupied, design in cooldown, 0 in setup,
  heat-max in warmup, cool-max in setback(!), 0 unoccupied.

Mapping: PARTIAL.

- CAN: the staging IDEA (valve-first-then-airflow) as an output map from our
  own per-zone PI signal — our `zone_pi` currently drives damper ∝ |u| and
  valve ∝ −u simultaneously; G36 says heat-min airflow + DAT ramp first,
  more air only above 50%. Emulating this open-loop is trivially portable
  and unexploited.
- CANNOT: DAT-setpoint tracking and the Tdis-vs-Troom+3 K gate (no DAT
  sensor/actuator); airflow-setpoint inner loop (we command damper %, and the
  sim reports only the resulting damper %, not flow); the 10 °C DAT floor
  (no DAT measurement; closest analog is the sim's 308.15 K discharge cap,
  which is a ceiling not a floor).

### 1.10 Zone temperature control loops

Mechanism (`ControlLoops.mo`, G36 §5.3.4; defaults k=0.1, Ti=120 s both
loops): dual PI loops with the enable/disable + 30-s-zero rule (§1.6),
software-point outputs 0–100%, operator-adjustable P/I.

Mapping: CAN. Our `zone_pi` lineage IS this (per-zone PI to band center,
KP=0.1/KI=0.005 class defaults) — aligned, no gap.

### 1.11 Zone setpoints + demand limiting + windows

Mechanism (`Setpoints.mo`, G36 §5.3.2/§3.1.1.1): VAV defaults occupied
21/24 °C, unoccupied 16/32 °C; heat ≤ cool − 0.5 K anti-overlap; local
adjust limits (cool 22–27, heat 18–22 °C, occupied modes only, frozen during
demand limit); demand-limit offsets Level 1/2/3: cool +0.5/+1/+2 K,
heat −0.5/−1/−2 K (non-cumulative, per-zone exemptable); window open →
heat 4 °C / cool 49 °C (effectively off) + ventilation 0.

Mapping: CAN for demand-limit-style setpoint offsets as a software policy
(shift our PI targets under grid/price events — but TOU is only 1.25x and
arbitrage is dead, so the USE case is weak; still portable as a
pre-cool/deep-setback lever). Setpoint defaults themselves are GIVEN by the
sim's per-zone sensors (we read them — already aligned). Windows CANNOT
(no window sensors; closest analog: none).

### 1.12 AFDD, alarms, and alarm suppression

Mechanism (Trane G36 deck + JCI/Modelica alarm blocks, G36 §5.6.6 etc.):
15 AHU diagnostics (leaking econ damper, low static at full speed, …);
terminal alarms (low flow, low DAT, leaking damper/valve, flow-sensor
calibration with 300–900 s persistence timers); hierarchical suppression
(load alarms suppressed when source is in fault); time-based suppression
after setpoint changes (§1.1).

Mapping: not scoratively applicable (no faults injected, no alarm channel in
reward), but two portable IDEAS: (a) persistence timers before acting on
deviation (our PTR_PERSIST=4 is exactly this pattern); (b) source-vs-load
attribution — don't crank zone outputs when the AHU-level resource (SAT/
pressure/water) is the constraint; ease the upstream setpoint instead
(our T&R pressure does this implicitly).

### 1.13 Hot-water / chilled-water reset + plant requests

Mechanism: same T&R engine (§1.1) fed by valve/DAT requests (§1.6) and
AHU-level coil requests (`PlantRequests.mo`, §5.16.16: SAT error >3 K/2 K for
2 min → 3/2 CHW requests; CHW valve >95% → 1; HW mirror with 17/8 K DAT
shortfalls over 5 min). Plant enable = any plant request (valve >95% until
<10%).

Mapping: PARTIAL. Hot-water T&R on valve>95% CAN (we have valve feedback +
water-temp action; our lineage has water reset + pump idle-off). DAT-based
3/2-request rules CANNOT (no DAT sensor). Chilled-water reset CANNOT (no
chilled-water action — SAT is direct). Plant staging CANNOT (no staging
actions; constant COP makes it pointless anyway).

### 1.14 What G36 does NOT contain (verified gaps)

- No optimal-start ALGORITHM (interface only, §1.5).
- No IM/rogue logic in the reference code (§1.1).
- No night purge / pre-occupancy flush sequence (only warmup/cooldown +
  optimal-start window).
- No humidity/dehumidification sequences in the surveyed AHU/VAV scope
  (beyond the humid-climate TSupCoo_max ≤ 16 °C guidance).
- No worst-zone control (see §1.6 divergence note).

## 2. Manufacturer application sequences

### 2.1 Trane

Opened: G36 certification blog; 2023 Engineer Summit "Achieving Benefits of
Guideline 36" deck (57 pp, primary concept source for §§1.1/1.12: T&R
equations, IM, AFDD, alarm suppression); Buildings.com "Intelligent Variable
Air Systems" article (pre-G36 Tracer SC VAS strategies).

Sequences:

- Fan-pressure optimization (Tracer SC VAS): poll all VAV controllers, find
  most-open damper, reset duct static setpoint so ≥1 damper is nearly wide
  open. This is critical-zone reset — functionally the ancestor of G36 T&R
  pressure (same effect, cruder algorithm: no ignore-count, no persistence).
- Ventilation optimization: zone-level DCV (CO2 / occupancy / schedule, mixed
  per-zone as fits) + system-level ASHRAE 62.1 ventilation-reset equations to
  compute AHU OA intake. Same two-layer shape as G36 §1.4.
- SAT reset: cold SAT when warm outside (unload fan, avoid humidity); reset
  upward in mild/cold weather (economizer benefit + less reheat); zone-temp
  override lowers SAT if any zone overheats. Same shape as G36 §1.2 minus the
  T&R-bounded TMax.
- G36 product line: pre-packaged G36 programming library; first CEC-certified
  vendor for CA Title 24-2025 §140.4.r (G36-2021-based programming mandatory
  for nonresidential HVAC from 2026-01-01); RT-VAV (rooftop VAV) adaptation of
  GL36 concepts since the Guideline covers applied AHUs/chilled-water plants
  but not DX.

Mapping: same as the corresponding G36 sections (CAN/PARTIAL per §1.2–1.4);
Trane adds no distinct portable mechanism beyond G36. The zone-overheat SAT
override is a worst-zone-flavored tweak (CAN, aligns with our reward better
than pure G36).

### 2.2 Johnson Controls (Metasys)

Opened: Controller Tool Help LIT-12011147 v10.2 PDF (1528 pp: DAT reset
module, VAV single-duct app, warmup/cooldown determination, summer/winter
mode); SCT Help LIT-12011964 R13.3 PDF (734 pp: Optimal Start object, used in
§1.5). NOT opened (JS-only help portal; §7): v16+ G36 CCT modules, AHU Reset
application note LIT-12012070 (rogue-zone exclusion, summer/winter reset
modes known only from search snippets — not used as evidence).

Sequences (from opened bodies):

- DischargeAirSetpointReset module: OAT-band reset — inputs OA Reset A/B
  (OAT band edges) + DAT low/high limits; output DAT-SP linearly interpolated
  on OAT (default reset variable 22 °C). Used across AHU mixed-air / 100%-OA /
  rooftop trees; dual-deck variants reset by EA-T/RA-T. Parameters: all four
  band endpoints operator-set ("depends on use").
- VAV single-duct application (selection tree): zone temp + flow control core;
  warmup/cooldown mode logic interfaced to building-wide Optimal Start;
  automated zone-setpoint shifts on OAT schedule; CO2 interface auto-resetting
  zone flow settings (DCV); pressure + temperature reset modules with network
  outputs driving a supply-air static/DAT T&R application; patented adaptive
  tuning of temp/flow loops; balancer override + auto-calibration.
- WarmupCooldownDetermination: warmup effective ANY time SA-T and ZN-T are
  reliable and SA-T > ZN-T + differential (local auto-warmup, not just
  network-commanded).
- SummerWinterModeDetermination: supply-water-temp-based season switchover
  (setpoint 38 °C/100 °F … actually compares water temp vs switchover SP with
  22 °C/40 °F … per table: SWO-SP 38 °C, SWO-DIFF 22 °C sic — operator-set in
  practice; summer = cold water → cooling modes allowed).
- Optimal Start object: §1.5 (quadratic + adaptation + optimal stop + OAT
  gating).
- Unoccupied Sequencing family (AHU/VAV/MZ/unit-vent variants, pp. 943–985):
  scheduled + optimal-start-driven mode arbitration per equipment type.

Mapping: OAT DAT-reset CAN (OAT + SAT action; cold-inert caveat per §1.2);
VAV-app DCV/flow-reset PARTIAL (no CO2/flow sensors; schedule+building-occ
substitutes CAN); T&R network-output architecture CAN (it IS §1.1);
warmup-on-warm-SAT CAN (we have SAT + zone temps); season switchover CAN
(OAT/water-temp driven; our lineage does this implicitly via OAT-gated
resets); optimal start/stop CAN (§1.5).

### 2.3 Siemens (Desigo/APOGEE-era solution set)

Opened: Siemens BT (Tianjin) "Solution set for AHU with VAV V07" (34 pp,
2008; cites Taylor's static-pressure-reset papers as the T&R source).

Sequences:

- Minimum OA intake (3 options): (A) mixing-plenum pressure control
  (commissioned DP setpoint matching design/min OA within 5%); (B) direct OA
  flow feedback (accurate but expensive sensing); (C) min-OA damper setting
  correlated with supply-fan speed (fixes the fixed-damper≈fixed-fraction
  VAV problem: low speed → bigger min opening). Rejects fan-tracking-derived
  OA (error stacking) for OA regulation.
- DCV with AQ sensor + free cooling (recommended): CO2-driven OA above the
  min-OA floor; free cooling = extra OA beyond min when supply air too warm.
  Return-duct CO2 reads low (short-circuit) → control to a lower ppm than
  in-space sensing. Fixed-OA option documented as the no-savings baseline.
- Building pressure (6 options): return-fan-to-supply TABLE tracking
  (recommended: easy, robust), fan-tracking PID on flow offset, inside/outside
  DP feedback to relief damper, relief fan on DP, relief-damper-only (small
  systems, <12–25 Pa drops), constant relief fan for fixed-OA systems.
- Supply-fan static control (4 options): (A) T&R on zone pressure requests
  (RECOMMENDED over PID: asymmetric trim/respond rates, works with end-switch
  boxes): 35–325 Pa, SP0 125 Pa, −10/+15 Pa per 2 min (I=2), or weighted
  +7.5×R capped +30; request = damper >95% until <80%; commissioning: plot
  setpoint vs requests, tune for slow reset (60→325 Pa in 1 h = too fast).
  (B) PID loop holding most-open damper at 90% (needs damper feedback; 90
  not 100 for two-sided deviation). (C) VAVPlus (patented): duct-pressure
  model predicts, per flow vector, which box-if-open needs max pressure;
  that box held open, fan set to the modeled pressure — continuous, no
  hunt-and-peck, sensor at fan discharge. (D) Constant pressure (cheapest,
  no networking; sensor 75–100% to most remote terminal).
- Safety: discharge high-static cutout; frost detector <3 °C kills fans.

Mapping: T&R variant CAN (same as §1.3; the 95%/80% hysteresis request rule
is directly portable and simpler than G36's flow-based rules); PID-on-max-
damper CAN (we have damper feedback; unexploited alternative to our T&R);
min-OA-by-fan-speed CAN as a recirc-vs-pressure scheduling idea (we have
both actions); VAVPlus CANNOT faithfully (needs a duct model; a learned
surrogate is conceivable but out of rule-based scope); building-pressure
options N/A (no pressure physics/relief actuators in sim); frost cutout N/A.

### 2.4 Daikin (Rebel rooftop, MicroTech III)

Opened: Rebel DPS IM 1125-11 (142 pp) via authorized-manual mirror.

Sequences (VAV = Discharge Air Controller flavor):

- Operating-state machine: Off → StartUp (3 min, fan off) → Recirculation
  (3 min, OA shut, fan on, mix duct+space air) → Fan Only (OA to minimum;
  stays while control temp within deadband) → Heating / Cooling /
  Economizing / Minimum DAT. VAV control temp = return temp; CAV = space
  temp. Mode entry at setpoint ± ½ deadband (e.g. heat 68 ± 0.5 °F →
  enter at 67.4 °F).
- Cooling: staged compressors on 4–5 min stage timers to hold DAT (VAV) or
  space temp (CAV); stage down/off at setpoint − ½ deadband.
- Economizer = FIRST stage of cooling: enabled when OAT < changeover
  (default 65 °F) AND differential-enthalpy switch made (optional); modulates
  to DAT (VAV) or space temp; every 4 min: add mechanical if 100% open,
  continue econ, or return to min + Fan Only if satisfied; exits to
  mechanical + min-OA on enthalpy-break or OAT rise.
- Resets (menu-exposed): cooling DAT reset (external, default 55–65 °F over
  0–100% signal); heating DAT reset (55–120 °F); MinOA reset sources
  None/External/CO2(IAQ 800 ppm DCV limit / 1000 ppm vent limit, 0–2000 ppm
  range)/CFM flow station; fan-speed reset (min 40%); BSP/Econ/Heating PI
  gains + PAT/MaxChg anti-windup caps; lockouts (cooling OAT 55 °F, heating
  0/55 °F, OAT diff 2 °F); unoccupied diff 3 °F.
- Minimum DAT mode: tempering heat when DAT < min limit in Fan Only
  (discharge-temp floor protection).

Mapping: state-machine + deadband + stage-timer structure CAN as a control
skeleton (our lineage's hold/latch/prestart logic is the same family);
economizer-first-then-mechanical CAN via recirc→SAT sequencing (our econ does
proportional, Daikin does staged — a portable variant); OAT-changeover 65 °F
≈ G36 fixed dry-bulb limit (CAN, consistent with §1.8); DAT/OA/CO2 resets
PARTIAL (external-signal resets CAN with OAT/schedule/building-occ as the
signal; CO2/CFM-sourced CANNOT — no sensors); Minimum-DAT tempering CANNOT
(no DAT sensor; sim discharge cap is a ceiling anyway); staged-DX N/A
(continuous SAT action); lockouts CAN (OAT-gated enables — our lineage does
this).

## 3. Open-source equipment / BMS / thermostat firmware and control stacks

Commercial binary firmware is out of scope (see header). Everything below is
open-source-licensed code or documentation that was opened and read.

### 3.1 Modelica Buildings Library + OpenBuildingControl (LBL)

Opened: G36 package sources (§1, 15 .mo files) + generated doc pages for
Generic, TerminalUnits.Reheat, AHUs.MultiZone.SetPoints; package inventory
via GitHub API (G36/AHUs, Generic, Plants, TerminalUnits, ThermalZones,
Types, VentilationZones, ZoneGroups; OBC root: ASHRAE, CDL, OutdoorLights,
RadiantSystems, Shade, UnitConversions, Utilities).

What it is: the reference executable specification of G36 in the Control
Description Language (CDL, a vendor-neutral Modelica subset): elementary
blocks + full AHU/VAV/plant sequences + validation models. This is the
artifact §§1.1–1.13 were verified against.

Mapping: it is a SIMULATION + code-generation library, not a runnable
policy for our sim — but every sequence in it ports per §§1.x. Verified
absent from it: optimal-start algorithm, IM/rogue weighting (§1.14).

### 3.2 Eclipse VOLTTRON (PNNL/DOE, open source)

Opened: readthedocs index + Agents Overview + Core Services pages (v9.0.4).

What it is: a message-bus-based Python platform for building/energy control
("an operating system for buildings"), NOT a sequence library. Verified
inventory: Agent Framework (lifecycle + pub/sub + config store + security);
core service agents — PlatformDriverAgent + BACnetProxy (device I/O),
ActuatorAgent (arbitrated control), OpenADRVenAgent (demand response),
historians (SQL/Mongo/MQTT/forward), tagging services, VolttronCentral
(fleet mgmt); operations agents — ThresholdDetectionAgent (deadband alarms),
watchers, SysMon; driver + historian + web + simulation-integration
frameworks.

Mapping: N/A as portable sequences — VOLTTRON ships no canned VAV/AHU
sequences (control logic is user-written agents). Relevant only as (a) the
deployment target class our policy would compile to in a real building
(schedule/agent structure), and (b) ThresholdDetection/OpenADR as prior art
for deadband alarming (§1.12 pattern) and demand-limit inputs (§1.11).

### 3.3 Home Assistant generic_thermostat + HASmartThermostat

Opened: HA `generic_thermostat` integration docs; HASmartThermostat README
(ScratMan, HACS custom integration).

Mechanisms:

- generic_thermostat: single-switch bang-bang with `cold_tolerance` /
  `hot_tolerance` hysteresis (heater on below target−cold, off above
  target+hot; e.g. 25 ± 0.5), `min_cycle_duration` anti-short-cycle,
  `keep_alive` re-drive interval, `initial_hvac_mode`, `precision` /
  `target_temp_step`. One switch per entity (heat/cool = two entities).
  Explicitly NOT a PID (community-confirmed in search; docs show no I/D).
- HASmartThermostat: virtual PID thermostat: `error = target − current`,
  `P = Kp·e`, `I += Ki·e·dt`, `D = −Kd·di/dt`, output 0–100% → PWM over a
  period (e.g. 15 min; 33% = 5 min on) driving on/off heaters/valves;
  example gains Kp=50/Ki=0.01/Kd=2000 (single heater) or 5/0.01/500
  (valves); `sampling_period` for fast sensors; Ziegler-Nichols autotuner
  (freezes setpoint during tuning, hysteresis fallback); outdoor-temp
  feedforward `E = Ke·(target − outdoor)` (Ke≈0.6 well-insulated — acts
  like fast integral); cascading PIDs (room→floor→PWM) for floor heating.

Mapping: single-zone on/off concepts. Hysteresis + min-cycle CAN (our AHU
on/off + hold timers are the same family; G36 uses this at zone level, we
at AHU level). PID+PWM N/A directly (our valves are continuous — PWM would
only add cycling; but the Ke outdoor-feedforward is a portable PI
augmentation: our zone PI could add `Ke·(Ttarget − OAT)` — CAN, unexploited).
Autotune N/A (offline; our gains are hand-tuned per policy generation).

### 3.4 ESPHome PID Climate + OpenTherm

Opened: ESPHome `climate/pid` docs; ESPHome `opentherm` hub docs.

Mechanisms:

- PID Climate: full PID over a temp sensor → float `heat_output` /
  `cool_output` (at least one required); `output_averaging_samples` and
  `derivative_averaging_samples` smoothing; deadband block
  (`threshold_high/low`, e.g. ±0.5 °C) with multipliers; autotune action
  (`climate.pid.autotune`); deadband-parameter actions for runtime tuning.
  Example gains Kp≈0.49/Ki≈0.0049/Kd≈12.6.
- OpenTherm hub: GPIO bit-banged OpenTherm master (in/out pins) exposing the
  full OT data-ID set: boiler supply-water control setpoint (write),
  DHW setpoint, max CH setpoint clamp, room-temp/setpoint feedback, CH/DHW
  enable flags. Three setpoint sources: HA sensor, PID Climate output, or
  static. I.e. the standard open pattern = room-side PID computes a
  boiler-water setpoint, OT carries it to the boiler.

Mapping: ESPHome PID ≈ G36 §1.10 zone loops (CAN, already aligned). The
OpenTherm pattern "room PID → water-temperature setpoint" is EXACTLY our
hot-water reset architecture (zone demand → `water_indices[0]`); their
max-CH-setpoint clamp ≈ our discharge-cap-aware water ceiling (CAN as a
design pattern; our lineage's water reset + removed hot-water floor in
`occ_gated4` already encodes the "hotter water past the cap is waste"
lesson).

### 3.5 Sinergym / CityLearn / BOPTEST rule-based baselines

Opened: Sinergym `utils/controllers.py`; CityLearn `agents/rbc.py` +
controller-examples docs; BOPTEST `baselines/` (README, `run_baselines.py`)
+ `examples/python/controllers/{baseline,pid,sup}.py`.

Mechanisms (all deliberately weak — they are RL foils, not SOTA):

- Sinergym: `RBC5Zone` = seasonal fixed bands (summer 23–26 °C Jun–Sep,
  winter 20–23.5 °C, ASHRAE 55-2004); `RBCDatacenter` = fixed 18–27 °C
  (ASHRAE TC9.9); `RBCIncrementalDatacenter` = ±1 K steps on the cooling
  setpoint when mean zone temp exits the band. No scheduling, no resets.
- CityLearn: `HourRBC` = 24-h action map per device (hour → action);
  `BasicRBC` = charge storage 9.1%/h overnight (22:00–08:00), discharge
  8%/h daytime, cooling at 40%/80% nominal, heating inverse. Pure
  time-of-day load shifting.
- BOPTEST: baseline = `u={}` (test case's built-in Modelica controller runs;
  the "baseline" is the emulator's default, price-blind); examples add a
  deadband-P heater loop (20–23 °C, Kp=2000 W/K) and fixed supervisory
  setpoints (22/23 °C).

Mapping: nothing to port beyond what we already exceed — EXCEPT the negative
result is informative: none of these stacks implement T&R, dual-max, DCV, or
optimal start; the open-source RL-gym world has NO advanced rule-based HVAC
baseline, so our G36 ports (§§1–2) stand on the LBL/manufacturer sources
only.

### 3.6 Deliberately not covered (open source)

- openHAB, ThingsBoard, Tasmota, OpenTherm Gateway firmware, MicroPython
  thermostat sketches, EnergyPlus example controllers: not inspected (time
  budget; ESPHome/HA already saturate the thermostat-firmware pattern and
  VOLTTRON the BMS-platform pattern). No claim is made about them.
- Project Haystack / Brick Schema: semantic tagging, not control sequences —
  out of scope by definition.

## 4. Consolidated mapping to our obs/action space

| Sequence | Verdict | What ports | What blocks the rest |
|---|---|---|---|
| T&R engine (timer, trim+respond, hold, suppression) | CAN | Whole engine; rescale to 5-min steps | — |
| SAT reset (OAT schedule + T&R TMax) | PARTIAL | Occupied-hours reset; cold-climate gate | SAT inert below OAT −6.8 °C; no AHU heat coil for warmup 35 °C |
| Static-pressure T&R + fan modes | CAN | Whole loop w/ damper+temp proxy requests | Per-zone flow-based 3/2-request rules (no sensor) |
| Zone request/state/prioritization + IM | CAN/PARTIAL | States, 1-request rules, IM weighting | Flow- and DAT-based 3/2 rules |
| DCV + 62.1 ventilation reset | PARTIAL/CANNOT | System-level OA via recirc + building-occ | Zone-level: no CO2 / per-zone occ / flow / DAT |
| Optimal start (JCI quadratic + adapt) | CAN | Whole algorithm | — |
| Optimal stop coasting | CAN | Whole algorithm (apply conservatively) | — |
| AHU/zone-group mode machine | CAN | All modes; sub fan-min+reheat warmup, max-recirc OA cut | Warmup SAT 35 °C; OA=0 (0.70 cap) |
| Economizer + fixed dry-bulb limit | PARTIAL | Fixed-bulb limit, fan-off disable, SAT tracking | Differential/enthalpy (no RAT/humidity); damper dynamics |
| Dual-max VAV staging idea | PARTIAL | Valve-first-then-airflow output map | DAT tracking/gates (no DAT sensor/actuator) |
| Zone PI loops | CAN | Aligned already | — |
| Demand-limit offsets | CAN | Portable; weak use case (TOU 1.25x) | — |
| HW reset (valve-driven) | CAN | Aligned already | DAT-based rules (no sensor) |
| CHW reset / plant staging | CANNOT | — | No such actions; constant COP |
| AFDD/alarms | N/A | Persistence-timer + source/load patterns | No faults/alarms in sim |
| Trane VAS trio | CAN/PARTIAL | Per G36 rows above | Same blockers |
| JCI OAT DAT-reset / season / WC modes | CAN | All (OAT+SAT driven) | — |
| Siemens PID-on-max-damper | CAN | Unexploited alternative to T&R | — |
| Siemens VAVPlus | CANNOT | — | Needs duct model |
| Daikin state machine / econ-first / lockouts | CAN | Skeleton + sequencing + lockouts | CO2/CFM resets; DAT tempering; staged DX |
| Thermostat PID/hysteresis/feedforward | CAN | Hysteresis, Ke·(T−OAT) feedforward | PWM (continuous valves); autotune (offline) |
| Gym RBCs (Sinergym/CityLearn/BOPTEST) | N/A | Nothing beyond current lineage | Weak by design |

## 5. Unexploited ideas for our lineage (ports not yet tried)

Ranked by implementability (all CAN/PARTIAL above; no new sensors needed):

1. G36 two-stage SAT reset (OAT schedule + T&R-bounded TMax on zone cooling
   requests) — our SAT resets are single-stage demand followers.
2. Dual-max-inspired heat staging (valve ramp at min airflow → add air above
   50%) — our damper/valve move simultaneously.
3. Adaptive optimal start (JCI quadratic + daily adaptation) + conservative
   optimal stop — ours is fixed-hour prestart with no early release.
4. Importance weighting / rogue-zone down-weighting in reset requests —
   no weighting exists in our lineage.
5. PID-on-most-open-damper (Siemens B) as a drop-in alternative to T&R
   pressure — one loop, two-sided deviation at 90%.
6. Outdoor-feedforward `Ke·(Ttarget − OAT)` on zone PI (HASmartThermostat) —
   faster-than-integral cold-weather response.
7. System-level pseudo-DCV: raise recirculation in unoccupied/low-occupancy
   hours (ventilation-driven, distinct from our economics-driven econ).

## 6. Sources (every body below was opened and read)

G36 reference implementation (raw source, lbl-srg/modelica-buildings master):

- `Buildings/Controls/OBC/ASHRAE/G36/Generic/TrimAndRespond.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/Generic/TimeSuppression.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/TerminalUnits/Reheat/Subsequences/SystemRequests.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/TerminalUnits/Reheat/Subsequences/DamperValves.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/TerminalUnits/Reheat/Subsequences/ActiveAirFlow.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/ThermalZones/ZoneStates.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/ThermalZones/ControlLoops.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/ThermalZones/Setpoints.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/ZoneGroups/OperationMode.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/ZoneGroups/ZoneStatus.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/AHUs/MultiZone/VAV/Controller.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/AHUs/MultiZone/VAV/SetPoints/SupplyTemperature.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/AHUs/MultiZone/VAV/SetPoints/SupplyFan.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/AHUs/MultiZone/VAV/SetPoints/PlantRequests.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/AHUs/MultiZone/VAV/Economizers/Controller.mo`
- `Buildings/Controls/OBC/ASHRAE/G36/AHUs/MultiZone/VAV/Economizers/Subsequences/Enable.mo`

LBL generated docs:

- https://simulationresearch.lbl.gov/modelica/releases/v12.1.0/help/Buildings_Controls_OBC_ASHRAE_G36_Generic.html
- https://simulationresearch.lbl.gov/modelica/releases/v9.1.0/help/Buildings_Controls_OBC_ASHRAE_G36_TerminalUnits_Reheat.html
- https://simulationresearch.lbl.gov/modelica/releases/v9.1.0/help/Buildings_Controls_OBC_ASHRAE_G36_Generic.html
- https://simulationresearch.lbl.gov/modelica/releases/v5.0.0/help/Buildings_Controls_OBC_ASHRAE_G36_PR1_AHUs_MultiZone_SetPoints.html

Manufacturer:

- http://www.trane.com/commercial/north-america/us/en/about-us/newsroom/blogs/ashrae-guideline-36.html
  (G36 certification, Title 24 §140.4.r, pre-packaged library)
- https://www.trane.com/content/dam/Trane/Commercial/north-america/about-us/newsroom/events/2023-engineer-summit/2023-EngSummit-Achieving-Benefits-of-Guideline-36-FINAL.pdf
  (57 pp: T&R equations, IM, AFDD, alarm suppression)
- https://www.buildings.com/home/article/10188679/trane-intelligent-variable-air-systems-increase-efficiency
  (Tracer SC VAS: fan-pressure optimization, ventilation optimization, SAT reset)
- JCI Controller Tool Help LIT-12011147 v10.2 (1528 pp, via
  https://docs.johnsoncontrols.com/bas/api/khub/documents/oljzrL52qtPGho8In2ASqA/content)
  (DAT reset module, VAV single-duct app, warmup/cooldown, summer/winter)
- JCI SCT Help LIT-12011964 R13.3 (734 pp, via
  https://docs.johnsoncontrols.com/bas/api/khub/documents/BSG75dQSEry67svv4sMXEQ/content)
  (Optimal Start object pp. 430–435: equation, adaptation, optimal stop)
- http://www.bt.siemens.com.cn/partner/DownloadCenter/upload/20121217015242429.pdf
  (Siemens "Solution set for AHU with VAV V07", 34 pp: min OA, DCV, building
  pressure, T&R/PID/VAVPlus/constant fan control)
- https://dxseng.com/wp-content/uploads/2020/10/Daikin-Rebel-Rooftop-3-28Tn-Manual-Daikin-IM-1125-11.pdf
  (Daikin Rebel IM 1125-11 mirror, 142 pp: state machine, economizer, resets)

Open source:

- https://esphome.io/components/climate/pid/ (PID Climate)
- https://esphome.io/components/opentherm (OpenTherm hub)
- https://www.home-assistant.io/integrations/generic_thermostat/
- https://github.com/ScratMan/HASmartThermostat (README: PID+PWM, autotune,
  outdoor feedforward, cascade)
- https://volttron.readthedocs.io/en/latest/ (+ `agent-framework/agents-overview.html`,
  `agent-framework/core-service-agents/index.html`)
- https://github.com/ugr-sail/sinergym (`sinergym/utils/controllers.py`)
- https://github.com/intelligent-environments-lab/CityLearn (`citylearn/agents/rbc.py`;
  https://www.citylearn.net/usage/agents.html)
- https://github.com/ibpsa/project1-boptest (`baselines/`, `examples/python/controllers/`)

Secondary:

- https://sustainable-eng.com/ecm-tip-2-optimal-start-stop/ (optimal start/stop
  practitioner summary with savings example)
- https://starduct.vn/en/news/-vav-system-control-and-monitoring-strategies
  (VAV strategy survey: pressure/SAT reset, DCV, occupancy setback)

In-repo (mapping baseline, read not modified):

- `smart_buildings/baselines/transfer_policy.py` (obs/action contract)
- `smart_buildings/baselines/{pid,pid_plus,zone_pi,occ_gated4,occ_gated7}.py`
  (lineage state for §§4–5)

## 7. Unresolved (could not open; not used as evidence)

- Trane Engineers Newsletter ADM-APN078-EN / Taylor "VAV System Static
  Pressure Setpoint Reset" and "Dual maximum VAV box control logic" (ASHRAE
  Journal): manuals.plus bot-challenged (403), thefreelibrary 403, no
  accessible mirror found. G36 reference code + Siemens doc cover the same
  mechanisms, so no evidence gap — only the original prose is uncited.
- JCI AHU Reset application note LIT-12012070 and v16+ CCT G36 modules
  (Cooling/Static-Pressure Request Determination, BSP/relief control):
  docs portal is JS-rendered (curl/Jina return "Loading application…");
  khub documents index truncates server-side before these entries; legacy
  cgproducts host redirects to the portal. Rogue-zone-exclusion specifics
  and JCI's exact G36 request thresholds are therefore snippet-level only
  and were excluded from evidence.
- JCI Premier rooftop DAT-reset "five methods" and TempMaster morning
  warm-up/optimal-start pages: same JS-portal block.
- Trane Tracer Summit BAS-PRC001-EN and UC400/UC210 VAV manuals: manuals.plus
  bot-challenged; trane.com/elibrary copies 404. Trane VAS behavior verified
  instead via the opened Buildings.com article + G36 deck.
- Daikin MicroTech OM 1141 (referenced economizer detail): not fetched;
  IM 1125-11 sufficed.
- MDPI Buildings 12(3):348 (SAT reset strategies S1–S3), ACHR News economizer
  high-limit article, DeepWiki G36 page: 403/429 at fetch time; claims covered
  by primary sources instead.
- Open-source: openHAB, ThingsBoard, Tasmota, OpenTherm Gateway, EnergyPlus
  example controllers (§3.6): not inspected; no claims made.
- No code was touched; this file is the sole write (per task).
