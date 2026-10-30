# HVAC domain reference: physics, standards, observability, and deployment

Research date: **2026-10-28**. Scope: commercial HVAC, especially multizone AHU/VAV/reheat systems, with an explicit mapping to this repository's competition environment.

This reference complements [03-rule-based-sota.md](03-rule-based-sota.md). It explains the physical and operational assumptions behind control sequences; it does not establish new policy scores or report new simulation experiments.

Evidence is separated into **primary technical bodies**, **official edition/scope descriptions**, **repository contract**, and **engineering inference**. Full normative texts of the four current ASHRAE base publications were not obtained. Their public descriptions establish edition and scope; accessible approved addenda and open implementations support only the particular mechanisms identified below. A software implementation is not a substitute for the applicable standard.

## 1. Standards: current editions, different responsibilities

The editions below were verified against official sources on the research date. An edition being current does not establish which edition a particular jurisdiction or owner has adopted.

| Publication | Edition verified | What it addresses | Boundary for this project |
|---|---|---|---|
| ASHRAE Standard 55 | **55-2023** | Thermal conditions, environmental and personal factors; standard and adaptive evaluation methods | Zone air temperature alone cannot establish compliance. [Official overview][S01] |
| ASHRAE Standard 62.1 | **62.1-2025** | Ventilation and IAQ in its covered nonresidential occupancies; filtration, controls, operations, and moisture provisions | The 2025 overview explicitly adds air-density adjustments, DCV sequences, and humidity requirements; old formulas need edition-specific checking. [Official overview][S02] |
| ASHRAE/IES Standard 90.1 | **90.1-2025** | Minimum energy requirements for covered buildings/sites and systems | Its overview includes revised equipment metrics, fan-power procedures, and ventilation/dehumidification alignment. It is not an unconstrained energy objective. [Official overview][S03] |
| ASHRAE Guideline 36 | **G36-2024**, with subsequent addenda | Implementable high-performance HVAC sequences | The official register lists 2024-edition addenda **a, e, f, h, k** dated in 2026. Listing is verified; this report does not claim a complete clause-by-clause audit of them. [Official register][S04] |

These documents answer different questions: whether occupants are thermally comfortable, whether ventilation/IAQ provisions are satisfied, whether energy requirements are met, and how a particular equipment arrangement should be controlled. A strong Kaggle reward is evidence for none of those compliance claims.

**Correction to the historical G36 survey.** “No humidity sequences” must be limited to the previously surveyed implementation. Approved **G36-2021 Addendum z, December 2024**, supplies humidity-limiting SAT logic using outdoor, return, and/or zone dew point. Separately, **90.1-2022 Addendum cy**, approved December 2025, explicitly discusses G36-2024 humidity sequences. These are opened primary documents, not an inference from a search result. [Addendum z][S05], [Addendum cy][S06]

**Correction to the historical ventilation formula.** The opened LBL AHU implementation uses `sysVenEff = 1 + VUncOutAir_flow/VSumZonPri_flow - uOutAirFra_max`. The maximum zone fraction is **subtracted**, not added. Its documentation identifies a May 2020 G36 sequence and a historical 62.1 ventilation-rate implementation; do not present it as the universal formula for every system or as a full 62.1-2025 implementation. [LBL outdoor-air calculation][S07]

## 2. Physical topology and units

A conventional arrangement is `outdoor + return air → mixing section → coils → supply fan → VAV terminal/reheat → zone → return/relief`. Fan placement and coil order vary. The water circuit is a separate energy transport system: heat source → pump → distribution → terminal coils → return. A controller must know which physical path each command affects.

The repository's [interface description](../../smart_buildings/baselines/transfer_policy.py) and [AGENTS.md](../../AGENTS.md) establish a narrower topology: cooling-only AHUs; VAV reheat provides heating; recirculation is capped at `0.70`; VAV discharge at `308.15 K`; dampers are floored at `10%`. These are simulator facts, not universal commercial-HVAC design values.

| Symbol | Meaning | SI unit / convention |
|---|---|---|
| `T`, `ΔT` | Absolute temperature; temperature difference | K; differences in K equal differences in °C |
| `T_C` | Dry-bulb temperature used in the approximate enthalpy equation below | °C, explicitly not K |
| `Q̇`, `P` | Heat transfer rate; electrical or fuel input rate | W = J/s; specify thermal versus electrical |
| `E` | Integrated energy | J; `1 kWh = 3.6 MJ` |
| `V̇`, `ṁ` | Volumetric flow; mass flow | m³/s; kg/s; `ṁ = ρV̇` |
| `ρ`, `c_p` | Density; specific heat | kg/m³; J/(kg K) |
| `C`, `G`, `R` | Lumped heat capacity; conductance; thermal resistance | J/K; W/K; K/W |
| `Δp`, `H` | Pressure difference; water head | Pa; m; `Δp = ρgH` |
| `w`, `h` | Humidity ratio; moist-air enthalpy | kg water/kg dry air; J/kg dry air |
| `φ`, `COP`, `η` | Relative humidity; coefficient of performance; efficiency | Fractions or dimensionless ratios, with `%` converted explicitly |
| `c`, `c_ppm` | CO₂ volume fraction; parts per million | Dimensionless; `c = c_ppm × 10⁻⁶` |

Useful approximate values near ordinary indoor conditions are `ρ_air ≈ 1.2 kg/m³`, `c_p,air ≈ 1006 J/(kg K)`, and `c_p,water ≈ 4180 J/(kg K)`. They are illustrative constants, not calibration values for every altitude, humidity, glycol mixture, or equipment model. EnergyPlus exposes pressure-, temperature-, and humidity-dependent psychrometric functions. [Psychrometric services][S08]

**Accounting rule:** `E_kWh = Σ P_W Δt_seconds / 3.6e6`. A 5-minute sample is `300 s`, not `1/12` of a joule-based timestep. Keep thermal load, electricity, natural gas, cost, and carbon as separate quantities before applying an objective.

## 3. Thermal dynamics: temperature is the result of several balances

For a well-mixed zone, a useful sensible balance is:

```text
C_a dT_z/dt = Σ h_i A_i (T_surface,i - T_z)
             + Σ ṁ_j c_p (T_j - T_z)
             + ṁ_inf c_p (T_out - T_z)
             + Q̇_internal,conv + ṁ_supply c_p (T_dis - T_z)
```

Every right-hand term is W. `C_a = ρ_air V_zone c_p` is air heat capacity in J/K; an effective lumped capacity may also represent sufficiently coupled contents. Solar and radiant internal gains generally affect surfaces before reaching the air. EnergyPlus explicitly separates these terms and integrates their dynamics. [Zone-air balance][S09]

For control, a two-state approximation can preserve slow storage:

```text
C_a dT_z/dt = G_oa(T_out-T_z) + G_am(T_m-T_z) + Q̇_conv + Q̇_HVAC
C_m dT_m/dt = G_am(T_z-T_m) + Q̇_rad
τ ≈ C/G                         # single isolated first-order approximation
```

This is an **engineering reduction**, not an assertion that the simulator is exactly a 2R2C model. Envelope conduction, surface radiation, furniture storage, and interzone coupling can produce several time constants. A fitted parameter may absorb multiple effects and change with ventilation or operating mode.

Three immediate implications follow from the equations:

1. Air temperature returning to setpoint does not imply that walls and contents have recovered. Aggressive night setback can leave a large morning recovery load.
2. A temperature slope observed with the fan off does not identify occupied-mode capacity; ventilation and heat transfer paths change when the fan starts.
3. A mean zone temperature can conceal one perimeter zone with much greater envelope loss or solar gain. Shared AHU decisions still require zone-level error information.

**Capacity check, worked example:** at `T_z = 21°C` and `T_dis,max = 35°C`, delivering `2 kW` of sensible heat requires at least `V̇ = 2000/(1.2×1006×14) ≈ 0.118 m³/s`. This calculation ignores duct loss and imperfect mixing. Reducing airflow below that value cannot be rescued by more valve opening once the discharge limit is reached. It directly explains why a very low fan-pressure policy can fail in this repository.

**Identification requirements:** usable temperature histories, physical units, action timing, operating mode, supply/discharge conditions, weather, and sufficient variation in independent inputs. Estimating separate envelope loss, internal load, and HVAC gain from almost constant temperature and tightly correlated actions is underdetermined. Treat residuals as evidence of model inadequacy before escalating controller aggressiveness.

## 4. Air distribution, fans, and hydronic delivery

### 4.1 Pressure does not equal flow

For a simple turbulent branch with fixed geometry, `Δp ≈ R_f V̇²`. An orifice approximation is `V̇ ≈ C_d A_open sqrt(2Δp/ρ)`. These are local approximations: terminal dampers, duct losses, filters, and parallel branches change the operating point. Damper position is neither measured airflow nor a universal linear flow fraction.

Fan electrical power is approximately `P_fan = V̇ Δp_total / η_total`. The duct-static sensor measures only one pressure quantity, not necessarily the fan's total pressure rise. EnergyPlus's detailed fan description models the intersection of fan and system curves and their changing efficiency. [Fan models][S10]

The familiar affinity-law result `P ∝ N³` assumes geometrically similar operation, roughly constant efficiency, and an appropriate unchanged system curve; `V̇ ∝ N` and `Δp ∝ N²` are its companion relations. Static-pressure reset, damper movement, minimum pressure, and efficiency variation break a blanket cubic extrapolation. The appropriate optimization variable is delivered service plus total power, not fan speed alone.

**Repository inference:** an almost-open damper plus persistent discomfort is a plausible demand signal, but cannot distinguish inadequate duct pressure, insufficient coil output, bad zone sensing, or insufficient supply-temperature difference. Pressure trim-and-respond is therefore only as informative as its request generator.

### 4.2 Water flow and coil output

```text
Q̇_water = ṁ_water c_p,water (T_supply - T_return)
P_pump  = V̇_water Δp_loop / η_total
Q̇_coil ≈ ε C_min (T_hot,in - T_cold,in)
C_min   = min(ṁ_hot c_p,hot, ṁ_cold c_p,cold)
```

The coil expression is the effectiveness formulation for sensible exchange; actual coil geometry and flow arrangement determine `ε`. It shows why water temperature, water flow, and air flow are coupled. Pump models also account for minimum/maximum flow, efficiency, speed, and heat entering the water. [Pump models][S11], [Coil models][S12]

Valve authority can be expressed at the design condition as `a = Δp_valve/(Δp_valve + Δp_other_branch)`. Low authority changes the installed flow characteristic and can make useful control occur over a narrow stroke. Increasing differential pressure can increase bypass and throttling losses; reducing it too far starves remote coils. A fully open valve is a capacity request only after confirming available water temperature and actual flow.

**Low-ΔT problem:** for a fixed thermal load, smaller water supply-return temperature difference requires higher water flow. That can raise pumping power and constrain plant capacity even when the load is modest. An optimizer that minimizes only boiler or chiller input can miss this system penalty.

**Missing inputs here:** terminal water flows, coil UA/performance data, terminal discharge-air temperature, valve characteristics, branch pressure drops, and trustworthy return-water measurements at the required locations. The competition offers hot-water temperature/pressure commands, not a fully instrumented plant optimization problem.

## 5. Psychrometrics: sensible cooling is not total cooling

Relative humidity depends on temperature. Humidity ratio measures the amount of water associated with dry air; dew point indicates the temperature at which that water content would become saturated. At barometric pressure `p`:

```text
p_v = φ p_sat(T)
w   = 0.62198 p_v/(p - p_v)
h   ≈ 1006 T_C + w(2501000 + 1860 T_C)    # J/kg dry air
```

Use consistent pressure units and `φ` as a fraction. The enthalpy expression is a conventional engineering approximation with a 0°C reference. A production implementation should use validated property routines and enforce their valid ranges. [Psychrometric functions and units][S08]

For a cooling coil, neglecting small condensate-energy corrections:

```text
Q̇_total ≈ ṁ_dry_air (h_in - h_out)
Q̇_sensible ≈ ṁ_air c_p (T_in - T_out)
ṁ_condensate = ṁ_dry_air (w_in - w_out), if positive
SHR = Q̇_sensible/Q̇_total
```

Cooling below the entering dew point can remove water. Reheating after dehumidification raises dry-bulb temperature without putting that water back, so relative humidity falls while humidity ratio remains approximately unchanged. A coil's sensible heat ratio depends on entering conditions, airflow, and operating state; the coil load is not identical to the zone load because outdoor air and fan heat intervene. [Coil treatment of sensible/latent load][S12]

A compact zone moisture balance is `M_dry dw_z/dt = G_water + Σ ṁ_dry,j(w_j-w_z) + G_surface`. Terms have units kg water/s; `G_surface` accounts for moisture buffering. This state can evolve differently from air temperature. [Moisture balance][S13]

**Worked example:** at `ṁ_dry = 1 kg/s`, reducing humidity ratio by `0.004 kg/kg` condenses `0.004 kg/s`, or `14.4 kg/h`. With latent heat approximated as `2.5 MJ/kg`, that is about `10 kW` of latent removal before sensible cooling is counted. A dry-bulb-only economizer comparison can miss a load of this scale.

**Control implication:** raising SAT may save sensible cooling/reheat while reducing latent removal. Running the fan continuously through a wet, inactive coil can also change delivered moisture behavior; performance depends on the actual equipment. Humidity constraints can require operation outside occupied comfort hours. Approved Addendum cy explains why dew-point histories, rather than a single occupied RH check, matter to its humidity-control framework. [Humidity-control rationale][S06]

**Repository boundary:** the observation contract exposes no zone/return/outdoor humidity or dew point. Its AHU cooling energy calculation uses sensible temperature differences and fixed COP. An enthalpy economizer, humidity-safe SAT reset, or mold-risk conclusion cannot be validated from this interface. More realistic moisture-aware objectives would require both new sensors and supported moisture physics.

## 6. Ventilation, IAQ, and CO₂

### 6.1 Outdoor-air quantity is a separate constraint

The familiar ventilation-rate construction is `V̇_bz = R_p P_z + R_a A_z`, followed by `V̇_oz = V̇_bz/E_z`. Here `R_p` is flow per person, `R_a` is flow per floor area, `P_z` is the design/allowed population basis, `A_z` is area, and `E_z` is zone air-distribution effectiveness. Values, occupancy adjustments, exceptions, and current density treatment must come from the applicable edition and space category. [62.1 scope/current changes][S02], [LBL zone ventilation implementation][S26]

For the **historical single-path multizone VAV calculation** documented by LBL:

```text
X_s   = V̇_ou / Σ V̇_pz
Z_pz  = V̇_oz / V̇_pz
E_v   = 1 + X_s - max_z(Z_pz)
V̇_ot = V̇_ou / E_v                # before implementation design caps
```

`V̇_ou` is uncorrected system outdoor-air demand, `V̇_pz` primary zone supply flow, and `V̇_ot` system outdoor-air intake. The opened implementation caps both demand and effective intake using design quantities. Zero flow, infeasible demand, and division guards require explicit handling. Multipath/fan-powered systems with secondary recirculation require their own calculation. [Exact AHU implementation][S07]

**Sign sanity check:** with `X_s = 0.20` and critical `Z_pz = 0.50`, efficiency is `0.70`; intake is `1.429 V̇_ou`. Using a plus sign yields `1.70` and `0.588 V̇_ou`, reversing the critical-zone penalty. This numerical example is a direct calculation, not a proposed design.

DCV can reduce the population-related demand when permitted and measured/estimated occupancy falls; it does not justify treating all pollutant sources as proportional to people. Maintaining a fixed damper position also does not maintain a fixed outdoor-air flow as fan flow and pressures change. DOE's ventilation guide emphasizes verification of actual ventilation and CO₂-sensor calibration. [DOE assessment guide][S14]

**Repository implication:** `recirculation ≤ 0.70` means at least a `0.30` outdoor-air fraction of modeled running supply flow. It establishes neither enough L/s per person nor enough flow to each zone. A `10%` damper floor likewise is not evidence of 62.1 compliance. Building-level occupancy cannot resolve a crowded conference room within a lightly occupied building.

### 6.2 What CO₂ can and cannot identify

For a single well-mixed space with balanced outdoor exchange, constant volume, and consistent volumetric reference conditions:

```text
V_zone dc/dt = G_CO2 + V̇_oa(c_out - c)
c_steady - c_out = G_CO2/V̇_oa
τ_vent = V_zone/V̇_oa
```

`G_CO2` is pure-CO₂ volume generation per second; concentrations are fractions, not raw ppm. The steady-state equation is unreliable before sufficient settling or when people, ventilation, mixing, or outdoor concentration change. A single concentration cannot identify both unknown occupancy emissions and unknown outdoor-air rate.

Persily's NIST study develops space-specific CO₂ interpretation based on occupants, generation, airflow, time, and outdoor concentration. CO₂ is not a universal indicator for particles, VOCs, or infection risk; particle filters remove particles without necessarily changing CO₂. Do not reuse `1000 ppm` as a universal IAQ/compliance threshold. [NIST primary paper][S15]

For particle control, a conceptual equivalent clean-air contribution is `η_filter V̇_recirc`, with actual efficiency depending on particle size, leakage, and installation. It is not interchangeable with outdoor-air dilution for gases. Outdoor smoke or pollution may also make additional outdoor air undesirable; use the applicable emergency/IAQ procedure and pollutant information, not a thermal economizer signal alone. [DOE ventilation assessment][S14]

## 7. Economizers and heat recovery

An airside economizer increases outdoor-air use when it can help satisfy cooling. A **fixed dry-bulb** limit needs outdoor temperature; a **differential dry-bulb** comparison also needs return temperature. Enthalpy variants need humidity as well; control may impose additional dew-point, dry-bulb, and operating limits. Minimum ventilation and relief-air handling remain separate requirements. [EnergyPlus outdoor-air controller][S16]

For ideal mixing, with outside dry-air mass fraction `f`:

```text
w_mix = f w_out + (1-f)w_return
h_mix = f h_out + (1-f)h_return
T_mix ≈ f T_out + (1-f)T_return    # sensible approximation
```

**Engineering objective:** compare avoided compressor power with changes in supply/return/relief fan power, latent treatment, and downstream reheat. “Outdoor air is colder” is only one term in that comparison. An economizer can still require mechanical cooling when outdoor conditions cannot meet supply demand.

**Implementation requirements:** a verified minimum-outdoor-air method; functioning linked dampers; relief/exhaust capacity; mixed/supply temperature; high-limit sensors; and freeze/smoke/pollution interlocks appropriate to the equipment. Use hysteresis and minimum dwell times to prevent mode chatter, while preserving faster protection overrides. Sensors must sample representative mixed air, since stratification can hide local coil-freezing conditions.

Heat recovery changes the calculation. An energy-recovery device transfers sensible and possibly latent energy between exhaust and outdoor streams; its effectiveness is separate for the two processes and varies with flow. Bypass or reduced recovery can be appropriate when economizing. Frost protection, leakage, and fan pressure drop must be included in the real system model. [Heat-exchanger models][S17]

**Repository boundary:** direct recirculation and fixed-COP sensible cooling allow a simplified economic mixing policy. There is no observed return humidity, pollutant state, or energy-recovery device. A zone-temperature average is at most an inferred return-temperature proxy; without return-flow weights, stratification, and duct gains it is not a measured mixed-air state.

## 8. VAV dual-maximum control and interactions

Dual-maximum control separates the heating and cooling airflow maxima. In the opened May 2020 G36 implementation, cooling demand raises the airflow target; deadband uses minimum airflow; early heating demand raises discharge-temperature target at heating-minimum airflow; stronger heating demand raises airflow after a discharge-temperature condition is satisfied. Inner loops track measured airflow and discharge temperature. [LBL terminal sequence][S18]

This can reduce avoidable reheat by avoiding high airflow of cold primary air during light heating. It also respects the heating capacity limit: sufficient warm air must eventually be delivered. Heating and cooling design flows need not be equal, and the minimum is constrained by ventilation, diffuser operation, sensor range, coil requirements, and operating mode.

**Do not equate three different controls:** zone-temperature loop → required service; discharge-temperature loop → reheat valve; airflow loop → damper. Opening both actuators in proportion to temperature error is a different architecture. A pressure-independent VAV box uses an airflow measurement and adjusts its damper against changing upstream pressure; a direct damper command has no such guarantee.

**Repository feasibility:** valve-first/airflow-second staging can be approximated from existing signals, but literal G36 tracking is blocked by absent terminal measured airflow and discharge temperature. Such an approximation should be called a heuristic and evaluated across climates, AHU sizes, and heating recovery events. No expected score gain is established by the domain argument alone.

**Interaction map:** higher SAT reduces cooling temperature difference, potentially increasing cooling airflow; lower SAT can increase reheat in cold zones; lower duct pressure can saturate critical dampers; lower hot-water temperature can saturate reheat valves. Independent reset loops can fight if they all respond to the same symptom. Add mode-aware requests, saturation handling, and separation of update rates before making resets more aggressive.

## 9. Plants, heat pumps, and objectives omitted by constant efficiency

For a chiller, `COP_c = Q̇_cooling/P_electric`; for a boiler, `η_b = Q̇_useful/P_fuel`, with heating-value convention stated. Real equipment has capacity limits, part-load behavior, cycling, minimum flow, and auxiliary power. A rated COP is an operating point, not a universal conversion factor.

EnergyPlus chiller models use capacity and energy-input-ratio curves depending on water/condenser conditions and part-load ratio; `EIR = 1/COP`. Thus chilled-water reset, condenser-water reset, tower fan effort, and chiller staging can trade against each other. Use equipment-specific curves within their fitted range. [Chiller models][S19]

Boiler models can make efficiency depend on water temperature and part load. For condensing equipment, return temperature and fuel/flue-gas conditions affect recovery of latent heat; lowering supply temperature helps only if coils still meet load and the complete loop permits it. A reset strategy therefore needs plant and terminal evidence, not a generic “lower is always better” rule. [Boiler models][S20]

For a heat pump, `COP_h = Q̇_heating/P_electric`. The ideal reversible upper bound is `T_hot/(T_hot-T_cold)` using reservoir temperatures in K; actual refrigerant temperature lifts and losses reduce performance. Outdoor-source capacity, frost, defrost, auxiliary heaters, and compressor operating envelope matter. A DOE/NREL experiment on one small cold-climate unit found major control-dependent defrost/pan-heater effects; this is evidence of mechanism, not a transferable percentage saving for commercial plants. [Primary test presentation][S21]

**Worked comparison:** at heat demand `Q̇`, heat-pump electricity is `Q̇/COP_h`, while boiler fuel input is `Q̇/η_b`. The cheaper or lower-carbon source depends on contemporaneous electricity/fuel factors and auxiliaries. Cost and carbon can favor different choices. This is a dimensional comparison; no tariff, marginal-emissions series, or switching recommendation is assumed here.

**Repository boundary:** chiller COP is `3.5` and boiler efficiency is constant by contract. Plant efficiency improvements from refrigerant lift, condensing operation, compressor staging, or heat-pump defrost are unavailable mechanisms. Water reset can still alter delivered heat, stored energy, losses, and comfort; it should not be justified by a condensing-efficiency benefit absent from the model.

Demand-charge avoidance, thermal storage, time-varying marginal carbon, and heat recovery may be material elsewhere. The verified repository TOU ratio is only `1.25x`, and its established conclusion is that load-shifting arbitrage is unattractive. Do not import a different building's economic case or override comfort to pursue an absent incentive.

## 10. Sensors, state estimation, and observability

Control points fall into four categories: **command**, **setpoint**, **measured physical response**, and **inferred state**. Their names can look similar while their evidentiary value differs. A writable valve point is not proof of water flow; a fan enable is not proof of airflow; a setpoint is not the achieved temperature.

| Method / claim | Needed information beyond a zone thermostat | What is missing or uncertain in this interface |
|---|---|---|
| PMV / Standard 55 assessment | Mean radiant temperature, air speed, humidity, clothing, activity, applicable method conditions | Most of these inputs are absent; temperature-band reward is a different metric |
| Enthalpy economizer | Outdoor and return humidity/temperature, pressure/property conversion | Humidity and measured return temperature absent |
| Multizone ventilation reset | Zone population/area/type, required rates, distribution effectiveness, measured primary flows, OA flow | No per-zone occupancy or measured terminal flow; geometry alone is insufficient |
| CO₂ DCV | Calibrated CO₂, outdoor baseline, generation/occupancy model, ventilation constraints | No CO₂; only building occupancy |
| Literal dual-max terminal sequence | Terminal discharge temperature and measured airflow, design minima/maxima | No terminal DAT or measured airflow |
| Coil/valve diagnosis | DAT/SAT, flow, water temperatures, actuator feedback, mode | Valve position plus room temperature leaves competing explanations |
| Plant COP optimization | Heat metering, electric/fuel metering, part-load/temperature maps, staging states | Fixed-efficiency physics and no staging action |
| Humidity-aware shutdown | Zone dew-point history, latent sources, infiltration, moisture storage, dehumidification capability | Humidity state and appropriate physics unavailable |
| Reliable optimal start | Time to occupancy, temperature history, disturbance estimate, achievable capacity, uncertainty | Some inputs available; exact latent/thermal state and future disturbances not observed |

CBE's comfort work illustrates why radiant temperature, clothing, metabolism, air speed, and humidity alter a comfort calculation even at the same air temperature. The adaptive method has an applicability domain and cannot simply replace the mechanical-HVAC temperature objective. [CBE primary tool description][S22]

**Observability example:** the same room warming rate can be explained by solar gain, occupancy, hotter discharge air, or more airflow. If only room temperature and commands are observed, these effects can be inseparable. A hidden-state model should express that uncertainty; adding an estimator does not create information.

**Practical sensing checks:** inspect calibration, location, range, saturation, stale values, timestamps, normalization, unit conventions, and physical plausibility. Align commands with the response interval they actually influence. Distinguish missing values from true zeros. Aggregation should preserve zone/AHU topology and account for whether averages are volume-, area-, flow-, or occupancy-weighted.

**Filtering tradeoff:** suppressing noise helps avoid false requests, but a long filter delays morning recovery and hides abrupt equipment failure. Retain raw signals for diagnosis, use mode-specific persistence, and keep actuator anti-windup tied to achieved limits. Five-minute policy steps cannot reproduce every subminute protection or actuator loop.

## 11. Commissioning, FDD, and real-building transfer

Commissioning verifies that installed systems and operating sequences meet documented owner/facility requirements. ASHRAE's public commissioning page distinguishes **Standard 202-2024** for new buildings/systems and **Standard 230-2022** for existing buildings. It also identifies documentation and operator-training responsibilities. [Official commissioning scope][S23]

NIST's field study demonstrates two different diagnostic ideas: AHU rules based on mass/energy balances and operating mode, and statistical performance checks for VAV boxes. Their usefulness depends on available sensors and contextual interpretation. Fault detection is not equivalent to automatically knowing the failed component. [NISTIR 6994][S24]

The following transfer workflow is an **engineering recommendation**, not a claimed standard checklist:

1. **Establish topology and authority:** map zones, AHUs, water loops, physical actuators, BAS points, local loops, overrides, and protection priorities.
2. **Verify instrumentation:** cross-check temperatures and flows, inspect mixed-air sensor placement, reconcile meters, and document units and normal operating ranges.
3. **Recover the operating baseline:** trend occupied/unoccupied transitions, heating/cooling modes, outdoor-air behavior, saturated actuators, comfort, energy, and known complaints.
4. **Test sequences functionally:** exercise representative modes, setpoint transitions, minimum/maximum flows, recovery, sensor loss, communication loss, and restart behavior with the responsible site team.
5. **Run in advisory/shadow mode:** compare proposed actions with the existing BAS while measuring whether required inputs are valid and constraints are respected.
6. **Introduce bounded supervisory control:** retain equipment protection and local stabilization; limit rates and ranges; preserve operator override and a verified fallback.
7. **Verify performance under comparable conditions:** account for weather, occupancy, schedule changes, and concurrent repairs; include comfort, humidity, ventilation, cycling, and complaints alongside energy.
8. **Maintain the result:** assign fault ownership, document resolution, retrend after repairs, and verify seasonal modes. A dashboard alert without corrective action does not establish savings.

| Observed symptom | Plausible competing causes | Useful discriminating measurement |
|---|---|---|
| Zone cold, damper/valve high | Low pressure, cold supply water, limited coil, sensor error, envelope load | Terminal airflow/DAT and water temperatures |
| Warm DAT with valve commanded closed | Leaking valve, hot upstream air, sensor offset, duct gain | SAT, water flow, independent DAT check |
| High fan power with modest load | High pressure target, restricted filters, closed dampers, bad flow reading | Total/static pressure, filter differential, calibrated flow |
| Excess cooling plus reheat | Intentional dehumidification, bad reset, valve leak, conflicting setpoints | Dew point, coil states, valve response, zone modes |
| High occupied CO₂ | Insufficient OA, unusually high population, poor mixing, biased sensor | OA flow, population, outdoor CO₂, calibration |

The LBL Smart Energy Analytics Campaign reports median annual savings of `9%` for FDD participants and `3%` for energy-information-system participants, across a program involving `104 organizations` and about `6,500 buildings`. Here only the opened publication-page abstract supports those figures; the large report PDF failed retrieval. This observational program is context for operational value, not a guaranteed causal gain from adding an algorithm or a prediction of competition score. [LBL publication record][S25]

## 12. Implications for policy research in this repository

**Directly relevant mechanisms:** allocate scarce heating/cooling delivery to persistent zone needs; coordinate SAT, pressure, and reheat; use occupancy/schedule transitions carefully; handle saturation; and adapt prestart from demonstrated recovery behavior. The physical argument motivates hypotheses, not successful experiments.

**Partially transferable mechanisms:** dual-max staging, inferred terminal-capacity requests, and demand-based water reset. Their real versions use measurements absent here. Record the proxy, the failure mode it cannot distinguish, and the fallback before evaluating a heuristic.

**Unsupported mechanisms:** CO₂ compliance, humidity-safe dehumidification, true enthalpy economizing, heat-pump dispatch, variable-COP plant optimization, and full Standard 55 assessment. They cannot be claimed from temperature reward, actuator limits, or building-level occupancy alone.

For every candidate, write a small applicability contract: required sensors; actuators; physical mechanism; modes where it has authority; expected saturation; missing information; and measurable success/failure. This prevents a physically sensible real-building method from becoming an inert or harmful competition control.

Keep comparisons aligned with [AGENTS.md](../../AGENTS.md): full `8640`-step episodes; loss `abs(df.agent_reward.sum()) / len(df)`; deterministic same-policy/same-scenario results; no policy edits while an importing validation run is in flight. Use building/weather transfer cases, including cold recovery and capacity limits. This document itself ran no simulations and changes no policy.

## Source register and access audit

Access codes: **BODY** = relevant primary technical text opened and inspected; **OVERVIEW** = official publication/scope page opened, not the normative base document; **ABSTRACT** = publisher-hosted abstract only. All accesses below are dated **2026-10-28**. “Latest” documentation URLs are moving targets; their stated sequence vintage is recorded where applicable.

| ID | Primary source / version | Access and evidentiary use |
|---|---|---|
| S01 | [ASHRAE 55 official overview][S01] | OVERVIEW; 2023 edition, comfort factors and method families |
| S02 | [ASHRAE 62.1 official overview][S02] | OVERVIEW; 2025 edition/scope and advertised updates; no reproduced rate table |
| S03 | [ASHRAE 90.1 official overview][S03] | OVERVIEW; 2025 edition/scope and mechanical updates |
| S04 | [ASHRAE approved-addenda register][S04] | OVERVIEW/register body; G36-2024 existence and listed 2026 addenda; not their full contents |
| S05 | [G36-2021 Addendum z][S05], approved 2024-12-03 | BODY; humidity SAT limits; historical addendum lineage explicitly preserved |
| S06 | [90.1-2022 Addendum cy][S06], approved 2025-12-05 | BODY; dew-point/dehumidification rationale and explicit G36-2024 reference |
| S07 | [LBL G36 AHU outdoor-air calculation][S07] | BODY; exact ventilation-efficiency sign and design caps; documentation says May 2020 |
| S08 | [EnergyPlus psychrometric services][S08] | BODY; property definitions, units, and available functions |
| S09 | [EnergyPlus zone-air balance][S09] | BODY; sensible balance, storage, and supply-air heat delivery |
| S10 | [EnergyPlus fan models][S10] | BODY; total/static pressure, system curves, power/efficiency coupling |
| S11 | [EnergyPlus pump models][S11] | BODY; pressure/flow/power and heat-addition model |
| S12 | [EnergyPlus coil models][S12] | BODY; sensible/latent distinction, coil-load versus zone-load distinction |
| S13 | [EnergyPlus moisture balance][S13] | BODY; moisture sources, transport, and buffering |
| S14 | [DOE Ventilation Assessment and Action Guide][S14], September 2025 | BODY; ventilation verification, DCV sensing, economizers, and filtration context |
| S15 | [Persily, indoor CO₂ metric][S15], 2022 NIST-hosted paper | BODY; transient/space-specific interpretation and limitations of CO₂ |
| S16 | [EnergyPlus outdoor-air controllers][S16] | BODY; economizer inputs, limits, mixing, minimum OA, and relief control |
| S17 | [EnergyPlus heat exchangers][S17] | BODY; sensible/latent recovery and flow-dependent effectiveness |
| S18 | [LBL G36 reheat-terminal subsequences][S18] | BODY; dual-max topology and required measured inputs; May 2020 sequence vintage |
| S19 | [EnergyPlus chillers][S19] | BODY; temperature/part-load capacity and EIR models |
| S20 | [EnergyPlus boilers][S20] | BODY; temperature and part-load efficiency modeling |
| S21 | [DOE/NREL very-cold-climate heat-pump evaluation][S21], 2024 review | BODY; one-unit experiment, defrost/auxiliary control effects; preliminary results labeled in source |
| S22 | [CBE Thermal Comfort Tool research description][S22] | BODY; primary developers' model/input explanation; not certification of a proposed control |
| S23 | [ASHRAE commissioning publications][S23] | OVERVIEW; 202/230 scopes and operational documentation/training |
| S24 | [NISTIR 6994][S24], 2003 | BODY; field-tested AHU/VAV FDD mechanisms and diagnostic ambiguity |
| S25 | [LBL building-analytics campaign publication record][S25], 2020 | ABSTRACT; program size and aggregate reported savings only |
| S26 | [LBL ventilation-zone setpoints][S26] | BODY; historical zone ventilation calculation, inputs, effectiveness |

**Access limitations:** the G36-2024 store URL returned `403` on direct opening; its search-visible marketing text was not treated as an inspected technical body. The linked full LBL campaign PDF exceeded the browser retrieval size limit; only S25's accessible abstract is used. A DOE heat-pump overview returned `404`; S21 supplies the primary technical evidence instead. The public-review G36-2024 Addendum b search result was not used as an approved standard. No claim is made that all 2026 addenda, proprietary controller firmware, or every HVAC system family were exhaustively audited.

[S01]: https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy
[S02]: https://www.ashrae.org/technical-resources/bookstore/standards-62-1-62-2
[S03]: https://www.ashrae.org/technical-resources/bookstore/standard-90-1
[S04]: https://www.ashrae.org/technical-resources/standards-and-guidelines/standards-addenda
[S05]: https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/g36_2021_z_20241203.pdf
[S06]: https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/90_1_2022_cy_20251205.pdf
[S07]: https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Controls_OBC_ASHRAE_G36_AHUs_MultiZone_VAV_SetPoints_OutdoorAirFlow_ASHRAE62_1.html
[S08]: https://energyplus.readthedocs.io/en/latest/guides/module-developer/7.14-psychrometric-services.html
[S09]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/2.1-basis-for-the-zone-and-air-system-integration.html
[S10]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/16.4-air-system-fans.html
[S11]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/17.5-pumps.html
[S12]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/15.2-coils.html
[S13]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/2.4-moisture-predictor-corrector.html
[S14]: https://www.energy.gov/sites/default/files/2025-09/ventilation-assessment-and-action-guide_september-2025.pdf
[S15]: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934011
[S16]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/15.3-hvac-controllers.html
[S17]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/16.8-heat-exchangers.html
[S18]: https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Controls_OBC_ASHRAE_G36_TerminalUnits_Reheat_Subsequences.html
[S19]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/14.3-chillers.html
[S20]: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/14.2-boilers.html
[S21]: https://www.energy.gov/sites/default/files/2024-11/bto-peer-2024_32264-detailed-air-source-hp-evaluation-for-very-cold-climates-munk.pdf
[S22]: https://cbe.berkeley.edu/research/cbe-thermal-comfort-tool/
[S23]: https://www.ashrae.org/technical-resources/bookstore/commissioning
[S24]: https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6994.pdf
[S25]: https://bies.lbl.gov/publications/proving-business-case-building
[S26]: https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Controls_OBC_ASHRAE_G36_VentilationZones_ASHRAE62_1.html
