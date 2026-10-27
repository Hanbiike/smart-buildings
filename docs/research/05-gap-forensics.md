# Track 5: Leader-gap forensics (repo-only, from our own data)

**Question:** our best (`occ_gated4`/`occ_gated7`) scores public 0.01323;
the leader scores 0.00521 (2.5x gap). The leader beats even our TRAIN mean
(0.00963), so this is a CONTROL gap, not composition. Where does our loss
concentrate, and which control axes have we never probed?

**Top finding:** our loss is **96% energy+carbon and only 4% comfort**,
with median occupied comfort regret exactly 0. We are massively
over-conditioning: near-perfect comfort bought at full energy price. The
valid same-window probe data proves the exchange rate is favorable --
`g4_kp03` spends 0.001/step of comfort to save 0.0044/step of energy+carbon
on hard cells (-12%) -- but the systematic comfort-energy frontier was never
mapped. The gap lives in efficiency and the operating point, not in comfort
chasing. Ranked suspects in §7; unprobed axes in §6.

All numbers below come from CSVs already in the repo (1136 runs) plus the
policy/reward source. No new sim runs were needed (§8).

## 1. Validity: which comparisons are real

Three run populations exist in `smart_buildings/results/`:

| Population | Steps | n | Window |
|---|---|---|---|
| `occ_gated4` full runs | 8640 (30 d) | 30 cells | training episodes |
| `_full30d_archive/*/zone_pi` | 8640 | 25 cells | **same windows as g4** (timestamps + occupancy bit-identical, verified) |
| probes (`dsat_*`, `g4_*`, `bc_*`, …) | 864 (3 d) | 1068 | **first 864 steps of the same episodes** (timestamps + occupancy trace match g4 exactly, verified over all 1068) |

Consequences:

- Archive `zone_pi` vs `occ_gated4` on 25 matched cells is a **valid
  full-horizon** comparison (§3).
- Every 864-step probe vs `occ_gated4` truncated to its first 864 steps is a
  **valid same-window** comparison (§4). Naive "probe mean vs g4 full mean"
  is INVALID (different horizons; Fairbanks is strongly non-stationary --
  fb1 first-864 loss 0.0286 vs full-episode 0.0516; fb2 0.0293 vs 0.0161).
- `occ_gated5/6/7`, `occ_sat*`, `occ_duty*`, `occ_coolshift`, `occ_floor287`,
  all `probe_*`, `slot_linear`, `nn_policy` have **zero runs** in `results/`
  (coded but never measured on train).
- The submitted `occ_gated7` (trim-and-respond, occupied-only) therefore has
  no train CSVs; all loss-structure analysis uses `occ_gated4`, whose chain
  g7 extends without changing the base operating point
  (`baselines/occ_gated7.py`).

## 2. Loss structure of occ_gated4 (30 cells x 8640 steps)

Reward (verified in
`smart_control/reward/setpoint_energy_carbon_regret.py:333-344` and gin,
e.g. `training_scenarios/building_train_2/atlanta_3/config.gin:360-375`):

- `agent_reward = 0.6*prod_regret - 0.2*norm_energy - 0.2*norm_carbon`
  (weights sum to 1, no rescale). Comfort regret is 0 iff every occupied
  zone is in band (and 0 whenever unoccupied); 1 K excursion = 0.2 shortfall
  = **0.12 loss/step/K** (all zones), linear to 5 K, quadratic unbounded
  beyond (`base_setpoint_energy_carbon_reward.py:106-118`).
- Comfort band 294-297 K (3 K), eco band 289-298 K, comfort hours 06-19
  workdays (`config.gin:124-136`).
- Electricity TOU 16/18/20 $/kWh (peak/night 1.25x), carbon 88-104 kg/MWh
  (`electricity_energy_cost.py`); gas carbon 0.181 kg/kWh, ~2x electricity
  (`utils/constants.py:33-41`, `natural_gas_energy_cost.py:53-57`).

### 2.1 Overall split

Mean loss/step **0.009629** over 259,200 steps:

| Component | Share of loss |
|---|---|
| comfort | **3.9%** |
| energy | **49.8%** |
| carbon | **46.3%** |

Occupied steps are 36.3% of steps but carry 67.7% of loss; **unoccupied
steps (63.7%) carry 32.3%** -- pure energy+carbon waste by construction.
Day 68.3% / night 31.7%; workdays 91.2% / weekends 8.8%. Worst-5% steps
carry 22.5% (moderately heavy tail, not a tail story). Gas is only 24.2%
of energy $ but drives the 46% carbon share (2x carbon intensity).

### 2.2 Per-climate split (mean loss/step, shares)

| Climate | loss/step | comf% | en% | carb% | occ-loss% |
|---|---|---|---|---|---|
| fairbanks | 0.01925 | 2.7 | 29.9 | **67.5** | 51.9 |
| phoenix | 0.00911 | 5.4 | **72.1** | 22.5 | 78.4 |
| singapore | 0.00692 | 1.9 | **77.7** | 20.4 | 85.2 |
| atlanta | 0.00663 | 3.1 | 53.5 | 43.4 | 73.4 |
| denver | 0.00623 | 8.1 | 44.2 | 47.7 | 75.6 |

Cold = carbon-dominated gas heating; hot = energy-dominated
chiller+blower. No climate is comfort-dominated anywhere.

### 2.3 Per-cell table (mean loss/step; C/E/Cb shares; occupied-loss share)

| cell | loss | C% | E% | Cb% | occ% |
|---|---|---|---|---|---|
| train_1/fairbanks_1 | 0.05158 | 2.9 | 28.5 | 68.7 | 55.7 |
| train_8/fairbanks_8 | 0.03047 | 2.2 | 27.4 | 70.4 | 42.1 |
| train_1/phoenix_5 | 0.01696 | 1.2 | 80.3 | 18.5 | 71.9 |
| train_2/fairbanks_2 | 0.01612 | 1.8 | 28.6 | 69.6 | 40.8 |
| train_8/phoenix_4 | 0.01321 | 7.4 | 74.3 | 18.3 | 74.7 |
| train_7/fairbanks_7 | 0.01107 | 1.3 | 28.2 | 70.5 | 54.0 |
| train_7/atlanta_8 | 0.00999 | 3.9 | 31.3 | 64.9 | 55.6 |
| train_2/phoenix_6 | 0.00915 | 5.7 | 75.6 | 18.7 | 77.9 |
| train_4/denver_2 | 0.00833 | 5.1 | 31.1 | 63.8 | 61.2 |
| train_4/singapore_3 | 0.00780 | 1.4 | 79.9 | 18.7 | 85.0 |
| train_8/atlanta_1 | 0.00770 | 2.9 | 34.1 | 63.0 | 49.7 |
| train_1/singapore_8 | 0.00736 | 1.6 | 78.8 | 19.6 | 89.1 |
| train_4/atlanta_5 | 0.00729 | 1.0 | 79.8 | 19.3 | 92.1 |
| train_7/phoenix_3 | 0.00728 | 7.4 | 72.9 | 19.7 | 93.8 |
| train_8/singapore_7 | 0.00717 | 0.2 | 75.3 | 24.5 | 81.1 |
| train_7/singapore_6 | 0.00717 | 4.3 | 75.6 | 20.1 | 86.3 |
| train_2/denver_8 | 0.00705 | 3.5 | 34.3 | 62.2 | 47.8 |
| train_5/singapore_4 | 0.00650 | 3.0 | 78.3 | 18.7 | 86.4 |
| train_5/denver_3 | 0.00647 | 13.0 | 31.1 | 55.9 | 85.0 |
| train_7/denver_5 | 0.00635 | 15.9 | 65.7 | 18.4 | 97.0 |
| train_2/singapore_1 | 0.00554 | 0.9 | 78.5 | 20.5 | 82.7 |
| train_1/atlanta_2 | 0.00542 | 0.6 | 50.4 | 49.0 | 76.3 |
| train_8/denver_6 | 0.00506 | 6.1 | 70.3 | 23.6 | 94.0 |
| train_5/atlanta_6 | 0.00501 | 0.5 | 79.5 | 20.0 | 94.2 |
| train_5/phoenix_1 | 0.00473 | 9.0 | 40.2 | 50.7 | 80.0 |
| train_2/atlanta_3 | 0.00440 | 11.0 | 69.0 | 20.1 | 97.0 |
| train_1/denver_7 | 0.00411 | 5.0 | 43.1 | 51.9 | 81.8 |
| train_4/fairbanks_4 | 0.00342 | 13.8 | 49.1 | 37.1 | 89.7 |
| train_4/phoenix_8 | 0.00333 | 9.3 | 55.6 | 35.1 | 91.4 |
| train_5/fairbanks_5 | 0.00285 | 2.0 | 71.9 | 26.1 | 95.8 |

Two Fairbanks cells (fb1, fb8) are 5-8x the mean and dominate the average:
fb1 alone (0.0516) contributes ~18% of total train loss. But even deleting
both, the mean is 0.0066 -- still above the leader's 0.0052.

### 2.4 The 02:00-06:00 prestart spike (our own prestart)

Unoccupied loss by hour: hours 2,3,4,5 carry **17.6/15.9/17.1/20.8% =
71.4% of ALL unoccupied loss**. The spike is workday-only (0.0170/step vs
0.0021 weekend) and is **22.5% of TOTAL loss**. By climate: fairbanks
33.2%, atlanta 24.1%, denver 22.8%, phoenix 9.9%, singapore 7.5%.
Unoccupied burn is 61.6% gas $ (vs 17.8% when occupied).

Mechanism (code + `ahu_log.csv`): `OccGatedPolicy` allows
`MAX_PRESTART_STEPS = 48` (4 h) with 1.3x safety + 30-min margin
(`occ_gated.py:55-60,155-161`), latched (`occ_gated3.py:78-83`). On fb1
all 4 AHUs run hours 2-5 with the day's HIGHEST total flow (9.4 -> 11.6;
occupied flow falls 10.1 -> 7.6 through the day), SAT ~280 K, i.e. maximum
airflow x maximum dT against -42..+1 C outside air (`ahu_log.csv`,
train_1/fairbanks_1). The PI screams (u=-1, dampers full open) and VAV
reheat (discharge capped 308.15 K) heats the 30%-outside-air stream.

### 2.5 Occupied hours: perfect comfort at full price

Occupied loss/step 0.0180, of which comfort **5.7%**. Comfort-regret
distribution over 93,996 occupied steps: median 0, mean -0.0017, 75th
percentile 0; only 0.57% of steps worse than -0.05. Occupancy itself:
workdays only (Sat/Sun exactly 0%), hours 6-18 at ~96% of steps.
So the building is held in-band essentially always, including the
06:00-09:00 arrival ramp (occupancy rises monotonically 06:00 -> 13:00 per
`occ_gated.py:183-188`) when few occupants are present to benefit.

### 2.6 AHU block behavior (no staging)

`ahu_log.csv`: atlanta_3 (9 AHUs) is all-9-on or all-off (partial states
<1.5% of steps); fb1 (4 AHUs) all-on 47%, all-off 36%. Per-AHU on/off by
local demand is effectively absent -- the chain forces ALL AHUs on in PI
hours (`zone_pi.py:220-226`) and shuts ALL down otherwise.

## 3. zone_pi -> occ_gated4: what worked (valid, 25 matched cells)

Archive `zone_pi` vs current `occ_gated4`, same windows: 0.014886 ->
0.009480 (-36%), g4 wins 25/25. Biggest win fb1: -0.139 -> -0.052.
Savings decompose as comfort 17.5% / energy 38.1% / carbon 44.4%; occupied
hours saved 0.0037/step, unoccupied 0.0017/step. I.e. the occupancy gating
project harvested mostly energy+carbon -- and 96% energy+carbon REMAINS.

## 4. Valid same-window probe ranking (864 vs g4-first-864)

Diff > 0 means the probe beats g4 on the identical 3-day slice:

| probe | cells | mean diff | wins | note |
|---|---|---|---|---|
| g4_kp03 (KP 0.03) | 5 | +0.00141 | 4/5 | gentlest PI; full KP gradient 0.08<..<0.03 monotonic better |
| g4_kp035/kp04/kp045 | 5 | +0.0013..+0.0014 | 4/5 | same gradient |
| dsat_fp_06/fp_02 (higher forced pressure) | 2 | +0.00126 | 2/2 | fairbanks only; counterintuitive, n=2 |
| g4_m25w08kp05 (triple combo) | 30 | +0.00014 | 14/30 | coin flip at full table |
| g4bo_params, g4_mild025 | 30 | +0.00003..+0.00009 | ~12/30 | noise |
| dsat_gen31 (= g4 skeleton w/o gating) | 30 | -0.00003 | 11/30 | gating worth ~0 on 3-day slices (1 weekend day) |
| zone_pi | 25 | -0.00185 | 1/25 | g4 beats it on slices too |
| g4_prestart_gate | 30 | -0.00246 | 1/30 | skipping prestart after empty days LOSES (cold catch-up) |
| dsat_nightdrop3 (3 K setback) | 2 | -0.00330 | 0/2 | deeper setback LOSES via energy, not comfort (§5) |
| dsat_shift_deepest (2.4 K shift) | 4 | -0.00558 | 0/4 | blunt shift LOSES via comfort (§5) |

No tested axis moves full-table loss more than ~2%; the only >5% wins are
on hard-cell 5-grids (KP gradient, forced pressure). The 2.5x gap is not in
any tested axis.

## 5. Micro-autopsies (fb1, first-864 slice; loss/step)

**g4_kp03 beats g4 (-12%: 0.02527 vs 0.02865)** by TRADING comfort for
gas: comfort 0.00112 vs 0.00009 (12x worse, still tiny), energy
0.00708 vs 0.00838, carbon 0.01707 vs 0.02018; gas $184.9 vs $217.7
(-15%). Exchange rate: ~1 comfort buys ~4 energy+carbon. We sit far on
the over-conditioned side of this frontier.

**dsat_nightdrop3 loses (+18%)** through ENERGY, not comfort: comfort
flat, gas $250.6 vs $217.7, electricity $46.8 vs $24.7 (nearly doubles).
A cold-soaked building + morning sprint with ventilation load costs more
than the night saves. Naive "deeper setback" is dead; the catch-up
physics must be solved, not the setpoint lowered.

**dsat_shift_deepest loses (+14%)** through COMFORT: saves gas ($175 vs
$218) but comfort explodes to 0.00965 vs 0.00009. A blunt static 2.4 K
shift crosses the band edge; dynamic gentleness (low KP) does not.

**Non-stationarity warning:** fb1 first-864 0.0286 -> full 0.0516 (deep
winter worsens / thermal mass depletes day-over-day); fb2 0.0293 ->
0.0161 (warms). 3-day probes understate fb1-type difficulty by ~2x; never
extrapolate probe % to full episodes in heating climates.

## 6. Control-axis inventory: tried vs never probed

### 6.1 Tried and measured (in `zone_pi.py`, `occ_gated*.py`, results)

Zone PI gains (KP 0.03-0.10, KI, OAT-scheduled KI); down-only target shift
0.5-2.0 K + caps; night heat drop 2-3 K; prestart 1/2/3 h + adaptive 4 h
latch + EMA gate; water reset (full-K, effort weight, deadband); water
scale 0.6-1.0; hot-water floor (tried, REMOVED in g4); pump frac/idle-off;
SAT depth (-0.25..-1.0), DSAT ramp 0/0.35/0.5, per-AHU mixedness/deep SAT,
adaptive SAT EMA; pressmin; forced pressure -1.0/-0.6/-0.2; heat damper
gain 1.0/0.85; occupancy gate; shutdown/floor/holiday modes; economizer
decode fix; behavior-cloned NNs (bc_v1..v4e, dagger_v1..v3 -- v1/v2
catastrophic, v3 ≈ chain).

### 6.2 Coded but NEVER run (zero dirs in results/)

- `occ_sat/occ_sat2/occ_sat3` (`occ_gated.py:307-392`): deep-cold
  saturation restraint (damper cap when heat plant saturated). Directly
  targets the fb1 prestart pathology (§2.4). Highest-value unmeasured code.
- `occ_gated5` (heating target -2 K), `occ_gated6/7` trim-and-respond
  pressure (g7 submitted with no train CSVs), `occ_duty*` (per-AHU duty
  cycling; 4/2-4/4 failed per docstring, 6/1 micro never run),
  `occ_coolshift` (0.3 K warm-side cooling shift), `occ_floor287`
  (floor cap -2 K), all `probe_*` headroom probes (weekend floor,
  forced-unit economizer, prestart margins, SAT bias, time-of-day cooling
  offsets/peak shifting, guards), `slot_linear`, `nn_policy`.

### 6.3 Never probed (no code, no runs) -- exhaustive

AIR-SIDE HEATING DELIVERY (the fb gas sink):
1. **Damper/reheat decoupling.** The PI couples them (`damper=|u|`,
   `reheat=-u`, `zone_pi.py:193-200`). Optimal heating is MIN airflow x
   MAX discharge temp (cap 308.15 K); only gain 0.85 was tried. Delivering
   the same heat with less air cuts ventilation load ~1:1.
2. **Low-flow prestart.** Prestart runs max flow (§2.4). Longer + lower
   airflow prestart, prestart to a below-band target, or per-zone prestart
   (cold zones only) were never tried -- only timing (1-4 h).
3. **Catch-up rate control.** Cap dampers/airflow during recovery to
   spread load (vs sprint); tradeoff vs TOU/carbon never mapped.
4. **Warmest-zone SAT reset (cooling G36 classic).** SAT set so the
   warmest damper rides ~90%; trim-and-respond exists for PRESSURE only.

AHU OPERATION:
5. **Occupied AHU staging.** Forced-ALL-on in PI hours; per-AHU on/off by
   local demand, rotation, or occupied cycling never measured (duty
   cycling coded, failed variants only).
6. **Per-AHU schedules.** Prestart/schedule per AHU (cold-side prestarts,
   warm-side does not); only per-AHU SAT-by-mixedness exists.
7. **Recirculation beyond the organizer econ.** Max-recirc heating
   sprints, OA modulation by OAT, forced-unit economizer (coded in
   `probe_econ`, never run). Recirc cap 0.70 verified as hard floor on OA.

COOLING SIDE (SG/PHX = 72-80% energy):
8. **Chiller/blower optimum.** Colder SAT = more chiller + less blower;
   the exact optimum (COP 3.5 + fan curves are IN the sim) never computed;
   only 4 coarse SAT depths probed.
9. **Pre-cooling / thermal-mass charging** with cheap clean night
   electricity (16 vs 20c, 88 vs 104 kg/MWh). TOU arbitrage declared dead
   (1.25x) but never quantified against real charge/discharge losses;
   `probe_tod` (time-of-day offsets) coded, never run.
10. **Cooling-side night setback.** `NIGHT_COOL_CEILING` does the
    OPPOSITE (prevents overnight drift). Letting hot buildings drift up
    overnight + morning pull-down never tried.

WATER / PLANT:
11. **Pump differential-pressure reset** (G36 trim-and-respond for pumps;
    only frac-follow/idle-off tried), boiler staging/cycling/minimum-fire,
    hot-loop cooldown timing before shutdown.

COMFORT TARGETING (the 4%-loss anomaly):
12. **Deadband riding / hysteresis.** PI holds CENTER; comfort is free
    anywhere in 294-297. Heat-to-294.0-and-stop / cool-to-297.0-and-stop,
    heating-season low-edge riding, per-zone edge by leakiness -- never
    tried (only static shifts, which lose when blunt, §5).
13. **Occupancy-proportional conditioning.** Arrivals ramp 06:00-13:00,
    departures 13:00-19:05, yet conditioning is binary (full from 06:00).
    Comfort loss scales with occupants: soft start + evening coast
    (stop ~17:30, coast on mass) are nearly free and NEVER tried.
14. **Comfort budgeting across zones.** No per-zone occupancy obs exists,
    but zone-temp divergence could proxy local load; uniform band for all
    zones always.

SCHEDULE / MODES:
15. **Evening coast + morning soft start** (see 13). No policy stops
    before 19:00 except empty-day detection.
16. **Weekend/Monday dynamics.** fb1 thermal depletion day-over-day;
    Sunday-night recharge vs Monday 02:00 sprint, freeze-protection-only
    hold (<< 288 K floor cap), holiday-week detection -- all reactive or
    absent (`probe_wfloor` coded, never run).
17. **Episode-startup transient.** Cold integrals/EMAs, uniform 294 K
    start; warm-start/first-day logic never studied; shutdown->prestart
    integral reset effects never studied.

WEATHER / HETEROGENEITY / LEARNING:
18. **Predictive weather use.** OAT rate/cold-snap pre-charge, solar
    sensors (exist? unused), wind/infiltration, humidity/latent -- none
    used beyond current-OAT rules.
19. **Zone exposure differentiation.** Perimeter/interior, top/ground,
    big/small zones share one target; mean-zone aggregation everywhere
    (`USE_WORST_ZONE=False`); mixed heating+cooling zones on one AHU
    handled only via mild-SAT.
20. **Thermal-mass identification per building.** Greybox/MPC loop never
    closed (brief); prestart learns scalar rates only, no feedforward.
21. **Reward-level:** the 0.12/K comfort slope means ~0.5 K systematic
    excursions cost ~0.06(if-all-zones) -- the frontier in §5 says small
    DYNAMIC excursions trade ~1:4; no policy explicitly budgets regret.

## 7. Ranked suspects for the 2.5x gap (need -46% from 0.00963)

| # | Suspect | Pool | Recoverable (est.) | Evidence |
|---|---|---|---|---|
| 1 | Occupied heating/cooling efficiency (airflow x dT; decoupled reheat; deadband riding; SAT optimum) | 63% occ. energy+carbon | 15-25% of total | §2.5 (comfort 5.7% occ.), §5 kp03 -12% hard-cell, §6.3.1/.4/.8/.12 |
| 2 | Prestart shape (low-flow / partial / below-band / staged-AHU) | 22.5% (spike) | 8-15% of total | §2.4 (max-flow 4 h sprint), AHU logs |
| 3 | Operating point: systematic small excursions (dynamic, occupancy-weighted) | buys 4:1 (§5) | 5-15% of total | §5 kp03; regret median 0 |
| 4 | Arrival soft-start + evening coast | ~10-15% (ramp/coast hrs) | 3-8% of total | §2.5 occupancy ramp; never tried |
| 5 | Deep-cold fb strategy (saturation restraint CODED/UNRUN; min-airflow heat; weekend recharge) | fb1+fb8 ~25% of train | 5-10% of total | §2.3; §6.2 occ_sat; day-over-day depletion |
| 6 | Occupied AHU staging (break all-on block) | part of #1 | 2-6% of total | §2.6 (<1.5% partial states) |
| 7 | Cooling optimality (warmest-zone SAT reset, pre-cool, night drift) | SG/PHX ~20% of train | 2-5% of total | §2.2 energy-dom.; §6.3.8-10 unrun |
| 8 | Weekend/holiday depth WITH smart catch-up | 8.8% weekend | 1-3% of total | naive versions LOSE (§4/§5); physics unsolved |

No single suspect reaches 46% alone; #1+#2+#3 together plausibly do. The
common thread: we hold band-center with maximum air; the leader most
likely rides edges with minimum air.

## 8. On new runs (task item b)

No new runs were executed. Existing CSVs proved sufficient: 1068 probes
window-validate against g4 (§1), the archive gives 25 valid full-horizon
baselines, and `ahu_log.csv` exposes equipment behavior. The machine is
loaded; the two permitted short runs are better spent on §6.2/§6.3
candidates (occ_sat, low-flow prestart, decoupled reheat) than on
re-measuring known structure.

## 9. Source paths (all inspected)

- `smart_buildings/results/building_train_*/ */occ_gated4/reward_components.csv`
  (30), `ahu_log.csv`, `summary.json`; `results/_full30d_archive/` (34);
  all 864-step probe CSVs (1068 window-validated).
- `smart_buildings/baselines/zone_pi.py` (full chain, 1136 lines),
  `occ_gated.py`, `occ_gated2.py`, `occ_gated3.py`, `occ_gated4.py`,
  `occ_gated7.py`, `occ_gated5.py`/`occ_gated6.py`/`occ_round5.py`/`occ_duty.py`/`probe_modes.py`
  (docstrings + keys), `pid_plus.py` (spot: reset params, schedule use).
- `smart_control/reward/setpoint_energy_carbon_regret.py`,
  `base_setpoint_energy_carbon_reward.py`, `electricity_energy_cost.py`,
  `natural_gas_energy_cost.py`, `utils/constants.py`.
- `training_scenarios/*/config.gin` (reward weights, bands, hours),
  `training_set.yaml` (30-scenario composition).
