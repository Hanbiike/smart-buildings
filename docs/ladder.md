# Improvement ladder (worst -> best)

Clean-room reproduction guide: every rung lists the exact mechanism,
its parameters, why it moved the score, and how to run it. Rungs are
ordered by public/train score, worst first. Failed rungs are included
with their decisive numbers so nobody re-tests them.

Score convention: `loss = mean(|per-step reward|)` over 8640-step
episodes (30 days x 5 min). Computed from CSV as
`abs(df.agent_reward.sum()) / len(df)`.
`train-30` = the 6 released buildings x 5 weathers
(`building_train_{1,2,4,5,7,8}`); `subset-8` = the 8-pair probe subset;
`transfer-12` = 3 sampled buildings x 4 unseen weathers;
`public` = Kaggle public LB (mean over 30 hidden pairs).
All `occ*`/`probe_*`/`dsat_*`/`slot_*`/`bc_*` names are `POLICY_MAP`
keys in `smart_buildings/evaluate.py`.

Reward recap (fixed for everyone): weights (comfort, energy, carbon) =
(0.6, 0.2, 0.2); comfort linear to 5K then quadratic unbounded;
electricity TOU peak/night = 1.25x (arbitrage dead); plant efficiencies
constant (boiler const-eff, chiller COP 3.5). Comfort slope (~0.18/K)
>> energy slope (~0.002/K). Full tables: `docs/2026/10/20/`.

## Rung 0 — shipped baselines

`bang_bang`, `pid`, `pid_water` (`PIDWaterResetPolicy`), `fixed_setpoint`.
Measured on 6 train scenarios: pid_water 0.020-0.212, bang_bang 0.077-0.308
(occ4 is 4-12x better). On transfer, pid_water is 2-3x worse than occ4
everywhere. `fixed_setpoint` wins only hot-humid corners.

`pid_water` mechanism (also the underneath of everything later): hot-water
reset = max(mean-zone shortfall / `RESET_FULL_K=4.0`, air-effort quantile
`RESET_EFFORT_QUANTILE=0.25` x `RESET_EFFORT_WEIGHT=1.0`), deadband 0.25K,
integrator built then rejected (`RESET_INTEGRAL_GAIN=0.0` — widens worst-zone
spread, the unbounded term punishes exactly that).

## Rung 1 — transfer / BaselinePolicy (public 0.06439)

`smart_buildings/baselines/zone_pi.py:BaselinePolicy`. Score from the class
docstring (earlier account, same competition).

Mechanism: sensor comfort bands, mean-zone control, OAT decoded from its
normalization stats. Found the organizer economizer bug: the AHU OAT sensor
arrives in raw kelvin, the starter code decoded it as normalized, so the
economizer saw ~3000K and never opened (`ZonePIEconFixPolicy` fixes it by
calling the organizer economizer with kelvin).

## Rung 2 — zone_pi k10 (public 0.01853)

Per-zone PI to band centre in work hours. One signal `u` in [-1,1] drives
both actuators: damper `= MIN_DAMPER + (1-MIN_DAMPER)*|u|*gain`
(`MIN_DAMPER=0.1`, `HEAT_DAMPER_GAIN=1.0`), reheat `= max(0,-u)`.
`KP=0.1`, `KI=0.005` (later `0.07/0.0035` in the dsat config), conditional
integration (freeze on saturation), prestart 1h (later 2h), forced-on AHUs
at `FORCED_SAT=-0.7` (~15C) / `FORCED_PRESSURE=-1.0`, cooling-run hold
`HOLD_STEPS=288`, `NIGHT_COOL_CEILING=True` (nights use eco-low/comfort-high).

Why it won: per-zone targeting instead of mean-zone + working economizer.

## Rung 3 — gen-3.1 dsat_gen31 (public 0.01350)

Notebook: `notebooks/smart-buildings-submission-gen31.ipynb`.
Stack (each a tested layer, full registry in `docs/2026/10/21/`):

- Demand-SAT: per-AHU supply setpoint from mean zone-PI demand `d`;
  `frac = clip(d/RAMP)`, `SAT = 1.0 + frac*(COLD_SAT-1.0)` (net heating
  demand -> SAT max = chiller off).
- pressmin: every running AHU pinned to static pressure -1.0 (min fan).
- Down-shift: zone target = centre - up to 1.0K (cap 0.4x band) when OAT <
  centre (heating economy). Up-shift in heat REJECTED (singapore +25%).
- Occupant gate: after arrival+2h, PI hours only while occupants reported
  (skips holidays/empty days).
- Mixedness-SAT: per-AHU 24h EMAs of heating share / cooling activity;
  mixed handlers capped at `MILD_SAT=-0.4` (no chill-then-reheat waste),
  sustained cooling-dominant (>0.7) saturate `DEEP_SAT=-1.0`.
- Night heating drop 2K, prestart 2h, water reset + pump idle-off underneath
  (pump discrete+head off when water frac < 0.05 and max_heat < 0.05).

Independent confirmation: Bayesian optimization over
(kp, ki, water, mild, hgate, shift, ndrop) converged to the hand values
(`results/g4bo_params.json`, git-ignored): kp 0.025, ki 0.0035,
water 0.76, mild -0.38, hgate 0.05, shift 1.0, ndrop 2.0.

## Rung 4 — occ round-1 (REGRESSION: +2..15% vs gen31)

`OccGatedPolicy` (`smart_buildings/baselines/occ_gated.py`, key `occ_gated`):
four modes from occupancy + schedule. Shutdown (plant off) when unoccupied
and recoverable; floor hold otherwise; prestart before arrival; normal when
occupied. Constants: `OCC_EPS=0.5`, `EMPTY_GRACE_STEPS=12`,
`RATE_INIT_K_PER_STEP=0.08` with EMA `ALPHA=0.2`, bounds [0.0005, 0.3],
`PRESTART_SAFETY=1.3`, `PRESTART_MARGIN_STEPS=6`, `MAX_PRESTART_STEPS=48`
(4h), `FLOOR_CAPABILITY=0.7`. Side branch: `OccSatPolicy` (`occ_sat`) +
tighter damper caps `occ_sat03` (0.3) / `occ_sat07` — saturation restraint,
did not ship.

Why it lost: prestart flicker (re-deciding every step) + an over-hot floor.
Kept the architecture, fixed the bugs in round-3.

## Rung 5 — occ round-2 (hot-water floor)

`OccGated2Policy` (`occ_gated2.py`, key `occ_gated2`): OAT-scheduled
water-temp floor under the reset: `floor = clip((273.15-OAT)/30)`
(`WATER_FLOOR_OAT_HI=273.15`, full at 243.15K), applied only when
heat-active (`max_heat > 0.05` or water frac > 0.10). Theory: hotter water
cuts airflow needs. Verdict: adds gas without cutting airflow (see round-4).

## Rung 6 — occ round-3 -> occ_gated3 (public 0.01330)

Fixes round-1: latched prestart (decide once, no flicker), bounded
hysteretic floor (`FLOOR_ENTER/EXIT_DB_K=0.5`, caps eco_low-1 /
eco_high+1), rate EMA learns from prestart steps only, idle pump-off in
the chain. First policy to beat gen31 on public.

## Rung 7 — occ round-4 -> occ_gated4 (train 0.00963, public 0.01323)

One-line change (`_apply_water_floor` -> no-op): the VAV coil analysis
showed discharge air is capped at 308.15K, so once the coil hits the cap,
hotter water is clipped and delivers no extra heat per unit air — the PI
equilibrium doesn't move while the hotter loop pays tank losses. Measured:
round-3 with floor vs without, without wins on full train-30.
Train-30: mean 0.009629, median 0.007167. Public 0.01323 (hidden/train 1.37).

## Failed rungs (do not rebuild)

**occ_gated5 (-2K heating target): train +1.4% (subset), public 0.01352
(+2.2% vs occ4).** `HEAT_OFFSET_K=-2.0` shifts the comfort window
(294 -> 292). Comfort slope >> energy slope, so the tradeoff loses on
train AND hidden. Only value was attribution: hidden does not reward
efficiency more than train.

**Duty cycling (`occ_duty` 4on/2off, `occ_duty_lean` 4/4, `occ_duty_micro`
6/1, margin 0.7K):** coasting with fan off saves ventilation+fan but the
comfort ripple and the ON-sprint cost more. All variants <= 0.

**Setpoint / SAT / margin / econ / load-shift probes (~15, `probe_*`):**
comfort window +/-2-4K (`probe_sp`, knobs `SP_DH`/`SP_DC`), AHU SAT bias
(`probe_sat`, knob `SAT_BIAS`), prestart cap/safety/margin (`probe_margin`,
knobs `M_CAP`/`M_SAFE`/`M_MARGIN`), forced-AHU economizer (`probe_econ`),
weekend floor instead of shutdown (`probe_wfloor`), cooling +0.3K shift
(`occ_coolshift`), floor cap -2K (`occ_floor287`), TOD precool -1K 00-10h /
coast +1K 12-19h (`probe_tod`). Measured vs occ4: wfloor +0.0% (n=2),
econ -0.1% (n=4), tod +5.1%, sat +9.5%, sp +1/+19%, margin +0.0%.
Every one <= 0 on the probe subsets. This is the evidence for
"train-optimal": nothing left on any tested axis.

**Slot/BC line (all below the occ4 teacher):** per-5-min-slot linear heads
(288 heads, blend 1-2-1 over N-1/N/N+1; `SlotFeatures`: FZ room+10 extra,
FA=14, FG=11), `tools/slot_train.py` (ridge 1.0, masked shutdown + OAT
feature). Masked-BC: subset-8 0.0368 vs teacher 0.0175 (mild 1.4x,
extremes 2.2-2.6x). Hybrid 0.009-0.27 (+243%), DAgger worse than BC
(+212%; teaches false gains from the stateful teacher). Failure
concentrates on houston/tucson/b3-atlanta (heating-head errors) and tampa.
Root cause: the teacher is stateful (PI integrals, latched prestart,
hysteresis); a memoryless linear cannot clone it, and compounding errors
explode. Per-scenario tables: `docs/2026/10/27/`.

**PPO on slots (diverged, closed):** `tools/slot_ppo.py`, GAE + V-pretrain +
per-minibatch KL (lr 3e-5, KL 0.01, per-dim logprob norm). Math and rollout
tests green, runs diverged 0.040 -> 0.076 with negative explained variance.
Continuous PPO is non-viable on this sharp comfort landscape (SNR ~ 1/s^2).

**Greybox/MPC (open, never closed):** lumped RC per zone,
`dT = a*(Tout-T) + b*flow*(Tsup-T) + c*flow*reheat*(Twater-T) + d*occ + e`,
RLS online ID + LOBO-validated predictors (`greybox.py`,
`thermal_greybox.py`, `causal_greybox.py`, `equipment_observer.py`). The
control loop was never closed; kept as plan B for unseen-buildings
generalization. Full registry: `docs/2026/10/23/`, `docs/2026/10/24/`.

## Rung 8 — transfer-suite method

Hidden = unseen buildings x unseen climates, so train scores don't predict
hidden. Local substitute: sample new buildings from the repo generator
(`tools/gen_transfer.py`, seeds 50000+i, paper ranges, materials rotated
over the 4 constructions), pair with unseen weather (minneapolis = unseen
station; atlanta_4 / singapore_2 = unseen months of seen stations).
Validity: <= 10 zones/AHU and moderate size (a 125-zone monster simulates
2h and is out of spec — dropped as xfer_4).
Current suite: 3 buildings x 4 weathers = 12 scenarios
(`smart_buildings/transfer_scenarios/`, git-ignored, regenerable).

Finding: occ4 holds 2-3x over pid_water everywhere (no uniform transfer
failure), but big-zone buildings bleed comfort in tails (xfer_3 January:
comfort -0.062, 25x worse than train; Singapore cooling -0.015).

## Rung 9 — trim-and-respond pressure (ASHRAE G36 style)

Root cause (proven via obs dump): pressmin starves airflow; water loop at
350K max, dampers wide open, yet zones at 13-18C. Only more air helps
(discharge capped). `probe_ptr` (env-tuned via `PTR_UP/DOWN/MAX/DSAT/DB/
DB_SLOPE/DB_REF/PERSIST/OCC_ONLY`): per-AHU pressure integrator,
+0.25/step while a zone is out of band with damper > 0.85, -0.03/step when
all dampers < 0.5.

- v1 (defaults, MAX +0.5): heat tail -6.5%, cooling tail -33%, but fan +374%
  and xfer_1 +17% (transient misfire).
- v2 (+persist 4, DB 1.0K, MAX 0.0): heat -28%, cooling -34%, xfer_1 +6%,
  but train_1 deep-cold +25% (ventilation at -30C).
- v3 (DB 2.0K): heat -28%, cooling -24%, xfer_1 +5.5%, train_1 +19%.
  Fixed DB can't separate edge-riding from starvation.
- v4 (+OAT-aware DB: 1.0 + 0.1/K over 10K |OAT-band|): heat -28%,
  cooling -34%, train_1 +0.9%, xfer_1 +3.8%. -> occ_gated6.
- v5 (occupied-only trigger, no prestart): heat -23%, cooling -34%, ALL
  misfire pairs bit-equal to occ4. -> occ_gated7.

`probe_guard` (force normal when occupied + zone >3K out,
`GUARD_K=3.0`/`GUARD_RELEASE_K=2.0`) is a structural no-op: occupied mode
is already normal. Never shipped.

## Rung 10 — occ_gated6 (transfer -13.0%, submitted PENDING)

`smart_buildings/baselines/occ_gated6.py`: v4 with fixed constants
(`PTR_DB=1.0`, `PTR_DB_SLOPE=0.1`, `PTR_DB_REF=10.0`, `PTR_PERSIST=4`).
Train-30 0.009695 (+0.7% vs occ4, cold-only); transfer-12 -13.0%
(xfer_3: -20% to -34% in ALL weathers, 42 runs). Notebook assembled +
300-step exec check + kernel pushed (v1 ERRORed on missing
`STEPS_PER_HOUR`, fixed in v2 — see `docs/2026/10/27/`).

## Rung 11 — occ_gated7 (6/6 clean, submitted PENDING)

`smart_buildings/baselines/occ_gated7.py`: v5 (occupied-only, same PTR
constants). Tails -23%/-34%, zero regressions anywhere measured (6/6
misfire checks bit-equal). Current best, awaiting public score.

## Submission log

| Policy | Kernel | State | Public |
|---|---|---|---|
| gen-3.1 (dsat_gen31) | smart-buildings-submission-gen31 | COMPLETE | 0.01350 |
| occ_gated3 | (10-26) submitted 17:20 UTC | COMPLETE | 0.01330 |
| occ_gated4 | occ4 notebook | COMPLETE | 0.01323 |
| occ_gated5 | occ5 notebook | COMPLETE | 0.01352 |
| occ_gated6 | occ6 v2 (v1 ERROR) | PENDING | — |
| occ_gated7 | occ7 v1 | PENDING | — |
| leader (Joe Penn) | — | LB | 0.00521 |

Scoring queue is slow (~14h). Watch the daily submission budget
(1 left on 10-27).

## Reproduction cookbook

```bash
./.venv/bin/python -B tools/fullrun.py /tmp/r 8640 8 1500 \
  occ_gated7 building_train_1/atlanta_2,...,building_train_8/singapore_7
# loss per run = abs(sum(agent_reward))/8640 from reward_components.csv
./.venv/bin/python -B tools/gen_transfer.py   # 16 scenarios (drop xfer_4)
./.venv/bin/python -B tools/fullrun_t.py /tmp/x 8640 8 1500 \
  occ_gated7 <pairs...> smart_buildings/transfer_scenarios
./.venv/bin/python -B tools/assemble_notebook.py <src> <dst> OccGated7Policy \
  DsatWaterPumpIdleOffPolicy DsatG4WPumpIdleOffPolicy 'occ_gated:<consts>' \
  occ_gated:OccGatedPolicy occ_gated2:_Round2Mixin occ_gated2:OccGated2Policy \
  occ_gated3:_Round3Mixin occ_gated3:OccGated3Policy \
  occ_gated4:OccGated4Policy occ_gated7:OccGated7Policy
# <consts> (STEPS_PER_HOUR) is REQUIRED - kernel ERRORs without it.
# Exec-check the notebook for 300+ steps (must cover night/unoccupied).
```
