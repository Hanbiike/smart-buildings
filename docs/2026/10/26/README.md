# 2026-10-26

## Summary

Evening implementation session (filesystem mtimes 21:44–22:36 local)
that built the occupancy-gating line on top of the gen-3.1 chain in four
rounds, plus a duty-cycling experiment and round-5 stubs. Round-1
(`OccGatedPolicy` in `smart_buildings/baselines/occ_gated.py`) added
empty-building shutdown, adaptive prestart, a recoverable overnight
floor, and holiday/early-departure shutdown. Round-2
(`occ_gated2.py`) added an OAT-scheduled hot-water floor plus pump
idle-off. Round-3 (`occ_gated3.py`) fixed two full-episode defects found
in round-1: prestart flicker (latched prestart) and an overshooting
recoverability floor (eco-anchored cap with hysteresis), and restricted
rate learning to prestart steps. Round-4 (`occ_gated4.py`) deleted the
hot-water floor again (one-line no-op) after episodes showed the VAV
discharge cap at 308.15 K clips hotter water without moving airflow
equilibrium. `occ_gated3` was submitted the same evening and stands at
public 0.01330 vs gen-3.1 0.01350 (both COMPLETE; scores and submit time
are draft-preserved — no leaderboard record exists in the repo).
`tools/test_occ_gated.py` covers the line with 22 tests, all passing
(re-verified this session: `Ran 22 tests ... OK`). Teacher dumps for
`occ_gated4` exist for all 30 scenarios
(`smart_buildings/results/<building>/<scenario>/nn_data_occ_gated4/obs_dump.npz`,
dump window 23:20–23:40 local). The evening closed with chat-derived
architecture decisions (marked as such below) that launched the 10-27
slot line: per-zone linear regression with 288 time-slot heads, 1-2-1
Pascal-soft-blended heads, fixed-dim input, supervised-first-then-RL
plus model-based, goal of beating leader Joe Penn (0.00521), and a pivot
to pause rule-based work.

## Status

| Who | Score | State |
|---|---|---|
| us (occ_gated3) | 0.01330 | COMPLETE |
| us (gen-3.1) | 0.01350 | COMPLETE |

Both rows are preserved verbatim from the prior draft. No in-repo
record (JSON, CSV, log, or script) containing `0.01330`, `0.01350`, or a
submit timestamp was found — a repo-wide grep over `*.py`, `*.md`, and
`*.sh` for those scores and for `submit`/`17:20` returned zero matches
outside the day logs — so both scores are kept as draft-preserved, not
as independently verified. The submit is recorded as 17:20 UTC; see
Measured for the timezone inconsistency flag.

## Done

All file bodies below were read in full this session before writing.
All constants are quoted exactly as coded.

### Round-1: `smart_buildings/baselines/occ_gated.py` (mtime 10-26 21:44)

`OccGatedPolicy(DsatGen31Policy)`, `KEY = "occ_gated"`. Driven by live
observations and metadata only (no scenario identities, no weather
files). Four modes from `_decide`, plus the inherited normal path:

- `shutdown`: AHUs off, water loop cold and still (`_shutdown_action`
  drives temp/pressure/water/damper/reheat to -1.0, recirculation to
  +1.0, discrete to 0). Taken when the building is empty and no comfort
  period is in force or approaching.
- `prestart`: full gen-3.1 chain with `_prestart_active = True`, which
  forces `_pi_hours` True. Triggered when steps-to-comfort fall within
  the predicted need (see sizing below).
- `floor_heat` / `floor_cool`: gen-3.1 chain run against a temporary
  floor window (eco band, night-cool ceiling, sensor band, and hold
  steps overridden and restored in `finally`). `HOLD_STEPS` is forced to
  1 so a cooling-run hold cannot extend a `floor_cool` recovery into a
  24 h forced run; `USE_SENSOR_BAND` is forced False so the VAV layer
  judges zones against the floor window instead of the schedule eco
  band (otherwise cold-but-in-eco zones get minimum cold air and no
  reheat and the mean never reaches the floor).
- `normal`: occupied, or empty-but-arrivals-pending inside the grace
  window — the untouched gen-3.1 chain.

Round-1 constants (class attributes, verbatim):

- `OCC_EPS = 0.5` — occupied iff the occupant signal exceeds its
  running minimum by 0.5. `_occupied` seeds the running minimum at
  `min(raw, 0.0)` so an episode that starts occupied still reads
  occupied, and tracks the minimum down afterward, which decodes both
  raw-count and normalized signals.
- `EMPTY_GRACE_STEPS = 12` — 1 h into comfort hours with nobody still
  reads `normal` (arrivals pending); after that, an empty building in
  comfort hours shuts down (holiday case).
- `RATE_INIT_K_PER_STEP = 0.08` (~1 K/h), `RATE_EMA_ALPHA = 0.2`,
  `RATE_MIN = 0.0005`, `RATE_MAX = 0.3` — separate heat/cool recovery
  rates adapted by EMA from observed mean-zone-temperature deltas.
- `PRESTART_SAFETY = 1.3`, `PRESTART_MARGIN_STEPS = 6` (+30 min),
  `MAX_PRESTART_STEPS = 48` (never more than 4 h early).
- `FLOOR_CAPABILITY = 0.7` — the floor uses 70% of predicted
  capability: `floor_heat = heat_lo - 48 * rate_heat * 0.7`
  (symmetric on the cool side). `FLOOR_DEADBAND_K = 0.5`.
- Holiday/early-departure rule: occupancy rises monotonically from
  06:00 (arrivals only, no departures before 13:00) and nobody can
  remain past 19:05 (both per the module docstring), so a building that
  was occupied and is now empty stays empty, and a still-empty building
  past the grace window is a holiday — both shut down immediately.

Side branch in the same file (documented briefly; experimental):

- `OccSatPolicy` (`KEY = "occ_sat"`): deep-cold saturation restraint.
  When full heat runs for `STALL_STEPS = 24` (2 h) without
  `STALL_MIN_RISE_K = 0.5` of mean-zone recovery, and OAT is below
  `SAT_ARM_OAT_K = 263.15` (-10 C) with heat demand above
  `SAT_MIN_HEAT = 0.5`, the plant is declared saturated and heating
  dampers are capped at `SAT_DAMPER_CAP = 0.5` so the boiler heats zones
  instead of feeding the mandatory 30%-outside-air stream. Releases on
  recovery or dropped demand. The class docstring requires
  full-episode validation before any submission use.
- `OccSat03Policy` (`KEY = "occ_sat03"`, cap 0.3) and
  `OccSat07Policy` (`KEY = "occ_sat07"`, cap 0.7): tighter/looser
  damper-cap variants of the same restraint.

### Round-2: `smart_buildings/baselines/occ_gated2.py` (mtime 10-26 21:45)

Kept in a separate module so round-1 batch results stay comparable.
`_Round2Mixin` re-implements `get_action` (mode decision runs exactly
once; the chain is entered via `super(OccGatedPolicy, self)` so the
comfort grace is not double-counted) and applies two additions:

- F. Cold-weather hot-water floor. Rationale (module docstring): the
  water reset collapses to zone-PI demand in near-band steady state
  (~0.3), so lukewarm water forces high airflow for the same heat, and
  high airflow feeds the 30% outside-air ventilation load — over half
  the gas in occupied Fairbanks hours. The loop is therefore held at an
  OAT-scheduled floor while heating in cold weather; the zone PI
  self-balances to lower dampers. Schedule:
  `floor = clip((WATER_FLOOR_OAT_HI - OAT) / span, 0, 1)` with
  `WATER_FLOOR_OAT_HI = 273.15`, `WATER_FLOOR_OAT_LO = 243.15`
  (span = 30, matching the prior draft's `clip((273.15-OAT)/30)`
  verbatim). Gates: applies only when heat-active, i.e. zone demand
  `_max_heat > WATER_FLOOR_MIN_HEAT = 0.05` or loop fraction above
  `WATER_FLOOR_MIN_FRAC = 0.10`.
- G. Pump idle-off via `DsatG4WPumpIdleOffPolicy` (imported from
  `zone_pi`): pump head + discrete off when no zone wants heat,
  covering occupied no-heat hours the shutdown does not.
- Assemblies: `OccGated2Policy(KEY = "occ_gated2")` and
  `OccSat2Policy(KEY = "occ_sat2")` = round-2 on the sat branch
  (damper cap 0.5).

### Round-3: `smart_buildings/baselines/occ_gated3.py` (mtime 10-26 21:55)

`_Round3Mixin` on top of round-2, fixing the two round-1 full-episode
defects named in the module docstring:

1. Prestart flicker — the need-vs-time threshold has no hysteresis, so
   AHUs chatter through prestart, including a shutdown blip at 06:00
   with occupants arriving. Fix: `_prestart_latched`. Once prestart
   triggers it latches until comfort hours or occupancy; `occupied`
   and `comfort_now` both clear the latch.
2. Recoverability floor too hot in saturation — in deep cold the plant
   cannot reach the computed floor, so `floor_heat` burns maximum gas
   all night for a Monday arrival that is cold anyway. Fix: the floor
   window is bounded at `eco_low - 1 / eco_high + 1`
   (`FLOOR_HEAT_CAP_BELOW_ECO = 1.0`,
   `FLOOR_COOL_CAP_ABOVE_ECO = 1.0`; note the cool side also uses
   `min`, i.e. it never holds cooler than `eco_high + 1` either), with
   `FLOOR_ENTER_DB_K = 0.5` / `FLOOR_EXIT_DB_K = 0.5` hysteresis: entry
   needs a 0.5 K breach, exit needs 0.5 K of recovery, and an active
   floor mode persists inside the deadband instead of chattering.
- Prestart-only rate learning: `_update_rates` returns early (keeping
  `_last_mean` tracking so the first prestart delta stays genuine)
  unless `_prestart_active`. Occupied steady-state micro-rises — strong
  heat demand holding against losses with near-zero delta — used to
  drag the EMA toward zero and push the overnight floor to the band
  edge within days; prestart deltas are unoccupied and plant-driven, so
  they size the next prestart correctly.
- Pump idle-off and the hot-water floor are inherited unchanged from
  round-2 at this stage.
- Assemblies: `OccGated3Policy(KEY = "occ_gated3")` and
  `OccSat3Policy(KEY = "occ_sat3")`, each initializing
  `_prestart_latched = False`.

### Round-4: `smart_buildings/baselines/occ_gated4.py` (mtime 10-26 22:14)

`OccGated4Policy(OccGated3Policy)`, `KEY = "occ_gated4"`. The entire
round is a one-line no-op:

- `_apply_water_floor` returns `None` — the OAT-scheduled hot-water
  floor is disabled; the underlying water reset keeps the loop just hot
  enough.
- Rationale (module docstring): round-2/3 episodes showed the floor
  adding gas without reducing airflow. The VAV coil explains why:
  discharge air is capped at 308.15 K, so once the coil hits the cap,
  hotter water is clipped and delivers no extra heat per unit air — PI
  equilibrium and ventilation load do not move, while the hotter loop
  pays more tank dissipation and cycling.

### Duty experiment: `smart_buildings/baselines/occ_duty.py` (22:14)

`OccDutyPolicy(OccGated4Policy)` (`KEY = "occ_duty"`), experimental
per its docstring. Rationale: the 30% outside-air ventilation floor
applies whenever an AHU runs (over half the gas in occupied Fairbanks
hours, over half the chiller load in Phoenix hours), while zone thermal
mass smooths short air-side interruptions. Mechanism H: an AHU whose
zones all sit inside `[lo + MARGIN_K, hi - MARGIN_K]` with
`DUTY_MARGIN_K = 0.7` alternates `DUTY_ON_STEPS = 4` (20 min) on /
`DUTY_OFF_STEPS = 2` (10 min) off; any zone leaving the inner band
restarts ON immediately. Applies only in occupied steady state — never
during latched prestart, floor hold, or shutdown (a stale cycle after
shutdown self-heals via the margin check).
`OccDutyLeanPolicy` (`KEY = "occ_duty_lean"`): longer coast, 4 on / 4
off (20 min / 20 min).

### Round-5 stubs: `smart_buildings/baselines/occ_round5.py` (22:36)

Three experimental variants on `OccGated4`, all flagged in the module
docstring as needing full-episode proof: `OccDutyMicroPolicy`
(`occ_duty_micro`, duty 6 on / 1 off — the 4/2 and 4/4 variants cost
more comfort ripple than ventilation saved, so a single 5-min coast),
`OccCoolShiftPolicy` (`occ_coolshift`, +0.3 K warm-edge ride while
cooling-dominated, capped at `min(base + shift, hi - 0.1)` with
`COOL_SHIFT_K = 0.3`; the symmetric 1.0 K shift was rejected for
comfort), `OccFloor287Policy` (`occ_floor287`, overnight floor cap one
kelvin lower at `eco_low - 2 = gen31's hold`,
`FLOOR_HEAT_CAP_BELOW_ECO = 2.0`).

### Tests and dumps produced

- `tools/test_occ_gated.py` (mtime 10-26 22:14): single file, 6 test
  classes, 22 tests (full bodies read; names and assertions in the
  Appendix). Verified this session with the project venv
  (`.venv/bin/python -m unittest tools.test_occ_gated`):
  `Ran 22 tests in 0.032s — OK`.
- Teacher dumps: 30 `nn_data_occ_gated4` directories under
  `smart_buildings/results/<building>/<scenario>/`, each containing
  `obs_dump.npz` — 30/30 present. Dump mtimes span 10-26 23:20–23:40
  (5 files each at 23:20, 23:26, 23:30, 23:36, 23:38, 23:40), so the
  prior draft's "23:20" is the start of the dump window, confirmed.
  Example size: 8,959,795 bytes
  (`building_train_2/atlanta_3/nn_data_occ_gated4/obs_dump.npz`).
  `*.npz` is git-ignored (repo `.gitignore` lists `*.npz`), and
  `smart_buildings/results/` is git-ignored as a whole, so these live
  on disk only. Per-scenario `occ_gated4/summary.json` and
  `comparison.json` files sit alongside (example numbers in Measured).

## Measured

- Round-1 train-subset regression: +2..+15% vs gen31 (cost —
  draft-preserved verbatim). No in-repo record: no `occ_gated`
  (round-1) result directory exists under `smart_buildings/results/`
  (only `occ_gated4` + `nn_data_occ_gated4` were found), and no file
  contains per-subset numbers. Kept as stated, marked unverified here.
- `occ_gated3` public 0.01330, submit recorded as 17:20 UTC
  (draft-preserved verbatim). Honesty flag, not resolved by guessing:
  the recorded local build window is 21:44–22:36 (`occ_gated.py`
  through `occ_round5.py` mtimes) with tests at 22:14 and dumps at
  23:20–23:40 — a submit at 17:20 UTC on the same clock would precede
  the code it submits. System locale/timezone evidence is mixed (shell
  `ls` renders Russian month names; git log stamps `+0600`), so this
  is most likely a timezone mix between the submit log and the
  workstation clock. Flagged; the numbers themselves are untouched.
- `test_occ_gated`: 22/22 pass, verified by execution this session
  (see Done). Fixture windows used by the tests (from `_fake_building`,
  read in full): comfort (294.0, 297.0), eco (289.0, 298.0),
  morning/evening hours 6/19.
- Dump inventory: 30/30 `obs_dump.npz` present under
  `smart_buildings/results/*/nn_data_occ_gated4/`; mtime spread
  23:20–23:40 on 10-26 as tabulated above.
- Worked example, `building_train_2/atlanta_3/occ_gated4/summary.json`
  (read in full): `total_reward -37.97061657878294` over
  `total_steps 8640`, `avg_reward -0.0043947472892109886`,
  `avg_occupancy 33.445023148148145`,
  `avg_zone_temp 296.0969554267459`, weights
  productivity/energy/carbon 0.6/0.2/0.2,
  `total_normalized_productivity_regret -6.944230087365668`,
  `total_normalized_energy_cost 130.9073710972034`,
  `total_normalized_carbon_emission 38.11302159893967`. The sibling
  `comparison.json` carries the same `occ_gated4` block. Train-2
  scenario summaries share mtimes 10-26 23:26–23:27.

## Decisions

- Flicker finding: re-deciding prestart every step breaks round-1
  (AHU chatter, 06:00 shutdown blip at arrival) → latched prestart.
  Consequence: round-3.
- VAV-cap finding: the 308.15 K discharge cap clips hotter water, so
  the hot-water floor pays tank losses for nothing → floor removed.
  Consequence: round-4 one-liner.
- Floor-cap finding: unbounded recoverability floor burns max gas all
  night in deep cold for an arrival that stays cold → eco-anchored cap
  (eco_low - 1 / eco_high + 1) with 0.5 K hysteresis. Consequence:
  round-3 `_floor_window` + `_decide` rework.
- Rate-pollution finding: occupied steady-state micro-rises drag the
  recovery EMA to zero and lift the floor to the band edge within days
  → learn rates from prestart steps only. Consequence: round-3
  `_update_rates` gate.
- Chat-derived architecture decisions from that evening (no file
  backing — marked chat-derived, kept because they launched the 10-27
  slot line; details live in the 10-27 log, not here): user proposed
  per-zone linear regression with 288 time-slot heads (one per 5 min of
  day); soft blend of heads N-1/N/N+1 with weights 1-2-1 (Pascal
  pyramid); fixed-dim input vector; user confirmed
  supervised-first-then-RL plus model-based; session goal set to beat
  leader Joe Penn (public 0.00521, from memory); pivot agreed — pause
  rule-based work, compact trainable RL first attempt.

## Open

- Slot/BC/PPO follow-through — done 10-27 (closed below the teacher;
  see the 10-27 log).
- Headroom probes — done 10-27 (all ≤ 0; see the 10-27 log).
- Saturation branch (`occ_sat*`, damper caps 0.5/0.3/0.7, arm OAT
  263.15 K): still experimental, needs full-episode validation before
  any submission use (per its own docstring).
- Round-5 variants (duty-micro 6/1, cool-shift 0.3 K, floor287 cap
  2.0): need full-episode proof (per their docstring); the 4/2 and 4/4
  duty variants are already known to cost more comfort ripple than
  ventilation saved.
- Score/timezone hygiene: the 17:20 UTC submit stamp vs the 21:44+
  local build window remains unexplained (flagged in Measured).

## Appendix

All 22 tests in `tools/test_occ_gated.py` by exact name, each with its
assertion in one line. Grouping note: the file holds 6 classes sized
7/1/4/5/4/1 = 22. The prior draft's "8/6/5/3" matches on the 8 (7 + 1)
and the 5 (round-3); the remaining 9 split 4 (round-2) + 4 (duty) + 1
(round-4) in the current single-file layout, so the old 6/3 cut no
longer maps to file boundaries. The draft's per-group paraphrases
("asof", "hws unchanged", "band metrics", "cooling discharge only")
match no test name or body in the file — flagged, prior text preserved
only by this note.

### Round-1 family (8): `OccGatedTest` (7) + `OccSatTest` (1)

- `test_night_empty_shuts_down` — empty 02:00 night at 292 K gives
  discrete `[0, 0]`, water -1.0 (cold), damper -1.0 (min).
- `test_occupied_comfort_runs` — occupied 10:00 comfort at 292 K turns
  AHU discrete output 1 (on).
- `test_holiday_shuts_down_after_grace` — empty comfort-hour obs held
  for `EMPTY_GRACE_STEPS + 1` steps ends at discrete `[0, 0]` (holiday).
- `test_departed_afternoon_shuts_down` — occupied morning then empty
  15:00 same day gives `[0, 0]` (early departure stays empty).
- `test_prestart_runs_before_comfort` — cold (288 K) 05:00 Monday with
  `comfort_mode_soon` and OAT 250 K runs the AHU (discrete 1).
- `test_occupied_late_night_runs` — occupied 02:00 night runs the AHU
  despite the hour.
- `test_normalized_occupancy_decodes_from_running_min` — after the
  running minimum settles at -0.9, obs -0.9 stays shut down while -0.2
  (min + 0.7 > `OCC_EPS`) in comfort hours runs.
- `test_saturation_caps_heating_dampers` — with `_saturated` forced and
  heat demand on VAV_1, the heating damper stays ≤ `SAT_DAMPER_CAP`.

### Round-2 (4): `OccGated2Test`

- `test_cold_floor_raises_water_while_heating` — occupied heating at
  OAT 250 K pins the water fraction ≥ 0.7 (floor active).
- `test_mild_weather_leaves_water_reset` — same call at OAT 285 K keeps
  the water fraction < 0.7 (no floor).
- `test_no_heat_no_floor` — in-band (296 K) at OAT 250 K leaves the
  water fraction ≈ 0.0 (no heat demand, floor gated off).
- `test_occ_sat2_carries_damper_cap` — `OccSat2Policy.SAT_DAMPER_CAP`
  is 0.5 and forced saturation caps its heating damper at ≤ 0.5.

### Round-3 (5): `OccGated3Test`

- `test_prestart_latches_until_comfort` — prestart fires cold, then
  zones recover to 294 K with need < 12 steps left, yet the AHU stays
  on (latch, no re-evaluation shutdown).
- `test_floor_bounded_by_eco_band` — at the slowest rate
  (`_rate_heat = 0.0005`) the computed `floor_heat` stays ≤ 288.0
  (= eco_low 289.0 − 1).
- `test_floor_hysteresis` — weekend-night deep breach enters
  `floor_mode == "heat"`; mid-deadband recovery stays heating; above
  floor + 0.6 K the AHU shuts down (`[0, 0]`).
- `test_occ_sat3_imports` — `OccSat3Policy(...).KEY == "occ_sat3"`.
- `test_rates_learn_only_in_prestart` — occupied steady-state
  micro-rise (294.0 → 294.001) leaves `_rate_heat` unchanged, while the
  same-sized rise under `_prestart_active` moves the EMA to
  `0.8 * before + 0.2 * 0.05`.

### Duty + round-4 (5): `OccDutyTest` (4) + `OccGated4Test` (1)

- `test_cycles_off_in_band` — in-band occupied over
  `DUTY_ON_STEPS + DUTY_OFF_STEPS` steps runs ON then coasts OFF.
- `test_breach_restarts_on_phase` — after the ON phase, a zone at
  294.5 K (outside the inner band) forces the AHU back to 1 at once.
- `test_no_cycling_out_of_band` — out-of-band (292 K) occupied over a
  full cycle plus 2 steps never coasts (always 1).
- `test_lean_variant` — `OccDutyLeanPolicy` uses `(DUTY_ON_STEPS,
  DUTY_OFF_STEPS) == (4, 4)`.
- `test_water_floor_disabled` — cold-OAT occupied small shortfall
  leaves the water fraction < 0.77, where round-2/3 would pin the floor
  (~0.77 here).
