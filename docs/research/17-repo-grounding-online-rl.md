# Repository grounding for Online RL over occ_gated7

Date: 2026-11-01
Status: draft
Scope: every repository fact that constrains the design of Online RL built on
top of the rule-based rudder `occ_gated7`: scoring, the policy registry, the
rule stack and its fixed constants, simulator physics caps, reward economics,
the transfer-evaluation method, submission packaging, and documentation
conventions. No simulator episode, training job, or benchmark was run for
this report; all claims come from the inspected sources below.

## Sources

| # | Source | Lines read | Used for |
|---|--------|------------|----------|
| S-A | `AGENTS.md` | all (59 lines) | scoring, episode contract, drivers, simulator facts, conventions |
| S-B | `docs/ladder.md` | all (255 lines) | score convention, rung mechanisms, failed levers, transfer method, submission log, cookbook |
| S-C | `docs/research/06-repository-analysis.md` | 1-120 | interfaces, deployable boundary, evidence standard, gap arithmetic |
| S-D | `docs/research/10-control-research-synthesis.md` | 1-120 | control families, predictive formulation, horizon guidance |
| S-E | `smart_buildings/evaluate.py` | 218-377 | `POLICY_MAP` registry keys |
| S-F | `smart_buildings/baselines/occ_gated7.py` | all (85 lines) | rudder: occupied-only trim-and-respond constants and logic |
| S-G | `smart_buildings/baselines/occ_gated4.py` | all (24 lines) | rudder parent: water-floor removal rationale |
| S-H | `tools/fullrun.py` | 1-60 | batch driver CLI and loss computation |
| S-I | `tools/assemble_notebook.py` | 1-60 | notebook assembly mechanism and CLI |
| S-J | scout handoff (prior result 1) | evidence list | doc-formatting standard (numbering, headers, R1-R9, `docs/plan` absent) |

## 1. Competition scoring and episode contract

1. Leaderboard = mean |per-step reward| over 30 hidden (building, weather)
   pairs; lower is better. Provenance: `AGENTS.md`.
2. Full episode = 8640 steps = 30 days x 5-minute steps x 300 s/step.
   Short runs are not comparable to full runs. Provenance: `AGENTS.md`;
   step length 300 s from `docs/research/06-repository-analysis.md`.
3. Loss from CSV: `abs(df.agent_reward.sum()) / len(df)` computed on
   `reward_components.csv`. Provenance: `AGENTS.md`, `docs/ladder.md`,
   `tools/fullrun.py`.
4. Coverage labels must never be mixed unlabeled: `train-30` = 6 released
   buildings x 5 weathers (`building_train_{1,2,4,5,7,8}`); `subset-8` = the
   8-pair probe subset; `transfer-12` = 3 sampled buildings x 4 unseen
   weathers; `public` = Kaggle public LB mean over 30 hidden pairs.
   Provenance: `docs/ladder.md`.
5. The simulator is deterministic: same policy + scenario = bit-identical
   result. Deterministic reruns are not independent statistical replicates.
   Provenance: `AGENTS.md`, `docs/research/06-repository-analysis.md`.
6. Python is always `./.venv/bin/python -B`, never system python.
   Provenance: `AGENTS.md`.

## 2. Policy registry and naming

7. Policies live in `smart_buildings/baselines/` and are addressed by
   `POLICY_MAP` keys in `smart_buildings/evaluate.py`. Every `occ*`,
   `probe_*`, `dsat_*`, `slot_*`, `bc_*` name in the ladder is such a key.
   Provenance: `AGENTS.md`, `docs/ladder.md`.
8. The inspected registry window (`smart_buildings/evaluate.py` lines
   218-377) contains 133 keys and the map continues below line 377:
   `transfer`, 4 shipped baselines (`bang_bang`, `fixed_setpoint`, `pid`,
   `pid_water`), `all_off`, `geometry_zonal`, `ode_thermal`, 37
   `zone_pi`/`dsat_*` keys, 33 `g4_*` keys, 30 `occ*` keys (including
   `occ_gated` through `occ_gated7` plus `occ_*7` variants,
   `occ_econdbhhot9`, `occ_dcv7`), 3 `slot_*` keys, 2 `occ_duty*` keys,
   11 `probe_*` keys, 9 `bc_*`/`dagger_*` keys. Report 06 counted 138
   keys at its 2026-10-28 snapshot. Provenance: `smart_buildings/evaluate.py`;
   count 138 from `docs/research/06-repository-analysis.md`.
9. Throwaway probes go in `probe_modes.py` + `POLICY_MAP`; promoted winners
   become `occ_gatedN.py` with fixed constants; no environment variables in
   shipped code. Provenance: `AGENTS.md`.
10. A new policy/version gets a `docs/ladder.md` entry before submission
    (rule R5). Provenance: scout handoff (prior result 1).

## 3. The rudder: occ_gated4 -> occ_gated7

11. `OccGated4Policy` (`smart_buildings/baselines/occ_gated4.py`) subclasses
    `OccGated3Policy` and disables the hot-water floor by turning
    `_apply_water_floor` into a no-op. Rationale: discharge air is capped at
    308.15 K, so hotter water past the cap delivers no extra heat per unit
    air while the hotter loop pays tank dissipation. Provenance:
    `smart_buildings/baselines/occ_gated4.py`, `docs/ladder.md` rung 7.
12. `OccGated7Policy` (`smart_buildings/baselines/occ_gated7.py`) subclasses
    `OccGated4Policy` and adds occupied-only trim-and-respond static
    pressure with fixed constants: `PTR_UP=0.25`, `PTR_DOWN=0.03`,
    `PTR_MAX=0.0`, `PTR_DSAT=0.85`, `PTR_DB=1.0`, `PTR_DB_SLOPE=0.1`,
    `PTR_DB_REF=10.0`, `PTR_PERSIST=4`. Provenance:
    `smart_buildings/baselines/occ_gated7.py`.
13. Trigger logic (per AHU): starved = some zone damper > 0.85 while its
    temperature is outside the comfort band by more than the effective
    deadband; after 4 consecutive starved steps (`PTR_PERSIST=4`), pressure
    rises by 0.25 capped at 0.0; when all dampers are < 0.5, pressure falls
    by 0.03 floored at -1.0; pressure resets to -1.0 whenever the AHU is
    off; nothing happens unless `self._occupied(obs)` is true (prestart and
    floor keep minimum pressure). Provenance:
    `smart_buildings/baselines/occ_gated7.py`.
14. The deadband is OAT-aware: `db_eff = 1.0 + 0.1 * max(0, |OAT - band_mid|
    - 10.0)` in kelvin, computed from the decoded outside temperature.
    Provenance: `smart_buildings/baselines/occ_gated7.py`.
15. Validated effects (journal-reported): xfer_3 heat tail -23%, xfer_3
    cooling tail -34%; all misfire pairs flat to -1%, with train_1/train_2
    February bit-equal to `occ_gated4`. Provenance:
    `smart_buildings/baselines/occ_gated7.py` docstring.
16. The rudder stack underneath (inherited, journal-reported): per-zone PI
    to band centre (`KP=0.1`, `KI=0.005`, later `0.07/0.0035`),
    `MIN_DAMPER=0.1`, `HEAT_DAMPER_GAIN=1.0`, conditional integration,
    prestart 2 h, forced-on AHUs at `FORCED_SAT=-0.7` (~15 C) /
    `FORCED_PRESSURE=-1.0`, cooling-run hold `HOLD_STEPS=288`,
    demand-SAT, pressmin, down-shift up to 1.0 K capped at 0.4x band,
    mixedness-SAT (`MILD_SAT=-0.4`, `DEEP_SAT=-1.0`), night heating drop
    2 K, latched prestart (`MAX_PRESTART_STEPS=48` = 4 h,
    `PRESTART_SAFETY=1.3`, `PRESTART_MARGIN_STEPS=6`), hysteretic floor
    (`FLOOR_ENTER/EXIT_DB_K=0.5`, caps eco_low-1 / eco_high+1), occupancy
    gate (`OCC_EPS=0.5`, `EMPTY_GRACE_STEPS=12`), rate EMA
    (`RATE_INIT_K_PER_STEP=0.08`, `ALPHA=0.2`, bounds [0.0005, 0.3],
    prestart steps only), water reset
    (`RESET_FULL_K=4.0`, `RESET_EFFORT_QUANTILE=0.25`,
    `RESET_EFFORT_WEIGHT=1.0`, deadband 0.25 K,
    `RESET_INTEGRAL_GAIN=0.0`), and pump idle-off
    (water frac < 0.05 and max_heat < 0.05). Provenance: `docs/ladder.md`
    rungs 2-6; occupancy-gate constants from rung 4.
17. The teacher is stateful: PI integrators, latched prestart, hysteresis,
    24 h mixedness EMAs. A memoryless student cannot clone it (masked-BC
    subset-8 0.0368 vs teacher 0.0175; DAgger +212%). Any Online RL that
    learns from or around this rudder must handle hidden controller state,
    not just the observation vector. Provenance: `docs/ladder.md`
    (slot/BC line).
18. `probe_guard` (force normal when occupied + zone > 3 K out) is a
    structural no-op because occupied mode is already normal; never ship
    equivalent logic. Provenance: `docs/ladder.md` rung 9.

## 4. Simulator physics caps (hard constraints on actions)

19. The AHU has NO heating coil (cooling only); all heating is VAV reheat.
    Do not plan AHU-side heating, heat-pump COP curves, storage arbitrage,
    or humidity-control benefits. Provenance: `AGENTS.md`,
    `docs/research/10-control-research-synthesis.md`.
20. VAV discharge air is capped at 308.15 K; recirculation is capped at
    0.70; dampers have a 10% minimum. Plant efficiencies are constant
    (boiler constant efficiency, chiller COP 3.5). Provenance: `AGENTS.md`.
21. Action interface for K AHUs and V VAVs: `discrete_action: int32[K+1]`,
    `continuous_action: float32[3K+2+2V]`; discrete values are {0,1};
    continuous commands are normalized to [-1, 1] with zero = range
    midpoint, not shutdown. Plant entries are interleaved; index via
    `temp_indices`, `pressure_indices`, `recirculation_indices`,
    `water_indices`, `zone_action_indices`. Provenance:
    `docs/research/06-repository-analysis.md`.
22. `evaluate.py::validate_action` casts dtypes and checks shapes only; it
    does not enforce finite values or numeric bounds. Local shape
    acceptance is not sufficient submission validation: the submission
    interface requires in-range actions. Achieved actuator values can
    differ from commands (physical floors/caps apply). Provenance:
    `docs/research/06-repository-analysis.md`.
23. Observation is a building-dependent flat vector addressed by
    `observation_field_names` (zone temperatures and bands, achieved
    damper/reheat positions, AHU states, water readings, outside
    temperature, schedule/time features, building-level occupancy).
    Metadata adds floor/zone maps, floor count, AHU assignments, schedule
    windows/timezone/holidays, normalization data. NOT available:
    materials, diffuser arrays, hidden slab/wall temperatures, future
    weather, per-zone occupancy trajectories. Provenance:
    `docs/research/06-repository-analysis.md`.
24. Occupancy is building-level only; no per-zone gating is possible.
    Occupancy normalization has a known source mismatch
    (`environment.py` uses `(N-c)/(c+1)`; `evaluate.py` sets an
    underscored attribute distinct from the public constant), so raw-count
    assumptions must be bound to one inspected source/dump version.
    Provenance: `AGENTS.md`, `docs/research/06-repository-analysis.md`.
25. Buildings are valid iff <= 10 zones/AHU; monster buildings simulate
    catastrophically slowly (a 125-zone building took ~2 h and was dropped
    as xfer_4). Check zone count before launching a suite. Provenance:
    `AGENTS.md`, `docs/ladder.md` rung 8.
26. The deployable policy interface is
    `Policy(building_meta_data, action_spec)` + `get_action(obs)`. The
    inference server creates a fresh policy per scenario: cross-scenario
    learned state is NOT retained. The policy receives metadata + obs, not
    an environment object, scenario path, material fields, or future
    weather replay. `tools/oracle_search.py` and `tools/ckpt.py` operate
    outside this boundary and cannot be embedded in the submitted policy.
    Provenance: `docs/research/06-repository-analysis.md`.
27. Observation dimensions are metadata, not constants: the template cites
    271 for `building_train_8` while `eda/buildings.csv` records 283, and
    `eda/observation_normalization_metadata.json` is an example artifact,
    not live metadata. Online RL feature code must read live
    `observation_field_names` and `action_spec`. Provenance:
    `docs/research/06-repository-analysis.md`.

## 5. Reward economics (what the score punishes)

28. Reward weights (comfort, energy, carbon) = (0.6, 0.2, 0.2). Comfort
    loss is linear to 5 K then quadratic unbounded; comfort slope (~0.18/K)
    >> energy slope (~0.002/K). Never trade comfort for energy: the -2 K
    heating-target probe (`HEAT_OFFSET_K=-2.0`) lost on train (+1.4%
    subset) AND public (+2.2% vs occ4, 0.01352 vs 0.01323). Provenance:
    `AGENTS.md`, `docs/ladder.md`.
29. Electricity TOU peak/night = 1.25x: load-shifting arbitrage is dead
    (TOD precool/coast probe +5.1%). Provenance: `AGENTS.md`,
    `docs/ladder.md`.
30. Comfort loss = `-normalized_productivity_regret`; the exact reward uses
    zone occupancy that the participant does not observe. A controller
    must declare its occupancy-allocation proxy or uncertainty bounds.
    Optimize the repository's actual normalized components, not
    degrees/kWh/dollars substituted into the weights. Provenance:
    `docs/research/10-control-research-synthesis.md`.
31. Scoreboard (journal-reported): gen-3.1 public 0.01350; occ_gated3
    public 0.01330; occ_gated4 train-30 mean 0.009629 / median 0.007167,
    public 0.01323 (hidden/train 1.37); occ_gated5 public 0.01352;
    occ_gated6 train-30 0.009695 (+0.7% vs occ4, cold-only), transfer-12
    -13.0%, PENDING; occ_gated7 tails -23%/-34% with zero measured
    regressions (6/6), PENDING; leader (Joe Penn) 0.00521. Moving
    0.01323 -> 0.00521 on the same public evaluation would require a
    60.62% reduction; comparing train 0.00963 with hidden 0.00521 mixes
    coverages and is not a valid target. Provenance: `docs/ladder.md`,
    `docs/research/06-repository-analysis.md`,
    `docs/research/10-control-research-synthesis.md`.
32. "Train-optimal" evidence: ~15 probes on setpoint/SAT/margin/econ/
    load-shift axes all <= 0 vs occ4 (wfloor +0.0% n=2, econ -0.1% n=4,
    tod +5.1%, sat +9.5%, sp +1/+19%, margin +0.0%); duty-cycling variants
    all <= 0. Expected Online RL gains must come from transfer tails, not
    from re-tuning train axes. Provenance: `docs/ladder.md`.
33. Continuous PPO on slots diverged (0.040 -> 0.076, negative explained
    variance; lr 3e-5, KL 0.01, per-dim logprob norm) and is closed:
    continuous PPO is non-viable on this sharp comfort landscape
    (SNR ~ 1/s^2). Online RL must pick a different algorithm or
    parameterization, not retry the same setup. Provenance:
    `docs/ladder.md`.

## 6. Transfer-evaluation method (local substitute for hidden)

34. Hidden = unseen buildings x unseen climates, so train scores do not
    predict hidden. Local substitute: sample new buildings from the repo
    generator (`tools/gen_transfer.py`, seeds 50000+i, paper ranges,
    materials rotated over the 4 constructions) paired with unseen weather
    (minneapolis = unseen station; atlanta_4 / singapore_2 = unseen months
    of seen stations). Provenance: `docs/ladder.md` rung 8, `AGENTS.md`.
35. Transfer scenarios live in `smart_buildings/transfer_scenarios/`
    (git-ignored, regenerable with fixed seeds). Current suite: 3
    buildings x 4 weathers = 12 scenarios. Transfer limits: only 3
    building samples, all from one generator with rotated materials; 4
    weathers with only 1 unseen station; results/, scenarios, and *.npz
    are git-ignored, so local presence is weaker than durable provenance.
    Provenance: `docs/ladder.md` rung 8, `AGENTS.md`,
    `docs/research/06-repository-analysis.md`.
36. Known transfer finding: occ4 holds 2-3x over pid_water everywhere (no
    uniform transfer failure), but big-zone buildings bleed comfort in
    tails (xfer_3 January comfort -0.062, 25x worse than train; Singapore
    cooling -0.015). Root cause (proven via obs dump): pressmin starves
    airflow while the water loop sits at 350 K max with dampers wide open
    and zones at 13-18 C; only more air helps because discharge is capped.
    Provenance: `docs/ladder.md` rungs 8-9.
37. Batch drivers: train `tools/fullrun.py <results> 8640 <conc> <timeout>
    <policies> <pairs>`; transfer `tools/fullrun_t.py ... <scenarios_dir>`.
    Keep total concurrent sims <= 10 with 1500 s+ timeouts or jobs time
    out. Never edit policy code while a validation run using it is in
    flight (fresh subprocesses import at start; mixed versions corrupt
    results). Provenance: `AGENTS.md`; driver mechanics (subprocess per
    job, `ThreadPoolExecutor(max_workers=conc)`, CSV loss read) from
    `tools/fullrun.py`.
38. Reproduction cookbook (verbatim commands): see `docs/ladder.md`
    "Reproduction cookbook" (fullrun 8640/8/1500, gen_transfer,
    fullrun_t, assemble_notebook with the full occ class chain).
    Provenance: `docs/ladder.md`.

## 7. Submission packaging constraints

39. Assemble notebooks only via `tools/assemble_notebook.py` with verbatim
    class blocks: it copies `class <Name>` blocks from
    `smart_buildings/baselines/*.py`, appends them as a new controller
    cell after the existing chain cell (located by `class DsatGen31Policy`),
    re-points `SubmissionPolicy` at the requested class, verifies every
    block is copied exactly modulo trailing whitespace, and is idempotent
    via the `# >>> assembled occ cells (verbatim from repo) >>>` marker.
    CLI: `assemble_notebook.py <src_ipynb> <dst_ipynb> <SubmitClass>
    <CLASS...> <file:CLASS...>`; bare CLASS names come from `zone_pi.py`,
    `file:<consts>` copies module-level UPPER_CASE assignments.
    Provenance: `tools/assemble_notebook.py`, `AGENTS.md`.
40. The assembly MUST include `occ_gated:<consts>`: the module constant
    `STEPS_PER_HOUR` is required at runtime (kernel ERROR without it;
    occ6 v1 ERRORed exactly this way, fixed in v2). Provenance:
    `AGENTS.md`, `docs/ladder.md` rung 10.
41. Local exec check must run 300+ steps covering night / shutdown /
    prestart; a 12-step check once missed a night-only NameError.
    Provenance: `AGENTS.md`.
42. Submission flow: `kaggle kernels push` -> wait COMPLETE ->
    `competitions submit -k ... -v N -f submission.parquet`. Scoring queue
    is slow (~14 h). Watch the daily submission budget (1 left on 10-27).
    Push code only via `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_kaggle_push -o
    IdentitiesOnly=yes" git push origin main`; remote `main` is the source
    of truth for multi-machine work. Provenance: `AGENTS.md`,
    `docs/ladder.md`.
43. Shipped policy code implications for Online RL: fixed constants only
    (no env vars), standalone notebook cell (no repo imports beyond the
    assembled blocks, no checkpoint files, no torch dependency unless the
    notebook carries it), fresh instance per scenario (no cross-scenario
    memory). Anything the RL layer needs at inference must be inside the
    assembled classes or their constants. Provenance: `AGENTS.md`,
    `tools/assemble_notebook.py`,
    `docs/research/06-repository-analysis.md`.

## 8. Planning horizons and forecasting rules

44. A 6-12-step horizon is 30-60 minutes; a 2-4-hour horizon is 24-48
    steps. Fixed short horizons miss prestart, nighttime storage, and
    next-day recovery; longer horizons can use move blocking but must be
    evaluated against the original 5-minute system. Provenance:
    `docs/research/10-control-research-synthesis.md`.
45. Use only causally available forecasts: persistence and learned periodic
    patterns are deployable; exact future weather/occupancy from the
    simulator is an oracle input. Report oracle and deployable results
    separately. Provenance:
    `docs/research/10-control-research-synthesis.md`.
46. The air supply is shared within each AHU while zone dampers/reheat
    allocate locally, and the water system couples heating terminals:
    independent per-zone optimization misses both couplings. A supervisory
    plant decision plus local allocations must be checked against the
    common plant; decomposition is a compute tradeoff, not an assumed
    equivalence. Provenance:
    `docs/research/10-control-research-synthesis.md`.

## 9. Documentation and evidence conventions

47. Committed docs are English only (`docs/*`, code comments); chat may be
    Russian. Provenance: `AGENTS.md`, scout handoff (prior result 1).
48. Research docs: files `docs/research/01..10-kebab-slug.md` exist;
    numbers 11-17 are free; naming pattern `NN-kebab-case.md` with an
    index entry in `research/README.md`; header carries `# Title`,
    Date, and Scope; report 06 additionally records snapshot SHA,
    reconciliation SHA, a no-simulation disclaimer, and evidence-standard
    definitions (Source-verified / Recomputed / Journal-reported /
    Hypothesis). Provenance: scout handoff (prior result 1),
    `docs/research/06-repository-analysis.md`.
49. Journal rules R1-R9 (checked at write time): R1 no bare better/worse
    without delta; R2 every measurement names coverage, never mixed
    unlabeled; R3 every file/command exists (repo-root-relative paths);
    R4 negative results mandatory; R5 ladder entry before submission; R6
    submissions logged on day sent; R7 timestamps on world-facing events;
    R8 irreproducible not recorded as fact; R9 English only. Daily
    structure: Summary + Status + Done + Measured + Decisions + Open +
    optional Appendix. Provenance: scout handoff (prior result 1).
50. `docs/plan/` did not exist at scout time; this track creates it. Greybox/MPC
    control loop was never closed (kept as plan B). No demonstrated
    participant-only MPC improves full-episode transfer scores.
    Provenance: scout handoff (prior result 1), `docs/ladder.md`,
    `docs/research/06-repository-analysis.md`.

## Open questions

1. What are the public scores of occ_gated6 and occ_gated7 (both PENDING at
   ladder time)? The Online RL baseline cannot be fixed until at least one
   lands. Provenance: `docs/ladder.md` (submission log).
2. Where is `STEPS_PER_HOUR` defined and what is its value? The constant is
   required by statement 40 but its definition site (`occ_gated.py`) was
   outside this track's assigned sources. Provenance: `AGENTS.md`.
3. What is the exact `tools/fullrun_t.py` CLI (argument order, dump support)?
   Only `tools/fullrun.py` was in the assigned sources; transfer commands
   are quoted from `docs/ladder.md`. Provenance: `docs/ladder.md`.
4. What are the exact inherited `OccGated3Policy` prestart/floor/shutdown
   state machines? Only the constants quoted in `docs/ladder.md` rungs 4-6
   were available; `occ_gated3.py` was outside the assigned sources.
5. What is the full current `POLICY_MAP` size? 133 keys were counted in the
   inspected window (lines 218-377) with more lines below; report 06's 138
   belongs to its 2026-10-28 snapshot. Provenance:
   `smart_buildings/evaluate.py`, `docs/research/06-repository-analysis.md`.
6. What are the current transfer-scenario contents (building sizes, zone
   counts, weather months)? The directory is git-ignored and was not
   inspected. Provenance: `docs/ladder.md` rung 8.
7. Which occupancy-allocation proxy will Online RL declare for the comfort
   term, given that true zone occupancy is unobserved (statement 30)?
   Provenance: `docs/research/10-control-research-synthesis.md`.
8. Which algorithm/parameterization replaces the diverged continuous PPO
   setup (statement 33), and what is its stability evidence on the sharp
   comfort landscape? Provenance: `docs/ladder.md`.
