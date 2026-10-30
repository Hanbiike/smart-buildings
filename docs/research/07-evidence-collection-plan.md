# HVAC evidence collection plan

Date: 2026-10-28. Basis: the [repository audit](06-repository-analysis.md),
the [current journal](../2026/10/28/README.md), and the primary resources below.
This is the collection specification for the next research and experimental
stage. It distinguishes material already inspected from data still needed.

The highest priority is to recover a reproducible description of existing
experiments, then collect matched full episodes for the current policy family.
More papers or more hourly meter records cannot substitute for the missing
action, observation, code-version, and scenario alignment.

## 1. Research questions and the evidence each needs

| Question | Existing evidence | Required addition | Decision enabled |
|---|---|---|---|
| What actually controls the scored plant? | Simulator source, policy interface, metadata builder, AGENTS facts | A versioned field dictionary for every evaluated building; requested versus applied action traces | Identify reachable control improvements |
| How much of the energy loss is avoidable? | Recomputed occ4 train-30 decomposition | Matched baseline/candidate trajectories with equal comfort outcomes, plus credible counterfactuals | Distinguish necessary conditioning from recoverable loss |
| Do econ, heating deadband, and heatopt transfer? | Oct-28 selected-cell results and notebooks | Full current-baseline train suite; fixed transfer suite; component ablations; startup and night tails | Select a reproducible reference policy |
| Why do prestart restrictions lose? | Full-episode negative cases in the journal | Aligned heat delivery, thermal state estimates, comfort timing, and next-day recovery | Test a specific recovery mechanism |
| Can a model support planning? | Greybox, equipment observer, reduced thermal diagnostics | Causal held-out forecasts under alternative actions, excitation evidence, calibrated errors | Decide whether an MPC experiment is justified |
| Can learning outperform the current controller? | Prior BC, DAgger, PPO work; audited literature | Equal-compute comparisons, training seeds, held-out buildings, action support and fallback statistics | Compare algorithms in this environment |
| What transfers to real buildings? | Plant mapping and domain reference | Humidity, ventilation, sensor quality, equipment curves, commissioning and field constraints | Define the deployment gap |

The hidden leaderboard cannot reveal the leader's algorithm, the hidden
climate distribution, or which actuator created a score difference. Keep
these as unknowns rather than inputs to a research ranking.

## 2. Collection priorities

### P0 — recover and freeze what already exists

| Item | Location or source | Fields to collect | Acceptance condition |
|---|---|---|---|
| Repository identity | Git and the current worktree | Commit, dirty diff, hashes of policy and inherited modules | A result resolves to the exact executed code |
| Policy configuration | Constructor defaults, environment overrides, notebook | All effective constants, mixin order, normalization assumptions | A policy key alone is never treated as version identity |
| Experiment inventory | `smart_buildings/results/` and surviving journal-named `/tmp` directories | Scenario, policy, file hash, row count, first/last timestamp, return, status | Truncated and full episodes are distinguishable |
| Scenario inventory | `smart_buildings/training_set.yaml`, generated configuration, building/weather YAML, maps | Building seed, construction, geometry, AHU assignment, weather strip, start time, hashes | A scenario can be regenerated and compared |
| Transfer inventory | `tools/gen_transfer.py`, generated transfer configs | Fixed seeds, every planned pair, actual completion/timeout state | Report 16 planned separately from 12 completed |
| Submission evidence | Notebook, kernel/submission record, journal | Kernel owner/slug/version, notebook hash, submitted artifact, UTC times, score/status | Journal-reported status distinguished from live status |
| Previous negative results | Day logs and underlying files | Coverage, control version, delta, failure mechanism | Closed variants do not silently return as untested priorities |

Existing `/tmp` files should be preserved in a durable artifact location with a
manifest before reuse. A surviving CSV proves its stored trajectory; it does
not by itself prove which historical environment variables produced it.
Use `provenance_unknown` for any missing configuration, and retain the data as
exploratory evidence until the missing link is recovered or a run is repeated.

### P1 — collect measurements that discriminate current hypotheses

| Priority | Data package | Required coverage | Why it comes first |
|---|---|---|---|
| 1 | occ7, econ, dbh, heatopt, occ8/occ9 aligned comparisons | Identical 8640-step scenarios, all train-30 pairs for final comparison | The current promising family is ahead of the old research plan |
| 2 | Comfort and recovery event tables | All scored steps; arrival, shutdown, overnight and prestart subsets | Heating deadband gains carry documented night-tail risk |
| 3 | Applied actuator and equipment traces | Per-zone and per-AHU signals aligned to observations and rewards | Detect saturation, ineffective commands and cooling/reheat interaction |
| 4 | Causal model evaluation set | Prefix fit; later observations and unseen buildings reserved for evaluation | Prediction after an update is insufficient evidence for planning |
| 5 | Transfer development suite and fresh holdout | Existing 16-cell suite plus untouched seeds/regimes, with timeout accounting | Repeated tuning has made the existing suite development data; freezing it now does not restore holdout status |
| 6 | Runtime and packaging record | Full policy runtime plus at least 300-step assembled-notebook check | A runnable controller must fit deployment constraints |

These are proposed data collections, not experiments performed by this
documentation update. The user-requested repository analysis and research
do not require launching a tuning campaign.

### P2 — collect broader HVAC and field evidence

For each control method, collect the plant diagram, measured and commanded
variables, control cadence, occupancy and weather information available at
decision time, baseline tuning, evaluation duration, raw energy and comfort
outcomes, compute requirements, and whether the study used real buildings.

For field transfer, additionally collect ventilation design rates, humidity
limits, CO2 sensor placement/calibration, supply and return temperatures,
airflow calibration, equipment part-load curves, minimum cycle times,
overrides, fault logs, local sequence requirements, and commissioning reports.
These are needed to evaluate a real control system; the competition loss does
not measure all of them. See the [domain reference](08-hvac-domain-reference.md).

## 3. Required data contracts

### Run manifest

One record per attempted policy/scenario episode, including failures:

```text
run_id, created_utc, completed_utc, code_commit, dirty_patch_sha256
policy_key, policy_source_sha256, inherited_source_sha256, effective_parameters
building_id_for_analysis, weather_id_for_analysis, scenario_sha256
building_seed, weather_file_sha256, simulation_start, timestep_seconds
planned_steps, completed_steps, termination_reason, timeout_seconds
runtime_seconds, concurrency, python_version, dependency_versions
reward_csv_path, reward_csv_sha256, observation_dump_path
notebook_sha256_if_applicable, provenance_status, comparison_group
```

Building/weather identities are analysis metadata. Do not expose labels or
future weather to a deployed controller unless the actual interface and
competition rules allow them. A future stored trajectory is ground truth for
evaluation, not a forecast that was available when an action was selected.

### Step and equipment records

| Grain | Minimum contents | Main validation |
|---|---|---|
| Episode step | Run, index, timestamp, observation before action, requested action, next observation, scored reward, done flag | No duplicate or missing scored step; explicit timing convention |
| Zone step | Zone and serving AHU, temperature, heating/cooling setpoints, actual damper/reheat, requested outputs | Resolve identifiers from metadata; preserve units and masks |
| AHU step | On/off, SAT command and sensor, pressure command and sensor, airflow, recirculation, OAT, requests | Command clipping and inactive actuators are visible |
| Water system step | Temperature and differential-pressure commands/sensors, pump state | Delay and saturation recorded |
| Reward step | `agent_reward`, normalized comfort/energy/carbon terms, raw costs/emissions | Weighted components reconstruct reward within recorded numerical precision |
| Model forecast | Origin time, last assimilated sample, horizon, future input assumption, prediction interval, arriving target | No post-update target leakage |

For this simulator, store sensor temperature and post-solve reward temperature
as separate fields with separate information-availability labels. Record
command time, achieved-actuator observation time, physical-state time, and
the occupancy/setpoint interval used by the reward. The response of zone
temperature to `u[t]` arrives at `obs[t+2]`; do not align it as an immediate
`obs[t+1]` response. Mark post-solve diagnostic state and reward-only zone
occupancy as unavailable to the participant at action selection time.

The existing tracker exposes raw electricity/gas cost and carbon alongside
normalized components. Preserve all of them. A dollar cost is not electrical
energy, a valve command is not delivered heat, and a damper position is not a
measured outdoor airflow.

Missing fields stay missing. An inferred heat rate must carry its model and
assumptions; never write it into a column labeled as a sensor measurement.
Similarly, building-level occupancy must not be duplicated as if it were
observed zone occupancy.

### Research source record

```text
title, authors_or_organization, publication_date, accessed_date
doi_or_canonical_url, version_or_edition, source_type, access_level
specific_claim, section_table_or_equation, original_units
plant, climate, occupancy, horizon, baseline, metric, uncertainty
code_data_availability, applicability_to_this_repo, unresolved_limit
```

Access levels: `body inspected`, `abstract only`, `publisher description`,
`search lead`, and `unavailable`. Only the first supports detailed algorithm
or table claims; a publisher description can establish the existence and
scope of a standard without establishing its detailed clauses.

## 4. Evaluation design

### Coverage and aggregation

The shipped set is the 30 pairs in `smart_buildings/training_set.yaml`, not
the cross product of every building and every weather file on disk.
Transfer generation defines four new buildings and four weather strips;
report all 16 planned pairs even if a historical table includes only 12.
Count zones per AHU before running a new suite and keep the repository's
maximum of 10.

For scenario s with N=8640 scored steps:

```text
L_s = abs(sum_t reward[s,t]) / N
L_suite = mean_s L_s
delta_s = L_candidate,s - L_reference,s       # negative means improvement
relative_delta = (L_candidate / L_reference - 1) * 100 percent
```

Compute the aggregate only on a declared identical scenario set. If a run
times out, report the failure and a separately labeled completed-pair mean.
Do not silently shorten episodes or let a missing difficult building improve
the reported mean. A diagnostic subset can rank hypotheses; it cannot stand
in for train-30 or the hidden leaderboard.

Use the exact reward decomposition and occupancy normalization from the
running configuration. Preserve the official total loss as the primary
metric and show comfort tails, raw costs and carbon as explanatory metrics.
Record maximum violation, violation duration, morning recovery, night events
and per-scenario deltas; an average alone can hide a serious local regression.

### Generalization and uncertainty

Separate tuning data from a final local holdout. Hold out complete buildings
and weather regimes, rather than randomly splitting five-minute rows.
Evaluate cold, hot, shoulder and mixed-load regimes, new constructions,
different AHU/zone counts, and both episode startup and later operating days.
Where practical, use building-held-out and weather-held-out analyses
separately before their joint holdout to diagnose the source of failure.

The simulator's deterministic rollout removes repeated-environment sampling
noise for a fixed policy/scenario. It does not remove uncertainty about new
buildings or variation between learned policies trained with different
random seeds. Report training-seed variation for learning experiments and
scenario-level dispersion for transfer. If bootstrapping, resample building
or scenario groups with stated assumptions; five-minute rows are not
independent experimental units. Six training buildings limit precision.

### Counterfactuals and model identification

Heating, airflow, occupancy gains, envelope losses and unmeasured wall state
can compensate for one another in fitted models. Low in-sample temperature
error does not identify their physical parameters. Collect excitation and
response by operating mode, compare one-step and multistep errors, and
evaluate on actions outside the fitting controller's usual operating range.
See `tools/thermal_prefix_identification_check.py` for the existing explicit
warning that excitation checks are necessary rather than sufficient.

Evaluate the original controller, the observer without optimization, the
planner using estimated states, and a diagnostic oracle separately. Access
to simulator snapshots or future weather changes the information available;
an oracle gain establishes local opportunity, not deployable performance.
Extend short search windows to complete episodes before claiming a score gain.

### Execution contract

Use `./.venv/bin/python -B` for every Python command. The existing train
driver is `tools/fullrun.py`; transfer uses `tools/fullrun_t.py` with the
scenario-directory argument. Both accept explicit policy and pair lists.
Use 8640 steps and timeouts of at least 1500 seconds; keep total concurrent
simulators across all processes at or below 10. Freeze policy code and all
inherited dependencies for the entire run. A return code alone is insufficient:
inspect row count, time coverage, termination and artifact identity.

Assemble a candidate only with `tools/assemble_notebook.py`, including
`occ_gated:<consts>` for `STEPS_PER_HOUR`. The local execution check must cover
at least 300 steps including night/shutdown/prestart paths. Preserve the
assembled artifact and effective constants separately from the research
prototype. These are future implementation gates, not checks run here.

## 5. External material worth collecting

The following primary source bodies were opened on 2026-10-28. Dataset
archives and complete model distributions were not downloaded. Their roles
below are research judgments based on the inspected descriptions.

| Resource | What it provides | Use here | Limitation |
|---|---|---|---|
| [BOPTEST](https://ibpsa.github.io/project1-boptest/) and [official repository](https://github.com/ibpsa/project1-boptest) | Building/HVAC emulators, control API and KPI reporting | Independent control-method comparison with explicit test-case contracts | Different plants, observations, forecasts and scoring; no direct score conversion |
| [ComStock public datasets](https://comstock.nlr.gov/page/datasets) | Commercial-building characteristics, end-use results, weather and model access | Prior ranges and coverage design for commercial buildings | Simulated stock data; pin a release and inspect its known issues |
| [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) | Hourly meter records for 1,636 buildings, metadata and weather | Load diversity, anomaly analysis and forecasting research | Hourly meters do not supply five-minute intervention labels or zone actuator responses |
| [LBNL FDD data](https://faultdetection.lbl.gov/data/) | Fault and fault-free time series across AHUs, VAVs and plant equipment | Fault robustness and sensor-quality stress tests | Mixed simulation/lab/field provenance; use documented faults and equipment scope |
| [Brick documentation](https://docs.brickschema.org/brick/overview.html) | Semantic representation of building entities and relationships | A portable zone/AHU/point dictionary | A schema supplies meaning, not missing measurements or identification |
| [EnergyPlus weather format](https://energyplus.readthedocs.io/en/stable/guides/interface-developer/6-weather-data.html) | Location/time metadata and weather-file structure, including subhourly support | Document weather conversion and alignment | File-format support does not create observed high-frequency variation |
| [Climate.OneBuilding source documentation](https://climate.onebuilding.org/sources/default.html) | Weather source periods and construction of typical-year files | Broader climate sensitivity cases | A typical year is not an extreme-weather stress test or a causal forecast |
| [CityLearn feature documentation](https://www.citylearn.net/guides/features.html) | Configurable buildings, thermal models, storage and community energy resources | Compare coordination ideas when matching enabled assets | Storage/PV/EV flexibility differs from this VAV-reheat task; pin the study's version |

Collect small representative samples and their dictionaries before obtaining
large archives. A dataset is useful only if its granularity, interventions,
physics and metadata answer a listed question. For model initialization,
document the mapping and distribution mismatch; do not concatenate unrelated
temperature/meter streams into apparent transition data.

## 6. Literature acquisition priorities

1. Read the corrected evidence in [09](09-evidence-audit.md) before quoting
   RL/MPC savings from [04](04-rl.md). Capture each exact baseline and metric.
2. Use the official standards and Modelica implementation links in
   [08](08-hvac-domain-reference.md) for mechanisms. Pin edition and code
   version; a vendor or implementation example does not establish every
   requirement of the current standard.
3. For MPC, prioritize measured uncertainty, state estimation, online
   adaptation, actuation constraints and fair tuned baselines. Generic
   energy-savings percentages are secondary to information compatibility.
4. For RL, collect action support, training budget, normalization, delayed
   observations, reward details, validation split and deployment fallback.
   Compare complete algorithms rather than attributing success to an acronym.
5. For field relevance, collect ventilation, humidity, equipment/fault and
   commissioning evidence in addition to temperature and energy.

The old reports contain useful search leads that remain unread or only
abstract-supported. They are a secondary retrieval backlog, not numerical
evidence for the current recommendations. A source is not considered
verified because another review repeats its number.

## 7. Remaining access and decision gaps

The public Kaggle overview and rules URLs returned no readable body through
the web reader on 2026-10-28. Local `README.md` and `AGENTS.md` establish the
repository workflow; they are not a fresh verification of current external
rules, deadlines, scores or remaining submission quota. Capture the actual
competition page or API record before a submission decision. No submission
is part of this documentation work.

The closing journal records occ8 and occ9 submitted with scores pending;
their current hidden scores remain unknown in this research snapshot.
Unarchived run parameters, incomplete transfer
coverage, realistic humidity/IAQ behavior, and causal planner performance
also remain open. Each requires the specific data above, rather than a
stronger claim from the available evidence.
