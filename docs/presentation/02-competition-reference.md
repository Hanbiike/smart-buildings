# NeurIPS 2026 Smart Buildings Competition — consolidated reference

> Multi-agent extraction of https://smart-buildings-2026.github.io/competition/index.html
> plus same-site pages, run 2026-11-02: 4 scope extractors (overview-logistics,
> rules-submission, evaluation, data-simulator), 1 completeness critic, 1 gap
> follow-up, 1 synthesis. Sections 1-12 below are the synthesis verbatim;
> source URLs inside are verbatim as reported. The addendum records parent-side
> verification that closed the critic's residual micro-gaps.

# NeurIPS 2026 Smart Buildings Competition — Consolidated Reference

Consolidated from prior results 1, 4, 2, 3, 5, 6. English only. Facts grouped per section with verbatim source URLs.

## 1. Competition identity and hero block

Source: https://smart-buildings-2026.github.io/competition/index.html

- Page title: "Smart Buildings Competition 2026"
- Eyebrow: "NeurIPS 2026 · Competition Track" (prior result 1, item 1); variant recorded as "NeurIPS 2026 Competition Track" (prior result 6 verification).
- H1: "The Smart Buildings Challenge" / sub: "Learning Foundation Models for Real-World Optimization at Scale"
- Lede: "Build a single HVAC control policy that generalizes across commercial buildings it has never seen, jointly minimizing energy, carbon, and discomfort. A step toward scaling ML-driven building optimization across the globe."
- Meta description: "A NeurIPS 2026 challenge to build one HVAC control policy that generalizes across unseen commercial buildings."
- Verification (prior result 6): web_fetch https://smart-buildings-2026.github.io/competition/index.html returned 200 OK, 12681 bytes; Title Smart Buildings Competition 2026, NeurIPS 2026 Competition Track, H1 The Smart Buildings Challenge; lede describes a single HVAC policy generalizing across unseen buildings minimizing energy/carbon/discomfort.

## 2. Goal and motivation

Source: https://smart-buildings-2026.github.io/competition/index.html

- Section: "Building control policies rarely transfer"
- Quote: "Buildings cause nearly 40% of carbon emissions in the U.S. Of that total, HVAC systems are responsible for roughly 40 to 60%, and account for about 15% of global energy consumption. Yet despite advances in AI, most buildings still run about as inefficiently as they did decades ago."
- Quote: "The biggest obstacle is the lack of policy transfer. Most building control systems, whether simulation-based or learned, only work well for the building they were developed for, and fail to generalize to others."
- Quote: "To break through this, we propose a challenge to develop **building foundation models** that adapt to new buildings with minimal configuration, so building optimization can finally scale globally."
- Quote: "Participants build control policies on a set of simulated buildings, then have those policies evaluated on an **unseen set of buildings** that vary in size, layout, weather, material properties, and HVAC system."

Source: https://smart-buildings-2026.github.io/competition/competition.html

- Quote: "The objective is deliberately *not* to tune a controller for one building. It is to learn one policy that generalizes across arbitrary floor plans, HVAC topologies, and weather."
- Quote: "Participants develop a single control policy that operates the HVAC systems of commercial buildings across a range of layouts and weather conditions."
- Quote: "The same policy is deployed, unchanged, on buildings it has never encountered, differing in number of floors, geometry, room layout, zoning, equipment count, and climate."
- Quote: "The policy must infer an effective strategy from the building description and live sensor stream it is given at runtime."
- Quote: "This portability across the entire distribution of buildings, rather than competence on a single instance, is the defining requirement of a building foundation model and the central scientific challenge of the competition."

## 3. Organizers, contacts, and sponsors

Source: https://smart-buildings-2026.github.io/competition/organizers.html

- Heading: "Organizing committee"; sub: "A cross-institution group spanning machine learning, control, building science, and industry."
- Main contacts: "Judah Goldfeder — Columbia University · PhD Candidate & research fellow, NSF AI Institute in Dynamic Systems — Main contact"; "Ayush Rai — Harvard University · Postdoc, Electrical Engineering & Applied Mathematics — Main contact"
- Committee: "Na (Lina) Li — Harvard University"; "Philippe Wyder — Distyl AI"; "Victoria Dean — Princeton University"; "Donna Vakalis — Mila"; "Jiong Lin — Columbia University"; "Bing Dong — Syracuse University"; "Michael Rossetti — Google"; "J. Nathan Kutz — University of Washington"; "Sohei Arisaka — Kajima"; "John Sipple — Google"; "Hod Lipson — Columbia University"; "Yoshua Bengio — Mila"
- Acknowledgments: "This competition is made possible by the generous support of our prize sponsors."
- Main-contact emails: jag2396@columbia.edu (mailto:jag2396@columbia.edu); ayush_rai@seas.harvard.edu (mailto:ayush_rai@seas.harvard.edu)

Sponsors (verbatim URLs):

- Kajima Corporation: https://www.kajima.co.jp/english/
- Distyl AI: https://www.distyl.ai/
- KODE Labs: https://kodelabs.com/
- Distyl careers link listed on site: https://jobs.ashbyhq.com/Distyl/form/neur-ips-2026

## 4. Tracks, key dates, and prizes

Source: https://smart-buildings-2026.github.io/competition/index.html

- Quote: "Total prize pool: Kaggle code competition, first paper round, and second paper round"
- Header: "$40k" / "Total prize pool: Kaggle code competition, first paper round, and second paper round"; also "$40k Total prize pool"
- Track 1 — Main leaderboard: "Ranked on the Kaggle private leaderboard. Deadline January 15, 2027." / "Main leaderboard: $25,000"
- Track 2 — Paper track: "NeurIPS Poster Paper Awards. Deadline November 13, 2026." and "Final Paper Awards. Deadline January 15, 2027." / "Paper track: $15,000"
- Phases shown under Key dates: "Round One deadline (NeurIPS poster)"; "NeurIPS Workshop in Paris"; "Final deadline"
- Key dates: "Nov 13, 2026 — Round One deadline (NeurIPS poster)"; "Dec 13, 2026 — NeurIPS Workshop in Paris"; "Jan 15, 2027 — Final deadline"
- Main leaderboard prizes: 1st $5,750; 2nd $3,750; 3rd $3,000; 4th to 8th $2,500 each
- NeurIPS Poster Paper Awards prizes: 1st $1,000; 2nd $750; 3rd $500; 4th to 14th $250 each
- Final Paper Awards prizes: 1st $3,000; 2nd $1,750; 3rd $750; 4th to 12th $500 each
- Rules pointer: "The official competition rules are on the [Kaggle rules page](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules)."

## 5. Rules location, entry, submission format, and policy API

Source: https://smart-buildings-2026.github.io/competition/index.html

- Quote: "The official competition rules are on the [Kaggle rules page](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules)."
- Entry links: "[Compete on Kaggle](https://www.kaggle.com/competitions/neurips-2026-smart-buildings)", "[Starter kit on GitHub](https://github.com/smart-buildings-2026/starter-kit)", "[Submit papers on OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge)".
- Finding: no eligibility text, team-size or team-merge rules, or registration steps appear on this same-site page; it defers rules to Kaggle.

Source: https://raw.githubusercontent.com/smart-buildings-2026/starter-kit/main/README.md (mirror of https://github.com/smart-buildings-2026/starter-kit)

- Quote: "The competition runs on Kaggle. For the rules, the leaderboard, the submission template and how to enter, see the competition page: **<https://www.kaggle.com/competitions/neurips-2026-smart-buildings/>**"
- Quote: "The official ranking is computed server-side on Kaggle (https://www.kaggle.com/competitions/neurips-2026-smart-buildings/) over held-out evaluation buildings, drawn from the same generator and parameter ranges as the shipped six."
- Quote: "Edit `smart_buildings/baselines/transfer_policy.py`. It carries the full interface contract in its module docstring ... Read that first; it is the reference, not this section."
- Quote: "A policy is a plain Python class with two methods. No base class, nothing to subclass: class TransferPolicy: def __init__(self, building_meta_data, action_spec) ... def get_action(self, obs) -> dict: return {continuous_action: np.float32[3K+2+2V], discrete_action: np.int32[K+1]}"
- Quote: "Keep the class name and you need change nothing else — it is already registered in `POLICY_MAP`."
- Quote: "Your policy also receives a **building metadata** dict at construction time (action indices, observation field names, zone-to-AHU assignment, floor plan, schedule, normalization constants)"
- Commands quoted: `pip install -e .`; `python -m smart_buildings.run_environment --building building_train_1 --weather fairbanks_1`; `python -m smart_buildings.evaluate --policies transfer`; full episode default `--steps` is 8640.
- Quote: "The evaluation spans **6 buildings x 5 weather regimes = 30 scenarios**. A submission is one policy that must generalize across all of them."
- Scoring quotes: "The leaderboard metric is the **mean absolute error against a target of zero**, taken over the per-scenario mean per-step reward. **Lower is better.**" and "Returns from different episode lengths are not comparable, so scoring is refused outright if your policy was run at any other step count."

## 6. Submission mechanics, starter kit, paper channel, and support

Source: https://smart-buildings-2026.github.io/competition/competition.html

- Quote: "At each control step, a five-minute interval over a 30-day episode of 8,640 steps in which every step is scored, the policy observes the building state and must simultaneously issue: Discrete actions ... Continuous setpoints ..."
- Quote: "the discrete component has shape `(K+1)`" and "the continuous component has shape `(3K+2+2V)` ... each normalized to [-1, 1]".
- Quote: "The simulator, the training scenarios, the baseline controllers, the building generator, and a local scorer are in the [starter kit on GitHub](https://github.com/smart-buildings-2026/starter-kit)."

Source: https://github.com/smart-buildings-2026/starter-kit

- Repo file listing includes: `smart_buildings/`, `README.md`, `baseline_returns.json`, `check_scales.py`, `environment.yml`, `pyproject.toml`, `score_leaderboard.py`, `Dockerfile`, `LICENSE`, `NOTICE`.
- Layout quotes: `smart_control/` "the building simulator", `baselines/` "reference policies", `buildings/` "the 6 building definitions", `training_scenarios/` "30 generated scenarios (building x weather)", `evaluate.py` "test YOUR policy", plus `run_floor_generator.py` and `run_scenario_generator.py`.
- Baselines table quotes: `bang_bang` "per-air-handler on/off", `fixed_setpoint` "one comfort band around the clock, no eco setback", `pid`, `pid_water`, `all_off` "never runs any equipment"; "There is no learned baseline. Bring your own."
- Generation quotes: "Building generation is procedural, so you can sample as many additional buildings and scenarios as you like during development (the held-out evaluation buildings come from the same generator and ranges)." Commands: `python -m smart_buildings.run_floor_generator --start-seed 500 --count 10` and `python -m smart_buildings.run_scenario_generator --buildings ... --weathers ...`.
- Quote: "Apache License 2.0 (see `LICENSE`). Builds on Google's open-source Smart Buildings Simulator; see `NOTICE`."

Paper channel. Source: https://smart-buildings-2026.github.io/competition/index.html

- Quotes/links: "[Submit papers on OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge)" and "[Paper template](https://drive.google.com/file/d/1NHtpV2Kwz39wuPDwA2FiyVWaQr7vGbWt/view)".
- Source: https://drive.google.com/file/d/1NHtpV2Kwz39wuPDwA2FiyVWaQr7vGbWt/view — only verified fact: file is named "Smart_Buildings_Competition_NeurIPS_2026_Template.zip"; page body requires login-walled Drive preview.

Support. Sources: https://smart-buildings-2026.github.io/competition/index.html and https://smart-buildings-2026.github.io/competition/competition.html

- Finding: neither same-site page states any compute grant/credits, GPU/TPU support, office hours, forum, email, or contact channel. The only outbound participation links found by href enumeration are the Kaggle competition page, Kaggle rules page, starter-kit GitHub repo, OpenReview group, paper-template Drive file, sponsor sites, and one careers link.
- Same-site scope finding: href enumeration of index.html found only one same-site .html link (`competition.html`), and competition.html contains no further same-site .html links in that enumeration.

## 7. Evaluation protocol, reward, and leaderboard score

Source: https://smart-buildings-2026.github.io/competition/competition.html

- "At each step the policy receives a weighted sum of three competing quantities: comfort, energy cost, and carbon emission."
- Formula: "$$ r_t = \\alpha_{\\text{comfort}}\\,\\delta_t \\;-\\; \\alpha_{\\text{energy}}\\,c_t \\;-\\; \\alpha_{\\text{carbon}}\\,e_t $$"
- "delta_t: comfort (productivity) term, 0 when occupied zones are in band. c_t: energy cost (electricity and gas). e_t: carbon emission."
- "The comfort term (delta_t) equals 0 when occupied zones are comfortable and falls as zones leave the comfort band: linearly up to 5 K outside the band, then quadratically, with no lower limit. So abandoning comfort is never free."
- "The organizer-fixed weights are identical for every submission and sum to one: 0.6 comfort, 0.2 energy, 0.2 carbon"
- "This makes comfort dominant while still charging materially for energy and carbon"
- "Because comfort is scored only in *occupied* zones"; "Comfort is scored only against the currently active band, so deviations are penalized during occupied hours but may drift within the wider eco band at night."

Source: https://smart-buildings-2026.github.io/competition/index.html

- "A regret-based reward forces a genuine tradeoff: condition aggressively and burn energy, or switch off and lose comfort."
- "At each 5-minute step the policy issues discrete on/off commands and continuous setpoints, and their dimensions change from building to building."

Source: https://smart-buildings-2026.github.io/competition/scoring.html

- "One rollout per scenario."
- "each building and weather pair is run exactly once and there is no sampling noise to average away."
- "Grouping by weather or building is deliberately avoided: any grouping imposes an arbitrary weighting and hides within-group variance. A flat equal-weight mean over scenarios is the cleanest choice, so each building and weather pair counts exactly once."
- "Policies are ranked by their mean absolute error against a target of zero, where lower is better."
- "Each scenario contributes its mean per-step reward. The leaderboard reports the mean absolute error of those values against a target of zero. Lower is better."
- Formula: "$$ \\text{score} = \\frac{1}{|S|}\\sum_{s \\in S} \\bigl| \\bar{r}_s - 0 \\bigr| \\;=\\; -\\frac{1}{|S|}\\sum_{s \\in S} \\bar{r}_s $$"
- "r-bar_s: the policy's mean per-step reward on scenario s, that is its episodic return divided by 8,640 steps. S: the set of evaluation scenarios. Lower is better."
- "Why zero. The reward charges for three things and credits none: comfort shortfall, energy cost, and carbon emission. So the per-step reward is never positive. Zero means perfect comfort delivered at no energy and no carbon cost, which is the ideal no controller can reach. Every score is a distance from that ideal, and a smaller distance is a better policy."
- "This page explains the metric behind the standings."
- "The leaderboard reports the mean absolute error of those values against a target of zero."

## 8. Hidden test, reproducibility, baselines, and judging

Source: https://smart-buildings-2026.github.io/competition/scoring.html

- "Development runs on 6 public buildings x 5 weather stations = 30 scenarios. Final evaluation runs on the held-out set: 6 unseen buildings x 5 unseen weather stations = 30 scenarios."
- "The generator supports unlimited scenario creation, so participants can sample arbitrarily many additional buildings during development, while the held-out buildings and stations stay completely unseen."
- "The simulator is deterministic: the same policy on the same scenario reproduces the same return bit for bit, across processes and machines."
- "Repeated rollouts would return the same number, so each building and weather pair is run exactly once"
- At-a-glance: "30 Development scenarios - 6 public buildings x 5 weather stations, used to build and validate policies."; "30 Evaluation scenarios - 6 held-out buildings x 5 held-out weather stations, completely unseen during development."; "0.0 The unreachable floor - Zero is perfect comfort at no energy and no carbon cost."
- Companion panel labels: "30 Development scenarios", "30 Evaluation scenarios", "0.0 The unreachable floor".

Source: https://smart-buildings-2026.github.io/competition/competition.html

- Training envelope: "Across the 6 training buildings, K runs from 3 to 9 and V from 28 to 85"; "One AHU serves at most 10 zones; the training buildings have 3 to 9 AHUs."; "The training buildings have 28 to 85 zones"
- "The simulator, the training scenarios, the baseline controllers, the building generator, and a local scorer are in the [starter kit on GitHub](https://github.com/smart-buildings-2026/starter-kit)."
- Finding: no baseline scores, names, or values are published on the same-site pages inspected.

Source: https://smart-buildings-2026.github.io/competition/index.html

- "Main leaderboard: $25,000"; "Ranked on the Kaggle private leaderboard. Deadline January 15, 2027."; leaderboard prizes "1st $5,750"; "2nd $3,750"; "3rd $3,000"; "4th to 8th $2,500 each"
- Compete link verbatim: https://www.kaggle.com/competitions/neurips-2026-smart-buildings
- "Total prize pool: Kaggle code competition, first paper round, and second paper round"; "$40k Total prize pool"
- "Key dates: Nov 13, 2026 Round One deadline (NeurIPS poster)"; "Dec 13, 2026 NeurIPS Workshop in Paris"; "Jan 15, 2027 Final deadline"
- "Paper track: $15,000"; "[Submit papers on OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge)"; "[Paper template](https://drive.google.com/file/d/1NHtpV2Kwz39wuPDwA2FiyVWaQr7vGbWt/view)"
- "NeurIPS Poster Paper Awards. Deadline November 13, 2026." prizes "1st $1,000"; "2nd $750"; "3rd $500"; "4th to 14th $250 each"
- "Final Paper Awards. Deadline January 15, 2027." prizes "1st $3,000"; "2nd $1,750"; "3rd $750"; "4th to 12th $500 each"
- Finding: no paper review criteria, reviewer process, or Kaggle code-competition enforcement details (runtime limits, submission format, private/public split) are stated on the same-site pages inspected; rules page is external.

## 9. Buildings, weather scenarios, and simulator mechanics

Source: https://smart-buildings-2026.github.io/competition/competition.html

- "One AHU serves at most 10 zones; the training buildings have 3 to 9 AHUs."
- "The training buildings have 28 to 85 zones, depending on size and layout."
- "Each AHU serves a distinct group of rooms, and this mapping is unique to every building."
- "Figure 2. AHU-to-zone mapping for training building `building_train_1`: three floors, 39 zones, 4 AHUs (serving 10, 10, 9, and 10 zones)."
- "Figure 3. Three training buildings. Left to right: `building_train_8` (two floors, 3 AHUs, 28 zones), `building_train_1` (three floors, 4 AHUs, 39 zones), and `building_train_2` (two floors, 9 AHUs, 85 zones). Footprint, floor count, zone count, AHU grouping, and envelope construction all vary."
- "the generator samples a broad configuration space: 2 to 4 floors; a 400 to 700 ft x 300 to 500 ft footprint; composite layouts; 30 to 60 ft rooms; and 1 to 2 occupants per zone."
- "Across the 6 training buildings, K runs from 3 to 9 and V from 28 to 85" where K = AHUs, V = zones.
- "At each control step, a five-minute interval over a 30-day episode of 8,640 steps in which every step is scored"
- HVAC equipment: "AHU — Air Handler Units: Central systems that cool and mix air and distribute it via ducts."; "VAV Boxes: Terminal units in each zone. The policy sets each box's damper (the zone's share of air) and reheat valve (hot water to warm the air). They are the only per-zone controls."; "HWS — Hot Water System: A central boiler and pump supplying heated water to AHU coils and zone terminals."; "Zone: Individual rooms with independent temperature sensors and occupancy schedules."
- "Figure 1. The principal HVAC devices in a commercial building. Chillers and air handlers condition and distribute air, the boiler and hot-water system provide heating, and meters record the energy and carbon that the reward penalizes."
- "A coupled dynamical system: Changing one AHU's setpoint affects multiple zones; occupancy in one zone shifts the thermal load across the building."
- Simulator-enforced constraints: "a running AHU keeps a minimum static pressure, recirculation is capped at 70%, and every damper stays at least 10% open."
- "An AHU sends one supply-air temperature to all its zones, so the dampers and reheat valves are the only per-zone decisions."

Sources: https://smart-buildings-2026.github.io/competition/scoring.html and https://smart-buildings-2026.github.io/competition/index.html

- "Development runs on **6 public buildings x 5 weather stations = 30 scenarios**. Final evaluation runs on the held-out set: **6 unseen buildings x 5 unseen weather stations = 30 scenarios**."
- "The generator supports unlimited scenario creation, so participants can sample arbitrarily many additional buildings during development, while the held-out buildings and stations stay completely unseen."
- "**One rollout per scenario.** The simulator is deterministic: the same policy on the same scenario reproduces the same return bit for bit, across processes and machines."
- "A flat equal-weight mean over scenarios is the cleanest choice, so each building and weather pair counts exactly once."

## 10. Observation space, action space, and building metadata

Source: https://smart-buildings-2026.github.io/competition/competition.html

- "Both spaces are determined by the floor plan, chiefly the number of zones and AHUs, and are not fixed across the competition."
- "Each observation's length scales with the building (it grows with zones and air handlers). It encodes:"
- "Per zone: air temperature, current heating and cooling setpoints, damper position, reheat valve position, airflow setpoint."
- "Per AHU: supply-air temperature, static pressure, and airflow (sensed and setpoint), recirculation, a cooling-request count, and fan run status."
- "Hot-water system: supply-water temperature (sensed and setpoint)."
- "Occupancy & schedule: building-level occupancy count, plus comfort-mode flags for now and imminently."
- "Weather: outside-air temperature."
- "Time: time of day and day of week (cyclically encoded)."
- "Optionally, the policy may also consume the static floor plan and the AHU-to-zone mapping, supplied through the building metadata at runtime."
- "Discrete actions: an on/off command for each Air Handler Unit (AHU) and for the central Hot Water System (HWS)"
- "Continuous setpoints: for each AHU, the supply-air temperature, duct static pressure, and recirculation fraction; for the HWS, the boiler supply-water temperature and pump differential pressure; and for each zone's VAV box, the damper position and reheat valve position."
- "Because the number of AHUs and zones differs from building to building, both the observation and action vectors change shape per building, and there is no fixed input or output mapping the policy can memorize."
- "The action is a hybrid, variable-dimensional dictionary. For a building with **K** AHUs and **V** zones (one VAV box each): the **discrete** component has shape `(K+1)`, one on/off bit per AHU plus the Hot Water System; the **continuous** component has shape `(3K+2+2V)`... each normalized to [-1, 1]."
- "the discrete space spans **4 to 10 bits** and the continuous space **67 to 199 values**" across the 6 training buildings.
- "Training building `building_train_1` (Figure 2), with 4 AHUs and 39 zones, has a **5-bit** discrete action and a **92-value** continuous action. Read the index layout from the building metadata rather than assuming it."

Source: https://smart-buildings-2026.github.io/competition/submission.html

- "The class is initialized once per scenario with `building_meta_data` and `action_spec`. At each step the gateway calls `get_action(obs)`."
- "The metadata is what lets a single policy condition its control on the specific building it operates, rather than rely on a fixed input layout."
- "Action structure: The number of discrete and continuous actions, and the indices of the temperature, pressure, recirculation, water, and per-zone (damper and reheat) setpoints within the continuous action vector."
- "Observation structure: The observation field names and the indices of the zone temperature sensors."
- "Building layout: The floor plan, the zone map, the assignment of zones to AHUs, and the number of floors."
- "Schedule & comfort targets: Morning and evening start hours, the comfort window during work hours, the eco window for unoccupied periods, the time zone, and the holiday calendar."
- "Normalization: Observation and action normalization metadata that place raw sensor readings and setpoints on a common scale."
- "Because the floor plan, zone map, and zone-to-AHU assignments are delivered for every building, a policy can parse each building's structure and adapt"; guideline: "Read action and observation positions from the building metadata. Do not hardcode them."

## 11. Datasets, downloads, papers, and references

Source: https://smart-buildings-2026.github.io/competition/competition.html

- "The simulator, the training scenarios, the baseline controllers, the building generator, and a local scorer are in the [starter kit on GitHub](https://github.com/smart-buildings-2026/starter-kit)."

Source: https://smart-buildings-2026.github.io/competition/submission.html

- "The simulator, training scenarios, baselines, building generator, and local scorer are on GitHub. The starter notebook is on Kaggle."
- Links verbatim: [Starter kit on GitHub](https://github.com/smart-buildings-2026/starter-kit) and [Starter notebook on Kaggle](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/code)
- "A complete, runnable version is the **TransferPolicy Notebook** on the [Kaggle Code tab](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/code)."
- "To score against all 30 training scenarios, use `evaluate.py` and `score_leaderboard.py` from the [starter kit](https://github.com/smart-buildings-2026/starter-kit)."
- "Running the notebook yourself plays one public example scenario, so you can check that your policy loads and runs."
- "**Paper track.** Papers are submitted separately, on OpenReview. The NeurIPS poster round closes November 13, 2026, and the final round closes January 15, 2027."
- Same verbatim links: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge and https://drive.google.com/file/d/1NHtpV2Kwz39wuPDwA2FiyVWaQr7vGbWt/view
- Finding: no academic papers, citations, or reference list are published on any same-site page inspected (index, competition, submission, scoring, organizers). The only paper-related content is the paper-track submission channel and template.

Source: https://smart-buildings-2026.github.io/competition/index.html

- Entry links verbatim: [Compete on Kaggle](https://www.kaggle.com/competitions/neurips-2026-smart-buildings), [Starter kit on GitHub](https://github.com/smart-buildings-2026/starter-kit).
- Paper-track links verbatim: [Submit papers on OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge) and [Paper template](https://drive.google.com/file/d/1NHtpV2Kwz39wuPDwA2FiyVWaQr7vGbWt/view)
- "Paper track: $15,000"; "**NeurIPS Poster Paper Awards.** Deadline November 13, 2026."; "**Final Paper Awards.** Deadline January 15, 2027."
- Finding: no direct dataset file downloads are hosted on the same-site github.io pages; datasets/scenarios are distributed via the GitHub starter kit and Kaggle.

## 12. Verbatim link catalog and same-site probe findings

Same-site pages inspected:

- https://smart-buildings-2026.github.io/competition/index.html
- https://smart-buildings-2026.github.io/competition/competition.html
- https://smart-buildings-2026.github.io/competition/organizers.html
- https://smart-buildings-2026.github.io/competition/scoring.html
- https://smart-buildings-2026.github.io/competition/submission.html
- Same-site nav: https://smart-buildings-2026.github.io/competition/competition.html ("See observation & action spaces"); https://smart-buildings-2026.github.io/competition/organizers.html

Off-site participation links (verbatim):

- https://www.kaggle.com/competitions/neurips-2026-smart-buildings
- https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules
- https://www.kaggle.com/competitions/neurips-2026-smart-buildings/code
- https://github.com/smart-buildings-2026/starter-kit
- https://raw.githubusercontent.com/smart-buildings-2026/starter-kit/main/README.md
- https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge
- https://drive.google.com/file/d/1NHtpV2Kwz39wuPDwA2FiyVWaQr7vGbWt/view
- https://www.kajima.co.jp/english/
- https://www.distyl.ai/
- https://kodelabs.com/
- https://jobs.ashbyhq.com/Distyl/form/neur-ips-2026

Probe findings recorded in prior reports (HTTP 404 unless noted):

- https://smart-buildings-2026.github.io/competition/papers.html — 404; paper-track facts are on index.html.
- https://smart-buildings-2026.github.io/competition/faq.html — 404.
- https://smart-buildings-2026.github.io/competition/rules.html — 404; rules are off-site at https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules
- https://smart-buildings-2026.github.io/competition/schedule.html — 404; dates are on index.html.
- https://smart-buildings-2026.github.io/competition/workshop.html — 404; only workshop mention is Dec 13, 2026 NeurIPS Workshop in Paris.
- https://smart-buildings-2026.github.io/competition/evaluation.html — 404; protocol lives on scoring.html and competition.html.
- https://smart-buildings-2026.github.io/competition/baselines.html and https://smart-buildings-2026.github.io/competition/baseline.html — 404.
- https://smart-buildings-2026.github.io/competition/leaderboard.html — 404; ranking is Kaggle private leaderboard.
- https://smart-buildings-2026.github.io/competition/sitemap.xml and https://smart-buildings-2026.github.io/competition/robots.txt — 404.
- https://smart-buildings-2026.github.io/competition/data.html, simulator.html, buildings.html, scenarios.html, references.html — probed 404; simulator/scenarios/baselines/generator/scorer live in https://github.com/smart-buildings-2026/starter-kit
- https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules and https://www.kaggle.com/competitions/neurips-2026-smart-buildings — JS-shell only (5655-byte shell reported); rules/leaderboard/template text not retrievable server-side.
- https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge — JS shell text only (774 chars, 'Loading').

## 13. Unresolved

- rules-submission: missing complete evidence result
- evaluation: missing complete evidence result
- data-simulator: missing complete evidence result
- index.html closing sentence of About section not quoted in any report: 'Success here would mark a real leap for global sustainability and show what ML can do against climate change.'
- index.html 'A foundation model for building control' heading + intro sentence ('The same policy is deployed, unchanged, on buildings it has never encountered, inferring its strategy from the building description and live sensor stream it is handed at runtime.') not verbatim captured; only the competition.html paraphrase is quoted (result 4 item 4, result 2 item 1).
- index.html 'Generalize, don't tune' card heading + text ('The goal is not to optimize one controller for one building. It is to learn one policy that works across arbitrary floor plans, HVAC topologies, and climates.') not quoted in any report; only the competition.html paraphrase is quoted (result 1 item 3).
- index.html second sentence of Comfort-vs-energy-vs-carbon card not quoted: 'The objectives pull against each other.' (first sentence is covered via result 2 item 2).
- index.html 'Careers at Distyl AI' section heading + QR image (assets/images/distyl-ai-qrcode.png) + 'Apply to Distyl AI' label not described; only the bare careers URL is listed (result 1 item 7).
- index.html structural eyebrows/headings 'About the challenge', 'The shape of the problem', 'Key dates & prizes' not cited in any report (content under them is covered, labels are not).

## 14. Parent verification addendum (2026-11-02)

Direct parent-side fetch of `index.html` (200 OK, 12681 bytes) confirms the 6
verbatim items the critic flagged as unquoted. All quotes below are exact.

- About-section closing sentence: "Participants build control policies on a
  set of simulated buildings, then have those policies evaluated on an
  **unseen set of buildings** that vary in size, layout, weather, material
  properties, and HVAC system. Success here would mark a real leap for global
  sustainability and show what ML can do against climate change."
- Heading "A foundation model for building control" with intro: "The same
  policy is deployed, unchanged, on buildings it has never encountered,
  inferring its strategy from the building description and live sensor stream
  it is handed at runtime."
- Card "Generalize, don't tune": "The goal is not to optimize one controller
  for one building. It is to learn one policy that works across arbitrary
  floor plans, HVAC topologies, and climates."
- Comfort-vs-energy-vs-carbon card second sentence: "The objectives pull
  against each other." (full card: "A regret-based reward forces a genuine
  tradeoff: condition aggressively and burn energy, or switch off and lose
  comfort. The objectives pull against each other.")
- Section "Careers at Distyl AI": QR image `assets/images/distyl-ai-qrcode.png`
  (alt text "QR code: Distyl AI careers application for NeurIPS 2026") linking
  to `https://jobs.ashbyhq.com/Distyl/form/neur-ips-2026`, plus link label
  "Apply to Distyl AI" to the same URL.
- Structural eyebrows: "About the challenge", "The shape of the problem",
  "Key dates & prizes".
- Prize-table column header is "Prize (USD)"; all place/prize values match
  Section 4.

## 15. Residual unresolved

- Kaggle rules and competition pages render as JS shells server-side (5655-byte
  shell reported); official eligibility, team limits, submission format/limits,
  and deadline details could not be read from
  `https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules` and
  `https://www.kaggle.com/competitions/neurips-2026-smart-buildings`.
- OpenReview venue page returns JS-shell "Loading" text only; paper-track
  requirements and review process unread at
  `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge`.
- Drive template preview is login-walled; only the filename
  `Smart_Buildings_Competition_NeurIPS_2026_Template.zip` is verified.
- Guessed same-site pages (papers, faq, rules, schedule, workshop, evaluation,
  baselines, baseline, leaderboard, data, simulator, buildings, scenarios,
  references, sitemap.xml, robots.txt) all return HTTP 404; the live same-site
  set is index, competition, organizers, scoring, submission (.html).
- The three "missing complete evidence result" flags in the workflow terminal
  value are script-side spill artifacts: all four extractor submissions were
  delivered in full and inspected by the parent (results 1-4 above); only the
  off-site JS-walled items in this section remain genuinely unresolved.

