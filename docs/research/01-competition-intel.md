# Track 1 — Competition Intel: NeurIPS Smart-Buildings family + sibling benchmarks

> Historical research track. For current repository state and recommendations,
> read [06](06-repository-analysis.md), [09](09-evidence-audit.md) and
> [10](10-control-research-synthesis.md). Related competitions have different
> plants and metrics; their results do not identify this competition's leader
> or establish a universal algorithm ranking. Unopened sources remain leads.

**Question:** what has won (and by how much) in this competition family and its
siblings — and what does that imply for OUR actuators
(SAT / static pressure / recirculation / HWS temps + pump / VAV damper + reheat / on-off)?

**Method.** Every source below is labeled with an inspection level:
`FULL` = full body opened and read (HTML/PDF/README),
`ABS` = abstract/landing page only (full text paywalled or bot-walled),
`SNIP` = search-result snippet only (body not accessible, flagged in §9 Unresolved).
No code was touched; this file is the only write.
All margins are quoted as reported in the cited bodies; "lower is better" unless noted.

**Inspection date:** 2026-10-27. Total: 20 FULL bodies, 3 ABS, 4 SNIP/unresolved.

---

## 1. This competition: the NeurIPS 2026 Smart Buildings Challenge

### 1.1 Competition paper (FULL)

**Source:** Goldfeder et al., *The Smart Buildings Challenge: Learning Foundation
Models for Real-World Optimization at Scale*, arXiv:2610.18511, 13 Oct 2026.
`https://arxiv.org/html/2610.18511` — FULL (entire HTML incl. metrics, baselines, protocol, references).

Key facts (all verified against our repo README where overlapping):

- **Transfer framing.** 6 training buildings × 5 climate stations = 30 dev scenarios;
  eval is 6 *unseen* buildings × 5 *unseen* stations = 30 rollouts.
  The generator varies floors (2–4), footprint, perimeter, room layout, occupancy,
  6 envelope material properties, AHU count and zone assignment (≤10 zones/AHU);
  hot-water system stays one-per-building, one VAV box per zone.
- **Reward.** `r_t = 0.6·p_t − 0.2·c_t − 0.2·e_t` per 5-min step, T=8640 (30 days),
  every step scored. Comfort is linear to 5 K outside band, then quadratic and
  *unbounded*; summed over zones weighted by **occupancy, not floor area**.
  `r_t ≤ 0` always; the largest per-step reward ever observed is exactly 0
  (95 steps, all unoccupied + plant idle). Leaderboard = MAE vs 0 = mean per-step loss.
- **Baselines shipped** (all run the plant): `bang_bang`, `fixed_setpoint`
  (no night setback), `pid` (P-control + fan pressure ∝ effort), `pid_water`
  (+ modulating reheat + demand-driven hot-water reset). `all_off` is a reference only.
  On public training scenarios **bang_bang is the floor; pid_water or (hot/humid)
  fixed_setpoint is the ceiling. No baseline wins everywhere.**
- **Action/obs.** Hybrid: per AHU + HWS discrete on/off; per AHU 3 continuous
  (SAT, static pressure, recirc ≤70%); HWS 2 (supply-water T, pump dP); per VAV 2
  (damper ≥10%, reheat). Shapes `(K+1,)` discrete, `(3K+2+2V,)` continuous.
  Obs = per-zone T/setpoints/damper/reheat/airflow-SP; per-AHU SAT/pressure/
  airflow/recirc/fan; HWS T; building occupancy + schedule flags; OAT; time.
  Plus runtime metadata (floorplan, AHU map, indices, normalization).
- **Anti-cheat / protocol.** Public board = rotating visible subset; private = full 30.
  No-network sandbox; **wall-clock ≤ 10× bang-bang runtime per rollout**
  (unbounded online planning is disqualified); code scanned for hard-coded building IDs;
  top-7 re-run end-to-end at a fixed simulator commit. Simulator is bit-deterministic.
- **Prizes.** $25k leaderboard + $15k paper track; top teams get a NeurIPS podium talk,
  consortium-writeup co-authorship, and **post-contest live trials in real buildings**.

### 1.2 Competition website (FULL)

**Source:** `https://smart-buildings-2026.github.io/competition/index.html` — FULL.
Confirms §1.1 framing, dates (R1 paper 13 Nov 2026, Paris workshop 13 Dec 2026,
final 15 Jan 2027), sponsors (Kajima, Distyl, KODE Labs), and the "same policy
deployed unchanged, inferring strategy from building description + live sensors" API.

### 1.3 Starter kit (FULL)

**Source:** `https://github.com/smart-buildings-2026/starter-kit` (README via page fetch) — FULL.
17 stars / 6 forks at fetch time. Ships `smart_buildings/`, `evaluate.py`-style harness,
`score_leaderboard.py`, `baseline_returns.json`, `check_scales.py`, Dockerfile.
README's action/observation/ventilation-floor text matches the paper verbatim —
**"close everything" is not an available energy strategy** (min static pressure,
≤70% recirc, ≥10% damper).

### 1.4 Rules/organizers (ABS)

NeurIPS 2026 competitions announcement (`https://blog.neurips.cc/2026/07/28/neurips-2026-competitions-announced`)
lists the Challenge in the program — ABS (listing only; rules live on the Kaggle page,
which is JS-rendered, see §9).

---

## 2. Simulator origin: Google smart-buildings lineage (all FULL unless noted)

### 2.1 Open-source announcement (FULL)

**Source:** Google Open Source Blog, 17 Sep 2024,
`https://opensource.googleblog.com/2024/09/google-open-sources-smart-buildings-simulator-and-dataset-accelerate-sustainable-innovation.html` — FULL.
Google open-sourced (1) the **TensorFlow Smart Buildings Simulator** (RL training env)
and (2) **6 years of telemetry from 3 Google buildings**. Stated purpose: train RL
agents balancing comfort/energy/carbon; links Authors: Core Enterprise ML team (John Sipple).

### 2.2 Simulator repo (FULL)

**Source:** `https://github.com/google/sbsim`, README (raw, `master` branch) — FULL.
(Note: `main` branch 404s; `master` is live.)
README says the repo accompanies Goldfeder & Sipple's NeurIPS-2024-under-review paper
*Real-World Data and Calibrated Simulation Suite for Offline Training of RL Agents…*
and builds on the BuildSys'23 paper (§2.3). Entry point is a **Soft Actor-Critic demo
notebook** (TF-Agents); dataset on TFDS (`smart_buildings`) plus a small zip;
official API docs at `https://google.github.io/sbsim/`.

### 2.3 Lightweight calibrated simulation, BuildSys 2023 (FULL abstract page)

**Source:** Google Research pubs page for Goldfeder & Sipple (BuildSys 2023),
`https://research.google/pubs/a-lightweight-calibrated-simulation-that-enables-efficient-offline-learning-for-optimal-control-of-real-buildings/` — FULL (page incl. full abstract).
**Numbers:** on a 2-story 68,000 sq ft building with 127 devices, telemetry-calibrated
sim drifted **just over 0.5 °C from reality over a 6-hour horizon**. Thesis: one
lightweight calibrated sim *per building* to train offline RL at scale.

### 2.4 Smart Buildings Control Suite (FULL)

**Source:** Goldfeder, Dean, Jiang, Wang, Dong, Lipson, Sipple,
*The Smart Buildings Control Suite…*, arXiv:2410.03756 (v2, Jan 2025),
`https://arxiv.org/html/2410.03756` — FULL.
Three components: **(i) real telemetry from 11 buildings / 6 years** (TFDS),
**(ii) lightweight data-driven simulator per building** (Gym-compatible),
**(iii) modular PINN building model** as a simulator alternative.
Stated limitations (important for us): **no radiative heat-transfer model**
(added later by Ham et al. 2026, "tensorized radiative heat transfer", cited in the
competition paper) and calibration targeted temperature, not energy.
This suite is the direct ancestor of the competition sim ("HVAC simulation platform
from the Smart Buildings Control Suite, plus open-source improvements" — §1.1).

### 2.5 RL benchmark + policy extraction on the suite, ICCVW 2025 (FULL)

**Source:** Goldfeder, Trigo, Wyder, Kachappilly, Lipson, *Evaluating Performance of
RL Agents to Control Buildings Efficiently*, ICCVW 2025 (SEA workshop),
`https://openaccess.thecvf.com/content/ICCV2025W/SEA/papers/Goldfeder_Evaluating_Performance_of_Reinforcement_Learning_Agents_to_Control_Buildings_Efficiently_ICCVW_2025_paper.pdf` — FULL (7-page PDF).
Same-authors, same-simulator-family evidence — the closest thing to "what works in OUR sim":

- **SAC and DDPG both beat the baseline schedule** on Building SB1 (15-min steps),
  converge to similar returns ("near-optimal policy for this fixed condition").
- **Seasonal generalization works *within one building*:** trained on a single July
  14-day episode, both agents beat the schedule on Aug/Sep/Oct/Nov episodes;
  DDPG marginally better Sep–Nov, SAC slightly better in Aug.
- **Policy extraction (the BuildSys'24 idea):** averaging the DDPG agent's actions
  into static schedule bins of **2/4/8 h loses minimal return** — thermal inertia
  makes buildings insensitive to high-frequency control. **Even a 168-hour (weekly)
  static schedule beats the baseline**, i.e. the baseline is grossly inefficient and
  *simple improved heuristics have large headroom*.
- Takeaway for us: on this sim family, RL trains fine *per building*; the competition's
  cross-building gap is the entire game; and coarse schedules ≈ dynamic policies.

### 2.6 Policy-extraction paper, BuildSys 2024 (SNIP → unresolved)

Goldfeder & Sipple, *Reducing Carbon Emissions at Scale: Interpretable and Efficient
to Implement RL via Policy Extraction* (BuildSys 2024, doi 10.1145/3671127.3699535).
Full text not opened (ACM paywall; see §9). Its content is summarized *inside* the
ICCVW paper (§2.5, FULL): record RL actions over a forecast period → aggregate into a
static setpoint-vs-time table → deploy the table, not the agent. The ICCVW paper notes
the original study used an LSTM dynamics model (not the physics sim) and lacked
baseline comparisons and forecast-error sensitivity — gaps the ICCVW paper fills.

### 2.7 HOT dataset: 150k buildings for transfer (ABS/SNIP → unresolved full text)

Berkes et al., *A HOT Dataset: 150,000 Buildings for HVAC Operations Transfer Research*
(BuildSys 2025). ACM page (`https://dl.acm.org/doi/10.1145/3736425.3770110`) is
Cloudflare-walled; abstract visible via search: **159,744 building-weather combos,
envelope/occupancy/climate variations over all 19 ASHRAE zones and 76 global sites**,
plus a similarity framework for transfer feasibility. Relevant as the "procedural
infinite buildings" precedent for our generator; full text not opened (§9).

---

## 3. CityLearn family (NeurIPS 2020–2023): what won and by how much

CityLearn is battery/demand-response control, not HVAC airside — but it is the closest
*competition-series* precedent (same NeurIPS crowd, same generalization-across-buildings
pressure, same RBC→MPC→RL methods ladder). Pattern across all four editions:
**optimization + forecasting ensembles beat pure RL, and the winner is decided by
generalization to unseen buildings/weeks, not by peak public-board score.**

### 3.1 CityLearn 2020 (cited inside FULL bodies)

Per the MARL benchmark paper (§3.5, FULL): centralized **SAC took 2nd place**
(Kathirgamanathan et al. 2020). The 2021 winner paper (§3.2, FULL) adds that the
official MARLISA baseline (Yu et al.) scored 1.011 total vs RBC=1.0 — i.e. early RL
barely beat rules. Full 2020 overview not opened (§9).

### 3.2 CityLearn 2021 winner: ZO-iRL — adaptive optimization + guided evolution (FULL)

**Source:** Khattar & Jin, *Winning the CityLearn Challenge: Adaptive Optimization with
Evolutionary Search under Trajectory-based Guidance*, arXiv:2212.01939,
`https://arxiv.org/html/2212.01939` — FULL.
Method: use the **solution function of a (virtual-price) optimization as the policy**,
adapt its parameters online with **evolutionary search + trajectory-based guidance**;
per-day episodes, reward `−max(0,e_t)³`, only 3 candidate policies, 7-day buffer.
Official top-4 (total score, RBC ≡ 1.0, lower better):

| team | total | coord. score |
|---|---|---|
| **ZO-iRL (1st)** | **0.944** | **0.915** |
| ICD-CA (2nd) | 1.070 | 1.107 |
| IDLab (3rd) | 1.070 | 1.098 |
| Breakfast Club (4th) | 1.130 | 1.095 |

**Margin: −11.8% total and −17% coordination vs 2nd/3rd.**
Ablation vs RL baselines (Climate Zone 1, 10 runs): ZO-iRL 0.962 total vs SAC 1.092,
DDPG 1.102, PPO 1.199, MARLISA 1.011; on **ramping cost 0.781 vs 1.022 (best baseline)**,
i.e. −24%. Pure model-free RL *underperformed RBC* on most metrics.
Lesson: no train period existed (online control) — the winner was the team that
**planned with a model and adapted its parameters**, not the team with the best RL.

### 3.3 CityLearn 2022 winner: Team Together (Alibaba) — forecast + stochastic-opt + MAPPO ensemble (FULL ×2)

**Sources:**
(a) Nweye et al., *The CityLearn Challenge 2022: Overview, Results, and Lessons Learned*,
PMLR 220:85–103, `https://proceedings.mlr.press/v220/nweye23a/nweye23a.pdf` — FULL (20-page PDF).
(b) Jiang et al. (same Alibaba authors), *A Stochastic Online Forecast-and-Optimize
Framework… (SOFO)*, arXiv:2309.08642, `https://arxiv.org/html/2309.08642` — FULL.
(Both describe the same 1st-place entry: GBDT+linear forecast ensemble with
self-adaptive adjustment → decision ensemble of **stochastic optimization (MindOpt
solver) + MAPPO** (shared params/buffers) → rolling-horizon re-schedule.)

Official placements (PMLR): **1st Team Together** (forecast+stoch-opt+MAPPO);
2nd ambitiousengineers (forecast + DP→NN + CMA-ES evolution strategies);
3rd CUFE (linear/autoregressive forecasting); 4th DivMARL (diverse-model ensemble + SAC —
one shared SAC policy beat one-policy-per-house); 5th Greener (imitation learning,
decomposed per-building sub-objectives). 201 participants / 105 teams; median
participant score 0.907 vs fine-tuned RBC above median.
SOFO post-competition numbers (normalized to no-battery baseline; lower better):

| method | overall avg | test-set avg |
|---|---|---|
| RBC | 0.921 | 0.944 |
| MPC | 0.861 | 0.906 |
| best RL baseline (PPO) | 0.808 | 0.887 |
| **SOFO** | **0.795** | **0.862** |
| SOFO gain vs best baseline | **+1.3%** | **+2.1%** |

Component ablation (overall avg): MPC 0.861 → +rolling horizon 0.839 (+2.2%) →
+stochastic opt 0.805 (+3.4%) → +online fine-tuning **0.795 (+1.0%)**.
Team Together's own lessons (PMLR appendix, verbatim in spirit):
(i) **pretrained forecasting-then-decide beats feeding everything end-to-end**;
(ii) **optimization generalizes to unknown buildings, MARL wins on known buildings** —
hence the ensemble, since the two disagreed by month;
(iii) data augmentation for generalization is the open problem.

### 3.4 CityLearn 2023 winner: CHESCA — hierarchical optimization + RBC heuristics (FULL)

**Source:** Garmendia, Morri, Cappart, Le Cadre, *Winning the 2023 CityLearn Challenge:
a Community-based Hierarchical Energy Systems Coordination Algorithm*,
ECAI 2024 / HAL-04685791, `https://hal.science/hal-04685791v1/document` — FULL (9-page PDF).
Context: 600+ participants, 2500+ submissions, tiny 4-month hourly training set —
most entries overfit. CHESCA: **building-level individual optimization first, then a
central community controller for grid metrics**; interpretable stack
(XGBoost-ensemble forecasts + MPC-style battery/temp control + hourly RBC for DHW +
peak clamps); *"nothing tuned specifically on the dataset."*
2nd place: MPC (battery+temp) + RBC (DHW). 3rd: RL + curriculum + hand penalties.
Final leaderboard (lower better):

| team | private | public | comfort sub-cost |
|---|---|---|---|
| RBC baseline | 1.124 | 1.085 | 2.190 |
| **CHESCA (1st)** | **0.565** | 0.562 | **0.132** |
| Team 2 (2nd) | 0.575 | **0.464** | 0.304 |
| Team 3 (3rd) | 0.582 | 0.508 | 0.203 |
| CHESCA* (post-fix) | 0.548 | — | 0.129 |

**Margins: CHESCA −50% vs RBC; −1.7% vs 2nd on private — but note Team 2 won public
(0.464) and collapsed on private (0.575): the win came from zero public→private
variance, and the decisive sub-cost was comfort (0.132 vs 0.304/0.203).**
Forecast ablation (grid cost): ensemble 0.874 vs **oracle (perfect foresight) 0.848**
vs single XGBoost 0.900 — the forecaster was near the ceiling, so the remaining gains
were all in control/generalization. Discussion: "RL lacked the ability to generalize…
because of the small amount of data."

### 3.5 MARL benchmark on CityLearn 2026 (FULL)

**Source:** Khouja et al. (InstaDeep), *Characterizing MARL for Energy Control:
A Multi-KPI Benchmark on the CityLearn Environment*, arXiv:2602.19223,
`https://arxiv.org/html/2602.19223v1` — FULL.
Six MARL algorithms × DTDE vs CTDE × MLP vs temporal architectures, multi-seed,
beyond-mean KPIs (battery lifetime, per-agent contribution, worst case). Findings:
**DTDE (decentralized) consistently beats CTDE on stability/variability**; centralized
critics (MAPPO) reach higher peaks but vary more; **SAC best on carbon, IPPO most
stable** across IQM and worst-case; temporal dependencies help ramping/battery-DoD but
hurt short-horizon discomfort; no "lazy agents" — coordination is loose, so per-agent
policies deploy robustly. Confirms the 2020-SAC-2nd-place and 2022-MAPPO-winner history.

### CityLearn → us (synthesis)

1. **Forecast-then-optimize ensembles won 2 of 3 editions with published details**
   (2021 ZO-iRL, 2022 Together/SOFO); hierarchical opt+RBC won 2023. Pure RL never won.
2. Victory margins over the runner-up are **small on private (1–2%)** once everyone is
   near the ceiling — but huge vs RBC (−12% in 2021, −50% in 2023 with comfort weight).
3. The binding constraint is always **unseen-building generalization under data
   scarcity**; public-board leaders repeatedly fail private (Team 2, 2023).
4. Comfort-dominated scoring (2023: w_comfort=0.3 with comfort sub-cost spread 0.13–2.19)
   rewards **never-violate-comfort controllers** over clever energy plays — directly
   analogous to our 0.6 comfort weight and unbounded quadratic tail.

---

## 4. BOPTEST: MPC vs RL on physics simulators (FULL)

### 4.1 Framework (ABS + FULL-hosted derivatives)

Blum et al. 2021, *Building Optimization Testing Framework (BOPTEST)…*,
J. Building Performance Simulation 14(5):586–610 — ABS (abstract via publisher page;
full text paywalled, §9). What the abstract + the FULL papers below establish:
high-fidelity **Modelica emulators + standardized run-time environment + common KPIs**
(cost, discomfort in Kh, emissions), explicitly built for MPC-vs-RBC-vs-RL benchmarking.
No competition/league structure — it is a benchmark harness, so "margins" come from
head-to-head studies (§4.2–4.4), all pointing the same way.

### 4.2 TreeC: evolved decision trees ≈ best MPC, crush RL (FULL)

**Source:** Ruddick et al., *TreeC: a method to generate interpretable energy management
systems using a metaheuristic algorithm*, arXiv:2304.08310 (Knowledge-Based Systems),
`https://arxiv.org/html/2304.08310` — FULL.
Test case: BOPTEST **single-zone hydronic heat pump** (residential). TreeC evolves
*interpretable decision trees* with CMA-ES (12k evals, ~150 s each, parallelized).
Results: TreeC **matches the best MPC in 4 of 6 price×season scenarios** on both cost
and comfort, is slightly worse on cost in 2, and **beats RBC and every RL controller
in all 6** — including an RL controller with *identical* inputs/actions.
Adjusted averages (distance to per-scenario best; lower better):

| EMS | electricity cost [EUR/m²] | discomfort [Kh] |
|---|---|---|
| TreeC | 0.033 ± 0.021 | **0.42 ± 0.77** |
| MPC | 0.037 ± 0.029 | 2.91 ± 3.67 |
| RBC | 0.208 | 6.05 |
| RL | 0.266 ± 0.085 | 24.41 ± 23.47 |

The winning tree is trivial: **hold ~21.2 °C, just above the 21 °C occupied lower
bound** — MPC's extra cleverness (pre-cooling to 20 °C when unoccupied) added nothing.
Paper's verdict on RL: needs "multiple years of bad performance" to train; black-box;
uninterpretable. **Margin summary: TreeC/MPC ≈ 6–8× better than RL on cost,
≈ 8–60× on discomfort in this test case.**

### 4.3 Surrogate-model MPC on BOPTEST emulators (FULL)

**Source:** Mostafavi et al. (PARC), *Benchmarking MPC Algorithms in BOPTEST*,
arXiv:2301.13447v2, `https://arxiv.org/html/2301.13447v2/` — FULL.
Differentiable surrogate models + nonlinear MPC across BOPTEST cases: **reduces heating
and cooling loads while satisfying comfort and constraints** (qualitative win, no single
margin number). Failure modes documented: LSTM surrogate failed on the 5-zone case;
iLQR controller failed constraints; the **main scaling bottleneck is feasible data
generation** (sampling inputs that don't break the emulator). Relevant warning for any
greybox/MPC attempt on our side.

### 4.4 RL vs MPC on the BOPTEST hydronic heat pump (SNIP → unresolved full text)

Wang et al. 2023 (*Comparison of RL and MPC for building energy system optimization*,
BOPTEST Hydronic Heat Pump): search snippets (ResearchGate/IDSciences mirrors of the
abstract) report **only DDPG beats the baseline in both typical and peak-heating
scenarios; MPC is superior to all RL and baseline controllers in both** "because it can
take the best [actions]." A related MDPI review's comparison table (Arroyo et al. 2022:
RL beats PI on cost/discomfort tradeoff) was bot-blocked on fetch. Full texts not
opened (§9) — but the direction matches TreeC (§4.2) and the CityLearn record (§3).

### BOPTEST → us (synthesis)

On physics-grounded emulators with comfort constraints, the ranking is stable:
**MPC ≈ evolved-interpretable-policies > RBC ≫ model-free RL**, with RL often worse
than rules. The TreeC mechanism (evolve a tiny interpretable policy in sim, deploy the
tree) is directly portable to our per-zone damper/reheat + AHU-reset structure.

---

## 5. Sinergym + Building2Building: RL-vs-RBC and generalization (FULL)

### 5.1 Sinergym DRL evaluation (FULL)

**Source:** Manjavacas et al., *An experimental evaluation of DRL algorithms for HVAC
control*, arXiv:2401.05737 (v3, Aug 2024), `https://arxiv.org/html/2401.05737` — FULL.
Sinergym = EnergyPlus + Gymnasium wrapper (5ZoneAutoDXVAV VAV building,
2ZoneDataCenterHVAC, more). Head-to-head with raw obs, modest tuning:

- **5-zone VAV (our closest analogue): RBC wins overall; SAC is the closest DRL**,
  then PPO ≈ TD3. No agent beats RBC on reward. SAC saves power vs RBC at small
  comfort cost, mainly in hot climates. TD3 is the most energy-hungry.
- **2-zone datacenter: TD3 beats RBC on reward** via power savings at higher comfort
  risk (rides the comfort limit; RBC stays conservative). SAC/PPO ≈ RBC.
- **Climate transfer:** training≠eval climate costs only *minor* reward loss
  (SAC 5Zone table: e.g. train-cool/test-hot −0.258 vs train-hot/test-hot −0.454 —
  the *test climate dominates*, train climate barely matters). All differences p<0.05
  but "generally acceptable."
- **Sequential (continual) training across climates fails** — catastrophic forgetting,
  agents specialize in the latest climate.
- **Reward engineering:** past a point, raising comfort weight stops helping
  (physical limit); sometimes comfort gains cost ~zero extra energy.
  (The Sinergym *framework* paper itself, Campoy-Nieves et al., E&B 327:115075, is
  paywalled — ABS via PyPI/citations only, §9.)

### 5.2 Building2Building: 6000-building generalization benchmark (FULL)

**Source:** Taboga et al. (Mila), *Building2Building: A Large Scale Benchmark for
Generalizable Real-World RL*, arXiv:2607.16534, `https://arxiv.org/html/2607.16534` — FULL.
Parametric EnergyPlus generator → 6000 Gymnasium envs, heterogeneous obs/action spaces;
task splits: goal/dynamics/action-space/cross-domain adaptation. Baselines:
**ASHRAE-G36-style reactive control (zone PI loops + Trim-and-Respond SAT reset)** as
the primary baseline + PPO. Findings:

- **Per-building PPO specialists beat generalist PPO by a large gap** (dynamics
  adaptation, 100 test houses); generalists show a **long tail of transfer failures**.
- **Conditioning the policy on building parameters** (floor area, year, actuator
  count…) closes part of the specialist gap — i.e. metadata-aware policies win.
- Cross-domain (retail/restaurant/2 offices, heterogeneous spaces): an
  **Amorpheus-style transformer** (typed encoders + shared transformer + typed decoders)
  adapts to unseen buildings and **beats the reactive baseline on some of them**
  (normalized return < 1.0 on a subset of 20 test buildings).
- B2B is the sibling that most resembles our competition (procedural buildings,
  heterogeneous spaces, G36-ish baselines) — and its headline is that
  **generalization is still wide open: specialists ≫ generalists**.

---

## 6. In-context / pretrained transfer: HVAC-DPT + live RL deployments (FULL)

### 6.1 HVAC-DPT: decision-pretrained transformer (FULL)

**Source:** Berkes, *HVAC-DPT: A Decision Pretrained Transformer for HVAC Control*,
arXiv:2411.19746 (NeurIPS'24 Climate-AI), `https://arxiv.org/html/2411.19746v1` — FULL.
Method: train **100 diverse PPO policies** across training buildings (EnergyPlus+COBS)
→ distill interaction histories into a **GPT-2 causal transformer** predicting
expert-labeled actions from (query state + in-context history) with MSE loss →
deploy in unseen buildings *with no weight updates*, refining in-context.
Year-long eval in an unseen Denver building (different size + HVAC design +
state/action dims vs train):

- **−45% HVAC energy vs Baseline** (50%-damper controller);
- **only 5% worse than the per-building Expert** (engineer-designed for that building);
- **SARL/MARL trained from scratch on the target: +74%/+70% worse** (RL needs up to
  **1,250 years** of training interactions — citing Krishna et al. 2023).
Conclusion numbers: −45% vs baseline ops, −70% vs RL agents in year one.
This is the strongest published "foundation-model-for-HVAC" result and the direct
precedent for any transformer/policy-distillation approach on our side.

### 6.2 BCOOLER: DeepMind+Google+Trane live chiller-plant RL (FULL)

**Source:** Luo et al., *Controlling Commercial Cooling Systems Using RL*,
arXiv:2211.07357 (via `https://ar5iv.labs.arxiv.org/html/2211.07357`) — FULL.
Two **live 3-month A/B tests** on real commercial facilities vs Trane's
heuristics-based sequence of operations (SOO) — with the SOO first *retro-commissioned*
("AI ready" tuning), so the baseline was above average. Method honesty worth copying:
alternating-day control, 2 h post-handover exclusion, temp×load bucket normalization.
Results: **−9% and −13% plant energy at equal comfort** (manager constraints respected);
savings concentrate at **cooler OAT and lower loads** (headroom vanishes near max
capacity); savings grow over time (better model + fixes + fall weather).
Behaviors found: e.g. colder condenser water (more tower power, less chiller power).
Takeaway: even tuned vendor heuristics leave ~10% on the table for RL — but only
where the plant has headroom, and only with constraint-safe deployment.

---

## 7. Normative anchor: what a well-run building does (FULL)

**Source:** Trane, *Trane Certified for ASHRAE Guideline 36*,
`http://www.trane.com/commercial/north-america/us/en/about-us/newsroom/blogs/ashrae-guideline-36.html` — FULL.
Vendor statement of the G36 pieces: **Trim-and-Respond** dynamically resets AHU
discharge-air T, duct static pressure, CHW/HW supply T from zone requests; alarm
suppression (VAV alarm suppressed if the AHU is also in fault); AHU auto-FDD; dynamic
ventilation setpoints per system and per zone. This is the "practitioner ceiling" our
`pid_water`-style baselines approximate — and what B2B uses as its reactive baseline
(§5.2). (Richer G36 sources — Reliable Controls explainer, LBL Modelica G36 docs —
found but Cloudflare/unfetched; snippets confirm canonical T&R params, e.g. SAT reset
12–18.3 °C with +0.1/−0.2/−0.6 °C trim/respond steps and rogue-zone counting, §9.)

---

## 8. Actionable takeaways mapped to OUR actuators

Context: VAV-reheat-only plant (no AHU heating coil), VAV discharge cap 308.15 K,
recirc ≤70%, dampers ≥10%, weights (0.6/0.2/0.2), comfort slope ~0.18/K ≫ energy slope,
TOU peak/night only 1.25×, constant COP 3.5, building-level occupancy only, ≤10 zones/AHU.

1. **Comfort-first, never-violate is the winning objective shape.** CHESCA won 2023 on
   the comfort sub-cost (0.132 vs 0.304/0.203, §3.4); TreeC won by hugging the occupied
   lower bound (§4.2); our 0.6 weight + unbounded quadratic tail says the same.
   → Actuators: **reheat valves and dampers must close comfort gaps before any energy
   play**; energy/CO₂ trims (SAT reset, static-pressure trim, recirc, HWS reset, night
   setback) only in the comfort-safe residual. The leader beating even our *train mean*
   smells like fewer comfort violations, not better arbitrage.
2. **Forecast-then-optimize beats end-to-end learning — twice.**
   ZO-iRL 2021 (−11.8%, §3.2) and Together/SOFO 2022 (+1–2% at ceiling, §3.3) both won as
   forecaster→optimizer pipelines with online adaptation (guided ES / rolling horizon +
   self-adaptive adjustment). SOFO's ablation (+2.2% rolling, +3.4% stochastic,
   +1.0% fine-tune) is a recipe.
   → Actuators: a **receding-horizon scheduler over SAT/static-pressure/recirc/HWS with
   per-zone damper+reheat tracking**, re-solved on a slow cadence (ICCVW: 8 h bins ≈
   lossless, §2.5), fed by occupancy/OAT forecasts — not a per-step neural policy.
3. **Evolve tiny interpretable policies in sim; deploy the artifact, not the learner.**
   TreeC (§4.2): CMA-ES-evolved tree ≈ best MPC, 6–8× better than RL on cost.
   ICCVW (§2.5): extracted static schedules ≈ DDPG, weekly bins still beat baseline.
   Our 10× bang-bang wall-clock cap (§1.1) *forces* this shape anyway.
   → Actuators: evolve per-AHU reset curves (SAT/pressure vs OAT + request count) and
   per-zone damper/reheat deadbands; ship lookup tables/curves.
4. **Condition on building metadata; expect specialist≫generalist.**
   B2B (§5.2): param-conditioned policies close the gap; unconditioned generalists have
   a failure tail; specialists win big. We get floorplan/AHU-map/indices at runtime (§1.1).
   → Actuators: normalize every loop by zones-per-AHU, use AHU→zone assignment for
   request counting (T&R needs per-AHU cooling-request counts — present in obs, §1.1),
   and treat any no-metadata policy as a known-loser.
5. **Trim-and-Respond is the right *structure* for SAT/static-pressure/HWS — but our
   hidden-eval failure says the *parameters* don't transfer.**
   G36/B2B/CHESCA all validate request-based reset; our suite predicted −13% yet hidden
   scored 0. CHESCA's fix: *"nothing tuned specifically on the dataset"* + ensemble
   forecasts ≈ oracle (§3.4); Together's fix: self-adaptive prediction adjustment (§3.3).
   → Actuators: keep T&R for SAT/pressure/HWS-T/pump-dP but **self-tune trim/respond
   rates online** (request-rate feedback, rogue-zone exclusion) instead of fixed constants.
6. **Don't pay for arbitration the tariff can't fund; do exploit unoccupied hours.**
   CHESCA's outage/comfort split, BCOOLER's low-load concentration (§6.2), and our dead
   1.25× TOU all say: **the prize is switching things OFF (or to setback) when
   unoccupied**, not load-shifting. Obs has building-level occupancy + schedule flags (§1.1).
   → Actuators: aggressive discrete on/off + night setback on AHUs/HWS (the `fixed_setpoint`
   baseline explicitly lacks it and still wins hot/humid evals — i.e. setback handling
   separates policies); keep ventilation floors legal (≥10% damper, min pressure, ≤70% recirc).
7. **Per-zone air allocation (dampers) is the only per-zone lever — protect it.**
   Paper: "dampers and reheat valves are the only per-zone levers" (§1.1); HVAC-DPT
   controlled *only dampers* and reached −45% / near-Expert (§6.1).
   → Actuators: damper logic deserves the most optimization budget (per-zone comfort
   tracking + AHU-level air balancing); reheat second (comfort rescue within the 308.15 K
   cap); SAT/pressure/recirc are shared compromises driven by zone requests.
8. **What NOT to do (evidence-backed):** pure model-free RL from scratch (never won
   CityLearn; 6–60× worse on BOPTEST; +70% worse in HVAC-DPT year one; 1250-year sample
   needs); sequential multi-climate fine-tuning (catastrophic forgetting, §5.1);
   public-board hill-climbing (2023 Team 2: public 0.464 → private 0.575, §3.4 —
   mirrors our discredited transfer suite); unbounded online planning (disqualified, §1.1).

---

## 9. Sources table + unresolved gaps

### Opened (FULL) — 20 bodies

| # | source | URL | what was inspected |
|---|---|---|---|
| 1 | competition paper | `https://arxiv.org/html/2610.18511` | full HTML: task/reward/baselines/protocol/refs |
| 2 | competition site | `https://smart-buildings-2026.github.io/competition/index.html` | full page: prizes/dates/framing |
| 3 | starter kit | `https://github.com/smart-buildings-2026/starter-kit` | README + file list via page fetch |
| 4 | Google OSS blog | `https://opensource.googleblog.com/2024/09/google-open-sources-smart-buildings-simulator-and-dataset-accelerate-sustainable-innovation.html` | full post |
| 5 | sbsim README | `https://raw.githubusercontent.com/google/sbsim/master/README.md` | full README (main-branch 404 noted) |
| 6 | BuildSys23 sim | `https://research.google/pubs/a-lightweight-calibrated-simulation-that-enables-efficient-offline-learning-for-optimal-control-of-real-buildings/` | full page incl. abstract + numbers |
| 7 | Control Suite | `https://arxiv.org/html/2410.03756` | full HTML: abstract/intro/conclusion/refs |
| 8 | ICCVW25 RL eval | `https://openaccess.thecvf.com/content/ICCV2025W/SEA/papers/Goldfeder_Evaluating_Performance_of_Reinforcement_Learning_Agents_to_Control_Buildings_Efficiently_ICCVW_2025_paper.pdf` | full 7-page PDF |
| 9 | CityLearn21 win | `https://arxiv.org/html/2212.01939` | full HTML incl. Tables 1–2, appendix |
| 10 | CityLearn22 PMLR | `https://proceedings.mlr.press/v220/nweye23a/nweye23a.pdf` | full 20-page PDF: placements/lessons/board |
| 11 | SOFO (CL22 win) | `https://arxiv.org/html/2309.08642` | full HTML incl. Tables 1–2 ablations |
| 12 | CHESCA (CL23 win) | `https://hal.science/hal-04685791v1/document` | full 9-page PDF incl. Tables 1–2 |
| 13 | MARL benchmark | `https://arxiv.org/html/2602.19223v1` | full HTML: history/benchmark/conclusion |
| 14 | TreeC/BOPTEST | `https://arxiv.org/html/2304.08310` | full HTML incl. Tables 2–3 |
| 15 | BOPTEST-MPC | `https://arxiv.org/html/2301.13447v2/` | full HTML: method/conclusion/failures |
| 16 | Sinergym DRL | `https://arxiv.org/html/2401.05737` | full HTML: results/conclusions/Table 5 |
| 17 | B2B | `https://arxiv.org/html/2607.16534` | full HTML: baselines/§6/conclusion |
| 18 | HVAC-DPT | `https://arxiv.org/html/2411.19746v1` | full HTML: method/results/conclusion |
| 19 | BCOOLER live RL | `https://ar5iv.labs.arxiv.org/html/2211.07357` | full HTML: eval method/§5.2 numbers |
| 20 | Trane G36 | `http://www.trane.com/commercial/north-america/us/en/about-us/newsroom/blogs/ashrae-guideline-36.html` | full article text |

ABS-level (landing/abstract only): NeurIPS'26 competitions announcement (listing);
BOPTEST official paper page (abstract); HOT-dataset search abstract.

### Unresolved (body not opened; claims above flagged SNIP/ABS, not FULL)

1. **Kaggle competition forum/kernels** (`kaggle.com/competitions/neurips-2026-smart-buildings/discussion`,
   `/code`): page fetch returns a 6 KB JS shell with no content; notebooks/discussions need
   login + JS rendering. **No shared-solution intel obtained.** Retry with browser/API creds.
2. **BOPTEST official paper full text** (Blum et al. 2021, T&F paywall) — KPI/test-case
   details beyond the abstract not verified.
3. **Wang et al. 2023 RL-vs-MPC on BOPTEST** full text — direction (MPC > DDPG > rest) from
   snippets only; MDPI mirror bot-blocked (Access Denied).
4. **Sinergym framework paper** (Campoy-Nieves et al., E&B 327:115075) — paywalled;
   framework claims rest on the DRL-eval paper + PyPI metadata.
5. **Policy-extraction full text** (Goldfeder & Sipple, BuildSys'24, ACM paywall) —
   covered via its ICCVW'25 summary (§2.5).
6. **HOT dataset full text** (ACM, Cloudflare-walled) — scale numbers from abstract snippet.
7. **LBNL Modelica G36 docs + Reliable Controls T&R explainer** — found, not fetched
   (Cloudflare); canonical T&R parameters from snippets only.
8. **CityLearn 2020 official overview** — 2nd-place-SAC fact via the MARL paper (§3.5);
   winner/margins not verified; 2023 edition official overview (aicrowd/NeurIPS pages)
   likewise not opened beyond CHESCA's account.

---
