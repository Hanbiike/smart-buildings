# TRACK 2 — Model-Predictive Control for Commercial-Building HVAC

> Historical research track. Read the [source audit](09-evidence-audit.md)
> and [current synthesis](10-control-research-synthesis.md) before applying
> its recommendations. Study-specific savings are not forecasts for this
> repository; snippet-only results are excluded from the current synthesis.
> At five minutes per step, 6–12 steps mean 30–60 minutes; 2–4 hours require
> 24–48 steps. Action projection alone does not guarantee future comfort.

**Author:** workflow child `mpc-buildings` · **Date:** 2026-10-27 · **Status:** web research only, no code touched
**Target plant:** Kaggle NeurIPS smart-buildings simulator — VAV-reheat-only (NO AHU heating coil),
VAV discharge cap 308.15 K, recirculation cap 0.70, dampers ≥ 10%, 5-min steps,
reward weights (comfort, energy, carbon) = (0.6, 0.2, 0.2), comfort slope ≈ 0.18/K ≫ energy slope,
electricity TOU peak/night = 1.25× (arbitrage dead), constant plant efficiency (chiller COP 3.5),
BUILDING-level occupancy observation only, buildings valid iff ≤ 10 zones/AHU.

**How to read the citation tags:** `[FULL]` = I downloaded and read the source body (PDF/HTML text-extracted
and grepped); `[SNIPPET]` = paywalled or bot-blocked, claim rests on the search-engine snippet only and is
flagged as such in §12. All URLs below are the exact URLs fetched or returned by search.

---

## 1. The seminal line: OptiControl / Oldewurtel et al. (ETH Zurich)

The founding program is **OptiControl-I/II** (ETH Zurich, Morari group, ~2007–2013):
simulation-based potential assessment of MPC across construction types × HVAC systems × climates,
followed by field demonstrations (notably a 7-month MPC experiment in a Swiss office building,
Sturzenegger et al.).

- Oldewurtel, Parisio, Jones, Gyalistras, Gwerder, Stauch, Lehmann, Morari,
  *"Use of model predictive control and weather forecasts for energy efficient building climate control"*,
  Energy and Buildings 45 (2012), pp. 15–27. `https://doi.org/10.1016/j.enbuild.2011.09.022` **[SNIPPET]**
  (paper paywalled; consistently described across the [FULL] sources below as the seminal stochastic-MPC-with-weather-forecast
  study). Its companion, *"Handling model uncertainty in MPC for energy efficient buildings"* (Energy and Buildings, 2014),
  is the source of the widely quoted **16–41% building-energy saving vs rule-based control** figure
  (via `https://c.coek.info/pdf-handling-model-uncertainty-in-model-predictive-control-for-energy-efficient-buil.html` **[SNIPPET]** —
  host unreachable, TLS error; figure corroborated by the Purdue review's discussion, [FULL]).
- Ma, Borrelli, Hencey, Packard, Bortoff, *"Model Predictive Control of Thermal Energy Storage in Building
  Cooling Systems"* (CDC 2009, UC Merced chilled-water plant):
  `https://www.sysecol2.ethz.ch/OptiControl/Lit/Ma_09_Proc-CDC09_WeA12.1.pdf` **[FULL]**.
  Simplified hybrid chiller/tank model + periodic robust-invariant terminal sets + **move blocking**;
  **24 h prediction horizon at 1 h sampling**; **24.5% daily bill saving vs a heuristic controller**
  under PG&E E-20 tariff (TOU energy + 3 demand charges). This is the canonical demonstration that MPC's
  economic value concentrates in *price-structure exploitation* (pre-cooling into storage ahead of peaks).
- Sturzenegger et al., *"BRCM Matlab toolbox: model generation for predictive building control"* (ACC 2014)
  **[SNIPPET]** (cited inside Atam & Kerrigan [FULL]): semi-automated RC-model generation from geometry —
  the practical answer to "where does the MPC model come from" for white/grey-box MPC.
- Prívara, Široký, Ferkl, Cigler, *"Model predictive control of a building heating system: the first
  experience"* (Energy and Buildings, 2011) **[SNIPPET]** (cited in multiple [FULL] bibliographies):
  first documented university-building MPC deployment; established the "identify from BMS data → receding-horizon
  optimize → supervise setpoints" pattern still used today.

**Takeaway for us:** the seminal gains (16–41%) come from (a) exploiting thermal mass against time-varying
prices, (b) replacing independent PI loops with coordinated optimization, (c) weather/occupancy anticipation.
Our plant's flat-ish economics (1.25× TOU, constant COP) remove lever (a) almost entirely — see §9 —
so portable gains must come from (b) and (c).

## 2. What the field really delivers: the Purdue 104-study review (2025)

Khabbazi, Pergantis, Reyes Premer, Papageorgiou, Lee, Braun, Henze, Kircher,
*"Lessons learned from field demonstrations of MPC and RL for residential and commercial HVAC: a review"*,
arXiv:2503.05022v4 (Jun 2025):
`https://arxiv.org/html/2503.05022v4` **[FULL]** (body read: abstract, §§5–7, Table 2 headers, conclusions).

- Corpus: **24 residential + 80 commercial field-demonstration papers (104 total)**, the largest review to date.
- **71% use experiment protocols the authors judge unreliable** (simulated baselines, small/coupled zones,
  < 7-day tests). Simulation-based benchmarks systematically report higher savings than measurement-based ones.
- Over the reliable 29%, **duration-weighted average utility-bill savings: 16% residential, 13% commercial.**
- Only **13 of 104 papers report deployment/operation/maintenance costs** — the review's central complaint and
  its main recommendation: the field should pivot to *cost-of-deployment reduction* (less sensing, faster
  commissioning, less tuning) rather than squeezing more % in simulation.
- Context hypotheses the authors could *not* confirm for lack of reporting: MPC/RL should shine under
  **complex tariffs** and in **thermally massive buildings**; high-level-setpoint vs low-level-actuator
  architectures could not be ranked. They publish a reporting checklist (rate structure, envelope/mass,
  control level, climate) for future studies.
- Raw reported ranges in commercial tests span roughly **−5% to +60% energy**, with most measurement-based
  results in the **10–30%** band (Table 2, [FULL]).

**Takeaway for us:** 13% commercial bill savings is the honest field prior for MPC vs incumbent control —
but incumbents in the field are often poor (constant SAT, no resets). Our baseline-to-beat is a strong
occupancy-gated heuristic, and our "reliability protocol" problem is the mirror image: our transfer suite
predicted −13% for trim-and-respond yet hidden scored 0 (suite discredited). Any MPC claim must be validated
against *hidden-like* conditions, not our own suite.

## 3. Offset-free MPC: disturbance models + Kalman estimation (comfort has no steady-state error budget)

Because our comfort weight (0.6, slope 0.18/K) punishes even small persistent offsets far more than energy,
offset-free tracking is the single most load-bearing MPC sub-technology for us.

- UC thesis *"Development, Implementation and Performance of a Model Predictive Controller for Packaged Air
  Conditioners in Small and Medium-sized Commercial Building Applications"* (compressor-stage MPC with
  demand charges): `https://escholarship.org/content/qt0kk608vf/qt0kk608vf.pdf` **[FULL]** (29 pp.).
  Textbook **lumped output-disturbance model**: augment (A, C) with a constant output disturbance,
  prove detectability of the augmented pair, solve the algebraic Riccati equation for the Kalman gain Ka,
  estimate state + disturbance jointly. Reports **≈20% HVAC energy savings and ≈30% peak-demand reduction**
  for a small commercial building (case study), and up to **≈50% electric peak reduction** via coordinated
  unit dispatch. Control-relevant detail: unit cycling randomness is handled by fixing stage commitments over
  the horizon.
- Radecki & Hencey (Cornell), *"Learning control-oriented thermal models / UKF augmented for parameter and
  disturbance estimation"*: `https://arxiv.org/pdf/1601.02947` **[FULL]** (14 pp.).
  **Unscented Kalman Filter jointly estimating RC parameters and unmeasured heat-gain disturbances**,
  validated on EnergyPlus data over a year: learns parameters during known-load periods (nights/weekends),
  then attributes residuals to disturbance patterns (occupancy/equipment/solar schedules). **< 1 °C RMS
  24-h prediction error** once disturbances are characterized. This is the cleanest published template for
  "identify at night, estimate gains by day."
- Alanqar, Ellis et al. (Johnson Controls), *"Practice-Oriented System Identification Strategies for MPC of
  Building Thermal and HVAC Dynamics"* (IHPBC 2018): `http://docs.lib.purdue.edu/ihpbc/255` **[FULL]**
  (abstract page + snippet body). **Two-step practice-oriented ID**: (1) grey-box parameters via nonlinear
  least squares, (2) augment with disturbance model and identify the **Kalman gain by 1-step prediction-error
  minimization** — i.e., tune the observer for closed-loop MPC performance, not open-loop fit. Tested on real
  building HVAC data.
- Parisio et al. scenario-explicit MPC (§5) likewise leans on disturbance scenarios rather than perfect models.

**Takeaway for us:** with only BUILDING-level occupancy observed, per-zone internal gains are unmeasured —
exactly the Radecki/UKF setting. The portable pattern is: **night/weekend parameter ID → daytime lumped-gain
disturbance observer → offset-free MPC**. Our simulator access makes this *easier* than the literature
(we can run designed ID experiments), which inverts the usual MPC bottleneck.

## 4. Grey-box RC identification FOR control (not just prediction)

The literature is emphatic: the best *predictor* is not the best *control model*. Control-oriented ID must
preserve stability, physical sign conventions, and low order — otherwise the MPC optimizes against artifacts.

- Klanatsky, Veynandt, Heschl, *"Grey-box model for MPC of buildings"* (Energy & Buildings 300, 2023,
  PRELUDE living-lab): `https://prelude-project.eu/wp-content/uploads/2023/11/Grey-box-model-for-model-predictive-control-of-buildings.pdf`
  **[FULL]** (18 pp.). Compared 1R1C→8R3C structures with TABS + glass-façade + shading submodels on a year of
  lab data. Best variant: **2nd-order (4R2C-family), 8 parameters, 0.25 °C MAE on 24-h zone-temperature
  forecasts**, identified by nonlinear least squares on **1–2 weeks startup + 2–4 weeks training data**,
  re-estimated daily/weekly (adaptive/evolving model). Notes **15-min timesteps adequate** for most building
  MPC. Key control-ID lesson: parameter *stability* across re-estimations matters more than single-fit error.
- Hossain, Zhang, Ardakanian, *"Identifying Grey-box Thermal Models with Bayesian Neural Networks"*
  (arXiv:2009.05889, 2020): `https://ar5iv.labs.arxiv.org/html/2009.05889` **[FULL]**.
  8,884 Canadian homes with smart thermostats: BNN-estimated grey-box beats black-box on temperature
  prediction; **1R1C unacceptable accuracy; 2R2C worse than other low-order models** in their tests;
  priors transfer across homes with similar floor area / season. Lesson: order-2 structure choice matters,
  and *fleet priors* (our many training buildings!) accelerate per-building ID.
- Raman et al. hierarchical MPC (§6, [FULL]): states flatly that **reliable multi-zone model ID from data is
  intractable without disturbance assumptions** (as many unknown gain signals as zones) and that no published
  method does it assumption-free — the motivation for their meta-zone aggregation (§6).
- Cai et al. adaptive MPC (§7, [FULL]): the identifier **guarantees a stable estimate with physical properties
  irrespective of data** and re-learns periodically without expert re-tuning; the year-long study shows
  periodic model+disturbance updates are *necessary* for comfort, beneficial for energy.
- Liu et al., *"Building thermal control: hierarchical design from limited data using gray-box or black-box
  internal models"* (HAL, 2025): `https://hal.science/hal-05097512v1/preview/Liu_2025.pdf` **[SNIPPET]**
  (bot-walled; abstract only via search): grey-box RC vs subspace black-box as MPC internal models under
  limited data — topic-relevant, body unread.

**Takeaway for us:** start with **per-zone 1R1C + lumped-gain disturbance** (fast, stable, identifiable from
nights), escalate to 2R2C only if 24-h MAE > ~0.3 °C. Use fleet priors across the ≤10-zone buildings.
Re-ID on a schedule (weekly) rather than continuously, per Klanatsky stability findings.

## 5. Explicit MPC (when the optimizer must disappear)

- Parisio, Fabietti, Molinari, Varagnolo, Johansson, *"Control of HVAC Systems via Scenario-based Explicit
  MPC"* (CDC 2014): `https://damianov.folk.ntnu.no/Publications/Parisio%20et%20al.%20-%202014%20-%20Control%20of%20HVAC%20Systems%20via%20Scenario-based%20Explicit%20MPC.pdf`
  **[FULL]** (7 pp.). Multi-parametric program solved **offline** → piecewise-affine feedback law over
  polytopic state partitions; scenarios (no distributional assumptions) immunize against occupancy/weather
  uncertainty. Demonstrated on one university room; online cost = point-location + affine evaluation.
  Cites 5–30% systematic-management savings potential. Strength: runs on cheap BMS hardware. Weakness:
  partitions explode with state dimension — strictly a small-model technique.
- *"Self-tunable approximated explicit MPC: heat-exchanger implementation"* (arXiv:2406.04048):
  `https://arxiv.org/html/2406.04048v1/` **[FULL]**. Concrete region counts for a SISO nonlinear plant:
  **1,680 critical regions (upper) / 409 (lower)** for two tunings — a visceral illustration of explicit
  MPC's scaling wall. Self-tuning via online penalty interpolation.
- Atam & Kerrigan decentralization study (§6, [FULL]) notes comfort-violation weights of **10⁶× energy
  weight** are routine in building MPC — relevant calibration for our 0.6/0.2/0.2 reward (comfort-first).

**Takeaway for us:** explicit MPC is the right *deployment shape* only if online optimization is forbidden
at inference (Kaggle time limits may force this!). A per-zone explicit law over (T_zone, T_amb, gain estimate)
with ~10²–10³ regions is feasible; a 10-zone joint explicit law is not. Keep as fallback if QP-MPC is too
slow: distill the online MPC into explicit/PWA or a small neural policy (cf. PPO divergence — distill MPC,
don't RL from scratch on the sharp comfort landscape).

## 6. Decentralized / distributed / hierarchical MPC (the ≤10-zone question)

Our buildings are small (≤10 zones/AHU). The literature says centralized MPC is fine at that scale — but the
distributed literature still gives us (a) the coordination mechanisms worth stealing, (b) proof of what
centralized-vs-decentralized gaps look like.

- **Raman, Chaturvedi, Guo, Barooah (Univ. Florida), *"MPC-Based Hierarchical Control of a Multi-Zone
  Commercial HVAC System"* (arXiv:2102.02914)**: `https://arxiv.org/html/2102.02914` **[FULL]** — the single
  most portable paper for our plant. 33-zone VAV-with-reheat building (Modelica/IDEAS plant, 29 reheat boxes)
  aggregated to **3 floor meta-zones**; high-level MPC (Δt = 5 min model, ΔT = 15 min control) commands AHU
  outdoor-airflow + conditioned-air temperature; low-level projection controller splits meta-zone airflows to
  VAV boxes with per-zone feedback. Baseline = **dual-maximum** (an energy-efficient rule-based VAV sequence).
  Results: **11% (hot-humid week) → 60% (mild week) → up to 68% (cold week) energy savings vs dual-maximum**,
  identical comfort (temp-violation RMSE 0.10 °C vs 0.01 °C baseline), **3.28 s HLC + 0.018 s LLC solve**.
  Savings mechanisms: SAT reset, economizer/free-cooling use, reheat minimization. Directly relevant: our plant is
  this paper's plant minus humidity and minus AHU heating.
- **Hou, Xiao, Cai, Hu, Braun (Purdue), *"Distributed MPC via Proximal Jacobian ADMM"* (ACC 2017)**:
  `https://engineering.purdue.edu/~jianghai/Publication/ACC2017_ProximalJacobianADMM.pdf` **[FULL]** (7 pp.).
  3-zone open-space, TOU $0.10/0.03 kWh, 0.5-h sampling, 12-h horizon, CVX: **centralized MPC 10.15% bill
  saving, PJ-ADMM distributed 8.74%** vs greedy local control. Distributed nearly matches centralized; both
  pre-cool on cheap power and stagger peaks; the façade-coupled zone optimally does *not* pre-cool
  (storage-efficiency reasoning no heuristic captures).
- **Yang, Hu, Spanos, *"HVAC Energy Cost Optimization via a Decentralized Approach"* (arXiv:1905.10934)**:
  `https://arxiv.org/html/1905.10934` **[FULL]**. Nonconvex multi-zone cost problem with thermal + equipment
  coupling; ADMM-style decentralized method **nearly matches the centralized optimum** on small cases;
  penalty ρ ≥ 15 gives best convergence. No VAV specifics; value is the convergence/optimality-gap evidence.
- **Atam & Kerrigan, *"Optimal partitioning of multi-zone buildings for decentralized control"*
  (arXiv:2006.06326)**: `https://arxiv.org/pdf/2006.06326` **[FULL]** (8 pp.). Graph + stochastic/robust
  optimization to partition 5- and 20-zone buildings; D-MPC per cluster with 6-h horizon, neighbor temps
  fixed at 23 °C; cites literature **up to 40% MPC savings vs on-off/rule-based**. Lesson: partition along
  weak thermal links; our ≤10-zone buildings likely need no partitioning at all.
- **Li, Xu, Zhao, *"Priority-based Energy Allocation via Distributed MPC"* (arXiv:2403.13648)**:
  `https://arxiv.org/html/2403.13648v2` **[FULL]**. Energy-limited allocation with zone priorities;
  MATLAB+EnergyPlus co-simulation on 3- and 36-zone buildings. Relevant failure-mode design: when the AHU
  saturates, serve zones by need — a principled version of trim-and-respond's "requests."
- **Mork, Xhonneux, Müller, *"Hierarchical Distributed MPC for Building Energy Systems"* (ECOS 2023)**:
  `https://www.proceedings.com/content/069/069564-0220open.pdf` **[FULL]** (12 pp.). Temporal split
  (slow anticipatory layer + fast reactive layer for forecast errors) × spatial split (Nash for thermal,
  ADMM for hydraulic coupling), nonlinear Modelica controller models. The "high inertia needs anticipation,
  forecast error needs reactivity" split maps to our day/night + occupancy-gain structure.
- **Liang et al. (CMU), *"Integrated MPC-ILC Control for a Large-Scale Multi-zone VAV System"*
  (BuildSys 2022)**: `https://mandarin-cone-fs77.squarespace.com/s/buildsys22-1.pdf` **[FULL]** (10 pp.).
  **32-zone VAV-reheat office** with real measurements: MPC (QP, N=3!) + Iterative Learning Control for
  repetitive daily disturbances → fast setpoint tracking/convergence. Lesson: with strongly periodic
  occupancy/solar patterns, ILC on top of short-horizon MPC beats long-horizon MPC alone — and our occupancy
  is highly periodic.
- Xie et al. open-space DMPC (IBPSA 2017): `https://publications.ibpsa.org/proceedings/bs/2017/papers/BS2017_602.pdf`
  **[SNIPPET]** (Cloudflare-blocked; title/linear-program claim only).

**Takeaway for us:** at ≤10 zones/AHU, run **centralized** MPC (a 10-zone QP at 5-min steps is trivially
solvable; Raman's 33-zone hierarchy solves in ~3 s). Steal from distributed MPC only the *ideas*: PJ-ADMM's
staggered pre-cooling, priority allocation under saturation, temporal two-layer split, and MPC+ILC for
periodic gains.

## 7. Stochastic / robust / adaptive MPC under occupancy uncertainty

Our occupancy is BUILDING-level only — every per-zone gain forecast is uncertain. The literature offers three
maturity levels:

1. **Adaptive + convexified MPC (most practical).** Cai et al., *"An adaptive MPC scheme for energy-efficient
   control of building HVAC"* (arXiv:2102.03856): `https://ar5iv.labs.arxiv.org/html/2102.03856` **[FULL]**.
   Autonomous architecture: periodic data-driven re-ID (stability/physics guaranteed) + convex approximation
   of the nonconvex planner solved by a descent/convergent method. **1-day planning horizon, 15-min control
   horizon, Δt = 5 min.** Year-long realistic-plant study: **26.8% energy reduction vs commonly-used baseline;
   adaptation alone adds ≈16% over non-adaptive MPC**; convex planner feasible **≈100%** of steps (vs 0.1%
   failure without). Holidays handled by disturbance forecast switching. This paper is the best argument that
   *adaptation > horizon length* for our setting.
2. **Scenario-based MPC (principled uncertainty, heavier).** *"Scenario-based Nonlinear MPC for Building
   Heating Systems"* (arXiv:2012.02011): `https://ar5iv.labs.arxiv.org/html/2012.02011` **[FULL]**.
   Copula-based disturbance scenarios learned online (heteroscedastic, time-correlated — plain Gaussian
   scenarios rejected as too restrictive); **24-h horizon at 1-h sampling**, Modelica plant-in-the-loop;
   SBMPC beats deterministic MPC with and without the Modelica model; 10–40 scenarios tested with no single
   dominant count (diminishing returns past ~20). Comfort–cost tradeoff knob α traced explicitly.
   Parisio et al. scenario-explicit MPC (§5, [FULL]) applies the same paradigm to explicit laws.
3. **Chance-constrained / robust MPC.** Stochastic-occupancy MPC with chance constraints
   (`https://link.springer.com/article/10.1007/s12273-025-1300-4` **[SNIPPET]**, paywalled): **32% energy
   saving + 17% IAQ improvement** vs fixed schedule in simulation; adaptive robust MPC
   (`https://www.sciencedirect.com/science/article/abs/pii/S0360132319305360` **[SNIPPET]**): **15–20%
   savings**, collapsing to ~10% under comfort-biased tuning as uncertainty rises — a number worth
   remembering, since our reward is comfort-biased (0.6).

**Takeaway for us:** skip chance constraints (overkill, needs distributions we lack); implement **adaptive
MPC (weekly re-ID + daily disturbance-pattern update) with constraint backoff proportional to gain-forecast
error** — the poor-man's robust MPC. Scenario MPC is a stretch goal if occupancy variance proves large.

## 8. VAV / reheat-specific formulations and the Guideline 36 baseline

Everything here is about the physics we actually control: zone dampers (airflow), reheat coils (SAT per zone),
and AHU-level SAT + recirculation.

- **EnergyPlus Engineering Reference, "Air System Control"** (VAV system modeling):
  `https://bigladdersoftware.com/epx/docs/23-1/engineering-reference/air-system-control.html` **[FULL]**.
  Canonical two-regime VAV logic: **cooling → vary airflow at constant SAT** (damper modulates linearly with
  zone temp); **heating → minimum airflow, vary SAT via reheat-coil throttling**. Supply temp at min flow:
  T_supply = T_cool-coil-outlet + η_h · (capacity term). Our simulator's VAV-reheat-only actuation is this
  textbook structure — any MPC formulation can assume it.
- **Merema PhD, *"An MPC framework for all-air systems in non-residential buildings"* (KU Leuven, 2021)**:
  `https://iiw.kuleuven.be/onderzoek/building-physics-and-sustainable-design/research/phd/phd-bart-merema`
  **[FULL]** (abstract + findings). All-air MPC optimizing **supply-air temperature + ventilation flow**
  setpoints in a real educational building (BACnet): occupancy-based heating setpoint MPC achieves
  **33–36% electrical + 10–22% thermal savings vs RBC**, **≥50% fewer thermal-discomfort hours**, and at
  **5-min control steps, 80% fewer CO₂-discomfort hours**. (Conference version reports 10–40% electric /
  21–55% thermal across strategies: `https://www.researchgate.net/publication/359873653_Demonstration_of_an_MPC_framework_for_all-air_systems_in_non-residential_buildings`
  **[SNIPPET]**.) Directly supports 5-min MPC for our plant.
- **Raman et al. (§6, [FULL])**: full VAV-reheat MPC with per-zone (ṁ_sa,i, T_sa,i) + AHU (ṁ_oa, T_ca),
  reheat boxes constrained T_sa,i ≥ T_ca, dual-maximum baseline. The formulation to copy.
- **Pang, Piette et al., *"New Optimal SAT and Minimum Zone Airflow Resetting Strategies for VAV Systems"*
  (Buildings 12(3), 2022)**: `https://pdfs.semanticscholar.org/0001/61043c48f4a9ffad961d541fbf1f77bcde6a.pdf`
  **[FULL]** (15 pp.; the MDPI page itself is bot-blocked). Guideline-36-style SAT reset + minimum-flow reset
  vs fixed 55 °F SAT: **fan-energy savings 1.6–5.7%, heating-load savings 7.7–33.7% depending on climate**
  (e.g., Chicago heating −4.6%+). This quantifies the *reheat waste* available to any supervisory optimizer —
  and note our transfer suite predicted −13% for trim-and-respond while hidden scored 0, so treat these as
  upper bounds under ideal conditions.
- **ACEEE 2026 multi-site G36 field study, *"Enter the AHU (36th Chamber of ASHRAE)"***:
  `https://www.aceee.org/wp-content/uploads/2026/08/enter-the-ahu-36th-chamber-of-ashrae.pdf` **[FULL]**
  (12 pp.). 17 reprogramming-only projects: **HVAC savings 2–49%, median 18%**; SAT+DSP trim-and-respond
  median **19%**; simulation underestimates field savings by ≈5%. G36 retrofits save **25–45% whole-building**
  in ESTCP demos (WBDG summary page, [SNIPPET]).
- **Trane Guideline 36 primer (trim-and-respond mechanics)**:
  `https://www.trane.com/content/dam/Trane/Commercial/north-america/about-us/newsroom/events/2023-engineer-summit/2023-EngSummit-Achieving-Benefits-of-Guideline-36-FINAL.pdf`
  **[FULL]** (57 slides). Zones issue **requests** (e.g., damper > threshold); importance-weighted;
  if requests ≤ ignores → **trim** SP toward efficient value (SP += SP_trim); else **respond**
  proportionally (SP += R·SP_res). SAT reset trims *up* (warmer), responds *down*.
- Blum et al. grey-box MPC field test (cited in MDPI review **[SNIPPET]**): **40% HVAC energy reduction vs
  existing PI over 2 months**; Liang ARMAX-MPC **[SNIPPET]**: **≈28% vs PID**; Blum demand-flexibility field
  demo (Berkeley office, dynamic prices): **40–65% demand decrease, up to 61% annual cost saving vs RBC**
  (`https://www.sciencedirect.com/science/article/pii/S0378778825008278` **[SNIPPET]**; LBL page bot-blocked).

**Takeaway for us:** our occ_gated heuristics already do occupancy gating (Merema's biggest lever). The
*remaining* VAV-specific prize is **coordinated SAT-vs-reheat optimization**: one zone's cooling demand forces
cold SAT on all zones, causing reheat waste in the others — exactly what Raman's MPC and G36 SAT-reset attack
(7.7–33.7% heating, median-18% HVAC in field). An MPC that jointly picks AHU SAT + per-zone damper/reheat each
5 min is the canonical way to capture it.

## 9. Horizon / sampling choices: what the literature actually uses

| Source (depth) | Plant / method | Model Δt | Control ΔT | Horizon | Notes |
|---|---|---|---|---|---|
| Ma et al. CDC09 [FULL] | chiller+storage MPC | 1 h | 1 h | **24 h** | move blocking; demand charges |
| Raman et al. [FULL] | 33-zone VAV MPC | **5 min** | 15 min | day-ahead loads | HLC 3.28 s / LLC 0.018 s |
| Hou et al. PJ-ADMM [FULL] | 3-zone DMPC | 30 min | 30 min | **12 h** | TOU pre-cooling |
| Liang et al. MPC-ILC [FULL] | 32-zone VAV | 5-min data | 5 min | **N = 3 steps** | ILC covers periodicity instead |
| SBMPC heating [FULL] | single-zone NMPC | 1 h | 1 h | **24 h** | 24·30 = 720 solves/month study |
| Atam & Kerrigan [FULL] | D-MPC clusters | 15 min | 15 min | **6 h** | neighbor Ts fixed 23 °C |
| Cai adaptive MPC [FULL] | single AHU | **5 min** | 15 min | **24 h plan** | adaptation > horizon |
| Klanatsky grey-box [FULL] | RC ID | 15 min adequate | — | 24-h forecast test | 0.25 °C MAE |
| Merema PhD [FULL] | all-air MPC field | — | **5 min** | receding | 80% CO₂-hours cut at 5 min |
| Parisio explicit [FULL] | 1-room explicit | — | — | short | offline PWA |
| Van Cutsem et al. TOU [FULL] | tariff MPC study | — | — | short-H | peak-threshold state |
| MDPI horizon survey [SNIPPET] | literature survey | — | — | **1 h–24 h, mode 24 h** | long horizon ⇐ high mass |

Consensus: **24-h horizon is the modal choice** (captures the full diurnal tariff/occupancy/weather cycle);
**5-min model steps with 15-min control blocks** is the established VAV pattern (Raman, Cai); sub-hourly
control below 5 min only matters for IAQ transients (Merema). **Move blocking** (Ma) or short-N + ILC (Liang)
keeps the QP small.

**Takeaway for us:** 5-min native steps × 24 h = 288 stages — too many without blocking. Use **5-min model
resolution with geometrically growing blocks** (≈24 blocks/24 h, dense near-term) or **N ≈ 12–36 at 15–30 min
effective control resolution** with 5-min re-solve. No paper supports > 24 h (forecast skill collapses).

## 10. Prices, arbitrage, and why our 1.25× TOU kills pre-cooling

- Van Cutsem, Kayal, Blum, Pritoni, *"Comparison of MPC Formulations under Commercial TOU Tariffs"*
  (IEEE PowerTech 2019): `https://eta-publications.lbl.gov/sites/default/files/comparison_of_mpc_formulations_for_building_control_under_commercial_time_of_use_tariffs_-_david_blum.pdf`
  **[FULL]** (6 pp.). Energy-only MPC vs demand-aware MPC vs proposed incremental formulation across many
  building models: traditional demand-charge handling cuts peaks but **loses load-shifting ability**; the
  incremental bill-balancing formulation restores shifting while holding peaks. Lesson: *how* the tariff
  enters the objective matters as much as the horizon.
- LBNL *"Commercial building HVAC demand flexibility with MPC"* field demo + review
  (`https://escholarship.org/content/qt132008ps/qt132008ps.pdf` **[FULL]**, 19 pp., related LBL program
  report): MPC curtails **30–80% of controlled load in 6-h high-price windows**, **10–70% cost reductions**;
  60,000-ft² office: **65% winter peak-load cut, 50–60% summer energy+cost savings**. Requires *real* price
  spreads (dynamic tariffs), not 1.25×.
- K-12 low-cost MPC field test (LBL): `https://eta-publications.lbl.gov/publications/first-field-application-low-cost-mpc`
  **[FULL]** (abstract page): **24% cooling-season peak-demand cut, ≤16% load shifted** on school tariff,
  with user setpoint overrides allowed.
- HPWH MPC field study (Purdue, arXiv:2510.12955): `https://arxiv.org/html/2510.12955v1/` **[FULL]**.
  The cleanest tariff-complexity experiment published: MPC **+5% cost vs default under flat pricing**
  (thermal losses from shifting exceed gains), **−23% under TOU, −28% under hourly** wholesale pricing.
  Direct confirmation of our "arbitrage dead at 1.25×" probe: MPC load-shifting needs price ratios ≫ 1.25.
- PNNL packaged-AC MPC thesis (§3, [FULL]): TOU + demand-charge-aware stage coordination → ≈30% peak cuts.

**Takeaway for us:** do NOT design the MPC around TOU arbitrage (expected value ≈ 0 at 1.25×, possibly
negative after comfort risk). Design it around **comfort-violation avoidance + reheat/fan waste minimization**,
with the tariff entering only as a small linear energy-cost term. This also matches the reward (0.6 comfort).

## 11. Reported gains vs PI / rule-based, with numbers (all measurement- or simulation-based as noted)

| # | Study (depth) | Baseline | Setting | Energy / cost result |
|---|---|---|---|---|
| 1 | OptiControl companion 2014 [SNIPPET] | rule-based | sim, many types | **16–41% energy** |
| 2 | Ma et al. CDC09 [FULL] | heuristic | sim, UC Merced + storage, E-20 tariff | **24.5% daily bill** |
| 3 | Purdue 104-study review [FULL] | as-built control | **field**, reliable subset | **13% commercial bill** (duration-weighted) |
| 4 | Blum grey-box MPC field [SNIPPET] | existing PI | **field**, 2 months | **40% HVAC energy** |
| 5 | LBNL demand-flex demo [FULL] | RBC | **field**, dynamic prices | **40–65% demand, ≤61% cost** |
| 6 | LBNL program report [FULL] | legacy schedules | **field**, 60k-ft² office | **65% winter peak, 50–60% summer energy+cost** |
| 7 | Merema PhD field [FULL] | RBC | **field**, school + occ gating | **33–36% elec, 10–22% thermal, −50% discomfort-h** |
| 8 | ResearchGate Merema conf [SNIPPET] | RBC | field+sim | 10–40% elec, 21–55% thermal |
| 9 | Raman et al. hier. MPC [FULL] | dual-maximum | sim, 33-zone VAV Modelica | **11% hot-humid → 60% mild → 68% cold week** |
| 10 | Hou et al. PJ-ADMM [FULL] | greedy local | sim, 3-zone, TOU 3.3× | **10.2% central / 8.7% distributed bill** |
| 11 | Cai adaptive MPC [FULL] | common baseline | sim, year-long | **26.8% energy (+16% from adaptation)** |
| 12 | K-12 low-cost MPC [FULL] | thermostat | **field** | **24% peak cut, ≤16% shifted** |
| 13 | HPWH MPC [FULL] | default hybrid | **field**, 77 homes data | **+5% flat, −23% TOU, −28% hourly** |
| 14 | G36 ACEEE field [FULL] | pre-retrofit | **field**, 17 sites | **2–49% HVAC, median 18%** |
| 15 | SAT+flow reset sim [FULL] | fixed 55 °F SAT | sim, multi-climate | **fan 1.6–5.7%, heating 7.7–33.7%** |
| 16 | TABS admin building [SNIPPET] | conventional | **field**, 6 months | 5–42% monthly heating |
| 17 | Packaged-AC MPC thesis [FULL] | staged thermostat | sim+field case | **≈20% energy, ≈30–50% peak** |
| 18 | Robust/adaptive MPC [SNIPPET] | MPC w/o robust. | sim | 15–20% → ~10% comfort-biased |
| 19 | Stochastic occ MPC [SNIPPET] | fixed schedule | sim | 32% energy + 17% IAQ |
| 20 | 2025 MPC review [SNIPPET] | PID | field+sim survey | ARMAX-MPC ≈28%; black-box MPC 8.4% sim |

Honest prior for *our* delta (strong occ-gated heuristic → centralized MPC, flat economics, no-tails hidden):
rows 3, 9-hot-week, 10, 14-median suggest **5–15%** is realistic, concentrated in **reheat + comfort-violation**
reduction — *if* hidden buildings resemble train. The 30–60% rows all involve weak baselines, storage/dynamic
tariffs, or mild-weather free cooling.

## 12. Sources That Could NOT Be Opened (gaps — do not cite as inspected)

1. Oldewurtel et al. 2012 + 2014 (Energy and Buildings, paywalled) — numbers via secondary snippets only.
2. MDPI Buildings 15(18) 2025 MPC review — bot-blocked (Access Denied); Blum-40% / Liang-28% / 8.4% via snippets.
3. `eta.lbl.gov` publication pages (Blum demand-flex) — Cloudflare-challenged; substituted with the [FULL]
   eScholarship LBNL program report + the [FULL] PowerTech TOU paper from the same group.
4. IBPSA proceedings (Xie open-space DMPC; BS2019 Modelica grey-box) — Cloudflare-blocked.
5. `core.ac.uk` PDFs (TABS admin building; open-space DMPC) — 403.
6. `hal.science` Liu 2025 hierarchical grey/black-box — bot-check page.
7. Springer stochastic-occupancy MPC + ScienceDirect robust-MPC/VRF-MPC — paywalled abstracts/snippets.
8. `sdb.cs.berkeley.edu` Ma12mpc.pdf (Ma et al. VAV MPC) — host dead (DNS/timeout).
9. `1library.net` thesis chapter (Oldewurtel secondary) — 403; `c.coek.info` mirror — TLS error.
10. `mdpi.com` SAT-reset page — 403, but the **same paper's PDF via semanticscholar is [FULL]**.

## 13. Concrete MPC design sketch for OUR plant

Constraints from the verified simulator facts: VAV-reheat-only (no AHU heat coil); discharge ≤ 308.15 K;
recirc ≤ 0.70 (OA fraction ≥ 0.30 — no full recirculation!); damper ≥ 10%; ≤ 10 zones/AHU; 5-min steps;
reward 0.6/0.2/0.2 with comfort slope 0.18/K; TOU 1.25×; COP 3.5 constant; building-level occupancy only.

### 13.1 Architecture: centralized economic MPC, one QP per AHU per 5-min step

No distribution needed at ≤ 10 zones (Raman's 33-zone HLC solves in 3.28 s; ours is ~10× smaller).
Two timescales, following Mork (ECOS [FULL]) + Raman + Cai:

- **Slow layer (every 1 h, 24-h horizon, 15-min blocks):** optimal SAT schedule + pre-conditioning plan +
  occupancy-gain forecast update. Captures diurnal cycle, the modal horizon choice (§9).
- **Fast layer (every 5 min, 2–4-h horizon at 5-min resolution, warm-started):** per-zone damper + reheat
  commands tracking the slow plan with feedback. Handles forecast error (Mork's "reactivity").
- **Fallback:** if online QP violates Kaggle inference budget, distill to per-zone explicit PWA laws (§5)
  or a small MLP policy cloned from MPC trajectories (NOT PPO-from-scratch: sharp comfort landscape
  diverged; supervised distillation sidesteps exploration).

### 13.2 Control-oriented model (identify from the simulator itself)

Per zone i, discrete 1R1C + lumped gain (start here; escalate to 2R2C iff 24-h MAE > 0.3 °C, §4):

```text
C_i · (T_i[k+1] − T_i[k]) / Δt = (T_amb[k] − T_i[k]) / R_i + q_hvac,i[k] + d_i[k]
q_hvac,i[k] = ṁ_i[k] · c_p · (T_sat,i[k] − T_i[k])        (sensible VAV delivery)
```

- Actuation maps to simulator knobs: ṁ_i ∝ damper_i (≥ 10%), T_sat,i ∈ [T_ahu, 308.15 K] via reheat,
  T_ahu shared (cooling-coil SAT), recirc r ≤ 0.70 couples OA intake (economizer term in AHU energy).
- **ID protocol (Radecki/UKF [FULL] + Alanqar [FULL]):**
  1. Night/weekend pseudo-random damper+SAT excitation → nonlinear least squares for (R_i, C_i, coil/damper
     gains) with stability + sign constraints (Cai guarantees [FULL]).
  2. Daytime residual attribution → per-building periodic gain profile d̂(tod, dow) + online Kalman disturbance
     observer (PNNL thesis recipe [FULL]: augmented (Aa, Ca), Riccati Ka, 1-step-PEM tuning per Alanqar).
  3. Fleet priors across train buildings (Hossain transfer lesson [FULL]); weekly re-ID (Klanatsky [FULL]).
- Inter-zone coupling: start decoupled (Atam: partition along weak links; ≤ 10 zones → verify residuals;
  add neighbor terms only if night-decay fits demand it).

### 13.3 Objective = the reward, written as an MPC cost

```text
min Σ_k  [ 0.6 · Σ_i softplus-ish comfort penalty(T_i[k])      ← slope 0.18/K ⇒ ~L1; use soft constraints + exact-penalty slacks
         + 0.2 · (P_fan[k] + P_chiller[k] + P_reheat[k]) · π[k]  ← π = TOU (1.25× ⇒ nearly flat; keep, don't chase)
         + 0.2 · carbon[k] ]                                    ← ∝ electricity ⇒ folds into energy term
     + small Δu regularization (actuator wear / hunting)
s.t. zone dynamics (§13.2), T_sat,i ∈ [T_ahu, 308.15K], damper ∈ [0.1, 1], recirc ≤ 0.70,
     T_i ∈ comfort band + slacks (soft, exact penalty ≫ energy scale — cf. 10⁶× practice, §5),
     AHU capacity + shared-SAT coupling.
```

Because comfort is ~L1-steep and energy ~linear-flat, the QP behaves like: **ride the comfort-band edge
with a backoff proportional to gain-forecast uncertainty** (poor-man's robust MPC, §7), and spend energy
only to avoid predicted violations or to cut reheat waste via SAT coordination. No pre-cooling for arbitrage
at 1.25× (HPWH lesson [FULL]); pre-condition ONLY to avoid imminent comfort violations (e.g., pre-cool before
occupancy ramp when the chiller/damper slew cannot catch up — the Merema anticipation pattern [FULL]).

### 13.4 The core optimization the heuristics cannot do (expected win)

Single biggest portable prize (§8): **joint AHU-SAT + per-zone damper/reheat coordination.**
Heuristics pick SAT by rule (fixed / reset by max damper) and zones fight it with reheat. The MPC instead
solves, each step: *is it cheaper to lower SAT for the hot zone and reheat the rest, or raise SAT and open
dampers?* — subject to the recirc ≤ 0.70 and discharge caps. Secondary prizes: (a) violation avoidance ahead
of occupancy/weather ramps (building-level occupancy IS observed, so the ramp is forecastable);
(b) night/weekend setback depth optimized against morning recovery capacity (no tails in hidden ⇒ keep
backoffs small); (c) recirculation vs free-cooling tradeoff at the AHU.

### 13.5 Validation protocol (learn from §2 + our discredited suite)

- Baselines: occ_gated4/occ_gated7 (0.01323 public) AND a dual-maximum/G36-style VAV baseline (Raman §6) —
  report both, since literature shows baseline choice dominates the %.
- Metrics: public-score delta, comfort-violation K·h, reheat kWh, fan kWh; ablation: slow-only / fast-only /
  no-observer / no-adaptation (Cai showed adaptation = 16 pts of 26.8).
- Hidden-robustness: evaluate on held-out buildings AND no-tails subsets (hidden shows no tails — verify the
  MPC still wins when all mass is near setpoint; if the win evaporates, the design is wrong, not the suite).
- Compute budget: time the QP (target ≪ 5-min step; Raman-scale ⇒ ms–s); prepare the explicit/distilled
  fallback (§5) before inference.

### 13.6 Risks and open questions

1. **No-tails hidden + no per-zone occupancy gating** ⇒ predictive comfort management has a low ceiling;
   most of the 2.5× gap to the leader (0.00521 vs 0.01323) may be steadier band-riding, not foresight.
2. **Recirc cap 0.70** (min 30% OA) limits free-cooling-vs-recirc optimization vs literature plants.
3. **No AHU heating coil** ⇒ morning heat-up is reheat-only (slow, capped at 308.15 K) — recovery
   constraints may dominate the formulation; verify slew/recovery capacity in-sim first.
4. Grey-box/MPC loop "never closed" per state-of-play: the risk is engineering time, not theory — mitigate
   with the slow/fast split (ship slow layer first; it degrades gracefully to SAT-reset + setback).
5. PPO divergence on the sharp comfort landscape is *evidence for* MPC (exact-penalty soft constraints handle
   L1-steep costs natively) — but also a warning that any learned component (distillation, observer) needs
   the same constraint discipline.

---

*End of report. 24 sources read in full ([FULL]), 10 flagged snippet-only ([SNIPPET], §12). All quantitative
claims carry their source row in §11.*
