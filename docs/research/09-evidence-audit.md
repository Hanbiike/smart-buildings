# Evidence audit: what the HVAC literature and repository actually establish

**Audit date:** 2026-10-28. **Scope:** correct the evidentiary basis of the
2026-10-27 synthesis and RL report; reconcile them with the latest day log.
This audit creates no controller, launches no simulation, and claims no new score.
It supersedes unsupported interpretations in the earlier reports, not their raw data.

## 1. Repository evidence comes first

The starting records are [the original synthesis](2026-10-27-synthesis.md),
[the RL report](04-rl.md), [gap forensics](05-gap-forensics.md), and
[the 2026-10-28 log](../2026/10/28/README.md).
The RL report explicitly says that none of its cited paper bodies was opened.
Its recommendations therefore needed verification before becoming settled advice.
The synthesis is dated 2026-10-27; its top ten are historical hypotheses,
not ten still-untried opportunities after the work recorded on 2026-10-28.

The day log is the most recent experimental narrative available for this audit.
Its opening summary still says the dual-max screening is pending, but its later
Measured and end-of-session sections record completed results and supersede that text.
Likewise, early Open items must be read together with their later closures.
Numbers below are attributed to that log; this audit does not independently
recompute temporary `/tmp` run directories or verify their continued availability.
The evidence grades used here distinguish measurement from interpretation:

| Grade | Meaning | Appropriate use |
|---|---|---|
| LOCAL | Repository source or recorded experiment inspected | Describe the inspected implementation/result and its tested scope |
| BODY | Primary paper methods/results body inspected | Cite that study's actual baseline, outcome, and limitations |
| ABSTRACT | Primary abstract/preview only | Bibliographic lead; excluded from recommendations here |
| UNOPENED | Body unavailable or not inspected | Explicit unresolved item; no numerical support |
| INFERENCE | Reasoning from evidence, not a new observation | State assumptions and the observation that could refute it |

BODY does not mean independently reproduced, peer-reviewed, or applicable to this
competition. An accessible preprint remains a preprint. A simulator result remains
a simulator result, even when its emulator is detailed and publicly available.

## 2. Seven primary bodies checked

### A. Sinergym algorithm evaluation — arXiv:2401.05737v3

**Access:** BODY; methods §§4–5, discussion §6.1, Tables 9–10.
**Setting:** two EnergyPlus buildings, three climates, annual episodes,
15-minute control; seasonal setpoint RBC for 5ZoneAutoDXVAV and integral
setpoint correction for 2ZoneDataCenterHVAC. This is not direct joint control
of every VAV actuator and AHU in our competition.
**Verified:** under the paper's equal-weight comfort/consumption reward, RBC
beats all three tested DRL methods in the five-zone case. Mean rewards for
cool/mixed/hot climates are RBC `−0.198/−0.284/−0.416`, SAC
`−0.226/−0.344/−0.454`, PPO `−0.250/−0.367/−0.470`, and TD3
`−0.258/−0.368/−0.477`. TD3 beats RBC on reward in the data center.
**Limit:** data-center violation frequency rises, but the authors distinguish
frequency from severity; reported hot-case excursions stay below 0.7°C.
This paper has no MPC comparison. Its results establish neither a universal
SAC/TD3/PPO order nor inevitable catastrophic comfort failures.
**Disposition:** retain the five-zone RBC comparison with its setting; retract
the claim that this source shows MPC ties or beats RL.
[Primary body](https://arxiv.org/html/2401.05737v3).

### B. Gnu-RL — Frontiers 2020 extension of BuildSys 2019

**Access:** BODY; §§3–7, especially §§6.1–6.3 and Table 5.
**Method:** imitation-pretrained differentiable MPC, refined using PPO.
**Actual comparisons:** the `6.6%` simulation saving is against the
best-performing Zhang and Lam (2018) RL controller in the same calibrated
600 m² workspace. The controlled plant is radiant heating via supply-water
temperature, not our multizone VAV action vector.
The three-week field trial controls airflow in one 20 m² VAV conference room.
Its `16.7%` reduction is cooling demand against a fixed-schedule controller,
using weather/duration-normalized historical June data, not simultaneous
randomized whole-building electricity measurement. Table 5 reports
`99.4 → 82.8 kWh` cooling demand; the authors explicitly distinguish this
quantity from energy consumption.
**Limits:** future disturbances/occupancy assumptions, local linearization,
small field scale, and historical baseline normalization limit extrapolation.
**Disposition:** useful evidence for controller initialization and task-based
refinement. It does not validate a generic SAC residual recipe, prove DAgger
was used, or support banning PPO in every architecture.
[Primary body](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.562239/full).

### C. MB²C — arXiv:2302.00725v1

**Access:** BODY; §§V-A–V-C, Table II, conclusion.
**Setting:** simulated 463 m², five-zone office with central cooling and VAV
terminal reheat; 15-minute decisions. AHU control stays with EnergyPlus;
the learned controller changes zone heating/cooling setpoints.
**Actual baselines:** campus-style RBC; PPO; deterministic neural dynamics
with random-shooting model-based control.
**Verified:** average energy savings are `10.65%` against RBC and `8.23%`
against the random-shooting MBRL implementation, with comparable comfort.
Asymptotic performance is comparable to PPO; the claimed `10.52×` advantage
concerns required training data/convergence, not a corresponding score gain.
**Limits:** the result combines dynamics-model and optimizer changes.
It is not an isolated MPPI-versus-CEM test, a comparison against tuned
greybox MPC, or evidence that PPO cannot solve HVAC control.
**Disposition:** motivates testing learned dynamics plus sampled planning,
while leaving the planner, model class, and attainable local gain undecided.
[Primary body](https://arxiv.org/html/2302.00725v1).

### D. CLUE — arXiv:2407.12195v2

**Access:** BODY; §§V-D and VI, including Fig. 7 and baseline definitions.
**Setting:** five-zone office; January/July in Pittsburgh, Tucson, New York.
GP meta-kernels use reference-building data; confidence-aware MPPI can fall
back to the simulator's default controller.
**Actual baseline:** deep-ensemble MBRL with MPPI, plus default RBC and an
ablation without confidence-based control.
**Verified:** the Pittsburgh-January data-size experiment stabilizes with
seven target-building days; DE-MBRL needs 50 days to beat default and another
250 to reach peak performance. This supports a study-specific target-data
advantage, not a general `>30×` wall-clock acceleration. Meta-training uses
one year from each reference building and is additional to those seven days.
The body reports fewer violations with confidence control and slightly more
energy than variants without it; uncertainty-aware behavior is not free.
**Limits:** searched error thresholds, selected climates, reference-building
data, and fallback quality matter; no robust-invariance guarantee follows.
**Disposition:** retain uncertainty/fallback ablations; do not budget local
training or predict savings from the headline convergence ratio.
[Primary body](https://arxiv.org/html/2407.12195v2).

### E. Heat-pump SAC/PPO comparison — arXiv:2608.09453v1

**Access:** BODY; §§3–6 and Tables 1–2. The cited preprint exists.
**Setting:** single-zone BOPTEST hydronic heat pump, scalar capacity command,
15-minute steps, two 14-day periods, highly dynamic tariff. Training uses
a SINDy surrogate for five million steps; reward includes compressor wear.
**Baseline/outcomes:** built-in controller. Peak discomfort `8.382 → 0.777`
and cost `0.909 → 1.013` support the quoted `90.7%` comfort reduction and
about `11.5%` cost increase. Typical discomfort is `9.446 → 6.322` and cost
`0.413 → 0.631`: approximately `33.1%` less discomfort but `52.8%` more
cost, calculated from Table 2. The 11.5% premium is not universal.
SAC has zero counted starts; PPO cycles more than baseline in this setup.
**Limits:** the paper acknowledges an uncalibrated surrogate and permissive
`0.001` on-threshold; near-zero commands can count as continuously on.
This is not direct evidence about our fixed-efficiency VAV-reheat plant.
**Disposition:** a bounded empirical warning; not a diagnosis of our PPO run
or proof that entropy tuning guarantees smooth, comfortable control.
[Primary body](https://arxiv.org/html/2608.09453v1).

### F. Offline HVAC RL — arXiv:2408.07986v1

**Access:** BODY; §§5–6, especially §§5.3, 6.1–6.4 and Table 4.
**Setting:** MixedUse and two-zone DataCenter simulations, annual evaluation,
multiple training climates and a held-out weather location for each building.
**Actual baseline:** benchmark RBC with its night mode removed for alignment
with learned controllers; comparisons include SAC/TD3, CQL, BCQ, TD3+BC.
**Verified:** CQL generally leads the tested offline methods and is selected
for later experiments. Offline regularization helps when replay coverage is
limited. History-aware controllers improve the studied reward/comfort tradeoff.
The headline `28.5%` violation reduction corresponds to approximately
`30.9% → 2.36%`: about 28.54 percentage points, not 28.5% relative.
That DataCenter example increases power about 5%; the approximately `12.1%`
power saving is the separate MixedUse evaluation example.
**Limits:** these are not two simultaneous gains on one universal controller.
Generated datasets, simulator-based epoch selection, and baseline modification
do not establish improvement from our existing deterministic policy logs.
**Disposition:** keep CQL as a conditional candidate after a coverage audit;
neither promise improvement nor declare offline learning useless merely
because simulator access exists.
[Primary body](https://arxiv.org/html/2408.07986v1).

### G. Direct MPC/RL comparison — Arroyo et al., Frontiers 2022

**Access:** BODY; §§3.1–3.4, results/discussion, conclusion.
**Setting:** single-zone Brussels dwelling, radiant floor and air-source
heat pump, six combinations of heating period and electricity tariff.
**Baseline:** built-in PI controller tuned for comfort and energy; methods
include several MPC horizons/control periods and RL configurations.
**Verified:** none of the tested RL variants outperforms MPC in this study.
However, the shortest-horizon MPC configurations also fail to beat baseline;
the authors relate this to the building time constant. The paper identifies
limited hyperparameter coverage and the tradeoff of direct versus indirect
actuation as limitations.
**Disposition:** concrete support for including a well-specified MPC baseline,
not evidence that every MPC design beats every learned controller on every
HVAC plant. The horizon and actuation interface belong in any comparison.
[Primary body](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2022.849754/full).

## 3. Unresolved sources are excluded, not silently promoted

| Source | Access in this audit | Consequence |
|---|---|---|
| [Wang et al. 2023, Applied Thermal Engineering](https://www.sciencedirect.com/science/article/pii/S1359431123004593) | ABSTRACT/preview; direct body returned 403 | Its DDPG/MPC ranking is not used to support recommendations |
| [Energy 2023, load-shifting comparison](https://www.sciencedirect.com/science/article/pii/S036054422300467X) | ABSTRACT/preview; direct body returned 403 | A potential contrary comparison remains a reading lead, not verified numerical evidence |
| [LDLP citation in the old report](https://www.mdpi.com/2076-3417/16/18/9131) | UNOPENED; page returned 429, linked PDF inaccessible | Savings ranges and claimed occupancy-gating equation remain unverified and excluded |
| Other sources listed in old `04-rl.md` §9 | Not comprehensively re-audited here | Their previous snippet status is not upgraded by proximity to verified papers |

Failure to retrieve a body does not establish that the paper is false or absent.
Conversely, a publisher URL or plausible abstract does not verify a result table.
This audit resolves the six named RL source gaps; it does not certify every
literature claim in research tracks 01–05 or every numerical prior in the MPC report.

## 4. Correct the causal and statistical claims

**Leaderboard identity is unknown.** A score does not reveal whether the leader
uses MPC, RL, rules, an ensemble, analytic optimization, or a hybrid. The old
phrases “leader-shaped bet” and “most likely a planner” have no identified
solution disclosure or controller trace behind them. Treat them as speculation.

**Separate the public gap from train composition.** On the reported public
evaluation, `0.01323` versus `0.00521` is a real comparative performance gap.
The reduction required to match it is approximately `60.62%` of our public loss.
Comparing our training mean `0.00963` to the leader's public `0.00521` gives
approximately `45.90%`, but mixes populations and cannot exclude composition.
Neither calculation determines which mechanisms create the gap.

**Energy share is not avoidable waste.** The forensic `96.1%` energy-plus-carbon
share describes the measured `occ_gated4` training loss decomposition.
It is not a lower bound on waste or an estimate of recoverable savings.
Low observed comfort loss may be purchased by necessary energy, especially
before occupancy. Zero contemporaneous unoccupied comfort penalty does not
make preheating waste: actions change subsequent thermal state and recovery.
Recoverable waste requires a feasible counterfactual with no unacceptable
comfort deterioration on the same full trajectory.

**Equal displayed score is not action equivalence.** The day-log statement
`trim ≡ 0 on hidden` is stronger than `occ_gated4` and `occ_gated7` both
displaying `0.01323`. Rounding, aggregation, cancellation across scenarios,
and different action paths with similar reward are all possible.
Only matching hidden trajectories or action instrumentation could prove
the stronger statement. Determinism does not make a scalar score injective.

**Hidden tails remain unknown.** An intervention's aggregate tie or regression
does not identify the hidden distribution of comfort regret, weather, or loads.
“Hidden has no tails” must not become a premise for weakening comfort protection.
Likewise, current public standings quoted from the log are a dated snapshot,
not a live leaderboard verification performed by this audit.

## 5. Reconcile the old action plan with completed work

Use the [day log](../2026/10/28/README.md) for the full matrix; the following
are changes of decision state, not a duplicate top-ten ranking.

- Items 1–3: naive dual-max loses `+5.0%` on the eight full episodes;
  the 600-step gain reversed. Edge riding is closed, with gentle-centre
  around `−1.0%` banked. Blanket prestart shape/timing changes lose.
- Item 4: fair-port heating saturation control loses; the cooling port
  is a small local win, around `−0.9%` on four hot cells, not promoted.
- Items 5–6: tested start-margin/coast changes and two-stage SAT/T&R lose.
  This closes the tested implementations, not every adaptive controller.
- Item 7: occupancy proportional v3 is around `−0.2%` on eight cells;
  its coolshift combination regresses, showing gains are not additive.
- Item 9: demand-based staging is negative or nearly inert after correction.
  It is no longer an untested easy saving in the shutdown architecture.
- Later additions matter more than the old ranking: explicit economizer
  control improves several mild cells without reported comfort changes;
  heating-only deadband improves loss but grows some midnight tails.
- At the initial audit, `occ8` was submitted and `occ9` held. The closing
  reconciliation of commits `9cd060e` and `59a307b` records both submitted,
  scores pending. Local improvements must not be presented as scored wins.
- The same closing update records pseudo-DCV inert: bit-identical to occ7 on
  sg1 and loss-equal on phx6/atl2/den2. The unimplemented warmest-zone idea
  was dropped by analogy to an external notebook and an inert hot port;
  that decision does not establish global cooling-SAT optimality.

The day log's explanation that delivery restriction proves the waste is
plant-level is itself an inference. A failed output-map change shows that
that change lost under the tested PI/recovery settings; it does not uniquely
identify the optimal actuator allocation or eliminate all delivery redesigns.

## 6. Implications for the next decision

Prioritize mechanisms with matched full-episode evidence and useful failure
diagnostics; select the algorithm after specifying the actuator interface,
available forecasts, adaptation data, runtime budget, and fallback behavior.
Preserve a strong current controller as the comparison baseline. Old literature
percentages against weaker or different baselines are not local gain forecasts.

Planning still generalizes imperfectly: a learned model, objective, uncertainty
threshold, horizon, or scenario tuning can overfit. Reoptimization does not
remove model bias or confer immunity to unseen buildings.
Action clamping establishes admissible commands, not future temperature
feasibility; a genuine comfort guarantee needs dynamics/uncertainty assumptions
and a feasible recovery set. Clipping penalties changes the evaluated tradeoff
unless the original objective is retained for selection and final evaluation.

The `≤10 zones/AHU` validity rule is not a `≤10 zones/building` bound; an
entire building can have many AHUs and a much larger joint action dimension.
At five minutes per step, a 6–12-step horizon is 30–60 minutes; 2–4 hours
requires 24–48 steps. Specify temporal blocking explicitly when mixing scales.
Sample efficiency still consumes wall-clock budget even with unlimited resets;
measure actual simulator throughput before assuming millions of steps are cheap.

A defensible research result reports both full-episode competition loss and
its comfort/energy/carbon components, subgroup failures, and inference time.
Run-to-run determinism removes simulation noise, not uncertainty about unseen
buildings, climate composition, training seeds, or selection among many variants.
No paper inspected here identifies the hidden optimum or guarantees closing
the public gap. They support conditional designs and controlled comparisons.
