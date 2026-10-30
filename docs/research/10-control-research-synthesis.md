# HVAC control research synthesis

Date: 2026-10-28. Scope: one policy for unseen commercial buildings in this
repository, with a separate account of real HVAC applicability.

The next useful research result is a reproducible comparison of the current
economizer, heating-deadband and coil-aware family, followed by a causal
test of whether a model can coordinate air and water decisions. Repeating
the old top-ten heuristics without accounting for their measured failures
would spend compute on questions the repository has already investigated.
The evidence supports comparing several control families; it does not
identify the hidden leader's method or establish a universal winner.

## 1. What is established

The [repository analysis](06-repository-analysis.md) defines the actual
information and actuator contract. The [domain reference](08-hvac-domain-reference.md)
explains the physical mechanisms and omitted field constraints. The
[source audit](09-evidence-audit.md) checks high-impact literature claims
against primary bodies. The [collection plan](07-evidence-collection-plan.md)
specifies the missing records and evaluations.

| Finding | Evidence and scope | Consequence |
|---|---|---|
| Comfort failures have a steep, unbounded tail | Repository reward contract and AGENTS facts | Inspect worst episodes and time-resolved recovery alongside mean loss |
| Historical occ4 train loss is predominantly energy/carbon | Existing train-30 CSVs, recomputed in report 06 | Investigate conditioning efficiency; do not equate this share with removable waste |
| Real VAV rules do not automatically improve this simulator | Full-episode dual-max and G36 SAT results in the Oct-28 journal | Transfer the mechanism and its prerequisites, then evaluate the implemented variant |
| The current useful heuristic evidence is econ/dbh/heatopt | Selected full episodes, journal and surviving artifacts | Compare these with a versioned parent; quantify the deadband's night-tail risk |
| A model/tool existing in the tree is not evidence of successful control | Diagnostic scripts explicitly limit their claims | Require predictive and closed-loop evidence before promoting a planner |
| RL studies produce setting-dependent outcomes | Primary-study audit in report 09 | Match plant, baseline, information, metric and compute before using a result |
| Many real HVAC obligations are outside the score | Humidity, IAQ, faults and standards mapping in report 08 | Keep competition gains and field-readiness claims separate |

The historical public scores 0.01323 and 0.00521 imply a reduction of about
60.62% to reach that recorded leader value. They are journal evidence, not a
live leaderboard check. Comparing 0.00963 on training scenarios with 0.00521
on hidden scenarios cannot determine the attainable improvement on either
population. No reliable percentage forecast for MPC or RL follows from it.

## 2. Control families and their information requirements

| Family | Useful mechanism | Prerequisites | Main failure to investigate | Position in this project |
|---|---|---|---|---|
| Rules and PI with supervisory reset | Reliable local feedback, mode scheduling, plant reset | Consistent units, zone/AHU mapping, well-tuned timing and integrators | Interacting loops, hunting, delayed recovery, simultaneous conditioning | Essential reference and fallback |
| Analytic equipment allocation | Match flow and reheat to required duty within known caps | Credible flow/temperature relation and plant state | Commanded flow mistaken for delivered flow; unobserved capacity | Current heatopt family provides a concrete starting point |
| Deterministic greybox MPC | Predict temperature and coordinate coupled actions | Causal state estimate, calibrated dynamics, feasible forecasts and solver budget | Model bias, latent wall state, poor alternative-action predictions | Worth a gated research comparison |
| Robust or scenario MPC | Account for forecast/model uncertainty | Valid uncertainty sets or calibrated scenario distribution | Over-conservatism, optimistic coverage under transfer | Add only after nominal error is understood |
| Shooting, CEM or MPPI planning | Optimize nonlinear coupled decisions without exact derivatives | Fast predictive model and meaningful action parameterization | Compute cost, local search, terminal/recovery errors | Oracle probes motivate testing, not promotion |
| Model-free RL | Learn feedback across diverse scenarios | Large representative transition set, training stability and transfer evaluation | Reward exploitation, narrow action support, out-of-distribution actions | A controlled comparator; no universal ban on PPO or guarantee for SAC |
| Model-based RL | Improve dynamics and policy/planning from data | Identifiable useful dynamics plus epistemic error handling | Model exploitation and compounding rollout error | Natural bridge to the model/observer work |
| Offline RL and imitation | Reuse logged actions or strong-controller trajectories | Aligned transitions and sufficient behavioral coverage | Actions unsupported by logs; copying a teacher's failures | Useful only after the run/transition contract is recovered |
| Shared zone/AHU or graph policy | Handle variable layouts and equipment counts | Semantic topology, masks, stable normalized features | Treating zone order as identity, losing shared-plant coupling | Research hypothesis for transfer, not a measured advantage here |
| Residual or distilled policy | Improve a reference or compress a planner | Reliable reference, action limits, diverse teacher coverage | Residuals break stability; student misses rare events | Deployment option after a controller earns its advantage |

The graph-policy and robust-planning entries are proposed research designs.
They are not claims that these architectures have already won this task.
Temperature history and topology can help infer dynamics, but no architecture
can reconstruct unobserved occupancy or equipment limits without assumptions
and informative data.

## 3. A physically compatible predictive formulation

Treat the joint air/water decision as a constrained dynamic control problem.
The observable state includes zone temperatures, plant measurements,
schedules and aggregate occupancy. The estimated state may include wall
temperatures, disturbance loads and uncertain equipment capacity. Keep these
two categories explicit in any implementation.

An illustrative model-based objective over H future five-minute steps is:

```text
minimize sum(k=0..H-1) [0.6 * comfort_loss(k)
                       + 0.2 * normalized_energy_cost(k)
                       + 0.2 * normalized_carbon(k)]
         + terminal_recovery_cost

subject to estimated thermal/equipment dynamics,
           actual action bounds and equipment modes,
           uncertainty-aware operating constraints.
```

Here `comfort_loss = -normalized_productivity_regret`. The exact reward uses
zone occupancy that is not directly observed by the participant. A predictive
controller must declare its occupancy allocation proxy or uncertainty bounds;
future true zone occupancy is an oracle input, not an available forecast.
Use the repository's actual normalized components rather than substituting
degrees, kWh and dollars directly into these weights. Extra smoothness,
cycling or terminal penalties are design choices whose effects need
ablation. The real score still decides whether the resulting trajectory is
useful. A terminal estimate should discourage apparent savings obtained by
deferring recovery beyond the optimization horizon.

The air supply is shared within each AHU, while zone dampers and reheat
allocate local conditioning. The water system couples the heating terminals.
Independent per-zone optimization can miss both couplings. A possible
decomposition is a supervisory plant decision plus local zone allocations,
with explicit checks that their combined actions respect the common plant.
Compare this with a joint optimization; decomposition is a compute tradeoff,
not an assumed equivalence.

Do not insert a heat-pump COP curve, storage arbitrage, AHU heating coil or
humidity-control benefit into this simulator's optimization when the scored
plant does not contain the corresponding mechanism. Conversely, a field
controller must represent the plant and constraints actually installed.

A 6–12-step horizon is 30–60 minutes. A 2–4-hour horizon is 24–48 steps.
Longer horizons can use move blocking or coarser predictions, but evaluate
their approximation against the original five-minute system. Fixed short
horizons may miss prestart, nighttime storage and next-day recovery.

Use forecasts that are available causally. Persistence and learned periodic
patterns are deployable candidates; exact future weather/occupancy from
the simulator is an oracle input unless explicitly provided by the policy
interface. Report oracle and deployable results separately.

## 4. What the model must prove before planning

First, validate observation timing and action-to-response alignment. The
existing equipment and thermal diagnostics contain explicit delayed-state
logic; fitting an immediate response to a delayed sensor can produce a good
correlation with the wrong causal interpretation.

Second, evaluate temperature and energy predictions on complete held-out
periods, including start/stop transitions, cold saturation, hot cooling and
mixed-load operation. Compare with persistence and the present controller's
simple predictors. Report bias and horizon-dependent error, rather than
selecting a single aggregate RMSE threshold with no relation to the control
band or the intended action change.

Third, assess counterfactual usefulness. Predictions along a teacher's
trajectory do not establish accuracy after a different damper/reheat/SAT
decision. Preserve supported action ranges, use modest interventions during
research, and test whether predicted action rankings match realized ones.
For uncertain modes, the controller can revert to its measured reference.

Fourth, evaluate the complete feedback loop. A model can predict average
temperature accurately while repeatedly selecting the wrong action near a
comfort boundary. Record model rejection, optimization failure, projected
action corrections, fallback use and delayed recovery. A box projection
guarantees action limits only; it does not guarantee future comfort.

The current `AdequacyMonitor` explicitly distinguishes lack of detected
rejection from control permission. Preserve that distinction. A calibrated
prediction interval on one set of operating regimes does not automatically
retain its coverage on an unseen building or controller-induced state.

## 5. Updated experimental ordering

These stages describe the work needed after this research. They do not assert
that new policies, simulations or submissions were completed here.

1. **Recover the evidence base.** Archive code/configuration identity for
   existing full episodes; reconcile canonical and temporary results. Retain
   unknown-provenance files as exploratory material. This is the prerequisite
   for comparing any candidate with the actual current controller.
2. **Establish a current reference.** Evaluate parent occ7, econ/dbh/heatopt
   components and their combination on identical full scenarios. Keep energy
   gains and the heating-deadband night tails visible. Neither an old best
   public score nor a six-cell improvement establishes a full-suite winner.
3. **Isolate recovery and coupling.** Examine the gas/comfort timing behind
   the failed prestart and dual-max changes. Reopen a closed mechanism only
   with a changed hypothesis, such as joint water/air coordination, rather
   than repeating the same blanket restriction.
4. **Validate causal prediction.** Use the model gates above on untouched
   buildings and action regimes. Establish which modes and horizons are
   adequate and which still require the reference controller.
5. **Compare a bounded planner.** Use the same information as the deployed
   policy, account for all planning time, and run complete episodes. Compare
   observer-only, planner, and uncertainty/fallback variants.
6. **Compare learning under a declared budget.** Record transitions, gradient
   updates, wall time, training seeds and validation selection. A modest
   residual/imitation comparator may be cheaper to investigate than a large
   end-to-end learner, but neither is presumed superior.
7. **Package only a measured candidate.** Preserve fixed parameters and
   notebook identity; complete the assembly/runtime checks in report 07.
   A competition submission requires current external status/rules evidence.

Closed results remain specific to their tested implementation. For example,
the naive dual-max output map loses on its eight full-episode cells; that
does not prove every joint air/reheat optimization is unhelpful. Conversely,
a generic success in the literature does not overturn that local result.

## 6. Transfer and practical HVAC scope

Transfer changes geometry, thermal storage, envelope conductance, equipment
capacity, occupancy and weather jointly. A useful representation should
respect topology and units, distinguish applied from requested actions, and
retain uncertainty for unobserved quantities. Do not use a weather-file name
or training-building identifier as a substitute for adaptation.

Collect geometry/topology from allowed metadata and material priors from
public or training evidence; true held-out material fields are unavailable
to the participant. Update only what observations can support. Online adaptation needs an explicit budget and
evaluation convention: all adaptation steps in the 30-day episode are
scored, including startup. A free unscored identification period would define
a different task.

For field use, the main additional research topics are latent loads and
dehumidification, ventilation and outdoor pollution, actual airflow and CO2,
part-load efficiency, faults and overrides, minimum equipment cycle times,
freeze/condensation protection and commissioning. The competition's favorable
energy score is not evidence that these constraints have been satisfied.
The domain reference identifies the measurements and primary standards
needed to assess them.

## 7. Research conclusions and remaining uncertainty

The available evidence favors an engineering sequence: recover exact
experiments, strengthen the measured baseline, validate a causal model,
then compare coordination and learning. This ordering follows the gaps in
this repository; it is not a universal hierarchy of HVAC algorithms.

The most consequential unknowns are current-stack full-suite transfer,
the cause and magnitude of its night tails, alternative-action model
accuracy, robust runtime, and the current hidden score. Field relevance
additionally depends on missing humidity, ventilation and equipment evidence.
Report 07 turns these into explicit collection tasks.

This synthesis supersedes the action ranking in the
[October 27 research synthesis](2026-10-27-synthesis.md). The older reports
remain useful source and experiment records subject to the corrections and
access limits in report 09. No savings target, hidden leader method, or
field-deployment claim is inferred from unverified material.
