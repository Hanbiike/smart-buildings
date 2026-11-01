# Online RL for HVAC, Track A: Model-Free Algorithms (PPO / SAC / TD3 / Recurrent)

Date: 2026-11-01
Status: draft
Scope: Which model-free RL algorithm wins where on HVAC control, with numbers,
settings, and failure modes; transfer limits to our competition (5-minute joint
VAV+AHU control, reward weights 0.6/0.2/0.2, one frozen policy on hidden
buildings). Residual-on-occ_gated7 framing. No simulation was run for this
document; all numbers carry provenance below.

## Sources

| ID | Source | Access | Covers |
|---|---|---|---|
| S1 | Session evidence bundle, Sinergym/BOPTEST/TreeC/CityLearn claims (`.../tool-outputs/readable-text/456116843e31461cf8ec6ea90832694b30bcf1611e06161e8893c41bbe142a2a.txt`) | Inspected body (JSON claims + excerpts) | Claims 1-8, 13, 16 |
| S2 | Session evidence bundle, resolved result 4 only, SAC/TD3/PPO/MBRL claims (`.../tool-outputs/readable-text/42641ec471e2087178e19d0d394643f652d75416802c7a1ae23ac4506ccd5066.txt`, lines 19-24) | Inspected body (lines 19-24 only) | Claims 6, 9-10, 13-14, 17-18 |
| S3 | `docs/research/04-rl.md` | Inspected full body (492 lines) | Claims 9-12, 15, 19; context |
| S4 | `docs/research/09-evidence-audit.md`, sections A-C (Sinergym, Gnu-RL, MB2C) | Inspected full body; cite A-C only | Claims 1-2, 9-10; corrections |
| S5 | `tools/slot_ppo.py`, first 120 lines (our PPO setup) | Inspected full body; cite lines 1-120 | Claim 15 |
| S0 | Scout formatting standard (prior result 1: English-only, NN-kebab naming, header pattern, R1-R9) | Handoff summary | Document shape only, no numbers |

Evidence grades follow S4: BODY = primary paper body inspected by the auditor;
ABSTRACT = abstract/preview only; UNOPENED = body unavailable. S1/S2 excerpts
are quoted passages from primary bodies or reproductions, not independent
reproductions by us.

## Numbered claims

### Where each algorithm wins

1. On Sinergym `5ZoneAutoDXVAV` (the closest public analogue to our VAV
problem), the rule-based controller beats SAC, PPO, and TD3 on mean episode
reward in every climate, under the paper's equal-weight linear reward
(`w = 0.5/0.5`, 20 one-year training episodes, 15-minute steps, seasonal
setpoint RBC). Mean rewards, cool/mixed/hot: RBC `-0.198/-0.284/-0.416`,
SAC `-0.226/-0.344/-0.454`, PPO `-0.250/-0.367/-0.470`, TD3
`-0.258/-0.368/-0.477`. SAC is the closest DRL agent to RBC in all
climates, followed by PPO and TD3. (S1 excerpts from
https://arxiv.org/html/2401.05737v3 ; S4 section A, BODY, same URL, Tables
9-10.) Transfer limits: this is NOT joint control of every VAV actuator
and AHU (setpoint actions only); reward is 0.5/0.5 comfort/energy, not our
0.6/0.2/0.2; 15-minute steps, not our 5-minute steps. S4 explicitly
retracts any universal SAC/TD3/PPO order and any MPC comparison drawn
from this source (the paper has no MPC comparison).

2. On Sinergym `2ZoneDataCenterHVAC`, TD3 outperforms the RBC on mean reward
in all climates via large power savings at the cost of a higher comfort
violation rate. The measured temperature deviations stay small: they
exceed the upper comfort limit by less than 0.7 C and never breach the
lower limit. SAC, PPO, and RBC are more conservative. (S1 excerpts from
https://arxiv.org/html/2401.05737v3 ; S4 section A, BODY.) Transfer
limits: a data center (always-occupied, cooling-dominated, wide 18-27 C
band) differs structurally from our comfort-weighted commercial VAV task;
the violation-frequency-vs-severity distinction in the paper means the
headline "TD3 wins" depends on how violations are priced — under our
0.18/K comfort slope the same tradeoff could easily lose.

3. In the Sinergym testbed paper's `2ZoneDataCenter`-mixed example (40
one-year episodes, Stable-Baselines3 defaults), PPO is the most stable
learner and achieves the best, most consistent evaluation reward; SAC
performs similar to RBC/default, while TD3 generally beats both reactive
baselines. DRL saves 15-18% energy vs reactive control in several months,
concentrated in shoulder seasons. (S1 excerpts from
https://arxiv.org/html/2412.08293v1.) Transfer limits: single building,
single climate, SB3 defaults, reactive (weak) baselines — this is a
"beats weak baselines" result of the kind S3 section 1.2 warns against
extrapolating, and it directly contradicts any fixed PPO<SAC<T3 ranking
from Claim 1, proving the winner is configuration-dependent.

4. On a BOPTEST `bestest_hydronic_heat_pump` with identical MDP and a
compressor-wear-augmented reward, SAC learns continuous inverter-style
modulation with zero compressor starts/day on both peak and typical days,
cutting thermal discomfort by up to 90.7% for an 11.5% cost increase
(+11.0% source energy); PPO collapses to bang-bang control cycling MORE
than baseline (4.29 starts/day peak, 2.64 typical) while barely improving
comfort (17.9% peak, 3.8% typical). (S1 excerpts from
https://arxiv.org/pdf/2608.09453v1.pdf ; S2 resolved result 4 notes the
same study.) S4 section E (BODY, Tables 1-2) bounds this: peak discomfort
8.382 -> 0.777 and cost 0.909 -> 1.013 support the 90.7%/11.5% figures,
but typical-period discomfort 9.446 -> 6.322 with cost 0.413 -> 0.631 is
only ~33.1% less discomfort for ~52.8% more cost — the 11.5% premium is
not universal. Transfer limits: single-zone, scalar capacity command,
5M training steps on a SINDy surrogate the authors call uncalibrated, a
permissive 0.001 on-threshold (near-zero commands can count as
continuously on), 15-minute steps — not evidence about our
fixed-efficiency multi-zone VAV-reheat plant.

5. The PPO-vs-SAC winner is objective-dependent: in an 8-zone VRF annual
simulation, the PPO agent cuts comfort violations 67% vs rule-based
control while the SAC agent saves 27.6% energy — PPO favored comfort,
SAC favored efficiency. (S2, resolved result 4, citing
https://arxiv.org/abs/2608.17235, Ziadi et al., IEEE CCTA 2026.)
Transfer limits: residential VRF plant, abstract-level detail in our
evidence (full reward weights and violation pricing not inspected); the
point stands as a counterexample to "SAC always beats PPO on HVAC."

6. TreeC (2023): all tested RL algorithms perform poorly on the BOPTEST
case, even vs the simple RBC baseline (summary table rates RL "Bad" vs
RBC "Average"); on the second benchmark (ANM6Easy) SAC is good but PPO
is not. There is no consistent RL winner across benchmarks. (S1 excerpts
from http://arxiv.org/pdf/2304.08310v1.) Transfer limit: BOPTEST cases
are single-zone/small plants; ANM6Easy is a grid task, not HVAC.

7. CityLearn centralized storage-management comparison (Princeton senior
thesis, 2/4/8-building districts): SAC achieves the strongest cost and
energy metrics, slightly better than PPO; PPO produces smoother, more
stable electricity use with significantly less memory than SAC and less
compute than MPC; both RL methods beat MPC on most metrics and MPC
struggles to scale; PPO is judged the most balanced controller. (S1
excerpt from
https://theses-dissertations.princeton.edu/entities/publication/2a3a4067-d8e5-4158-9849-8adfaadd20e1.)
Transfer limit: storage dispatch (no comfort cliff, no thermal dynamics)
is the least HVAC-like setting in this document; it does not transfer to
comfort-weighted VAV control.

### Failure modes

8. PPO boundary-collapse failure mode (single-seed Sinergym 5-zone
replication, SB3, linear reward 0.5/0.5): all three PPO runs (200k, 1M,
1M reweighted 0.3/0.7) produce bit-identical deterministic evaluation
policies pinned at heating setpoint 12.0 C and cooling setpoint 23.25 C
(both lower bounds, std 0.0 over 35,040 steps), yielding 5,994 kWh/yr
but 6,311 C-h/yr violations (reward -13,821); SAC at 1M steps uses the
full action space: 17,499 kWh/yr, 282 C-h/yr violations (reward -4,065).
Neither 5x samples nor +40% comfort weight moves PPO off the boundary
corner. Mechanism stated in the reproduction: the Gaussian policy mean
reaches a boundary, std collapses, and updates push further into the
boundary. (S1 excerpts and S2 resolved result 4, both citing
https://github.com/shiplusheba/hvac-rl-sinergym and
https://raw.githubusercontent.com/shiplusheba/hvac-rl-sinergym/main/README.md.)
Caveats: explicitly single-seed results; SB3 defaults; setpoint action
space. Transfer note: our action space has hard bounds on every channel
(dampers >= 10%, discharge <= 308.15 K, recirculation <= 0.70), so the
same collapse geometry applies unless the policy is residual-centered
with small output scale.

9. Our own PPO run diverged on the sharp comfort landscape (S3 context
lines 16-18: occ_gated4/occ_gated7 public 0.01323 vs leader 0.00521, and
"PPO diverged on the sharp comfort landscape in our own experiments").
Our PPO setup (S5, `tools/slot_ppo.py` lines 60-85, 88-101): BC
initialization with memory-feature columns zeroed (stable P-only start),
Gaussian exploration with init std 0.02 / min std 0.01, clip 0.2,
lr 3e-5, gamma 0.995, gae-lambda 0.95, entropy coefficient 0.0 (default),
rollout 2016 steps x 8 workers, 6 epochs, minibatch 1024, KL early-stop
0.01, two-phase value fit (v-lr 1e-3, v-steps 100). S3 sections 4.1-4.4
attribute the divergence class to: stiff two-scale reward (comfort slope
~0.18/K vs near-flat energy slope) dominating advantage estimates; PPO
clipped-surrogate bang-bang/saturation collapse (Claim 4/8 mechanism);
zero entropy pressure (ent-coef 0.0) vs SAC auto-tuned entropy; missing
reward/observation normalization on a two-scale landscape. S4 section E
disposition cautions that the BOPTEST SAC/PPO paper is a bounded
empirical warning, not a proven diagnosis of our run — our divergence
mechanism is INFERENCE-grade until an instrumented rerun confirms it.

10. Agents trained in one climate perform best in that same climate with
significant reward gaps on transfer (Mann-Whitney p < 0.05 in all cases);
sequential multi-climate training shows catastrophic forgetting and does
not beat single-climate specialists. (S1 excerpts from
https://arxiv.org/html/2401.05737v3.) A model-based continual-RL
framework with a hypernetwork dynamics model re-converges within 5
episodes after a task switch, outperforming MFRL and mitigating
catastrophic forgetting. (S2, resolved result 4, citing
https://arxiv.org/abs/2503.19212v1.) Transfer note: our evaluation is
one frozen policy on 30 hidden (building, weather) pairs with no
per-building training, so cross-climate/cross-building transfer without
fine-tuning is the binding test — few-shot re-convergence numbers do
not directly apply.

### Model-based reference points (context for the residual-policy decision)

11. Gnu-RL (differentiable MPC policy, imitation-pretrained, PPO-refined):
simulation saving 6.6% total energy vs the best-performing Zhang and Lam
(2018) RL controller in the same calibrated 600 m^2 workspace; three-week
field trial in one 20 m^2 VAV conference room: 16.7% cooling-demand
reduction (Table 5: 99.4 -> 82.8 kWh) vs a fixed-schedule controller on
weather/duration-normalized historical June data. (S3 section 1.1; S4
section B, BODY, https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.562239/full.)
Transfer limits: radiant-heating plant in sim (supply-water temperature,
not our multizone VAV action vector); small field scale; historical
(not simultaneous randomized) baseline; cooling demand, explicitly not
energy consumption. S4 disposition: useful evidence for controller
initialization + task-based refinement (imitate occ_gated7, then refine);
does not validate a generic SAC-residual recipe and does not support
banning PPO in every architecture.

12. MB2C (ensemble NN dynamics + MPPI, EnergyPlus 5-zone VAV-reheat office,
15-minute decisions): average energy savings 10.65% vs campus-style RBC
and 8.23% vs a random-shooting MBRL implementation, with comparable
comfort; asymptotic performance comparable to PPO; the 10.52x advantage
concerns required training data/convergence, not score. (S3 section 5.2;
S4 section C, BODY, https://arxiv.org/html/2302.00725v1, Table II.)
Transfer limits: learned controller changes zone heating/cooling
setpoints only (AHU stays with EnergyPlus); result confounds dynamics
model + optimizer (not an isolated MPPI-vs-CEM test); no comparison vs
tuned greybox MPC. Relevance to Track A: sets the bar a residual
model-free policy must beat — MB2C ~ PPO asymptotically on 5-zone VAV,
so a residual SAC/TD3 that cannot beat MB2C-level gains is not obviously
the best use of compute vs planning.

13. ADAPT (physics-aware diffusion world models, 2026): -7.3% energy and
-30.2% discomfort vs strongest baselines under IID control on
SemibuildingSim and Sinergym, with robust transfer to unseen seasons and
climates OOD with only marginal degradation. (S2, resolved result 4,
citing https://arxiv.org/abs/2608.19804.) Grade: ABSTRACT-level in our
evidence (S2 notes flag the PDF body as unreadable with granted tools;
head-to-head numbers vs DreamerV3/MBPO/MBVE unresolved). Reading lead
only, excluded from recommendations.

14. Higher-order action (jerk) regularization preserves competitive RL
performance while cutting HVAC equipment switching by 60%. (S2, resolved
result 4, citing https://arxiv.org/abs/2601.02061v1, NeurIPS workshop
2025.) Grade: ABSTRACT-level. Portable idea for Track A training logic
(action-rate penalty), pending full-text check.

### Cost, scale, and practice notes

15. Reward practice is uniform across successful studies: weighted
comfort-dominant sum `r = -alpha*E - beta*comfort_violation`; Sinergym
normalizes power and violation terms so neither dominates numerically;
SB3 canonical recipe is VecNormalize (obs + reward) plus advantage
whitening; action-rate |du| penalties kill cycling. (S3 section 7 with
per-item URLs, including
https://ugr-sail.github.io/sinergym/compilation/main/pages/rewards.html
and https://stable-baselines3.readthedocs.io/en/master/guide/examples.html.)
Grade: search-snippet inspection in S3, not BODY-verified — treat as
practice checklist, not proven-on-our-sim fact.

16. Compute gap (same 5-zone replication as Claim 8, CPU-only): PPO ran
~770 steps/s (4 vec envs, ~22 min for 1M steps) vs SAC ~70 steps/s
single-env (~4 h for 1M steps), ~10x slower per step because SAC updates
every env step while PPO batches every 8,192 steps; the bottleneck is
EnergyPlus, not the 17->64->64->2 MLP. (S1 excerpt from
https://github.com/shiplusheba/hvac-rl-sinergym.) Transfer note: S4
section 6 warns that sample efficiency still consumes wall-clock budget
even with unlimited resets — our simulator throughput must be measured
before assuming millions of steps are cheap (see plan checks C-01, C-02).

17. DreamerV3-style latent world models are claimed (secondary survey) to
offer 10-100x better sample efficiency than model-free methods, but are
harder to implement/debug and suffer compounding errors in long
imagination rollouts. (S2, resolved result 4, citing the rlox survey
https://raw.githubusercontent.com/wojciechkpl/rlox/HEAD/docs/research/08-dreamer.md,
which itself cites Hafner et al.) S3 section 5.1 adds: no direct HVAC
application found; Dreamer strength (pixels, long-horizon sparse reward)
mismatches HVAC (low-dim states, dense stiff reward). Grade: secondary
survey; excluded from recommendations.

18. Biemann et al. (as cited in the Sinergym evaluation): off-policy
algorithms predominate in a 2-zone datacenter, SAC cutting consumption
~15% while guaranteeing comfort and data efficiency. (S1 excerpt from
https://arxiv.org/html/2401.05737v3.) Grade: citation-within-paper, not
independently inspected — supporting context only.

### Synthesis: why each algorithm wins where it does

19. No universal model-free ranking exists for HVAC (S4 section A
disposition; Claims 1 vs 3 vs 5 vs 7 disagree by design). The pattern
across inspected evidence: (a) SAC wins on bounded continuous HVAC
actions when entropy-regularized exploration avoids the PPO boundary
collapse of Claims 4/8 — S2 synthesis: "SAC wins on bounded continuous
HVAC actions via entropy-regularized exploration (avoids PPO boundary
collapse); TD3 is competitive especially in data-center-like plants";
(b) PPO wins on stability of the learning curve and can win outright on
comfort-weighted objectives (Claims 3, 5) when it does not collapse;
(c) TD3 wins where aggressive energy/comfort tradeoffs are rewarded and
mild violations are cheap (Claim 2); (d) all model-free methods reliably
beat weak baselines by 10-30%, roughly tie well-tuned RBC, and lose to
MPC/planning on standardized testbeds with strong baselines (S3 section
1.2 pattern; S4 section G Arroyo et al. BODY: no tested RL variant beats
MPC on a single-zone Brussels dwelling — cited here as S3/S4 context,
full claim outside A-C scope). For our competition — strong rule-based
baseline (occ_gated7), stiff 0.18/K comfort slope, frozen-policy
transfer — the literature predicts model-free will struggle to beat a
good RBC (Claim 1 is the direct warning) unless the residual framing +
normalization + entropy + multi-seed protocol in the companion plan
(`docs/plan/11-algorithms-checks.md`) all hold.

## Open questions

1. Residual SAC vs residual PPO vs TD3 head-to-head on OUR simulator and
reward: literature cannot predict the winner (Claims 1, 3, 5 disagree);
requires the controlled comparison in plan group A.
2. Recurrent (memory) vs feedforward residual policy: no inspected source
in S1-S5 quantifies an MLP-vs-LSTM ablation for 5-minute commercial
HVAC; our partial observability (building-level occupancy only, hidden
thermal states) motivates the check, but it is currently evidence-free —
plan group D is exploratory, not literature-backed.
3. Which entropy schedule prevents boundary collapse in our ~92-dim
residual action space: SAC auto-tuning vs fixed entropy bonus on PPO vs
none (our S5 default ent-coef 0.0 is a suspect, not a proven cause).
4. Do LDLP comfort-normalized energy gains (-4% to -26% vs PPO/SAC) hold?
UNOPENED in both S4 section 3 (page 429, PDF inaccessible) and S2 notes
(HTTP 403) — excluded until the body is inspected.
5. ADAPT vs DreamerV3/MBPO/MBVE head-to-head numbers: UNOPENED (S2
notes) — excluded until the body is inspected.
6. MDPI 2026 SAC-vs-PPO/TD3/DDPG comparisons: UNOPENED (HTTP 403 per S2
notes) — excluded until inspected.
7. Gnu-RL PPO-refinement hyperparameters (clip, entropy, KL handling)
that made PPO succeed as a refiner while it fails from scratch: not
extracted in S4 section B scope — needed before copying the
imitate-then-refine recipe.
8. Optimal control timescale interaction: all benchmark evidence is at
15-minute steps; our 5-minute steps triple the horizon (8640-step
episodes) and no inspected source covers long-episode convergence for
these algorithms.
