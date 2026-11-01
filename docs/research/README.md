# HVAC deep research

Updated: **2026-10-28**. Initial snapshot: `2603d8e31eef3465ec54d02e1d62eb4d743c9b93`;
closing reconciliation: `59a307b7aa6d47210c2f921a6b9a0ea7d6428605`.
Report 06 records the working changes and intervening journal updates. Documentation is
English; repository code, experiments, primary research and real HVAC
requirements are kept distinct in the evidence.

The repository already contains an extensive stateful controller, learning
experiments, equipment/thermal models and thousands of stored result files.
The immediate research need is reproducible evaluation of the current
controller family and causal evidence for coupled air/water optimization.
The October 27 action ranking is superseded by later experiments and the
source audit. There is no evidence identifying the hidden leader's algorithm
or a reliable percentage of loss that a new method will recover.

## Read in this order

| Document | Purpose |
|---|---|
| [06 Repository analysis](06-repository-analysis.md) | Architecture, exact interfaces and timing, scenario inventory, recomputed results, policy/model status and provenance gaps |
| [07 Evidence collection plan](07-evidence-collection-plan.md) | What to collect, where to get it, required fields, priorities, evaluation design and acceptance conditions |
| [08 HVAC domain reference](08-hvac-domain-reference.md) | Thermal/air/water physics, psychrometrics, ventilation, standards, commissioning and simulator-to-field limitations |
| [09 Evidence audit](09-evidence-audit.md) | Primary-paper verification, corrected RL/MPC claims, unsupported inferences and source exclusions |
| [10 Control research synthesis](10-control-research-synthesis.md) | Updated method comparison, model/planning requirements and research priorities grounded in the current repository |

## Earlier research tracks

These reports retain detailed literature and historical experimental work.
Their dates, access limits and corrections matter. They are not a list of
currently untested ideas, and unverified numerical claims are excluded from
the current synthesis.

| Track | Useful material | Current qualification |
|---|---|---|
| [01 Competition intelligence](01-competition-intel.md) | Related challenges, published approaches and benchmark leads | Different task/asset/reward contracts limit comparisons; no disclosed hidden leader method |
| [02 Model predictive control](02-mpc.md) | Model classes, observers, constraints, coordination and implementation literature | Savings priors are study-specific; use report 09's evidence standard |
| [03 Rule-based control](03-rule-based-sota.md) | G36 and vendor mechanisms, actuator/sensor mapping | Several local ports now have negative full-episode evidence; standards/version and ventilation-equation corrections apply |
| [04 Reinforcement learning](04-rl.md) | Broad source-discovery inventory and candidate algorithms | Original text was based on search results; report 09 verifies selected primary bodies and supersedes overbroad advice |
| [05 Gap forensics](05-gap-forensics.md) | Historical occ4 loss decomposition and probe coverage | Recomputed train-30 decomposition is retained; new policy coverage/provenance is in report 06 |
| [October 27 synthesis](2026-10-27-synthesis.md) | Historical ranking and unresolved reading leads | Archived; superseded by report 10 |

## Online RL series (2026-11-01)

Seven dated reports plus an executable check plan in
[docs/plan/](../plan/README.md). Discovery-grade web evidence (two
deep-research runs whose verification stage returned `no_evidence`, so all
web numbers below are read pages/PDFs, not verifier-confirmed claims),
a line-by-line reverse-engineering of the public residual-PPO notebook and
dataset (`yes-rl-can-run-a-building-residual-ppo.ipynb`, `rl.zip`),
and repo grounding. Track F (notebook) is the only fully body-verified
report; literature numbers in tracks A–E carry per-claim provenance and
transfer limits.

| Report | Contents | Status |
|---|---|---|
| [11 Online RL algorithms](11-online-rl-algorithms.md) | PPO/SAC/TD3/recurrent evidence per benchmark; no universal ranking; residual SAC primary, TD3 backup, gated PPO; STOP rule | Draft 2026-11-01 |
| [12 Online RL architectures](12-online-rl-architectures.md) | Shared per-entity MLP, temporal-conv+LSTM, in-context transformer, morphology-aware transformer; 23 findings, 8 open questions | Draft 2026-11-01 |
| [13 Online RL training logic](13-online-rl-training-logic.md) | Adv normalization, masking, DR curricula, checkpoint judges, LR schedules, reward normalization, honest billing | Draft 2026-11-01 |
| [14 Online RL failures and exploits](14-online-rl-failures-exploits.md) | Cycling, boundary collapse, residual authority, billing/metric exploits, lever saturation | Draft 2026-11-01 |
| [15 Online RL model-based](15-online-rl-model-based.md) | ADAPT, LDLP, Counter-Dyna, hypernetwork, DreamerV3 head-to-heads; CPU-deployment exclusions | Draft 2026-11-01 |
| [16 Skynet residual-PPO notebook](16-skynet-residual-ppo-notebook.md) | Notebook+dataset reconstruction; shipped `m3r_ramp_cmagen_seed1@752640`, board 0.01128; honest gain -10.6% vs shipped -16.9% | Draft 2026-11-01 |
| [17 Repo grounding for Online RL](17-repo-grounding-online-rl.md) | Our policies, ladder, transfer method, simulator and submission constraints bounding RL design | Draft 2026-11-01 |

## Findings that change the next action

- Existing occ4 train-30 CSVs reproduce loss `0.009628636765325417`;
  comfort contributes `3.850576%`. This is a measured decomposition, not
  an estimate of removable energy waste.
- Full episodes reversed the apparent short-window dual-max improvement.
  Later experiments also close the tested blanket prestart, G36 SAT and
  staging variants. Read the [October 28 journal](../2026/10/28/README.md).
- Economizer/heating-deadband/heatopt combinations have promising local
  evidence and unresolved night-tail, coverage and version-identity issues.
  A selected-cell win does not establish a hidden or full-transfer win.
  The closing journal update records occ8 and occ9 submitted, scores pending,
  and the tested pseudo-DCV variant inert; these are journal-reported statuses.
- Thermal response appears in a delayed sensor; causal model evaluation must
  preserve the action/observation sequence. Diagnostic success alone does
  not establish that a controller can safely optimize alternative actions.
- Primary RL studies support conditional comparisons. They do not justify
  a universal algorithm ranking or a prohibition on every PPO-based design.
- Real HVAC also requires humidity, ventilation, fault and commissioning
  evidence that this competition score does not establish.

## Evidence and completion boundaries

This package includes a current repository audit, a concrete collection
specification, a broad HVAC reference, a focused primary-source audit and
an updated research synthesis. Existing result files were read and selected
metrics recomputed; no new simulations, controller changes or submissions
were performed for this documentation update.

The source audit is targeted rather than a claim that every paper ever
mentioned in tracks 01–05 has been verified. Unavailable bodies and old
search-only claims remain explicitly excluded from present recommendations.
Standard editions and implementation versions have separate access labels.
The new reports distinguish inspected bodies, publisher descriptions,
journal claims, recomputed data and research proposals.

Remaining collection work is explicit in report 07: exact historical run
identity, current-stack full-suite/transfer outcomes, counterfactual model
accuracy, live competition status and field constraints. These are future
experimental inputs; the documentation does not manufacture their answers.
