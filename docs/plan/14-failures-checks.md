# Failure-mode and anti-exploit checks for online RL (Track D plan)

Date: 2026-11-01. Status: draft. Scope: safety and anti-exploit checks to run
before and during online/residual RL above occ_gated7. Research basis:
`docs/research/14-online-rl-failures-exploits.md` (claims F1-F28).

Rules: full episodes only (8640 steps); loss = abs(df.agent_reward.sum()) /
len(df); keep total concurrent sims <= 10 with 1500s+ timeouts; never edit
policy code while a validation run using it is in flight (AGENTS.md).

## Check table

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|---|---|---|---|---|
| B1 | Audit boiler/tank billing path in source | Read `smart_buildings/smart_control/reward/setpoint_energy_carbon_regret.py` and `smart_buildings/smart_control/reward/natural_gas_energy_cost.py`; grep for setpoint vs tank/actual temperature in the billing computation | Billing quantity identified line-by-line; setpoint-vs-actual gap confirmed or refuted in writing | P0 |
| B2 | Confirm PR #2 fix state in this checkout | `git log --oneline -5 -- smart_buildings/smart_control/` and read the PR #2 diff if present; record commit SHA or ABSENT | Fix commit recorded, or ABSENT with re-audit date | P0 |
| B3 | Re-bill one Fairbanks 30d CSV at tank actual | `./.venv/bin/python -B tools/fullrun.py results_b3 8640 4 1500 occ_gated7 <fairbanks-pairs>`; recompute energy cost from logged tank temps vs billed values | Unbilled kWh over 30d Fairbanks reported; target 0 after fix; digest reference was +914 kWh pre-fix | P0 |
| B4 | Recompute shipped-vs-honest gain on same window | Same CSVs as B3: compare billed loss vs re-billed loss for rule and residual policies | Report both numbers; digest reference: shipped -16.9% (0.01251 vs 0.01040) -> honest -10.6% (0.01207 vs 0.01079) -> PR#2 -9.0% | P0 |
| B5 | Audit reward blind spots (what is NOT billed/scored) | Read `smart_buildings/smart_control/reward/setpoint_energy_carbon_regret.py:333-344` and one gin config, e.g. `smart_buildings/training_scenarios/building_train_2/atlanta_3/config.gin`; list unscored quantities (ventilation heating, shading, humidity, wear) | Written blind-spot list; each item has a monitor or a stated reason to ignore | P1 |
| C1 | Measure occ_gated7 baseline switching rate | `./.venv/bin/python -B tools/fullrun.py results_c1 8640 4 1500 occ_gated7 <train-30-pairs>`; count on/off and mode switches per zone per day from action logs | Baseline switches/zone/day on train-30 recorded; literature reference points: 4.1 (SARSA) and 1.07-1.50 (BOPTEST baseline) | P0 |
| C2 | Count residual switches behind the rule layer | Same runs with residual policy registered in `POLICY_MAP` (`smart_buildings/evaluate.py:218`); log residual sign flips and equipment commands separately from base commands | Residual-attributable switches/zone/day reported next to C1 baseline | P0 |
| C3 | Add jerk/switching penalty to residual training | Implement third-order action-derivative penalty (start lambda=0.1 per U2, retune to our action scale); keep penalty weight in the run config, not in shipped policy code | Penalty term logged per step; lambda value recorded per run | P1 |
| C4 | Ablate the jerk penalty (with vs without) | Two fullrun sweeps identical except penalty on/off: `./.venv/bin/python -B tools/fullrun.py results_c4a 8640 4 1500 <residual-pen> <pairs>` and `results_c4b` with `<residual-nopen>` | Report loss delta AND switching delta; gain that vanishes under penalty is a cycling exploit, not efficiency | P1 |
| C5 | Deep-cold heating-off probe | Night/deep-cold Fairbanks pair with residual enabled; assert heating commands never drop to zero while occupied and cold | Zero occupied heating-off events; digest failure reference: Fairbanks 0.02604 -> 0.07143 -> 0.56009 at 750k | P0 |
| G1 | Log rail saturation during training | Log per-dim action mean/std and fraction of steps within 5% of each rail; alert on std collapse toward any rail | Saturation report per checkpoint; Sinergym failure reference: std 0.0 over 35,040 steps at both lower bounds | P0 |
| G2 | Guard-off evaluation of the shipped checkpoint | Evaluate the same residual checkpoint twice on identical pairs: safety filter on vs filter bypassed (filter code path flagged, policy code untouched during the run) | Loss delta on/off reported; a checkpoint that needs the filter to avoid catastrophe is rejected | P0 |
| G3 | Worst-cell (not mean-only) comfort report | From every eval CSV: max per-cell comfort regret, hot-side and cold-side maxima, occupied violation hours | Worst-cell numbers accompany every mean; AH-RL reference: good cost/carbon/cold-comfort can hide severe overheating | P0 |
| A1 | Authority sweep: rail width | Register 3 residual variants in `POLICY_MAP` with narrow/medium/wide rails; `./.venv/bin/python -B tools/fullrun.py results_a1 8640 4 1500 <r-narrow,r-medium,r-wide> <screen-pairs>` | Loss vs rail width curve; pick the narrowest rail that still improves | P1 |
| A2 | Authority sweep: adaptive comfort throttle | Add per-step authority weight reduced by comfort risk (AH-RL rule); sweep 2 throttle gains on the same screen pairs | Throttle gain recorded; comfort-risk authority reduction demonstrated on at least one tail cell | P1 |
| A3 | Full-authority vs gated comparison | Same screen: full-authority residual vs gated residual (same net, e.g. d64x2) | Gated must beat full on held-out; digest reference: 0.011175 full vs 0.010868 gated | P1 |
| A4 | Action projection / clipping audit | Verify element-wise clipping onto the admissible action space in the residual wrapper; unit-check corners (min, max, mixed) | Clipping test green; no NaN or out-of-range action in any eval CSV | P1 |
| W1 | Worst-climate screen: deep cold | Include Fairbanks January pair(s) in every screen; report loss, comfort, unbilled-energy proxy | No occupied heating-off; Fairbanks loss within the multi-seed noise band (sd ~0.00025) of the claimed gain | P0 |
| W2 | Worst-climate screen: hot tail | Include hottest pair(s) (e.g. Tampa / Tucson summer); report hot-side comfort regret and cooling energy | No hot-side blowout; Sinergym/AH-RL hot-side pattern explicitly checked | P0 |
| W3 | Transfer-12 gate before any gain claim | `./.venv/bin/python -B tools/fullrun_t.py results_w3 8640 4 1500 <policies> <pairs> smart_buildings/transfer_scenarios/` (regenerate scenarios first via `tools/gen_transfer.py` with fixed seeds if the dir is empty; it is git-ignored) | Gain holds on transfer-12 within noise; board/holdout digest band 1.05-1.16 used as sanity range | P0 |
| O1 | Observation ablation: base-controller state | Compare residual with vs without base-controller state features (PI integrals, prestart latch, hysteresis flags) on one cold + one hot pair | State-ablated variant must not reproduce the slot-BC failure shape (subset-8 0.0368 vs teacher 0.0175, extremes 2.2-2.6x) | P1 |
| O2 | Observation ablation: rule-state unseen penalty | Same ablation evaluated on unseen buildings; watch for the digest pattern (better on seen, worse unseen) | Report seen and unseen losses separately; digest reference: b8 0.01021 vs 0.01039, unseen 0.011727 vs 0.011411 | P1 |
| N1 | Multi-seed checkpoint selection | Train >= 3 seeds; select checkpoints by mean over seeds on a holdout screen, never by single best checkpoint | Seed-wise losses tabled; single-checkpoint gains below ~0.0005 (2x noise sd) are not claims | P0 |
| N2 | Capacity check before scaling nets | Keep d64x2 as the default; any larger net (d128/256/512x3) needs a same-budget comparison | Larger net must beat d64x2 on the holdout screen; digest prior is that all larger nets were worse | P2 |
| E1 | Local exec check 300+ steps incl. night | Assemble via `tools/assemble_notebook.py` only (verbatim class blocks, keep `occ_gated:<consts>` with `STEPS_PER_HOUR`); exec-check 300+ steps covering night/shutdown/prestart | 300+ steps green with no night-only NameError or rail error | P0 |

## Definition of done

Billing group (B1-B5): done when the billing path is quoted line-by-line,
the PR #2 state is recorded with a SHA or ABSENT, a Fairbanks 30d re-billing
number exists (target 0 unbilled kWh), shipped-vs-honest gains are tabled on
the same window, and the reward blind-spot list is written.

Cycling group (C1-C5): done when the occ_gated7 switching baseline on
train-30 is recorded, residual-attributable switches are counted separately,
the jerk penalty ablation (C4) shows the gain survives the penalty, and the
deep-cold probe (C5) shows zero occupied heating-off events.

Guard group (G1-G3): done when rail-saturation logs exist for every shipped
checkpoint, a guard-off evaluation is tabled next to the guarded number, and
every eval reports worst-cell hot/cold comfort alongside the mean.

Authority group (A1-A4): done when the rail-width and throttle sweeps
identify the narrowest improving authority, gated beats full authority on
held-out, and the clipping audit test is green.

Climate/transfer group (W1-W3): done when every screen contains a deep-cold
and a hot-tail pair with per-tail reports, and no gain is claimed without a
transfer-12 gate.

Observation/noise group (O1-O2, N1-N2, E1): done when both observation
ablations are tabled (seen and unseen separately), checkpoints are selected
by multi-seed mean with the ~0.0005 resolution floor respected, net capacity
stays at d64x2 unless beaten on holdout, and the 300+ step exec check is
green.
