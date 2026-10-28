# 2026-10-28

## Summary

Goal resumed with the research-plan objective (beat 0.00521 top-down).
Started item 1 (dual-max damper/reheat decoupling): bit-identical refactor
of the PI output map into an overridable hook, dual-max mixin + occ_dualmax7
+ 0.85-gain control arm, off-exactness proven, +3.1%/+2.7% on the first
600-step smoke. Hard-cell full-8640 screening (6 fairbanks + 2 mild,
3 arms) launched; results pending. (Overnight 10-27 work — oracle/CEM,
P-a/P-b/P-c probes, transfer-12, deep research, occ6/occ7 scores — is
logged in the 10-27 appendix update below.)

Later research update: audited the current repository and rebuilt the HVAC
research index around the completed experiments. Added a collection specification,
domain reference, primary-paper evidence audit and current control synthesis.
Corrected unsupported leader/algorithm inferences and a ventilation equation.
Existing CSVs were inspected; this update launched no simulations or submissions.

## Status

| Whose | Score | State |
|---|---|---|
| leader (Joe Penn) | 0.00521 | LB |
| us (occ_gated4) | 0.01323 | COMPLETE (best, tied) |
| us (occ_gated7) | 0.01323 | COMPLETE (tied 10-27) |
| us (occ_gated6) | 0.01337 | COMPLETE (+1.1% regression) |

Research update: no new scores. The table is retained as the day's historical
record, not a fresh live leaderboard query. The closing reconciliation of
`9cd060e` and `59a307b` reports both occ8 and occ9 submitted, scores pending;
the research audit has no new public score for either.

## Done

- `zone_pi.py`: extracted `ZonePIPolicy._zone_output_map(u)` hook from
  `_command_zones` (pure method extraction, same op order).
- `occ_dualmax.py` (new): `DualMaxOutputMixin` (G36 valve-first-then-airflow
  staging, knobs `$DM_ON=1/$DM_SPLIT=0.5/$DM_HEAT_MIN=0.1`),
  `OccDualMax7Policy` (KEY `occ_dualmax7`), `OccHDamp085G7Policy` (KEY
  `occ_hdamp085_7`, control arm). Registered in `POLICY_MAP`.
- Bit-identity chain: refactor hash e9888825756a4637 == pre-refactor;
  `dualmax DM_ON=0` == occ7 exactly.
- Launched `/tmp/dm1`: 8 pairs × 3 policies full-8640 (fairbanks_1/2/4/5/7/8
  + atlanta_2 + denver_2). Commit `769dc61` pushed before launch.

- Audited interfaces, causal timing, scenarios, stored results and notebooks
  in `docs/research/06-repository-analysis.md` to establish the research baseline.
- Specified evidence priorities, run/transition/source records, external datasets
  and evaluation gates in `docs/research/07-evidence-collection-plan.md`.
- Added physics, ventilation, standards and field applicability in
  `docs/research/08-hvac-domain-reference.md` to explain simulator limitations.
- Verified seven primary paper bodies in `docs/research/09-evidence-audit.md`
  and corrected overbroad RL/MPC conclusions and unsupported leader inference.
- Replaced `docs/research/README.md` with the current reading guide, archived
  its dated ranking in `docs/research/2026-10-27-synthesis.md`, and added
  `docs/research/10-control-research-synthesis.md` for updated priorities.
- Corrected the section 1.4 ventilation-efficiency sign in
  `docs/research/03-rule-based-sota.md` against the linked LBNL implementation;
  added historical-scope notices to all five original tracks.

## Measured

- occ_dualmax7 — b1/fairbanks_1 first-600 — 0.01648 vs 0.01701 (+3.1%).
- occ_hdamp085_7 — same window — 0.01655 (+2.7%).
- dm1 full-8640 (6 fairbanks + atlanta_2 + denver_2), means: occ7 0.016157,
  dualmax 0.016964 (+5.0%), hdamp085 0.016288 (+0.8%). Dual-max loses on
  ALL 8 pairs (+2.5% fb1 … +18.3% fb4, +15.2% denver); hdamp mixed
  (−1.7% fb8 … +3.0% fb1). The 600-step +3.1% was startup-transient
  illusion — full-horizon reversal, exactly the S-U2 warning.

- occ_gated4 — train-30, existing 8640-row CSVs recomputed — 0.009628636765325417 — delta 0 versus the same archived reference.
- occ_gated7 — hard-cell-8, existing `/tmp/dm1` CSVs recomputed — 0.016157327656207122 — reference.
- occ_dualmax7 — hard-cell-8, existing `/tmp/dm1` CSVs recomputed — 0.01696358643371738 — +0.000806258777510258 versus occ_gated7.
- occ_coolshift7 — hot-cell-4, existing `/tmp/csh` CSVs recomputed — 0.009672823510725462 — -0.000077852260133571 versus occ_gated7.

## Decisions

- Naive dual-max CLOSED as specified (output-map swap loses full-episode).
  Lesson: the waste is plant-level over-conditioning (water/SAT), not
  zone delivery rate; restricting delivery loses to catch-up dynamics
  (same family as nightdrop3 +18%). Combination dual-max × retuned PI
  parked as follow-up after items 2–3.
- Item 3 shape levers (damper cap, below-band target) TRADE morning
  comfort for night gas at a bad rate on the anchor cell (fb1: gas −1.5%,
  regret −0.018 → −0.033, stop-rule FAIL). Dead ramp-gate idea dropped
  (grace already ends prestart ~05:00 — verified by instrumentation).
- Item 3 TIMING precision also CLOSED negative: (1.0/0) and (1.0/2) flat
  on atl2/fb2/fb8 but CRASH denver (+9.2%/+5.8%, morning deep-regret
  1.6% → 13.1%, stop-rule FAIL). Margin is load-bearing on cold-mornings
  climates; blanket cuts are wrong. Handoff: ADAPTIVE timing = item 5
  (JCI optimal start, daily adaptation), not fixed cuts.

- A leaderboard score does not identify the leader's controller or hidden
  failure distribution. Proof: the repository contains scalar scores without
  hidden action traces or a solution disclosure. Consequence: remove these
  inferences from the current research recommendations.
- The occ4 energy/carbon share is a decomposition, not an avoidable-waste
  estimate. Proof: existing full trajectories quantify costs but no matching
  zero-cost comfort-preserving counterfactual. Consequence: require paired
  complete-episode evidence for each claimed saving.
- Prioritize exact current-policy provenance and causal model evaluation.
  Proof: recent results are scattered across temporary paths, some policy
  keys cover several variants, and thermal diagnostics explicitly limit
  their control claims. Consequence: use the collection contract in report 07.

## Open

- Research update: old top-ten recommendations are reconciled in reports 06
  and 09; completed negative variants stay closed in their tested form.
- Moved to the collection plan: archive effective run settings, evaluate the
  current stack on complete declared train/transfer coverage, attribute night
  tails, and validate counterfactual predictions with participant-only inputs.
- Current competition status, unread historical paper bodies and field IAQ/
  humidity constraints remain explicit evidence gaps; none is treated as a
  verified result by the updated synthesis.

- Item 2 CLOSED (edge dead; gentle-centre banked −1.0%, stop-rule clean,
  tails even better). Item 3 CLOSED negative in blanket form (shape trades
  mornings; timing crashes Denver; adaptive timing → item 5).
- Item 4 heating CLOSED negative: fair-port satcap7 ALSO loses on the
  anchor (fb1 +10.7%, fb2 +2.1%, rest flat) — stall-detector cap slows
  recovery, same catch-up family. Both r1 and fair versions dead.
- Item 5 stop-side: evening coast with point-target drop LOSES uniformly
  (fb2 +10.9%, fb8/fb7 +5.7%): evening regret explodes (fb2 eve −0.0003
  → −0.034), mornings unchanged — 17:00 occupancy (17 people) is NOT
  negligible. Stop-rule FIRED. Lesson: exchange improves as occupancy
  falls → item 7 (occupancy-proportional) is the principled version.
- Item 4 cooling tail CLOSED as small win, BANKED not promoted:
  fair-port coolshift7 on 4 hot cells full-8640 (/tmp/csh): sg8
  0.007364 → 0.007322 (−0.6%), phx5 flat +0.05%, sg1 −0.2%, phx6
  0.009146 → 0.008880 (−2.9%); mean ≈ −0.9%, uniform direction.
  Stop-rule clean: phx6 tails BETTER (frac < −0.05: 0.15% → 0.00%),
  regret −0.00087 → −0.00053, elec −1.4% (4364 → 4302), gas +3
  negligible. Candidate for a combined build (coolshift + gentle-centre
  + item 7) rather than its own submit.
- Item 7 CLOSED as banked small win (`occ_prop7` v3, commit `b71e0ca`).
  v1 (err × frac): fb8 −2.6%, atl2 −1.2%, den2 +0.5%, fb1 +5.6% LOSE —
  cold-anchor mornings pay the savings back (morn regret ×7). v2 added
  warmth-gate (coast only when empty AND warm): fb1 → +0.4% neutral,
  cold wins kept. But hot cells lost: phx5 +10.7%, sg8 +5.5%, sg1 +3.0%
  — night coasting overheats the building (cooling mirror of fb1).
  v3 symmetric ride-gate (`ride = m` below centre / `1−m` above):
  phx5 +0.15%, sg8 −0.07%, phx6 +0.09%, sg1 −0.13%, fb1 +0.4%,
  fb8 −0.2%, atl2 −0.5%, den2 −0.6%. Mean ≈ −0.2%, stop-rule clean
  (tails ≤ control on all 8). Combo prop+coolshift REJECTED: coolshift
  inactive in cold (bit-identical to prop) but adds +0.3..+2% on hot —
  target-shift × err-scale double-shifts. Too small to submit alone;
  stack with item 5 next (item 5 has since CLOSED negative, see below).
  Code: `OccPropMixin` in
  `smart_buildings/baselines/occ_prop.py` (`OccProp7Policy`, KEY
  `occ_prop7`, knobs `$OP_ON=1/$OP_MIN=0.5/$OP_MAXN=5`), registered in
  `POLICY_MAP`; off-exactness `OP_ON=0` bit-proven vs occ7 (phoenix_6
  full-8640, /tmp/prop). v3 formula: `scale = 1−(1−OP_MIN)*(1−frac)*ride`.

## Open (end of session)

- Item 5 CLOSED negative in margin-schedule form (`occ_start7`,
  `smart_buildings/baselines/occ_start.py`, KEY `occ_start7`, knobs
  `$ST_ON/$ST_OAT_K/$ST_OAT_REF=283.15`; off-exact bit-proven vs occ7).
  `margin = 6 + K*(REF−OAT_predawn)`: K=1 loses everywhere (den2 +1.7%,
  fb1 +2.8%, fb2 +3.0%, atl2 +0.6%), K=2 worse (den2 +6.0%, fb1 +3.1%,
  fb2 +4.4%, atl2 +2.4%). Extra margin burns gas (+4–11%) while morning
  regret improves (fb1 −0.0025→−0.0007) at a bad exchange rate — the same
  catch-up family as items 3/4: earlier pre-heat costs more than it saves
  in recovery, and even margin CUTS (mild cells) lose via deeper-dip
  recovery sprints (atl2 gas 205→207). occ7's EMA-adapted gap/rate×SAFETY
  equation is at the local optimum; both start-side and stop-side (coast)
  of item 5 are dead.
- Item 6 CLOSED negative (`occ_gsat7`, `smart_buildings/baselines/occ_gsat.py`,
  KEY `occ_gsat7`, knobs `$SAT2_ON/$SAT2_TRIM=0.25/$SAT2_RESP1=-0.5/
  $SAT2_RESP2=-1.5/$SAT2_LO=285.15/$SAT2_HI=291.15`; off-exact bit-proven).
  Two-stage (OAT base 12–18 C + Trim&Respond) vs single-stage demand
  follower: fb8 bit-identical (SAT dead in cold, gate works), atl2 −0.1%,
  den2 +0.6%, sg1 +22% (!). Isolation: pure OAT base (TRIM=RESP=0) sg1
  +2.1% with clean regret but gas 20→41 (cold base forces reheat); the
  T&R loop adds the rest (elec +14%, regret ×13–14). Warmer base 15–18 C
  even worse (+23.5%) — level is not the driver, the quantized hunting
  loop is. Lesson: our instant demand follower already beats G36 T&R
  here; G36 assumes reheat waste dominates, here chiller+comfort do.
  Rogue-zone variant not tried (failure is systematic, not rogue).
- Item 9 CLOSED neutral/negative (`occ_stage7`, knobs `$AHU9_ON/
  $AHU9_OFF_DELAY=12/$AHU9_DB=0.2/$AHU9_U=0.1`; off-exact bit-proven).
  v1 (temp-breach need): atl3 +10.5%, fb2 +4.3%, atl2 +1.6%, den2 −0.8% —
  saves energy (gas −10%, elec −2%) but doubles regret (late re-on).
  v1.1 (u-based need |u|>0.1): atl3 −0.36%, fb2/den2 exactly neutral —
  safe but ~inert (all-zones-calm for 1 h is rare mid-day). Lesson:
  staging doesn't pay in the occ-shutdown architecture (nights already
  off); the 0.012 notebook's force-gate is specific to no-shutdown
  policies. Next: item 10, then distillation ports.
- DISTILLATION of sb-t-combo-d.ipynb (public 0.012, same code family):
  (a) `occ_econ7` explicit PI-hours economizer — atl2 −8.7%, atl3 −3.9%,
  den2 −4.2%, fb2/fb1/fb8/sg1 neutral-inert; stop-rule PERFECT (tails +
  regret bit-equal, gas equal, elec −5..−34%). (b) `occ_dbh7`
  heating-side-only deadband 0.15 K + leak 0.97 — fb1 −2.1%, fb2 −7.3%,
  atl2 −11.2%, den2 −7.8%, BUT tails grow (fb1 1.5%→2.5%, den2
  0.2%→0.44%) at midnight hours (less evening pre-charge); item-2
  closure stands for the BOTH-sided variant only. (c) `occ_hot7`
  Sat12Hot port + `occ_econdbh7` combo written; combo matrix running.
  (d) HeatOpt coil math verified against local `vav.py` source
  (discharge = T_s+e(T_w−T_s), e=min(1,C_w/C_a,(35−T_s)/(T_w−T_s))).
- occ8 SUBMITTED (kernel v1 COMPLETE, submission sent, 4 left today):
  `occ_econdbh7` = occ7 + econ + dbh. Local full-8640: fb1 −2.1%, fb2
  −7.3%, atl2 −19.1%, den2 −11.3%; tails = dbh tails (midnight family).
  Notebook `notebooks/smart-buildings-submission-occ8.ipynb` assembled
  via tool (ZonePIPolicy block refreshed for the integral hook) and
  proven BIT-IDENTICAL to the repo policy on atl2 full-8640. Assembly
  trap fixed: module consts (lowercase `_SAT_*`) are NOT copied by the
  assembler — econ mixin made self-contained (class attrs).
- occ9 STACK (`occ_econdbhhot9` = occ7 + econ + dbh + heatopt; OFF
  bit-exact): fb1 −2.9%, fb2 −11.5%, fb8 −9.5%, atl2 −23.1%, den2 −16.8%,
  sg1 −3.1% — uniform, all better than occ8. Tails = dbh midnight family
  (fb1 1.5%→2.6%, den2 0.2%→0.36%); heatopt slightly amplifies dbh's fb1
  regret (−0.0037→−0.0044) via slower recovery, net still wins. HELD for
  submit until occ8 scores (tail-risk call needs hidden evidence).
- `occ_hot7` (Sat12Hot port) CLOSED as inert-duplicate: bit-equal to occ7
  on all 4 cells (sg1/phx6/atl2/den2) — our mixedness/deep SAT already
  drives −1.0 where OAT > return, and the gate never fires elsewhere.
- occ9 SUBMITTED (kernel v1 COMPLETE, 3 left today): occ8 + HeatOpt.
  Both occ8 and occ9 PENDING in the ~10 h scoring queue.
- Full train-30 suite occ7 vs occ9 (/tmp/suite30, 60×8640, commit
  `59a307b` + uncommitted docs only): mean 0.009630 → 0.008782 (−8.80%),
  occ9 wins 29/30 (only phx5 +0.6%). Best: phx1 −28.6%, fb4 −27.4%,
  denver_7 −26.9%. Tails grow on ~6 cells (fb1 1.5%→2.6%, denver_3
  0.7%→1.6%, atlanta_8 0.3%→0.9%) — ALL midnight family (hours 0/1/23),
  no morning failures. Event table: night regret −0.00204→−0.00259
  (gas −14%), MORNING regret IMPROVES −0.00042→−0.00026 (gas −25%),
  midday flat, evening zero both. Provenance: clean code tree at commit
  `59a307b` (docs-only dirty); no code edits during the run. occ9 6-cell
  numbers reproduce exactly inside the suite.
- SCORES LANDED: occ8 **0.01260** (−4.8% vs occ7), occ9 **0.01236**
  (−6.6%) — both COMPLETE. Transfer ≈ 75% of local (occ9 local −8.8%).
  Leader 0.00521 still 2.37× away; all heuristics exhausted, next is
  report-10 stage 4 (causal gates) + fresh holdout, no noise ships.
- Component ablations full train-30 (/tmp/abl30, 90×8640): econ −1.67%
  (mild cells −3..−22%, inert on hot/deep-cold), dbh −5.39% (wins 29/30,
  only phx5 +0.6%), heatopt −2.21% (heating cells); stack −8.80%
  near-additive. Tails: econ flat 0.0021, heatopt improves to 0.0018,
  dbh raises to 0.0030 (midnight family only).
- New research context (docs/research/06–10 + README, 2026-10-28):
  Oct-27 ranking SUPERSEDED by report 10; no leader-method evidence;
  priorities now = reproducible current-family reference (this suite),
  night-tail quantification (done above), component ablations (/tmp/abl30
  running: econ/dbh/heatopt × 30), causal model gates before any MPC.
- Item 10 CLOSED: (b) pseudo-DCV (`occ_dcv7`, off-exact + ON bit-identical
  to occ7 on sg1, loss-equal on phx6/atl2/den2) is fully inert — the
  organizer economizer already minimizes outside air whenever OAT >
  inside, so the DCV clamp never fires; the plan's "never tried" was
  wrong in effect. (a) warmest-zone SAT reset not built: the 0.012 stack
  tried it (MILD_SatWarm) and dropped it from the final (external
  negative), and our Hot-inert result shows cooling SAT depth already
  optimal on our base. (c) cooling night-drift covered by architecture
  (occ shutdown drifts freely). All heuristic items 1–10 now closed:
  shipped/econ+dbh+heatopt, banked/prop-v3+coolshift+gentle-centre,
  rest negative.

## Appendix

### 10-27 overnight session (22:00–24:00, logged here, file 10-27 frozen)

- Oracle/CEM: `tools/ckpt.py` + `tools/oracle_search.py` (perstep + const
  + CEM modes); numba global-RNG trap found and fixed via per-branch reseed
  (common random numbers); CEM on xfer_3 January: Tue +11.9%, Wed +9.7%,
  Thu +16.8% over occ7; 2H check confirms real (Tue: pure energy win,
  regret 0 both; Thu: 9x comfort-regret cut for +3% energy).
- P-a/P-b/P-c probes over occ7 (fast-trim/prop-kick/high-ceil): off-exact
  proven (caught an NEP-50 float32 promotion bug in P-b); transfer-8:
  pa +0.3%, pb +0.02% (flat), pc +7.2% (dead, x3-Jan +16%). xfer_2 16/16
  timeouts at 1500s (slowest building, not re-run).
- Kaggle: occ6 COMPLETE 0.01337 (+1.1% vs occ4 — prestart misfire);
  occ7 COMPLETE 0.01323 (exact tie — trim ≡ 0 on hidden).
- Deep research workflow: `docs/research/` (6 files, 3082 lines) + ranked
  top-10 plan; headline: our loss is 96% energy+carbon / 4% comfort.
- Speed analysis: reward computed 2x/step (~10% win available), zone-temp
  stats 234 calls/step (~5%); SuperLU 35% structural. Tier 0 not started.

### HVAC research documentation audit

The research pass began at `2603d8e` and reconciled concurrent commits
`9cd060e` and `59a307b`. The later journal records occ9 submitted and
pseudo-DCV inert; no fresh external score is inferred. Research edits are
documentation only, and the three policy/evaluator working-file hashes
recorded in report 06 were rechecked unchanged.

| Requested outcome | Inspectable evidence | Audit result |
|---|---|---|
| Analyze the repository first | `docs/research/06-repository-analysis.md`: source interfaces, causal timing, configuration inventory, current policy/model status, existing CSV reductions | Covered; source evidence, recomputed values and journal-only claims labeled separately |
| Determine what must be collected | `docs/research/07-evidence-collection-plan.md`: prioritized questions, manifests, transition/source schemas, datasets, holdouts and acceptance conditions | Covered; future measurements explicitly distinguished from collected evidence |
| Provide comprehensive HVAC research for this project | Existing tracks 01–05 plus new domain reference, seven-paper primary audit and current synthesis | Covered across physics, equipment, IAQ, standards, rule control, MPC, learning, transfer, diagnostics and commissioning; inaccessible claims excluded |
| Write documentation in English | All research Markdown, documentation index and day log checked for Cyrillic prose | Check passed |
| Preserve navigable and usable documentation | All local Markdown targets resolve; code fences balanced; 26 domain source-reference IDs defined and used; whitespace check | Check passed |
| Verify quantitative anchors | All 30 occ4 reward CSVs independently read; every one has 8640 rows; mean reproduces `0.0096286367653254` at shown precision | Check passed; no new simulator scores produced |
| Account for later evidence and prior failures | Historical synthesis archived; each original track has a correction notice; Oct-28 closures and concurrent submission updates reconciled | Covered |

The independent review identified four substantive issues, all corrected:
material properties are not exposed in participant metadata; temperature
logs need explicit sensor/reward timing and visibility labels; an extensively
tuned transfer suite is development data; and predictive comfort needs an
explicit proxy for unobserved zone occupancy. The ventilation-efficiency
minus sign was checked against the opened LBNL primary implementation.

No code tests, training, new simulation episodes, submissions, commits or
pushes were performed by this documentation pass. Source access limitations,
missing historical run identity, current pending scores and future field
measurements remain in the collection plan rather than being reported as
completed experiments.
