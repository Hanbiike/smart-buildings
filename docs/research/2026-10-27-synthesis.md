# Research synthesis: ranked action plan

> Historical snapshot, superseded on 2026-10-28. Read the [current index](README.md),
> [repository analysis](06-repository-analysis.md), [evidence audit](09-evidence-audit.md),
> and [current synthesis](10-control-research-synthesis.md) before using these rankings.
> Several proposed experiments have since failed full episodes. The claims that the
> leader is probably a planner, that train/public comparison identifies the gap's
> cause, and that energy share establishes avoidable waste are unsupported.
> Original text below is retained as a dated research record, not current advice.

**Date:** 2026-10-27. **Scope:** synthesize the 5 completed research tracks
(`docs/research/01`–`05`) into one ranked plan. No code touched; this file is
the only write of this task.

**Situation (verified facts shared by all tracks).** Our best
(`occ_gated4`/`occ_gated7`) scores public 0.01323 vs leader 0.00521 — a 2.5x
gap, and the leader beats even our *train* mean (0.00963), so this is a
**control gap**, not composition. Plant: VAV-reheat-only (no AHU heating coil),
VAV discharge cap 308.15 K, recirc cap 0.70, dampers ≥ 10%, 5-min steps,
reward 0.6/0.2/0.2 with comfort slope ≈ 0.12–0.18/K ≫ energy slope, TOU
peak/night 1.25x (arbitrage dead), chiller COP 3.5, building-level occupancy
only, ≤ 10 zones/AHU, wall-clock ≤ 10x bang-bang per rollout.

---

## 1. Index of the 5 track reports

| # | Report | Question | Bottom line |
|---|---|---|---|
| 1 | [01-competition-intel.md](01-competition-intel.md) (572 lines; 20 FULL + 3 ABS bodies) | What won in this competition family and siblings, by how much? | Forecast-then-optimize won 2/3 CityLearn editions with details; comfort-first wins; pure model-free RL never won; evolve tiny interpretable policies and deploy the artifact. |
| 2 | [02-mpc.md](02-mpc.md) (491 lines; 24 FULL, 10 snippet-fenced in §12) | What does building-MPC literature say for our plant? | Honest field prior 13% commercial bill savings; realistic delta strong-heuristic → centralized MPC is **5–15%**; design around violation-avoidance + SAT/reheat coordination, not arbitrage. Concrete design sketch in its §13. |
| 3 | [03-rule-based-sota.md](03-rule-based-sota.md) (860 lines; all bodies opened) | What do G36 / vendors / open-source do that we don't? | 7 portable unexploited ideas: two-stage SAT reset, dual-max staging, JCI optimal start/stop, rogue-zone weighting, PID-on-max-damper, outdoor feedforward, pseudo-DCV. Full CAN/PARTIAL/CANNOT map in its §4. |
| 4 | [04-rl.md](04-rl.md) (snippet-level: direct fetch blocked, see Unresolved) | Is there an RL path worth our compute? | Model-free loses to tuned RBC/MPC on standardized testbeds; our PPO divergence is the documented structural failure (bang-bang collapse). Bet 1 = planner (MPPI/CEM + shielding); Bet 2 = one disciplined SAC run, never PPO. |
| 5 | [05-gap-forensics.md](05-gap-forensics.md) (363 lines; repo-only, 1136 CSVs) | Where does OUR loss concentrate; what never got probed? | Loss is **96% energy+carbon / 4% comfort**, median occupied comfort regret 0 — we over-condition. Own 02:00–06:00 prestart = 22.5% of total loss. 21 never-probed axes; 8 ranked suspects; #1+#2+#3 plausibly close the 46%. |

Track provenance notes (preserved verbatim in meaning): track 02-mpc read 24
sources in full body with 10 paywalled/bot-blocked/dead sources fenced at
snippet level in its §12, created only `docs/research/02-mpc.md` (491 lines),
and its design sketch targets the verified plant facts above. Track 04-rl
created `docs/research/` plus only `docs/research/04-rl.md`; direct HTTP
fetches were blocked in its sandbox (curl hung), so its claims rest on
web-search result bodies with 7 full-text verification gaps listed in its §10.
Track 05-gap-forensics modified no code (only untracked `docs/research/`),
ran no new sims (existing 1136 CSVs sufficed after window validation), and
used throwaway mining scripts in `/tmp/gapmine*.py` (not committed).

---

## 2. Top-10 actionable ideas, ranked by expected gain × feasibility

Each item maps to our actuators (per-AHU: SAT / static pressure / recirc /
on-off; per-VAV: damper / reheat; HWS: supply-T / pump-dP / on-off) and
observations (per-zone T + setpoints + damper/reheat positions; per-AHU SAT /
pressure / airflow / recirc / request counts / fan; building occupancy +
schedule flags; OAT; hod/dow), with a concrete next experiment.
Expected-gain pools come from track 05; mechanisms from tracks 01–04.

### 1. Decouple damper from reheat (dual-max staging) — heat with min-air × max-temp
- **Why:** our PI couples `damper=|u|`, `reheat=−u` (05 §6.3.1). Optimal
  heating delivers the same heat with minimum airflow at max discharge temp
  (cap 308.15 K); less air cuts ventilation load ~1:1. G36's dual-maximum
  (03 §1.9) ramps the valve at min airflow first, adds air only above 50%
  loop. Suspect #1 pool: 63% occupied energy+carbon, 15–25% recoverable.
- **Actuators/obs:** per-VAV damper + reheat outputs remapped from the
  existing per-zone PI signal; no new sensors (open-loop emulation — DAT
  tracking CANNOT, no DAT sensor, 03 §1.9).
- **Next experiment:** implement valve-first-then-airflow output map in the
  `zone_pi` chain (keep heat-damper-gain 0.85 variant as control); 30-cell
  full-horizon runs vs `occ_gated4`; report comfort K·h, reheat kWh, fan kWh.

### 2. Ride the comfort-band edge instead of band center (deadband riding)
- **Why:** comfort is free anywhere in 294–297 K but our PI holds center
  (05 §6.3.12); median occupied regret is exactly 0 — pure waste. TreeC won
  BOPTEST by holding just above the occupied lower bound (01 §4.2); the
  competition reward's unbounded quadratic tail punishes crossings, not
  edge-riding. `g4_kp03` already proved a 1:4 comfort-for-gas exchange rate
  (−12% on hard cells, 05 §5).
- **Actuators/obs:** per-zone PI target → heat-to-294.0-and-stop /
  cool-to-297.0-and-stop with hysteresis; backoff ∝ gain-forecast error
  (poor-man's robust MPC, 02 §7); uses zone T + setpoints only.
- **Next experiment:** map the comfort–energy frontier: edge-riding variants
  × backoff {0, 0.25, 0.5 K} × KP {0.03, 0.05} on the 5-cell hard grid, then
  30-cell full runs of the winner; stop rule: any variant with tail regret
  beyond −0.05 on >1% of occupied steps is rejected.

### 3. Fix prestart shape: low-flow / below-band / staged-AHU prestart
- **Why:** our own 02:00–06:00 max-flow 4-hour sprint (all AHUs, SAT ~280 K
  against −42 °C OAT on fb1) is **22.5% of total loss**, gas-dominated
  (05 §2.4). Only prestart *timing* was ever tuned; shape never was
  (05 §6.3.2). Naive deeper setback LOSES via catch-up energy (+18%, 05 §5),
  so the catch-up physics — not the setpoint — must change.
- **Actuators/obs:** per-AHU on/off + SAT + pressure during prestart window;
  per-VAV damper caps during recovery (catch-up rate control); OAT + zone T
  for gating. Recirc to max 0.70 during heating sprints (min OA).
- **Next experiment:** (a) low-flow prestart (longer window, capped dampers,
  below-band target) vs current sprint on fb1/fb8 full runs; (b) per-AHU
  prestart (cold-side AHUs only); measure gas $, electricity $, and 06–09
  comfort ramp.

### 4. Run the coded-but-unrun backlog (cheapest information in the repo)
- **Why:** `occ_sat/2/3` (deep-cold saturation restraint — directly targets
  the fb1 prestart pathology), `occ_gated7` (the *submitted* T&R policy, zero
  train CSVs), duty cycling, `occ_coolshift`, `probe_*` headroom probes all
  have zero run dirs (05 §6.2). Feasibility is maximal: the code exists.
- **Actuators/obs:** as-coded; `occ_sat` caps dampers when the heat plant
  saturates (cold-climate gate on OAT).
- **Next experiment:** 30-cell full-horizon runs of `occ_sat` and
  `occ_gated7` first (highest value), then `occ_coolshift` + `probe_tod`;
  this also resolves Unresolved item 05-U2 (g7's loss structure is currently
  inferred, not measured).

### 5. Adaptive optimal start (JCI quadratic + daily adaptation) + conservative optimal stop
- **Why:** ours is fixed-hour prestart with no early release (03 §1.5); G36
  standardizes only the interface, while JCI's published algorithm
  (`OST = C/100·ΔT² + deadtime`, adapted daily on miss/made-early) ports
  directly to our hod/dow + OAT + zone-T obs (03 §§1.5, 2.2). Evening coast
  (stop ~17:30, ride thermal mass while departures ramp 13:00–19:05) is
  nearly free and never tried (05 §6.3.13/15). Pool: arrival/coast hours
  ~10–15%, recoverable 3–8%.
- **Actuators/obs:** AHU/HWS on-off schedule + per-zone setpoint release;
  building occupancy + schedule flags gate adaptation (skip on OAT outside
  design limits, per JCI).
- **Next experiment:** implement JCI optimal start vs fixed prestart on full
  30 cells; then add optimal-stop coasting with a conservative Temp
  Differential (0.6 comfort weight ⇒ coast must be verifiable tail-free);
  ablate start-only vs stop-only.

### 6. Two-stage SAT reset (G36 OAT schedule + T&R-bounded TMax) as the heuristic core
- **Why:** our SAT resets are single-stage demand followers (03 §5.1); G36's
  two-stage form (OAT-scheduled base 12–18 °C over 16–21 °C OAT + TMax
  trimmed up / responded down on zone cooling requests, defaults
  +0.1/−0.2/−0.6 K per 2 min rescaled to our 5-min steps) is the
  practitioner ceiling that ships the reheat-waste prize (fan 1.6–5.7%,
  heating 7.7–33.7%, 02 §8). Self-tune trim/respond rates online
  (request-rate feedback, rogue-zone exclusion) — fixed constants are what
  failed to transfer (01 §8.5). This is also the graceful-degradation floor
  of the MPC slow layer (item 8).
- **Actuators/obs:** per-AHU SAT setpoint; requests from damper + temperature
  proxies (flow-based G36 3/2-rules CANNOT — no per-zone airflow sensor,
  03 §1.6); gate the loop inert below OAT ≈ −6.8 °C (SAT action dead there,
  03 §1.2).
- **Next experiment:** implement G36 two-stage SAT reset + online trim
  self-tuning; compare vs `DemandSat` chain on full 30 cells; add
  importance-weighting / rogue-zone down-weighting (03 §5.4) as a second
  variant. Siemens PID-on-most-open-damper at 90% (03 §5.5) is the
  drop-in alternative arm.

### 7. Occupancy-proportional conditioning: arrival soft-start + explicit comfort budgeting
- **Why:** occupancy ramps 06:00→13:00 yet conditioning is binary full-from-
  06:00; comfort loss scales with occupants, so early/late occupied hours
  are cheap to soften (05 §6.3.13). The measured 1:4 exchange rate (item 2)
  says small *dynamic*, occupancy-weighted excursions buy large savings;
  no policy explicitly budgets regret today (05 §6.3.21).
- **Actuators/obs:** per-zone PI targets scaled by building-occupancy level
  during ramp/coast; per-VAV damper/reheat follow; uses building occupancy +
  hod only (no per-zone occupancy exists).
- **Next experiment:** soft-start variant (conditioning effort ∝ occupancy
  06–09) + evening coast (item 5) vs binary gating; track comfort-regret
  distribution (median must stay 0, 99th percentile bounded) alongside
  energy+carbon.

### 8. Close the greybox/MPC loop: centralized economic MPC per AHU (the leader-shaped bet)
- **Why:** all four literature tracks converge: the 0.00521 leader is most
  likely a planner, not a policy — it beats our train mean, hidden has no
  tails (planners re-optimize per state; they don't memorize tail responses),
  policy tweaks didn't transfer, and MPC ≈ evolved-interpretable ≫
  model-free RL on every physics-grounded benchmark (01 §§4–5, 02, 04 §8.2).
  The portable prize is joint AHU-SAT + per-zone damper/reheat coordination
  each 5 min (heuristics pick SAT by rule and zones fight it with reheat)
  plus violation avoidance ahead of occupancy/weather ramps (02 §13.4).
  Realistic delta over our strong heuristic: 5–15% (02 §11).
- **Actuators/obs:** all actuators jointly optimized; per-zone 1R1C + lumped
  gain model (escalate to 2R2C iff 24-h MAE > 0.3 °C); night/weekend
  parameter ID + daytime Kalman disturbance observer for offset-free
  tracking (building-level occupancy ⇒ per-zone gains unmeasured, 02 §3);
  exact-penalty soft comfort constraints matching 0.6/0.2/0.2; NO
  pre-cooling for arbitrage at 1.25x (02 §§9–10).
- **Next experiment (staged, each shippable):** (a) night-ID + 1R1C fit,
  verify <1 °C RMS 24-h prediction; (b) slow layer first (1-h cadence, 24-h
  horizon, 15-min blocks: SAT schedule + setback depth) — degrades to
  item 6 if stopped here; (c) fast layer (5-min re-solve, 2–4-h horizon,
  MPPI/CEM, horizon 6–12, hard action projection inside the planner);
  (d) validate on held-out buildings AND no-tails subsets with
  occ_gated4/7 + dual-maximum baselines, ablate
  slow-only/fast-only/no-observer/no-adaptation (adaptation was 16 of 26.8
  pts in Cai, 02 §7); keep the explicit-PWA/distilled fallback ready for
  the 10x wall-clock cap (02 §5, 04 Bet 3).

### 9. Break the all-on block: occupied AHU staging + per-AHU schedules
- **Why:** atlanta_3 (9 AHUs) is all-9-on or all-off (partial states <1.5%);
  the chain forces ALL AHUs on in PI hours (05 §2.6). Per-AHU on/off by
  local demand, rotation, or occupied cycling was never measured (duty
  variants coded, only failed ones run, 05 §6.2). Pool: part of suspect #1,
  2–6% (05 §7.6).
- **Actuators/obs:** per-AHU discrete on/off by local request counts (in
  obs) + hysteresis/min-cycle timers (03 §§1.12, 3.3 patterns); cold-side
  AHUs prestart while warm-side ones don't (05 §6.3.6).
- **Next experiment:** per-AHU demand staging with persistence timers vs
  forced-ALL-on on full 30 cells; second arm: occupied duty cycling
  (6-on/1-off micro-cycle, coded-never-run).

### 10. Cooling-side + ventilation optimization: SAT/chiller-blower optimum, pseudo-DCV, night drift
- **Why:** Singapore/Phoenix loss is 72–80% energy (05 §2.2) yet the exact
  colder-SAT(cold-chiller/less-blower) optimum was never computed — only 4
  coarse SAT depths probed (05 §6.3.8); COP 3.5 + fan curves are IN the sim.
  Ventilation-driven recirc (raise recirc in unoccupied/low-occupancy hours)
  is distinct from our economics-driven economizer and never tried
  (03 §5.7); cooling-side night setback (let hot buildings drift up +
  morning pull-down) is inverted today by `NIGHT_COOL_CEILING` (05 §6.3.10).
- **Actuators/obs:** per-AHU SAT (warmest-zone reset: SAT set so warmest
  damper rides ~90%) + recirc scheduled on building occupancy + OAT
  (fixed-dry-bulb high limit; differential/enthalpy CANNOT — no RAT/humidity
  sensors, 03 §1.8); HWS pump-dP reset (G36 T&R for pumps, only
  frac-follow/idle-off tried, 05 §6.3.11) as a follow-on.
- **Next experiment:** (a) compute the chiller/blower SAT optimum in-sim and
  test warmest-zone SAT reset on SG/PHX cells; (b) pseudo-DCV recirc schedule
  vs economics-only econ; (c) cooling night-drift + pull-down on one hot
  cell before full rollout.

### Deliberately NOT recommended (evidence-backed)
- More PPO or any model-free-from-scratch primary bet (structural bang-bang
  collapse on stiff HVAC landscapes, 04 §4; never won CityLearn, 6–60x
  worse on BOPTEST, 01 §§3–4). One disciplined SAC-residual run on top of a
  behavior-cloned occ_gated7, with full normalization and a kill rule, is
  the only sanctioned model-free spend (04 Bet 2).
- TOU arbitrage / pre-cooling for price (1.25x is dead; MPC *lost* 5% vs
  default under flat pricing, 02 §§9–10). Pre-condition only to avoid
  imminent comfort violations.
- Pure offline RL on our own logs (narrow RBC coverage ⇒ distributional
  shift binds; free simulator removes the point, 04 §6).
- Public-board hill-climbing / trusting our transfer suite (predicted −13%,
  hidden scored 0; mirrors CityLearn 2023 Team 2's public→private collapse,
  01 §3.4). Validate on held-out buildings + no-tails subsets.
- Sequential multi-climate fine-tuning (catastrophic forgetting, 01 §5.1);
  unbounded online planning (disqualified by the 10x cap, 01 §1.1).

---

## 3. What remains unresolved (honest list)

Every unresolved item from all 5 tracks, preserved. Prefix = source track.

**Track 01 (competition intel) — unopened bodies:**
- 01-U1. Kaggle forum/kernels: JS-shell only, no shared-solution intel —
  needs browser/API creds.
- 01-U2. BOPTEST official paper full text (T&F paywall).
- 01-U3. Wang et al. 2023 RL-vs-MPC full text (MDPI bot-blocked; direction
  from snippets only).
- 01-U4. Sinergym framework journal paper (paywalled; claims rest on the
  DRL-eval paper).
- 01-U5. BuildSys24 policy-extraction full text (ACM paywall; covered via
  the ICCVW25 summary).
- 01-U6. HOT 150k dataset full text (ACM Cloudflare-walled).
- 01-U7. LBNL Modelica G36 docs + Reliable Controls T&R explainer (not
  fetched).
- 01-U8. CityLearn 2020 official overview + 2023 official overview
  (winner/margins via secondary FULL sources only).

**Track 02 (MPC) — snippet-level numbers, fenced in its §12:**
- 02-U1. Oldewurtel et al. 2012/2014 full bodies (paywalled) — the 16–41%
  figure rests on secondary snippets.
- 02-U2. MDPI Buildings 15(18) 2025 review body (bot-blocked) —
  Blum-40%/Liang-28%/blackbox-8.4% rest on snippets.
- 02-U3. Springer stochastic-occupancy MPC + ScienceDirect
  robust-MPC/VRF-MPC bodies (paywalled) — snippet-level numbers.
- 02-U4. IBPSA/core.ac.uk/HAL MPC papers (bot-blocked/403) — topic-relevant
  but unread.
- 02-U5. Ma et al. VAV MPC (Ma12mpc, Berkeley host dead) — VAV-SAT
  coordination details unverified from that source.

**Track 03 (rule-based SOTA) — unopened, not used as evidence:**
- 03-U1. Taylor originals (T&R paper, dual-max ASHRAE Journal) + Trane
  newsletter ADM-APN078-EN: 403/bot-blocked (mechanisms covered by primary
  sources).
- 03-U2. JCI LIT-12012070 AHU Reset note + v16 G36 CCT modules +
  Premier/TempMaster pages: JS-only portal, index truncated; snippet-level
  only, excluded.
- 03-U3. Trane Tracer Summit + UC400/UC210 manuals: bot-block/404 (VAS
  verified via Buildings.com + G36 deck instead).
- 03-U4. MDPI SAT paper, ACHR econ article, DeepWiki G36: 403/429 (covered
  by primaries).
- 03-U5. openHAB/ThingsBoard/Tasmota/OTGW/EnergyPlus examples: not
  inspected, no claims.

**Track 04 (RL) — full-text verification gaps (sandbox blocked direct fetch):**
- 04-U1. Full PDF/HTML bodies not opened for ANY cited source — all claims
  rest on inspected search snippets; verify before external citation.
- 04-U2. LDLP exact numbers (−4..−26%/−1..−19%) and gated-reward equation
  need full-text check.
- 04-U3. MB2C +8.23% baseline identity; CLUE >30× convergence setting;
  Sinergym reward-table means; SAC 90.7% BOPTEST table; Gnu-RL 6.6% sim
  baseline; CQL-vs-rest margins in 2408.07986 — all need full-text
  verification before external citation. (This is the "7 gaps" set: U2 + six
  items in U3.)

**Track 05 (gap forensics) — repo/measurement gaps:**
- 05-U1. Hidden-eval building/climate composition unknown from repo (only
  train+xfer scenarios visible); suspect sizes are train pools, hidden
  weights may differ.
- 05-U2. occ_gated7 (submitted policy) has zero train CSVs: its loss
  structure is inferred from parent occ_gated4; trim-and-respond effect
  unmeasured on train. (Addressed by action item 4.)
- 05-U3. dsat_fp_02/fp_06 win (+0.00126) rests on n=2 Fairbanks slices;
  counterintuitive sign needs a confirmatory run.
- 05-U4. First-3-day slices include episode-startup transient (cold
  integrals/EMAs, uniform 294 K start); slice-vs-full differences conflate
  transient with weather trend.
- 05-U5. Per-AHU zone count / ≤10-zones-per-AHU validity rule cited in
  briefs not independently verified (metadata not opened).

**Synthesis-level caveats (new, not from any single track):**
- S-U1. The 5–15% MPC-delta prior (02) and the 46%-needed gap (05) are in
  tension: if hidden truly has no tails and our heuristic reaches the
efficiency frontier via items 1–7, the residual MPC win could sit
  at the low end (~5%). Items are ordered so heuristic gains land first and
  de-risk the MPC investment.
- S-U2. Gain pools in §2 are train-pool estimates (05-U1 applies to all of
  them); hidden weights may differ, and 3-day probe % must never be
  extrapolated to full episodes in heating climates (05 §5 warning).
