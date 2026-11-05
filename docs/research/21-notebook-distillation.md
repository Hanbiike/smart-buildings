# Public-notebook distillation triage

Date: 2026-11-04. Status: triage + first ports in flight.
Scope: all 31 public kernels on `neurips-2026-smart-buildings`
(pulled 2026-11-04 to `/tmp/kaggle_nb/`, Kaggle API auth refreshed),
plus `alexchilton/smart-buildings-residual-rl` dataset (updated
2026-11-04, `/tmp/sb_alex/`). Goal: port rule mechanisms worth more
than the 0.0002 floor; feed RL-side findings into plan-19.

## 1. Public board (notebook scores, lower better)

| Score | Notebook | Disposition |
|---|---|---|
| 0.01107 | alexchilton/yes-rl-can (latest) | MINED (see §2); dataset pulled |
| 0.01128 | ndaedzomukhuba/sb-alex-rl | CLOSED: 28-cell copy of alex at the §4c era (no 4d/4e/5/ladder); same v11 weights, nothing over alex-latest |
| 0.01420 | malyshevdanil/probe-zt-best | MINED (see §3); 3 probes ported |
| 0.01489 | jazivxt/facilities-v17 | superseded by v42-v46; skimmed |
| 0.01499 | beraterolelk/adaptive-zt-controller | same code as zt-best minus pressmin; narrative only |
| 0.01546 | jazivxt/facilities-v5 | superseded; skimmed |
| 0.01582 | malyshevdanil/probe-ds-k07 | ancestor diff read (see §3) |
| 0.01619 | jazivxt/facilities-v2 | draft; skipped |
| 0.01853 | malyshevdanil/probe-zone-pi-k10 | ancestor; skipped |
| 0.01853 | chaitanyajamble/competition | EDA + controllers; pending |
| 0.01981-0.02009 | mdmahfujulkarim x3 (hold variants) | weak; skipped |
| 0.03424 | rishabhbafnaiiitd/beginner-friendly | weak; skipped |
| 0.05195-0.06439 | first-analysis baselines | weak; skipped |
| 0.05720 | royalacecat/smartpolicy | pending |
| 0.06398 | prvsiyan/ship-a | CLOSED: 3-stage clock floor-raise, 1-pair evidence (-0.00041 vs weak PID-water), no official score. Same family as alex's ramp (our J1 closed it); our adaptive prestart subsumes the shape |
| 0.09078-0.09162 | avikdas thermodynamic, competition-2 | weak; skipped |
| unscored | ayushrai1114/transferpolicy (HOST) | trivial constant-action template; nothing |
| unscored | danielwuu/cpu-a, cpu-b (724KB each) | base of facilities line; DEEP READ pending |
| unscored | jazivxt/facilities-v42..v46 | CLOSED-read: v42-v46 are BYTE-IDENTICAL code (only headers differ) = danielwuu base + v40 NaN-guard + v41 bounded-RLS online thermal ID + v42 error reserve (shrink band by min(clip(0.5x(err-0.08),0,0.25), 0.25xband)) + 1-param variations (margin .08/.20, recovery, heating-cost). No LB score anywhere; dev-only 576-step numbers untrusted. Queued: error-reserve analogue probe (D4, after D1-D3) |
| unscored | mallakossama/mazenabdou/rushikeshpundkar/mallakossama | EDA/generator notebooks; skipped |
| unscored | shkhan922/baseline | starter-kit runner; skipped |

Our occ9 (0.01236, rank 273) sits behind the two alex-RL entries
ONLY among scored public notebooks. Full public leaderboard
(547 teams, 2026-11-04): Daniel Wu 0.01061 (rank 149, VERIFIED --
header claim corroborated); alex 0.01107 (rank 183); us 0.01236
(rank 273, p50 = 0.01244 -- we are at the median). Top ~150 teams
at 0.0047-0.0106, mostly submitted Nov 1-4: almost certainly the
boiler-billing exploit wave (cf. PR #2: exploit policy -36.0 vs
honest -85.5). Hosts aim final verification at the CORRECTED sim,
so those scores should evaporate. Our lane stays honest.

## 2. alex-latest (30 cells vs our Oct-30 copy's 26)

Version ladder (their table): v10 ramp-rule+250k board 0.01132;
v11 same run @750k picked on 20 held-out GENERATED buildings board
**0.01128** (shipped weights); v12 retrain on 7+720 buildings board
0.01133; v14/15 B1 run pending at pull time (listing now shows
0.01107 -- probably B1 scored after our pull).

- §4d checkpoint judge: 20 generated buildings (4/climate,
  7 days), validated against the board (ranks m3r>m4, gap
  0.000164 vs LB 0.00018); b8 alone misleads (m4 beats m3r on
  b8, loses on LB). Now 60 buildings. Matches our judge-first
  doctrine; replicate LB-agreement validation for our judge.
- 720 generated buildings in the TRAIN pool: unseen improves
  to 750k (0.010868 -> 0.010744); 2700 does not beat 720.
  Direct input to plan-19 P7 pool design (we have
  `tools/gen_transfer.py` + `buildings_xfer/`).
- B1 schedule: linear LR -> 0 + 2048 steps/update removes
  post-peak drift but adds no gain (flat within 0.00007 to 2M).
  We already decay LR; note the batch-size half.
- Checkpoint averaging (4 ckpts): +0.00003. Noise; skip.
- FA1 full authority / thermal storage: RL keeps the tank at
  59-66 C overnight, cheaper warm-up -- but NO NET GAIN
  (0.02510 vs 0.02511, night losses cancel). Skip; watch their
  10M/20M runs.
- Pre-start ramp (RAMP_K=1.0, floor 289 K): target sits
  K x hours-to-morning below the band, meets band at arrival.
  ALREADY PORTED by us (C3 blanket: +0.60% cooling damage; J1
  heating-gated: dead tie -0.00000140). Their base gains
  b8 0.01109 -> 0.01077 from it; on occ9 it is a tie (adaptive
  prestart + econ already capture it). CLOSED, no action.
- "Hash your training pool" (manifest checksum): we have
  `pool_checksum` (E4). Done.

## 3. malyshevdanil line (alex's RL base; our stack's ancestor)

Same class names as our `zone_pi.py` (CoolingHoldPolicy,
ZonePIPolicy, DemandSatPolicy, TunedPIPolicy, ...). zt-best
shipped constants (0.01420): KP=0.1, KI=0.005, HOLD_STEPS=288,
NIGHT_COOL_CEILING=False, ZONE_SHIFT_K=0.6 (EMA 144 steps, full
at |u|=0.2), DEMAND_QUANTILE=0.5 (median SAT), WEEKEND_DROP 6.0K,
PRESSURE_MIN_ALL=True.

Ported to occ9 probes (this round):

- D1 MEDIAN-DEMAND SAT: AHU supply temp from the MEDIAN (not
  mean) of zones' PI demand; outliers stop setting SAT for all.
  Ours uses mean (+mild/deep caps). Probe: `DEMAND_QUANTILE`
  hook (None default) + `probe_medsat05`.
- D2 PER-ZONE ECONOMIC TARGET: per-zone u-EMA (alpha 1/144)
  shifts that zone's target up to 0.6 K toward its cheap edge
  (0.3 K inside band, 0.4 x band cap). Ours has only a global
  OAT-gated down-shift. Probe needs the D5 `_cur_vav` hook
  (1-line base edit) + `probe_ushift06`. ALSO BUILDS the D5
  hook plan-19 P7 needs.
- D3 WEEKEND SETBACK: heating band edge -6 K on weekends
  (day >= 5, not PI-active). Ours has no weekend-specific
  drop. Probe `probe_wknd6` (pure probe-level).

Research knobs they built but did NOT ship (neutral in zt-best;
weak evidence, fallback queue, no probes yet):

- EVENING_STOP_H (evening coast), HEAT_SEQUENCE_SPLIT /
  REHEAT_FULL_AT / HEAT_U_SCALE (dual-maximum heating),
  INTEGRAL_LEAK / INTEGRAL_LIMIT_FRAC (anti-windup variants),
  IDLE_U (stop AHU when all zones |u| < thr), DYN_START
  (dynamic warm-up start; ours `_needed_prestart` is the richer
  analogue -- compare formulas before any probe),
  TARGET_SCHEDULE (24 hourly offsets, CMA-fitted oracle-style).

Also noted: their null-group methodology (z vs perturbation
nudges, |z|<2 noise) addresses switching-chaos; our G8 3x
screen + determinism parity covers the same risk differently.

## 4. danielwuu analytic controller (0.01061, best verified honest)

Archive decoded from cpu-b cell 2 (sha ok): `policy.py` (28 KB) +
2 ensemble .npz (~266 KB each). Architecture (from-scratch
MPC-lite, NOT PI-based):

- Per-zone 6-dim linear thermal model (1, (OAT-T)/20,
  (slow-T)/5, (group-T)/5, mode, power/1000), geometry priors
  from `zone_map`, online residual disturbance observer
  (rate 0.35, clip +-5 K) + model_error tracker. Theta init
  scaled by `response_scale` 0.75. CORRECTION (2026-11-05,
  round-5 angle-2, parent-verified): NO RLS update exists --
  `covariance` is allocated once (`policy.py:229`,
  sha-verified archive) and never referenced; theta is
  read-only after init. "alpha075" is the ResidualPolicy
  blend weight (`round2_runtime.py`), not `response_scale`
  (coincidentally also 0.75). The v41 "bounded-RLS" label
  in the facilities row above refers to jazivxt's own code,
  not danielwuu's.
- "Observation delay = 1" is STANDARD MDP causality (action at
  t affects obs at t+1; verified in
  `environment.py:1651+`: action -> wait_time -> obs), NOT a
  sim bug. Their "compensation" is a 1-step model nowcast of
  the true zone temp from the last applied action. Marginal
  for us (PI + integral + EMA already absorb it).
- `_choose_ahu`: joint grid search over 11 supply-temp
  setpoints (285-305 K + band-adaptive) x 2 recirc options:
  invert implicit model for cool/heat flows, cap heating flow
  at water-coil saturation (discharge <= 308.15 K), fan
  pressure = DESIGN 2000 Pa x fraction^2 (min 200 Pa),
  objective = fan + chiller/3.5 + 0.30 x gas + 65000 x comfort
  penalty (violation + 2 x violation^2). Plus an OFF candidate
  (run=0, min damper) when thermal inertia suffices.
- Water: 345 K target with 24-step heat-recency hysteresis,
  else 310 K standby. cpu-b experiment: conditional 345 -> 350
  K boost (unverified, NOT the scored weights).
- No tank cycling, no billing games visible: plausibly honest.

Distillation verdict: porting the whole MPC-lite is a rewrite,
not a probe. Candidate pieces for later probes (after D1-D3):
(a) OFF-candidate logic (run AHU only if inertia insufficient)
vs our occ-gate/hold; (b) pressure = f(fraction^2) reset shape
vs our pressmin + POPT; (c) heat-recency water hysteresis vs
our pump idle-off. None started yet.

## 5. Pending reads (priority order)
2. facilities-v42..v46 reserve_thermal variations: bounded
   comfort reserve from prediction error. Dev-only 576-step
   numbers (NOT comparable per AGENTS.md) -- mechanisms only,
   no score trust.
3. sb-alex-rl diff vs alex (whose §4d is newer?).
4. prvsiyan/ship-a (morning floor steps -- likely covered by
   our prestart work; quick check).
5. chaitanyajamble/competition (0.01853), royalacecat/smartpolicy.

## 5. Execution log

- 2026-11-04 triage: 31 kernels + alex dataset pulled;
  alex-latest + zt-best line mined; D1-D3 probes
  (medsat/ushift/wknd) ported, train-30 suite launched.
- 2026-11-04 D1-D3 verdict (day note `docs/2026/11/04/`):
  120/120, occ9 0.00878189 (== J4/J6): medsat05
  -0.00002398 (13W/14 ties, median rarely flips binary
  SAT), ushift06 +0.00006532 (denver +0.000248 kills it),
  wknd6 -0.00013008 (7W/23 ties/0L, fairbanks-driven).
  ALL below the 0.0002 floor: no judge screen, no
  promotion. D1-D3 CLOSED.
