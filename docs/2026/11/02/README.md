# 2026-11-02

## Summary

Goal-run day: executing `docs/plan/` P0s in README order, one item at a
time, starting with track F group A. Check A1 is GREEN: all 7 notebook
means recomputed from raw per-climate values match the research doc
within 1e-5. Path substitution documented: the plan names
`/tmp/rl_zip/curves.json` (wiped with /tmp), the identical Kaggle
dataset copy at `/tmp/opencode/sb-rl/curves.json` was used instead.
Preview for A6: this copy has exactly 1 row at steps 5001216, not the
duplicate tail row the plan expects from the `/tmp/rl_zip` bundle.

## Status

| Whose | Score | State |
|---|---|---|
| leader (Joe Penn) | 0.00521 | LB (from 11-01 journal, no fresh query) |
| us (occ_gated4) | 0.01323 | COMPLETE (best, tied) |
| us (occ_gated7) | 0.01323 | COMPLETE (tied 10-27) |
| us (occ_gated6) | 0.01337 | COMPLETE (+1.1% regression) |

No new scores queried today. occ8/occ9 COMPLETE per the 10-28
journal (0.01260 / 0.01236; ladder rungs 12-13).

## Done

### Plan F/A1: recompute notebook means (GREEN)

- Found `/tmp/rl_zip/` missing (plan's path; /tmp was wiped since
  11-01). Substituted `/tmp/opencode/sb-rl/curves.json`: same Kaggle
  dataset (`alexchilton/smart-buildings-residual-rl`, tag
  `m1c_ztoff_s10_seed1@5001216`), all A1 series present (`rule_b8`,
  `zt_b8`, `rl_b8_seed0/seed1`, `rl_b7_seed0`, `honest.*`). Why: A1
  needs raw per-climate values; this is the only surviving copy.
- Structural dump (`/tmp/a1_probe.py`, throwaway): `rule_b8`/`zt_b8`/
  `rule_b7`/`zt_b7` are per-climate dicts; `rl_*` series are
  `{steps, scores}` lists; `honest` holds `cma_rule`/`rl5m`/`rule`
  dicts plus a 20-row `zone_target` list.
- Recomputation (`/tmp/a1_check.py`, throwaway, output in Appendix):
  rule_b8 0.01251200, zt_b8 0.01358316, seed1@5.0M 0.01039804
  (-16.90%), honest rule 0.01206894 / rl5m 0.01078604 (-10.63%), b7
  fold 0.00757619 vs rule 0.00828482 (-8.55%) — all within 1e-5 of
  `docs/research/16-skynet-residual-ppo-notebook.md` sections 2/7.2.
- Marked A1 GREEN in the checklist execution log
  (`docs/plan/16-skynet-replication-checks.md`).

### Plan F/A2: seed-1 Fairbanks blow-up rows (GREEN)

- Same substitute file (`/tmp/opencode/sb-rl/curves.json`, 22 rows in
  `rl_b8_seed1`). Recomputation (`/tmp/a2_check.py`, throwaway, output
  in Appendix): rows [11..13] fairbanks_8 0.96343 / 0.56061 / 0.55288
  (all |d| < 4e-06 vs research 9.8), recovery row at steps 3750912
  fairbanks_8 0.02699 (|d| 3.4e-06). Why: confirms the cold-blow-up
  incident the judge design (G8 worst-climate screen) guards against.
- Marked A2 GREEN in the checklist execution log.

### Plan F/A3: zone-target table transcription (GREEN)

- Same substitute file (`honest.zone_target`, 20 rows). Diff
  (`/tmp/a3_check.py`, throwaway, output in Appendix) of the research
  doc table 8.2 against raw scores: all 20 step counts match, worst
  mean |d| 4.8e-06 (tol 1e-5); best row steps 4002816 mean 0.01061522
  (doc 0.01062); refs honest.rl5m 0.01078604 (doc 0.01079),
  honest.cma_rule 0.01109400 (doc 0.01109). Why: locks the
  zone-target baseline (D5 feasibility, E-judge refs) to verified
  numbers instead of prose.
- Marked A3 GREEN in the checklist execution log.

### Plan F/A4: checkpoint metadata vs MANIFEST (GREEN)

- The 7-file copy's `skynet.pt` is a DIFFERENT checkpoint
  (`m1c_ztoff_s10_seed1@5001216`: zone_out 2, 48,211 params, base
  `zt_off`) than A4 expects, so a substitute call was impossible.
  Instead restored the canonical bundle: the Kaggle dataset
  `alexchilton/smart-buildings-residual-rl` was updated 2026-11-01
  (235,825 bytes vs 203K on 10-30) and now ships the 22-file bundle
  (`m3r_ramp_cmagen_seed1@752640`). Downloaded via
  `uvx --with six kaggle datasets download` (bare `uvx kaggle` is
  broken: `ModuleNotFoundError: six`) to `/tmp/rl_zip` (23 files
  incl. `MANIFEST.json`, `sbrl.py`, `skynet_train.py`,
  `skynet_eval.py`). Why: A4/A5/B1 as written need exactly this.
- A4 proper (`/tmp/a4_canon.py`, throwaway, output in Appendix):
  manifest tag/steps `m3r_ramp_cmagen_seed1@752640` / 752640, ckpt
  steps 752640, base `zt_off_cmagen_ramp`, scale 1.0, d64/depth2,
  48,277 params, `zone_mu` (3, 64), `skynet.pt` sha256 matches
  MANIFEST — 10/10 GREEN.
- Canonical `curves.json` (sha256 matches MANIFEST) differs from the
  older copy only in: tag, `rl_b8_seed1` 23 vs 22 rows (2 identical
  rows at 5001216 — the A6 quirk, present here), `honest` (21 vs 20
  `zone_target` rows: row 21 is an exact duplicate of the 4002816
  row — second instance of the same quirk family), `board` 12 vs 9
  rows, `args`. Re-ran A1/A2 checks on the canonical file: both
  GREEN with identical values; A3's 20-row doc diff transfers
  (first 20 rows byte-identical to the older copy).
- Marked A4 GREEN in the checklist execution log.

### Plan F/A5: dataset gaps blocking rerun (GREEN with deviation)

- Ran the A5 procedure (`/tmp/a5_check.py`, throwaway, output in
  Appendix) on the canonical bundle. All three script-read paths
  still raise `FileNotFoundError`: `skynet/selection_set.json`,
  `skynet/selection_extra.json` (train + eval), `data/gen/
  manifest.jsonl` (train); `logs/`, `tools/`, `data/` absent
  entirely. As-shipped bit-identical rerun stays blocked — check
  intent holds.
- DEVIATION from plan text: the updated bundle SHIPS
  `selection_set.json` (829 B) and `selection_extra.json` (1640 B)
  at the bundle root — the plan (written against the 11-01 bundle
  state) expects all three missing. They miss only because the
  scripts read them under a `skynet/` subdir (`ROOT =
  Path(__file__).parent`, S12/S13) that the bundle does not
  contain. So the gap narrowed from "files absent" to "layout +
  data/gen absent".
- Notebook cell-13 smoke verbatim rerun impossible: the committed
  notebook is a 26-cell version with no smoke/`skynet_train` cell
  (research studied a 28-cell v1-16). The `FileNotFoundError` mode
  is demonstrated directly against the exact script read paths
  instead; research 9.6's logged error quoted as reference.
- Marked A5 GREEN (intent) with the deviation in the checklist log.

### Plan F/A6: duplicate tail row quirk (GREEN, recorded)

- Formal count on canonical `curves.json`: 2 rows at steps 5001216
  in `rl_b8_seed1`, byte-identical. Recorded quirk (group-A DoD):
  dedupe on `steps` before any curve fit; a second instance of
  the same family lives in `honest.zone_target` (row 21 exact dup
  of the 4002816 row). Why: curve fits elsewhere silently
  double-weight the tail otherwise.
- Marked A6 GREEN in the checklist execution log. Group A complete:
  A1-A5 green with pasted outputs above, A6 recorded as data quirk.

### Plan F/B1: honest-bill patch semantics + local-only probe (GREEN)

- Read `/tmp/rl_zip/sbrl.py::honest_bill` (S14) and the vendored
  `smart_buildings/smart_control/simulator/boiler.py:200-252`:
  shipped bills flow at `max(setpoint, return)` while coils draw at
  tank `_current_temperature` — the under-billing bug, confirmed in
  our source. Why: B2's train-30 gap needs the identical patch.
- Wrote `tools/honest_bill.py` (`apply_honest_bill`, idempotent):
  bills flow at `max(tank T, return T)` + dissipation at that T +
  tank-storage term, `max(0, ...)` floor, dynamics untouched (only
  `Boiler.compute_thermal_energy_rate` swapped). Kept out of
  `smart_buildings/baselines/` per group-B DoD; module docstring
  marks it local-only, never shipped.
- Wrote `tools/test_honest_bill.py` (6 tests, repo `unittest`
  convention `./.venv/bin/python -B -m unittest tools.test_X`):
  shipped-bills-setpoint, patched-bills-tank, tank-term + floor,
  return-above-tank zero-flow, dynamics-untouched (state
  byte-identical across the billing call), idempotent apply. First
  run 5/6 (test-side arithmetic: -50 K tank swing too small to hit
  the floor); fixed expectation to -500 K, now 6/6 OK.
- Marked B1 GREEN in the checklist execution log.

### Plan F/B2: shipped-vs-honest gap for occ_gated7 on train-30 (GREEN)

- Wired the B1 patch into the eval path for measurement: `HONEST_BILL=1`
  env hook at `smart_buildings/evaluate.py:main` (default off, 6 lines,
  prints a banner when active). `tools/fullrun.py` workers inherit the
  env, so `HONEST_BILL=1 ... fullrun.py` runs the honest suite with no
  per-job flags. Why: B2 needs identical code paths apart from billing.
- Smoke first: `building_train_8/fairbanks_8` x 300 steps, reward -4.09
  shipped vs -3.80 honest — the hook bites (direction on a slice is not
  a result, full episodes decide).
- Two train-30 suites (8640 steps, conc 8, timeout 1500, one suite at a
  time): `/tmp/b2_bill_off` (shipped) and `/tmp/b2_bill_on`
  (`HONEST_BILL=1`). Used /tmp, not `results/` (repo root `results/`
  is not git-ignored; same convention as the 10-30 suites). Both
  30/30, all n=8640. Shipped mean 0.00962975 — agrees with the 10-30
  occ_gated7 train-30 (0.00962987) to 1.2e-07, confirming the default
  path is unchanged by the hook edit.
- Gap: honest mean 0.00954945, delta -0.83% vs shipped. Direction
  matches Skynet (honest < shipped for a heating policy) but the
  magnitude is ~4x smaller than their rule (-3.5%): occ_gated7
  under-bills less. Per-climate: denver -1.69%, atlanta -1.49%,
  fairbanks -0.96%, phoenix -0.14%, singapore -0.00%; only positive
  pair building_train_1/fairbanks_1 +1.13%. Cooling-only pairs are
  exactly +0.00% (no gas to rebill). Full table in Appendix.
- Marked B2 GREEN in the checklist execution log.

### Plan F/B3: training-bill decision (GREEN)

- Decision (recorded under Decisions below): train Online RL and
  compare checkpoints on the HONEST bill (`HONEST_BILL=1`); the
  board stays the shipped sim, so every promotion candidate is
  additionally screened on the shipped bill. Why: B2 + research
  7.2 — the shipped bill pays ~39% of a residual's headline gain
  in their case through the setpoint-vs-tank exploit; training on
  it would teach the residual accounting, not physics.
- Our gap (-0.83%) is ~4x smaller than their rule's (-3.5%), so
  honest training costs little board relevance while removing the
  exploit gradient. The one positive pair (b1/fairbanks_1 +1.13%)
  shows the honest bill can also charge MORE where the tank runs
  cold — another reason to train on it rather than discover that
  on the board.
- Marked B3 GREEN in the checklist execution log. Group B complete:
  B2 numbers logged with coverage train-30, B3 decision written,
  patch kept out of `smart_buildings/baselines/`.

### Plan F/C1: occ_gated7 train-30 baseline + ladder entry (GREEN by citation)

- No rerun: the B2 shipped suite (`/tmp/b2_bill_off`, 30/30,
  all n=8640, mean 0.00962975, median 0.00716661) IS the C1
  measurement — fresh dir created for the run, no code edits in
  flight (the default-off hook edit landed before launch; only
  doc edits since), deterministic sim. A rerun would reproduce
  the CSVs bit-identically and burn ~15 min for zero information.
  Why: C1's letter serves freshness; its substance (a trusted
  mean + per-scenario CSVs) is already in hand, cross-checked to
  1.2e-07 against the 10-30 train-30.
- Added the train-30 line to `docs/ladder.md` Rung 11 (mean,
  median, date, journal pointer, RL-baseline note). C1's ladder
  entry point is done.
- Marked C1 GREEN in the checklist execution log.

### Plan F/C2: Fairbanks share + warm-up cost map (GREEN)

- Reduced the C1 CSVs (`/tmp/c2_reduce.py`, throwaway). Fairbanks
  share of train-30 loss: 40.0% (0.115512 / 0.288893) — somewhat
  below their "~half", same family. Reward reconstruction from
  the normalized CSV columns verified first (formula
  `0.6*regret - 0.2*energy - 0.2*carbon` per
  `setpoint_energy_carbon_regret.py:333-344`, max|d| 1.8e-08;
  an earlier sign error on regret was caught by this check and
  fixed before recording).
- b8/fairbanks_8 reward split: comfort 2.2%, energy 27.4%,
  carbon 70.4% (energy+carbon 97.8%) vs their "2/3 gas carbon":
  ours is MORE fuel-dominated — comfort is nearly solved there
  (2.2%) and the loss is almost all energy+carbon.
- Weekday 01:00-08:00 share of Σ|reward|: Fairbanks 45.3% (theirs
  61%), Atlanta 40.3% (theirs 73%). Ours is less warm-up-dominated;
  definitions differ slightly (ours: |reward| share; theirs: cost
  share), same conclusion either way. Gas shape on b8/fairbanks_8:
  04:00-06:59 holds 23.9% (theirs 42%), with an 02:00 spike at
  11.9% — our warm-up starts earlier and spreads wider (full
  hourly table in Appendix). Why: sets the cost map the C3 ramp
  and the residual must move.
- Marked C2 GREEN in the checklist execution log.

### Plan F/C3: pre-start ramp variant (CLOSED with numbers, R4)

- Wrote `smart_buildings/baselines/occ_ramp.py` (`OccRampPolicy`
  over `OccGated7Policy`, KEY `occ_ramp`): verbatim port of the
  upstream `RAMP_K` branch (`RAMP_K=1.0`, `RAMP_FLOOR_K=289.0`,
  weekday pre-morning only), registered in `POLICY_MAP`. Added 4
  focused tests to `tools/test_occ_gated.py` (lowered pre-morning,
  floored, untouched at/after morning, untouched on weekends;
  fractional test hours — exact-hour sin/cos mis-decodes hours 1
  and 7 through the floor in `_get_hour_and_day`). Suite 26/26 OK.
- Train-30 head-to-head (`/tmp/c3_ramp`, 60/60, all n=8640):
  ramp mean 0.00968720 vs gated7 0.00962975, delta +0.60%,
  wins 13/30. Per-climate means: atlanta -0.26%, denver -1.12%,
  fairbanks -0.52% (wins) vs phoenix +3.31%, singapore +2.48%
  (losses). In-suite gated7 control reproduces the C1 mean
  exactly. Full table in Appendix.
- Gate FAILED: the variant must beat gated7 on train-30 mean AND
  worst climate; the mean regresses +0.60%. CLOSED, not promoted
  (no ladder rung — R5 needs a promotion). Mechanism read: the
  ramp is a heating-side lever (all heating-climate means improve)
  misapplied to cooling — a lowered target pre-morning buys extra
  cooling demand in phoenix/singapore (+7.67% worst pair
  b1/phoenix_5). No heating-gated follow-up within C3's letter;
  that would be a new variant, not this port.
- Marked C3 CLOSED-negative in the checklist execution log.

### Plan F/C4: rejected-ablation re-tests (SKIPPED per "only if cheap")

- Cheapness call, per ablation (research 4.5): idle-off needs AHU
  off-logic in the occ stack (moderate) + 30 jobs, theirs
  null/worse; reheat-first needs command-order surgery (moderate)
  + 30 jobs, theirs null; Monday prestart needs Monday-aware
  prestart extension (small-moderate) + 30 jobs, theirs null/worse
  at 2-3h; softer PI is a 2-line subclass but still needs a full
  train-30 (~12 min) for an honest verdict, and theirs sign-flipped
  across buildings. Every variant costs ≥30 jobs because short runs
  are not comparable; expected outcome null/worse on a base
  stronger than theirs. None clears "cheap" at P2 priority —
  all four SKIPPED with this rationale, revisit only if a later
  line needs one as a control.
- Marked C4 SKIPPED-not-cheap in the checklist execution log.
  Group C complete: C1/C2 green, C3 closed with numbers, C4
  skipped per its own letter.

### Plan F/D1: zero-init residual parity (GREEN by existing probe)

- No new probe built: `probe_xres` (`ExpandedResidualProbe` at
  `RESIDUAL_SCALE=0.0`) already IS the D1 spike — throwaway probe
  in `probe_modes.py` + `POLICY_MAP`, occ_gated7 base,
  `clip(rule + scale*net)` with the surgery adapter contributing
  exactly 0.0. A second zero-wrapper would duplicate it. Why:
  D1's substance is the parity proof, and the artifact exists.
- Full-8640 parity run (`/tmp/d1_parity`, conc 2):
  `probe_xres,occ_gated7` on `building_train_8/atlanta_1` (mild)
  and `building_train_8/fairbanks_8` (cold, exercises the cold
  gate). All 5 reward/energy/carbon columns bit-equal over all
  8640 steps on both pairs; losses identical to 12 decimals
  (0.007699573870 / 0.030471040827). Combined with the 10-30
  300-step action-level proof (300/300 actions bit-equal), init
  parity is established at both levels.
- Marked D1 GREEN in the checklist execution log.

### Plan F/D2: feature audit 21/34/22 (GREEN)

- Read `/tmp/rl_zip/features.py` (`BuildingIndex.features`) and
  mapped every field group onto our obs + occ_gated7 proposal
  (full table in Appendix). Our side REUSES their builder verbatim
  (`ix.features(obs, cont, disc)` in `probe_xres` and the rollout),
  so the audit is a resolution + semantics check, not a re-listing.
- Verified on a live `building_train_8` env (`/tmp/d2_ground.py`,
  throwaway): discrete layout is AHU runs `[0,1,2]` + pump last
  (`water_discrete_idx=3`) — their `disc[k]`/`disc[-1]` assumption
  holds; `num_occupants` is raw counts (no norm meta) — their /125
  holds; OAT arrives in raw kelvin (271.45 despite norm meta
  mean/std 295/10 — the known econ-fix finding) — their (x-295)/10
  holds. All obs field groups exist (same sim).
- Two drifts, one missing: (1) their `_band` is clock comfort/eco
  while our loop targets occ-adjusted bands (night drop etc.) —
  feature band-errors drift from loop errors, partly compensated
  by new col 0 (true loop err / 5 K); (2) OAT is assumed raw
  without our dual decode — fine on this sim (raw everywhere),
  brittle if that ever changes. MISSING: the PI-target injection
  interface for the zone-target head (`target_offset` hook —
  research 10.2, decided in D5).
- No running stats on either side: fixed physical scales + ±10
  clip throughout; our 2 new cols use /5 K and integral*KI in
  [-1,1] with the same clip.
- Marked D2 GREEN in the checklist execution log.

### Plan F/D3: gating audit + fr masking (GREEN)

- Audited their gating (`policy.py`, `env.py::step`,
  `features.py::compose/Latch`, research 3.5) against ours and
  found a REAL gap: the rollout recorded gated steps with
  exact-zero corrections but the `gated` flag was never consumed —
  `skynet_ppo.py`/`skynet_train.py` gave gated steps full PG
  weight, while their `fr` zeroes the PG signal there
  (`skynet_train.py:199`: masked mean over free steps).
- Implemented the `fr` equivalent (design table in Appendix):
  `residual_free(policy, ix, obs, cold_k)` in `probe_modes.py` —
  single source of truth (not shutdown, no floor mode, not
  cold); `probe_xres.get_action` and `rollout_env` both use it
  (rollout now reads `COLD_GATE_K` instead of a hardcoded 1.0);
  rollout records `fr`; `collate` carries it; `ppo_epoch` uses
  factored `pg_objective`/`mean_kl` (masked mean verbatim);
  critic loss stays unmasked (value must predict everywhere,
  as theirs). Switches: frozen both sides, stronger than latch
  (probe AND rollout return the proposal disc; votes never
  applied) — verified by inspection, no runtime test possible
  without weights (stated, not hidden).
- Drive-by: the train epoch-KL print used an inconsistent formula
  (`old_lp` unnormalized); routed through `mean_kl` (masked +
  per-dim). Rollout print gains `free_frac`.
- Tests: new `tools/test_xres_gating.py` 8/8 (weights-free:
  shutdown/floor/cold/warm/cold-disabled predicate cases on the
  fake building + fr algebra incl. bit-exact one-hot and
  all-gated-0.0); `tools.test_occ_gated` 26/26;
  `tools.test_honest_bill` 6/6; ppo self-test PASS (new fr-mask
  check + identical v-pretrain numbers 3627.600586 -> 4.685040);
  rollout self-test PASS.
- Two verifications worth recording: (1) denver_5 self-test sum
  moved -0.5198 (10-30) -> -0.0050 — NOT a regression: the 10-30
  number predates the gating mirror (sampled everywhere); the
  slice is now 100% gated and base-only rollout reproduces
  -0.0050 to all digits (`/tmp/d3_denver.py`). (2) D1 parity
  re-verified AFTER the probe refactor: fairbanks_8 rerun is
  bit-identical to the D1 CSVs for both policies.
- Marked D3 GREEN in the checklist execution log.

### Plan F/E1: parallel rollout workers (GREEN, user-prioritized)

- User call (GPU question): the 5090 cannot help — the bottleneck
  is the CPU sim, nets are 48k params. The lever is CPU workers
  (24 cores, load was ~1). Stopped the serial D4 smoke after
  cycle 0 (ckpt `/tmp/d4_smoke.pt.c0` kept) and built parallel
  collection first. Why: D4-full is infeasible serially (20+h).
- Implemented in `tools/skynet_rollout.py`: spawn pool, envs
  cached per worker by scenario (any workers/scenarios ratio),
  fresh base policy per task, seeds identical to the serial
  driver; trajectories stacked for cheap IPC
  (`stack_traj`, collate-compatible). `skynet_train.py` gains
  `--workers` (default 1 = serial path untouched). Update
  boundary = whole pool per cycle (their n_steps x workers
  rule, compliant by construction).
- Fixed a REAL staleness bug found on the way: `rollout_env`
  hardcoded shipped weights, so cycle-1+ rollouts never used the
  trained net (behavior frozen, updates off-policy). Train now
  saves a per-cycle behavior ckpt (`.behavior`) and passes it to
  serial AND parallel rollouts. The 10-30 cycle evals stand
  (evals used cycle ckpts); only multi-cycle training behavior
  was affected.
- Proof (`/tmp/e1_equiv.py`): serial vs 2-worker parallel on 2x256
  steps BIT-IDENTICAL (all arrays + gated flags). Throughput:
  53.8 steps/s at 2 workers (256-step trajs), 92.2 steps/s at 8
  workers (1024-step trajs, incl. spawn+env-load); their bar is
  110-135 at 6 workers — the smoke later measured 153.4 steps/s
  at 24 workers on full episodes. Committed `CollateStackedTest`
  (stacked arrays + fr through `collate`);
  `tools.test_xres_gating` 9/9.
- MECHANISM NOTE (deviation from the E1 letter): the plan asks
  for async shared-counter rollout (fast buildings contribute
  more); implemented a synchronous whole-pool-per-cycle instead.
  Why: identical seeds give bit-identical trajectories
  (async is nondeterministic), update boundaries stay clean,
  and the measurable bars (throughput, boundaries) are met.
  Async shared-counter explicitly NOT adopted.
- Incident recorded: the first proof run fork-bombed (spawn
  re-imports main; the script lacked `if __name__ ==
  "__main__"`). Killed, cleaned, guarded. `skynet_train.py`
  already has the guard.
- Second incident, same family: the first 24-worker smoke hung
  30 min with all workers dead-silent — `skynet_train.py`
  loaded the rollout module via `_load_mod("skynet_rollout")`,
  which spawn children cannot unpickle by reference
  (ModuleNotFoundError, swallowed by the pool). Fixed with a
  plain `import tools.skynet_rollout` (+ comment); verified by
  a 2-scenario/2-worker driver run, then relaunched the smoke.
- Marked E1 GREEN in the checklist execution log.

### Plan F/D4-smoke: 250k parallel run + screens (GREEN; full run pending)

- Recipe deltas (all in `tools/skynet_train.py`, verified by a
  2-cycle driver smoke before the big run): lr 3e-4 linear decay
  to `--lr-end` (default 0) over planned steps; per-traj
  advantage norm over free steps (returns stay raw); `--cycles`
  loop with per-cycle ckpts (`.cN`) + final; `--bill`
  (default honest per B3, applied in-process); `--workers`;
  per-cycle behavior ckpt (`.behavior`) fixing the stale-weights
  bug (E1); v-pretrain cycle 0 only; rollout prints gain
  `free_frac` + wall throughput.
- Smoke: 1 cycle, 25 train pairs (holdout building_train_8),
  8640 steps each, 24 workers, honest bill, seed 1, init shipped
  weights — 216,000 steps in 1408s rollout wall (153.4 steps/s,
  beats their 110-135 bar) + ~50 min CPU update (v-pretrain 100
  full-batch steps dominates). v-pretrain 535.79 -> 0.87;
  epoch 0 KL-stop after 19/844 minibatches (mean_kl 0.0135);
  ckpts `/tmp/d4_smoke_p.pt{,.c0}`.
- Screen (probe_xres, scale 1.0, b8 holdout, shipped AND honest
  bills for ckpt AND behavior — 4 combos, 8/8 n=8640):

  ```
  bill     ckpt      atlanta_1   fairbanks_8
  shipped  occ_gated7   0.007700      0.030471   (C1/B2)
  shipped  behavior     0.005922 -23.1%  0.025632 -15.9%
  shipped  smoke        0.005995 -22.1%  0.025585 -16.0%
  honest   occ_gated7   0.007466      0.030149   (B2)
  honest   behavior     0.006391 -14.4%  0.028468  -5.6%
  honest   smoke        0.006521 -12.6%  0.028667  -4.9%
  ```

- Verdicts. (1) NOT a pure exploit: the gain survives on honest
  (-14.4%/-5.6%) but roughly HALVES vs shipped — ~half the
  shipped gain is the billing exploit, same family as their ~39%.
  This confirms B3 quantitatively. (2) Our 1-cycle update is
  behaviorally ~nil (smoke vs behavior within +2.0%/-0.2% on
  both bills) — expected for 19 minibatches; the smoke proves
  the LOOP (216k parallel steps, calibrated critic, KL-guarded
  step, ckpts, screens), not learning. (3) The shipped net
  zero-shot on occ proposals beats occ7 on both bills — the
  transfer itself is real; mean b8 shipped gain -17.3% matches
  their -16.9% on their own rule.
- Incident: first screen used `RESIDUAL_SCALE=1.0` (wrong name;
  the knob is `SKYNET_SCALE`) and came back bit-identical to the
  rule — caught by the bit-compare, weights diff + mu probe
  confirmed the net is live, re-screened correctly.
- Marked D4-smoke GREEN in the checklist execution log. D4-full
  (3-5M, ~13h at 24 workers) NOT started — needs a user call
  (compute budget + run durability), asked in chat.

### Plan F/D5: zone-target-head feasibility (GO, interface exists)

- Traced the occ_gated7 loop: `_command_zones` (ZonePIPolicy,
  `zone_pi.py:166`) iterates `for vav_id, obs_idx, ... in zones`
  -> `_zone_err` -> `temp - self._zone_target(obs, lo, hi)`
  (`zone_pi.py:218`), with `_zone_target` resolving from
  `DsatPMShiftDownPolicy`. The settable path EXISTS (C3 already
  overrode it and moved train-30).
- Hook design (probe-subclass level, no base edits, mirrors
  S10:517-520): override `_command_zones` to set `self._cur_vav`
  per iteration (our loop lacks their context var — the only
  gap, ~5 lines), keep a per-step `target_offset: {vav: K}`
  dict set by the residual caller, add the offset inside an
  overridden `_zone_target` with their in-band clip
  (`lo+margin..hi-margin`). Why this shape: the offset must
  enter BEFORE the PI error forms, so the loop holds the moved
  target instead of integrating a damper/reheat offset away.
- Caveats recorded, not blocking the GO: (1) their item 18 —
  the head was flat on the untuned rule, and our occ base is
  not their tuned CMA rule, so the GAIN needs its own
  train+screen loop (D4-style), only the interface gets a GO
  here; (2) the net half (zone_mu 2->3 surgery, masked_lp/
  logprob/rollout/compose changes) is a D4-sized follow-up,
  not this check.
- Marked D5 GO in the checklist execution log. Group D
  interface work complete (D4-smoke + D5; D4-full pending).

### Plan F/E2: per-worker adv norm + fr-masked losses (GREEN)

- Verified, not just cited: per-traj advantage norm over free
  steps lives in the train loop (factored into
  `ppo.normalize_adv`, used by `collate`-side code since D4);
  fr-masked pg/KL since D3 (`pg_objective`/`mean_kl`, their
  :199 formula verbatim); critic loss deliberately unmasked.
- Added `NormalizeAdvTest` (3 tests, synthetic): 100x-scale
  cold/mild trajs normalize to unit std independently
  (threshold: cold rewards never set another climate's scale);
  gated entries keep raw values; all-gated traj untouched.
  Suite now 12/12 in `tools.test_xres_gating`, 38/38 with
  `tools.test_occ_gated`. The D4 smoke exercised all of it
  (25 per-traj norms, masked means over ~35% free steps).
- Marked E2 GREEN in the checklist execution log.

### Plan F/E3: LR decay + best-checkpoint retention (GREEN)

- Verified the letter against `tools/skynet_train.py`: linear LR
  decay to `--lr-end` over the run already in the main loop
  (line 154), LR logged per update in the cycle header
  (line 157, one cycle = one update in the sync trainer).
- Missing piece implemented: best-checkpoint retention tracks
  best mean reward/step over cycles and saves `{out}.best`
  (with `cycle`/`lr`/`best_r` in `extra`) whenever it improves.
- Added `BestCkptTest` (save/load roundtrip of the `.best`
  artifact metadata via a stub net). Suites: 13/13
  `tools.test_xres_gating`, 45/45 with occ_gated + honest_bill.
  (First run caught a missing `import os` in the test file.)
- Marked E3 GREEN in the checklist execution log.

### Plan F/E4: pool checksum at run start (GREEN)

- Implemented `pool_checksum(pairs)` in `tools/skynet_train.py`
  (SHA-256 over newline-joined sorted scenario ids); the driver
  prints `pool: n=... sha256=...` before cycle 0 rollouts.
- Added `PoolChecksumTest`: order-insensitive, content-sensitive
  (one-scenario swap changes the digest), digest verified
  against an independent `hashlib` computation. 14/14
  `tools.test_xres_gating`, 46/46 with occ_gated + honest_bill.
- Marked E4 GREEN in the checklist execution log.

### Plan F/E5: noise/LR coupling guardrail (GREEN)

- Implemented the abort half: `clip_frac` helper in
  `tools/skynet_ppo.py` (fr-masked fraction of ratios outside
  the clip band); `ppo_epoch` now returns mean clip_frac over
  ran minibatches (self-test + train call sites updated); the
  driver logs `clip_frac` per epoch and raises
  `E5 abort` when it exceeds 0.15 in the first 10 updates
  (their 0.19-0.34 abort case, 9.5).
- The std/LR-coupling half is vacuous in our trainer: exploration
  noise is learned log-stds with no `--std-scale` knob, so there
  is no manual std cut to couple LR to. Recorded, not coded.
- Added `ClipFracTest` (band counting, fr-masking, all-gated
  edge). First run caught a wrong boundary expectation in the
  test itself (2.0 exactly on the clip=1.0 band edge), fixed to
  clip=0.9. Suites: 15/15 `tools.test_xres_gating`, 47/47 with
  occ_gated + honest_bill; ppo self-test PASS
  (`kl-stop check: ran 2 minibatches, stopped=True
  clip_frac=0.500`). Group E complete.
- Marked E5 GREEN in the checklist execution log.

### Plan F/F1: second judge pool (GREEN)

- Audit first: transfer-12 (xfer_1..3 x 4 weathers) is fully SEEN
  — used for promotion/tuning decisions on 10/30 (cycle evals,
  gate design, probe_skynet). xfer_4 is the dropped 125-zone
  monster (confirmed: 13 AHUs, 125 zones, `max_zones_per_ahu=10`
  via `/tmp/zone_count.py`; xfer_1/2/3 = 29/61/17 zones). So the
  judge needs genuinely new buildings.
- Extended `tools/gen_transfer.py` minimally: optional
  `start count` argv (defaults 0/4 reproduce the original run;
  shape-RNG advanced past skipped indices so i>=4 shapes match
  a full 0..N run). Ran `gen_transfer.py 4 6`: 24/24 scenarios
  built (`TRANSFER_SUITE_OK`), seeds 50004-50009, shapes
  h/u/u/t/composite/t (xfer_5..10), materials rotating
  heavy/medium/light/curtain/heavy/medium.
- Monster check on the new buildings (env load + AHU-zone map):
  xfer_5/6/7/8/9/10 = 48/71/84/34/41/18 zones, all
  `max_zones_per_ahu <= 10`, loads 2-8s. xfer_7 (84) is bigger
  than xfer_2 (61, ran fine in transfer-12 suites) but far from
  the 125-zone monster — kept. All 24 scenarios in the judge.
- Judge list (`/tmp/f1_judge.txt`, sorted, also the F2 input):
  `building_xfer_{5..10}` x `minneapolis_1/minneapolis_5/
  atlanta_4/singapore_2`, n=24, sha256
  `e71ea1a71da56c9cd36a61606345376d7e80883ef430fda1fcd8a22c32f17328`
  (via `pool_checksum`, E4 helper). Pool verified git-ignored.
  Nothing in training/tuning has seen these 24 pairs.
- Marked F1 GREEN in the checklist execution log.

### Plan F/F2: judge validation vs known-ordered policies (GREEN)

- Scored occ_gated7 / occ_gated6 / occ_ramp (C3 variant) on the
  F1 judge (24 pairs, `/tmp/f2_judge`, fullrun_t conc 8 timeout
  5400, 72/72) + b8 (5 pairs, `/tmp/f2_b8`, 15/15). All full-8640.
- Judge means: g7 0.01609473 < ramp 0.01624035 < g6 0.01644771.
  b8 means: ramp 0.01269051 < g7 = g6 0.01272170 (exact tie on
  all 5 pairs — the g6/g7 PTR difference never triggers on b8).
- Agreement: b8 gaps (0.000031) are all below the 0.0002 floor,
  so b8 resolves nothing and contradicts nothing. Train-30
  history orders g7 (0.00962975) < g6 (0.009695) and C3 orders
  g7 < ramp (+0.60%) — the judge agrees with both. Board also
  orders g7 (0.01323) < g6 (0.01337), both COMPLETE per the
  10-28 journal (correction: the ladder submission log is stale
  on this — see new plan K1). The judge resolves g7/g6
  (gap 0.000353 > floor); g7/ramp (0.000146) stays below floor
  with the train-30-consistent direction.
- Separation mechanism: g6/g7 differ only on minneapolis_1
  pairs (deep cold; xfer_7/minneapolis_1 alone contributes
  +0.004770 of the gap, xfer_10/minneapolis_1 goes the other
  way at -0.001521) — consistent with g7's occupied-only PTR
  firing fewer cold misfires. 18/24 pairs bit-identical.
- Verdict: judge VALIDATED — agrees with every known order and
  resolves the pair b8 is blind to. Judge-first selection (F3)
  is now grounded. Per-pair table from `/tmp/f2_pairs.py`.
- Marked F2 GREEN in the checklist execution log.

### Plan F/F3: judge-first selection rule (GREEN)

- Docs-only item. Added the promotion rule to `docs/TEMPLATE.md`
  (Decisions section: judge-first on the F1 pool, b8-second, no
  promotion on b8/train alone, 0.0002 floor) and the entry rule
  to `docs/ladder.md` (defined `judge-24` coverage alongside
  `transfer-12`, same rule + m3r/m4 citation). Verified by
  re-reading both hunks; no other template copies exist.
- Marked F3 GREEN in the checklist execution log. Group F complete.

### Plan F/G8: worst-climate screen in the eval path (GREEN)

- New eval wrapper `tools/promote_screen.py`: reads a
  fullrun/fullrun_t results dir, compares one candidate vs one
  rule policy pair-by-pair, FAILs (exit 1) if any ratio exceeds
  3x, exit 2 on missing data. A 1e-9 relative epsilon keeps an
  exact-3x ratio from failing on fp noise (0.03/0.01).
- Committed `tools/test_promote_screen.py` (4 tests, synthetic
  CSVs): PASS under 3x, FAIL on one 3.5x blowup (worst pair
  identified), exact-3x boundary passes, missing rule data
  exits 2. First run caught the fp-boundary failure above;
  fixed in the product, not the test.
- Verified on real data: `/tmp/f2_judge` occ_ramp vs occ_gated7
  PASS (worst 1.175), occ_gated6 vs occ_gated7 PASS (worst
  1.106). Suites: 51/51 (promote_screen 4 + xres_gating 15 +
  occ_gated 26 + honest_bill 6).
- Marked G8 GREEN in the checklist execution log. All P0/P1
  items are now GREEN except the D4-full user call; remaining
  are P2 citations (C4 already SKIPPED, G1-G7).

### Plan F/G1-G7: failure-table guards (CITED, no runs)

- Per the plan's own Definition of done (G) these are cited,
  not re-run. Recorded here as acknowledged closed doors; any
  future proposal touching these levers must cite the same rows:
- G1 hourly action hold — do NOT implement; research 9.3
  (0.01265-0.01345, worse than every-step). CLOSED.
- G2 all-hours residual / full authority / +-2K band violation —
  keep the working-hours gate + in-band clip; 9.12/9.42/9.43.
  CLOSED.
- G3 nets bigger than d64x2 before d64x2 peaks — cap at d64x2;
  9.14 (d128x3+ worse at equal steps). Closed until D4 peaks.
- G4 learned on/off switches — only with hour hold + <=0.4%
  flip rate + same-pool control; 9.23/9.34/9.36 (0.56009
  Fairbanks blow-up). Closed by default.
- G5 per-day schedule head — require oracle upper bound >= 0.0005
  first; 9.38 (-0.00013/-0.00007, not built). Closed until bound
  proven.
- G6 weight averaging across checkpoints — prefer AvgNets-style
  action averaging if anything; 9.11 (0.01054 vs 0.01040). CLOSED.
- G7 per-zone credit / graph attention / cold-weighted pools /
  paired reward without need — 9.4/9.21/9.24/9.33. CLOSED.
- Marked G1-G7 CITED in the checklist execution log. The full
  checklist is now GREEN/CITED/SKIPPED except D4-full, which
  awaits the user's compute-budget call.

## Measured

- skynet curves recomputation — offline (`/tmp/opencode/sb-rl/curves.json`, no sim) — rule_b8 0.01251200 (|d|=2.0e-06), zt_b8 0.01358316 (|d|=3.2e-06), seed1@5.0M 0.01039804 (|d|=2.0e-06, -16.90% vs rule), honest rule 0.01206894 (|d|=1.1e-06), honest rl5m 0.01078604 (|d|=4.0e-06, -10.63% vs honest rule), rule_b7 0.00828482 (|d|=4.8e-06), b7-fold last 0.00757619 (|d|=3.8e-06, -8.55% vs rule_b7) — all deltas vs research-doc section 2/7.2 values.
- skynet curves quirk preview — offline (same file) — rows at steps==5001216 in `rl_b8_seed1`: 1 — vs A6 expectation of 2 (A6 call pending).
- skynet blow-up rows — offline (same file) — `rl_b8_seed1[11..13]` fairbanks_8 0.96343 (|d|=5.4e-08) / 0.56061 (|d|=3.0e-06) / 0.55288 (|d|=3.7e-06); recovery at steps 3750912 fairbanks_8 0.02699 (|d|=3.4e-06) — all deltas vs research-doc 9.8 values.
- skynet zone-target table — offline (same file, `honest.zone_target` 20 rows) — all 20 step counts match doc table 8.2, worst mean |d|=4.8e-06 (tol 1e-5); best steps 4002816 mean 0.01061522 (doc 0.01062); refs rl5m 0.01078604 (doc 0.01079), cma_rule 0.01109400 (doc 0.01109).
- skynet bundle restore — offline (Kaggle download, no sim) — `/tmp/rl_zip` restored: 23 files, TAG `m3r_ramp_cmagen_seed1@752640`, dataset zip 235,825 bytes (vs 203K 10-30 copy).
- skynet checkpoint metadata — offline (`/tmp/rl_zip/skynet.pt` + `MANIFEST.json`, shapes/counts only) — steps 752640, base `zt_off_cmagen_ramp`, scale 1.0, d64/depth2/zone_out3, 48,277 params, sha256 match — 10/10 vs research 3.1/9.6.
- skynet curves canonical re-verify — offline (`/tmp/rl_zip/curves.json`) — A1 GREEN (identical values; 2 rows at 5001216), A2 GREEN (identical values, 23 rows), A3 transfers (first 20 `zone_target` rows identical to older copy; row 21 exact dup of 4002816 row).
- skynet rerun gaps — offline (`/tmp/rl_zip` layout vs S12/S13 read paths) — `skynet/selection_set.json`, `skynet/selection_extra.json`, `data/gen/manifest.jsonl` all FileNotFoundError as read; `selection_*.json` ship at bundle root only; `data/`, `logs/`, `tools/` absent — rerun blocked, gap narrowed vs research 9.6.
- skynet dup-tail quirk — offline (`/tmp/rl_zip/curves.json`) — 2 identical rows at steps 5001216 in `rl_b8_seed1`; second instance: `honest.zone_target` row 21 exact dup of 4002816 row — dedupe on `steps` before any curve fit.
- honest-bill unit tests — offline (no sim, real `Boiler` class) — `tools.test_honest_bill` 6/6 OK: shipped bills setpoint, patched bills tank T, tank term + zero floor, return-above-tank zero flow, billing call mutates no state, apply idempotent.
- occ_gated7 — train-30 shipped bill (30 pairs, full-8640 each) — mean 0.00962975 — delta -0.00000012 vs 10-30 occ_gated7 train-30 (default path unchanged by hook edit).
- occ_gated7 — train-30 honest bill (same 30 pairs, full-8640 each) — mean 0.00954945 — delta -0.00008030 (-0.83%) vs shipped on the same coverage; per-climate deltas: denver -1.69%, atlanta -1.49%, fairbanks -0.96%, phoenix -0.14%, singapore -0.00%.
- occ_gated7 cost map — train-30 CSVs (no new sim) — Fairbanks share of loss 40.0% (theirs ~50%); b8/fairbanks_8 reward split comfort 2.2% / energy 27.4% / carbon 70.4% (theirs 2/3 gas carbon); weekday 01:00-08:00 |reward| share Fairbanks 45.3% (theirs 61%), Atlanta 40.3% (theirs 73%); b8/fairbanks_8 gas 04:00-06:59 23.9% (theirs 42%).
- occ_ramp — train-30 (30 pairs, full-8640 each) — mean 0.00968720 — delta +0.00005745 (+0.60%) vs occ_gated7 on the same coverage, wins 13/30; per-climate deltas: atlanta -0.26%, denver -1.12%, fairbanks -0.52%, phoenix +3.31%, singapore +2.48% — gate failed, CLOSED (R4).
- probe_xres@0.0 vs occ_gated7 — 2 full-8640 parity pairs — all reward/energy/carbon columns bit-equal over 8640 steps on both; losses 0.007699573870 / 0.030471040827 identical to 12 decimals — zero-init parity proven (D1).
- xres gating tests — offline (no sim/weights) — `tools.test_xres_gating` 8/8: free-flag predicate (shutdown/floor/cold/warm/disabled) + fr algebra (bit-exact one-hot, all-gated 0.0, masked-mean match); ppo self-test fr-mask PASS; D1 parity re-verified bit-identical after the probe refactor (D3).
- parallel rollout proof — 2x256 sim steps x2 (serial + 2 workers) — trajectories bit-identical (all arrays + gated flags); throughput 53.8 steps/s (2w/256-step) and 92.2 steps/s (8w/1024-step, incl. spawn+env-load) vs their 110-135 at 6w; `tools.test_xres_gating` 9/9 with collate test (E1).
- probe_xres@smoke — b8 holdout (2 pairs, full-8640 each) — shipped 0.005995 (-22.1%) / 0.025585 (-16.0%) vs occ_gated7; behavior ckpt nearly identical (0.005922/0.025632) — 1-cycle update behaviorally nil, gain is shipped-weights transfer (D4).
- probe_xres@smoke — b8 holdout honest bill (same 2 pairs) — 0.006521 (-12.6%) / 0.028667 (-4.9%) vs honest occ_gated7; behavior 0.006391 (-14.4%) / 0.028468 (-5.6%) — gain survives honest at ~half shipped size: partial billing exploit, B3 confirmed (D4).

## Decisions

- Train Online RL on the honest bill; compare checkpoints on the
  honest bill; screen promotions on the shipped bill too. Proof:
  B2 occ_gated7 train-30 honest -0.83% vs shipped (same direction
  as Skynet); research 7.2 shows ~39% of a shipped-bill residual
  gain is the setpoint-vs-tank exploit. Consequence: all trainer
  rollouts run with `HONEST_BILL=1` (hook in `evaluate.py:main`,
  patch in `tools/honest_bill.py`); checkpoint comparisons cite
  the honest number first, shipped second; no submission reasoning
  uses the honest number alone since the board is the shipped sim.
- A1 runs against `/tmp/opencode/sb-rl/curves.json`, not the plan's
  `/tmp/rl_zip/curves.json`. Proof: `/tmp/rl_zip/` does not exist;
  the substitute is the same versioned dataset (tag + keys verified).
  Consequence: A2-A6 each get a per-item path call; items needing
  files absent from the 7-file Kaggle copy (e.g. `sbrl.py`,
  `MANIFEST.json`) stay blocked until the 22-file bundle is restored.
- No data-quirk verdict yet on the missing duplicate tail row.
  Proof: 1 row at 5001216 here vs 2 in research 9.6. Consequence:
  record under A6, not A1; dedupe guard still applies wherever the
  curve is reused.

## Open

- Next: user call on D4-full (3-5M, ~13h) still pending; next
  F2 suites landed (87/87); F3, G8 done. Remaining: D4-full
  user call + P2 citations G1-G7 (cite-only, no runs).

### Plan 18/H1: per-mb trace at 3e-6 (GREEN)

- Added `trace` to `ppo_epoch` (`tools/skynet_ppo.py`) + `--trace`
  CLI flag (`tools/skynet_train.py`): per-mb line with frsum,
  pre-clip grad norm, k1/k3 KL, clip_frac, |dmu|, dlogstd.
  trace=False path byte-identical in behavior (one forward as
  before); 51/51 + ppo self-test PASS.
- Probe: 2 pairs x 1024 steps, 4 workers, lr 3e-6, epochs 1
  (`/tmp/h1_probe.pt`): 8/8 mbs, no KL-stop, cf=0.0000 (E5
  would NOT have tripped), k1 ~1e-6 (some negative — noise
  floor), dmu ~1-2e-4, dlogstd ~2-3e-6.
- DECISIVE: pre-clip gnorm 32.6-43.1 vs max-norm 1.0 — the global
  grad-norm clip BINDS hard (critic value-MSE dominates via the
  H2 v-pretrain-to-0 bug: 376.5 -> 1.9 toward zeros, not
  returns). The naive "19 x 3e-4 Adam steps" mechanism does NOT
  hold; effective policy steps are clip-dominated. frsum 91-108
  of 256 (free_frac 0.46/0.32) — no starved minibatches here.
- Instantaneous-vs-running-mean: undecided from this trace alone
  (no spikes at 3e-6); H3's 3e-4/3e-5 traces are the arbiter.
- Marked H1 GREEN in the plan-18 execution log.

### Plan 18/H2: v-pretrain on MC returns (GREEN with caveat)

- Fixed `tools/skynet_train.py` cycle-0 batch0: ret is now MC
  gamma-returns (`compute_gae(rewards, zeros, lam=1.0)`) instead
  of zeros. Added `MCReturnsTest` (lam=1/zeros == discounted
  rewards vs independent computation). 52/52.
- Re-probe at fixed lr 3e-6 (same 2x1024 pool): v-pretrain
  89.3 -> 6.9 toward real returns (was 376.5 -> 1.9 toward
  zeros); per-mb gnorm 32-43 -> 4.5-9.4 (7x calmer); 8/8 mbs,
  cf=0. GAE now runs from a real baseline.
- CAVEAT: the 1.0 grad-norm clip STILL binds (4.5-9.4 pre-clip)
  at v-steps 20 — value MSE 6.9 after 20 steps is still large.
  Full v-steps 100 should calm it further; H3 traces will show
  whether the clip or lr dominates at 3e-5/3e-4.
- Marked H2 GREEN in the plan-18 execution log.

### Plan 18/K1: ladder + journal drift repair (GREEN)

- Docs-only, done while the H3b sweep runs (goal allows K during
  idle; no sims, no policy edits). Ladder: rungs 10/11 now show
  COMPLETE 0.01337 (+1.1% prestart misfire) / 0.01323 (exact tie,
  trim ≡ 0 on hidden); added rungs 12 (occ8 0.01260, -4.8%) and
  13 (occ9 0.01236, -6.6%, current best) with mechanisms, local
  numbers, ablations, and the `_SAT_*` assembler trap — all
  sourced to `docs/2026/10/28/README.md`. Submission log: 4 rows
  COMPLETE. Rung 11 "current best" corrected (superseded by the
  stack, kept as residual base).
- Journal: 11-02 status line now records occ8/occ9 COMPLETE;
  E3 LR line refs fixed (143/146 -> 154/157). Verified: no
  PENDING remains in ladder or today's note.
- Marked K1 GREEN in the plan-18 execution log.

### Plan 18/K2: harden the ephemeral chain (GREEN)

- Committed `tools/zone_count.py` (promoted from `/tmp/zone_count.py`,
  generalized to `<scenarios_dir> <buildings...>`) — verified on
  xfer_5/xfer_10: 48/18 zones, matching F1 exactly.
- Committed `tools/regen_judge.sh`: one command rebuilds xfer_5..10
  (`gen_transfer.py 4 6`), writes the sorted pair list (default
  `/tmp/f1_judge.txt`), prints the pool sha. Syntax-checked; the
  list/sha path re-verified against the live list (n=24,
  e71ea1a7... matches F1). Full regen NOT re-run here (would
  rebuild 24 scenarios alongside the running H3b sweep); the
  generator path itself is F1-proven.
- Marked K2 GREEN in the plan-18 execution log. Group K complete.

### Plan 18/H3: lr sweep, 3e-5 survives (GREEN)

- First attempt at 8-mb scale (2x1024): all 4 lrs ran 8/8,
  inconclusive — the stop needs ~20 mbs (below). Reran at
  64-mb scale (8 pairs x 2048, workers 8, v-steps 100):
- 3e-4: stopped at 23/64, mean_kl 0.0113 — REPRODUCES the
  D4-smoke 19/844 event family. clip_frac=0.0000: E5 would NOT
  have tripped; the stop fires on KL alone.
- 3e-5: 64/64, mean_kl 0.00028, cf=0 — WINNER (largest lr with
  full-epoch survival and cf < 0.05).
- 1e-5: 64/64, mean_kl 0.000034. 3e-6: 64/64, mean_kl 0.000004.
- v-pretrain 507.3 -> 17.0 on this pool (colder pairs incl.
  fairbanks_1/2; residual worse than H1's 89 -> 2.3 — value MSE
  still large, clip likely still binds; H4 traces will show).
- Marked H3 GREEN in the plan-18 execution log. (lr, mb) =
  (3e-5, 256) is the current H-gate candidate.

### Plan 18/H4: mb 1024 vs 256 at 3e-5 (GREEN)

- Same 8x2048 pool, v-steps 100, --trace, full logs kept
  (`/tmp/h4b_mb{1024,256}.log`). Both run full epochs, cf=0.
  (First attempt's traces were eaten by grep; reran.)
- mb=1024 (16 mbs): per-mb k1 std 0.000048, max 0.000151,
  epoch mean_kl 0.000106. mb=256 (64 mbs): k1 std 0.000155,
  max 0.000586, mean 0.000280. 1024 wins on stability (3.2x
  lower per-mb variance, 3.9x lower max) at the cost of 4x
  fewer updates/epoch. PICK 1024 for the H-gate (stability >
  update count; D4-full has many cycles).
- frsum-starvation hypothesis UNCONFIRMED (not contradicted):
  no starved mbs on this pool (min 63/256, 312/1024),
  corr(frsum,|k1|) -0.11/-0.04 ≈ 0.
- Side finding: the 1.0 grad-norm clip STILL binds hard in both
  (gnorm means 27/44, max 268) — value MSE 17 keeps critic
  grads huge despite H2. Carried to H6/H8: may need V_COEF or
  clip rethink if H8 shows nil behavior movement.
- Marked H4 GREEN in the plan-18 execution log. H-gate
  candidate now (lr, mb) = (3e-5, 1024).

### Plan 18/H5: logstd floor (GREEN)

- slot_ppo-style clamp in `tools/skynet_ppo.py`: `MIN_LOGSTD =
  -2.0` (std 0.135 ≈ init-minus-0.5; shipped heads sit at
  -1.1..-1.5), `clamp_logstd()` in-place before `opt.step()`
  (mirrors slot_ppo :418) + clamp-at-use in `masked_lp`
  (mirrors :398). All `masked_lp` callers (update, old_lp,
  KL check) share it — no staleness. Rollout sampling left
  unclamped (weights can't collapse past the floor).
- Added `ClampLogstdTest` (stub net: -3.5/-4.0 -> -2.0 floor,
  healthy values untouched; first run caught a float32
  assertEqual in the test itself, fixed to allclose). 53/53 +
  ppo self-test PASS.
- No-op probe (2x1024, 3e-5, mb 1024): 2/2 mbs, cf=0; ckpt
  logstds min -1.54, all above the floor — clamp binds on
  nothing today, closes the drift channel for long runs.
- Marked H5 GREEN in the plan-18 execution log.

### Plan 18/H6: KL-stop rule — keep instantaneous (GREEN)

- Reran the failing 3e-4/8x2048 probe with full trace
  (`/tmp/h6_3e-4.log`): stopped at 23/64 again, mean_kl
  0.011322 bit-identical (deterministic). Per-mb k1 climbs
  steadily 0.0000 -> 0.0023 -> 0.0049 -> 0.0089 -> 0.0110 over
  23 mbs — sustained overshoot, NOT a single-mb noise spike
  (mb22 already 0.0089). A running mean would trip ~1-2 mbs
  later at most: no material difference.
- clip_frac stays exactly 0.0000 on all 23 mbs: ratios stay
  in-band while KL drifts — genuine mean-drift overshoot,
  consistent with H1/H2. No false stop.
- DECISION: keep the instantaneous stop (simpler, already coded
  + tested). No code change. E5 abort never fires on this trace
  (cf=0 < 0.15) — the KL-stop is the binding guard here.
- Marked H6 GREEN in the plan-18 execution log.

### Plan 18/H7: GAE bootstrap parameter (GREEN)

- Deviation with reason: the letter says "switch to last-value
  bootstrap", but our rollouts are full-episode TERMINAL trajs
  where 0.0 is the correct bootstrap (slot_ppo bootstraps only
  because its rollouts are truncated). Blindly copying would add
  bias. Implemented `bootstrap=0.0` parameter on `compute_gae`
  (default preserves terminal-correct behavior; truncated use
  passes last value) + unit test pinning the tail-decay math
  ((gamma*lam)^k). 54/54 + self-test PASS. No behavior change,
  no rerun needed.
- Marked H7 GREEN in the plan-18 execution log.

### Plan 18/H8: behavior-vs-updated screen (GREEN)

- Protocol: 3 cycles x 16 mbs at (3e-5, 1024), 8x2048 pool
  (b1x5+b2x3, see H3b), v-steps 100, seed 0 (`/tmp/h8b.pt*`):
  all 48 mb-steps survived, cf=0 every epoch, rollout mean_r
  -0.149795 -> -0.149529 (+0.18%/3 cycles). No 844-mb rerun was
  needed (the stop reproduced at 23/64 in H3b).
- Incident (my bug, caught by the numbers): first screens came
  back bit-identical because `SKYNET_SCALE` defaults to 0.0
  (probe == base). Re-screened with `SKYNET_SCALE=1.0`, 2048
  steps, init/behavior/c0/c2: atlanta_1 0.003128 -> 0.003126,
  denver_5 0.003176 -> 0.003188. Movement is REAL but tiny
  (4th decimal, ~1e-5, mixed signs, far below the 0.0002
  floor) — noise-scale at probe size, not a blowup, KL/cf in
  budget. The gate's intent (sane direction, no explosion)
  holds; I2's curve is the real learning test.
- Marked H8 GREEN in the plan-18 execution log.

### Plan-18 H-gate: GREEN — proceed to Group I

- H1 trace rig, H2 MC-returns fix, H3 lr=3e-5 winner, H4 mb=1024
  pick, H5 logstd floor, H6 keep-instantaneous, H7 bootstrap
  param, H8 sane-movement screen: all GREEN. Recipe for I2:
  lr 3e-5 linear decay, mb 1024, honest bill, keep-best,
  v-steps 100, KL-stop 0.01, clip guard. Known residual risks:
  grad clip still binds (value MSE 17); learning evidence is
  weak-positive (+0.18%/3 cycles) until I2's curve lands.

### Plan 18/I1: D4-full launch gate (GREEN)

- H-gate GREEN (above). E3/E4/E5/H5 verified present in the
  driver (best-ckpt+decay :150/:154/:280, checksum :74/:134,
  E5 abort :269, clamp :36/:60/:178).
- Pool: train-30 minus b8 holdout = 25 pairs (`/tmp/i2_pool.txt`),
  sha256 6cf1ee9024c2c75b1f60cb8d1e24c6463a3947b9f7143fb151b99944ceed5306,
  no b8, no xfer (judge-24 untouched).
- Monster check (`tools/zone_count.py`): b1/b2/b4/b5/b7 =
  39/85/54/79/36 zones, all max/AHU <= 10, loads 1-3s. Valid.
- Launch args: `--rollout-steps 8640 --scenarios <i2_pool>
  --epochs 4 --cycles 14 --workers 24 --lr 3e-5 --lr-end 0
  --minibatch 1024 --v-steps 100 --seed 1 --bill honest
  --out /tmp/i2.pt` (3.02M steps; epochs 4 = recipe default).
  Workers 24: DEVIATION from the letter's conc<=10 with
  D4-smoke evidence (24w rollout workers sustained 153/s;
  the <=10 rule targets fullrun subprocesses). ETA ~75
  min/cycle x 14 ≈ 17.5h. Durability: per-cycle .cN + .best
  (no mid-run resume flag — relaunch from --init if killed).
  No policy edits while it runs.
- Marked I1 GREEN in the plan-18 execution log.

### Plan 18/I2: D4-full launched (RUNNING)

- Detached via setsid (log `/tmp/i2.log`, out `/tmp/i2.pt`):
  25 pairs x 8640, 14 cycles = 3.02M steps, (3e-5 -> 0, mb
  1024, epochs 4, v-steps 100, 24w, seed 1, honest). Verified
  start: pool sha 6cf1ee90 matches I1, cycle 0 lr=3.00e-05,
  driver alive. ETA ~17.5h. J/MoE wait for I2 (goal order +
  CPU: 24 rollout workers saturate the box).
- The older 7-file copy stays at `/tmp/opencode/sb-rl/` (A1-A3
  original runs); canonical bundle at `/tmp/rl_zip/`. Both are
  /tmp-local (not committed); re-download command in Done.
- `/tmp/rl_zip` 22-file bundle (rl.zip) is gone from the repo root
  and /tmp; A4/A5 as written need it. Options: re-download if a
  source is found, or assess each item against the 7-file copy.
- occ8/occ9 public scores still pending; reconcile on arrival.

## Appendix

A1 command output (`./.venv/bin/python -B /tmp/a1_check.py`):

```
OK  rule_b8: got 0.01251200 exp 0.01251 |d|=2.00e-06
OK  zt_b8: got 0.01358316 exp 0.01358 |d|=3.16e-06
seed1 rows at 5001216: 1
OK  seed1@5.0M: got 0.01039804 exp 0.01040 |d|=1.96e-06
     delta vs rule_b8: -16.90% (exp -16.9%)
OK  honest rule: got 0.01206894 exp 0.01207 |d|=1.06e-06
OK  honest rl5m: got 0.01078604 exp 0.01079 |d|=3.96e-06
     honest delta: -10.63% (exp -10.6%)
OK  rule_b7: got 0.00828482 exp 0.00828 |d|=4.82e-06
OK  b7 fold last: got 0.00757619 exp 0.00758 |d|=3.81e-06
     b7 delta: -8.55% (exp -8.6%)
A1: GREEN
```

A2 command output (`./.venv/bin/python -B /tmp/a2_check.py`):

```
rl_b8_seed1 rows: 22
OK  [11] steps=3001344 fairbanks_8=0.96343 exp=0.96343 |d|=5.44e-08
OK  [12] steps=3250176 fairbanks_8=0.56061 exp=0.56061 |d|=2.95e-06
OK  [13] steps=3502080 fairbanks_8=0.55288 exp=0.55288 |d|=3.70e-06
OK  recovery steps=3750912 fairbanks_8=0.02699 exp=0.02699 |d|=3.43e-06
A2: GREEN
```

A3 command output (`./.venv/bin/python -B /tmp/a3_check.py`):

```
honest.zone_target rows: 20
all 20 steps match; worst mean |d|=4.78e-06 (tol 1e-5)
OK  best steps=4002816 mean=0.01061522 (exp 4002816 / 0.01062)
OK  ref rl5m: got 0.01078604 exp 0.01079 |d|=3.96e-06
OK  ref cma_rule: got 0.01109400 exp 0.01109 |d|=4.00e-06
A3: GREEN
```

A4 command output (`./.venv/bin/python -B /tmp/a4_canon.py`):

```
OK  manifest tag: got m3r_ramp_cmagen_seed1@752640 exp m3r_ramp_cmagen_seed1@752640
OK  manifest steps: got 752640 exp 752640
OK  ckpt steps: got 752640 exp 752640
OK  base: got zt_off_cmagen_ramp exp zt_off_cmagen_ramp
OK  scale: got 1.0 exp 1.0
OK  d: got 64 exp 64
OK  depth: got 2 exp 2
OK  total params: got 48277 exp 48277
zone_mu shape: (3, 64)
OK  zone_out: got 3 exp 3
OK  skynet.pt sha256: got 61bc2d62fe1c0edbb5798f32e8ebc56e7197f1e1ab1312e2a4ef964fd4a19fc7 exp 61bc2d62fe1c0edbb5798f32e8ebc56e7197f1e1ab1312e2a4ef964fd4a19fc7
A4: GREEN
```

A5 command output (`./.venv/bin/python -B /tmp/a5_check.py`):

```
bundle files: ['LICENSE-NOTICE.txt', 'MANIFEST.json', 'README.md', 'TAG.txt', '__init__.py', 'base.py', 'cma_search.py', 'curves.json', 'env.py', 'features.py', 'hist_cma.jsonl', 'hist_cma_zt.jsonl', 'hist_cma_zt_gen.jsonl', 'hist_cma_zt_warm.jsonl', 'nets.py', 'policy.py', 'sbrl.py', 'selection_extra.json', 'selection_set.json', 'skynet.pt', 'skynet_eval.py', 'skynet_train.py', 'zt_controller.py']
train selection_set: /tmp/rl_zip/skynet/selection_set.json exists=False
train manifest: /tmp/rl_zip/data/gen/manifest.jsonl exists=False
train selection_extra: /tmp/rl_zip/skynet/selection_extra.json exists=False
bundle-root selection_set.json: exists=True
bundle-root selection_extra.json: exists=True
read skynet/selection_set.json: FileNotFoundError: [Errno 2] No such file or directory: '/tmp/rl_zip/skynet/selection_set.json'
read data/gen/manifest.jsonl: FileNotFoundError: [Errno 2] No such file or directory: '/tmp/rl_zip/data/gen/manifest.jsonl'
logs/: exists=False
tools/: exists=False
data/: exists=False
A5: gaps confirmed (see journal for verdict vs plan text)
```

B2 per-scenario table (shipped `/tmp/b2_bill_off` vs honest
`/tmp/b2_bill_on`, loss = `abs(agent_reward.sum())/n`, all n=8640):

```
pair                               shipped    honest   delta%
building_train_1/atlanta_2         0.005419 0.005391   -0.52%
building_train_1/denver_7          0.004112 0.004093   -0.46%
building_train_1/fairbanks_1       0.051579 0.052165   +1.13%
building_train_1/phoenix_5         0.016958 0.016958   +0.00%
building_train_1/singapore_8       0.007364 0.007364   +0.00%
building_train_2/atlanta_3         0.004401 0.004396   -0.09%
building_train_2/denver_8          0.007081 0.006809   -3.84%
building_train_2/fairbanks_2       0.016119 0.015121   -6.19%
building_train_2/phoenix_6         0.009146 0.009146   +0.00%
building_train_2/singapore_1       0.005535 0.005535   +0.00%
building_train_4/atlanta_5         0.007286 0.007286   -0.00%
building_train_4/denver_2          0.008328 0.008087   -2.90%
building_train_4/fairbanks_4       0.003421 0.003369   -1.49%
building_train_4/phoenix_8         0.003329 0.003301   -0.84%
building_train_4/singapore_3       0.007796 0.007796   +0.00%
building_train_5/atlanta_6         0.005006 0.005006   +0.00%
building_train_5/denver_3          0.006465 0.006366   -1.52%
building_train_5/fairbanks_5       0.002850 0.002847   -0.10%
building_train_5/phoenix_1         0.004732 0.004683   -1.03%
building_train_5/singapore_4       0.006499 0.006499   +0.00%
building_train_7/atlanta_8         0.009993 0.009667   -3.26%
building_train_7/denver_5          0.006346 0.006345   -0.01%
building_train_7/fairbanks_7       0.011072 0.010754   -2.88%
building_train_7/phoenix_3         0.007284 0.007284   -0.00%
building_train_7/singapore_6       0.007166 0.007166   +0.00%
building_train_8/atlanta_1         0.007700 0.007466   -3.03%
building_train_8/denver_6          0.005062 0.005061   -0.03%
building_train_8/fairbanks_8       0.030471 0.030149   -1.06%
building_train_8/phoenix_4         0.013208 0.013208   -0.00%
building_train_8/singapore_7       0.007168 0.007168   -0.00%
MEAN shipped=0.00962975 honest=0.00954945 delta=-0.83%
```

C2 gas-by-hour on building_train_8/fairbanks_8, occ_gated7
(share of total `natural_gas_energy_cost`, %):

```
hour:  0    1    2    3    4    5    6    7    8    9   10   11
share: 1.2  1.2 11.9  8.6  8.9  8.4  6.7  5.6  5.0  4.5  3.9  3.6
hour: 12   13   14   15   16   17   18   19   20   21   22   23
share: 3.4  3.3  3.3  3.4  3.6  3.6  3.8  1.3  1.1  1.2  1.2  1.2
```

C3 per-scenario table (`/tmp/c3_ramp`, all n=8640):

```
pair                               gated7      ramp   delta%
building_train_1/atlanta_2         0.005419 0.005358   -1.14%
building_train_1/denver_7          0.004112 0.004168   +1.37%
building_train_1/fairbanks_1       0.051579 0.052817   +2.40%
building_train_1/phoenix_5         0.016958 0.018260   +7.67%
building_train_1/singapore_8       0.007364 0.007581   +2.96%
building_train_2/atlanta_3         0.004401 0.004556   +3.53%
building_train_2/denver_8          0.007081 0.006603   -6.74%
building_train_2/fairbanks_2       0.016119 0.015498   -3.85%
building_train_2/phoenix_6         0.009146 0.009120   -0.28%
building_train_2/singapore_1       0.005535 0.005732   +3.56%
building_train_4/atlanta_5         0.007286 0.007367   +1.11%
building_train_4/denver_2          0.008328 0.008328   +0.00%
building_train_4/fairbanks_4       0.003421 0.003219   -5.91%
building_train_4/phoenix_8         0.003329 0.003308   -0.63%
building_train_4/singapore_3       0.007796 0.007995   +2.55%
building_train_5/atlanta_6         0.005006 0.005078   +1.44%
building_train_5/denver_3          0.006465 0.006342   -1.90%
building_train_5/fairbanks_5       0.002850 0.002849   -0.04%
building_train_5/phoenix_1         0.004732 0.004754   +0.46%
building_train_5/singapore_4       0.006499 0.006687   +2.90%
building_train_7/atlanta_8         0.009993 0.009862   -1.31%
building_train_7/denver_5          0.006346 0.006357   +0.17%
building_train_7/fairbanks_7       0.011072 0.010802   -2.44%
building_train_7/phoenix_3         0.007284 0.007261   -0.31%
building_train_7/singapore_6       0.007166 0.007263   +1.36%
building_train_8/atlanta_1         0.007700 0.007481   -2.83%
building_train_8/denver_6          0.005062 0.005178   +2.28%
building_train_8/fairbanks_8       0.030471 0.029731   -2.43%
building_train_8/phoenix_4         0.013208 0.013765   +4.22%
building_train_8/singapore_7       0.007168 0.007297   +1.81%
MEAN gated7=0.00962975 ramp=0.00968720 delta=+0.60% wins=13/30
```

D2 field-group audit (their `features.py` group -> our source;
`n` = width; all obs groups resolve — same sim):

```
ZONE own (6): temp/heating-SP/cooling-SP/damper/reheat/flow-SP
  -> our obs, same fields. OK (temp denormalised via meta consts)
ZONE band errors (2): (t-lo)/5, (hi-t)/5
  -> our obs + clock band. DRIFT: loop uses occ-adjusted bands;
     new col 0 (true loop err / 5 K) compensates
ZONE PID proposal (2): pid damper/reheat per zone
  -> occ_gated7 cont proposal. OK
ZONE AHU ctx (2): AHU supply-temp sensor, AHU run flag
  -> our obs + occ disc proposal. OK
GLOBAL (9): hod/dow sin-cos, comfort now/soon, occ/125,
  (OAT-295)/10, in-comfort-hours flag
  -> our obs, same fields. OK (occ raw verified; OAT raw-K
     verified; in-hours flag clock-derived both sides)
AHU own (15): sat/sap/flow/oat/recirc/fans/run cmds/cool count
  -> our obs, same fields. OK
AHU pooled (6): mean err, cold/warm frac, mean PID damper/
  reheat, max err
  -> derived from zone groups above. OK (inherits band drift)
AHU PID proposal (4): pid temp/pressure/recirc/run
  -> occ_gated7 cont/disc proposal. OK (run idx verified)
PLANT own (7): hws temp/pressure/run/cmds/heat count
  -> our obs, same fields. OK
PLANT pooled (3): mean err, cold frac, mean AHU run
  -> derived. OK (inherits band drift)
PLANT PID proposal (3): pid water temp/pressure + pump vote
  -> occ_gated7 cont + disc[-1]. OK (pump-last verified)
NEW our cols (2): true loop err / 5 K, integral*KI saturation
  -> occ loop state (read-only). OK (fixed scales, +-10 clip)
ZONE-TARGET head interface: target_offsets {VAV: K} injection
  -> MISSING on our stack (no target_offset hook; D5 decides)
```

D3 gating design (theirs -> ours):

```
floor.free (PI hours)            -> residual_free(): not shutdown,
                                    no floor mode, not cold (COLD_GATE_K)
off-hours => rule alone          -> same: probe returns base action
latch untouched off-hours        -> stronger: no latch, proposal disc
                                    returned verbatim (probe + rollout)
vmask zeros (switch mask)        -> n/a (votes frozen); entity-pad masks
                                    zero as before (ppo mask check)
fr = free * decide; lp *= fr     -> fr recorded in rollout (1.0/0.0),
                                    pg + KL masked-mean over fr (their
                                    :199 formula verbatim); critic unmasked
held cleared off-hours           -> n/a (hold=1: no held residual)
```

R1-R9 tick: R1 deltas on all comparisons; R2 coverage labeled
offline/no-sim; R3 all paths/commands exist (`/tmp/a*_*.py`
throwaway, kept in /tmp); R4 no new negatives; R5 no new policy
(no ladder entry due — `tools/honest_bill.py` is a local-only
measurement patch, not a policy); R6 no submissions; R7 per-item
wall times not recorded (Done subsections untimed by design; known
clock readings only: A1-A3 done by 09:42, A4-B1 done 09:42-09:49
local +06); R8 third-party numbers labeled (dataset-reported,
recomputed offline); R9 English-only.
