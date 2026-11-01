# Phase 2 Model-Based Checks — Exhaustive Checklist

**Date:** 2026-11-01
**Status:** draft
**Scope:** Everything to verify before committing Phase 2 (Online RL over the `occ_gated7` helm) to any model-based component: grey-box loop closure, Dyna short rollouts, synth:real ratio, physics-regularized world models, and CPU-incompatibility gates for diffusion/MCTS. Research basis: `docs/research/15-online-rl-model-based.md` (claims A1-G5 referenced below as R:A1 etc.).
**Conventions:** Python is `./.venv/bin/python -B` (never system python). Full episode = 8640 steps. Loss from CSV: `abs(df.agent_reward.sum()) / len(df)`. Keep total concurrent sims <= 10 with 1500s+ timeouts. Never edit policy code while a validation run using it is in flight.

## G1. Grey-box loop closure (model -> controller, not just prediction)

| ID | Check | How (exact command / file) | Expected / threshold | Priority |
|----|-------|----------------------------|----------------------|----------|
| G1.1 | Grey-box L-step-ahead error on banked `occ_gated7` dumps at default horizons | `./.venv/bin/python -B tools/greybox_check.py --scenarios building_train_1/fairbanks_1` (harness: `tools/greybox_check.py`; model: `smart_buildings/baselines/greybox.py`) | Record MAE vs persistence at horizons (1, 3, 6, 12); proceed only if model beats persistence at horizon >= 6 (R:D2 band) | P0 |
| G1.2 | Repeat G1.1 across all 8 model kinds | Same command with `--model-kind` in `legacy,causal,sensors,delayed,calibrated,water,mass,thermal` (kinds defined in `tools/greybox_check.py` lines 27-44) | Pick the kind with lowest horizon-12 MAE; record delta vs `legacy` per R1 (no bare better/worse) | P0 |
| G1.3 | Fit-window sensitivity (`batch_rls` warm-up) | Same harness, vary `--fit-window` around default 144 (12h): 72 / 144 / 288 | Horizon-12 MAE within 10% of best across windows, else pin window per zone-count class | P1 |
| G1.4 | `rls` (no warm-up) vs `batch_rls` (one-day warm-up) episode-start error | Same harness, vary `--mode rls` vs `--mode batch_rls` (modes: `smart_buildings/baselines/greybox.py` lines 22-27) | First-day (steps 0-288) MAE recorded separately; `rls` must not exceed 2x `batch_rls` error or loop closure needs warm-up handling | P1 |
| G1.5 | Timing-contract compliance of any new rollout code | Code review against `smart_buildings/baselines/equipment_observer.py` lines 1-8: transition at obs t belongs to command t-1; new command t must not replace pending transition | Zero violations; add a regression test next to `tools/test_equipment_observer.py` if rollout code touches command timing | P0 |
| G1.6 | Close the loop: grey-box warm-up-time prediction drives `occ_gated7` prestart, full-episode eval | New probe policy in `probe_modes.py` + `POLICY_MAP` entry in `smart_buildings/evaluate.py`; eval via `tools/fullrun.py <results> 8640 <conc<=10> 1500 <probe-policy> <pairs>` | Mean per-step \|reward\| delta vs frozen `occ_gated7` on same pairs, train-30 coverage, R2-labeled; negative result (R4) still recorded | P0 |

**Definition of done (G1):** horizon-error table for all 8 model kinds on `occ_gated7` dumps exists; winning kind + window pinned; one closed-loop probe evaluated full-episode (8640 steps) with a recorded delta (positive or negative).

## G2. Dyna short rollouts (branch from real states, never full-length)

| ID | Check | How (exact command / file) | Expected / threshold | Priority |
|----|-------|----------------------------|----------------------|----------|
| G2.1 | Rollout-length ablation L in {6, 12, 24, 48} at 5-min resolution | Dyna probe built on `smart_buildings/baselines/greybox.py` + `tools/greybox_check.py` harness; rollout code branches from banked real states per R:D1 | Policy improvement peaks at moderate L (R:B2); L=48 must not beat L<=24 or the error-accumulation assumption is re-examined | P0 |
| G2.2 | Branch-from-real-states vs from-initial-distribution control | Same Dyna probe with rollout-start switch (real visited states vs episode-start states); doctrine R:D1 | From-real-states wins; if not, world-model error diagnosis before any ratio tuning | P0 |
| G2.3 | Prediction error under the *learning* policy's actions, not the logging policy's | Evaluate world-model MAE on rollouts driven by updated-policy actions (counterfactual regime, R:D3) vs replay actions | Report both MAEs; counterfactual MAE > 3x replay MAE triggers shorter L or re-identification (cf. 0.08 vs 0.90 C in R:D3) | P0 |
| G2.4 | 24h-reset analogue: re-grounding cadence | Vary re-ground interval {144, 288, 576} steps (12h/24h/48h at 5-min) in Dyna probe (cf. Counter-Dyna 24h resets, R:B2) | Best cadence recorded; improvement must be monotonic in groundedness or horizon error dominates | P1 |
| G2.5 | Equal-budget Dyna vs model-free (SAC-class) comparison | Same simulator-step budget for Dyna probe vs model-free baseline; eval `tools/fullrun.py` full-episode; baseline must be SAC-class per R:A6, never PPO-on-bounds | Report comfort and energy separately (R:A4: no comfort trading); Dyna wins or the sample-efficiency claim is rejected for our plant | P1 |

**Definition of done (G2):** optimal L + re-ground cadence pinned with numbers; counterfactual-action MAE measured; one equal-budget Dyna-vs-SAC-class comparison recorded with comfort/energy split.

## G3. Synthetic-to-real ratio

| ID | Check | How (exact command / file) | Expected / threshold | Priority |
|----|-------|----------------------------|----------------------|----------|
| G3.1 | Synth:real ratio sweep {1:1, 5:1, 10:1, 20:1} | Dyna probe from G2 with synthetic-batch multiplier; model/harness as in G1.1 | Performance saturates at >= 10:1 per R:B2; record the knee; 20:1 is the starting baseline, not the assumed optimum | P0 |
| G3.2 | Ratio x horizon interaction | Cross {10:1, 20:1} x L {12, 24} from G2.1's winner neighborhood | No super-linear blow-up: 20:1 + long L must not collapse vs 10:1 + short L, else cap ratio by horizon | P1 |
| G3.3 | Wall-clock cost per ratio step | Time the Dyna update loop (world-model query cost only; no simulator in the synthetic leg) with `time ./.venv/bin/python -B <dyna-probe>` | Synthetic updates must be >= 10x cheaper per step than simulator steps or the ratio buys nothing under a fixed compute budget | P1 |

**Definition of done (G3):** knee of the synth:real curve identified with full-episode deltas; ratio cap (if any) documented with the horizon it pairs with.

## G4. Physics-regularized world model (CPU-cheap analogue of ADAPT physics terms)

| ID | Check | How (exact command / file) | Expected / threshold | Priority |
|----|-------|----------------------------|----------------------|----------|
| G4.1 | Water-physics ablation on OOD prediction | `tools/greybox_check.py` model kinds `water` vs `mass` vs `thermal` (`smart_buildings/baselines/thermal_greybox.py`: `physical_water`, `thermal_mass` switches; Kv=0.108, mass tau 24 steps) | `thermal` (both on) beats single-term ablations OOD, mirroring ADAPT's both-terms-required result (R:C1: MAE 0.035 vs 0.040) | P0 |
| G4.2 | Grey-box-residual world model vs pure grey-box OOD | Residual MLP on grey-box one-step error; eval OOD split (train summer / test winter or train-climate / transfer-climate via `tools/gen_transfer.py` scenarios + `tools/fullrun_t.py`) | Residual model Room-Temp CVRMSE delta vs pure grey-box recorded; must improve OOD without regressing in-sample (cf. R:C1 11.560 vs 16.134) | P0 |
| G4.3 | Physics-prior sanity: Kv / pressure / mass-tau ranges | Vary `water_kv` around 0.108, `mass_time_steps` around 24.0 in `smart_buildings/baselines/thermal_greybox.py` lines 18-29 (`ValueError` guards: tau >= 1, kv >= 0) | OOD error flat within +/-50% of priors; sharp cliffs mean the prior is load-bearing and must be identified, not assumed | P1 |
| G4.4 | Reward-model accuracy (comfort + energy heads) | Fit reward head on banked dumps; report R2 (Counter-Dyna bar: R2 = 0.98, R:C2) | R2 >= 0.95 on held-out steps or Dyna policy gradients are untrustworthy; comfort head error reported in K-equivalent via 0.18/K slope | P1 |

**Definition of done (G4):** ablation table (water/mass/thermal) + residual-vs-pure OOD numbers recorded; priors either validated as flat or promoted to identified parameters; reward-head R2 on record.

## G5. CPU-incompatibility gates (diffusion / MCTS ruled out for deployment)

| ID | Check | How (exact command / file) | Expected / threshold | Priority |
|----|-------|----------------------------|----------------------|----------|
| G5.1 | Diffusion-inference cost ruling | No implementation: cite R:E1 (20 denoising steps, A100-only, ~150ms/step, no CPU numbers) against measured Kaggle kernel CPU budget per step | Formal RULED-OUT for inference unless a distilled <= 5ms CPU variant is proposed with its own accuracy table | P0 |
| G5.2 | MCTS-at-inference cost ruling | No implementation: cite R:E3/A5 (100 simulations x 7 candidates per decision at 15-min; ours is 5-min x 8640 steps) against kernel CPU budget | Formal RULED-OUT for inference; admissible only as offline upper-bound study with its own compute log | P0 |
| G5.3 | Symbolic / short-horizon planner CPU probe (if any planner survives G5.1-G5.2) | Candidate planner + `./.venv/bin/python -B` timing harness on CPU-only box; horizon <= 12 steps (R:D2 band) | Per-step p99 CPU time recorded; must fit kernel budget with 2x headroom or planner is training-time-only (Dyna) with reactive deployed policy | P1 |
| G5.4 | Notebook assembly constraint check | `tools/assemble_notebook.py` (verbatim class blocks; MUST include `occ_gated:<consts>` with `STEPS_PER_HOUR`) + local exec check 300+ steps covering night/shutdown/prestart | Any surviving model-based code assembles and runs 300+ steps without NameError; failure here vetoes deployment regardless of G1-G4 | P0 |

**Definition of done (G5):** written RULED-OUT / ADMITTED verdict for diffusion and MCTS with the CPU numbers that justify it; surviving components (if any) pass the 300+ step notebook exec check.

## G6. Transfer and generalization (train one regime, control another)

| ID | Check | How (exact command / file) | Expected / threshold | Priority |
|----|-------|----------------------------|----------------------|----------|
| G6.1 | Seasonal transfer: train-summer / control-winter world-model error | Banked dumps split by season; `tools/greybox_check.py` cross-split MAE (protocol mirror of R:C1) | Report S->W and W->S MAE + Room-Temp CVRMSE; physics-regularized variant must narrow the IID-OOD gap vs pure model | P1 |
| G6.2 | Climate transfer on transfer suite | Scenarios via `tools/gen_transfer.py` (fixed seeds); eval via `tools/fullrun_t.py ... <scenarios_dir>`; world model trained on train climates only | Control reward (not just prediction MAE) on unseen climates; comfort/energy split per R:A4 | P1 |
| G6.3 | Comfort-trading veto on every model-based candidate | Every G6 eval reports comfort violations and energy separately; weights (0.6, 0.2, 0.2), comfort slope ~0.18/K >> energy slope | Any candidate buying energy with comfort (EfficientZero pattern, R:A4) is rejected even if total loss improves | P0 |
| G6.4 | Jerk/smoothness penalty add-on | Add action-regularization term to the winning recipe (R:A7: -60% switching precedent) | Switching count delta recorded; comfort must not regress | P2 |

**Definition of done (G6):** cross-season and cross-climate numbers exist for the winning world model; comfort-trading veto applied to every candidate; jerk add-on measured or explicitly deferred.

## Global definition of done

1. G1 closed-loop probe evaluated full-episode with recorded delta (R1/R2/R4 compliant).
2. G2 optimal L + counterfactual MAE on record.
3. G3 ratio knee identified.
4. G4 ablation + residual-vs-pure OOD table on record.
5. G5 written verdicts for diffusion and MCTS; notebook exec check green for anything deployable.
6. G6 transfer numbers with comfort/energy split; no comfort-trading candidate survives.
7. All measurements labeled with coverage (train-30 / transfer-12 / subset) per R2; every file/command in this checklist verified to exist per R3.
