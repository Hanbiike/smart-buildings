# Online RL Track E — Model-Based Alternatives for HVAC Control

**Title:** Model-based alternatives to model-free Online RL (ADAPT, LDLP, Counter-Dyna, hypernetwork MBRL, DreamerV3, MBPO doctrine) against the in-repo grey-box substrate
**Date:** 2026-11-01
**Status:** draft
**Scope:** Head-to-head numbers of model-based vs model-free HVAC controllers, sample-efficiency ratios, transfer evidence, compounding-error / horizon doctrine, and compute limits — each with transfer limits to our competition setting (VAV-reheat-only plant, 5-min steps, 8640-step episodes, one frozen policy on unseen buildings). Web research synthesis only; no simulator runs, no training, no code changes.

## Sources

| ID | Source | Body inspected |
|----|--------|----------------|
| S1 | Session evidence file `.../tool-outputs/readable-text/9110ebceb1ea4483b7d920df960329981b218817ec76218c76323687cdbb59cb.txt` (ADAPT / LDLP / Counter-Dyna / hypernetwork / DreamerV3 / MBPO claims with excerpts) | Yes, full file (3 long lines, no truncation remainder) |
| S2 | Session evidence file `.../tool-outputs/readable-text/42641ec471e2087178e19d0d394643f652d75416802c7a1ae23ac4506ccd5066.txt`, resolved-result-4 section only (lines 19-24: SAC/TD3/PPO/HVAC SOTA, hypernetwork, ADAPT, Dreamer claims) | Yes, lines 1-45 |
| S3 | `docs/research/02-mpc.md`, lines 1-80 (OptiControl line, Purdue 104-study review figures) | Yes, lines 1-80 |
| S4 | `smart_buildings/baselines/greybox.py`, lines 1-60 (lumped-RC zone balance, estimator modes) | Yes, lines 1-60 |
| S5 | `smart_buildings/baselines/thermal_greybox.py`, lines 1-60 (offline ablations, slow thermal state, water physics) | Yes, lines 1-60 |
| S6 | `smart_buildings/baselines/equipment_observer.py`, lines 1-60 (observation/command timing, equipment priors) | Yes, lines 1-60 |
| S7 | `tools/greybox_check.py`, lines 1-60 (offline validation harness on banked dumps) | Yes, lines 1-60 |

Note: the assigned path for S1 carried a truncated hash (`...ec7623687cdbb59cb`); the on-disk file has hash `...ec76218c7623687cdbb59cb`. The closest (only) prefix match was read; no other candidate exists.

Evidence standard: every number below carries its provenance (URL or repo path). Figures quoted from paper text are journal-reported, not recomputed here.

---

## A. Head-to-head numbers: model-based vs model-free

**A1. ADAPT (physics-aware conditional-diffusion world model + downstream RL) reduces HVAC energy by 7.3% and occupant discomfort by 30.2% vs the strongest baseline under in-distribution control on SemiBuildingSim/Sinergym.**
Provenance: S1, source `https://arxiv.org/pdf/2608.19804`.

**A2. ADAPT in-distribution head-to-head (SemiBuildingSim Summer, mean over 3 seeds, lower better). Energy (kWh): ADAPT 138.49 vs DreamerV3 149.43 vs TransformerRL 149.34 vs MBVE 150.73 vs MBPO 150.99 vs PPO 154.83. PPD (%): ADAPT 7.01 vs DreamerV3 9.88 vs TransformerRL 10.05 vs PPO 12.37. Abs PMV: ADAPT 0.22 vs DreamerV3/MBVE 0.36 vs PPO 0.41. Action fluctuation: ADAPT 6.73 vs DreamerV3 8.37 vs PPO 9.75.**
Consequence: DreamerV3 roughly ties TransformerRL on energy (149.43 vs 149.34) and is only ~3.5% better than PPO on energy ((154.83-149.43)/154.83). The diffusion + physics world model, not latent imagination alone, carries the ADAPT margin.
Provenance: S1, Table 1 via `https://arxiv.org/pdf/2608.19804`.

**A3. LDLP (latent dynamics + MCTS planning) vs PPO/SAC/TD3/Sampled-EfficientZero on Sinergym 5Zone under equal budget (4 parallel collectors, 100 simulations per decision, 7 root actions, batch 4096, 30,000 updates; 1-year episodes at 15-min resolution = 35,039 steps; single-climate train+eval per climate). Comfort-normalized energy (Power@100%Comfort): Hot — LDLP 10.489 vs PPO 12.624 vs SAC 12.165 vs EfficientZero 22.305; Mixed — 10.299 vs 12.050 vs 11.228 vs 21.358; Cool — 7.843 vs 10.663 vs 8.159 vs 17.044. Headline: 4-26% less than PPO/SAC under the standard reward, 1-19% under the gated reward.**
Provenance: S1, source `https://mdpi-res.com/d_attachment/applsci/applsci-16-09131/article_deploy/applsci-16-09131.pdf?version=1789457351`.

**A4. Sampled EfficientZero exhibits comfort-violation trading on HVAC: lowest raw energy in hot/cool (7.034/6.005) but comfort ratios ~0.32-0.36 vs LDLP ~0.72-0.80, giving ~2x worse comfort-normalized energy. The normalization itself is author-flagged as a first-order device that assumes linear energy-comfort trade and over-corrects as comfort ratio falls.**
Transfer limit: our reward weights comfort 0.6 with slope ~0.18/K vs energy 0.2 — any method that buys energy with comfort fails here by construction; EfficientZero-style trading is disqualified.
Provenance: S1, same MDPI PDF.

**A5. LDLP planning ablation: one simulation per decision (planning removed) multiplies normalized energy fivefold; each reported decision evaluates 7 candidate setpoint pairs over 100 simulations.**
Transfer limit: 100 simulations per decision is a CPU-incompatible inference budget for a Kaggle kernel; LDLP is evidence for planning value, not a deployable recipe.
Provenance: S1, same MDPI PDF.

**A6. Model-free baseline context (Sinergym): SAC and TD3 are the empirically validated model-free SOTA for continuous HVAC control (Manjavacas et al.). On bounded setpoint action spaces SAC beats PPO because PPO's Gaussian policy collapses onto action bounds (bit-identical deterministic eval: heating 12.0 C and cooling 23.25 C at lower bounds, std = 0.0 across 35,040 eval steps) while SAC's auto-tuned entropy preserves exploration. The winner is environment-dependent: in an 8-zone VRF annual simulation PPO cut comfort violations 67% vs rule-based control while SAC saved 27.6% energy.**
Transfer limit: our action space is bounded damper/reheat/setpoint commands, so the PPO-collapse mechanism applies to us; any model-free baseline we compare a world model against should be SAC-class, not PPO.
Provenance: S2 lines 21-22; `https://arxiv.org/abs/2401.05737`, `https://raw.githubusercontent.com/shiplusheba/hvac-rl-sinergym/main/README.md`, `https://arxiv.org/abs/2608.17235`.

**A7. Higher-order action (jerk) regularization preserves competitive RL performance while cutting HVAC equipment switching by 60%.**
Transfer limit: directly portable as a training-logic add-on regardless of model-free/model-based choice.
Provenance: S2 line 21; `https://arxiv.org/abs/2601.02061v1`.

---

## B. Sample efficiency

**B1. Counter-Dyna (Dyna-style MBRL with counterfactual surrogate, BOPTEST hydronic heat pump, PPO, chronological episodes, 30 seeds, reward weights wD=1/wC=100) trains in 5 weeks what model-free needs 50 weeks for: Counter-Dyna-5/10 surpass asymptotic MF-50 after only 5 weeks; MF-50 converges only after ~30 weeks (6x longer to convergence). Claimed 10x speedup in real environment interactions. Hypothetical-deployment savings vs MF-10 (discomfort < 100 Kh): Counter-Dyna-10 5.3% peak-heating / 17.0% typical-heating; Counter-Dyna-5 3.5% / 13.2% at half the training.**
Settings that bound the claim: perfect forecasts, single-MLP dynamics, episode-boundary resets to new operating points, single test case, and BOPTEST lacks minimum run-times/startup losses (favoring bang-bang control).
Provenance: S1, source `https://arxiv.org/html/2605.04555`.

**B2. Counter-Dyna operating recipe: rollout length L=24 with 24h resets; synth:real = 20 baseline; ablation over L in {6, 12, 24, 48, 96} shows performance is insensitive inside a moderate regime provided (i) at least a 10:1 synthetic-to-real ratio and (ii) rollout length stays moderate — long rollouts accumulate model error (citing Janner et al. 2019).**
Transfer limit: the portable Phase-2 recipe is short rollouts + synth:real >= 10:1; the exact L=24 (24h at their resolution) must be re-derived at our 5-min resolution.
Provenance: S1, same arXiv page.

**B3. Prior HVAC MBRL context reported inside Counter-Dyna: Gao & Wang 2023 (5-MLP ensemble, MBPO-style, DDQN/SAC on BOPTEST) model-based variants converge 25-38% faster but still need >= 46 weeks with time-travel training; Saeed et al. 2024 Dyna-PINN reaches ~1 C MAE at 1-2-day horizon but uses 25-50 weeks; model-free RL needs years/decades to converge.**
Provenance: S1, same arXiv page (literature summary therein).

**B4. ADAPT sample-efficiency reporting is qualitative plus a figure-only metric: all methods trained 1e6 timesteps with MultiDiscrete factorized-categorical actions; rollout/imagination horizon 8 for MBVE/MBPO/DreamerV3; Fig. 2 stars mark the first point reaching 95% of final episodic return with relative convergence steps/return normalized to ADAPT = 1.0; text states ADAPT "converges substantially faster and reaches a higher return". No numeric sample-ratio appears in text.**
Transfer limit: do not quote a numeric ADAPT sample-efficiency ratio — none exists in the paper body.
Provenance: S1, source `https://arxiv.org/pdf/2608.19804`.

**B5. DreamerV3 generic trade-off (non-HVAC tutorial account, Atari/DMC): 10-100x sample-complexity cut vs PPO/SAC (DMC Walker: PPO ~10M steps vs DreamerV3 500k, 20x); ~10x more GPU time (1M steps: PPO ~2h vs DreamerV3 8-10h on RTX 3090). This contrasts sharply with HVAC, where DreamerV3 only marginally beats PPO (claim A2).**
Transfer limit: Atari/DMC ratios do not transfer to HVAC; on HVAC the demonstrated DreamerV3 margin over PPO is ~3.5% energy.
Provenance: S1, source `https://tildalice.io/dreamerv3-world-model-rl-pytorch-implementation/` (opinion-grade source).

---

## C. Transfer and generalization

**C1. ADAPT transfer protocol and OOD prediction numbers: 5-min control interval (matches ours); seasonal transfer summer/winter on SemiBuildingSim; climate-region transfer Stockholm/Arizona on Sinergym 2ZoneDataCenterHVAC. OOD prediction (Table 2): Summer->Winter overall MAE 0.035 (ADAPT) vs 0.040 w/o-physics vs 0.119 MambaFormer vs 0.131 VAE; Room-Temp CVRMSE 11.560 vs 16.134 vs 43.210 vs 44.983. Winter->Summer MAE 0.049 / 0.059 / 0.181 / 0.204. Removing either physics-loss term consistently deteriorates Winter->Summer OOD prediction.**
Transfer limit: strongest OOD-prediction evidence in this track, but it is prediction error, not control reward under transfer; and the world model is a GPU diffusion model (see D/E).
Provenance: S1, source `https://arxiv.org/pdf/2608.19804`.

**C2. Counter-Dyna generalizes winter->spring: trained on pre-Jan-17 winter only, tested on Apr-19/May-3 typical heating; Counter-Dyna-5/10 outperform MF-50 (which saw varied regimes) while MF-10 fails. Surrogate accuracy: final building model in-sample MAE 0.08 +/- 0.04 C / RMSE 0.19 +/- 0.05 C vs out-of-sample (counterfactual actions) 0.90 +/- 0.84 / 1.66 +/- 2.77; reward model R2 = 0.98.**
Transfer limit: closest control-reward generalization evidence (train one regime, control another), but single building, single test case, perfect forecasts.
Provenance: S1, source `https://arxiv.org/html/2605.04555`.

**C3. Hypernetwork continual MBRL (BOPTEST hydronic heat pump, SAC + Dyna-style synthetic rollouts; hypernetwork generates dynamics/reward nets from task/layer IDs + Gaussian-noise ensembles): after training Task 2 (3-action control), minimal fine-tuning on Task 1 (1-action) reconverges within 5 episodes, outperforming MFRL and mitigating catastrophic forgetting via regularization. Failure mode: sudden performance drop in Task 2 despite prior stability (hypernetwork training instability).**
Transfer limit: scope is action-space transfer on ONE building, not unseen buildings/climates; multi-zone and real-building validation are listed as future work. The 5-episode reconvergence does not transfer to our zero-shot unseen-building protocol.
Provenance: S1 + S2 line 21; `https://arxiv.org/html/2503.19212v1` and `https://arxiv.org/abs/2503.19212v1`.

**C4. LDLP tested no cross-climate/building transfer: each experiment trains and evaluates independently under a single weather file (hot, mixed, or cool). No wall-clock/GPU training times reported.**
Provenance: S1, MDPI PDF above.

**C5. MPC field prior (Purdue 104-study review, arXiv:2503.05022v4): corpus 24 residential + 80 commercial field demonstrations; 71% use experiment protocols judged unreliable; over the reliable 29%, duration-weighted average utility-bill savings are 16% residential / 13% commercial; only 13 of 104 papers report deployment/operation/maintenance costs; raw commercial range roughly -5% to +60% energy, measurement-based mostly 10-30%.**
Transfer limit: the seminal 16-41% MPC-vs-rule-based figure concentrates in price-structure exploitation, which is dead under our 1.25x TOU and constant COP 3.5 — portable gains must come from coordinated optimization and weather/occupancy anticipation.
Provenance: S3 (`docs/research/02-mpc.md` lines 58-80); `https://arxiv.org/html/2503.05022v4`.

---

## D. Compounding error and horizon doctrine

**D1. MBPO doctrine: single-step accuracy can hold under policy shift, but chaining predictions compounds errors over the horizon and makes long rollouts unreliable; the remedy is short rollouts branched from previously seen real states rather than full-length rollouts from the initial distribution (the Dyna/MBPO recipe).**
Provenance: S1, source `https://bair.berkeley.edu/blog/2019/12/12/mbpo/`.

**D2. DreamerV3 imagination horizon is 15 with divergence after 10-15 steps (tutorial account); ADAPT's MBVE/MBPO/DreamerV3 baselines use rollout/imagination horizon 8; Counter-Dyna uses L=24 with 24h resets and warns against longer (Janner et al.).**
Transfer limit: all staying power sits at horizons 8-24 steps. Our grey-box harness already evaluates L-step-ahead prediction at horizons (1, 3, 6, 12) — inside the survivable band (see F4).
Provenance: S1 (TildAlice tutorial + ADAPT Table 9 + Counter-Dyna); S7.

**D3. Counter-Dyna surrogate error explodes off-distribution: in-sample MAE 0.08 +/- 0.04 C vs counterfactual-action out-of-sample 0.90 +/- 0.84 C (RMSE 0.19 +/- 0.05 vs 1.66 +/- 2.77).**
Transfer limit: any Dyna loop on our simulator must measure prediction error under the *learning* policy's action distribution, not just replay of the logging policy.
Provenance: S1, `https://arxiv.org/html/2605.04555`.

---

## E. Compute limits

**E1. ADAPT compute: all runs on NVIDIA A100; offline world-model training ~12h; downstream online RL ~10h (SemiBuildingSim) / ~16h (Sinergym); inference one diffusion step ~150ms (Temporal U-Net, 20 denoising steps, hidden dim 256, EMA 0.995, L1 loss). No CPU-only inference numbers reported.**
Transfer limit: 20 denoising steps x ~150ms-class cost with no CPU numbers = CPU-incompatible for a Kaggle notebook kernel; diffusion world models are ruled out for Phase 2 deployment, admissible only as an offline teacher.
Provenance: S1, source `https://arxiv.org/pdf/2608.19804` (section D.4).

**E2. DreamerV3 compute (tutorial account): ~10x more GPU time than model-free for the same sample count (1M steps: PPO ~2h vs DreamerV3 8-10h on RTX 3090); harder to implement/debug than model-free; latent imagination horizon 15.**
Provenance: S1 (TildAlice tutorial); S2 line 21 (rlox survey via `https://raw.githubusercontent.com/wojciechkpl/rlox/HEAD/docs/research/08-dreamer.md`).

**E3. LDLP compute: 100 simulations per decision x 7 candidate setpoint pairs at every control step (15-min resolution); no wall-clock/GPU times reported.**
Transfer limit: MCTS-at-inference is CPU-incompatible at our 5-min resolution and 8640-step episodes; same verdict as diffusion — offline-study only.
Provenance: S1, MDPI PDF above.

---

## F. Our repo's grey-box substrate (what Phase 2 builds on)

**F1. `smart_buildings/baselines/greybox.py` implements a lumped-RC per-zone balance identified online from policy-visible observations only: T[t+1]-T[t] = a*(T_out-T) + b*flow*(T_sup-T) + c*flow*reheat*(T_water-T) + d*occ + e, with 5 parameters per zone (envelope, supply-air, reheat, occupancy, constant). Two estimator modes: `rls` (recursive least squares with forgetting factor, warm-started from a fixed prior, no warm-up window) and `batch_rls` (ridge least squares over the first `fit_window` steps, then RLS). The module is explicitly a tool, not a controller.**
Provenance: S4 (`smart_buildings/baselines/greybox.py` lines 1-60).

**F2. `smart_buildings/baselines/thermal_greybox.py` adds offline-only ablations: switchable physical water model (valve Kv = 0.108, Kv*sqrt(bar) in m3/hour converted to m3/second) and a causal slow thermal-mass state (default mass time constant 24.0 steps = 2h at 5-min resolution), with fit_window = 144 (12h), sensor_delay = 1, calibrate_plant = True.**
Provenance: S5 (`smart_buildings/baselines/thermal_greybox.py` lines 1-60).

**F3. `smart_buildings/baselines/equipment_observer.py` fixes the timing contract a Dyna loop must respect: zone temperatures latch BEFORE the simulator's thermal solve while actuator/flow observations arrive AFTER it, so the pending temperature transition at observation t belongs to command t-1 with outdoor/water inputs from t-1 — a new command t must not replace that pending transition. Explicit global priors (not inferred specs): design air pressure 2000.0 Pa, valve Kv 0.108, header resistance 0.0.**
Provenance: S6 (`smart_buildings/baselines/equipment_observer.py` lines 1-60).

**F4. `tools/greybox_check.py` replays a recorded observation sequence through the grey-box model and measures L-step-ahead zone-temperature prediction vs a persistence baseline (T[t+1] = T[t]), with 8 model kinds (`legacy`, `causal`, `sensors`, `delayed`, `calibrated`, `water`, `mass`, `thermal`), default horizons (1, 3, 6, 12), default policy `dsat_gen31`, and default fit_window = 144.**
Provenance: S7 (`tools/greybox_check.py` lines 1-60).

---

## G. Synthesis: what transfers to Phase 2

1. The portable model-based recipe is Counter-Dyna-shaped: short rollouts (8-24 steps) branched from real visited states, synth:real ratio >= 10:1 (20:1 baseline), error measured under the learning policy's actions (claims B2, D1-D3). Our grey-box + `tools/greybox_check.py` stack already covers the model and the horizon-error measurement (F1-F4); the missing piece is closing the loop into policy updates.
2. The portable world-model idea is physics-regularized dynamics: ADAPT's ablation shows both physics-loss terms are required for OOD prediction (C1); our grey-box equation (F1) plus water/mass ablations (F2) are the CPU-cheap analogue of that physics regularization.
3. Ruled out for deployment: diffusion world models at inference (E1: 20 denoising steps, A100-only numbers) and MCTS planning at inference (E3/A5: 100 simulations x 7 candidates per decision). Both stay admissible as offline teachers / upper-bound studies only.
4. DreamerV3-class latent imagination is a weak fit: ~3.5% energy margin over PPO on HVAC (A2), 10x GPU cost (E2), divergence after 10-15 steps, and known weakness on low-dimensional states — our observation is a low-dimensional sensor vector.
5. Any comparison baseline must be SAC-class with entropy regularization (A6), and comfort must never be traded for energy (A4, A7): our 0.6/0.2/0.2 weights with comfort slope ~0.18/K make comfort-trading methods fail by construction.

## Open questions

1. What is our grey-box L-step-ahead error at horizons (1, 3, 6, 12) on banked `occ_gated7` dumps, and at which horizon does it cross the persistence baseline? (Measurement exists in `tools/greybox_check.py`; numbers not yet recorded for Phase 2.)
2. What synth:real ratio and rollout length maximize policy improvement per simulator step on our plant — do the Counter-Dyna >= 10:1 / moderate-L settings re-derive at 5-min resolution?
3. Can a physics-regularized (grey-box-residual) world model beat the pure grey-box OOD error the way ADAPT's physics terms beat its w/o-physics ablation (MAE 0.035 vs 0.040, Room-Temp CVRMSE 11.560 vs 16.134)?
4. What is the CPU inference budget per step inside the Kaggle kernel, and does any learned-dynamics + short-rollout planner fit it, or is the world model training-time-only (Dyna) with a reactive deployed policy?
5. Is hypernetwork-style continual adaptation worth its instability risk (sudden Task-2 drop) given our protocol is zero-shot frozen-policy, not continual fine-tuning?
6. Unresolved from prior evidence (bodies not inspected): TD3-beats-SAC detail in 2ZoneDataCenterHVAC (arXiv:2401.05737v3 sections 5-6 fetch truncation); MDPI 2026 SAC-vs-PPO/TD3/DDPG comparisons (HTTP 403); Linnk DRL-for-HVAC summary (HTTP 403). The ADAPT-vs-DreamerV3/MBPO/MBVE head-to-head gap flagged in S2 is now closed by claim A2 from S1.
