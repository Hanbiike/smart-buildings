# Time-aware MoE verdict: Pascal slot-blend + static-context routing

Date: 2026-11-02
Status: verdict
Verdict: BUILD (narrow, gated — soft-spread BC prototype with kill criteria)

Scope: decide whether the per-timeslot-expert idea (288 slots/day x 2
banks, binomial/Pascal smoothing, static building-context routing) earns
a prototype, or closes. Inputs are four prior results: repo-fit (prior
result 1), literature survey (prior result 2), idea anatomy (prior
result 3), and the critic spot-check (prior result 4). No simulator
episode or training job was run for this note.

## 1. What the prior results establish (verified)

Idea (prior result 3, critic-verified): per-bank (workday/weekend) per-slot
(288 x 5 min) linear heads `y(s,x) = W_eff(s) x + b_eff(s)` with
`W_eff(s) = sum_j k_j W_{s+j}`, `k = (1,2,1)/4` circular (mod 288).
Bank decode from the dow harmonic is a hard cut, never smoothed.
Current BC (`tools/slot_train.py`) is hard slot assignment + ridge 1.0
(bias unpenalized), deleg-gated (shutdown steps excluded, `update=False`
memory freeze), empty slots fall back to the bank mean; the blend is
applied only at inference. PPO (`tools/slot_ppo.py:303-311`) forward-blends
then samples, so its autograd gradient already spreads
(`dL/dW_{s+j} = k_j dL/dW_eff`); lr 3e-5, KL-stop 0.01 defaults
(`tools/slot_ppo.py:71-74`).

Scale (recomputed exact by the critic): FZ=18, FA=14, FG=11; 134 params
per (bank, slot); 576 heads; total 77,184 weights + 1,152 rate
thresholds. Alternatives at the same budget: single global linear 134
(576x smaller), per-hour heads 6,432 (12x smaller), FiLM-linear ~30k
(2.5x smaller, diagonal-only), ControllerNet attention 81,673-82,441
(measured, d64/layers2/heads4) — same order as slots but shared across
time instead of partitioned-then-smoothed.

History (docs, critic-verified line refs): Pascal 1-2-1 origin
`docs/2026/10/26/README.md:27-28`; subset-8 teacher 0.0175 vs masked-BC
0.0368 vs hybrid +243% `docs/2026/10/27/README.md:33,69,116` — and that
page CLOSED the slot/PPO line ("no further slot/PPO runs"). Any BUILD
below is therefore an explicit REOPEN decision, justified in section 4.
Timing constants the kernel must respect: prestart margin 6 / max 48
(`occ_gated.py:56-58`), duty 4/2 (`occ_duty.py:32-33`) and 6/1
(`occ_round5.py:25-26`), PTR persist 4 (`occ_gated7.py:32`), EMA 1/12
(`slot_linear.py:191,211-212`).

Repo rails (prior result 1, critic-verified): `POLICY_MAP`
`smart_buildings/evaluate.py:220-388`; probe-to-`occ_gatedN` promotion
per AGENTS.md; rule-side plug = `_base_action`/`_zone_err`/`_decide`
override; residual-side plug = `probe_xres` harness
(`probe_modes.py:160-182`, gated by `residual_free` `:63-80`); judge-24 =
xfer_5..10 x 4 weathers, tuning-unseen, 0.0002 floor (`docs/ladder.md:17-19`,
`docs/TEMPLATE.md:26-27`); G8 screen `tools/promote_screen.py` (FAIL any
pair > 3x, exit 2 on missing); sim is the bottleneck (153.4 steps/s,
216k steps/1408s; conc <= 10, timeouts 1500/5400); baseline occ_gated7
train-30 0.00962975 (`docs/ladder.md:228-231`); honest bill halves
shipped gains (B3).

Literature (prior result 2, key transfers): static routing precedent —
task-level MoE pins whole inputs to fixed K experts, preloadable, +1.0
BLEU over token-MoE on 30 pairs
([pdf](https://arxiv.org/pdf/2110.03742),
[ACL](https://aclanthology.org/2021.findings-emnlp.304.pdf)); static
context as weight generator — HyperNetworks
([pdf](http://arxiv.org/pdf/1609.09106v1)); temporal consistency loss —
StickyMoE penalizes adjacent-position expert switches, -60% switch rate
([arXiv](https://arxiv.org/abs/2607.08780)); binomial filters = cheapest
separable Gaussian approximators
([Aubury & Luk](https://www.ncdc.noaa.gov/monitoring-content/monitoring-references/docs/aubury-and-luk-1996.pdf),
[pyramid standard](https://graphsearch.epfl.ch/en/concept/15966023));
binomial windows beat uniform for energy time series
([MDPI](https://www.mdpi.com/1996-1073/18/10/2507)); closest
static+temporal hybrid — TESTAM 3-expert MoE + Time2Vec
([paper](https://arxiv.org/html/2403.02600)); generalization benchmark —
Building2Building, 6000 EnergyPlus buildings
([pdf](https://arxiv.org/pdf/2607.16534)). Explicit gaps: no paper
studies 288-granularity per-timeslot experts with static-context routing
+ binomial smoothing jointly; no trainable-binomial-regularizer theory
(binomal appears only as fixed preprocessing); no static-metadata MoE
routing benchmark. All transfer claims are analogical.

Critic corrections applied (prior result 4): (a) MATH-ERROR 1 — K is
SINGULAR at the Nyquist mode for k3/k5/k7 alike
(`H(pi) = (1-2+1)/4 = 0`), so `W_eff` cannot represent alternating slot
patterns; the "same reachable function class" claim is false and raw
weights can oscillate invisibly under smooth `W_eff`. Practical impact
is small (alternating 5-min patterns are noise, not signal) but the
expressivity loss is real and disclosed. (b) MATH-ERROR 2 — the
Pascal-row `n` labels are off by 2x vs the `var = n/4` / `H = cos^n`
convention; variances below use the row convention. (c) Verified exact:
variance-reduction factors k3 0.375 / k5 0.2734 / k7 0.2256, sigmas
0.707/1.0/1.225 slots (3.5/5.0/6.1 min), param count 77,184, and the
BC-vs-post-smooth nonequivalence condition. (d) Caution carried forward:
a concurrent study finds locality/smoothing losses HURT quality on
larger multi-domain models ([pdf](https://arxiv.org/pdf/2608.18261)) —
smoothing strength must be ablated, not assumed. (e) The 26xx-arXiv
transfer numbers were spot-checked from snippets only, not full text.

## 2. Kernel width recommendation: k3, deploy and train

| Kernel | Weights | Sigma | Footprint | Var x | Edge grad | Verdict |
|---|---|---|---|---|---|---|
| k3 | (1,2,1)/4 | 3.5 min | 15 min | 0.375 | 0.25 | DEFAULT everywhere |
| k5 | (1,4,6,4,1)/16 | 5.0 min | 25 min | 0.273 | 0.25/0.0625 | train-time only, sparse slots |
| k7+ | /64 | 6.1+ min | 35+ min | 0.226 | outer 0.0156 | never uniform |

Reasoning. k3 kills boundary chatter while preserving the 06:00/19:00
steps to +-5 min, leaves the >= 30 min prestart ramp untouched, and
passes duty 4/2 + PTR persist-4 signals. k5 blurs edges +-10 min and
starts softening the 6/1 micro-duty (7-step period near Nyquist); its
outer-head effective LR is 16x down, borderline under PPO lr — hence
train-time-only for deleg-starved night/weekend slots, k3 at deploy.
k7+ blurs +-15 min (≈ the prestart margin itself), attenuates
micro-duty/persist > 50%, and its outer gradient (0.0156) freezes under
KL-stop. Two hard constraints at any width: midnight wrap stays
modulo-circular; the Fri->Sat bank cut stays HARD (a true discontinuity
— smoothing across it is a bug). Frequency view: `H(w) = cos^n(w/2)`
with the day-night square wave + 48-slot prestart ramp caps defensible
uniform width at k5, k3 preferred.

## 3. The one untested mechanism (why this is not re-litigating 10-27)

Forward blending always equals output blending (linearity) — an
implementation choice. But BC training with soft slot assignment is NOT
equal to hard-train + post-smooth: smoothing solutions differs from the
solution of smoothed stats unless second moments are proportional
across the window with uniform counts (critic: correct
ridge-equivalence condition). Our features violate this structurally
(occupancy, comfort now/soon, temperatures differ day/night), so
soft-spread BC is strictly MORE: (a) variance pooling into masked night
slots — today 0 samples falls back to the global bank mean, soft spread
gives local neighbor interpolation; (b) conditioning of ill-posed slots;
(c) bias coupling (prestart borrows occupied statistics) at the cost of
edge bias at sharp steps. What 10-27 closed was hard-assign masked-BC
(0.0368) and diverging slot-PPO — soft-spread BC (kernel ridge with a
binomial kernel, equivalently a Laplacian-penalty prior
`||W_{s+1} - W_s||^2`) has never been run. It is also offline-cheap:
ridge solves in minutes, no sim for training. That combination —
untested + addresses the identified failure mode + cheap — is the whole
case for BUILD.

## 4. Verdict reasoning: narrow BUILD with kill criteria

BUILD, scoped to exactly one question: does train-time soft-spread BC
with k3 beat hard-train + post-smooth k3 on held-out buildings offline,
and does the winner survive judge-first/G8 online? Rationale: (1) the
mechanism is genuinely new relative to the closed line; (2) offline
training cost is negligible — the budget is eval runs only; (3) kernel
choice is grounded in fixed-filter theory (binomial = Gaussian
approximator, pyramid standard) with the critic's singularity caveat
bounded to noise frequencies; (4) literature gives directional support
(static routing +1 BLEU precedent, binomial energy-forecasting win)
balanced by an explicit ablation duty (locality-loss caution). Kill
criteria (any one closes the line again): soft-spread BC fails to beat
hard+post-smooth on hold-one-building-out offline by >= 0.0002; the
prototype fails judge-first vs occ_gated7 or trips the G8 3x screen;
offline gains evaporate under the honest bill. Static-context routing
(HyperNetwork/FiLM-style heads from building metadata) is explicitly OUT
of this prototype — no benchmark exists, and 6 train buildings cannot
identify a router; revisit only if the time-aware core survives.

## 5. Minimal prototype spec

Experts: keep the existing 2-bank x 288-slot linear heads
(`slot_linear.py`, 77,184 weights) — no new expert count, no per-building
tables. Rule/mode authority stays with occ_gated7 (shutdown, floor,
prestart, run bits); slot heads modulate continuous commands only where
`residual_free`, mirroring the SlotHybrid rule/learned split.

Kernel: k3 (1,2,1)/4 circular for both train-time soft assignment and
deploy blending. k5 as ONE ablation, train-time-only on deleg-starved
slots (k3 at deploy). Bank cut hard. Offline comparison first:
soft-spread BC vs hard+post-smooth, hold-one-building-out
(`tools/bc_train.py --hold-building`, line 54), honest bill.

Router: NONE in this prototype — clock (hod/dow harmonics) is the fixed
router. Static-context routing is deferred (section 4). Register the
winning variant as ONE throwaway `probe_*` POLICY_MAP key, promote to
`occ_gatedN`-style fixed-constants on pass.

Budget (full-8640 runs; train-30 = 30, judge-24 = 24, 54 per scored
candidate; conc <= 10, timeouts 1500 train / 5400 judge; never edit
policy code while a validation batch is in flight): offline ablations
0 runs; prototype screen 2-4 pairs x 3 configs ≈ 12 runs; full
evaluation of winner + 2 ablations = 3 x 54 = 162 runs; total ≈ 175
runs (~2-3 fullrun batches at F2-like throughput) before any promote
decision. Promotion needs judge-first/b8-second, gaps >= 0.0002, G8
PASS, and shipped-AND-honest screens.

## 6. Plan sketch

| ID | Check | How | Expected | Priority |
|---|---|---|---|---|
| T1 | Offline soft-spread BC vs hard+post-smooth, hold-one-bldg-out | extend `tools/slot_train.py` with k-weighted sufficient stats per slot (k3 circular, bank-hard); compare vs current hard fit + `_blended()` on held-out building; honest bill | soft wins by >= 0.0002 on held-out; night-slot predictions move from bank mean toward neighbor interpolation; else CLOSE the line | P0 |
| T2 | k5 train-time ablation (offline) | same harness, k5 soft assignment, k3 deploy blend; check 06:00/19:00 edge bias and micro-duty attenuation | k5 helps only deleg-starved slots without edge damage; else keep uniform k3 and drop k5 | P1 |
| T3 | Probe-scale online screen (12 runs) | register winner as `probe_*` POLICY_MAP key; 2-4 pairs x 3 configs (winner, hard+post-smooth, occ_gated7) full-8640 | winner >= hard+post-smooth on probe pairs; no pair > 3x rule; else CLOSE | P0 |
| T4 | Full train-30 + judge-24 + G8 on the winner | `tools/fullrun.py` train-30, `tools/fullrun_t.py` judge-24 (xfer_5..10, never tuned on), `tools/promote_screen.py` vs occ_gated7, shipped AND honest | judge-first/b8-second pass, gap >= 0.0002, G8 PASS; any fail = CLOSE, no iteration without a new mechanism | P0 |
| T5 | Smoothing-strength ablation (online, only if T4 passes) | hard+post-smooth vs soft-spread vs FiLM-middle-ground on the same heads | per the locality-loss caution, confirm smoothing helps rather than hurts at scale; pick the weakest smoother that keeps the gain | P1 |

## 7. Open gaps (carried, not resolved here)

1. Per-slot delegation sample-count distributions (min/med/zeros printed
   by `slot_train.py` but no saved artifact) — T1 should persist them.
2. Exact wall-clock per full-8640 judge batch at conc 8 (only the 5400
   timeout is known) — budget above counts runs, not hours.
3. Whether learned weights/router logits can ship in-notebook
   (`tools/assemble_notebook.py` handles verbatim class blocks; the torch
   weights path for the Kaggle kernel is unclear) — must be answered
   before any promotion, not before T1-T3.
4. 26xx-arXiv transfer numbers rest on search snippets, not full-text
   fetch — treat as directional.

## 8. Execution log

- 2026-11-03 T1 PASS (day note `docs/2026/11/03/`): 30/30 occ9
  teacher dumps (fresh, none existed); soft-k3 vs hard+blend on
  held-out b5: overall MSE 0.004582 vs 0.004904, win +0.000322
  >= 0.0002; night -11%. Harness: `--soft`, holdout enforced,
  counts persisted (gap #1 closed).
- 2026-11-03 T2 done (same note): k5 weaker than k3 everywhere
  (-0.000021 overall, worse night + edge). k5 dropped, k3 kept.
- 2026-11-03 T3 CLOSE (same note): 9/9 probe screen — soft
  loses the held-out pair to hard (1.035x), both hybrids
  1.16-2.39x worse than occ7, soft 2.5x its occ9 teacher on
  fb8 (BC compounding). §4 kill: offline gain evaporates.
  T4/T5 gated shut. Line closed.
