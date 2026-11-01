# Online RL Policy Architectures for Cross-Building Transfer

Date: 2026-11-01
Status: draft
Scope: Policy-network architectures for online RL over the rule-based
controller occ_gated7 in the Kaggle NeurIPS 2026 Smart Buildings challenge:
shared per-entity MLPs, temporal-conv + recurrent (LSTM/GRU) policies,
transformer / in-context policies, and morphology-aware (per-type
encoder/decoder) architectures. Transfer numbers, validated settings, failure
modes (aliasing, input overfit, large-net degradation), and limits of transfer
for each claim. No simulators, training, or installs were run for this doc;
every number below carries provenance (file path or URL).

## Sources

| ID | Source | Type |
|----|--------|------|
| S1 | https://www.mendeley.com/catalogue/cb6e23e3-e206-34ee-827a-2fd1dafa1bbb/ | shared per-zone MARL library transfer (e-Energy 2022 record) |
| S2 | https://arxiv.org/html/2411.19746v1 | HVAC-DPT decision pretrained transformer (arXiv 2411.19746) |
| S3 | https://arxiv.org/html/2609.05822 | NOMAD-RL domain-randomized recurrent RL (arXiv 2609.05822) |
| S4 | https://rlj.cs.umass.edu/2026/papers/Paper103.pdf | Building2Building benchmark (RLJ 2026 Paper103) |
| S5 | https://github.com/vtaboga/building2building/blob/HEAD/docs/baselines/cross-domain.md | Building2Building cross-domain Amorpheus baseline doc |
| S6 | https://arxiv.org/html/2008.03625v2 | One for Many transfer learning (arXiv 2008.03625) |
| S7 | https://arxiv.org/html/2610.19790 | Decentralised per-zone control under partial observability (arXiv 2610.19790) |
| S8 | https://ojs.aaai.org/index.php/AAAI/article/download/30219/32168 | Active RL for robust building control (AAAI) |
| S9 | https://www.alphaxiv.org/overview/2307.08794v1 | Multi-timescale MARL overview (Emami et al. 2307.08794) |
| S10 | https://mila.quebec/en/article/neural-differential-equations-for-temperature-control-in-buildings-under-demand-response | Neural ODE vs SSM building models (Mila) |
| S11 | https://www.scribd.com/document/661918739/PowerCon2023-TSAC | Transformer-SAC HEMS summary (secondary, weak provenance) |
| S12 | https://freederia.com/ai-driven-predictive-maintenance-and-adaptive-energy-optimization-for-commercial-hvac-systems-a-commercially-viable-framework-for-environmental-management/ | Conv-LSTM + PPO hybrid (self-declared synthetic record) |
| S13 | /tmp/rl_zip/nets.py | Our M1 FlatNets: shared per-entity residual MLPs |
| S14 | /tmp/rl_zip/features.py | Our per-entity features: fixed scales, pooled context, zone-target head |
| S15 | smart_buildings/nn_model.py (lines 1-140) | Our FeatureContext room/AHU/global decoding |
| S16 | smart_buildings/baselines/nn_policy.py (lines 1-80) | Our BcV0Policy windowed transformer controller |

Evidence-surface note: S1-S12 bodies were inspected via the scout
evidence bundles at
`/Users/admin/.local/share/muse/sessions/2026/11/01/01a19822-f55c-7551-bdbc-cd806a3ee19f/tool-outputs/readable-text/8154ad0c139e3871923a01f45b2dc9854051fe2d4b37f1a40d123ac32f41c5f4.txt`
and
`/Users/admin/.local/share/muse/sessions/2026/11/01/01a19822-f55c-7551-bdbc-cd806a3ee19f/tool-outputs/readable-text/42641ec471e2087178e19d0d394643f652d75416802c7a1ae23ac4506ccd5066.txt`
(resolved result 2 section). S13-S16 were read directly with read_file.
S11 and S12 are low-trust sources; claims resting on them are marked.

## Numbered findings

### A. Shared per-entity MLP policies

A1. A library of shared per-zone MARL policies (PPO MLPs) transfers zero-shot
to a different target building by assigning the best pretrained library
policy to each target zone: HVAC energy drops 40.4% vs a fixed-schedule
baseline and 48.97% vs agents trained on the target building itself for
5,000 months, with no retraining on the target.
Provenance: S1.
Transfer limit: cross-building but within the studied simulator family;
assignment needs a per-zone selection step on the target (not a single
frozen forward pass); comfort-constraint handling is not quantified in the
inspected excerpt.

A2. Our M1 baseline (S13) implements the same family directly: one shared
zone MLP (a copy per VAV), one shared AHU MLP (a copy per AHU), one plant
MLP, each reading only its own entity features. Defaults are width d=64,
depth=2, tanh activations; residual heads are zero-initialized so an
untrained policy equals the base controller exactly; per-head Gaussian
log-std initializes at -1.0. The critic is separate: mean-pooled entity
embeddings (3*d) through Linear(d)+tanh+Linear(1). Batches pad to the
largest building; masks exclude padded entities from every sum, mean, and
loss; a latch mask (vmask) excludes switch votes held by the latch from
log-probability and entropy.
Provenance: S13 (`FlatNets`, `masked_mean`, `log_prob_entropy`, `pad`,
`pad_vmask` in /tmp/rl_zip/nets.py).
Transfer limit: not yet measured; this is our architecture under test, not
a reported transfer result.

A3. The M1 feature contract (S14) that makes one weight set serve any
building: fixed physical scales per field type (e.g. zone/AHU temps as
(K-295)/10, pressures/1000, flows/3.0, counts/10-30, occupancy/125), no
running statistics ("they would be per building and would have to ship
with the weights"), values clipped to [-10, 10]. Zone rows carry own
sensors plus band errors, PID damper/reheat proposal, and AHU supply-temp
plus run context; AHU rows carry pooled zone signals (mean error, cold/warm
fractions, mean PID damper/reheat, worst-zone error); plant rows carry
pooled AHU signals. Residual composition adds scaled deltas to the PID
proposal; switch votes use a dead zone at +-0.5 (defer to PID inside).
Provenance: S14 (module docstring, `ZONE_FIELDS`/`AHU_FIELDS`/`PLANT_FIELDS`,
`BuildingIndex.features`, `compose` in /tmp/rl_zip/features.py).
Transfer limit: scales were fit on building_train_8 (atlanta_1 night,
phoenix_4 day, 2026-10-25); unseen buildings/climates outside those ranges
hit the clip rails — the clip rate on transfer is an open measurement.

A4. Our shipped BC stack (S15, S16) already validates the per-entity
decoding pattern in-repo: `FeatureContext` decodes room features [Z, 8],
AHU features [A, 10], and a global vector (occupancy, comfort flags,
hour-of-day/day-of-week harmonics, HWS) from `building_meta_data` alone,
with AHU wiring in numeric-suffix order truncated to reportable temp
indices; `BcV0Policy` runs one tiny CPU forward pass per step (d=64
transformer over window tokens, batch B=1).
Provenance: S15 (module docstring, `FeatureContext`, SCALE_K=10.0,
DIFF_K=8.0 in smart_buildings/nn_model.py), S16 (module docstring, ring
buffer + `ControllerNet` forward in smart_buildings/baselines/nn_policy.py).
Transfer limit: BC, not RL; imitation quality bounds the policy and no
cross-building RL transfer number exists for it yet.

### B. Temporal-conv + recurrent (LSTM) policies

B1. NOMAD-RL (S3) is the validated reference for this family: forecast
trajectories are encoded with a temporal convolution module, fused with
current measurements (zone temp, power, comfort bounds, outdoor,
occupancy, price, time features, plus short-term forecasts of exogenous
inputs) into a compact latent, then fed to a recurrent LSTM-based
actor-critic trained with Recurrent PPO (SB3-Contrib). Actions are
supervisory thermostat setpoints. Motivation is explicit: HVAC is a POMDP
— the controller never observes full thermal state and must infer hidden
dynamics from interaction history.
Provenance: S3.
Transfer limit: validated within a flow-based domain-randomization
distribution over 25 thermal parameters (see B2), not across arbitrary
unseen buildings.

B2. NOMAD-RL transfer numbers after a 6M-step run: single-zone nominal
-70.4% cost vs PID but +8.7% vs non-randomized recurrent PPO (robustness
tax on the nominal zone); shifted sampled zone -60.1% vs PID and -51.4%
vs non-randomized PPO but +46.6% vs exact-model MPC; 3-zone coupled
building -67.9% vs constant-setpoint PID and +14.5% vs per-zone MPC, with
the MPC gap narrowing as complexity grows.
Provenance: S3.
Transfer limit: the +8.7% nominal tax and the remaining +46.6%/+14.5% MPC
gaps bound what recurrence + randomization buys; no zero-shot
cross-morphology (different zone counts/AHU layouts) number is reported.

B3. No direct MLP-vs-LSTM ablation for 5-minute commercial HVAC was found
in the inspected pages: NOMAD-RL compares recurrent PPO with vs without
domain randomization, not vs a feedforward MLP.
Provenance: scout notes in the 42641ec4 bundle (resolved result 2, notes
field).
Transfer limit: the marginal value of recurrence over a well-featurized
MLP on our 5-minute step is unmeasured — this is the central ablation for
our track.

B4. Discrete-time building models are State Space Models implemented as
RNN + encoder/decoder with lagged observations encoded in a latent state;
continuous-time Neural ODE vector-field models are reported as more
sample-efficient, while both work for MPC planning.
Provenance: S10.
Transfer limit: a modeling (not control) result; sample-efficiency claim
has no control-transfer number attached.

### C. Transformer / in-context policies

C1. HVAC-DPT (S2) is a GPT-2 causal transformer (3 layers, 8 heads,
embedding dim 128, MSE loss, AdamW, 118 epochs) pretrained on 100 diverse
PPO teacher policies (2x64 tanh MLPs, eps=0.2, lr=3e-4). It transfers
from a 5-zone small office to a 15-zone medium office in the same city
(Denver): -45.62% yearly HVAC energy vs a 50%-damper baseline, +5.78% vs
a per-building Expert controller, -70.56%/-74.12% vs SARL/MARL trained on
the target. No parameter updates at deployment (in-context adaptation).
Provenance: S2.
Transfer limit: same-city transfer only (Denver->Denver); episodes are 1
month at 15-minute steps (not our 30-day 5-minute episodes); reward is
energy-only with comfort left to the EnergyPlus controller (incompatible
with our comfort-weighted reward); +5.78% vs the per-building Expert caps
the zero-shot claim.

C2. A transformer-based Soft Actor-Critic for home energy management
treats control as a POMDP over observation history and uses a transformer
policy to leverage history without loss, avoiding catastrophic forgetting
reported for RNNs.
Provenance: S11 (secondary Scribd summary of PowerCon2023-TSAC; the
primary TechRxiv preprint was not fetchable — 403/empty).
Transfer limit: weak provenance (office-grade discussion source, not
primary evidence); home (not commercial multi-zone) setting; no transfer
numbers inspected.

C3. NOMAD-RL authors list transformers and larger architectures as future
work for HVAC generalization, which implies LSTM + temporal-conv is the
currently validated choice in that line and transformers are not yet
validated there.
Provenance: S3.
Transfer limit: by definition — no transformer number from this source.

C4. Our in-repo BC controller (S16) already runs a small transformer
(d=64 over window tokens, WINDOW=72 steps pooled to N_HIST=12 history
slots at STRIDE=6, warmup clamps to step 0) in a single CPU forward pass
per step — proof that a small attention policy fits our CPU budget.
Provenance: S15 (WINDOW=72, STRIDE=6, N_HIST=12 in
smart_buildings/nn_model.py), S16 (d=64 transformer, B=1 CPU forward in
smart_buildings/baselines/nn_policy.py).
Transfer limit: BC inference only; no RL training or transfer number.

### D. Morphology-aware architectures

D1. Fixed-size MLPs are incompatible with heterogeneous building
observation/action spaces; the Building2Building cross-domain baseline
(S4, S5) therefore uses an Amorpheus variant: per-node-type linear
encoders -> shared acausal transformer layers -> per-node-type decoders
producing per-node Beta-distribution action parameters, trained with
custom PPO (GAE, KL early stopping), embed_dim=64. Building-type one-hot
is appended to local observations.
Provenance: S4, S5.
Transfer limit: architecture enables cross-morphology training; it does
not guarantee beating baselines (see D2).

D2. That Amorpheus policy, trained 1M steps on 4 buildings (one per type)
with resampling every 10 PPO updates, adapts to 20 unseen test buildings
and beats the reactive baseline on some of them (normalized return < 1.0;
dynamic-setpoint tracking task with w_E=0). In dynamics adaptation,
per-building specialists beat generalized policies with a substantial
gap, and generalized policies show a long tail of transfer failures on a
few of 100 test houses.
Provenance: S4.
Transfer limit: "beats baseline on some" plus a long failure tail is the
honest shape of morphology transfer; specialists still win substantially
per building; the w_E=0 tracking task differs from our comfort+energy+
carbon reward.

D3. One-for-Many (S6) splits the policy into a building-agnostic front-end
plus a building-specific back-end: zero-epoch transfer from a 4-zone
source to 5-zone/7-zone targets satisfies the <5% violation criterion
(cost 339.73/332.07 vs ON-OFF 373.90/392.56), while DQN trained from
scratch on the target still violates at 150 epochs.
Provenance: S6.
Transfer limit: needs a building-specific back-end (a per-target
component, unlike our single-frozen-policy setting); zone-count transfer
only (4->5/7), not full morphology/weather transfer.

D4. M1 (S13/S14) is morphology-aware by construction without attention:
shared per-entity weights + padding/masking to the largest building in
the batch + mean pooling over members (AHU members, all zones). This is
the lightweight alternative to Amorpheus-style per-type transformers.
Provenance: S13 (`pad`, `masked_mean`), S14 (`ahu_members` mean pooling).
Transfer limit: pooling discards which-zone-is-which beyond the pooled
statistics plus worst-zone error; whether attention over entities beats
mean pooling on our buildings is unmeasured.

### E. What breaks: aliasing, overfit, degradation

E1. Per-zone partial-observability aliasing: a zone agent cannot
distinguish changes caused by its own thermal load from changes induced
by other zones on the same air handler and must infer them from local
observations; supply-air conditions and other zones' demand act as
unobserved latent disturbances. CTDE multi-agent RL absorbs this coupling
into network weights rather than representing it explicitly, yielding a
policy that is hard to inspect.
Provenance: S7.
Transfer limit: this is the core argument for giving zone nets explicit
AHU context (as M1 does via supply-temp + run state, S14) or recurrence —
a context-free per-zone MLP cannot resolve the alias in principle.

E2. At the 5-minute HVAC timescale, time-unaware agents suffer observation
aliasing: identical partial observations at different times need different
actions (other agents' phase, periodic environment patterns). Generic
recurrent networks must learn temporal dependencies from scratch and are
reported as inefficient vs explicit periodic time encoding with
Phase-Functioned Neural Networks; the inspected overview cites Recurrent
COMA at 75% success on an easy variant and 0% on a hard variant.
Provenance: S9.
Transfer limit: MARL coordination setting, not single-policy HVAC
transfer; supports our hour-of-day/day-of-week harmonic features (S14
GLOBAL_FIELDS, S15) as the cheap fix before reaching for recurrence.

E3. HVAC RL overfits to its training environment: agents are brittle and
fail to generalize to new settings (e.g. normal-weather training vs
extreme-weather test), which motivates unsupervised environment design /
environment-generation curricula for robustness.
Provenance: S8.
Transfer limit: qualitative (no numbers in the inspected excerpt); the
practical corollary is that input-distribution shift (weather, occupancy,
price) must be part of the training distribution, not just building
morphology.

E4. Weather transfer is asymmetric: transfer from large-range high-variance
weather (Riverside) to small-range low-variance weather (Buffalo, Los
Angeles) works, but not vice versa.
Provenance: S6.
Transfer limit: single-source claim; dictates training-weather ordering
(train on the widest-variance climates available).

E5. Input-scale overfit risk is concrete for M1: fixed scales fit on one
building x two weather snapshots (S14 module docstring) will rail-clip
([-10, 10]) on out-of-range transfer inputs; running statistics are
rejected as unshippable (per-building state). Either failure (clip
saturation or stale running stats) silently degrades the policy on unseen
buildings.
Provenance: S14.
Transfer limit: the clip rate and the scale-coverage of transfer
scenarios are unmeasured.

E6. Large-network degradation has two inspected data points. (a) The
NOMAD-RL robustness tax: the randomized recurrent policy costs +8.7% vs
the non-randomized recurrent policy on the nominal zone — capacity spent
on robustness hurts the training distribution. (b) The specialist gap:
per-building specialists beat generalized cross-domain policies
substantially (S4), i.e. one big shared net underfits each building
relative to a dedicated small one.
Provenance: S3 (a), S4 (b).
Transfer limit: neither is a pure "bigger net = worse" ablation; both
constrain how much generality to buy with shared capacity.

E7. A deployed-style hybrid reference point (weak provenance): Conv-LSTM
encoder over a 30-second window into a 256-d latent z_t, then an MLP PPO
policy (512, 256 units, gamma=0.99) on {z_t, faults, setpoints} with a
safety-rule override.
Provenance: S12 — the source page labels itself synthetic/unverified;
treat as discussion only, not evidence.
Transfer limit: none — no transfer numbers and no verification.

## Open questions

1. What is the marginal value of recurrence (GRU/LSTM latent) over the
well-featurized M1 feedforward MLP on 30-day 5-minute episodes? No
inspected source ablates this (B3).
2. Does attention over entities (Amorpheus-style, D1) beat M1 mean pooling
(D4) on our buildings, within the CPU budget?
3. What is the M1 feature clip rate ([-10, 10] rails, A3/E5) on transfer
scenarios, and which fields saturate first?
4. Can a 5-zone->15-zone in-context result (C1) be reproduced under our
comfort-weighted reward and 5-minute steps, or is it an artifact of the
energy-only reward?
5. How large is the specialist-vs-generalist gap (D2/E6) on our train set,
i.e. what does one shared policy cost vs per-building heads?
6. Does training-weather ordering (wide-variance first, E4) measurably
change transfer loss on our hidden climates?
7. What CPU budget per step (forward-pass ms on the eval machine) bounds
policy size, and where do d=64 MLP vs GRU-latent vs d=64 transformer sit
relative to it (A4/C4)?
8. Is the zone-target offset head (TARGET_K=1.0 K per unit, S14) necessary,
or do damper/reheat residuals suffice?
