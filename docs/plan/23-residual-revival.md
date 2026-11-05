# Plan 23 — residual revival push (round-6 synthesis)

Source: round-6 workflow (angle-0 + p99-followup + synthesis read;
angles 1-4 + critic schema-failed and were LOST -- T4-path,
tie-diag, ship, climategate survive only as synthesis ranks, which
the parent treats as unverified-until-screened). Synthesis ranked
8 candidates; parent-verified before writing: (1) bank1 weekend
heads EXACTLY 0 (W~1e-15, run-bias 1.0) -- T==B weekends;
(2) train/infer memory skew REAL (deploy slot.step update=True
always vs train update=deleg); (3) /tmp/pooled_p99.py exists with
per-channel constants (damp 0.872/reheat 0.846/SAT 0.624/press
0.046/recirc 2.102/waterT 1.160).

Shared gates: probe gate (fb8 gap-cut >= 50% vs occ7 + no >floor
regression on phx1/sg8) for revival steps; F/F3 (judge-24 first,
b8/train second, >= 0.0002, G8 3x, shipped AND honest) for any
promotion. Short runs protocol-only.

| ID | Item | Work | Gate |
|---|---|---|---|
| P23-1 | Scale ablation (RANK1, zero refit, FIRST) | Probes s in {0.05,0.3} + per-channel p99-normalized scale (consts above) on frozen slot_resid.npz; 3-pair x8640 (fb8+sg8+phx1) ~6 arms = 18 runs | Probe gate; winner → F/F3 track; all fail → KILL scale-up |
| P23-2 | Memory-skew fix | Deploy slot.step(update=deleg) to match train (deleg from AHU-run obs); 3-pair x8640 | Beats s=0.15 baseline (no-regression + any gain) else REVERT (keep skew, document) |
| P23-3 | Noise-head kill (RANK3) | Zero pressure + sg8-waterT heads (offline, no refit); scale break-test {0.15,0.5,1.0} on sg8 + HWS-cooling pair + fb8 | Self-limit claim holds (sg8-class bill diff < 2e-4 at all s) else KILL |
| P23-4 | Per-mode K=2 resid (RANK2) | OAT-band K=2 modes x2 banks x288 slots + cold scale-up; refit + 3-pair x8640 | Probe gate + no sample-starvation (min/med/zeros) else KILL |
| P23-5 | Feature augmentation (RANK4) | Add dp/SAT-cmd/u-proxy/unclipped-tw to zone+AHU heads; refit + R2 check + 3-pair x8640 | Offline R2 lift (phx1 SAT/water) w/o sg8 noise inflation, then probe gate; else KILL |
| P23-6 | DAgger-on-resid cold-only (RANK5) | FIRST commit tools/replay_base.py (B-replay writer, unblocks tooling); B on student obs; cold-only refit + screens | P21-13 revisit: cold wins + sg8 guard; dag2-pattern (non-convergent) = KILL |
| P23-7 | OAT-gated stack track (RANK6, SEPARATE) | 6-pair decision table; stage-0 liveness 6x6x1500 then x8640 survivors (P2 EXCLUDED; dmax/dwu never bundled unscreened) | 6-pair honest >= 0.0002 floor per arm; below floor = CLOSED (catastrophe-avoidance value only) |
| P23-8 | Ship hardening (RANK7, parallel, code-only) | f32 b64+zlib weights cell (182KB), self-containment (no assembler surgery), local exec-check 300+ steps 2 scenarios | P21-3 bars (no-exception, f32/f64 <1e-4 + discrete equality, determinism, p99 time); 0 sim |
| P23-9 | Doc closes | Round-6 angle loss record; bank1/skew findings to research; day notes | 0-sim; all landed |

NOT RANKED: nonlinear heads (RANK8 LAST -- only if ranks 1-4
plateau AND notebook path clears smoke); torch smoke stays
DEFERRED (no torch candidate).

Needs-external (unchanged): PR #1/#2/#3 diff bodies, forum
bodies, submission budget + rescore timing.

## Execution log

- 2026-11-05 plan written from round-6 synthesis (3/8 results
  read; angles 1-4 + critic lost to schema failure -- named
  gap: T4-path/feature-audit/tie-diag/ship/climategate have
  no dedicated evidence, synthesis ranks only). Starting
  action at P23-1.
- 2026-11-05 P23-1 done: PASS, res30
  new baseline (same note). NEW
  ResidS05/S30/PCProbes (uniform +
  parent-designed per-channel) + keys +
  8 tests green; refactor parity
  (resid 0.022739 exact). 18-run
  screen: fb8 monotone (res05 0.030070
  / resid 0.029590 / res30 0.028963;
  res30 cut 52% of occ7->occ9 gap,
  0.0014 from occ9); respc 0.029585 ~=
  resid (per-channel no win, killed);
  sg8 all tie (5e-6, self-limit holds
  at 0.3); phx1 res30 +0.00009 (below
  floor, predicted risk). Gate met.
  Winner res30 -> revival baseline
  (not promotion: still -0.0014 vs
  occ9). 161,520 sims (6,000 live +
  155,520).
- 2026-11-05 P23-2 done: REVERTED by
  analysis, 0 sim (same note). Deploy
  _chain_action runs only in non-shutdown
  modes (memory auto-frozen on shutdown)
  -- train update=deleg matches deploy
  GIVEN same modes. Mismatch (resid-vs-
  occ7 deleg 9.6%) IS closed-loop
  divergence, not an update-rule bug:
  no update rule can match unknown
  teacher-deleg in deploy (obs-run proxy
  lags a step; train-side update=True
  would be worse). Skew belongs to
  DAgger (P23-6), not to a deploy patch.
  Kept skew, documented. 0 sim.
- 2026-11-05 P23-3 done: KILLED (claim
  broken at s=1.0; same note). NEW
  ResidS50/S100Probes + keys + 8 tests
  green. Break-test 9-run: fb8 res50
  0.028820 (better than res30!) res100
  0.124747 (explosion); sg8 res50 tie
  (4e-6) res100 0.072405 (10x break);
  phx4 res50 0.037091 vs res15 0.013813
  (2.7x worse). Self-limit holds to
  0.5, breaks at 1.0: gate (all s)
  FAILED. res50 not promoted (hot
  risk); baseline stays res30. 80,760
  sims (3,000 live + 77,760).
- 2026-11-05 P23-4 done: KILLED (fb8
  worse above floor; same note). NEW
  mode_of_oat + hyst state in
  SlotFeatures, per-mode accums in
  slot_train (--modes/--mode-bounds,
  K=1 bit-exact legacy), mode-aware
  _blended in SlotHybridPolicy,
  ResidK2Probe (s=0.30) + key + 4
  tests green (301 total). Refit
  slot_resid_k2 (30 runs, holdout b5,
  soft k3): zone/ahu 1 zero-slot each
  (fallback covers), water-mode1
  370/576 fallback (semantically
  empty: no heat in warm mode).
  6-run screen vs res30: fb8 0.029268
  vs 0.028963 (-0.000305, gate fail),
  sg8 -0.000035, phx1 +0.000018 (both
  below floor); mean-3 -0.000107.
  Mechanism: sample-split cost beats
  specialization gain (mode0 zone med
  212 vs pooled 1730). K=1 holds;
  baseline stays res30. 52,140 sims
  (300 live + 51,840).
- 2026-11-05 P23-5 done: KILLED by
  offline R2 gate, 0 sim (same note).
  NEW tools/slot_r2.py (per-channel R2
  on teacher demos, deploy-blend).
  Baseline slot_resid: phx1 a.sat
  .176, w.temp -.844; sg8 a.sat .399.
  Augment (+2 zone unclipped tw/oat,
  +4 AHU satsp/sat-err/dp-err/fan, +1
  water sp-gap; prefix slices keep old
  weights bit-identical, R2-parity
  proven) + refit slot_resid_f5 + 4
  tests green (305 total): phx1 a.sat
  .176->.176 (ZERO lift, target
  channel), w.temp -.844->-.827
  (still worse than constant); sg8 no
  inflation. Mechanism: sat-err/dp-err
  weights EXACTLY 0 (teacher holds
  setpoints exactly -> degenerate in
  demos); zone lift +.003 noise. No
  screen (gate ordered offline first).
  Baseline stays res30. 0 sims.
