# Plan 22 — loss pockets + danielwuu port (round-5 synthesis)

Source: round-5 workflow (5 angles + critic + followup + synthesis, 8/8 read).
Parent-verified before writing: (1) angle-0 two-teacher theory REFUTED for
P21-13 runs — nn_data_dag1 bit-equals SlotDaggerProbe output whose teacher is
occ9 in committed code (single-teacher divergence stands); old
tools/dagger_collect.py (occ4, Oct 30) was never used for dag1/dag2.
(2) angle-2 NO-RLS correction CONFIRMED — covariance allocated once
(/tmp/dwu/policy.py:229, sha-verified archive) and never referenced; online
adaptation is the residual disturbance observer. (3) angle-4 pockets
REPRODUCED exactly from 30x occ9 reward_components.csv: total -2276.3,
comfort weighted -114.1, hour-6+7 = 76.7% of comfort, hour-6 alone 57.7%.

Shared gates (F/F3 law): judge-first (judge-24 F1 holdout), b8-second,
0.0002 floor, G8 3x any-pair guard. Short runs protocol-only.
Kill-conditions are per-item law; KILL means close without escalation.

| ID | Item | Work | Gate |
|---|---|---|---|
| P22-1 | UGAS cold-unoccupied-gas trim | Probe: lower unocc heating floor + occupancy-EMA prestart gate on occ9 (precedent zone_pi.py:745-768); screen fb1+fb8 x8640 | Same-sign win vs occ9 both pairs >= 0.0002 mean; then train-30 + judge-24 + G8; else KILL |
| P22-2 | PRE arrival-edge prestart | Probe: schedule-anchored min-prestart / earlier latch (latch exists occ_gated3.py:54-83, trigger too late); screen denver_3 + fb pair x8640 | Comfort cut with no en+co regression > floor; then train-30 + judge-24 + G8; else KILL |
| P22-3 | DWU dwusat 11x2 SAT-search port | DwusatProbe(OccEconDbhHot9Policy) post-chain: free-drift + residual observer + slow EMA + per-AHU 11 SAT x 2 recirc cost-min (_choose_ahu core); keep occ9 dampers/reheat/occ-gate/water; DWUSAT_ON flag + parity test | Liveness mixed-climate pair, then train-30 shipped-bill vs occ9 >= 0.0002; then judge-24 + G8 + determinism parity; else KILL (faithful OFF follow-up only if SAT-search sub-floor but live) |
| P22-4 | DMAX dual-max staging | Stack occ_dualmax.py DualMaxOutputMixin into occ9; screen fb1+fb8 x8640 | Same-sign win (fan + ventilation gas) both pairs; then train-30 + judge-24 + G8; else KILL |
| P22-5 | F2B one-line ablation | Clear floor.pol.target_offset={} on SkynetColdProbe fallback path; 1 run xfer_10/msp1 x8640 | Revive iff Monday regret <= 3x occ9 same episode AND full-episode same-sign win >= 0.0002 vs occ9; else F2 stays KILLED |
| P22-6 | RESID residual-ized slot | Refit delta=T-replay(B), B=occ7 T=occ9, new --residual-base flag (~30 lines); deploy cont=clip(B+delta*scale), scale from per-channel p99|T-B| start 0.15 ablate {0.05,0.15,0.3}; 0-sim refit + fb8/sg8/phx1 x8640 | fb8 gap cut >= 50% with no >floor regression elsewhere; then F/F3; unblocks P21-13 on pass; else P21-13 stays BLOCKED |
| P22-7 | OFFM mild-only OAT-gated OFF | OAT-gated N1OffMixin (fire only when OAT inside comfort band); screen >= 2 atlanta/denver pairs x8640 + sg7 check | Same-sign win vs occ9 both mild pairs AND sg7 gap within 0.0002; else KILLED (general OFF stays dead) |
| P22-8 | P2M heat-interlocked P2 | N1P2 pressure only when OAT mild (>10C) or clipped to occ9 range [-1.0,0.0]; 5 n1bo-SCEN pairs x864 + fb checks | Same-sign win vs occ9 all 5 AND no train-fb regression > floor; else KILLED |
| P22-9 | DAGX dag1 fb1/fb2 falsification | SlotDag1Probe vs soft7 on fb1+fb2 x8640 (stop if fail) | Revive iff same-sign win both with gap cut >= 50% AND no >floor regression on fb4/fb5/sg; else DAG line stays dead |
| P22-10 | Doc closes | danielwuu RLS->observer correction to research/21 + day notes; angle-4 stat scripts/paths to repo (repro); reward-weight value source cite | 0-sim; all closes landed |
| P22-11 | Torch smoke (DEFERRED) | P21-3 kernels-push smoke (torch + 200KB base64, print __version__); ONLY after P22-1..4 resolve | Smoke COMPLETE + version pinned + ARCH-10 p99<5ms (time_policy.py); else RL push stays deferred |

NOT RANKED: P22-COOL deep-cool electricity (smallest headroom, mostly
optimized) — run last, only if ranks 1-4 all close.

Needs-external (unchanged): PR #1/#2/#3 diff bodies, forum
749531/752142/753804/749896/752678/751998 bodies, submission budget balance +
rescore timing. Corrected-sim re-validation blocked on hosts.

## Execution log

- 2026-11-05 plan written from round-5 synthesis (8/8 read,
  critic corrections applied, 3 parent verifications above).
  Starting action at P22-1.
- 2026-11-05 P22-1 done: KILLED (same
  note). NEW UgasCap20Policy (floor cap
  eco_lo - 2K) + key + 3 tests green.
  Directionality record: FLOOR_CAPABILITY
  0.5/0.7/1.0 all VACUOUS bit-identical
  (gated3 caps floor at eco_lo - 1K,
  binds always; proven + offline replay
  modes). Cap20 liveness via replay-diff
  (x1500 window floor-quiet): 791
  mode-diffs + 2293 action-diffs.
  Screen fb1+fb8 x8640: fb8 -0.000152
  (win, below floor), fb1 +0.000126
  (loss, below floor) -- mixed, mean
  ~0. Components: fb1 comfort -46.8 vs
  -38.3 (deeper floor breaks recovery
  in -21C), fb8 all better but small.
  48,060 sims.
- 2026-11-05 P22-2 running. NEW
  PreMargin12Policy (margin 6->12) + key
  + 3 tests green. Liveness denver_3
  x1500 DIFFERS better (-0.0008,
  protocol). Screen denver_3+fb8 x8640:
  denver_3 -0.000515 (2.6x floor,
  comfort cut 44%), fb8 -0.000045,
  same-sign; en+co regression within
  floor both. Gate PASS -> train-30
  (518,400) running in background. No
  code edits until it lands.
- 2026-11-05 P22-2 done: CLOSED
  below-floor (same note). Screen PASS
  (denver_3 -0.000515 2.6x floor,
  comfort -44%; fb8 -0.000045; en+co
  within floor). Train-30 30/30 clean:
  prem12 0.00871097 vs occ9 0.00878189
  = +0.00007092, 22/30 wins, 4
  above-floor wins (denver_3 +0.000515,
  phx4 +0.000423, fb4 +0.000308,
  atlanta_3 +0.000201), 1 above-floor
  loss (denver_5 -0.000315). Mean below
  floor: no judge-24 (F2 law), no
  promotion. Lever live but heterogeneous;
  climate-conditional margin queued for
  research. 555,960 sims (3,000 live +
  34,560 screen + 518,400 train-30).
- 2026-11-05 P22-3 running. NEW
  DwusatProbe (phase 1: drift + observer
  + band + 11x2 search; SAT/recirc/
  pressure post-chain, occ9 rest) + key
  + 3 tests green. Fixed reset-before-init
  crash (guard). Parity DWUSAT_ON=False:
  0/1235 replay mismatches. Liveness
  x1500 mixed (atl1 better -0.00047,
  fb8 worse +0.00037, protocol).
  Train-30 (518,400) running in
  background. No code edits until it
  lands.
- 2026-11-05 P22-3 done: phase-1 KILLED
  (same note). DwusatProbe + key + 3
  tests green; parity 0/1235. Train-30
  30/30: dwu 0.00985167 vs occ9
  0.00878189 = -0.00106978 (5x floor,
  wrong sign), wins 10/30. Hot pairs
  catastrophe (phx5 -0.0061, phx4
  -0.0047, phx6 -0.0042, sg4 -0.0025);
  mild wins only (+0.0003..+0.0005 x5).
  Mechanism: search rides SAT 5K warmer
  (phx5 supply 296.9K vs 291.8K) --
  without joint dampers/reheat (phase-1
  keeps occ9's) warm supply + open
  dampers = comfort collapse. Mapping
  verified correct (spec ranges match).
  Super-floor loss (not sub-floor):
  no phase-2 OFF. Full-joint port +
  climate-gate queued for research.
  524,400 sims (6,000 live + 518,400).
- 2026-11-05 P22-4 done: KILLED (same
  note). NEW OccDmax9Policy (dual-max
  staging in occ9, fixed split 0.5/min
  0.1) + key + 3 tests green. Screen
  fb1+fb8 x8640: fb1 0.060201 vs
  0.050064 (-0.0101), fb8 0.031267 vs
  0.027586 (-0.0037) -- same-sign
  WRONG, 18-50x floor. Mechanism:
  min-airflow heating under the 308.15K
  discharge cap under-delivers heat in
  deep cold (reheat-first cannot
  compensate capped air). Dual-max is a
  mild-climate lever at best. 37,560
  sims (3,000 live + 34,560).
- 2026-11-05 P22-5 done: F2 stays KILLED
  (same note). NEW SkynetColdBProbe
  (fallback + target_offset clear) + key
  + 6 tests green. Decisive 1-run
  ablation xfer_10/msp1 x8640: F2b
  0.026814 == F2 0.026814 BIT-IDENTICAL
  all 3 CSVs. Stale-offset causation
  REFUTED (consistent with F7: no target
  head in ckpt, offset never set, clear
  is no-op). F2 degradation is
  residual-debt/underheat, not stale
  shifts. Gate unmet (22.7x stands).
  8,640 sims.
- 2026-11-05 P22-6 done: PASS (same
  note). Residual-ize WORKS. --residual-
  base flag in slot_train.py (delta=T-B
  targets, run cols keep T) + 30x B
  replays (occ9 self-replay MSE=0 bit-
  exact; mean|T-B|=0.17 p99=1.18) +
  slot_resid.npz refit + ResidualSlotProbe
  (clip(base+delta*0.15)) + key + 7 tests
  green. 9-run screen: fb8 resid 0.029590
  vs soft7 0.069742 vs occ7 0.030471
  (gap cut 102%, beats occ7 by 0.00088);
  phx1 0.004730 vs 0.004732 (tie 2e-6);
  sg8 0.007362 vs 0.007364 (tie 2e-6).
  Gate met (cut >=50%, no >floor
  regression). Mean-3 beats occ7 by
  0.000295 (T4 bar). P21-13 UNBLOCKED.
  82,260 sims (4,500 live + 77,760;
  refit/replay 0-sim).
- 2026-11-05 P22-7 done: CLOSED
  below-floor, mechanism confirmed (same
  note). NEW N1OffMildProbe (OAT-gate
  in mixin, default off = existing
  probes unchanged) + key + 9 tests
  green. Screen 2 mild + sg7 x8640:
  atlanta_2 +0.000026, denver_7
  +0.000028 (same-sign wins, 7x below
  floor); sg7 BIT-IDENTICAL all 3 CSVs
  (gate fully blocks hot fire; ungated
  off lost -1.18 there). Gate conditions
  met but below floor: revive confirmed,
  no escalation. 54,840 sims (3,000
  live + 51,840).
- 2026-11-05 P22-8 done: KILLED (same
  note). NEW N1P2MildProbe (OAT >= 10C
  gate, default None = existing probes
  unchanged) + key + 11 tests green.
  SCEN-5 x864: fb1/fb2/fb8 TIES (gate
  blocks cold fire, explosions gone)
  BUT atlanta_2 -0.000545 and sg7
  -0.000168 LOSE. Gate needs all-5 win:
  FAIL, no stage-2 fb checks. P2 dead
  everywhere: cold = explosion, mild =
  loss, hot = loss. 8,640 sims.
- 2026-11-05 P22-9 done: CLOSED,
  cold-transfer confirmed, revival gate
  failed (same note). Stage-1 fb1+fb2
  x8640: dag1 wins both, gap cut 93.4%
  (fb1: 0.0586 vs soft7 0.1586 vs occ7
  0.0516) and 50.5% (fb2: 0.0252 vs
  0.0344 vs 0.0161) -- fb8-fix DOES
  transfer cold-to-cold. Stage-2
  skipped 0-sim: sg8 poisoning already
  proven (/tmp/dag9 dag1 0.108 vs soft7
  0.0085), revival gate AND-condition
  unmeetable. DAG-overwrite line stays
  dead (resid dominates anyway).
  51,840 sims (34,560 + 17,280 occ7).
