# 2026-10-27

## Status

- Leaderboard: leader ~0.00521 (Joe Penn), median ~0.0142, us 0.01323 (occ4).
- Submissions: occ4 COMPLETE 0.01323; occ5/occ6/occ7 PENDING (queue ~14h).
- Hidden/train ratio for occ4: 1.37 (hidden is tail-heavier than train).

## Changed

- `smart_buildings/baselines/occ_gated6.py` (new): round-4 + OAT-aware
  trim-and-respond static pressure (respond on starved dampers, persistence 4,
  deadband 1.0K + 0.1K per degree of |OAT-band| over 10K, cap 0.0).
- `smart_buildings/baselines/occ_gated7.py` (new): occ6 with occupied-only
  trigger (no prestart response).
- `smart_buildings/baselines/probe_modes.py`: `probe_ptr` (tunable trim-respond),
  `probe_guard` (kept for reference; proven no-op — see below).
- `smart_buildings/transfer_scenarios/` (git-ignored, regenerable):
  4 new buildings x 4 unseen weathers; xfer_4 dropped (125 zones, 31/AHU —
  out of spec, 2h/sim).
- `tools/`: `gen_transfer.py`, `fullrun.py`, `fullrun_t.py`, `xfer_analyze.py`,
  `assemble_notebook.py` (repo-relative paths).
- Notebooks `occ6`/`occ7` assembled, 300-step exec check green, kernels pushed.

## Measured

- Train-30: occ4 0.009629; occ6 0.009695 (+0.7%, cold-only misfires).
- Transfer-12: occ6 -13.0% vs occ4 (xfer_3: -20% to -34% in ALL weathers).
- v5 (occ7) on 6 pairs: tails -23%/-34%, all misfire pairs bit-equal to occ4.
- Baselines on train: pid_water 4-12x worse than occ4; bang_bang 8-60x worse.

## Decided

- Root cause of big-zone bleed: pressmin (-1.0) starves airflow; AHU has NO
  heating coil (cooling only), VAV discharge capped at 308.15K — heat delivery
  is purely airflow-limited. Confirmed via dump (water 350K max, dampers open,
  zones 13-18C).
- GuardProbe (force normal when occupied + zone >3K out) is a structural no-op:
  occupied mode is already normal. Deleted from plans, kept in code.
- Comfort slope (~0.18/K) >> energy slope (~0.002/K): never trade comfort for
  energy. occ5 (-2K heating) directionally wrong on train math; pending score
  is free info only.
- TOU arbitrage dead: peak/night price 1.25x < thermal storage round-trip cost.
- All sim energy paths are metered (blower, chiller, pump, boiler, ASHP) —
  no exploit possible. Leader wins on control.
- Per-zone occupancy is NOT in obs (building-level only) — zone-level gating
  impossible.
- Kernel failure lesson: notebook assembler must include `occ_gated:<consts>`
  (`STEPS_PER_HOUR`); local exec check must cover night steps (300+, not 12).

## Open

- occ5/occ6/occ7 public scores (attribution: transfer fix value on hidden?).
- Remaining gap 0.013 -> 0.005: likely hidden tails + mild efficiency; needs
  score feedback to direct.
- 1 submission left today (after occ7).
