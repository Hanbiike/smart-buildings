# AGENTS.md — neurips-2026-smart-buildings

Kaggle competition: one HVAC policy for unseen commercial buildings.
Leaderboard = mean |per-step reward| over 30 hidden (building, weather) pairs.
Lower is better.

## Environment

- Python: `./.venv/bin/python -B ...` (never system python).
- Full episode = 8640 steps (30 days x 5 min). Short runs are not comparable.
- Loss from CSV: `abs(df.agent_reward.sum()) / len(df)`.
- Simulator is deterministic: same policy + scenario = bit-identical result.

## Evaluate

- Policies live in `smart_buildings/baselines/`, registered in `POLICY_MAP`
  in `smart_buildings/evaluate.py`.
- Batch driver: `tools/fullrun.py <results> 8640 <conc> <timeout> <policies> <pairs>`
  (train) and `tools/fullrun_t.py ... <scenarios_dir>` (transfer).
- Keep total concurrent sims <= 10 or jobs time out (use 1500s+ timeouts).
- Never edit policy code while a validation run using it is in flight
  (fresh subprocesses import at start — mixed versions corrupt results).

## Submissions

- Assemble notebooks only via `tools/assemble_notebook.py` (verbatim class
  blocks). MUST include `occ_gated:<consts>` — module constant
  `STEPS_PER_HOUR` is required at runtime (kernel ERROR without it).
- Local exec check must run 300+ steps (cover night/shutdown/prestart);
  12 steps once missed a night-only NameError.
- Flow: `kaggle kernels push` -> wait COMPLETE -> `competitions submit -k ... -v N -f submission.parquet`.
  Scoring queue is slow (~14h). Watch the daily submission budget.

## Simulator facts (verified, don't re-derive)

- AHU has NO heating coil (cooling only); all heating is VAV reheat.
- VAV discharge capped at 308.15K; recirculation capped at 0.70; dampers >= 10%.
- Reward weights (comfort, energy, carbon) = (0.6, 0.2, 0.2); comfort slope
  (~0.18/K) >> energy slope — never trade comfort for energy.
- Electricity TOU peak/night = 1.25x — load-shifting arbitrage is dead.
- Plant efficiencies are constant (boiler const, chiller COP 3.5).
- Obs has building-level occupancy only — no per-zone gating possible.
- Buildings are valid iff <= 10 zones/AHU; monster buildings simulate
  catastrophically slowly (check zone count before a suite).

## Conventions

- Reply in Russian; preserve paths/commands/scores verbatim.
- Throwaway probes go in `probe_modes.py` + `POLICY_MAP`; promote winners to
  `occ_gatedN.py` with fixed constants (no env vars in shipped code).
- Transfer scenarios live in `smart_buildings/transfer_scenarios/`
  (git-ignored, regenerate via `tools/gen_transfer.py` with fixed seeds).
- Record each active day in `docs/YYYY/MM/DD/README.md` (see `docs/README.md`),
  negative results included.
