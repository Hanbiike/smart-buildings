# -*- coding: utf-8 -*-
"""Bayesian-optimization ask-tell driver for the frozen gen-4 knob stack.

Rounds 4-6 (manual single-axis probes) mapped the KP/WATER valley but could
not reach the -5% submit gate (-2.0% exact at best; gap 4.3 pp of base). This
driver replaces manual probes with a Gaussian process over the whole 7-knob
subspace, warm-started from every config already measured on the 5-scenario
screen (fairbanks_1, fairbanks_2, fairbanks_8, atlanta_2, singapore_7).

Workflow (bayes_opt 3.4, manual ask-tell with f=None per the repo skill):
  1. warm-register all measured configs - exact targets re-derived from the
     on-disk reward CSVs with one shared procedure;
  2. loop FRESH times: suggest -> write g4bo_params.json -> evaluate the
     screen via tools/parallel_eval.py -> register the exact target ->
     save_state;
  3. if a fresh point beats the best warm config by >= CONFIRM_MARGIN on the
     screen, auto-run the full 30-table and print the gate verdict.

All knobs are plain class attributes on the gen-3.1 chain (no identities).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from bayes_opt import BayesianOptimization

ROOT = Path(__file__).resolve().parent.parent
PARAMS_JSON = ROOT / "smart_buildings" / "results" / "g4bo_params.json"
BO_STATE = ROOT / "smart_buildings" / "results" / "g4bo_state.json"
RESULTS = ROOT / "smart_buildings" / "results"

# The 7 live knobs on the gen-3.1 chain; bounds from rounds 1-6 probes.
PBOUNDS = {
    "kp": (0.025, 0.095),
    "ki": (0.002, 0.0065),
    "water": (0.45, 1.0),
    "mild": (-0.55, -0.2),
    "hgate": (0.04, 0.12),
    "shift": (0.0, 1.5),
    "ndrop": (0.0, 3.0),
}
# gen-3.1 standing values (DsatPMShiftDownPolicy carries TARGET_SHIFT 1.0).
DEFAULTS = {
    "kp": 0.07, "ki": 0.0035, "water": 1.0, "mild": -0.4,
    "hgate": 0.05, "shift": 1.0, "ndrop": 2.0,
}
# Measured-config knob overrides (only axes a config actually changed).
WARM_OVERRIDES = {
    "dsat_gen31": {},
    "g4_kp08": {"kp": 0.08},
    "g4_kp06": {"kp": 0.06},
    "g4_kp05": {"kp": 0.05},
    "g4_kp045": {"kp": 0.045},
    "g4_kp04": {"kp": 0.04},
    "g4_kp035": {"kp": 0.035},
    "g4_kp03": {"kp": 0.03},
    "g4_water08": {"water": 0.8},
    "g4_w07": {"water": 0.7},
    "g4_w06": {"water": 0.6},
    "g4_mild025": {"mild": -0.25},
    "g4_heatgate08": {"hgate": 0.08},
    "g4_shift15": {"shift": 1.5},
    "g4_nightdrop3": {"ndrop": 3.0},
    "g4_m25w08": {"mild": -0.25, "water": 0.8},
    "g4_kp06w07": {"kp": 0.06, "water": 0.7},
    "g4_kp06w08": {"kp": 0.06, "water": 0.8},
    "g4_kp05w07": {"kp": 0.05, "water": 0.7},
    "g4_m25w08kp06": {"mild": -0.25, "water": 0.8, "kp": 0.06},
    "g4_m25w08kp05": {"mild": -0.25, "water": 0.8, "kp": 0.05},
}
SCEN = [
    ("building_train_1", "fairbanks_1"),
    ("building_train_1", "atlanta_2"),
    ("building_train_2", "fairbanks_2"),
    ("building_train_8", "fairbanks_8"),
    ("building_train_8", "singapore_7"),
]
STEPS = 864
CONFIRM_MARGIN = 1e-4   # screen-sum improvement over best warm config


def screen_target(policy: str) -> float:
    """Exact screen sum |total reward| / steps over the 5 scenarios."""
    total = 0.0
    for building, weather in SCEN:
        csv = RESULTS / building / weather / policy / "reward_components.csv"
        total += abs(pd.read_csv(csv).agent_reward.sum()) / STEPS
    return total


def vec(overrides: dict) -> dict:
    v = dict(DEFAULTS)
    v.update(overrides)
    for key, (lo, hi) in PBOUNDS.items():
        assert lo - 1e-9 <= v[key] <= hi + 1e-9, (key, v[key])
    return {k: float(v[k]) for k in PBOUNDS}


def run_screen(policy: str) -> None:
    """Evaluate one config on the 5-scenario screen (3 chained batches)."""
    groups = [("building_train_1", "fairbanks_1,atlanta_2"),
              ("building_train_2", "fairbanks_2"),
              ("building_train_8", "fairbanks_8,singapore_7")]
    for building, weathers in groups:
        subprocess.run(
            [sys.executable, "tools/parallel_eval.py",
             "--policies", policy, "--buildings", building,
             "--weathers", weathers, "--steps", str(STEPS),
             "--concurrency", "6"],
            cwd=ROOT, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )


def run_full(policy: str) -> None:
    buildings = ",".join(f"building_train_{b}" for b in (1, 2, 4, 5, 7, 8))
    subprocess.run(
        [sys.executable, "tools/parallel_eval.py",
         "--policies", policy, "--buildings", buildings,
         "--weathers", "all", "--steps", str(STEPS), "--concurrency", "6"],
        cwd=ROOT, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def main() -> None:
    fresh = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    bo = BayesianOptimization(
        f=None, pbounds=PBOUNDS, random_state=42,
        allow_duplicate_points=True,   # deterministic evals, skip no-crash rule
    )

    print("== warm registration (exact CSV targets) ==")
    best_warm, best_warm_vec = None, None
    for name, overrides in WARM_OVERRIDES.items():
        target = screen_target(name)
        bo.register(vec(overrides), -target)
        print(f"  {name:16s} sum={target:.6f}")
        if best_warm is None or target < best_warm:
            best_warm, best_warm_vec = target, vec(overrides)
    print(f"best warm: {best_warm:.6f}  <- {best_warm_vec}")

    print(f"== {fresh} fresh BO evaluations ==")
    best_fresh, best_fresh_vec, best_fresh_sum = None, None, None
    for it in range(fresh):
        params = bo.suggest()
        PARAMS_JSON.write_text(json.dumps(params))
        run_screen("g4bo_params")
        target = screen_target("g4bo_params")
        bo.register(params, -target)
        bo.save_state(str(BO_STATE))
        print(f"  it{it:02d} sum={target:.6f} params="
              + json.dumps({k: round(v, 4) for k, v in params.items()}))
        if best_fresh_sum is None or target < best_fresh_sum:
            best_fresh_sum, best_fresh, best_fresh_vec = target, it, params

    champ = best_fresh_sum if best_fresh is not None else best_warm
    champ_vec = best_fresh_vec if best_fresh is not None else best_warm_vec
    print(f"BO best: {champ:.6f} | warm best: {best_warm:.6f}")
    if best_fresh is not None and best_fresh_sum <= best_warm - CONFIRM_MARGIN:
        print("== auto-confirm: full 30-table on the champion ==")
        PARAMS_JSON.write_text(json.dumps(champ_vec))
        run_full("g4bo_params")
        full = screen_target  # same routine, but SCEN covers only 5 -> reuse:
        # full-table target computed by the shared exact routine below
        total, n = 0.0, 0
        for building in (1, 2, 4, 5, 7, 8):
            for weather_dir in sorted((RESULTS / f"building_train_{building}").iterdir()):
                csv = weather_dir / "g4bo_params" / "reward_components.csv"
                if csv.exists():
                    total += abs(pd.read_csv(csv).agent_reward.sum()) / STEPS
                    n += 1
        base_total = 0.0
        for building in (1, 2, 4, 5, 7, 8):
            for weather_dir in sorted((RESULTS / f"building_train_{building}").iterdir()):
                csv = weather_dir / "dsat_gen31" / "reward_components.csv"
                if csv.exists():
                    base_total += abs(pd.read_csv(csv).agent_reward.sum()) / STEPS
        champ_mean, base_mean = total / n, base_total / n
        print(f"FULL champ={champ_mean:.6f} vs base={base_mean:.6f} "
              f"({(champ_mean / base_mean - 1) * 100:+.2f}%) "
              f"gate(-5%)={base_mean * 0.95:.6f} -> "
              f"{'PASS' if champ_mean <= base_mean * 0.95 else 'FAIL'}")
    else:
        print("champion not better than the manual stack; no confirm run")


if __name__ == "__main__":
    main()
