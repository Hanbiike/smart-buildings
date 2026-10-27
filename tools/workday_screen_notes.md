# Fresh-reset workday baseline screen

Outputs: `smart_buildings/results/screens/screen_umim7m67/`.
Machine-readable comparison/source hashes: `run_manifest.json` in that directory.

```sh
./.venv/bin/python -B tools/parallel_eval.py --policies g4_m25w08kp05,dsat_gen31 --buildings building_train_1 --weathers fairbanks_1 --steps 288 --concurrency 2 --timeout 600 --start-timestamp 2023-01-03T14:00:00+00:00 --dump-obs
```

Both runs: exactly 288 physical steps, no preceding warm-up; timestamp is bound
before environment construction. Requested UTC time resolves to the simulator's
naive-UTC convention (06:00 US/Pacific in this January schedule). Recorded result
timestamps run from January 3 14:05 through January 4 14:00 UTC. Fresh policy
and thermal state per run. Initial zone observations are uniformly 294 K; the
first latched-temperature duplicate is identical. Reward occupancy trajectories
match exactly, with 97 positive-occupancy intervals and average 17.80556 people.

Exact loss = -mean reward; values read from JSON, not rounded stdout:

| Policy | Loss | Comfort contribution | Energy contribution | Carbon contribution |
| --- | ---: | ---: | ---: | ---: |
| g4_m25w08kp05 | 0.0445707424 | 0.0040930441 | 0.0116708657 | 0.0288068324 |
| dsat_gen31 | 0.0449967450 | 0.0041284561 | 0.0117811417 | 0.0290871472 |

Champion improves this single paired case by **0.94674%**. This is an occupied
cold-reset diagnostic, not an archive replay, full-episode score, new controller
result or evidence of beating the leaderboard. It is not comparable to earlier
Sunday 288-step averages.

## Flow-preserving pressure opportunity audit

Replayed the achieved equipment observations and commands only, without a new
simulation. Per policy: 480 active AHU observations, actual pressure 200..1000 Pa.
Tested P' >= max(causally known floor / lowest prior running pressure,
P * max_zone_damper^2), with all dampers rescaled by sqrt(P/P'). This preserves
the unrestricted airflow products before common handler capacity capping.
The floor bound is causal; no true configured floor or future observation is
used. No lower feasible pressure was found at any of these operating points.

This closes this specific pressure-equivalence opportunity on this one case,
not all equipment optimization or all climates. It does not validate a new
online action mapping: achieved actuator positions are not automatically known
for a newly requested command when clipping is unidentified. No pressure sweep
or controller modification is justified by this audit.

Next: causal adequacy monitor and broader matched workday baseline coverage;
do not tune champion PI constants. Simulator physics and champion unchanged.
