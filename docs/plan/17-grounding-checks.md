# Grounding checks for Online RL work

Date: 2026-11-01
Status: draft
Scope: conformance checks that Online RL work on top of `occ_gated7` must
pass without breaking repository invariants: full-episode validity,
concurrency and policy-freeze discipline, registry hygiene, notebook
assembly, transfer evaluation, the deployable interface boundary, and the
journal/evidence rules R1-R9. Companion to
`docs/research/17-repo-grounding-online-rl.md` (statements cited as G-numbers
below). No simulator episode, training job, or benchmark was run to produce
this checklist.

## Check table

| ID | Check | How (exact command/file in this repo) | Expected / threshold | Priority |
|----|-------|----------------------------------------|----------------------|----------|
| E-01 | Every reported loss is a full 8640-step episode | Run `./.venv/bin/python -B tools/fullrun.py /tmp/r 8640 8 1500 <policy> <pairs>`; loss from `reward_components.csv` as `abs(df.agent_reward.sum()) / len(df)` (G3) | `n == 8640` in every CSV; short runs never quoted as scores | P0 |
| E-02 | Score uses the single loss formula | Compute `abs(df.agent_reward.sum()) / len(df)` exactly as `tools/fullrun.py` does; same policy+scenario rerun is bit-identical (G3, G5) | Formula verbatim; rerun diff is empty | P0 |
| E-03 | Coverage label on every number | Label each measurement `train-30` / `subset-8` / `transfer-12` / `public` per `docs/ladder.md` header; never mix unlabeled (G4, R2) | No bare number without a coverage tag | P0 |
| E-04 | Python interpreter discipline | All runs via `./.venv/bin/python -B ...` (G6) | No system-python runs in logs | P1 |
| C-01 | Concurrency cap respected | Pass `<conc>` so total concurrent sims <= 10; use timeout >= 1500 s, e.g. `./.venv/bin/python -B tools/fullrun.py /tmp/r 8640 8 1500 ...` (G37) | Zero `timeout after Ns` rows caused by overload | P0 |
| C-02 | Policy code frozen during validation | Do not edit anything under `smart_buildings/baselines/` or `smart_buildings/evaluate.py` while a `fullrun.py`/`fullrun_t.py` suite is in flight (G37) | `git status` shows no policy edits timestamped inside a run window | P0 |
| C-03 | Zone-count precheck before any suite | Inspect the building zone count (<= 10 zones/AHU required; monster buildings simulate catastrophically slowly, G25) before launching | Suite contains no >10-zones/AHU building; no multi-hour single job | P0 |
| R-01 | New policy registered in POLICY_MAP | Add key + class in `smart_buildings/evaluate.py` `POLICY_MAP` (dict starting at line 218); key naming follows `occ*`/`probe_*` convention (G7-G8) | `grep -n '"<key>"' smart_buildings/evaluate.py` hits exactly the new entry | P0 |
| R-02 | Probes stay probes until promoted | Throwaway variants live in `probe_modes.py` + `POLICY_MAP`; promotion creates `smart_buildings/baselines/occ_gatedN.py` with fixed constants, no env vars (G9) | Shipped file contains no `os.environ`/`getenv` reads | P0 |
| R-03 | Rudder constants untouched unless rung justifies | `occ_gated7.py` PTR block (`PTR_UP=0.25`, `PTR_DOWN=0.03`, `PTR_MAX=0.0`, `PTR_DSAT=0.85`, `PTR_DB=1.0`, `PTR_DB_SLOPE=0.1`, `PTR_DB_REF=10.0`, `PTR_PERSIST=4`, G12) and inherited stack (G16) change only with a ladder rung entry | `git diff smart_buildings/baselines/occ_gated7.py` empty, or rung documents the change | P0 |
| R-04 | Stateful-teacher handling declared | Document how the RL layer handles PI integrators, latched prestart, hysteresis, 24 h EMAs (G17); memoryless cloning already failed (BC 0.0368 vs teacher 0.0175) | Design note names state handling; no memoryless-clone retry without new evidence | P1 |
| R-05 | No re-test of closed levers | Cross-check proposals against `docs/ladder.md` failed rungs: -2 K heat offset, duty cycling, ~15 `probe_*` axes, slot/BC, continuous PPO (lr 3e-5/KL 0.01), load shifting (G28-G29, G32-G33) | Proposal cites the closing rung or shows why conditions changed | P1 |
| N-01 | Notebook assembled only via assembler | Build with `./.venv/bin/python -B tools/assemble_notebook.py <src> <dst> <SubmitClass> <classes...> <file:CLASS...>` (G39); never hand-edit class blocks | Assembler verification passes (verbatim modulo trailing whitespace) | P0 |
| N-02 | `occ_gated:<consts>` included | Assembly args contain `occ_gated:<consts>` so `STEPS_PER_HOUR` ships (G40); occ6 v1 ERRORed without it | `grep -c STEPS_PER_HOUR <dst_ipynb>` >= 1 | P0 |
| N-03 | Exec check 300+ steps incl. night/shutdown/prestart | Execute the assembled notebook locally for >= 300 steps covering night + unoccupied + prestart (G41); 12 steps once missed a night-only NameError | Log shows >= 300 executed steps across all three regimes, zero exceptions | P0 |
| N-04 | Submission flow and budget | `kaggle kernels push` -> wait COMPLETE -> `competitions submit -k ... -v N -f submission.parquet`; scoring ~14 h; watch daily budget (G42) | Kernel COMPLETE before submit; budget checked before push | P1 |
| N-05 | Inference self-containment | Assembled cell carries everything: fixed constants only, no repo imports, no checkpoint files, no missing torch-class deps (G43) | Notebook runs on a bare kernel with no repo checkout | P0 |
| T-01 | Transfer scenarios from fixed seeds | Generate with `./.venv/bin/python -B tools/gen_transfer.py` (seeds 50000+i, G34); evaluate with `tools/fullrun_t.py /tmp/x 8640 8 1500 <policies> <pairs> smart_buildings/transfer_scenarios` (G37-G38) | Regeneration reproduces the suite; suite = 3 buildings x 4 weathers unless documented otherwise | P0 |
| T-02 | Transfer limits stated with every claim | Quote the limits: 3 building samples from one generator, 1 unseen station (minneapolis), git-ignored scenarios (G35) | No "generalizes to hidden" claim from transfer-12 alone | P1 |
| T-03 | Tail-first reporting | Report worst episodes and time-resolved comfort tails (xfer_3 January comfort -0.062 precedent, G36) alongside means | Every transfer report shows per-scenario rows, not just the mean | P1 |
| T-04 | Misfire regression gate for rudder-adjacent changes | Re-run the 6 tail/misfire pairs; occ_gated7 bar is tails -23%/-34% with misfire pairs bit-equal to occ4 (G15) | Zero regressions on misfire pairs; any regression blocks promotion | P0 |
| B-01 | Deployable interface only | Policy entry is `Policy(building_meta_data, action_spec)` + `get_action(obs)`; fresh instance per scenario, no cross-scenario memory; inputs are metadata + obs only (G26) | Code review: no env object, scenario path, materials, or future-weather reads | P0 |
| B-02 | No oracle inputs | No exact future weather/occupancy, no hidden thermal state, no `tools/oracle_search.py` / `tools/ckpt.py` logic in the policy; oracle vs deployable results reported separately (G26, G45) | `grep -rn oracle <new_policy_file>` empty; forecast audit passes | P0 |
| B-03 | Occupancy proxy declared | Comfort optimization declares its building-level occupancy proxy (true zone occupancy unobserved, G24, G30); no per-zone gating assumed | Design note names the proxy and its failure mode | P1 |
| B-04 | Action bounds enforced in-policy | Continuous actions in [-1, 1] with 0 = midpoint (G21); `validate_action` does NOT enforce bounds, so the policy must (G22) | Fuzz `get_action` over recorded obs: 100% in-range and finite | P0 |
| B-05 | Live metadata indexing | Feature/action indexing reads live `observation_field_names` and `action_spec` (dims 271 vs 283 precedent, G27); no hardcoded obs size | Policy works on a held-out building with different obs width | P0 |
| B-06 | Physics-cap consistency | No AHU heating, no discharge > 308.15 K assumption, recirculation <= 0.70, dampers >= 10%, constant plant efficiencies (G19-G20) | Command audit: no plan relies on a capped mechanism | P1 |
| B-07 | Comfort-first guardrail | No comfort-for-energy trade: comfort slope ~0.18/K >> energy ~0.002/K; -2 K offset precedent lost on both train and public (G28); TOU arbitrage dead at 1.25x (G29) | Any energy-saving change shows comfort delta <= 0 before merge | P0 |
| B-08 | Horizon covers prestart/recovery | Effective horizon reaches prestart + night + next-day recovery (24-48 steps for 2-4 h); short-horizon approximations scored against the 5-minute system (G44); air/water coupling checked jointly, not per-zone alone (G46) | Horizon analysis in design note; joint-coupling check passes | P1 |
| D-01 | Ladder entry before submission | New policy/version recorded in `docs/ladder.md` (R5, G10) with mechanism, parameters, why, and how to run | Ladder diff merged before `kaggle kernels push` | P0 |
| D-02 | Daily journal kept | Active day recorded in `docs/YYYY/MM/DD/README.md` from `docs/TEMPLATE.md`: Summary + Status + Done + Measured + Decisions + Open + Appendix (G49) | File exists for each active day with all sections | P1 |
| D-03 | R1-R9 spot check at write time | R1 deltas on comparisons; R2 coverage labels; R3 paths/commands exist; R4 negatives recorded; R6 submissions logged day-sent; R7 timestamps on world events; R8 no irreproducible facts; R9 English only (G47-G49) | Checklist ticked in the journal Appendix | P1 |
| D-04 | English-only committed prose | Everything under `docs/*` and all code comments in English (G47) | No non-English prose in `git diff docs/` | P1 |
| D-05 | Evidence grading on new claims | Research claims graded Source-verified / Recomputed / Journal-reported / Hypothesis per report 06 standard (G48) | Each numbered claim carries a grade + provenance | P2 |
| D-06 | Push via project key; main is truth | Push only with `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_kaggle_push -o IdentitiesOnly=yes" git push origin main` (G42) | Plain-push rejection never bypassed by alternates | P2 |

## Definition of done

- Episode validity (E-01..E-04): every quoted score traces to a full 8640-step
  CSV via the verbatim loss formula, carries a coverage label, and was
  produced with the project interpreter. Done when a reviewer can recompute
  each number from a named CSV path.
- Concurrency and freeze (C-01..C-03): no suite exceeded 10 concurrent sims,
  no policy file changed mid-run, no monster building entered a suite
  unchecked. Done when run logs show zero overload timeouts and `git log`
  shows no in-flight policy edits.
- Registry hygiene (R-01..R-05): the new policy is a registered key, probes
  and shipped code are separated, rudder constants are intact or rung-
  justified, teacher-state handling is documented, and no closed lever was
  re-tested blind. Done when `POLICY_MAP` greps clean and the design note
  cites the closing rungs.
- Notebook and submission (N-01..N-05): the notebook was assembler-built with
  `<consts>`, exec-checked 300+ steps across night/shutdown/prestart,
  submitted through the COMPLETE-gated flow within budget, and runs
  standalone. Done when the kernel is COMPLETE and the exec log is attached.
- Transfer evaluation (T-01..T-04): scenarios came from fixed seeds, limits
  are stated, tails are reported per scenario, and misfire pairs show zero
  regressions. Done when the transfer report names all 12 per-scenario
  losses plus the 6 misfire comparisons.
- Deployable boundary (B-01..B-08): the policy uses the participant interface
  only, carries no oracle inputs, declares its occupancy proxy, enforces
  action bounds, indexes live metadata, respects physics caps, keeps the
  comfort-first guardrail, and covers prestart/recovery horizons. Done when
  the interface audit, fuzz test, and horizon analysis all pass.
- Documentation (D-01..D-06): ladder updated before submission, journal day
  filed, R1-R9 ticked, prose English-only, claims evidence-graded, and
  pushes use the project key. Done when `git diff docs/` is clean English
  and the Appendix carries the R1-R9 tick list.
