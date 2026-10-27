# 2026-10-28

## Summary

Goal resumed with the research-plan objective (beat 0.00521 top-down).
Started item 1 (dual-max damper/reheat decoupling): bit-identical refactor
of the PI output map into an overridable hook, dual-max mixin + occ_dualmax7
+ 0.85-gain control arm, off-exactness proven, +3.1%/+2.7% on the first
600-step smoke. Hard-cell full-8640 screening (6 fairbanks + 2 mild,
3 arms) launched; results pending. (Overnight 10-27 work — oracle/CEM,
P-a/P-b/P-c probes, transfer-12, deep research, occ6/occ7 scores — is
logged in the 10-27 appendix update below.)

## Status

| Whose | Score | State |
|---|---|---|
| leader (Joe Penn) | 0.00521 | LB |
| us (occ_gated4) | 0.01323 | COMPLETE (best, tied) |
| us (occ_gated7) | 0.01323 | COMPLETE (tied 10-27) |
| us (occ_gated6) | 0.01337 | COMPLETE (+1.1% regression) |

## Done

- `zone_pi.py`: extracted `ZonePIPolicy._zone_output_map(u)` hook from
  `_command_zones` (pure method extraction, same op order).
- `occ_dualmax.py` (new): `DualMaxOutputMixin` (G36 valve-first-then-airflow
  staging, knobs `$DM_ON=1/$DM_SPLIT=0.5/$DM_HEAT_MIN=0.1`),
  `OccDualMax7Policy` (KEY `occ_dualmax7`), `OccHDamp085G7Policy` (KEY
  `occ_hdamp085_7`, control arm). Registered in `POLICY_MAP`.
- Bit-identity chain: refactor hash e9888825756a4637 == pre-refactor;
  `dualmax DM_ON=0` == occ7 exactly.
- Launched `/tmp/dm1`: 8 pairs × 3 policies full-8640 (fairbanks_1/2/4/5/7/8
  + atlanta_2 + denver_2). Commit `769dc61` pushed before launch.

## Measured

- occ_dualmax7 — b1/fairbanks_1 first-600 — 0.01648 vs 0.01701 (+3.1%).
- occ_hdamp085_7 — same window — 0.01655 (+2.7%).
- Full-episode screening: PENDING.

## Decisions

- None yet (awaiting full-8640 screening).

## Open

- dm1 results → 30-cell + held-out/no-tails validation if promising.
- Research item 2 (band-edge riding) next after item 1 resolves.

## Appendix

### 10-27 overnight session (22:00–24:00, logged here, file 10-27 frozen)

- Oracle/CEM: `tools/ckpt.py` + `tools/oracle_search.py` (perstep + const
  + CEM modes); numba global-RNG trap found and fixed via per-branch reseed
  (common random numbers); CEM on xfer_3 January: Tue +11.9%, Wed +9.7%,
  Thu +16.8% over occ7; 2H check confirms real (Tue: pure energy win,
  regret 0 both; Thu: 9x comfort-regret cut for +3% energy).
- P-a/P-b/P-c probes over occ7 (fast-trim/prop-kick/high-ceil): off-exact
  proven (caught an NEP-50 float32 promotion bug in P-b); transfer-8:
  pa +0.3%, pb +0.02% (flat), pc +7.2% (dead, x3-Jan +16%). xfer_2 16/16
  timeouts at 1500s (slowest building, not re-run).
- Kaggle: occ6 COMPLETE 0.01337 (+1.1% vs occ4 — prestart misfire);
  occ7 COMPLETE 0.01323 (exact tie — trim ≡ 0 on hidden).
- Deep research workflow: `docs/research/` (6 files, 3082 lines) + ranked
  top-10 plan; headline: our loss is 96% energy+carbon / 4% comfort.
- Speed analysis: reward computed 2x/step (~10% win available), zone-temp
  stats 234 calls/step (~5%); SuperLU 35% structural. Tier 0 not started.
