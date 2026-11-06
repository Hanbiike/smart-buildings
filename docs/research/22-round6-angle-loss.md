# Round-6 workflow: angle loss record + parent-verified findings

Source: round-6 workflow (8 children: angles 0-4, p99-followup,
critic, synthesis). Only 3/8 results were read (angle-0,
p99-followup, synthesis); angles 1-4 + critic schema-failed and were
LOST. This note records the gap and the three findings the parent
verified before writing `docs/plan/23-residual-revival.md`.

## The gap (unverified-until-screened)

The synthesis ranked 8 candidates, but the evidence for ranks
covering the T4-path (feature-sufficiency audit: whether econ/dbh
/heatopt are linearly expressible in slot features), tie-diagnosis,
ship-readiness, and climategate lives only in the lost angles +
critic. Plan-23 treated synthesis ranks as hypotheses, not evidence:
every rank-2..7 item carried a kill gate, and all screening results
stand on their own (screens, not ranks, decided).

Concretely lost (names from the synthesis, no dedicated evidence):

- T4-path / feature-audit (would-be angle-1 territory).
- Tie-diagnosis (why sg8/phx ties hold).
- Ship-readiness analysis (covered independently by P23-8).
- Climategate analysis (covered independently by P23-4 KILL).

## Parent-verified findings (kept)

1. Bank-1 weekend heads are EXACTLY 0 (W ~1e-15, run-bias 1.0):
   T == B on weekends, i.e. occ9 and occ7 agree exactly when the
   plant is off. The residual has no weekend signal to fit.
2. Train/infer memory skew is REAL: deploy `slot.step(update=True)`
   runs always while train uses `update=deleg`. P23-2 showed the
   mismatch (9.6% deleg) is closed-loop divergence, not an
   update-rule bug -- no deploy patch can match unknown
   teacher-deleg; the skew belongs to DAgger (see P23-6 KILL).
3. `/tmp/pooled_p99.py` per-channel constants (damp 0.872 / reheat
   0.846 / SAT 0.624 / press 0.046 / recirc 2.102 / waterT 1.160)
   fed the P23-1 per-channel scale arm (killed: no win over uniform).

## Lesson

Size fan-out to the read budget: 8 children with a 3-read budget
loses 5. Critical angles (feature-sufficiency, ship) must be read
first or run as parent-verified probes, not as unverified ranks.
