# Documentation system

Three artifacts:

- Journal: `docs/YYYY/MM/DD/README.md`, append-only. One file per active
  day. This is the lab notebook: what was tried, what was measured, what
  was decided — including failures.
- Knowledge: `docs/ladder.md`. The improvement ladder (worst -> best)
  plus the reproduction cookbook. Distilled from the journal; updated
  whenever a rung moves.
- Template: `docs/TEMPLATE.md`. Starting point for a new day.

Language: English only (R9). Chat with the user may be in Russian, but
everything committed under `docs/` and `AGENTS.md` operational sections
is English so that any machine, any agent, and the public repo readers
share one language.

## Mandatory day sections (in this order)

1. `## Summary` — 5-15 lines. What happened today and what it changed.
   A reader who stops here must know the day's outcome.
2. `## Status` — score table: our submissions (id, state, score) plus
   leader/median when known. Rule: on an active day the numbers are
   always recorded, even if unchanged ("no new scores").
3. `## Done` — every item: what + file + why (one line each). Time
   subsections (`### 08:00-10:40`) are encouraged on heavy days.
4. `## Measured` — numbers only, one measurement per line:
   `policy — coverage — loss — delta vs reference`.
5. `## Decisions` — conclusions as `claim + proof + consequence`
   (e.g. "line closed", "shipped as occ_gatedN").
6. `## Open` — questions and next steps. When writing a new day,
   reconcile with the previous day's Open: each item is done/moved/closed.
7. `## Appendix` (optional, unlimited) — exhaustive detail: constant
   tables, per-scenario numbers, registries, command logs. This is where
   clean-room reproducibility lives: after reading the Appendix, a stranger
   must be able to re-run the day without asking questions.

## Rules (checked at write time)

- R1. No bare adjectives: "better/worse" only with a delta.
- R2. Every measurement names its coverage: `train-30`, `subset-8`,
  `transfer-12`, `public`. Coverages are never mixed without a label.
- R3. Every file/command exists (repo-root-relative paths).
- R4. Negative results are mandatory: a closed lever with numbers.
- R5. A new policy/version gets a `ladder.md` entry before submission
  (mechanism + parameters + validation).
- R6. Submissions are logged on the day they are sent (kernel id,
  version, message).
- R7. Timestamps on world-facing events (pushes, submits, scores).
- R8. Irreproducible is not recorded as fact: no artifact — mark
  "from memory" or skip.
- R9. English only. No Russian prose in committed docs.
