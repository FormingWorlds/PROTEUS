# Validation: `src/proteus/outgas/burial_ledger.py`

## What this page checks

This page checks the burial ledger, `proteus.outgas.burial_ledger`. When the magma ocean freezes, trapping buries volatiles in the solid; the ledger records at which melt fraction $\Phi$ each amount was buried, so that a remelt returns the most recently buried mass first. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If the ledger is wrong, a remelt returns the wrong amount of each volatile to the melt. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_burial_ledger.py::test_the_ledger_matches_a_fine_grid_over_a_random_history` | Cross-implementation check against an independent ledger on a grid of 2e5 melt-fraction bins, which deposits each freeze uniformly over its bins and empties every bin below the melt fraction after each remelt | A seeded history of 400 steps that mostly freeze and sometimes remelt. The mass held below four melt fractions agrees to `rel=1e-3`, the discretisation error of one grid bin, and the total held to `1e-9`. | 2026-10-01 |

## The ledger against a fine grid

**Why this check exists.** The ledger stores buried mass in layers that freezing adds and remelting removes. A wrong layer split or a wrong order of removal returns the wrong mass on a remelt without an error message.

**What it tests.** A seeded history of 400 steps that mostly freeze and sometimes remelt. After the history, the mass that the ledger holds below 4 melt fractions must agree with a second, independent ledger. The total mass held must also agree, and it must be less than 90 % of the mass buried, because the history remelts.

**How we know the right answer.** The second ledger uses a grid of 2e5 melt-fraction bins. It deposits each freeze uniformly over its bins and empties every bin below the melt fraction after each remelt. It uses no function of `burial_ledger`, and its discretisation error is about one grid bin.

**Why it matters.** `proteus.outgas.trapping` uses the ledger (`build_ledger`, `ledger_total`, `remelt_to`) for the volatiles held in the solid.

??? info "Test details"
    - Reference: cross-implementation check against an independent ledger on a grid of 2e5 melt-fraction bins.
    - Tolerance: relative 1e-3 on the mass held below each of 4 melt fractions (the discretisation error of one grid bin); relative 1e-9 on the total held.
    - Test id: `tests/outgas/test_burial_ledger.py::test_the_ledger_matches_a_fine_grid_over_a_random_history`
    - Command: `pytest tests/outgas/test_burial_ledger.py::test_the_ledger_matches_a_fine_grid_over_a_random_history`

## Invariants checked alongside

- A remelt returns the most recently buried mass first: after freezing from
  `Phi = 1` to 0.7 and on to 0.5, remelting to 0.6 returns half of the second
  layer, not a fifth of everything.
- Returned plus still held equals buried, after every remelt.
- Freezing and remelting the same interval cancel. In `tests/outgas/test_trapping.py`
  a closed melt cycled a hundred times between `Phi = 0.41` and 0.40 keeps its
  trapped water to `1e-12`, where releasing the mean of the solid raises it by
  4.4% a cycle.
