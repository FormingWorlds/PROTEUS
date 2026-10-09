# Validation: `src/proteus/outgas/burial_ledger.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.burial_ledger`, which records where in melt fraction trapping
buried each volatile, so that a remelt returns the most recently buried mass.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_burial_ledger.py::test_the_ledger_matches_a_fine_grid_over_a_random_history` | Cross-implementation check against an independent ledger on a grid of 2e5 melt-fraction bins, which deposits each freeze uniformly over its bins and empties every bin below the melt fraction after each remelt | A seeded history of 400 steps that mostly freeze and sometimes remelt. The mass held below four melt fractions agrees to `rel=1e-3`, the discretisation error of one grid bin, and the total held to `1e-9`. | 2026-10-01 |

## Invariants checked alongside

- A remelt returns the most recently buried mass first: after freezing from
  `Phi = 1` to 0.7 and on to 0.5, remelting to 0.6 returns half of the second
  layer, not a fifth of everything.
- Returned plus still held equals buried, after every remelt.
- Freezing and remelting the same interval cancel. In `tests/outgas/test_trapping.py`
  a closed melt cycled a hundred times between `Phi = 0.41` and 0.40 keeps its
  trapped water to `1e-12`, where releasing the mean of the solid raises it by
  4.4% a cycle.
