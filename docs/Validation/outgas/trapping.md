# Validation: `src/proteus/outgas/trapping.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.trapping`, the solid-phase volatile trapping step, against
the published mass balance and its analytic limits.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_trapping.py::test_incompatible_species_traps_through_the_interstitial_melt_alone` | Sim, Hirschmann & Hier-Majumder (2024), JGR Planets 129, e2024JE008346, Eq. 6 | Pins the buried mass `[(1 - F_tl) D_Z + F_tl] C_Z dM_RM` for water (`D_Z = 0.0017`) and carbon dioxide (`D_Z = 0`) at `F_tl = 0.02`, `dM_RM = 2e23` kg, against hand-computed values (4.3332e18 and 8e18 kg) at `rel=1e-12`, with a guard against dropping the `(1 - F_tl)` weight. Also pins the bracket's analytic limits, `D_Z` at `F_tl = 0` and 1 at `F_tl = 1`, and the supply cap. |

## Invariants checked alongside

The trapping tests also assert, without a published reference:

- per-element closure `total = atm + liquid + solid` after every step,
  including a remelting step, and a freeze followed by the matching remelt
  returning every reservoir to its start;
- the trapped share of the solid reservoirs surviving a chemistry solve that
  rewrites them, adding to condensed graphite rather than competing with it;
- species totals rebuilt from each fresh partition rather than frozen;
- desiccation keeping the trapped mass while the closure still holds.

## Scope

The trapped melt fraction `F_tl` comes from the front scheme in
`proteus.outgas.compaction`, validated on its own page. The pinned test above
covers the mass balance the fraction enters.
