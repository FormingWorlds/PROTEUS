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
  including a remelting step;
- remelting returning the most recently buried mass first, from the burial
  ledger, so that freezing and remelting the same interval cancel: a closed
  melt cycled a hundred times between `Phi = 0.41` and 0.40 keeps its
  trapped water (see `burial_ledger.md`);
- a mantle frozen from `Phi = 1` to 0.5, degassed a hundredfold and remelted
  to `Phi = 1` getting back every buried kilogram, helium included, and the
  history-free fallback releasing `(Phi(t) - Phi(t-1)) / (1 - Phi(t-1))` of
  each trapped reservoir;
- dissolved noble gases buried with the interstitial melt at `F_tl C_Z dM_RM`,
  with their moles following the masses after the chemistry solve;
- no burial of a species whose element the run does not carry, and a
  per-element closure refusing any break above 1 kg;
- a remelt after desiccation keeping what it releases however many steps it
  takes: a desiccated planet with 3.6e19 kg of water trapped, remelting from
  `Phi = 0.30` to 0.50 in 1000 steps that each release less than
  `mass_thresh`, ends with every total and the trapped 5/7 that 4 steps
  reach, and through the main loop in 8 such steps likewise;
- the drained fraction set by the front speed alone, the same for a step five
  times longer that moves the front beyond its own thickness;
- the trapped share of the solid reservoirs surviving a chemistry solve that
  rewrites them, adding to condensed graphite rather than competing with it;
- species totals rebuilt from each fresh partition rather than frozen;
- desiccation changing no total, with trapping on or off: with trapping on
  the trapped mass stays in the solid and the rest of each total in the melt,
  so the closure holds.

## Scope

The trapped melt fraction `F_tl` comes from the front scheme in
`proteus.outgas.compaction`, validated on its own page. The pinned test above
covers the mass balance the fraction enters.
