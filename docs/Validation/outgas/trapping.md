# Validation: `src/proteus/outgas/trapping.py`

## What this page checks

This page checks the trapping step, `proteus.outgas.trapping`. As the magma ocean freezes, the growing solid buries volatiles: some enter the crystals, and some stay in the melt that is caught between the crystals. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the solid holds the wrong amount of each volatile, and the atmosphere gets the rest. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_trapping.py::test_incompatible_species_traps_through_the_interstitial_melt_alone` | Sim, Hirschmann & Hier-Majumder (2024), JGR Planets 129, e2024JE008346, Eq. 6 | Pins the buried mass `[(1 - F_tl) D_Z + F_tl] C_Z dM_RM` for water (`D_Z = 0.0017`) and carbon dioxide (`D_Z = 0`) at `F_tl = 0.02`, `dM_RM = 2e23` kg, against hand-computed values (4.3332e18 and 8e18 kg) at `rel=1e-12`, with a guard against dropping the `(1 - F_tl)` weight. Also pins the bracket's analytic limits, `D_Z` at `F_tl = 0` and 1 at `F_tl = 1`, and the supply cap. |

## Trapping through the interstitial melt

**Why this check exists.** A species that does not enter the crystals ($D_Z = 0$) is still trapped by the melt between them. A code that treats $D_Z = 0$ as no trapping, or that drops the weight of the crystal term, buries the wrong mass without an error message.

**What it tests.** Eq. 6 of Sim, Hirschmann and Hier-Majumder (2024) gives the buried mass $[(1 - F_\mathrm{tl}) D_Z + F_\mathrm{tl}]\, C_Z\, \Delta M_\mathrm{RM}$. Here $D_Z$ is the partition coefficient of the species between crystal and melt, $F_\mathrm{tl}$ the fraction of trapped melt, $C_Z$ the concentration in the melt and $\Delta M_\mathrm{RM}$ the mass that freezes in the step. At $F_\mathrm{tl}$ = 0.02 and $\Delta M_\mathrm{RM}$ = 2e23 kg the test expects 8e18 kg of CO$_2$ ($D_Z = 0$, $C_Z$ = 2e-3) and 4.3332e18 kg of water ($D_Z$ = 0.0017, $C_Z$ = 1e-3). It also checks:

- that dropping the $(1 - F_\mathrm{tl})$ weight would give 4.34e18 kg of water;
- the limits of the bracket: $D_Z$ at $F_\mathrm{tl} = 0$ and 1 at $F_\mathrm{tl} = 1$;
- that a remelting step buries nothing;
- the supply cap: no step buries more of a species than the melt holds;
- that a negative $D_Z$ or an unphysical trapped fraction raises `ValueError`.

**How we know the right answer.** The values follow by hand from Eq. 6 of Sim et al. (2024).

**Why it matters.** The main loop calls `run_trapping`, which buries each species with this mass balance (`trapped_mass`).

![Buried mass in one step against the trapped melt fraction for CO2 and H2O](../../assets/validation/outgas_trapping_interstitial_melt.svg#only-light){ width="80%" }
![Buried mass in one step against the trapped melt fraction for CO2 and H2O](../../assets/validation/outgas_trapping_interstitial_melt_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the trapped melt fraction $F_\mathrm{tl}$ and the y-axis the mass buried in the step, for CO$_2$ ($D_Z$ = 0, $C_Z$ = 2e-3) and H$_2$O ($D_Z$ = 0.0017, $C_Z$ = 1e-3) with 2e23 kg of freezing melt. The lines are `trapped_mass` and the squares Eq. 6 (context). The filled circles are the 2 asserted values at $F_\mathrm{tl}$ = 0.02, 8e18 kg and 4.3332e18 kg. A pass means that the circles lie on their lines. The rejected form without the $(1 - F_\mathrm{tl})$ weight differs by about 1e15 kg here, too little to see at this scale; the test checks it.

??? info "Test details"
    - Reference: Sim, Hirschmann & Hier-Majumder (2024), JGR Planets 129, e2024JE008346, Eq. 6.
    - Tolerance: relative 1e-12 on 8e18 kg and 4.3332e18 kg; more than 1e15 kg from the value without the $(1 - F_\mathrm{tl})$ weight.
    - Test id: `tests/outgas/test_trapping.py::test_incompatible_species_traps_through_the_interstitial_melt_alone`
    - Command: `pytest -m reference_pinned tests/outgas/test_trapping.py --validation-plots`

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
- no burial of a species whose element the run does not track, and a
  per-element closure refusing any break above 1 kg;
- a remelt after desiccation keeping what it releases however many steps it
  takes: a desiccated planet with 3.6e19 kg of water trapped, remelting from
  `Phi = 0.30` to 0.50 in 1000 steps that each release less than
  `mass_thresh`, ends with every total and the trapped 5/7 that 4 steps
  reach. Through the main loop in 16 such steps the H and O totals end the
  same way, and the released N (2/7 of 1.5e18 kg, below `mass_thresh`) is
  emptied by the escape floor once the remelt stops, down to the locked 5/7;
- the escape floor reading the reachable mass, as the desiccation gate does:
  two runs that differ only in how much N is trapped, 5e14 or 1.1e16 kg, both
  empty a reachable 9e15 kg below `mass_thresh` to the locked mass, and on a
  step on which a remelt returned buried mass the floor is off;
- the drained fraction set by the front speed alone, the same for a step five
  times longer that moves the front beyond its own thickness;
- the trapped share of the solid reservoirs surviving a chemistry solve that
  rewrites them, adding to condensed graphite rather than competing with it;
- species totals rebuilt from each fresh partition rather than frozen;
- desiccation emptying the atmosphere into `M_desiccated` and keeping the
  solid, and with trapping on the melt as well, so the closure holds and a
  remelt's release stays in the planet.
- a mantle frozen with `freeze_volatiles` escaping a quarter of its atmosphere
  a step for 20 steps through the main loop, with the species masses at
  CALLIOPE's 1.0000177 times the element masses, every element closing to
  `1e-10` on every row: the frozen step removes from each element's
  atmosphere what escape took from its total.

## Scope

The trapped melt fraction `F_tl` comes from the front scheme in
`proteus.outgas.compaction`, validated on its own page. The pinned test above
covers the mass balance the fraction enters.
