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

**Known limit.** The case tests the mass balance for given $F_\mathrm{tl}$ and $C_Z$. $F_\mathrm{tl}$ comes from the front scheme in `proteus.outgas.compaction`, which [its own page](compaction.md) validates; this case does not test it.

**Why it matters.** The main loop calls `run_trapping`, which buries each species with this mass balance (`trapped_mass`).

![Buried mass in one step against the trapped melt fraction for CO2 and H2O](../../assets/validation/outgas_trapping_interstitial_melt.svg#only-light){ width="80%" }
![Buried mass in one step against the trapped melt fraction for CO2 and H2O](../../assets/validation/outgas_trapping_interstitial_melt_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the trapped melt fraction $F_\mathrm{tl}$ and the y-axis the mass buried in the step, from 2e23 kg of freezing melt. The 2 species are CO$_2$ ($D_Z$ = 0, $C_Z$ = 2e-3) and H$_2$O ($D_Z$ = 0.0017, $C_Z$ = 1e-3). Here $D_Z$ is the partition coefficient and $C_Z$ the concentration in the melt. The lines are `trapped_mass` and the squares Eq. 6 (context); the filled circles are the 2 asserted values at $F_\mathrm{tl}$ = 0.02. A pass means that the circles lie on their lines. For H$_2$O, the form without the $(1 - F_\mathrm{tl})$ weight differs by about 7e15 kg (the test requires more than 1e15 kg), too little to see.

??? info "Test details"
    - Reference: Sim, Hirschmann & Hier-Majumder (2024), JGR Planets 129, e2024JE008346, Eq. 6.
    - Tolerance: relative 1e-12 on 8e18 kg and 4.3332e18 kg; more than 1e15 kg from the value without the $(1 - F_\mathrm{tl})$ weight.
    - Test id: `tests/outgas/test_trapping.py::test_incompatible_species_traps_through_the_interstitial_melt_alone`
    - Command: `pytest -m reference_pinned tests/outgas/test_trapping.py --validation-plots`

## Invariants checked alongside

The trapping tests in `tests/outgas/test_trapping.py` also assert, without a published reference:

- the mass of each element equals the sum of its atmosphere, melt and solid parts after every step, also after a re-melt;
- a re-melt returns the most recently buried mass first, so freezing and re-melting the same layer cancel (see the [burial ledger](burial_ledger.md));
- a mantle that freezes and then melts again gets back every buried kilogram;
- dissolved noble gases are buried only with the trapped melt;
- nothing is buried for an element that the run does not track;
- the drained fraction depends on the speed of the front, not on the length of the step;
- desiccation and escape keep each element mass closed while volatiles are trapped.
