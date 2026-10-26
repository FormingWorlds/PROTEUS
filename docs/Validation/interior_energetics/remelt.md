# Validation: partial giant-impact re-melt

## What this page checks

This page checks how PROTEUS melts part of the mantle at a giant impact (`src/proteus/interior_energetics/remelt.py`).
An impact melts the mass fraction $f$ of the mantle.
The rule raises the outermost mantle cells (thin spherical shells) that hold this mass to the initial condition of the run.
Validation here means that we run the code on a case where another source gives the right answer.
The test then checks that the code gives that answer within a stated tolerance.
Here the right answer comes from geometry, since the melt front of a mantle of uniform density has a closed form, and from the masses of the mantle cells before the impact.
If this part of the code is wrong, an impact melts too much or too little of the mantle, or melts it at the wrong depth, and the evolution after the impact starts from the wrong state.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters.
The box "Test details" at the end of each case gives the reference, the tolerance, the test id and the command.

**Terms of this page.**

- Cell: the interior solver divides the mantle into cells, thin spherical shells; Aragog places them on a staggered mesh, with the cell centres between the mesh nodes.
- $r_c$, $R$ and $r_f$: the radius of the core-mantle boundary, of the surface and of the melt front.
- $S_\mathrm{end}$ and $S_\mathrm{ic}$: the specific entropy of a cell at the end of the step before the impact, and at the initial condition of the run.
- Weight $w$: the melted share of the mass of a cell, from 0 (not melted) to 1 (melted in full).

The table lists the `@pytest.mark.reference_pinned` test and the slow-tier test that anchor the re-melt in `proteus.interior_energetics.remelt`, with the reference and the date of the last comparison.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_remelt.py::test_the_melt_front_of_a_uniform_mantle_matches_the_closed_form` | Analytical limit: in a shell of uniform density between $r_c$ and $R$, the outer mass fraction $f$ lies above $r_f = (R^3 - f (R^3 - r_c^3))^{1/3}$ | `remelt_weights` on 400 cells uniform in radius between 3500 and 6400 km, at $f$ = 0.05 to 0.99. The front of the partly melted cell matches the closed form to `rtol=1e-12`. A fill linear in radius misses by 282 km at $f = 0.3$, and a fill from the base up misses by at least 1040 km at each tested fraction except $f = 0.5$, where the two fills meet. | 2026-10-24 |
| `tests/integration/test_slow_accretion_partial_remelt.py::test_the_melted_mass_closes_on_the_pre_impact_cell_masses` | Wiring check against the interior solver: the staggered mesh and its structural density in the Aragog solver PROTEUS builds (fwl-aragog 26.10.22), after a short cooled run | Slow tier. At $f = 0.3$ on a real solver, for the `energy_balance` and the `core_module` core, the weights recovered from the re-melted entropy close against $f$ times the structural mantle mass of the pre-impact mesh to `rel=1e-9`; against the cell volumes and against the equation of state (EOS) masses $\rho(P, S_\mathrm{end}) V$ they miss by more than 1e-3. | 2026-10-25 |

## Melt front against the closed form

**Why this check exists.** The rule must put the melt into the outermost shells by mass, not by radius. A fill that is linear in radius, or a fill from the base up, puts the melt at the wrong depth.

**What it tests.** `remelt_weights` on 400 cells uniform in radius between 3500 and 6400 km, at $f$ = 0.05 to 0.99, for a mantle of uniform density. In a shell of uniform density between $r_c$ and $R$, the outer mass fraction $f$ lies above $r_f = (R^3 - f (R^3 - r_c^3))^{1/3}$. The front of the partly melted cell must match this closed form.

**How we know the right answer.** The closed form follows from the mass of a spherical shell of density $\rho$: $\frac{4}{3}\pi\rho(R^3 - r_f^3) = f\,\frac{4}{3}\pi\rho(R^3 - r_c^3)$.

**Why it matters.** The melt front sets which part of the mantle starts hot after the impact. A fill linear in radius misses by 282 km at $f = 0.3$. A fill from the base up misses by at least 1040 km at each tested fraction except $f = 0.5$, where the two fills meet.

![Melt front radius against the melted mass fraction](../../assets/validation/interior_energetics_remelt_front.svg#only-light){ width="80%" }
![Melt front radius against the melted mass fraction](../../assets/validation/interior_energetics_remelt_front_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the melted mass fraction $f$ and the y-axis is the radius of the melt front in km, for a mantle of uniform density between 3500 and 6400 km. The line is the closed form, the dashed line is a fill linear in radius, and the circles are the front of the weights from `remelt_weights` at the 7 fractions of the test. The front sits above the linear fill because the outer shells hold more mass per unit radius. A pass means that the circles lie on the line.

??? info "Test details"
    - Reference: the closed form above, from the shell mass $\frac{4}{3}\pi\rho(R^3 - r_f^3) = f\,\frac{4}{3}\pi\rho(R^3 - r_c^3)$.
    - Tolerance: `rtol=1e-12`, the round-off of the cumulative cell masses.
    - Test id: `tests/interior_energetics/test_remelt.py::test_the_melt_front_of_a_uniform_mantle_matches_the_closed_form`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_remelt.py --validation-plots`

## Melted mass on a real Aragog solver

**Why this check exists.** The Aragog solver that PROTEUS builds has a staggered mesh and its own structural density. The melted mass must close on the cell masses of that mesh, not on the cell volumes or on densities from the equation of state (EOS).

**What it tests.** A short cooled run of the Aragog solver that PROTEUS builds (fwl-aragog 26.10.22), for the `energy_balance` and the `core_module` core. At $f = 0.3$, the test recovers the weights from the re-melted entropy. They must close against $f$ times the structural mantle mass of the mesh before the impact.

**How we know the right answer.** Mass balance gives it. The melted mass is $f$ times the mantle mass before the impact. The reference uses the same mesh and density as the weights, so the case checks the wiring and the mass closure, not the density itself. Against the cell volumes and against the EOS masses $\rho(P, S_\mathrm{end}) V$, the weights miss by more than 1e-3, so the test tells the right masses from the wrong ones.

**Why it matters.** This case checks the rule on the solver that a coupled run uses, with its real mesh and density.

??? info "Test details"
    - Reference: wiring check against the interior solver: the staggered mesh and its structural density in the Aragog solver PROTEUS builds (fwl-aragog 26.10.22), after a short cooled run.
    - Tolerance: `rel=1e-9` against $f$ times the structural mantle mass of the mesh before the impact.
    - Test id: `tests/integration/test_slow_accretion_partial_remelt.py::test_the_melted_mass_closes_on_the_pre_impact_cell_masses` (slow tier)
    - Command: `pytest -m slow tests/integration/test_slow_accretion_partial_remelt.py`

## Limits of the rule and the partly melted cell

**Why this check exists.** At $w = 1$ the blend $S_\mathrm{end} + w (S_\mathrm{ic} - S_\mathrm{end})$ can land 1 unit in the last place off $S_\mathrm{ic}$, so the full re-melt must take $\max(S_\mathrm{end}, S_\mathrm{ic})$ itself. Without the clamp $\max(S_\mathrm{ic} - S_\mathrm{end}, 0)$ in the blend, a cell that is hotter than the initial condition would cool. Both faults change the profile where the right answer is known exactly.

**What it tests.** `tests/interior_energetics/test_remelt.py` pins the ends of the rule. $f = 1$ gives `np.maximum(S_end, S_ic)` bit for bit, the full re-melt. $f = 0$ leaves every cell unchanged, and no cell cools. The scalar mantle of the dummy interior moves to $T_\mathrm{end} + f\,(T_\mathrm{init} - T_\mathrm{end})$, the enthalpy mean of a melted and an unmelted part of one heat capacity. Here $T_\mathrm{end}$ is its temperature before the impact and $T_\mathrm{init}$ the initial temperature.

**How we know the right answer.** These are the limits of the rule itself. At $f = 1$ the whole mantle melts, and at $f = 0$ none of it melts, so the expected profile is known cell by cell without a model. The blends are computed by hand: from $S_\mathrm{end}$ = 2400 to $S_\mathrm{ic}$ = 3000 J kg$^{-1}$ K$^{-1}$, a cell at $w = 1/3$ takes 2600 and one at $w = 0.7$ takes 2820; the scalar mantle at $f = 0.3$ moves from 2000 K towards 4000 K to 2600 K.

**Why it matters.** With $f = 1$ the rule must give exactly the full re-melt of `accretion.remelt = "full"`. A cell that cooled at an impact would remove heat that the impact never took out.

**Known limit.** The partly melted cell takes the mass-weighted mean entropy of its melted and unmelted parts. This mean holds less heat than the two parts. The deficit is first order in $dT/dS$, the rise of temperature with entropy, so it vanishes when the temperature does not change over the jump: $w(1-w)\,m\,\Delta S\,\Delta T/2$ for a temperature linear in entropy over the cell, with $w$ the melted share of the cell, $m$ its mass, $\Delta S = S_\mathrm{ic} - S_\mathrm{end}$ and $\Delta T$ the temperature rise over $\Delta S$. Relative to the injected heat it is largest when all the melt fits in the top cell. The booked `step_dE_impact_J` integrates the heat of the actual re-melted profile, so the energy budget of an Aragog run closes.

??? info "Test details"
    - Reference: the limits $f = 0$ and $f = 1$ of the rule, and the hand-computed blends above.
    - Tolerance: bit for bit at $f = 0$ and $f = 1$; relative 1e-15 for the blends and the scalar mantle.
    - Test ids: `tests/interior_energetics/test_remelt.py::test_weights_at_the_ends_are_exact`, `::test_full_weight_is_the_maximum_bit_for_bit`, `::test_partial_weight_blends_by_mass_share_and_never_cools`, `::test_scalar_mantle_moves_by_the_melted_share`
    - Command: `pytest tests/interior_energetics/test_remelt.py`
