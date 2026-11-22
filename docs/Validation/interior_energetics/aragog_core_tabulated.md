# Validation: the PALEOS core across a structure update

## What this page checks

This page checks what PROTEUS does to a core on the PALEOS profile at a structure update (`src/proteus/interior_energetics/aragog_core_tabulated.py`).
With `profile_source = "paleos"`, the core of the Aragog `core_module` boundary lies on an isentrope of the PALEOS liquid iron table: a line of constant specific entropy through the temperature at the core-mantle boundary (CMB).
PROTEUS refuses this profile when it reads the config, because the profile needs a 2-phase iron core in the Zalmoxis structure, which PROTEUS does not build. The tests below therefore run the code on 2 stored structures.
When the structure solver changes the core, PROTEUS gives the core the CMB temperature that keeps its mass and its total entropy, and books the change of the core heat content in `step_dE_struct_core_J`.
The second case checks the default Gaussian profile, which every `core_module` run uses.
If this part is wrong, the core starts too hot or too cold after each structure update, or the core energy ledger books a heat that depends on the table and not on the planet.

Validation here means that we run the code on a case where another source gives the right answer, and the test checks that the code gives that answer within a stated tolerance. Each case below says why the check exists, what it tests, how we know the right answer and why it matters; the box "Test details" gives the reference, the tolerance, the test id and the command.

## Terms and tests on this page


- Structure pair: the cores of 2 stored Zalmoxis planets of 1 Earth mass, with the same core mass, 1.9409e24 kg, and CMB pressures of 127.7 and 141.3 GPa. They come from 2 structure solves at surface temperatures of 3000 and 2510 K, so they stand for one core before and after a structure update. Zalmoxis solved them with an iron-sulfur outer core; the tests use their radius, mass, pressure, density and gravity on the pure-iron liquid table.
- $C_s(T)$: the secular heat capacity of the core in J K$^{-1}$, the heat per kelvin of CMB cooling at a fixed structure.
- Heat content: $T\,C_s(T) - R(T)$ in J, with $T$ the CMB temperature and $R$ the latent and gravitational energy that the inner core released below its onset (0 above it).
- Table floor: the lowest CMB temperature for which Aragog tabulates the isentropes of a structure. It follows from the lowest temperature of the table.
- Carried temperature: the CMB temperature on the second structure of the pair that keeps the mass and the total entropy of the core on the first structure at a given CMB temperature. "Carried from 5000 K" means that the core on the first structure is at 5000 K.
- Structure term: the change of the core heat content across the update, from the first structure at its CMB temperature to the second structure at the carried temperature. PROTEUS books it in `step_dE_struct_core_J`.
- The 3 TPa cut: the part of the PALEOS liquid iron table that the tests read (`tests/interior_energetics/fixtures/core_tabulated/paleos_iron_liquid_cut_3tpa.npz`): 74 pressures from 44.8 GPa to 3 TPa and 41 temperatures from 1996 to 20000 K.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_aragog_core_tabulated.py::test_the_structure_term_does_not_see_the_table_floor` | Analytical invariant: a heat content counted from 0 K does not change with the table floor | The structure pair at 5000 K on 2 cuts of one table. The carried temperature and the structure term agree to `rel=1e-12`; the difference of Aragog's own heat contents moves by more than 1e27 J. | 2026-11-17 |
| `tests/interior_energetics/test_aragog_core_tabulated.py::test_the_content_of_a_gaussian_budget_is_its_heat_content` | Identity: on the Gaussian profile $T\,C_s - R$ is Aragog's heat content | 4 CMB temperatures above and below the onset, `rel=1e-12` | 2026-11-17 |
| `tests/interior_energetics/test_aragog_core_tabulated.py::test_the_update_keeps_the_entropy_and_books_the_parcel_heat` | Thermodynamics: at constant entropy the heat content of a parcel changes by $c_p\,\Delta T$ | The structure pair at 7 CMB temperatures from 4600 to 6400 K. Entropy at the CMB kept to `rel=1e-10`, structure term against the parcel sum to `rtol=1e-5`. | 2026-11-17 |

## The structure term does not depend on the lowest table temperature

**Why this check exists.** Aragog counts the heat content of a tabulated core from the table floor, a temperature that differs between 2 structures and between 2 cuts of one table. A structure term built from that content would carry this constant: 4.4e27 J on the structure pair.

**What it tests.** The structure pair is carried from 5000 K on the 3 TPa cut of the table and on the same cut without its lowest temperature column. The test compares the carried temperature and the structure term on the 2 cuts.

**How we know the right answer.** The heat content $T\,C_s(T) - R(T)$ counts from 0 K. It reads the table only near the core temperatures, so it must not change when the lowest table column goes.

**Why it matters.** The structure term enters the core energy ledger at every structure update. A term that follows the table floor would book heat that the planet never had.

This case has no plot: it compares 2 values of one quantity, and the plot of the third case shows that quantity over the temperature range.

??? info "Test details"
    - Reference: analytical invariant of the heat content counted from 0 K.
    - Tolerance: `rel=1e-12` for the carried temperature and the term (measured 4.4e-16 and 9.6e-14); the difference of Aragog's own heat contents moves by more than 1e27 J.
    - Test id: `tests/interior_energetics/test_aragog_core_tabulated.py::test_the_structure_term_does_not_see_the_table_floor`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_aragog_core_tabulated.py`

## On the Gaussian profile the content is Aragog's heat content

**Why this check exists.** PROTEUS uses one heat-content formula, $T\,C_s - R$, for both core profiles. On the Gaussian profile, the core energy ledger uses this formula for the jump of the heat content at a structure update or an impact. If the formula did not reduce to Aragog's own heat content, that jump would not match the booked term, and the ledger of a `core_module` run on the default Gaussian profile would not close.

**What it tests.** An iron-alloy core on the Gaussian profile at 3800, 4100, 4500 and 5200 K, below and above its onset near 4149 K. The heat content that PROTEUS computes must equal Aragog's `heat_content`, and below the onset it must differ from $T\,C_s$ alone.

**How we know the right answer.** On the Gaussian profile $C_s$ does not depend on temperature, so Aragog's heat content is $T\,C_s - R(T)$ by its own definition.

**Why it matters.** The Gaussian profile is the default, so every `core_module` run reads this heat content in its core energy ledger.

This case has no plot: it checks an identity of the code.

??? info "Test details"
    - Reference: the definition of Aragog's heat content on the Gaussian profile.
    - Tolerance: `rel=1e-12`.
    - Test id: `tests/interior_energetics/test_aragog_core_tabulated.py::test_the_content_of_a_gaussian_budget_is_its_heat_content`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_aragog_core_tabulated.py`

## The update keeps the entropy and books the heat of the warmer parcels

**Why this check exists.** At a structure update the core is compressed without time to exchange heat, so its entropy must stay the same. A wrong rule gives a core that is too hot or too cold after the update, and a structure term that does not match the warming.

**What it tests.** The structure pair is carried from 7 CMB temperatures between 4600 and 6400 K. The specific entropy at the CMB must stay the same. The structure term must equal the sum over 256 parcels of equal mass of $c_p\,\Delta T$, with $c_p$ = 840 J kg$^{-1}$ K$^{-1}$ and $\Delta T$ the warming of the parcel.

**How we know the right answer.** At constant entropy each parcel warms by $\Delta T$. Its heat content changes by $c_p\,\Delta T$ with the $c_p$ of the core heat capacity. Aragog uses 840 J kg$^{-1}$ K$^{-1}$ there, because the PALEOS liquid has no electronic heat capacity. With the table $c_p$ at the CMB, 516 to 547 J kg$^{-1}$ K$^{-1}$ over these temperatures, the same sum is about 1.6 times smaller.

**Why it matters.** The term is the heat that the compression adds to the core, which the core later gives to the mantle.

![Structure term against the CMB temperature before the update](../../assets/validation/core_tabulated_structure_term.svg#only-light){ width="80%" }
![Structure term against the CMB temperature before the update](../../assets/validation/core_tabulated_structure_term_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the CMB temperature before the update in K, the y-axis the structure term in units of 1e29 J. The circles are the change of $T\,C_s - R$ that PROTEUS books. The crosses are the parcel sum with $c_p$ = 840 J kg$^{-1}$ K$^{-1}$, which the test asserts. The squares use the table $c_p$ and show the size of that choice. A pass means that the circles and the crosses coincide.

??? info "Test details"
    - Reference: $c_p\,dT$ per parcel at constant entropy, with the core heat capacity of the budget.
    - Tolerance: `rel=1e-10` for the CMB entropy; `rtol=1e-5` for the term (measured 4.7e-6 at 5000 K).
    - Test id: `tests/interior_energetics/test_aragog_core_tabulated.py::test_the_update_keeps_the_entropy_and_books_the_parcel_heat`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_aragog_core_tabulated.py --validation-plots`

## Other checks of this file

- With an inner core, the update keeps $M\,s_c - \Delta s_f\,M_{ic}$ to `rel=1e-9`, with $s_c$ the CMB entropy, $\Delta s_f$ the entropy of fusion and $M_{ic}$ the inner-core mass: `test_an_update_with_an_inner_core_keeps_the_total_entropy`.
- The ledger jump at the next call equals the booked term: `test_the_ledger_jump_across_the_update_is_the_booked_term`.
- The dynamo columns are 0 on this profile: `test_the_dynamo_columns_are_zero_on_the_tabulated_profile`.
