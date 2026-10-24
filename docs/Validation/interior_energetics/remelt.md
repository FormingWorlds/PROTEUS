# Validation: `src/proteus/interior_energetics/remelt.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the partial giant-impact re-melt in `proteus.interior_energetics.remelt`. The rule raises the outermost mantle shells that hold the melted mass fraction $f$ to the initial condition, so its contract is geometric: the melt front of a mantle of uniform density has a closed form. The slow-tier test below checks the same rule on a live Aragog solver.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_remelt.py::test_the_melt_front_of_a_uniform_mantle_matches_the_closed_form` | Analytical limit: in a shell of uniform density between $r_c$ and $R$, the outer mass fraction $f$ lies above $r_f = (R^3 - f (R^3 - r_c^3))^{1/3}$ | `remelt_weights` on 400 cells uniform in radius between 3500 and 6400 km, at $f$ = 0.05 to 0.99. The front of the partly melted cell matches the closed form to `rtol=1e-12`. A fill linear in radius misses by 280 km at $f = 0.3$, and a fill from the base up misses by more than 100 km at each tested fraction except $f = 0.5$, where the two fills meet. | 2026-10-24 |
| `tests/integration/test_slow_accretion_partial_remelt.py::test_the_melted_mass_closes_on_the_pre_impact_cell_masses` | Wiring check against the interior solver: the staggered mesh and the EOS density of the Aragog solver PROTEUS builds (fwl-aragog 26.10.22), after a short cooled run | Slow tier. At $f = 0.3$ on a real solver, for the `energy_balance` and the `core_module` core, the weights recovered from the re-melted entropy close against $f \sum \rho(P, S_\mathrm{end}) V$ to `rel=1e-9`; against the cell volumes alone they miss by more than 1e-3. | 2026-10-24 |

## Melt front against the closed form

### Plot: melt front

![Melt front radius against the melted mass fraction](../../assets/validation/interior_energetics_remelt_front.svg#only-light){ width="80%" }
![Melt front radius against the melted mass fraction](../../assets/validation/interior_energetics_remelt_front_dark.svg#only-dark){ width="80%" }

Radius of the melt front against the melted mass fraction $f$ for a mantle of uniform density between 3500 and 6400 km: the closed form (line), a fill linear in radius (dashed), and the front of the weights from `remelt_weights` (circles) at the 7 fractions of the test. The front sits above the linear fill because the outer shells hold more mass per unit radius.

- Reference: the closed form above, from the shell mass $\frac{4}{3}\pi\rho(R^3 - r_f^3) = f\,\frac{4}{3}\pi\rho(R^3 - r_c^3)$.
- Tolerance: `rtol=1e-12`, the round-off of the cumulative cell masses.
- Test id: `tests/interior_energetics/test_remelt.py::test_the_melt_front_of_a_uniform_mantle_matches_the_closed_form`
- Command: `pytest -m reference_pinned tests/interior_energetics/test_remelt.py --validation-plots`

## Limits checked alongside

`tests/interior_energetics/test_remelt.py` also pins the ends of the rule: $f = 1$ gives `np.maximum(S_end, S_ic)` bit for bit, the full re-melt; $f = 0$ leaves every cell unchanged; no cell cools; and the scalar mantle moves to $T_\mathrm{end} + f\,(T_\mathrm{init} - T_\mathrm{end})$, the enthalpy mean of a melted and an unmelted part of one heat capacity.
