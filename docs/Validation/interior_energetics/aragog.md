# Validation: the mixing length at the core-mantle boundary

## What this page checks

This page checks how the mixing length of the Aragog interior solver sets the heat flux out of the core (`interior_energetics.mixing_length`, read in `src/proteus/interior_energetics/aragog.py`).
The mixing length is the distance a hot or cold parcel of mantle rock moves before it mixes with its surroundings; a longer length gives stronger convection.
PROTEUS uses the `constant` length by default, and `nearest` on request; [Mixing-length theory](https://proteus-framework.org/aragog/Explanations/mixing_length.html) in the Aragog documentation explains both.
Validation here means that we run the code on cases where physics gives a clear requirement, and check that the runs meet it.
The requirement is that heat leaves a hot core while the mantle above it convects, and that the result does not depend on the mesh.
If this part of the model is wrong, the core stops cooling for tens of thousands of years, and the core temperature and the mantle melt fraction at later times are wrong.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters.
The box "Test details" at the end of each case gives the reference, the tolerance, the test id and the command.

**Terms of this page.**

- CMB: the core-mantle boundary.
- $T_\mathrm{cmb}$: the temperature at the CMB, in K.
- $F_\mathrm{cmb}$: the heat flux out of the core at the CMB, in W m$^{-2}$; positive means heat flows from the core into the mantle.
- Row: one line of the run output `runtime_helpfile.csv`; PROTEUS writes one row per coupling step.
- $D$: the thickness of the mantle, from the CMB to the surface.
- `constant`: the mixing length is $0.25\,D$ everywhere in the mantle.
- `nearest`: the mixing length is the distance to the nearer of the 2 boundaries, so it goes to 0 at the CMB and at the surface.
- Mesh: Aragog divides the mantle into cells, thin spherical shells. A uniform mesh has cells of equal thickness. A mesh with a CMB cell has a thin bottom cell, here 1500 m or 3000 m thick.
- Dip: a row after 10 kyr with $F_\mathrm{cmb}$ below 100 W m$^{-2}$.

All runs use the lid-seal tutorial configuration (`input/tutorials/tutorial_lid_seal.toml`, 1 Earth mass) for 100 kyr, except the last case, which uses a planet of 1.5 Earth masses for 300 kyr.
Each run sets the mesh and the mixing length in its configuration.
The tool `tools/verification/mixing_length/cmb_flux.py` stores $T_\mathrm{cmb}$ and $F_\mathrm{cmb}$ of every row of the 12 runs in `tests/data/mixing_length/cmb_flux.npz`, with the PROTEUS and Aragog commits of the runs.
The tests read this file; they do not run PROTEUS.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_cmb_flux_coupled_on_every_mesh` | Heat leaves a hot core under a convecting mantle; the same result on every mesh | `constant` on 4 meshes: no 2 dips in a row, no step of $T_\mathrm{cmb}$ below -10 K after 2 kyr, $T_\mathrm{cmb}$ at 100 kyr within 25 K. | 2026-11-14 |
| `tests/interior_energetics/test_mixing_length_validation.py::test_the_nearest_length_decouples_on_80_nodes_and_on_the_3000_m_cell_only` | The same requirement, shown not met by `nearest` | `nearest` on 3 meshes: dips for more than 10 kyr on 2 meshes, none on the finest. | 2026-11-14 |
| `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_flux_coupled_with_the_rheology_on` | As the first case | `constant` with the rheology on, against `nearest`. | 2026-11-14 |
| `tests/interior_energetics/test_mixing_length_validation.py::test_a_15_earth_mass_planet_keeps_its_core_temperature_under_the_default` | A run of the same configuration that sets `constant` in its file | 1.5 Earth masses: $T_\mathrm{cmb}$ at 300 kyr within 1 K of 5199.9 K, no 2 dips in a row. | 2026-11-14 |

## The constant length keeps the heat flux out of the core on every mesh

**Why this check exists.**
Aragog gives the CMB node the eddy diffusivity of the node above it, as SPIDER does.
If that node turns stable, its diffusivity is 0, and the flux out of the core falls to about 0.
The test fails if, on any of the 4 meshes, 2 rows in a row have $F_\mathrm{cmb}$ below 100 W m$^{-2}$ after 10 kyr.

**What it tests.**
4 runs with `constant`: uniform meshes of 80 and 160 nodes, and meshes of 80 nodes with a CMB cell of 1500 m and of 3000 m.
The 3000 m cell is the default of PROTEUS.
After 10 kyr, no 2 consecutive rows may be dips; after 2 kyr, no step of $T_\mathrm{cmb}$ between 2 rows may fall by more than 10 K.
$T_\mathrm{cmb}$ at 100 kyr of the 4 runs must lie within 25 K of each other.
The test also asserts that the uniform meshes have no dip, and that both meshes with a CMB cell have at least 1.

**How we know the right answer.**
The core is hotter than the mantle above it, and that mantle convects, so heat must flow out of the core in every row.
A model with a correct boundary treatment gives the same $T_\mathrm{cmb}$ on a coarse and on a fine mesh.
The 160-node mesh is the reference: its cells are about half as thick as on the 80-node mesh.

**Why it matters.**
$F_\mathrm{cmb}$ sets how fast the core cools, and with it the heat that reaches the base of the mantle.
A run that loses this flux keeps a hot core and a hot mantle base for tens of thousands of years.

![Absolute heat flux out of the core against time for the constant mixing length on 4 meshes](../../assets/validation/interior_energetics_aragog_constant_cmb_flux.svg#only-light){ width="80%" }
![Absolute heat flux out of the core against time for the constant mixing length on 4 meshes](../../assets/validation/interior_energetics_aragog_constant_cmb_flux_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The x-axis is the time in kyr, from 2 kyr on.
The y-axis is $|F_\mathrm{cmb}|$ in W m$^{-2}$ on a log scale, one line per mesh.
The dashed line marks 100 W m$^{-2}$.
The lines stay above it, apart from single rows on the 2 meshes with a CMB cell.
A pass means that no line stays below the dashed line for 2 rows.

**Known limit.**
With a CMB cell, `constant` gives single dips: 3 rows on the 3000 m cell, with $F_\mathrm{cmb}$ as low as -595.5 W m$^{-2}$, and 2 rows on the 1500 m cell, as low as -322.5 W m$^{-2}$.
A negative $F_\mathrm{cmb}$ means that, for one row, heat flows into the core.
In the deepest dip, at 38.1 kyr on the 3000 m cell, the entropy gradient at the CMB turns slightly positive, and the large diffusivity of the node above acts on it.
The next row has a flux above 100 W m$^{-2}$ again, and $T_\mathrm{cmb}$ never rises by more than 0.61 K between 2 rows.
A run with the planet mass changed by 1 part in $10^{12}$ has its 3 dips within 1 yr of the same times, so they come from the model, not from round-off.

??? info "Test details"
    - Reference: heat flows out of a hot core under a convecting mantle; the uniform 160-node mesh for the mesh comparison.
    - Tolerance: at most 1 dip in a row after 10 kyr; steps of $T_\mathrm{cmb}$ of at least -10 K after 2 kyr; spread of $T_\mathrm{cmb}$ at 100 kyr at most 25 K.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_cmb_flux_coupled_on_every_mesh`
    - The measure of dips and steps is checked on a hand-made series: `tests/interior_energetics/test_mixing_length_validation.py::test_the_dip_measure_counts_consecutive_rows_after_10_kyr`; the stored file is checked for every case and its commits: `tests/interior_energetics/test_mixing_length_validation.py::test_the_stored_runs_hold_every_case_and_name_their_commits`.
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

## The nearest-boundary length loses the heat flux on coarse cells at the CMB

**Why this check exists.**
This case shows why PROTEUS does not use `nearest` by default.
With `nearest`, the mixing length at the node above the CMB is about the thickness of the bottom cell, so convection there is weak and the node can turn stable.
The test fails if the 2 coarse meshes do not lose the flux, or if the 160-node mesh does.

**What it tests.**
3 runs with `nearest`: uniform meshes of 80 and 160 nodes, and 80 nodes with the 3000 m CMB cell.
On the 80-node uniform mesh and on the 3000 m cell, the dips must last more than 10 kyr.
On the 160-node mesh, no row after 10 kyr may be a dip.

**How we know the right answer.**
The requirement is the same as in the case above, and here the coarse meshes do not meet it.
The dips last 27.0 kyr on the 80-node mesh and 30.0 kyr on the 3000 m cell, with $F_\mathrm{cmb}$ below 1 W m$^{-2}$ in each dip row.
On 160 nodes the flux stays above 359 W m$^{-2}$.
So the loss depends on the bottom cell: it is a property of the mesh, not of the planet.

**Why it matters.**
Abe (1997)[^cite-abe1997] and Bower et al. (2018)[^cite-bower2018] use the distance to the nearest boundary as the mixing length.
Tachinami et al. (2011)[^cite-tachinami2011] use 0.82 times this distance, and Wagner et al. (2019)[^cite-wagner2019] calibrate a profile that also goes to 0 at both boundaries.
The `constant` length of PROTEUS departs from all 4: it does not go to 0 at the boundaries.
PROTEUS accepts this, because on the meshes it uses, `nearest` cuts the core off from the mantle.
The Aragog issue [#113](https://github.com/FormingWorlds/aragog/issues/113) tracks a CMB treatment for `nearest`.

![Absolute heat flux out of the core against time for the nearest-boundary mixing length on 3 meshes](../../assets/validation/interior_energetics_aragog_nearest_cmb_flux.svg#only-light){ width="80%" }
![Absolute heat flux out of the core against time for the nearest-boundary mixing length on 3 meshes](../../assets/validation/interior_energetics_aragog_nearest_cmb_flux_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The axes are as in the plot above.
The lines of the 80-node mesh and of the 3000 m cell fall below the dashed line, to about 1 W m$^{-2}$ and lower, for about 30 kyr.
When the flux returns, at 48.6 and 57.5 kyr, one row reaches 2.6 to 4.5 times $10^5$ W m$^{-2}$.
The line of the 160-node mesh stays above the dashed line.
A pass means exactly this pattern: the 2 coarse meshes lose the flux for more than 10 kyr and the fine mesh does not.

??? info "Test details"
    - Reference: the requirement of the first case, which `nearest` does not meet on the coarse meshes.
    - Tolerance: a span of dips longer than 10 kyr on the 2 coarse meshes; 0 dips after 10 kyr on 160 nodes.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_the_nearest_length_decouples_on_80_nodes_and_on_the_3000_m_cell_only`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

## The constant length keeps the heat flux with the rheology on

**Why this check exists.**
With the rheology on, the viscosity of the mantle follows its melt fraction, and the viscous part of the eddy diffusivity uses a length of its own.
The test fails if `constant` loses the flux in this setup.

**What it tests.**
The 3000 m CMB cell with the rheology on (`interior_energetics.aragog.rheology.enabled = true`), once with `constant` and once with `nearest`.
For `constant`, no 2 consecutive rows may be dips after 10 kyr, and no step of $T_\mathrm{cmb}$ may fall by more than 10 K after 2 kyr.
The `nearest` run is shown for comparison; the test does not bound it.

**How we know the right answer.**
The requirement is that of the first case.
The viscous part of the diffusivity does not depend on the `mixing_length` choice, so we expect the 2 runs to differ only where the inviscid part acts, near the CMB.

**Why it matters.**
The default must keep the core coupled with the rheology on as well as off.

![Absolute heat flux out of the core against time with the rheology on, for the constant and the nearest-boundary mixing length](../../assets/validation/interior_energetics_aragog_rheology_cmb_flux.svg#only-light){ width="80%" }
![Absolute heat flux out of the core against time with the rheology on, for the constant and the nearest-boundary mixing length](../../assets/validation/interior_energetics_aragog_rheology_cmb_flux_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The axes are as in the plots above.
The `constant` line stays above the dashed line, apart from 2 single rows.
The `nearest` line falls below it for 30.3 kyr, as on the 3000 m cell without the rheology.
A pass means that the `constant` line never stays below the dashed line for 2 rows.

??? info "Test details"
    - Reference: the requirement of the first case.
    - Tolerance: at most 1 dip in a row after 10 kyr; steps of $T_\mathrm{cmb}$ of at least -10 K after 2 kyr.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_flux_coupled_with_the_rheology_on`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

## A planet of 1.5 Earth masses keeps its core temperature under the default

**Why this check exists.**
The cases above use 1 Earth mass. This case checks the default on a larger planet with the 2-phase PALEOS equation of state of MgSiO$_3$.
The test fails if the run under the default does not reach the core temperature of the run that sets `constant` in its file.

**What it tests.**
The 1.5 Earth-mass configuration with 80 nodes and the melt-solid mixture viscosity for melt separation (`separation_viscosity = "mixture"`), run for 300 kyr without `mixing_length` in its configuration, so under the default.
On the uniform mesh, $T_\mathrm{cmb}$ at 300 kyr must be 5199.9 K within 1 K, and no row may have $|F_\mathrm{cmb}|$ below 1 W m$^{-2}$.
On both meshes, the uniform one and the default one with the 3000 m CMB cell, no 2 consecutive rows may be dips after 10 kyr.
The test does not bound the steps of $T_\mathrm{cmb}$ between rows in this case.

**How we know the right answer.**
The reference is a run of the same configuration that sets `mixing_length = "constant"` in its file; it reaches 5199.9 K at 300 kyr.
The run under the default reaches 5199.88 K on the uniform mesh, and 5206.63 K on the default mesh.

**Why it matters.**
A configuration without the key must give the same result as one that sets `constant`.

![Core-mantle boundary temperature against time for a planet of 1.5 Earth masses on 2 meshes under the default mixing length](../../assets/validation/interior_energetics_aragog_15_earth_mass_tcmb.svg#only-light){ width="80%" }
![Core-mantle boundary temperature against time for a planet of 1.5 Earth masses on 2 meshes under the default mixing length](../../assets/validation/interior_energetics_aragog_15_earth_mass_tcmb_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The x-axis is the time in kyr, and the y-axis $T_\mathrm{cmb}$ in K.
The 2 lines are the uniform and the default mesh.
The dot at 300 kyr marks 5199.9 K, the value of the reference run on the uniform mesh.
A pass means that the uniform line ends on the dot.

**Known limit.**
The steps of $T_\mathrm{cmb}$ between rows are not a measure of coupling in this run, so the 10 K bound of the first case does not apply.
The core cools smoothly, by up to 46 K per kyr early on and by 2.6 K per kyr at 100 kyr, and the rows are up to 4000 yr apart.
So 28 steps on the uniform mesh and 29 on the default mesh fall by more than 10 K.
The largest, -39.7 K, comes at 3.6 kyr, where the rows are 860 yr apart; the reference run has the same step.
$F_\mathrm{cmb}$ is at least 667 W m$^{-2}$ at each of these steps, and about 13,000 W m$^{-2}$ at the largest, so the core stays coupled.

??? info "Test details"
    - Reference: a run of the same configuration that sets `constant` in its file: $T_\mathrm{cmb}$ = 5199.9 K at 300 kyr.
    - Tolerance: 1 K at 300 kyr; 0 rows with $|F_\mathrm{cmb}|$ below 1 W m$^{-2}$ on the uniform mesh; at most 1 dip in a row after 10 kyr on both meshes.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_a_15_earth_mass_planet_keeps_its_core_temperature_under_the_default`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

[^cite-abe1997]: Y. Abe, *[Thermal and chemical evolution of the terrestrial magma ocean](https://doi.org/10.1016/S0031-9201(96)03229-3)*, Physics of the Earth and Planetary Interiors, 100, 27-39, 1997. [SciX](https://scixplorer.org/abs/1997PEPI..100...27A/abstract).
[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
[^cite-tachinami2011]: C. Tachinami, H. Senshu, S. Ida, *[Thermal evolution and lifetime of intrinsic magnetic fields of super-Earths in habitable zones](https://doi.org/10.1088/0004-637X/726/2/70)*, The Astrophysical Journal, 726, 70, 2011. [SciX](https://scixplorer.org/abs/2011ApJ...726...70T/abstract).
[^cite-wagner2019]: F. W. Wagner, A.-C. Plesa, A. B. Rozel, *[Calibrating mixing-length theory for thermal convection in rocky planets](https://doi.org/10.1093/gji/ggy543)*, Geophysical Journal International, 217, 75-89, 2019. [SciX](https://scixplorer.org/abs/2019GeoJI.217...75W/abstract).
