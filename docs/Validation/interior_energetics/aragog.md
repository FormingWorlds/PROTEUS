# Validation: the mixing length at the core-mantle boundary

## What this page checks

This page checks how the mixing length of the Aragog interior solver sets the heat flux out of the core (`interior_energetics.mixing_length`, read in `src/proteus/interior_energetics/aragog.py`).
The mixing length is the distance a hot or cold parcel of mantle rock moves before it mixes with its surroundings; a longer length gives stronger convection.
PROTEUS uses the `constant` length by default, and `nearest` on request; [Mixing-length theory](https://proteus-framework.org/aragog/Explanations/mixing_length.html) in the Aragog documentation explains both.
The same setting selects the mixing length of SPIDER (`-mixing_length 1` for `nearest`, 2 for `constant`); this page checks Aragog only.
A resume that changes the setting is refused unless `interior_energetics.aragog.allow_resume_law_change` is true; this field of the Aragog table also covers a SPIDER run.
Validation here means that we run the code on cases where physics gives a clear requirement, and check that the runs meet it.
The requirement is that heat leaves a hot core while the mantle above it convects, and that the result does not depend on the mesh.
If this part of the model is wrong, the core stops cooling for tens of thousands of years, while the mantle above it cools too fast.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters.
The box "Test details" at the end of each case gives the reference, the tolerance, the test id and the command.

## Terms of this page

- CMB: the core-mantle boundary.
- $T_\mathrm{cmb}$: the temperature of the bottom cell of the mantle, in K. Aragog reports it as the core temperature, as SPIDER does; it follows the core only while heat flows out of the core.
- $F_\mathrm{cmb}$: the heat flux out of the core at the CMB, in W m$^{-2}$; positive means heat flows from the core into the mantle.
- Row: one line of the run output `runtime_helpfile.csv`; PROTEUS writes one row per coupling step.
- Eddy diffusivity: the diffusivity, in m$^2$ s$^{-1}$, through which convection carries heat in the mixing-length model. It is 0 at a stable node, where the entropy does not fall with height; the CMB node is the exception, because it copies the diffusivity of the node above it.
- `constant`: the mixing length is 0.25 times the thickness of the mantle everywhere in the mantle.
- `nearest`: the mixing length is the distance to the nearer of the 2 boundaries, so it goes to 0 at the CMB and at the surface.
- Mesh: Aragog divides the mantle into cells, thin spherical shells. A uniform mesh has cells of equal thickness. A mesh with a CMB cell has a thin bottom cell, here 1500 m or 3000 m thick.
- Dip: a row after 10 kyr with $F_\mathrm{cmb}$ below 100 W m$^{-2}$.
- Lose the heat flux: 2 or more dips in a row. The test names call this "decouple".

## How the runs were made

All runs use the lid-seal tutorial configuration (`input/tutorials/tutorial_lid_seal.toml`, 1 Earth mass) for 100 kyr, except the last case, which uses a planet of 1.5 Earth masses for 300 kyr.
Each run sets the mesh and the mixing length in its configuration.
The tool `tools/verification/mixing_length/cmb_flux.py` names the settings of each run and stores $T_\mathrm{cmb}$ and $F_\mathrm{cmb}$ of every row of the 12 runs in `tests/data/mixing_length/cmb_flux.npz`, with the PROTEUS and Aragog commits of the runs.
The tests read this file; they do not run PROTEUS.
The table lists the 4 `reference_pinned` tests of this page.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_cmb_flux_coupled_on_every_mesh` | Heat leaves a hot core under a convecting mantle; the same result on every mesh | `constant` on 4 meshes: no 2 dips in a row, no step of $T_\mathrm{cmb}$ below -10 K after 2 kyr, $T_\mathrm{cmb}$ at 100 kyr within 25 K. | 2026-11-14 |
| `tests/interior_energetics/test_mixing_length_validation.py::test_the_nearest_length_decouples_on_80_nodes_and_on_the_3000_m_cell_only` | The same requirement, shown not met by `nearest` | `nearest` on 3 meshes: dips for more than 10 kyr on 2 meshes, none on the finest. | 2026-11-14 |
| `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_flux_coupled_with_the_rheology_on` | As the first case | `constant` with the rheology on, against `nearest`. | 2026-11-14 |
| `tests/interior_energetics/test_mixing_length_validation.py::test_a_15_earth_mass_planet_keeps_its_core_temperature_under_the_default` | A run of the same configuration that sets `constant` in its file | 1.5 Earth masses: $T_\mathrm{cmb}$ at 300 kyr within 1 K of 5199.9 K on the uniform mesh; on both meshes no row with $|F_\mathrm{cmb}|$ below 1 W m$^{-2}$ and no 2 dips in a row. | 2026-11-14 |

## The constant length keeps the heat flux out of the core on every mesh

**Why this check exists.**
Aragog gives the CMB node the eddy diffusivity of the first node above it, as SPIDER does, so the flux out of the core follows that node.
If that node turns stable, its diffusivity is 0, and the flux out of the core falls to about 0.
The test fails if, on any of the 4 meshes, 2 rows in a row have $F_\mathrm{cmb}$ below 100 W m$^{-2}$ after 10 kyr.

**What it tests.**
4 runs with `constant`: uniform meshes of 80 and 160 nodes, and meshes of 80 nodes with a CMB cell of 1500 m and of 3000 m.
The 3000 m cell is the default of PROTEUS.
After 10 kyr, no 2 consecutive rows may be dips; after 2 kyr, no step of $T_\mathrm{cmb}$ between 2 rows may fall by more than 10 K.
$T_\mathrm{cmb}$ at 100 kyr of the 4 runs must lie within 25 K of each other.
The test also asserts that the uniform meshes have no dip, and that both meshes with a CMB cell have at least 1.

**How we know the right answer.**
The core is hotter than the mantle above it, and that mantle convects, so heat must flow out of the core.
The test accepts single rows without this flux (see the known limit below), but not 2 in a row.
A model with a correct boundary treatment gives the same $T_\mathrm{cmb}$ on a coarse and on a fine mesh, so the test asks that $T_\mathrm{cmb}$ at 100 kyr of the 4 meshes lie within 25 K of each other; they lie within 7.6 K.
Because $T_\mathrm{cmb}$ is the temperature of the bottom cell, the thickness of that cell enters this comparison.
The 160-node mesh is the finest of the 4: its cells are about half as thick as on the 80-node mesh.

**Why it matters.**
$F_\mathrm{cmb}$ sets how fast the core cools, and with it the heat that reaches the base of the mantle.
A run that loses this flux keeps the heat in the core for tens of thousands of years, while the bottom cell of the mantle cools.
In the `nearest` runs of the next case, that cell becomes up to 237 K (80 nodes) and 302 K (3000 m cell) colder than in the `constant` run of the same mesh.

![Absolute heat flux out of the core against time for the constant mixing length on 4 meshes](../../assets/validation/interior_energetics_aragog_constant_cmb_flux.svg#only-light){ width="80%" }
![Absolute heat flux out of the core against time for the constant mixing length on 4 meshes](../../assets/validation/interior_energetics_aragog_constant_cmb_flux_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The x-axis is the time in kyr, from 2 kyr on.
The y-axis is $|F_\mathrm{cmb}|$ in W m$^{-2}$ on a log scale, one line per mesh.
The dashed line marks 100 W m$^{-2}$.
The lines stay above it, apart from single rows on the 2 meshes with a CMB cell.
A triangle marks a row with $F_\mathrm{cmb} < 0$: the log axis shows its absolute value, so such a row can lie above the dashed line and is still a dip.
A pass means that no line stays below the dashed line for 2 rows.

**Known limit.**
With a CMB cell, `constant` gives single dips: 3 rows on the 3000 m cell, at 22.0, 28.4 and 38.1 kyr, and 2 rows on the 1500 m cell.
The next row has a flux above 100 W m$^{-2}$ again, and $T_\mathrm{cmb}$ never rises by more than 0.61 K between 2 rows.
Each dip falls on a row in which `T_magma` falls by more than 25 K while `Phi_global` rises; the uniform 80-node mesh has no such row after 10 kyr, and no dip.
In the dips on the 3000 m cell the entropy gradient at the CMB, which the solver carries as a state, falls to a small positive value, and the CMB node keeps the diffusivity copied from the node above, so heat flows into the core for that row: a negative $F_\mathrm{cmb}$.
Around the dips this gradient is 2000 to 2600 times smaller than its absolute tolerance in the solver ($10^{-8}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$), so the solver does not control it; this is a candidate cause, and why the gradient falls is not known.
The aragog mesh study that supports the 3000 m cell runs the core module and the `nearest_boundary` length; for the defaults of PROTEUS, the `energy_balance` boundary and `constant`, the runs of this page are the check, and they show the single dips only on meshes with a CMB cell.

??? note "The dips in the run log"
    - On the 3000 m cell the run log (`output/c3000_constant/aragog.log` of the run, not in the repository) gives the gradient as $-4 \times 10^{-12}$ to $-5 \times 10^{-12}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ in the rows around the dips, and $+1.9 \times 10^{-17}$ to $+2.3 \times 10^{-17}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at the end of each dip row.
    - At 22.0 and 28.4 kyr the gradient starts and ends the step at this value: $F_\mathrm{cmb}$ is -0.011 W m$^{-2}$, and $T_\mathrm{cmb}$ changes by less than $10^{-5}$ K in the row.
    - At 38.1 kyr the gradient reaches this value within the step; the mean flux of the row, -595.5 W m$^{-2}$, needs a larger positive gradient during the step.
    - The 2 dips on the 1500 m cell, -110.4 and -322.5 W m$^{-2}$, are not traced in this way.
    - A run with the planet mass changed by 1 part in $10^{12}$ has its 3 dips within 1 yr of the same times, so the times come from the model, not from round-off; a tolerance effect is also the same in both runs, so this does not rule out a numerical cause. In that run $F_\mathrm{cmb}$ in the dips at 22.0 and 28.4 kyr is 1.6 and 1.0 W m$^{-2}$, not -0.011 W m$^{-2}$.

??? info "Test details"
    - Reference: heat flows out of a hot core under a convecting mantle; for the mesh comparison, the spread of $T_\mathrm{cmb}$ at 100 kyr over the 4 meshes.
    - Tolerance: at most 1 dip in a row after 10 kyr; steps of $T_\mathrm{cmb}$ of at least -10 K after 2 kyr; spread of $T_\mathrm{cmb}$ at 100 kyr at most 25 K.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_cmb_flux_coupled_on_every_mesh`
    - The measure of dips and steps is checked on a hand-made series: `tests/interior_energetics/test_mixing_length_validation.py::test_the_dip_measure_counts_consecutive_rows_after_10_kyr`; the stored file is checked for every case and its commits: `tests/interior_energetics/test_mixing_length_validation.py::test_the_stored_runs_hold_every_case_and_name_their_commits`.
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

## The nearest-boundary length loses the heat flux on the meshes of 80 nodes

**Why this check exists.**
This case shows why PROTEUS does not use `nearest` by default.
With `nearest`, the mixing length at the node above the CMB is its distance to the CMB, so it depends on the mesh.
The test fails if the 2 meshes of 80 nodes do not lose the heat flux, or if the 160-node mesh does.

**What it tests.**
3 runs with `nearest`: uniform meshes of 80 and 160 nodes, and 80 nodes with the 3000 m CMB cell.
On the 80-node uniform mesh and on the 3000 m cell, the dips must last more than 10 kyr.
On the 160-node mesh, no row after 10 kyr may be a dip.

**How we know the right answer.**
The requirement is the same as in the case above, and here the 2 meshes of 80 nodes do not meet it.
Counted from the last row above 100 W m$^{-2}$, the dips last 27.0 kyr on the 80-node mesh and 30.0 kyr on the 3000 m cell, with $F_\mathrm{cmb}$ below 1 W m$^{-2}$ in each dip row.
In these dips the entropy gradient at the CMB, the state the solver carries, reaches -0.137 (80 nodes) and -1.33 J kg$^{-1}$ K$^{-1}$ m$^{-1}$ (3000 m cell) in the run logs, large and negative: the flux is lost through an eddy diffusivity of 0, copied from the stable node above, not through the gradient.
On 160 nodes the flux stays above 359 W m$^{-2}$.
So the loss depends on the mesh, not on the planet.
The bottom cell of the 160-node mesh is about half as thick as on the 80-node mesh and thicker than the 3000 m cell, so the thickness of the bottom cell alone does not decide it; which property of the mesh does is not known.

**Why it matters.**
This case is the reason for the default: on the meshes PROTEUS uses, `nearest` makes the core lose the heat flux for about 30 kyr.

![Absolute heat flux out of the core against time for the nearest-boundary mixing length on 3 meshes](../../assets/validation/interior_energetics_aragog_nearest_cmb_flux.svg#only-light){ width="80%" }
![Absolute heat flux out of the core against time for the nearest-boundary mixing length on 3 meshes](../../assets/validation/interior_energetics_aragog_nearest_cmb_flux_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The axes are as in the plot above.
The lines of the 80-node mesh and of the 3000 m cell fall below the dashed line, to about 1 W m$^{-2}$ and lower, for about 30 kyr.
When the flux returns, one row reaches 2.6 times $10^5$ W m$^{-2}$ on the 3000 m cell at 48.6 kyr, and 4.5 times $10^5$ W m$^{-2}$ on the 80-node mesh at 57.5 kyr: the heat held in the core leaves it.
The line of the 160-node mesh stays above the dashed line.
A pass means exactly this pattern: the 2 meshes of 80 nodes lose the heat flux for more than 10 kyr and the 160-node mesh does not.

**Known limit.**
The `constant` length departs from published models, because it does not go to 0 at the boundaries.
Abe (1997)[^cite-abe1997] and Bower et al. (2018)[^cite-bower2018] use the distance to the nearest boundary as the mixing length; Bower et al. (2018) use 0.25 times the thickness of the mantle only in their simplified model.
Tachinami et al. (2011)[^cite-tachinami2011] use 0.82 times this distance, and Wagner et al. (2019)[^cite-wagner2019] calibrate a profile that also goes to 0 at both boundaries.
The Aragog issue [#113](https://github.com/FormingWorlds/aragog/issues/113) tracks a CMB treatment for `nearest`.

??? info "Test details"
    - Reference: the requirement of the first case, which `nearest` does not meet on the meshes of 80 nodes.
    - Tolerance: a span of dips longer than 10 kyr on the 2 meshes of 80 nodes; 0 dips after 10 kyr on 160 nodes.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_the_nearest_length_decouples_on_80_nodes_and_on_the_3000_m_cell_only`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

## The constant length keeps the heat flux with the rheology on

**Why this check exists.**
With the rheology on, the viscosity of solid rock depends on temperature, pressure and stress, and a stagnant lid can form at the surface ([Solid-state convection](https://proteus-framework.org/aragog/Explanations/solid_state_convection.html)).
The viscous part of the eddy diffusivity then uses a length of its own, which goes to 0 at both boundaries.
The test fails if `constant` loses the heat flux in this setup.

**What it tests.**
The 3000 m CMB cell with the rheology on (`interior_energetics.aragog.rheology.enabled = true`), once with `constant` and once with `nearest`.
For `constant`, no 2 consecutive rows may be dips after 10 kyr, and no step of $T_\mathrm{cmb}$ may fall by more than 10 K after 2 kyr.
The `nearest` run is shown for comparison; the test does not bound it.

**How we know the right answer.**
The requirement is that of the first case.
The viscous part of the diffusivity does not depend on the `mixing_length` choice, so we expect the 2 runs to differ only where the inviscid part acts, near the CMB.

**Why it matters.**
The default must keep the heat flux with the rheology on as well as off.

![Absolute heat flux out of the core against time with the rheology on, for the constant and the nearest-boundary mixing length](../../assets/validation/interior_energetics_aragog_rheology_cmb_flux.svg#only-light){ width="80%" }
![Absolute heat flux out of the core against time with the rheology on, for the constant and the nearest-boundary mixing length](../../assets/validation/interior_energetics_aragog_rheology_cmb_flux_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
The axes are as in the plots above.
The `constant` line stays above the dashed line, apart from 2 single rows; a triangle marks a row with $F_\mathrm{cmb} < 0$.
The `nearest` line falls below it for 30.3 kyr, as on the 3000 m cell without the rheology.
A pass means that the `constant` line never stays below the dashed line for 2 rows.

??? info "Test details"
    - Reference: the requirement of the first case.
    - Tolerance: at most 1 dip in a row after 10 kyr; steps of $T_\mathrm{cmb}$ of at least -10 K after 2 kyr.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_the_constant_length_keeps_the_flux_coupled_with_the_rheology_on`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

## A planet of 1.5 Earth masses under the default mixing length

**Why this check exists.**
The cases above use 1 Earth mass; this case runs the default on a planet of 1.5 Earth masses with the 2-phase PALEOS equation of state of MgSiO$_3$.
Under the current default, `constant`, the run without the key and the reference run have the same settings, so the case is a record of these runs; `tests/config/test_defaults.py` pins the default itself.
The test fails if the run under the default does not reach, within 1 K, the $T_\mathrm{cmb}$ at 300 kyr of the run that sets `constant` in its file.

**What it tests.**
The configuration `tools/verification/mixing_length/planet_1p5_earth.toml`: 1.5 Earth masses, 80 nodes and the melt-solid mixture viscosity for melt separation (`separation_viscosity = "mixture"`), run for 300 kyr without `mixing_length` in its configuration, so under the default mixing length; the file holds the full configuration of the runs, so its other settings stay fixed.
On the uniform mesh, $T_\mathrm{cmb}$ at 300 kyr must be 5199.9 K within 1 K.
On both meshes, the uniform one and the default one with the 3000 m CMB cell, no row may have $|F_\mathrm{cmb}|$ below 1 W m$^{-2}$, and no 2 consecutive rows may be dips after 10 kyr.
The test does not bound the steps of $T_\mathrm{cmb}$ between rows in this case.

**How we know the right answer.**
The reference is a run of the same configuration that sets `mixing_length = "constant"` in its file; it reaches 5199.9 K at 300 kyr.
This run is not stored; the test holds only its value at 300 kyr.
The run under the default reaches 5199.88 K on the uniform mesh, and 5206.63 K on the default mesh; the test bounds only the uniform mesh.

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
The steps of $T_\mathrm{cmb}$ between rows are not a measure of the heat flux in this run, so the 10 K bound of the first case does not apply.
After 2 kyr, $T_\mathrm{cmb}$ falls by up to 46 K per kyr, and by 2.5 K per kyr (uniform mesh) and 2.6 K per kyr (default mesh) in the row nearest 100 kyr; the rows are up to 4000 yr apart.
So after 2 kyr, 28 steps on the uniform mesh and 29 on the default mesh fall by more than 10 K.
The largest, -39.7 K on the uniform mesh and -39.8 K on the default mesh, comes at 3.6 kyr, where the rows are 860 yr apart.
$F_\mathrm{cmb}$ is at least 666 W m$^{-2}$ at each of these steps, and about 13,500 W m$^{-2}$ at the largest, so the heat flux is kept.

??? info "Test details"
    - Reference: a run of the same configuration that sets `constant` in its file: $T_\mathrm{cmb}$ = 5199.9 K at 300 kyr.
    - Tolerance: 1 K at 300 kyr on the uniform mesh; on both meshes 0 rows with $|F_\mathrm{cmb}|$ below 1 W m$^{-2}$ and at most 1 dip in a row after 10 kyr.
    - Test id: `tests/interior_energetics/test_mixing_length_validation.py::test_a_15_earth_mass_planet_keeps_its_core_temperature_under_the_default`
    - Command: `pytest -m reference_pinned tests/interior_energetics/test_mixing_length_validation.py --validation-plots`

[^cite-abe1997]: Y. Abe, *[Thermal and chemical evolution of the terrestrial magma ocean](https://doi.org/10.1016/S0031-9201(96)03229-3)*, Physics of the Earth and Planetary Interiors, 100, 27-39, 1997. [SciX](https://scixplorer.org/abs/1997PEPI..100...27A/abstract).
[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
[^cite-tachinami2011]: C. Tachinami, H. Senshu, S. Ida, *[Thermal evolution and lifetime of intrinsic magnetic fields of super-Earths in habitable zones](https://doi.org/10.1088/0004-637X/726/2/70)*, The Astrophysical Journal, 726, 70, 2011. [SciX](https://scixplorer.org/abs/2011ApJ...726...70T/abstract).
[^cite-wagner2019]: F. W. Wagner, A.-C. Plesa, A. B. Rozel, *[Calibrating mixing-length theory for thermal convection in rocky planets](https://doi.org/10.1093/gji/ggy543)*, Geophysical Journal International, 217, 75-89, 2019. [SciX](https://scixplorer.org/abs/2019GeoJI.217...75W/abstract).
