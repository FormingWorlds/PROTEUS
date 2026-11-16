# Validation: `src/proteus/interior_energetics/boundary.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
boundary interior backend, the zero-dimensional energy balance of Schaefer et
al. (2016)[^cite-schaefer2016], against an analytical limit or a cross-implementation comparison.
The marker is registered in `pyproject.toml`. The pins on the radiogenic decay
and the solidification front this backend takes from the shared helpers are
listed on the [common helpers page](common.md).

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_boundary.py::test_adiabat_isothermal_limit_reproduces_the_linear_cooling_rate` | Cross-implementation: the solidification-front energy balance of Schaefer et al. (2016), ApJ 829, 63, section 2, as implemented by the linear melt model with `sensible_heat = "whole"` | With no adiabatic gradient, pressure-independent melting curves and a uniform density, the adiabat melt model's heat capacity must equal `M (c_p + L / (T_liq - T_sol))` and its cooling rate must equal the linear model's. Guards against charging only the molten part's sensible heat, dropping the latent term, and a scale or sign slip. | 2026-11-13 |
| `tests/interior_energetics/test_boundary.py::test_core_heat_capacity_is_added_only_when_requested` | Closed form: core heat capacity `core_heatcap * M_core` lumped with the mantle, as in the dummy backend | With `include_core_heat_capacity` the heat capacity gains 450 J kg-1 K-1 x 2e24 kg = 9.0e26 J/K whatever the melt state, so a wholly molten mantle holds 5.7e27 J/K; the reported `Cp_eff` stays the mantle's own. On an isothermal mantle T_cmb follows T_p one to one. | 2026-11-13 |
| `tests/interior_energetics/test_boundary.py::test_sensible_heat_whole_cools_the_whole_partly_molten_mantle` | Closed form: `c_p M + L M / (T_liq - T_sol)` against `c_p phi M + L M / (T_liq - T_sol)` | Pins 7.46667e27 J/K (`"whole"`) and 6.10987e27 J/K (`"molten"`) on the linear model at 1850.4 K, equal outside the mushy zone, and no jump in sensible heat at the solidus with `"whole"`. The adiabat model gives 7.46667e27 J/K with either setting. | 2026-11-13 |
| `tests/interior_energetics/test_boundary.py::test_tidal_viscosity_kervazo_hands_the_tides_the_dummy_rheology` | Kervazo et al. (2021)[^cite-kervazo2021], A&A 650, A72: melt-fraction rheology, as `eval_rheoparam` evaluates it | With `tidal_viscosity = "kervazo"` a wholly solid mantle is handed 1.2567e19 Pa s per shell and a wholly molten one 1.0 Pa s, either side of the 1e9 Pa s viscous-block threshold of LovePy; the convective viscosity is unchanged. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_top_melt_fraction_is_the_top_level_of_the_adiabat` | Cross-implementation: the top level of the full adiabat integration, `AdiabatMeltModel.state(T_p).phi[-1]` | The closed form `(T_p - T_sol,top) / (T_liq,top - T_sol,top)` used with `viscosity_phi = "top"` matches the integrated adiabat and exceeds the mass average, because the mantle freezes from the bottom up. | 2026-11-10 |

## Energy balance

While the mantle is partially molten, Schaefer et al. (2016) charge sensible
heat only to the magma ocean above the solidification radius $r_s$, and add the
latent heat the front releases as it moves:

$$\left\lbrack \tfrac{4}{3}\pi\rho c_p\,(R^3 - r_s^3)
- 4\pi r_s^2 \rho L \frac{\mathrm{d}r_s}{\mathrm{d}T_p}\right\rbrack \frac{\mathrm{d}T_p}{\mathrm{d}t}
= -4\pi R^2 q_m + M_\mathrm{mantle}\,(Q_\mathrm{radio} + Q_\mathrm{tidal}).$$

The linear melt model (`boundary.melt_model = "linear"`, the default) keeps
this form with `sensible_heat = "molten"`, and charges the whole mantle,
$c_p M_\mathrm{mantle}$, with `"whole"`. With a uniform density the molten
mass above the front is $\phi M_\mathrm{mantle}$, so the latent term is
$L M_\mathrm{mantle}\,\mathrm{d}\phi/\mathrm{d}T_p$.

The adiabat melt model (`boundary.melt_model = "adiabat"`) ties the
temperature of every level to $T_p$ along the adiabat, so a change of $T_p$
moves the whole mantle, solid or molten, and moves the deep levels more than
the top. The heat capacity is

$$C = c_p M_\mathrm{mantle} \left\langle \left(\frac{\partial T}{\partial T_p}\right)_P \right\rangle_m
+ L M_\mathrm{mantle} \frac{\mathrm{d}\phi}{\mathrm{d}T_p}
+ c_\mathrm{core} M_\mathrm{core} \frac{\mathrm{d}T_\mathrm{cmb}}{\mathrm{d}T_p},$$

with $\langle\cdot\rangle_m$ the mass mean over the mantle, $\phi$ the mass
melt fraction of the adiabat against the pressure-dependent solidus and
liquidus, and the core term present with `include_core_heat_capacity`.
`sensible_heat` is not read. All three responses come from the
[melt curve](resolved_mantle.md), so the heat drawn along a step closes on the
change in mean mantle temperature, melt fraction and core temperature. With a
constant $\nabla_\mathrm{ad}$ the adiabat is $T \propto T_p$, so
$\partial T/\partial T_p = T/T_p$ exceeds 1 at depth, and taking the
responses as 1 understates the heat capacity. A wholly molten or wholly solid
mantle has no front and draws no latent heat.

The sensible term does not depend on the melt fraction, so the heat capacity
changes only by the latent term where the top of the mantle crosses its
solidus. A molten-only sensible term, $c_p \phi M_\mathrm{mantle}$, would fall
there from about 1200 to about 50 J kg-1 K-1, because only a thin top layer is
molten and $\mathrm{d}\phi/\mathrm{d}T_p$ is small;
`test_adiabat_heat_capacity_follows_the_adiabat` checks both.

The pinned test drives both models to the same state, where the adiabat model
must reduce to the linear one with `sensible_heat = "whole"`: an isothermal
mantle between constant melting temperatures, uniform in density, at a
potential temperature between two nodes of the melt curve.

## Tides

Under the adiabat model the mantle is handed to the tides modules as
`num_levels - 1` shells of equal thickness, and the tidal heat in the energy
balance is each shell's rate times its mass, summed. A uniform rate reduces to
the single-layer `H M_mantle / A` of the linear model;
`test_adiabat_tidal_heating_is_summed_over_the_shells` checks this to rounding,
and that heating only the base shell deposits that shell's share alone.

## Structure re-solve

Under the adiabat model the backend publishes the adiabat on every level of the
structure's profile (`interior_o.temp_profile`), and a Zalmoxis re-solve
(`wrapper.interior_temperature_handoff`) follows it on pressure, so the top of
the mantle stays at $T_p$ wherever the re-solve puts it. The re-solve also
leaves `T_magma` and `T_surf` to the backend: Zalmoxis sets them from the
solved structure only in the initial-condition solves. Feeding the shell means
back instead, with the top node written to `T_magma`, raised $T_p$ by the top
shell's adiabatic increment on every re-solve.
`test_adiabat_structure_resolve_does_not_heat_the_mantle` runs five boundary
steps without net cooling, each followed by the real re-solve with a stand-in
Zalmoxis solve on a temperature-scaled adiabat, and requires `T_magma` to stay
at its starting value; without the change it rises by 300 K.

[^cite-kervazo2021]: Kervazo, M., Tobie, G., Choblet, G., Dumoulin, C. & Běhounková, M., *[Solid tides in Io's partially molten interior: Contribution of bulk dissipation](https://doi.org/10.1051/0004-6361/202039433)*, Astronomy & Astrophysics, 650, A72, 2021.

[^cite-schaefer2016]: Schaefer, L., Wordsworth, R.D., Berta-Thompson, Z. & Sasselov, D., *[Predictions of the atmospheric composition of GJ 1132b](https://doi.org/10.3847/0004-637X/829/2/63)*, The Astrophysical Journal, 829, 63, 2016.
