# Validation: `src/proteus/interior_energetics/boundary.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
boundary interior backend, the zero-dimensional energy balance of Schaefer et
al. (2016), against an analytical limit or a cross-implementation comparison.
The marker is registered in `pyproject.toml`. The pins on the radiogenic decay
and the solidification front this backend takes from the shared helpers are
listed on the [common helpers page](common.md).

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_boundary.py::test_adiabat_isothermal_limit_reproduces_the_linear_cooling_rate` | Cross-implementation: the solidification-front energy balance of Schaefer et al. (2016), ApJ 829, 63, section 2, as implemented by the linear melt model | With no adiabatic gradient, pressure-independent melting curves and a uniform density, the adiabat melt model's heat capacity must equal `M (c_p phi + L / (T_liq - T_sol))` and its cooling rate must equal the linear model's. Guards against charging the whole mantle's sensible heat, dropping the latent term, and a scale or sign slip. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_core_heat_capacity_is_added_only_when_requested` | Closed form: core heat capacity `core_heatcap * M_core` lumped with the mantle, as in the dummy backend | With `include_core_heat_capacity` the heat capacity gains 450 J kg-1 K-1 x 2e24 kg = 9.0e26 J/K whatever the melt state, so a wholly molten mantle holds 5.7e27 J/K; the reported `Cp_eff` stays the mantle's own. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_sensible_heat_whole_cools_the_whole_partly_molten_mantle` | Closed form: `c_p M + L M / (T_liq - T_sol)` against `c_p phi M + L M / (T_liq - T_sol)` | Pins 7.46667e27 J/K (`"whole"`) and 6.10987e27 J/K (`"molten"`) at 1850.4 K, equal outside the mushy zone, and no jump in sensible heat at the solidus with `"whole"`. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_tidal_viscosity_kervazo_hands_the_tides_the_dummy_rheology` | Kervazo et al. (2021), A&A 650, A72: melt-fraction rheology, as `eval_rheoparam` evaluates it | With `tidal_viscosity = "kervazo"` a wholly solid mantle is handed 1.2567e19 Pa s per shell and a wholly molten one 1.0 Pa s, either side of the 1e9 Pa s viscous-block threshold of LovePy; the convective viscosity is unchanged. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_top_melt_fraction_is_the_top_level_of_the_adiabat` | Cross-implementation: the top level of the full adiabat integration, `AdiabatMeltModel.state(T_p).phi[-1]` | The closed form `(T_p - T_sol,top) / (T_liq,top - T_sol,top)` used with `viscosity_phi = "top"` matches the integrated adiabat and exceeds the mass average, because the mantle freezes from the bottom up. | 2026-11-10 |

## Energy balance

While the mantle is partially molten, Schaefer et al. (2016) charge sensible
heat only to the magma ocean above the solidification radius $r_s$, and add the
latent heat the front releases as it moves:

$$\left[\tfrac{4}{3}\pi\rho c_p\,(R^3 - r_s^3)
- 4\pi r_s^2 \rho L \frac{\mathrm{d}r_s}{\mathrm{d}T_p}\right]\frac{\mathrm{d}T_p}{\mathrm{d}t}
= -4\pi R^2 q_m + M_\mathrm{mantle}\,(Q_\mathrm{radio} + Q_\mathrm{tidal}).$$

With a uniform density, the molten mass above the front is
$\phi M_\mathrm{mantle}$, so the bracket is
$M_\mathrm{mantle}\,(c_p\,\phi + L\,\mathrm{d}\phi/\mathrm{d}T_p)$. The adiabat
melt model (`boundary.melt_model = "adiabat"`; the default is `"linear"`) keeps this form and
takes $\phi$ as the mass melt fraction of the adiabat integrated down the
structure's mantle profile, against the tabulated, pressure-dependent solidus
and liquidus. The response $\mathrm{d}\phi/\mathrm{d}T_p$ comes from the
[melt curve](resolved_mantle.md), so the latent heat drawn along a step closes
on the melt that froze. A wholly molten or wholly solid mantle has no front,
and the whole mantle cools with $c_p M_\mathrm{mantle}$ under either model.

The pinned test drives both models to the same state, where the adiabat model
must reduce to the linear one: an isothermal mantle between constant melting
temperatures, uniform in density, at a potential temperature between two nodes
of the melt curve.

## Tides

Under the adiabat model the mantle is handed to the tides modules as
`num_levels - 1` shells of equal thickness, and the tidal heat in the energy
balance is each shell's rate times its mass, summed. A uniform rate reduces to
the single-layer `H M_mantle / A` of the linear model;
`test_adiabat_tidal_heating_is_summed_over_the_shells` checks this to rounding,
and that heating only the base shell deposits that shell's share alone.
