# Validation: `src/proteus/interior_energetics/resolved_mantle.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
pieces the boundary backend uses once it resolves the mantle on an adiabat: the melt curve an ODE right-hand side reads the melt fraction from, and
the radial shells handed to the tides modules. The marker is registered in
`pyproject.toml`.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_resolved_mantle.py::test_melt_curve_reduces_to_the_linear_melt_fraction_without_a_gradient` | Analytical limit: an isothermal mantle between a pressure-independent solidus and liquidus melts linearly, `phi = (T - T_sol) / (T_liq - T_sol)` | Pins the melt fraction between two nodes of the curve and its response `1 / (T_liq - T_sol)`. Guards against a response over half the span, a node temperature read in Celsius, and a response of the wrong sign. | 2026-11-10 |
| `tests/interior_energetics/test_resolved_mantle.py::test_melt_curve_thermal_response_is_one_for_an_isothermal_mantle` | Analytical limit: an isothermal mantle sits at T_p at every level, so the mass mean of `dT/dT_p` and `dT_cmb/dT_p` are both 1 | Pins both responses at 1 inside the mushy zone and above the hottest liquidus. Guards against a slope taken over two node spacings. | 2026-11-13 |

## Melt curve

An ODE solver evaluates the melt fraction and its response to the magma
temperature many times per step, too often to integrate an adiabat each time.
`MeltCurve` integrates the adiabat once per node of a uniform 1 K grid in the
anchor temperature, keeps every node it has computed for as long as the
structure mesh is unchanged, and interpolates linearly in between. The
response it reports is the slope of the segment the temperature falls in, so

$$\int_{T_b}^{T_a} \frac{\mathrm{d}\phi}{\mathrm{d}T}\,\mathrm{d}T = \phi(T_a) - \phi(T_b)$$

holds exactly for any path, and the latent heat a solver draws while the mantle
cools from $T_a$ to $T_b$ is $L M_\mathrm{mantle}$ times the melt that froze.
`test_melt_curve_latent_heat_closes_on_the_change_in_melt_fraction` checks this
closure across 37 node boundaries on a sloped adiabat, and that the interpolant
stays within 1e-4 of the exact adiabat melt fraction there.

An anchor at or above the hottest liquidus anywhere in the mantle leaves every
level molten, because the integrated adiabat never cools with depth, so the
curve returns a wholly molten melt fraction there without integrating.

Each node also holds the mass-weighted mean temperature of the adiabat and its
temperature at the core-mantle boundary. Their segment slopes are the thermal
responses the boundary backend's heat capacity uses, the mass mean of
$(\partial T/\partial T_p)_P$ and $\mathrm{d}T_\mathrm{cmb}/\mathrm{d}T_p$,
so the sensible heat drawn between two temperatures closes on the change in
mean mantle temperature and in $T_\mathrm{cmb}$ in the same way.
`test_melt_curve_sensible_heat_closes_on_the_change_in_mean_temperature` checks
this on a temperature-scaled adiabat, where `test_melt_curve_thermal_response_matches_the_adiabat_derivative`
also checks the responses against the exact derivative $T/T_p$. A molten
mantle needs the adiabat for these, unlike for its melt fraction.

## Shells

The mantle is split into `num_levels - 1` shells of equal thickness for the
tides modules, each carrying the mass-weighted melt fraction and temperature of
the adiabat over it. The shells conserve the mantle mass and the mass integral
of any field. The boundary backend's pin that a uniform tidal rate summed over
the shells gives the single-layer flux is listed on the
[boundary page](boundary.md).
