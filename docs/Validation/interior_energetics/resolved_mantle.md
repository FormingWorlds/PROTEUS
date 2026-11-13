# Validation: `src/proteus/interior_energetics/resolved_mantle.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
pieces the boundary backend uses once it resolves the mantle on an adiabat: the melt curve an ODE right-hand side reads the melt fraction from, and
the radial shells handed to the tides modules. The marker is registered in
`pyproject.toml`.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_resolved_mantle.py::test_melt_curve_reduces_to_the_linear_melt_fraction_without_a_gradient` | Analytical limit: an isothermal mantle between a pressure-independent solidus and liquidus melts linearly, `phi = (T - T_sol) / (T_liq - T_sol)` | Pins the melt fraction between two nodes of the curve and its response `1 / (T_liq - T_sol)`. Guards against a response over half the span, a node temperature read in Celsius, and a response of the wrong sign. | 2026-11-10 |

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
curve returns a wholly molten mantle there without integrating.

## Shells

The mantle is split into `num_levels - 1` shells of equal thickness for the
tides modules, each carrying the mass-weighted melt fraction and temperature of
the adiabat over it. The shells conserve the mantle mass and the mass integral
of any field. The boundary backend's pin that a uniform tidal rate summed over
the shells gives the single-layer flux is listed on the
[boundary page](boundary.md).
