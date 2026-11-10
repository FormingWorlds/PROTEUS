# Validation: `src/proteus/interior_energetics/adiabat.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
depth-resolved melt state of the boundary interior module against an analytical
limit or a cross-implementation comparison. The marker is registered in
`pyproject.toml`.

The module turns the magma temperature into a melt fraction by integrating a
mantle adiabat down the radial profile the interior structure module produced,
and comparing it with the tabulated solidus and liquidus at each level's own
pressure. Three ingredients come from outside PROTEUS, and each is pinned
below: the adiabatic gradient from the Zalmoxis equation of state, the melting
curves from the interior lookup tables, and the closed-form isentrope that the
explicit stepping approximates.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_adiabat.py::test_integrate_adiabat_recovers_the_constant_gradient_isentrope` | Analytical limit of the isentrope `dT/dP = nabla_ad T / P` | Pins the power-law solution `T = T_top (P/P_top)^nabla_ad` over a pressure ratio of 140. Guards against an exponent that has picked up a factor of two, an adiabat that cools with depth, and a pressure-unit slip. | 2026-11-10 |
| `tests/interior_energetics/test_adiabat.py::test_zalmoxis_gradient_is_in_the_silicate_range` | Cross-implementation: the adiabatic gradient Zalmoxis reads for its own structure adiabat, from the equation of state named by `interior_struct.zalmoxis.mantle_eos` | Confirms the gradient reaching the boundary backend is the one Zalmoxis uses, in the range silicate melt occupies (a few to a few tens of kelvin per gigapascal), falling with depth as the bulk modulus stiffens. Also confirms the solidus lies below the liquidus at every level of the tabulated curves and that both rise with depth. | 2026-11-10 |
| `tests/interior_energetics/test_adiabat.py::test_paleos_melting_curves_follow_the_published_liquidus` | Published benchmark: Fei et al. (2021, Nat. Commun. 12, 876) MgSiO3 liquidus of 6000 K at 140 GPa, and the Belonoshko et al. (2005) zero-pressure intercept of 1831 K | With the Zalmoxis structure module the melt model takes the PALEOS curves rather than a tabulated folder. Pins the liquidus at both anchors and the solidus at `mushy_zone_factor` times the liquidus. Guards against the tabulated Wolf and Bower (2018) curves leaking back in (4760 K at 140 GPa), a Pa-versus-GPa slip, and a solidus scaled by the wrong factor. | 2026-11-10 |

## Adiabatic gradient

Zalmoxis builds its structure adiabat by stepping

$$\frac{\mathrm{d}T}{\mathrm{d}P} = \nabla_\mathrm{ad}(P,T)\,\frac{T}{P}$$

down the pressure profile, with the dimensionless gradient
$\nabla_\mathrm{ad}$ looked up from the equation-of-state tables: column 8 of
the PALEOS tables, or the tabulated `dT/dP` of the Wolf and Bower (2018) and
RTPress sets converted as $\nabla_\mathrm{ad} = (\mathrm{d}T/\mathrm{d}P)\,P/T$.
The boundary interior module takes the same gradient through
`zalmoxis.mixing.get_mixed_nabla_ad`, rather than re-reading the tables, so the
two cannot drift apart when an equation of state is added or its lookup
changed.

The stepping here reproduces the surface-anchored branch of
`zalmoxis/eos/temperature.py::compute_adiabatic_temperature`, including its
skip below one bar and its clamp of each level to 100 K to 100000 K.

## Melting curves

With the Zalmoxis structure module the solidus and liquidus are always the
PALEOS MgSiO3 curves: the Zalmoxis `PALEOS-liquidus`, Belonoshko et al. (2005)
below 2.55 GPa and Fei et al. (2021) above, and a solidus of
`interior_struct.zalmoxis.mushy_zone_factor` times that liquidus (0.8 by
default, the Stixrude (2014) cryoscopic ratio). These are the curves Zalmoxis
uses for a PALEOS mantle, the curves Aragog is given for a PALEOS mantle, and
the curves the `liquidus_super` initial condition is solved against, so the
melt state, the structure and the initial condition share one melting model.
A `melting_dir` set alongside is not read by the melt model, and the
configuration logs a warning saying so.

With any other structure module the curves are the two-column
pressure-temperature files under
`FWL_DATA/interior_lookup_tables/Melting_curves/`, selected by
`interior_struct.melting_dir` and defaulting to `Wolf_Bower+2018`, the folder
the basic interior lookup-table download provides. These are the same curves
the Aragog backend is configured with for a non-PALEOS mantle, read through the
same `proteus.utils.data.load_melting_curve` reader.

`tests/interior_energetics/test_adiabat.py::test_paleos_melting_curves_follow_the_published_liquidus`
pins the PALEOS liquidus against the Fei et al. (2021) value of 6000 K at
140 GPa and the Belonoshko et al. (2005) value of 1831 K at zero pressure.

The Wolf and Bower (2018) liquidus rises from 1980 K at the surface to about
4760 K at 140 GPa; the corresponding solidus runs from 1420 K to about 4340 K.
A level whose adiabat lies above the local liquidus is wholly molten, one below
the local solidus wholly solid, and the melt fraction is linear in temperature
between them.

## Mass and volume melt fractions

`Phi_global_vol` is the shell-volume integral of the local melt fraction and
`Phi_global` its density-weighted counterpart, both taken over the density
profile the structure module produced. Because the deep mantle is both denser
and closer to its liquidus, the two differ during crystallisation, with
`Phi_global < Phi_global_vol`. They coincide only for a mantle of uniform
density, which is the case
`tests/interior_energetics/test_adiabat.py::test_melt_fractions_agree_on_a_uniform_mantle`
pins against the closed-form shell integral.

`RF_depth` is the depth of the adiabat-liquidus crossing as a fraction of the
planet radius, $1 - r_\mathrm{liq}/R_\mathrm{int}$, matching the convention the
Aragog and SPIDER backends report.
