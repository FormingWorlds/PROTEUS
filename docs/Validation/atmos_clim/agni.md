# agni.py Validation

## Source under test
`src/proteus/atmos_clim/agni.py` (transparent-atmosphere branch of `run_agni`)

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_integration_agni_transparent_limit::test_transparent_greygas_olr_equals_blackbody_emission` | Stefan-Boltzmann law, CODATA 2018 constant 5.670374419e-8 W m-2 K-4 | Transparent grey-gas solve returns `F_olr` = sigma T_surf^4 at 1000, 1500, 2000 and 3000 K (rel=1e-5) |
| `test_integration_agni_transparent_limit::test_transparent_greygas_olr_scales_with_surface_emissivity` | Kirchhoff's law with the Schwarzschild solution for an isothermal slab | A surface of albedo 0.3 under a slab of optical depth tau emits `sigma T_surf^4 (1 - 0.3 exp(-tau))`; the surface boundary condition itself is 0.7 sigma T_surf^4 (rel=1e-9) |
| `test_integration_agni_transparent_limit::test_transparent_banded_olr_recovers_blackbody_emission` | Stefan-Boltzmann law, evaluated through the spectral file and the SOCRATES two-stream solver | The banded transparent solve recovers sigma T_surf^4 to better than 1e-4 relative from 1000 K to 3000 K (rel=5e-4) |

## Coverage

The three tests pin the analytical limit of radiative transfer. Transparent
mode holds the column isothermal at the surface temperature, so above a
black surface the column re-emits exactly what it absorbs and the outgoing
longwave radiation is the black-body flux `sigma * T_surf^4`, independent of
any residual opacity. The limit is exact for the grey-gas scheme and is
recovered to a few times 1e-5 relative by the banded scheme, where the
Planck function is evaluated at band centres and truncated outside the
spectral range.

A reflective surface separates the two contributions: the emitted flux
becomes `sigma * T_surf^4 (1 - a exp(-tau))`, the attenuated grey-body beam
plus the emission of the slab above it. Pinning that form tests the surface
emissivity and the slab emission at once, and the difference from the bare
grey-body flux is 0.5 per cent, well above the tolerance.

The comparison runs at four surface temperatures spanning a factor of three.
That span separates the T^4 law from a T^3 law by a factor of 81 against 27
in the flux ratio, which is the exponent guard. The remaining guards pin the
sign of the emitted flux, its absolute scale in W m-2, the constancy of the
upward longwave flux through a transparent column, and the equality of the
net flux with the emitted flux at zero instellation.

The tests drive the wrapper the same way the coupling loop does, through
`init_agni_atmos`, `update_agni_atmos` and `run_agni`, so they also pin the
selection of the transparent branch at `P_surf` below `agni.psurf_thresh`
and the mapping of the AGNI flux arrays onto the helpfile fields `F_olr`,
`F_atm` and `F_sct`.

## Scope and limits

The pin covers the transparent limit only. It constrains the surface
boundary condition, the band integration and the flux plumbing; it does not
constrain the treatment of gas opacity, convection, condensation or the
non-linear energy-conserving solver, which have no closed-form reference at
the conditions PROTEUS runs. The banded case builds its spectral file from
FWL_DATA and skips when that file is absent.

## Last verified
2026-08-11
