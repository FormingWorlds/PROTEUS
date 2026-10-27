# agni.py Validation

## What this page checks

This page checks the transparent limit of the AGNI atmosphere in PROTEUS, the branch of `run_agni` for an atmosphere with no opacity. In that limit the outgoing longwave radiation (OLR) is the thermal emission of the surface alone, which the Stefan-Boltzmann law gives in closed form.

AGNI has 2 radiation schemes. The grey-gas scheme treats the opacity as one value for all wavelengths; the banded scheme splits the spectrum into bands from a SOCRATES spectral file. PROTEUS maps the AGNI fluxes onto the helpfile, the table of output values that PROTEUS writes at each step.

Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the surface boundary condition, the band integration or the mapping of the AGNI fluxes onto the helpfile is wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

These are integration tests that need the Julia AGNI installation; they skip unless `PROTEUS_CI_NIGHTLY=1` is set, as in the nightly workflow.

## Source under test
`src/proteus/atmos_clim/agni.py` (transparent-atmosphere branch of `run_agni`)

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_integration_agni_transparent_limit::test_transparent_greygas_olr_equals_blackbody_emission` | Stefan-Boltzmann law, CODATA 2018 constant 5.670374419e-8 W m-2 K-4 | Transparent grey-gas solve returns `F_olr` = sigma T_surf^4 at 1000, 1500, 2000 and 3000 K (rel=1e-5) |
| `test_integration_agni_transparent_limit::test_transparent_greygas_olr_scales_with_surface_emissivity` | Kirchhoff's law: emissivity 1 minus the albedo | A surface of albedo 0.3 under a transparent column (optical depth 0) emits 0.7 sigma T_surf^4 at the top of the atmosphere and at the surface boundary (rel=1e-9) |
| `test_integration_agni_transparent_limit::test_transparent_banded_olr_recovers_blackbody_emission` | Stefan-Boltzmann law, evaluated through the spectral file and the SOCRATES two-stream solver | The banded transparent solve recovers sigma T_surf^4 from 1000 K to 3000 K. The largest residual measured across that range is 8.1e-5 relative, and the test admits 5e-4 |

## Black-body emission of a grey-gas transparent column

**Why this check exists.** In the transparent limit the OLR must equal $\sigma T_s^4$, with $T_s$ the surface temperature and $\sigma$ the Stefan-Boltzmann constant. A wrong temperature power, a wrong constant or a wrong mapping of the AGNI flux onto `F_olr` gives a wrong OLR without an error message.

**What it tests.** The test runs the grey-gas scheme in transparent mode, with a black surface (albedo 0) and no stellar flux, at surface temperatures of 1000, 1500, 2000 and 3000 K and compares `F_olr` with $\sigma T_s^4$ at each. The span of a factor of 3 separates the $T^4$ law from a $T^3$ law (flux ratios 81 against 27).

**How we know the right answer.** With no opacity, the surface emission reaches the top unchanged, so the OLR is the black-body flux of the surface, with the CODATA 2018 constant 5.670374419e-8 W m$^{-2}$ K$^{-4}$.

**Why it matters.** `F_olr` is the energy flux that the planet loses to space. PROTEUS uses this branch when the surface pressure is below `agni.psurf_thresh`, and the test drives it as the coupling loop does (`init_agni_atmos`, `update_agni_atmos`, `run_agni`).

![Outgoing longwave flux of the transparent grey-gas column against surface temperature](../../assets/validation/atmos_clim_agni_greygas_blackbody.svg#only-light){ width="80%" }
![Outgoing longwave flux of the transparent grey-gas column against surface temperature](../../assets/validation/atmos_clim_agni_greygas_blackbody_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the surface temperature $T_s$ and the y-axis the flux, both on log scales. The line is the black-body law $\sigma T_s^4$, with $\sigma$ the Stefan-Boltzmann constant, and the dashed line the $\sigma T_s^3$ form that the test rejects. The atmosphere is a transparent grey gas, with one absorption coefficient for all wavelengths, above a black surface. The circles are the asserted outgoing longwave radiation $F_\mathrm{olr}$ at 1000, 1500, 2000 and 3000 K. A pass means that the circles lie on the line (relative 1e-5).

??? info "Test details"
    - Reference: Stefan-Boltzmann law, CODATA 2018 constant 5.670374419e-8 W m-2 K-4.
    - Tolerance: relative 1e-5 at each surface temperature.
    - Test id: `tests/integration/test_integration_agni_transparent_limit.py::test_transparent_greygas_olr_equals_blackbody_emission`
    - Command: `PROTEUS_CI_NIGHTLY=1 pytest -m reference_pinned tests/integration/test_integration_agni_transparent_limit.py --validation-plots`

## Emission of a grey surface

**Why this check exists.** A surface with albedo $a$ emits only $(1 - a)$ of the black-body flux. A dropped emissivity, or $a$ used in place of $1 - a$, gives a wrong surface flux without an error message.

**What it tests.** With a surface albedo of 0.3 the test checks, at each of the 4 temperatures, that the column has zero optical depth and that both `F_olr` and the upward longwave flux at the surface equal $0.7\,\sigma T_s^4$. It also checks that the result is more than 25 % away from $\sigma T_s^4$ (a dropped emissivity) and away from $0.3\,\sigma T_s^4$ (the albedo used as the emissivity).

**How we know the right answer.** Kirchhoff's law sets the emissivity of the surface to 1 minus its albedo, and with zero optical depth that emission reaches the top unchanged.

**Why it matters.** The surface emissivity sets `F_olr`: a wrong emissivity scales it by the wrong factor, here $1 - a$ = 0.7.

![Outgoing longwave flux and surface upward flux of the transparent column above a grey surface](../../assets/validation/atmos_clim_agni_greygas_grey_surface.svg#only-light){ width="80%" }
![Outgoing longwave flux and surface upward flux of the transparent column above a grey surface](../../assets/validation/atmos_clim_agni_greygas_grey_surface_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the surface temperature $T_s$ and the y-axis the flux, both on log scales, for a surface albedo of 0.3. The solid line is $0.7\,\sigma T_s^4$, the emission of a surface with emissivity $1 - 0.3 = 0.7$. The dashed line is $\sigma T_s^4$ (emissivity dropped), and the dotted line $0.3\,\sigma T_s^4$ (albedo used as emissivity); the test rejects both. The circles are the asserted outgoing longwave radiation $F_\mathrm{olr}$, and the hollow squares the asserted upward longwave flux at the surface. A pass means that circles and squares lie on the solid line (relative 1e-9).

??? info "Test details"
    - Reference: Kirchhoff's law, emissivity 1 minus the albedo, with zero optical depth (analytical limit).
    - Tolerance: relative 1e-9 on `F_olr` and on the surface flux; absolute 1e-12 on the optical depth.
    - Test id: `tests/integration/test_integration_agni_transparent_limit.py::test_transparent_greygas_olr_scales_with_surface_emissivity`
    - Command: `PROTEUS_CI_NIGHTLY=1 pytest -m reference_pinned tests/integration/test_integration_agni_transparent_limit.py --validation-plots`

## Black-body emission through the banded scheme

**Why this check exists.** The banded scheme integrates the Planck function over the bands of a spectral file with the SOCRATES two-stream solver. A wrong band integration gives a wrong OLR without an error message.

**What it tests.** The banded transparent solve at the 4 surface temperatures must recover $\sigma T_s^4$, with more than 1 band. The largest residual measured over that range is 8.1e-5 relative.

**How we know the right answer.** The same Stefan-Boltzmann limit. The banded scheme falls slightly below it, because it evaluates the Planck function at band centres and truncates it outside the spectral range.

**Why it matters.** Runs with a SOCRATES spectral file take `F_olr` from this band integration.

**Known limit.** The test needs the spectral file in `FWL_DATA` and skips without it.

![Outgoing longwave flux of the transparent banded solve against surface temperature](../../assets/validation/atmos_clim_agni_banded_blackbody.svg#only-light){ width="80%" }
![Outgoing longwave flux of the transparent banded solve against surface temperature](../../assets/validation/atmos_clim_agni_banded_blackbody_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The axes and lines are those of the grey-gas case: $\sigma T_s^4$ (line) and $\sigma T_s^3$ (dashed, context). Here the solve uses the banded spectral file and the SOCRATES two-stream radiative transfer solver. The circles are the asserted outgoing longwave radiation $F_\mathrm{olr}$ at the 4 temperatures. The banded flux lies below $\sigma T_s^4$ by less than the tolerance, too little to see at this scale. A pass means that the circles lie on the line (relative 5e-4).

??? info "Test details"
    - Reference: Stefan-Boltzmann law, through the spectral file and the SOCRATES two-stream solver.
    - Tolerance: relative 5e-4 at each surface temperature.
    - Test id: `tests/integration/test_integration_agni_transparent_limit.py::test_transparent_banded_olr_recovers_blackbody_emission`
    - Command: `PROTEUS_CI_NIGHTLY=1 pytest -m reference_pinned tests/integration/test_integration_agni_transparent_limit.py --validation-plots`

## Coverage

Besides the values in the cases, the grey-gas test asserts that `F_olr` is positive, between 1e4 and 1e5 W m$^{-2}$ at 1000 K, and rises with the surface temperature. It also asserts that the upward longwave flux is the same at the top and at the bottom of the column, that the net upward flux `F_atm` equals `F_olr`, and that the reflected flux `F_sct` is 0, as there is no stellar flux.

## Scope and limits

The pin covers the transparent limit only. It constrains the surface
boundary condition, the band integration and the flux plumbing; it does not
constrain the treatment of gas opacity, convection, condensation or the
non-linear energy-conserving solver, which have no closed-form reference at
the conditions PROTEUS runs. The banded case builds its spectral file from
FWL_DATA and skips when that file is absent.

## Last verified
2026-08-11
