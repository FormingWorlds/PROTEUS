# dummy.py Validation

## What this page checks

This page checks the luminosity of the star and the stellar flux at the planet in the dummy star module, `src/proteus/star/dummy.py`. The dummy star computes both from the effective temperature and the radius of the star with the Stefan-Boltzmann law. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the energy that reaches the planet in a run with the dummy star is wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_star::test_calc_star_luminosity_solar` | IAU 2015 Resolution B3, nominal solar luminosity L_sun = 3.828e26 W | Solar Teff and radius yield L_sun via Stefan-Boltzmann law (rel=0.01) |
| `test_star::test_calc_instellation_earth_like` | IAU 2015 Resolution B3, nominal total solar irradiance S_0 = 1361 W/m^2 | Earth-like planet at 1 AU receives 1361 W/m^2 (rel=0.01) |

## Luminosity of the Sun

**Why this check exists.** The luminosity is $L = 4 \pi R^2 \sigma T_\mathrm{eff}^4$, with $R$ the stellar radius, $T_\mathrm{eff}$ the effective temperature and $\sigma$ the Stefan-Boltzmann constant. A wrong power of the temperature, a radius in km where m belongs, or a constant in cgs where SI belongs gives a wrong luminosity without an error message.

**What it tests.** `calc_star_luminosity` with the solar effective temperature and radius must return the nominal solar luminosity. A range check, $10^{26} < L < 10^{27}$ W, rejects a radius in km (which gives about 3.85e20 W) and a cgs-SI slip on $\sigma$ (a factor of 1000).

**How we know the right answer.** IAU 2015 Resolution B3 defines the nominal solar luminosity as 3.828e26 W.

**Why it matters.** With `star.module = 'dummy'`, `dummy.py` takes the stellar luminosity from `calc_star_luminosity`, and so does the orbit wrapper for `orbit.instellation_method = 'inst'`.

??? info "Test details"
    - Reference: IAU 2015 Resolution B3, nominal solar luminosity L_sun = 3.828e26 W.
    - Tolerance: relative 0.01; range check 1e26 to 1e27 W.
    - Test id: `tests/star/test_star.py::test_calc_star_luminosity_solar`
    - Command: `pytest tests/star/test_star.py::test_calc_star_luminosity_solar`

## Stellar flux at 1 AU

**Why this check exists.** The flux at the planet is the luminosity spread over a sphere with the radius of the orbit. A distance in km where m belongs, or a radius unit slip, changes the flux by large factors without an error message.

**What it tests.** `calc_instellation` with the solar effective temperature and radius at 1 AU (astronomical unit) must return the nominal total solar irradiance. A range check, 1000 to 2000 W/m$^2$, rejects a radius in km (about 1.4e-3 W/m$^2$) and an AU in km (about 1e6 times the right value).

**How we know the right answer.** IAU 2015 Resolution B3 defines the nominal total solar irradiance as 1361 W/m$^2$.

**Why it matters.** With `star.module = 'dummy'`, `update_instellation` takes the stellar flux at the planet from `calc_instellation`.

??? info "Test details"
    - Reference: IAU 2015 Resolution B3, nominal total solar irradiance S_0 = 1361 W/m^2.
    - Tolerance: relative 0.01; range check 1e3 to 2e3 W/m^2.
    - Test id: `tests/star/test_star.py::test_calc_instellation_earth_like`
    - Command: `pytest tests/star/test_star.py::test_calc_instellation_earth_like`

## Coverage

Both tests verify the Stefan-Boltzmann radiative transfer chain from stellar parameters to planetary irradiation. Discrimination guards pin the exponent (T^4, not T^3 or T^5) and the unit system (SI: metres, watts) against plausible unit-conversion regressions.

## Last verified
2026-05-24
