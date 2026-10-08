# star.py Validation

## Source under test
`src/proteus/star/star.py` (via `src/proteus/star/dummy.py`)

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_star::test_calc_star_luminosity_solar` | IAU 2015 Resolution B3, nominal solar luminosity L_sun = 3.828e26 W | Solar Teff and radius yield L_sun via Stefan-Boltzmann law (rel=0.01) |
| `test_star::test_calc_instellation_earth_like` | IAU 2015 Resolution B3, nominal total solar irradiance S_0 = 1361 W/m^2 | Earth-like planet at 1 AU receives 1361 W/m^2 (rel=0.01) |

## Coverage

Both tests verify the Stefan-Boltzmann radiative transfer chain from stellar
parameters to planetary irradiation. Discrimination guards pin the exponent
(T^4, not T^3 or T^5) and the unit system (SI: metres, watts) against
plausible unit-conversion regressions.

## Last verified
2026-05-24
