# Validation: `src/proteus/orbit/wrapper.py`

## What this page checks

This page checks the orbital period that the orbit wrapper, `proteus.orbit.wrapper`, computes from the semi-major axis and the masses. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, every orbital period in the helpfile is wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_wrapper.py::test_period_matches_keplers_third_law_for_earth_around_sun` | Kepler's third law (Kepler 1619, Harmonices Mundi Book V); modern statement in any classical-mechanics textbook | Standard | Pins the orbital period of Earth around the Sun against the observed sidereal year (365.256 days). Asserts sign and 0.5% agreement; catches AU-vs-m and M_sun-vs-kg unit slips. |

## Orbital period of the Earth

**Why this check exists.** The period depends on the cube of the semi-major axis and on the masses. A distance in the wrong unit or a mass in g where kg belongs gives a wrong period without an error message.

**What it tests.** `update_period` computes Kepler's third law, $T = 2 \pi \sqrt{a^3 / (G (M_\star + M_p))}$, with $a$ the semi-major axis, $G$ the gravitational constant and $M_\star$ and $M_p$ the masses of the star and the planet. For the Earth at 1 AU around the Sun the test expects the sidereal year, 365.256 days = 3.156e7 s. It also checks the closed form and that the period is positive.

**How we know the right answer.** The observed sidereal year of the Earth. The 0.5 % tolerance covers the rounding of `const_G` in `proteus.utils.constants` and the general-relativistic precession of about 0.001 %, which the Newtonian formula leaves out.

**Why it matters.** The orbit wrapper calls `update_period` in its orbit step and writes the orbital period to the helpfile.

??? info "Test details"
    - Reference: Kepler's third law and the observed sidereal year (365.256 days).
    - Tolerance: relative 5e-3 against the sidereal year; relative 1e-12 against the closed form.
    - Test id: `tests/orbit/test_wrapper.py::test_period_matches_keplers_third_law_for_earth_around_sun`
    - Command: `pytest tests/orbit/test_wrapper.py::test_period_matches_keplers_third_law_for_earth_around_sun`

## Re-derivation note

Kepler's third law:

```
T = 2 pi sqrt(a^3 / (G (M_star + M_planet)))
```

For Earth at 1 AU around the Sun, the result is 365.256 days =
3.156e7 seconds. The `rel=5e-3` tolerance covers the rounding of `const_G`
in `proteus.utils.constants` and the ~0.001% precession contribution from
GR omitted in the Newtonian formula.
