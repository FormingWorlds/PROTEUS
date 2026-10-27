# Validation: `src/proteus/orbit/wrapper.py`

## What this page checks

This page checks the orbital period that the orbit wrapper, `proteus.orbit.wrapper`, computes from the semi-major axis and the masses. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, every orbital period in the helpfile, the table of output values that PROTEUS writes at each step, is wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_wrapper.py::test_period_matches_keplers_third_law_for_earth_around_sun` | Kepler's third law (Kepler 1619, Harmonices Mundi Book V); modern statement in any classical-mechanics textbook | Standard | Pins the orbital period of Earth around the Sun to the closed form at `rel=1e-12`, the exact check. The comparison with the observed sidereal year (365.256 days) to 0.5% is a loose unit check that catches AU-vs-m and M_sun-vs-kg slips; the test also asserts a positive period. |

## Orbital period of the Earth

**Why this check exists.** The square of the period scales with the cube of the semi-major axis and inversely with the total mass. A distance in the wrong unit or a mass in g where kg belongs gives a wrong period without an error message.

**What it tests.** `update_period` computes Kepler's third law, $T = 2 \pi \sqrt{a^3 / (G (M_\star + M_p))}$, with $a$ the semi-major axis, $G$ the gravitational constant and $M_\star$ and $M_p$ the masses of the star and the planet. For the Earth at 1 AU around the Sun the test expects the sidereal year, the time of one orbit against the fixed stars: 365.256 days = 3.156e7 s. It also checks the closed form and that the period is positive.

**How we know the right answer.** The exact check is the closed form of Kepler's third law, which the test asserts to a relative 1e-12. The test also compares the period with the observed sidereal year of the Earth to 0.5 %; that is a loose check of the units.

**Why it matters.** The orbit wrapper calls `update_period` in its orbit step and writes the orbital period to the helpfile.

![Orbital period against semi-major axis for an Earth-mass planet around the Sun](../../assets/validation/orbit_wrapper_period.svg#only-light){ width="80%" }
![Orbital period against semi-major axis for an Earth-mass planet around the Sun](../../assets/validation/orbit_wrapper_period_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the semi-major axis and the y-axis the orbital period in days, both on log scales. The planet has the mass of the Earth and orbits the Sun. The line is `update_period`, and the squares are Kepler's third law (context). The filled circle is the asserted code value at 1 AU, and the hollow diamond the sidereal year, 365.256 days. The test allows 0.5 %. That is too little to see. A pass means that the circle sits on the line and in the diamond.

??? info "Test details"
    - Reference: Kepler's third law and the observed sidereal year (365.256 days).
    - Tolerance: relative 5e-3 against the sidereal year; relative 1e-12 against the closed form.
    - Test id: `tests/orbit/test_wrapper.py::test_period_matches_keplers_third_law_for_earth_around_sun`
    - Command: `pytest -m reference_pinned tests/orbit/test_wrapper.py --validation-plots`
