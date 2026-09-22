# Validation: `src/proteus/orbit/wrapper.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
orbital-mechanics helpers in `proteus.orbit.wrapper`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_wrapper.py::test_period_matches_keplers_third_law_for_earth_around_sun` | Kepler's third law (Kepler 1619, Harmonices Mundi Book V); modern statement in any classical-mechanics textbook | Standard | Pins the orbital period of Earth around the Sun against the observed sidereal year (365.256 days). Asserts sign and 0.5% agreement; catches AU-vs-m and M_sun-vs-kg unit slips. |

## Re-derivation note

Kepler's third law:

```
T = 2 pi sqrt(a^3 / (G (M_star + M_planet)))
```

For Earth at 1 AU around the Sun, the result is 365.256 days =
3.156e7 seconds. The `rel=5e-3` tolerance covers the rounding of `const_G`
in `proteus.utils.constants` and the ~0.001% precession contribution from
GR omitted in the Newtonian formula.
