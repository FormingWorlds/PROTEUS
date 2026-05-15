# Validation: `src/proteus/orbit/satellite.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.satellite` against a published source.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_satellite.py::test_update_satellite_angular_momentum_matches_korenaga_2023_formula` | Korenaga (2023) formulation, as implemented in `proteus.orbit.satellite.Ltot` | n/a | Pins the spin-plus-orbital decomposition on the present-day Earth-Moon configuration. Asserts sign and the 1e36-1e37 kg m^2 / s order of magnitude that the implemented formula returns. |

## Re-derivation note

The orchestrator populates `plan_sat_am` from

```
L_total = I_planet * omega_planet + M_planet * sqrt(G (M_planet + M_sat) a_sat)
```

with `I_planet = (2/5) M_planet R_planet^2` for a solid sphere. For Earth
parameters this evaluates to ~2.4e36 kg m^2 / s.

This is NOT the textbook orbital angular momentum of the Moon around
Earth. The textbook value, ~2.85e34 kg m^2 / s (Touma and Wisdom 1994),
uses the reduced mass `mu = M_planet * M_sat / (M_planet + M_sat)` instead
of `M_planet` in the sqrt prefactor. The Korenaga (2023) decomposition is
~80x larger because `M_planet / mu ~ M_planet / M_sat ~ 80` for the
Earth-Moon system.

The test pins the source-implemented value (closed form against the
formula in `Ltot`) and brackets the order of magnitude implied by that
formula. It does NOT certify that the formula matches the textbook
reduced-mass convention. Reconciling the two conventions, or confirming
that the Korenaga formulation is intentional, is a follow-up science item.
