# Validation: `src/proteus/orbit/satellite.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.satellite` against a published source.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_satellite.py::test_update_satellite_angular_momentum_matches_korenaga_2023_eq60` | Korenaga (2023) Icarus 400, 115564, Eq. 60 (cross-check Touma and Wisdom 1994) | n/a | Pins the spin-plus-orbital decomposition on the present-day Earth-Moon configuration. Asserts sign and the 1e34-1e35 kg m^2 / s order of magnitude expected from Korenaga's Eq. 60. |

## Re-derivation note

The orchestrator populates `plan_sat_am` from

```
L_total = I_planet * omega_planet + M_satellite * sqrt(G (M_planet + M_sat) a_sat)
```

with `I_planet = (2/5) M_planet R_planet^2` for a solid sphere. This is the
Korenaga (2023) Eq. 60 form, equivalent to the textbook reduced-mass
orbital angular momentum `L_orb = mu * sqrt(G (M_pl + M_sat) a)` in the
`M_sat << M_planet` limit (where `mu = M_pl M_sat / (M_pl + M_sat) -> M_sat`).
For the Earth-Moon system the substitution carries a 1.2% relative error.

For Earth parameters this evaluates to ~2.85e34 kg m^2 / s, in agreement
with the Touma and Wisdom (1994) inventory.

## Pre-fix history

An earlier implementation used `M_planet` in the prefactor instead of
`M_satellite`. This inflated the orbital angular momentum by
`M_planet / M_satellite ~ 81` for the Earth-Moon system, returning
~2.4e36 instead of ~2.85e34. The swap was surfaced by this reference-
pinned test (which pinned the textbook value via Korenaga's Eq. 60) and
corrected in the commit landing this validation page; the swap had been
propagating into `dω_dt` and `da_dt` through the integration constant
`L`. Future evolution simulations using this module should produce
physically correct lunar-recession trajectories within Korenaga's
formulation.
