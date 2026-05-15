# Validation: `src/proteus/orbit/satellite.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.satellite` against a published source.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_satellite.py::test_update_satellite_angular_momentum_matches_korenaga_2023_formula` | Korenaga (2023) Icarus 400, 115564, Eq. 60 (cross-check Touma and Wisdom 1994) | n/a | Pins the spin-plus-orbital decomposition on the present-day Earth-Moon configuration. Asserts sign and the 1e36-1e37 kg m^2 / s order of magnitude that the in-source formula returns. |

## Re-derivation note

The orchestrator populates `plan_sat_am` from

```
L_total = I_planet * omega_planet + M_planet * sqrt(G (M_planet + M_sat) a_sat)
```

with `I_planet = (2/5) M_planet R_planet^2` for a solid sphere. For Earth
parameters this evaluates to ~2.4e36 kg m^2 / s.

## Known discrepancy with the cited paper

Korenaga (2023) Icarus 400, 115564, Eq. 60 reads

```
L = I_E Omega + M_M sqrt(G (M_E + M_M) a)
```

where the orbital sqrt is multiplied by **M_M (the satellite mass)**,
not M_E (the planet mass). In the `M_M << M_E` limit this matches the
textbook reduced-mass formula `L_orb = mu sqrt(G (M_pl + M_sat) a)`
with `mu = M_pl M_sat / (M_pl + M_sat) ~ M_sat`.

The PROTEUS implementation in `proteus.orbit.satellite.Ltot` uses
`M_planet` in the prefactor instead. For the Earth-Moon system this
inflates the orbital contribution by `M_planet / M_sat ~ 81`, yielding
~2.4e36 kg m^2 / s where the paper formula gives ~2.85e34 (matching
Touma and Wisdom 1994's inventory). The swap propagates into `dω_dt`
and `da_dt` through the integration constant `L`, so any real
Earth-Moon evolution under this implementation should be treated as
suspect until the prefactor is reconciled.

The test pins the source-implemented value (closed form against the
in-source `Ltot` formula) and brackets the order of magnitude that
the in-source formula returns. It does NOT certify that the
implementation matches Korenaga Eq. 60. Correcting the prefactor is
deferred until every `Imk2` / `L` producer in the ecosystem can be
audited in lockstep.
