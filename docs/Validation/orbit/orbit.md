# Validation: `src/proteus/orbit/orbit.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.orbit` against a published source or analytical
limit. The marker is defined in
[`.github/.claude/rules/proteus-tests.md`](../../../.github/.claude/rules/proteus-tests.md)
section 3.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_orbit_evolve.py::test_de_dt_matches_driscoll_barnes_2015_eq16` | Driscoll & Barnes (2015), ApJ 815, 1, Eq. 16 | n/a (closed form) | Pins the prefactor `21/2`, the `a^-6.5` exponent, the `R_pl^5` scaling, and the linear-in-`e` dependence of the tidal eccentricity-damping rate at unit-scale parameters. Also asserts sign and order-of-magnitude. |

## Re-derivation note

Driscoll & Barnes (2015) Eq. 16 reads

```
de/dt = (21/2) * Imk2 * Mst^1.5 * G^0.5 * Rpl^5 / (Mpl * a^6.5) * e
```

With dimensionless unit inputs `Imk2 = Mst = G = Rpl = Mpl = 1`, `a = 2`,
`e = 0.5`, the closed-form result is

```
de/dt = (21/2) * 0.5 / 2^6.5 = 5.7996e-2
```

A regression to `a^5` would shift the answer to 0.164; the test asserts the
absolute difference exceeds 5e-2, well above any plausible float tolerance.
