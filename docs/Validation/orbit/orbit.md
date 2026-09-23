# Validation: `src/proteus/orbit/orbit.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.orbit` against a published source or analytical
limit. The marker is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_orbit.py::test_sp0d_de_dt_matches_driscoll_barnes_2015_eq16` | Driscoll and Barnes (2015), Astrobiology 15, 739 (DOI 10.1089/ast.2015.1325; arXiv:1509.07452), Eq. 16 | n/a (closed form) | Pins the prefactor `21/2`, the `a^-6.5` exponent, the `R_pl^5` scaling, and the linear-in-`e` dependence of `sp0d`'s tidal eccentricity-damping rate at unit-scale parameters. Also asserts sign and order-of-magnitude, with discrimination guards against `a^5` and `a^7` neighbouring exponents. |
| `tests/orbit/test_orbit.py::test_sp1d_conserves_total_angular_momentum` | Angular-momentum conservation for an isolated two-body system with only internal tidal torques (analytical limit, not paper-specific) | n/a (closed form) | For `sp1d` (which, unlike `sp0d`, tracks the planet's spin), pins that planet-spin + orbital angular momentum is conserved to `rel=1e-6` across a real, non-trivial step (`e`: 0.3 to below 0.1). |
| `tests/orbit/test_orbit.py::test_sp1d_spin_am_gain_matches_orbital_am_loss` | Same conservation law as above, more targeted | n/a (closed form) | The planet's spin AM is ~1e-6 of the system total, so the raw-sum check above is nearly blind to a bug confined to `domega_dt`. This test instead pins `Delta(spin AM) == -Delta(orbital AM)` directly, at a tolerance where the two comparable-magnitude quantities discriminate a coefficient bug the raw-sum check misses. |

## Sign convention note

The paper uses `Im(k2) < 0` for tidal dissipation (Eq. 4 expresses
`-Im(k2)` as the positive dissipation efficiency). The PROTEUS source
takes positive `Imk2` from callers (`run_dummy_tides`, `run_lovepy`),
so the formula evaluated with positive `Imk2` returns positive `de/dt`
and expands the orbit instead of circularizing it. This is documented
in the source docstring as a known science item; the test pins the
algebra under the source convention and does NOT certify the convention
matches the paper. Any future sign correction must visit every `Imk2` producer in the
ecosystem so the change propagates consistently.

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
