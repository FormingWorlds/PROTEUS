# Validation: `src/proteus/orbit/orbit.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.orbit` against a published source or analytical
limit. The marker is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_orbit_evolve.py::test_de_dt_matches_driscoll_barnes_2015_eq16` | Driscoll and Barnes (2015), Astrobiology 15, 739 (DOI 10.1089/ast.2015.1325; arXiv:1509.07452), Eq. 16 | n/a (closed form) | Pins the prefactor `21/2`, the `a^-6.5` exponent, the `R_pl^5` scaling, and the linear-in-`e` dependence of the tidal eccentricity-damping rate at unit-scale parameters. Also asserts sign and order-of-magnitude, with discrimination guards against `a^5` and `a^7` neighbouring exponents. |
| `tests/orbit/test_orbit_evolve.py::test_evolve_orbital_first_call_seeds_from_config_with_au_conversion` | AU constant from `scipy.constants` (IAU 2015 Resolution B2 nominal value, 1 AU = 1.495978707e11 m) | n/a (closed form) | Pins the AU-to-metres conversion that the orchestrator applies to `semimajoraxis` on the first call. A regression that dropped the AU factor would leave the semi-major axis at the config value in AU (e.g. 0.5) instead of the SI value (~7.48e10 m); the lower-bound `> 1e10 m` scale guard discriminates this. |

## Sign convention note

The paper uses `Im(k2) < 0` for tidal dissipation (Eq. 4 expresses
`-Im(k2)` as the positive dissipation efficiency). The PROTEUS source
takes positive `Imk2` from callers (`run_dummy_orbit`, `run_lovepy`),
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
