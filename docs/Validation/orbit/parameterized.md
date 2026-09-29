# Validation: `src/proteus/orbit/parameterized.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
prescribed migration laws in `proteus.orbit.parameterized`. The module is
closed-form, so both anchors are analytical limits rather than published
benchmarks: there is no external data set to compare against, and the
contract is the mathematics the laws are defined by.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_parameterized.py::test_sigmoid_matches_the_logistic_centre_and_quarter_point` | Analytical limit: the logistic function `s(x) = 1 / (1 + exp(x))` at `x = 0` and `x = ln(3)` | Standard | Pins `sigmoid_migration` at its centre (`s = 1/2`) and quarter point (`s = 1/4`). Fixes the migration width `tau_mig` in the exponent and catches a swapped `sma_init`/`sma_final`. |
| `tests/orbit/test_parameterized.py::test_high_ecc_conserves_orbital_angular_momentum` | Analytical limit: specific orbital angular momentum of a two-body orbit, `h = sqrt(G M a (1 - e^2))` | Standard | Pins `a(t)` against the closed form, which is the anchor on the law itself. The accompanying semi-latus-rectum equality `a (1 - e^2) = a_f` checks that `e(t)` is the exact inverse of `a(t)`; it is an identity of the implementation, not an emergent conservation law. Also asserts the pericentre `a (1 - e)` is not conserved. |

## Re-derivation notes

### Sigmoid migration

The law is

```
a(t) = (a_0 - a_f) / (1 + exp((t - t_mig) / tau)) + a_f
```

At `t = t_mig` the exponential is 1, so the logistic is `1/2` and
`a = a_f + (a_0 - a_f) / 2`, the arithmetic mean of the endpoints. At
`t = t_mig + tau ln(3)` the exponential is 3, so the logistic is `1/4`
and `a = a_f + (a_0 - a_f) / 4`.

With the test endpoints `a_0 = 2.0 AU` and `a_f = 0.8 AU` these are
1.4 AU and 1.1 AU. Halving `tau` in the exponent moves the second point
to `0.8 + 1.2 / 10 = 0.92 AU`, which is what makes it discriminating.
The `rel=1e-12` tolerance is machine precision: the expression is closed-form
algebra with no solver or lookup in the path.

### High-eccentricity migration

The law is

```
e_mig = sqrt(1 - a_f / a_0)
a(t)  = a_f / (1 - e_mig^2 exp(-2 (t - t_mig) / tau))
e(t)  = sqrt(1 - a_f / a(t))
```

Rearranging the eccentricity relation gives `1 - e^2 = a_f / a`, hence

```
a (1 - e^2) = a_f
```

for every `t >= t_mig`. Since the specific orbital angular momentum is
`h = sqrt(G M a (1 - e^2))`, holding `a (1 - e^2)` fixed is exactly
conservation of `h`.

Read carefully, though, that equality is weaker than it looks as a test.
The source computes `e` **from** `a` via `e = sqrt(1 - a_f/a)`, so
`a (1 - e^2) = a_f` follows algebraically for *any* `a(t)` whatsoever,
including a wrong one. It verifies that `e(t)` inverts `a(t)` exactly; it
cannot see an error in the time dependence. The test therefore also pins
`a(t)` directly against the closed form above, written out independently,
and that pin is the real anchor. A reader treating the semi-latus-rectum
equality as an emergent conservation law would overstate what is checked.

At `t = t_mig` the decay factor is 1, so
`a = a_f / (1 - e_mig^2) = a_0` and `e = e_mig`: the orbit is still at its
starting semi-major axis with the eccentricity excited to the value that
places it on the final angular momentum.

The pericentre `a (1 - e)` is **not** conserved: with the test endpoints it
rises from 0.45 AU to 0.80 AU over the circularisation. This distinguishes
the law from pericentre-conserving tidal circularisation, where the tide
acts at pericentre and `a_final` tends to `a_0 (1 - e_0)`. Anyone comparing
model tracks against a tidal-circularisation result should start from that
difference.

The `rtol=1e-10` tolerance on the invariant absorbs the round-off of the
`sqrt` round trip through `e` and back.

## End-to-end check against dummy runs

The pinned tests above exercise the three laws as pure functions. A
complementary check runs each regime as a dummy PROTEUS simulation and
compares the recorded `semimajorax` and `eccentricity` against the same
closed forms, so the config plumbing, the AU-to-metre conversion and the
dispatch in `evolve_orbit_star` are covered as well as the algebra.

| Regime | max abs da \[au\] | max abs de |
|---|---|---|
| `none` | 0 | 0 |
| `instant` | 0 | 0 |
| `sigmoid` | 3.8e-11 | 0 |
| `high_ecc` | 4.3e-11 | 2.8e-09 |

The figure and the run settings are in
[Star-planet models](../../Explanations/orbit.md#visualizing-the-four-parameterized-regimes).
