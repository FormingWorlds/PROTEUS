# Validation: `src/proteus/orbit/parameterized.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
prescribed migration laws in `proteus.orbit.parameterized`. The module is
closed-form, so the sigmoid anchor is an analytical limit and the
high-eccentricity anchor is the published law of Postolec et al. (2026):
there is no external data set to compare against, and the contract is the
mathematics the laws are defined by.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_parameterized.py::test_sigmoid_matches_the_cubic_smoothstep_across_the_window` | Analytical limit: the cubic Hermite interpolant with zero slope at both ends, `S(u) = 3u^2 - 2u^3`, at `u = 1/4`, `1/2` and `3/4` | Standard | Pins `sigmoid_migration` across its migration window. Separates the cubic from a linear ramp and from the quintic smootherstep, and catches a swapped `sma_init`/`sma_final`. |
| `tests/orbit/test_parameterized.py::test_high_ecc_energy_rate_at_the_epoch_is_twice_the_energy_change_over_tau` | Analytical limit: `dE/dt` of `E = -G M m (1 - e_mig^2 x) / (2 a_f)` at `x = 1` equals `2 dE / tau` | Derived below | Pins `orbital_energy_rate` at the high-eccentricity epoch for the TOI-561 b setup (`dE = -2.86e35 J`, `-1.8e21 W` at `tau = 1e7` yr), with guards on the factor 2, the sign and the unit. Last compared 2026-10-02. |
| `tests/orbit/test_parameterized.py::test_high_ecc_circularises_as_a_pure_exponential_in_eccentricity` | Postolec et al. (2026), submitted to ApJ, arXiv:2609.03144 (doi:10.48550/arXiv.2609.03144) | Sect. 2.2, Eqs. 1-4 | Pins `e(t) = e_mig exp(-(t - t_mig) / tau)`, a form the source never evaluates, plus the one-tau e-folding. The semi-latus-rectum equality `a (1 - e^2) = a_f` is kept as a sanity check only: it is an identity of the implementation, not an emergent conservation law. Also asserts the pericentre `a (1 - e)` is not conserved. |

## Re-derivation notes

### Sigmoid migration

The law is a cubic S curve on a migration window of finite length,
clamped to its endpoints outside it:

```
u    = (t - t_mig) / tau,  clamped to [0, 1]
S(u) = 3u^2 - 2u^3
a(t) = a_0 + (a_f - a_0) S(u)
```

`S` is the unique cubic with `S(0) = 0`, `S(1) = 1` and
`S'(0) = S'(1) = 0`. Both endpoint conditions matter: the value
conditions make `a(t)` continuous at the window edges, and the slope
conditions make `da/dt` continuous there too. A discontinuous orbit
would hand the atmosphere a discontinuous instellation, and a kinked
one would hand it a discontinuous heating rate.

With the test endpoints `a_0 = 2.0 AU` and `a_f = 0.8 AU`,
`S(1/4) = 5/32` gives 1.8125 AU, `S(1/2) = 1/2` gives 1.4 AU and
`S(3/4) = 27/32` gives 0.9875 AU. The quarter point is the
discriminating one: a linear ramp puts it at 1.7 AU and the quintic
smootherstep `6u^5 - 15u^4 + 10u^3` at 1.8757 AU, both far outside the
tolerance. The cubic is symmetric about the window centre,
`S(u) + S(1 - u) = 1`, which the test also asserts and which holds
whatever the endpoints are.

The clamp is not cosmetic. Continued past its window the cubic runs
away: `S(2) = -4` puts the orbit at 6.8 AU and `S(-1) = 5` puts it at
-4.0 AU, so an unclamped evaluation at any time outside the window
returns an unphysical orbit.

Here `tau` is the length of the migration window, not an exponential
decay constant. It carries that second meaning in the high-eccentricity
law below, so the two regimes read the same configuration key
differently.

The `rel=1e-12` tolerance is machine precision: the expression is
closed-form algebra with no solver or lookup in the path.

### High-eccentricity migration

The law (Postolec et al. 2026, Sect. 2.2, Eqs. 1-4) is

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

Substituting `a(t)` into `1 - e^2 = a_f / a` collapses the eccentricity
to a pure exponential with half the decay rate of the semi-major axis:

```
e(t) = e_mig exp(-(t - t_mig) / tau)
```

so `e` e-folds in exactly one `tau`. That is the form the test pins,
because the source never evaluates it: it recovers `e` from `a`, so an
error in the time dependence cannot show up in the semi-latus rectum.
Checked at the half decay point, `sqrt(0.6) exp(-0.5 ln 2) = 0.547723`,
which is `sqrt(0.3)`.

The same route costs precision at late times. Recovering `e` as
`sqrt(1 - a_f / a)` subtracts two numbers that approach each other, and
the quantity under the root is `e^2`, so it drops below double precision
once `e` falls near 1e-8 and the returned eccentricity floors to exactly
zero. The exponential is therefore pinned over the first five migration
widths, and the floor is asserted separately as a boundedness property:
the value reaches zero without passing through a negative intermediate,
which would have produced a nan.

The pericentre `a (1 - e)` is **not** conserved: with the test endpoints it
rises from 0.45 AU to 0.80 AU over the circularisation. This distinguishes
the law from pericentre-conserving tidal circularisation, where the tide
acts at pericentre and `a_final` tends to `a_0 (1 - e_0)`. Anyone comparing
model tracks against a tidal-circularisation result should start from that
difference.

The `rtol=1e-10` tolerance on the invariant absorbs the round-off of the
`sqrt` round trip through `e` and back.

### Orbital energy rate

The orbital energy is `E = -G M_star M_planet / (2 a)`, so at fixed masses
`dE/dt = G M_star M_planet (da/dt) / (2 a^2)`. On the high-eccentricity
track `1/a = (1 - e_mig^2 x) / a_f` with `x = exp(-2 (t - t_mig) / tau)`,
which gives

```
dE/dt = -G M_star M_planet e_mig^2 x / (a_f tau)
```

At the epoch `x = 1` and `e_mig^2 = 1 - a_f / a_0`, so the rate is
`2 dE / tau` with `dE = E(a_f) - E(a_0) = -(G M_star M_planet / 2)(1/a_f - 1/a_0)`
the energy change of the whole circularisation, negative for inward migration. The rate depends on time through `x`
alone, so one `tau` after the epoch it has fallen by exactly `e^-2`; a
decay written with `exp(-(t - t_mig) / tau)` would fall by `e^-1`. The
closure test integrates the rate with `scipy.integrate.quad` and recovers
`E(a_end) - E(a_0)` to `rel=1e-8` for the high-eccentricity track and for
the sigmoid inward and outward, which a missing factor of 1/2 or a rate
left per year instead of per second would miss by a factor of 2 or 3.2e7.

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
| `sigmoid` | 3.6e-11 | 0 |
| `high_ecc` | 4.3e-11 | 2.8e-09 |

The figure and the run settings are in
[Star-planet models](../../Explanations/orbit.md#visualizing-the-four-parameterized-regimes).
