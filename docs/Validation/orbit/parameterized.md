# Validation: `src/proteus/orbit/parameterized.py`

## What this page checks

This page checks the prescribed migration laws of `proteus.orbit.parameterized`, which move the orbit of the planet along a fixed law in time. The sigmoid law moves the semi-major axis along a smooth S curve; the high-eccentricity law circularises an eccentric orbit. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the orbit, and with it the stellar flux at the planet, follows the wrong path. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

The `@pytest.mark.reference_pinned` tests anchor the
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

## Sigmoid migration

**Why this check exists.** The sigmoid law must follow a cubic S curve with zero slope at both ends of its window. A linear ramp, a different smooth curve, or interchanged start and end distances gives a wrong orbit without an error message.

**What it tests.** `sigmoid_migration` with a start distance of 2.0 AU and an end distance of 0.8 AU must give 1.8125, 1.4 and 0.9875 AU at a quarter, a half and three quarters of the window. The test checks that a linear ramp (1.7 AU at the quarter point) and the quintic smootherstep (1.8757 AU) are clearly away, that the curve is symmetric about the centre of the window, and that the quarter point lies between the 2 distances.

**How we know the right answer.** The cubic $S(u) = 3u^2 - 2u^3$ is the unique cubic with $S(0) = 0$, $S(1) = 1$ and zero slope at both ends (see the derivation below).

**Why it matters.** The sigmoid law gives the orbit of a run that uses it, and so the stellar flux at the planet over time.

![Semi-major axis of sigmoid migration against the window fraction](../../assets/validation/orbit_sigmoid_migration.svg#only-light){ width="80%" }
![Semi-major axis of sigmoid migration against the window fraction](../../assets/validation/orbit_sigmoid_migration_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the fraction $u$ of the migration window and the y-axis the semi-major axis, from 2.0 to 0.8 AU. The line is `sigmoid_migration` (context); the dashed line is a linear ramp and the dotted line the quintic smootherstep. The test rejects both of these forms. The hollow squares are the cubic $3u^2 - 2u^3$ at $u$ = 1/4, 1/2 and 3/4. The filled circles are the asserted code values there: 1.8125, 1.4 and 0.9875 AU. A pass means that the circles sit in the squares, away from the dashed and dotted lines at $u$ = 1/4.

??? info "Test details"
    - Reference: the cubic Hermite interpolant with zero slope at both ends, `S(u) = 3u^2 - 2u^3` (analytical limit).
    - Tolerance: relative 1e-12 at u = 1/4, 1/2 and 3/4; the linear ramp more than 0.05 AU and the quintic more than 0.02 AU away.
    - Test id: `tests/orbit/test_parameterized.py::test_sigmoid_matches_the_cubic_smoothstep_across_the_window`
    - Command: `pytest -m reference_pinned tests/orbit/test_parameterized.py --validation-plots`

## Orbital energy rate at the start of circularisation

**Why this check exists.** The rate of orbital energy change follows from the migration law. A missing factor of 2, a wrong sign or a wrong time unit gives a wrong energy rate without an error message.

**What it tests.** For the TOI-561 b setup, the test checks that `orbital_energy_rate` at the start of the high-eccentricity migration equals $2\, \Delta E / \tau$, with $\Delta E$ the energy change of the whole circularisation (about -2.86e35 J) and $\tau$ = 1e7 yr. It also checks that the rate without the factor 2 is clearly away, that the rate is negative (the orbit loses energy as it shrinks), and that its size lies between 1e21 and 3e21 W.

**How we know the right answer.** The closed form of the energy rate at the epoch (see "Orbital energy rate" below).

**Why it matters.** The energy rate is the power that the orbit loses as it circularises.

![Orbital energy rate of high-eccentricity migration against time for TOI-561 b](../../assets/validation/orbit_high_ecc_energy_rate.svg#only-light){ width="80%" }
![Orbital energy rate of high-eccentricity migration against time for TOI-561 b](../../assets/validation/orbit_high_ecc_energy_rate_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the time since the migration start $t_\mathrm{mig}$ in units of the decay time $\tau$, for the TOI-561 b setup. The y-axis is the orbital energy rate in $10^{21}$ W, and the line is `orbital_energy_rate` (context). $\Delta E$ is the orbital energy change of the whole migration. The filled circle is the asserted rate at $t_\mathrm{mig}$, and the hollow square is $2\,\Delta E / \tau$. The dashed line is $\Delta E / \tau$, the form without the factor 2 that the test rejects. A pass means that the circle sits in the square, below the dashed line, between $-1$ and $-3 \times 10^{21}$ W.

??? info "Test details"
    - Reference: analytical limit, `dE/dt` of `E = -G M m (1 - e_mig^2 x) / (2 a_f)` at `x = 1` equals `2 dE / tau`.
    - Tolerance: relative 1e-10 against the closed form; relative 0.01 on dE = -2.86e35 J; size 1e21 to 3e21 W.
    - Test id: `tests/orbit/test_parameterized.py::test_high_ecc_energy_rate_at_the_epoch_is_twice_the_energy_change_over_tau`
    - Command: `pytest -m reference_pinned tests/orbit/test_parameterized.py --validation-plots`

## Eccentricity decay of high-eccentricity migration

**Why this check exists.** The eccentricity must decay as a pure exponential with time constant $\tau$. A wrong exponent or a wrong time constant changes the circularisation without an error message.

**What it tests.** The test compares the eccentricity of `high_eccentricity_migration` at 500 times with $e(t) = e_\mathrm{mig} \exp(-(t - t_\mathrm{mig}) / \tau)$, a form the source never evaluates. After one $\tau$ the eccentricity must have fallen by $e^{-1}$, not $e^{-1/2}$. It also checks that the eccentricity stays finite and not negative and ends at exactly zero, that the semi-latus rectum $a (1 - e^2)$ equals the final distance (an identity of the implementation, a sanity check only), and that the pericentre $a (1 - e)$ is not conserved.

**How we know the right answer.** Postolec et al. (2026), Sect. 2.2, Eqs. 1-4 (see the derivation below).

**Why it matters.** The eccentricity and the semi-major axis of a run with this law come from these equations.

![Eccentricity decay of high-eccentricity migration](../../assets/validation/orbit_high_ecc_eccentricity.svg#only-light){ width="80%" }
![Eccentricity decay of high-eccentricity migration](../../assets/validation/orbit_high_ecc_eccentricity_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is $x = (t - t_\mathrm{mig})/\tau$, the time since the migration start $t_\mathrm{mig}$ in units of the decay time $\tau$. The y-axis is the eccentricity, for $a_0 = 2.0$ AU, $a_f = 0.8$ AU, $e_\mathrm{mig} = \sqrt{0.6}$, $t_\mathrm{mig} = 10^5$ yr and $\tau = 10^4$ yr. Here $a_0$ and $a_f$ are the initial and final semi-major axes, and $e_\mathrm{mig}$ the eccentricity at the start. The markers are the asserted eccentricities of `high_eccentricity_migration`, every 25th of the 500 tested points over 5 $\tau$. The solid line is the closed form $e_\mathrm{mig} \exp(-x)$, and the dashed curve the law with half the decay rate, which the test rejects. A pass means that the markers lie on the solid line.

??? info "Test details"
    - Reference: Postolec et al. (2026), submitted to ApJ, arXiv:2609.03144, Sect. 2.2, Eqs. 1-4.
    - Tolerance: relative 1e-9 on all 500 points; relative 1e-10 on the one-tau e-folding and on the semi-latus rectum.
    - Test id: `tests/orbit/test_parameterized.py::test_high_ecc_circularises_as_a_pure_exponential_in_eccentricity`
    - Command: `pytest -m reference_pinned tests/orbit/test_parameterized.py --validation-plots`

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
decay constant. It has that second meaning in the high-eccentricity
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

## Check against complete dummy runs

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
