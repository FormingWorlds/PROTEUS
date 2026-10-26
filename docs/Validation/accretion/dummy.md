# dummy.py Validation

## What this page checks

This page checks the analytical accretion module, `src/proteus/accretion/dummy.py`. It builds a history of giant impacts: an exponential growth law sets the impactor masses, a mass-radius scaling sets their radii, and each impact merges 2 bodies. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, an impact has the wrong speed or leaves the planet on the wrong orbit. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

## Source under test
`src/proteus/accretion/dummy.py` (the analytical accretion module: the
exponential growth law, the Noack & Lasbleis mass-radius scaling, the
gravitationally focused collision speed, and the momentum-conserving merger).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_dummy::test_collision_velocity_never_falls_below_the_mutual_escape_velocity` | Analytical limit: the two-body mutual escape speed, $v_\mathrm{esc} = \sqrt{2G(M_1+M_2)/(R_1+R_2)}$ | The collision speed for a circular encounter, where the approach velocity vanishes and the focused speed collapses exactly onto the mutual escape speed. Pinned to `rel=1e-12` against the value computed from the masses and radii the record itself holds, with a discrimination guard showing that dropping the factor of two moves the result by 29%. |
| `test_dummy::test_a_circular_encounter_leaves_the_orbit_untouched` | Analytical limit: a perfect merger of two bodies sharing one circular orbit | The merged orbit for $e = 0$, where both bodies have identical velocities and the mass-weighted mean is that velocity, so the semi-major-axis ratio is exactly one and the eccentricity exactly zero. This is the limit that a sign error cannot reproduce. |
| `test_dummy::test_merged_orbit_conserves_angular_momentum_of_the_merged_body` | Analytical identity: $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ | Internal consistency of the returned orbit. The semi-major axis and eccentricity are two numbers derived from one velocity vector, so they are only consistent if the angular momentum they imply equals the radius times the tangential velocity that produced them, pinned to `rel=1e-9`. |

## Collision speed and the mutual escape speed

**Why this check exists.** Gravitational focusing makes 2 bodies collide at least at their mutual escape speed. A missing factor of 2 under the square root, or a wrong sum of masses or radii, gives a collision speed below that floor without an error message.

**What it tests.** For a circular encounter the approach speed is zero, so the focused collision speed must equal the mutual escape speed $v_\mathrm{esc} = \sqrt{2G(M_1+M_2)/(R_1+R_2)}$, with $M_1, M_2$ the masses and $R_1, R_2$ the radii of the 2 bodies. The test computes $v_\mathrm{esc}$ from the masses and radii that the impact record holds. It also checks that the collision speed is never below the escape speed.

**How we know the right answer.** The two-body escape speed is an analytical limit of the focused collision speed.

**Why it matters.** `accretion/atmloss.py` takes the collision speed of each impact (`v_impact`) for the atmospheric loss.

**Known limit.** The code computes the collision speed as the root of the sum of the squared approach and escape speeds, so it can never fall below the escape speed; that check, and the 29 % change from dropping the factor of 2, cannot fail through a code fault. Only the hand-computed escape speed of the first event tests the formula.

??? info "Test details"
    - Reference: two-body mutual escape speed (analytical limit).
    - Tolerance: relative 1e-12; the factor-of-2 error must miss by more than 20 %.
    - Test id: `tests/accretion/test_dummy.py::test_collision_velocity_never_falls_below_the_mutual_escape_velocity`
    - Command: `pytest tests/accretion/test_dummy.py::test_collision_velocity_never_falls_below_the_mutual_escape_velocity`

## A circular merger keeps the orbit

**Why this check exists.** A merger sets the new orbit from the mass-weighted velocity of the 2 bodies. A sign error gives a wrong orbit without an error message.

**What it tests.** When 2 bodies share one circular orbit ($e = 0$) they have the same velocity, so the mass-weighted mean is that velocity. The merged orbit must keep its semi-major axis (ratio 1) and stay circular ($e = 0$).

**How we know the right answer.** A perfect merger of 2 co-orbital bodies on a circular orbit is an analytical limit.

**Why it matters.** `accretion/wrapper.py` adds the eccentricity change of each impact to the orbit of the planet, and applies the semi-major-axis ratio unless `orbit.instellation_method = 'inst'`.

**Known limit.** At $e = 0$ both bodies have the same velocity, so an inverted mass weighting gives the same orbit and passes this case; the angular-momentum case below catches it.

??? info "Test details"
    - Reference: perfect merger of 2 bodies on one circular orbit (analytical limit).
    - Tolerance: relative 1e-12 on the semi-major-axis ratio; absolute 1e-12 on the eccentricity.
    - Test id: `tests/accretion/test_dummy.py::test_a_circular_encounter_leaves_the_orbit_untouched`
    - Command: `pytest tests/accretion/test_dummy.py::test_a_circular_encounter_leaves_the_orbit_untouched`

## Angular momentum of the merged orbit

**Why this check exists.** The semi-major axis and the eccentricity are 2 numbers derived from one velocity vector. If one of them is computed wrongly, or the mass weighting is inverted, the orbit does not match the velocity that produced it, and nothing reports it.

**What it tests.** For one merger of a 1.0 and a 0.4 Earth-mass body at $e = 0.2$, the specific angular momentum of the returned orbit, $h = \sqrt{\mu a (1 - e^2)}$ with $\mu = G M_\star$, must equal the radius times the tangential velocity, $r v_\theta$.

**How we know the right answer.** For a Keplerian orbit, $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ is an identity.

**Why it matters.** It checks that a merged orbit is consistent with the velocity that produced it.

**Known limit.** The eccentricity is derived from the semi-major axis, so a wrong semi-major axis gives a consistent orbit and passes this case.

??? info "Test details"
    - Reference: Keplerian identity $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ (analytical identity).
    - Tolerance: relative 1e-9.
    - Test id: `tests/accretion/test_dummy.py::test_merged_orbit_conserves_angular_momentum_of_the_merged_body`
    - Command: `pytest tests/accretion/test_dummy.py::test_merged_orbit_conserves_angular_momentum_of_the_merged_body`

## Coverage

The module derives an impact chain rather than integrating one, so what it
must show is that the chain is physically admissible, not that it
reproduces any particular system. Mass closes at every merger and over the
whole timeline; the collision speed satisfies the floor the timeline validator
enforces, with equality in the circular limit; and the merged orbit is bound
and internally consistent in angular momentum. The merged orbit is also never
wider than the one the two bodies shared, but that follows from the co-orbital
geometry the module assumes rather than from mergers in general: bodies meeting
from different semi-major axes can merge onto a wider orbit.

The growth law itself is a modelling choice rather than a measured quantity,
so it is pinned by its own structure: consecutive impactor masses differ by
exactly $\exp(-\Delta t / \tau)$, which fixes both the sign and the presence of
the exponential, and the delivered mass sums to the configured budget exactly,
which fixes the renormalisation.

Radii come from the Noack & Lasbleis (2020) scaling laws through
`utils.structure_estimate`, whose own anchors are checked with the dummy
interior structure. Those laws are calibrated for planets, and their implied
bulk density falls without bound as the mass does, so an impactor well below an
Earth mass would otherwise come out less dense than its own uncompressed
minerals. The radius is capped so the bulk density never falls below the
zero-pressure value of an iron and silicate mixture at the same iron fraction,
which is the correct limit for a small body and is where the check of
the scaling laws stops applying.
