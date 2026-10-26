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
| `test_dummy::test_collision_velocity_never_falls_below_the_mutual_escape_velocity` | Analytical limit: the two-body mutual escape speed, $v_\mathrm{esc} = \sqrt{2G(M_1+M_2)/(R_1+R_2)}$ | The collision speed for a circular encounter, where the approach velocity vanishes and the focused speed collapses exactly onto the mutual escape speed. Pinned to `rel=1e-12` against the value computed from the masses and radii the record itself holds, with a check that dropping the factor of two moves the result by 29% (arithmetic on the test's own numbers). |
| `test_dummy::test_a_circular_encounter_leaves_the_orbit_untouched` | Analytical limit: a perfect merger of two bodies sharing one circular orbit | The merged orbit for $e = 0$, where both bodies have identical velocities and the mass-weighted mean is that velocity, so the semi-major-axis ratio is exactly one and the eccentricity exactly zero. This is the limit that a sign error cannot reproduce. |
| `test_dummy::test_merged_orbit_conserves_angular_momentum_of_the_merged_body` | Analytical identity: $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ | Internal consistency of the returned orbit. The semi-major axis and eccentricity are two numbers derived from one velocity vector, so they are only consistent if the angular momentum they imply equals the radius times the tangential velocity that produced them, pinned to `rel=1e-9`. |

## Collision speed and the mutual escape speed

**Why this check exists.** Gravitational focusing makes 2 bodies collide at least at their mutual escape speed. A missing factor of 2 under the square root, or a wrong sum of masses or radii, gives a collision speed below that floor without an error message.

**What it tests.** For a circular encounter the approach speed is zero, so the focused collision speed must equal the mutual escape speed $v_\mathrm{esc} = \sqrt{2G(M_1+M_2)/(R_1+R_2)}$, with $M_1, M_2$ the masses and $R_1, R_2$ the radii of the 2 bodies. The test computes $v_\mathrm{esc}$ from the masses and radii that the impact record holds. It also checks that the collision speed is never below the escape speed.

**How we know the right answer.** The two-body escape speed is an analytical limit of the focused collision speed.

**Why it matters.** `accretion/atmloss.py` takes the collision speed of each impact (`v_impact`) for the atmospheric loss.

**Known limit.** The code computes the collision speed as the root of the sum of the squared approach and escape speeds, so it can never fall below the escape speed; that check, and the 29 % change from dropping the factor of 2, cannot fail through a code fault. Only the hand-computed escape speed of the first event tests the formula.

![Collision speed against mutual escape speed for the impacts of 2 timelines](../../assets/validation/accretion_dummy_collision_speed.svg#only-light){ width="80%" }
![Collision speed against mutual escape speed for the impacts of 2 timelines](../../assets/validation/accretion_dummy_collision_speed_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the mutual escape speed $v_\mathrm{esc}$ of each impacting pair, and the y-axis its collision speed $v_\mathrm{impact}$. The circles are the asserted impacts at encounter eccentricity 0.05; they must lie above the line $v_\mathrm{impact} = v_\mathrm{esc}$. The hollow squares are the asserted impacts at eccentricity 0, which must lie on the line. The star is the escape speed of the first pair, computed by hand from its masses and radii, against the code value. The dashed line is the escape speed without the factor 2, which the test rejects. A pass means that the squares and the star lie on the line, and the circles just above it.

??? info "Test details"
    - Reference: two-body mutual escape speed (analytical limit).
    - Tolerance: relative 1e-12; the factor-of-2 error must miss by more than 20 %.
    - Test id: `tests/accretion/test_dummy.py::test_collision_velocity_never_falls_below_the_mutual_escape_velocity`
    - Command: `pytest -m reference_pinned tests/accretion/test_dummy.py --validation-plots`

## A circular merger keeps the orbit

**Why this check exists.** A merger sets the new orbit from the mass-weighted velocity of the 2 bodies. A sign error in the tangential velocity gives a wrong orbit without an error message.

**What it tests.** When 2 bodies share one circular orbit ($e = 0$) they have the same velocity, so the mass-weighted mean is that velocity. The merged orbit must keep its semi-major axis (ratio 1) and stay circular ($e = 0$).

**How we know the right answer.** A perfect merger of 2 co-orbital bodies on a circular orbit is an analytical limit.

**Why it matters.** `accretion/wrapper.py` applies the eccentricity change and the semi-major-axis ratio of each impact to the orbit of the planet.

**Known limit.** At $e = 0$ both bodies have the same velocity, so an inverted mass weighting, or a sign error in the radial velocity of the impactor, gives the same orbit and passes this case; the angular-momentum case below catches it.

![Semi-major axis ratio and eccentricity of merged orbits against the mass ratio of each impact](../../assets/validation/accretion_dummy_circular_merger.svg#only-light){ width="80%" }
![Semi-major axis ratio and eccentricity of merged orbits against the mass ratio of each impact](../../assets/validation/accretion_dummy_circular_merger_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the mass ratio of impactor and target for each impact of a circular timeline at 1 AU (astronomical unit). The y-axis shows 2 dimensionless quantities of the merged orbit. The circles are the asserted semi-major axis ratios $a/a_0$, and the hollow squares the asserted eccentricities $e$. The solid line is $a/a_0$ = 1 and the dashed line $e$ = 0. A pass means that the circles lie on the solid line and the squares on the dashed line.

??? info "Test details"
    - Reference: perfect merger of 2 bodies on one circular orbit (analytical limit).
    - Tolerance: relative 1e-12 on the semi-major-axis ratio; absolute 1e-12 on the eccentricity.
    - Test id: `tests/accretion/test_dummy.py::test_a_circular_encounter_leaves_the_orbit_untouched`
    - Command: `pytest -m reference_pinned tests/accretion/test_dummy.py --validation-plots`

## Angular momentum of the merged orbit

**Why this check exists.** The semi-major axis and the eccentricity are 2 numbers derived from one velocity vector. If the eccentricity is computed wrongly, or the mass weighting is inverted, the orbit does not match the velocity that produced it, and nothing reports it.

**What it tests.** For one merger of a 1.0 and a 0.4 Earth-mass body at $e = 0.2$, the specific angular momentum of the returned orbit, $h = \sqrt{\mu a (1 - e^2)}$ with $\mu = G M_\star$, must equal the radius times the tangential velocity, $r v_\theta$.

**How we know the right answer.** For a Keplerian orbit, $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ is an identity.

**Why it matters.** It checks that a merged orbit is consistent with the velocity that produced it.

**Known limit.** The eccentricity is derived from the semi-major axis, so a wrong semi-major axis gives a consistent orbit and passes this case.

![Specific angular momentum of the merged orbit against encounter eccentricity](../../assets/validation/accretion_dummy_angular_momentum.svg#only-light){ width="80%" }
![Specific angular momentum of the merged orbit against encounter eccentricity](../../assets/validation/accretion_dummy_angular_momentum_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the encounter eccentricity $e$, for a 1.0 and a 0.4 Earth-mass body at 1 AU around a solar-mass star. The y-axis is the specific angular momentum $h$ (per unit mass) over $a\,v_\mathrm{K}$, with $v_\mathrm{K}$ the circular orbital speed. The line is $\sqrt{\mu a (1 - e^2)}$ of the orbit that `_merged_orbit` returns, with $\mu$ the gravitational parameter of the star. The squares are $r\,v_\theta$, the radius times the tangential speed of the merged body (context). The filled circle is the asserted case at $e$ = 0.2. A pass means that the squares and the circle lie on the line.

??? info "Test details"
    - Reference: Keplerian identity $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ (analytical identity).
    - Tolerance: relative 1e-9.
    - Test id: `tests/accretion/test_dummy.py::test_merged_orbit_conserves_angular_momentum_of_the_merged_body`
    - Command: `pytest -m reference_pinned tests/accretion/test_dummy.py --validation-plots`

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
