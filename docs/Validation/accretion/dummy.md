# dummy.py Validation

## Source under test
`src/proteus/accretion/dummy.py` (the analytical accretion module: the
exponential growth law, the Noack & Lasbleis mass-radius scaling, the
gravitationally focused collision speed, and the momentum-conserving merger).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_dummy::test_collision_velocity_never_falls_below_the_mutual_escape_velocity` | Analytical limit: the two-body mutual escape speed, $v_\mathrm{esc} = \sqrt{2G(M_1+M_2)/(R_1+R_2)}$ | The collision speed for a circular encounter, where the approach velocity vanishes and the focused speed collapses exactly onto the mutual escape speed. Pinned to `rel=1e-12` against the value computed from the masses and radii the record itself carries, with a discrimination guard showing that dropping the factor of two moves the result by 29%. |
| `test_dummy::test_a_circular_encounter_leaves_the_orbit_untouched` | Analytical limit: a perfect merger of two bodies sharing one circular orbit | The merged orbit for $e = 0$, where both bodies have identical velocities and the mass-weighted mean is that velocity, so the semi-major-axis ratio is exactly one and the eccentricity exactly zero. This is the limit that a sign error or an inverted mass weighting cannot reproduce. |
| `test_dummy::test_merged_orbit_conserves_angular_momentum_of_the_merged_body` | Analytical identity: $h = \sqrt{\mu a (1 - e^2)} = r v_\theta$ | Internal consistency of the returned orbit. The semi-major axis and eccentricity are two numbers derived from one velocity vector, so they are only consistent if the angular momentum they imply equals the radius times the tangential velocity that produced them, pinned to `rel=1e-9`. |

## Coverage

The module derives an impact chain rather than integrating one, so what it
must certify is that the chain is physically admissible, not that it
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
`utils.structure_estimate`, whose own anchors are certified with the dummy
interior structure. Those laws are calibrated for planets, and their implied
bulk density falls without bound as the mass does, so an impactor well below an
Earth mass would otherwise come out less dense than its own uncompressed
minerals. The radius is capped so the bulk density never falls below the
zero-pressure value of an iron and silicate mixture at the same iron fraction,
which is the correct limit for a small body and is where the certification of
the scaling laws stops applying.
