"""Tests for the analytical accretion module.

This file targets accretion/dummy.py (get_timeline and its helpers), which
builds a giant-impact history from scaling laws instead of integrating a
system of embryos. What it must guarantee is that the chain it produces is
physically self-consistent: mass closes across every merger and over the whole
timeline, the collision velocity never falls below the pair's mutual escape
velocity, the merged orbit follows from conserving momentum through the
collision, and the whole thing satisfies the same validator a model-derived or
file-read timeline must satisfy.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from proteus.accretion.dummy import _merged_orbit, get_timeline
from proteus.utils.constants import AU, M_earth, M_sun, const_G

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _config(
    mass_accreted=1.0,
    num_impacts=4,
    timescale=2.0e6,
    time_last=8.0e6,
    eccentricity=0.05,
    impact_parameter=0.5,
    mass_tot=1.0,
    semimajoraxis=1.0,
    orbit_eccentricity=0.0,
    star_mass=1.0,
    time_offset=0.0,
):
    """Build the minimal config shape the analytical module reads."""
    return SimpleNamespace(
        accretion=SimpleNamespace(
            module='dummy',
            time_offset=time_offset,
            dummy=SimpleNamespace(
                mass_accreted=mass_accreted,
                num_impacts=num_impacts,
                timescale=timescale,
                time_last=time_last,
                eccentricity=eccentricity,
                impact_parameter=impact_parameter,
            ),
        ),
        planet=SimpleNamespace(mass_tot=mass_tot),
        orbit=SimpleNamespace(semimajoraxis=semimajoraxis, eccentricity=orbit_eccentricity),
        star=SimpleNamespace(mass=star_mass),
        interior_struct=SimpleNamespace(core_frac=0.55, core_frac_mode='radius'),
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_timeline_conserves_mass_across_every_merger():
    """Mass closes per impact and over the whole timeline.

    Two separate closures matter and can fail independently. Each merger is a
    perfect merger, so the merged mass must equal the sum of the two bodies,
    and the next impact must inherit exactly that mass; a chain that re-derived
    the target mass from the growth law instead of from the previous merger
    would drift. Across the whole timeline the delivered mass must equal the
    configured budget exactly, because the increments are renormalised: a
    missing renormalisation would land short of it by the exponential tail.
    """
    events = get_timeline(_config(mass_accreted=1.0, mass_tot=1.0))

    for event in events:
        assert event.M_merged_after == pytest.approx(
            event.M_target_before + event.M_impactor, rel=1e-12
        )

    for previous, current in zip(events, events[1:]):
        assert current.M_target_before == pytest.approx(previous.M_merged_after, rel=1e-12)

    delivered = sum(event.M_impactor for event in events)
    assert delivered == pytest.approx(1.0 * M_earth, rel=1e-12)
    assert events[-1].M_merged_after == pytest.approx(2.0 * M_earth, rel=1e-12)

    # Discrimination: without the renormalisation the law only reaches
    # 1 - exp(-t_last/tau) of the budget, which for these settings is 98.17%.
    # That shortfall is 1.8e-2 relative, four orders above the tolerance above.
    unnormalised = 1.0 - math.exp(-8.0e6 / 2.0e6)
    assert abs(unnormalised - 1.0) > 1.0e-2


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impactor_masses_decay_and_the_first_impact_is_the_largest():
    """A decaying accretion rate delivers its mass front-loaded.

    The growth law is an exponential approach to an asymptote, so the mass
    accreted per unit time falls monotonically. With impacts spaced evenly in
    time the increments must therefore decrease, which is the ordering that
    distinguishes this law from a uniform delivery. The ratio of consecutive
    increments is exp(-dt/tau), a value the test pins directly, so an
    implementation that dropped the exponential or inverted its sign fails.
    """
    events = get_timeline(_config(num_impacts=4, timescale=2.0e6, time_last=8.0e6))

    masses = [event.M_impactor for event in events]
    assert masses == sorted(masses, reverse=True)

    # Consecutive weights differ by exactly exp(-dt/tau) with dt = 2 Myr and
    # tau = 2 Myr, so the ratio is 1/e. Renormalisation is a common factor and
    # cancels out of the ratio.
    expected_ratio = math.exp(-1.0)
    for previous, current in zip(masses, masses[1:]):
        assert current / previous == pytest.approx(expected_ratio, rel=1e-9)

    # Discrimination: a uniform delivery would give a ratio of 1, and a sign
    # error in the exponent would give e. Both are far outside the tolerance.
    assert abs(expected_ratio - 1.0) > 0.5


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_collision_velocity_never_falls_below_the_mutual_escape_velocity():
    """v_impact = sqrt(v_encounter^2 + v_esc^2) holds, including at e = 0.

    The timeline validator rejects any impact whose collision velocity is below
    the pair's mutual escape velocity, since that is unreachable for any
    approach. The equality case is the discriminating one: on a circular
    encounter the two bodies meet with no relative velocity, so the collision
    velocity must equal the escape velocity exactly rather than fall below it,
    which is where a formula that forgot the gravitational focusing would fail.
    """
    events = get_timeline(_config(eccentricity=0.05))
    for event in events:
        assert event.v_impact >= event.v_esc
        # The focused speed is the quadrature sum, so the encounter term is
        # recoverable and must be positive for a non-circular encounter.
        v_encounter_sq = event.v_impact**2 - event.v_esc**2
        assert v_encounter_sq > 0.0

    circular = get_timeline(_config(eccentricity=0.0))
    for event in circular:
        assert event.v_impact == pytest.approx(event.v_esc, rel=1e-12)

    # A hand-computed escape velocity for the first pair, from the masses and
    # radii the record itself carries, pins the formula rather than the code.
    first = circular[0]
    expected = math.sqrt(
        2.0
        * const_G
        * (first.M_target_before + first.M_impactor)
        / (first.R_target_before + first.R_impactor)
    )
    assert first.v_esc == pytest.approx(expected, rel=1e-12)
    # Discrimination: dropping the factor of two, the single most plausible
    # slip, changes the value by 29%, far outside the tolerance.
    assert abs(expected / math.sqrt(2.0) - expected) > 0.2 * expected


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a_circular_encounter_leaves_the_orbit_untouched():
    """With no encounter eccentricity the merger cannot move the orbit.

    Two bodies on the same circular orbit have identical velocities, so the
    mass-weighted mean is that velocity and the merged body stays on the same
    orbit. This is the analytic limit of the momentum-conserving merger, and it
    is the case that catches a sign error or a mass weighting applied the wrong
    way round, both of which move the orbit even here.
    """
    events = get_timeline(_config(eccentricity=0.0, semimajoraxis=1.0))

    for event in events:
        assert event.semimajoraxis_ratio == pytest.approx(1.0, rel=1e-12)
        assert event.e_after == pytest.approx(0.0, abs=1e-12)
        assert event.a_before == pytest.approx(1.0 * AU, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_merger_shrinks_the_orbit_and_leaves_it_bound():
    """Conserving momentum through a collision can only lower the orbit.

    Averaging two velocities of equal magnitude lowers the specific orbital
    energy, so under the co-orbital geometry this module assumes the merged
    semi-major axis must be smaller than the target's and the orbit must stay
    bound. That is a property of the shared semi-major axis, not a general
    result for mergers. An implementation that conserved energy instead of
    momentum, the plausible alternative, would not shrink the orbit at all.
    The merged orbit is also checked against vis-viva evaluated on the
    independently computed merged velocity.
    """
    events = get_timeline(_config(eccentricity=0.3, num_impacts=2))

    for event in events:
        assert event.a_after < event.a_before
        assert 0.0 <= event.e_after < 1.0

    # Cross-check the first merger against the closed-form momentum average.
    first = events[0]
    m_star = 1.0 * M_sun
    mu = const_G * m_star
    v_kep = math.sqrt(mu / first.a_before)
    e_enc = 0.3
    v_t = (0.0, v_kep)
    v_i = (v_kep * e_enc, v_kep * math.sqrt(1.0 - e_enc**2))
    m_merged = first.M_target_before + first.M_impactor
    v_m = (
        (first.M_target_before * v_t[0] + first.M_impactor * v_i[0]) / m_merged,
        (first.M_target_before * v_t[1] + first.M_impactor * v_i[1]) / m_merged,
    )
    speed_sq = v_m[0] ** 2 + v_m[1] ** 2
    a_expected = 1.0 / (2.0 / first.a_before - speed_sq / mu)
    assert first.a_after == pytest.approx(a_expected, rel=1e-12)


@pytest.mark.unit
def test_impact_times_are_evenly_spaced_and_carry_the_configured_offset():
    """Times run up to the configured last impact and shift with the offset.

    Even spacing is what makes the schedule finite: placing impacts at equal
    mass increments instead would put the final one at infinite time, because
    the growth law only approaches its asymptote. The offset maps the model's
    zero point onto the PROTEUS clock and must shift every impact by exactly
    the same amount without changing their spacing.
    """
    events = get_timeline(_config(num_impacts=4, time_last=8.0e6))
    times = [event.time for event in events]
    assert times == pytest.approx([2.0e6, 4.0e6, 6.0e6, 8.0e6], rel=1e-12)

    shifted = get_timeline(_config(num_impacts=4, time_last=8.0e6, time_offset=1.0e6))
    shifted_times = [event.time for event in shifted]
    assert shifted_times == pytest.approx([3.0e6, 5.0e6, 7.0e6, 9.0e6], rel=1e-12)

    spacing = [b - a for a, b in zip(times, times[1:])]
    shifted_spacing = [b - a for a, b in zip(shifted_times, shifted_times[1:])]
    assert spacing == pytest.approx(shifted_spacing, rel=1e-12)


@pytest.mark.unit
def test_a_single_impact_delivers_the_whole_budget():
    """The edge case of one impact is a complete timeline, not a degenerate one.

    With num_impacts = 1 the renormalisation has a single weight to scale, so
    that impact must carry the entire configured mass and land exactly at the
    configured final time. A division that assumed at least two impacts, or an
    off-by-one in the time spacing, fails here.
    """
    events = get_timeline(_config(num_impacts=1, mass_accreted=0.5, time_last=3.0e6))

    assert len(events) == 1
    assert events[0].M_impactor == pytest.approx(0.5 * M_earth, rel=1e-12)
    assert events[0].time == pytest.approx(3.0e6, rel=1e-12)
    assert events[0].M_merged_after == pytest.approx(1.5 * M_earth, rel=1e-12)


@pytest.mark.unit
def test_an_unusable_timescale_is_refused_with_an_actionable_message():
    """A timescale far below the first impact time cannot be silently divided by.

    If the whole accretion law completes before the first impact, every weight
    underflows and the renormalisation would divide by zero, producing NaN
    masses that only surface much later as an opaque solver failure. The module
    must reject it at generation time and name both parameters involved.
    """
    with pytest.raises(ValueError, match='timescale'):
        get_timeline(_config(timescale=1.0e-3, time_last=1.0e9, num_impacts=2))


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_merged_orbit_conserves_angular_momentum_of_the_merged_body():
    """The returned orbit reproduces the angular momentum it was built from.

    a_after and e_after are two numbers derived from one velocity vector, so
    they are only consistent if the specific angular momentum implied by the
    pair, sqrt(mu * a * (1 - e^2)), equals r times the tangential velocity that
    produced them. This is the internal consistency check that a mis-signed
    eccentricity or a radius/semi-major-axis confusion breaks.
    """
    m_star = 1.0 * M_sun
    mu = const_G * m_star
    a_target = 1.0 * AU
    m_target = 1.0 * M_earth
    m_impactor = 0.4 * M_earth
    eccentricity = 0.2

    a_after, e_after, v_encounter = _merged_orbit(
        m_target, m_impactor, a_target, 0.0, eccentricity, m_star
    )

    v_kep = math.sqrt(mu / a_target)
    m_merged = m_target + m_impactor
    v_theta = (
        m_target * v_kep + m_impactor * v_kep * math.sqrt(1.0 - eccentricity**2)
    ) / m_merged

    h_from_orbit = math.sqrt(mu * a_after * (1.0 - e_after**2))
    assert h_from_orbit == pytest.approx(a_target * v_theta, rel=1e-9)

    # The encounter velocity reduces to e * v_kep for small eccentricity, which
    # is the approximation the parameter is named for; at e = 0.2 it is within
    # a percent of it, and it must not be zero.
    assert v_encounter == pytest.approx(eccentricity * v_kep, rel=2e-2)
    assert v_encounter > 0.0


@pytest.mark.unit
def test_an_impactor_heavier_than_its_target_is_refused():
    """The target must survive the collision, so it cannot be the lighter body.

    Everything downstream treats the target as the survivor: its mantle re-melts,
    its atmosphere is stripped, its orbit moves. A timeline whose impactor
    outweighs the target describes the opposite collision, and the whole chain
    would silently model the wrong body. Asking for more mass than the planet has
    is the ordinary way to reach that, so it fails at generation.
    """
    with pytest.raises(ValueError, match='lighter than the impactor'):
        get_timeline(_config(mass_accreted=5.0, mass_tot=1.0, num_impacts=1))

    # Just under the planet's own mass is still a legal, if violent, merger, so
    # the guard discriminates rather than refusing every large impact.
    events = get_timeline(_config(mass_accreted=0.9, mass_tot=1.0, num_impacts=1))
    assert events[0].M_impactor < events[0].M_target_before


@pytest.mark.unit
def test_an_impactor_far_too_small_for_a_giant_impact_is_refused():
    """An impact must be large relative to the body it strikes, not just to the budget.

    Each impact re-melts the whole mantle, strips atmosphere and resets the orbit.
    A collision carrying a millionth of the target's mass cannot do any of that,
    and scheduling one silently applies a giant impact's consequences to a pebble
    strike. The share of the accreted budget is bounded separately: this case has
    a perfectly reasonable share of a tiny budget, so only a ratio against the
    target catches it.
    """
    with pytest.raises(ValueError, match='not a giant impact'):
        get_timeline(_config(mass_accreted=1.0e-6, mass_tot=1.0, num_impacts=2))

    # A budget large enough that each impact is a real collision passes, so the
    # floor does not simply reject small timelines.
    events = get_timeline(_config(mass_accreted=0.1, mass_tot=1.0, num_impacts=2))
    assert all(e.M_impactor / e.M_target_before > 1.0e-3 for e in events)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_small_impactor_is_not_less_dense_than_its_own_minerals():
    """The mass-radius scaling is capped at the uncompressed density.

    The scaling is fitted for planets and its radius grows as M**0.282, so the
    density it implies falls without bound as the mass does; extrapolated to a
    small impactor it returns a body less dense than the rock and iron it is made
    of. That radius feeds the mutual escape velocity and the erosion law, so the
    error does not stay local. Below roughly a tenth of an Earth mass the body is
    treated as uncompressed instead, which is the correct limit for a small body.
    """
    from proteus.accretion.dummy import _RHO_IRON, _RHO_SILICATE, _body_radius
    from proteus.utils.structure_estimate import iron_fractions

    config = _config()
    _, x_fe, _ = iron_fractions(0.55, 'radius', mass_tot_M_earth=1.0e-3)
    floor = 1.0 / (x_fe / _RHO_IRON + (1.0 - x_fe) / _RHO_SILICATE)

    tiny = 1.0e-3 * M_earth
    radius = _body_radius(config, tiny)
    density = tiny / (4.0 / 3.0 * math.pi * radius**3)

    # At this mass the cap governs, so the density sits at the floor exactly.
    assert density == pytest.approx(floor, rel=1e-9)
    assert density > 4000.0

    # Discrimination: the uncapped scaling would give a body under 1000 kg m-3,
    # less dense than water and impossible for rock and iron.
    from proteus.utils.structure_estimate import nl20_planet_radius_km

    uncapped_r = nl20_planet_radius_km(x_fe, 1.0e-3) * 1.0e3
    uncapped_rho = tiny / (4.0 / 3.0 * math.pi * uncapped_r**3)
    assert uncapped_rho < 0.5 * floor

    # An Earth-mass body is inside the scaling's range, so the cap is inactive
    # there and the scaling still governs. The iron fraction is mass-dependent
    # under the radius-mode core fraction, so it is evaluated at this body's own
    # mass rather than reused from the small one above.
    _, x_fe_earth, _ = iron_fractions(0.55, 'radius', mass_tot_M_earth=1.0)
    earth_r = _body_radius(config, M_earth)
    assert earth_r == pytest.approx(nl20_planet_radius_km(x_fe_earth, 1.0) * 1.0e3, rel=1e-9)

    # And the cap is genuinely inactive there: the uncompressed radius is the
    # larger of the two, so the scaling is what min() selects.
    floor_earth = 1.0 / (x_fe_earth / _RHO_IRON + (1.0 - x_fe_earth) / _RHO_SILICATE)
    r_uncompressed_earth = (3.0 * M_earth / (4.0 * math.pi * floor_earth)) ** (1.0 / 3.0)
    assert r_uncompressed_earth > earth_r


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_chain_carries_its_eccentricity_from_one_impact_to_the_next():
    """Each merger sees the orbit the previous one produced, not the original.

    The module computes a merged eccentricity and must hand it to the next
    collision as the target's state. Dropping that assignment leaves every
    impact computed against the configured orbit, so the chain describes a
    planet that is re-circularised between collisions, and the eccentricity it
    reports is the one it just discarded.
    """
    events = get_timeline(
        _config(num_impacts=3, eccentricity=0.3, orbit_eccentricity=0.2, mass_accreted=0.6)
    )

    # Each impact's stated pre-impact eccentricity is the previous impact's
    # result, which is exactly what carrying the value forward means.
    assert events[0].e_before == pytest.approx(0.2, rel=1e-12)
    for previous, current in zip(events, events[1:]):
        assert current.e_before == pytest.approx(previous.e_after, rel=1e-12)

    # Discrimination: without the carry every impact would start from the
    # configured 0.2, and the first merger moves it well away from that.
    assert abs(events[0].e_after - 0.2) > 1e-3


@pytest.mark.unit
def test_a_timescale_that_starves_a_late_impact_is_refused():
    """A weight that is small but non-zero is still an unusable impact.

    The growth law's increments decay geometrically, so a timescale far shorter
    than the impact spacing drives the later ones toward zero without ever
    reaching it. Testing for exact underflow therefore misses the whole regime
    the guard exists for: an impact carrying a millionth of the budget is not a
    giant impact, but every arithmetic in the module is perfectly happy with it.
    """
    with pytest.raises(ValueError, match='below the'):
        get_timeline(_config(num_impacts=4, timescale=1.0e5, time_last=5.0e6))

    # The smallest weight here is far above zero, so an exact-underflow test
    # would let this configuration through.
    weights = [
        math.exp(-k * 1.25e6 / 1.0e5) - math.exp(-(k + 1) * 1.25e6 / 1.0e5) for k in range(4)
    ]
    assert min(weights) > 0.0
    assert min(weights) / sum(weights) < 1.0e-4
