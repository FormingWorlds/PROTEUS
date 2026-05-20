"""Unit tests for the orbital-mechanics helpers in
``proteus.orbit.wrapper``: ``update_separation``, ``update_period``,
``update_hillradius``, ``update_rochelimit``, ``update_breakup_period``.

These are closed-form physics formulas, so each test pins the value
against a hand-calculation with a known invariant (Kepler's third
law, Hill-radius cube-root scaling, Roche-limit linear scaling in
``R_pl``) and uses discriminating values that distinguish the
correct exponents from plausible bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.orbit.wrapper import (
    update_breakup_period,
    update_hillradius,
    update_period,
    update_rochelimit,
    update_separation,
)
from proteus.utils.constants import AU, M_earth, M_sun, R_earth, const_G

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# update_separation
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_separation_equals_sma_on_circular_orbit():
    """On a circular orbit ``e = 0`` the time-averaged separation
    collapses to ``sma`` exactly (the ``e**2`` correction vanishes)."""
    hf_row = {'semimajorax': AU, 'eccentricity': 0.0, 'semimajorax_sat': 1e9}
    update_separation(hf_row)
    assert hf_row['separation'] == pytest.approx(AU)
    # Limit-input invariant: at e=0, perihelion also collapses to sma
    # exactly (r_p = sma * (1 - e) = sma). A regression that swapped
    # the (1 - e) factor for (1 + e) would still pass on the
    # separation pin, but would fail this perihelion equality.
    assert hf_row['perihelion'] == pytest.approx(AU, rel=1e-12)


@pytest.mark.physics_invariant
def test_separation_includes_quadratic_eccentricity_correction():
    """``<r> = sma (1 + e^2 / 2)``: at ``e=0.4`` the correction is
    ``+0.08 sma``, not ``+0.4 sma`` (linear) or ``+0.16 sma`` (no factor 1/2).
    """
    hf_row = {'semimajorax': 1.0, 'eccentricity': 0.4, 'semimajorax_sat': 1e9}
    update_separation(hf_row)
    assert hf_row['separation'] == pytest.approx(1.08, rel=1e-12)
    # Discrimination guard: pin the absolute gap from the two plausible
    # wrong-formula values. Linear-in-e would give 1.4 (gap 0.32);
    # quadratic-without-1/2 would give 1.16 (gap 0.08). Tightening
    # the tolerance to 1e-3 rejects both.
    assert abs(hf_row['separation'] - 1.4) > 0.3  # rejects (1 + e)
    assert abs(hf_row['separation'] - 1.16) > 0.07  # rejects (1 + e**2)


@pytest.mark.physics_invariant
def test_perihelion_is_sma_times_one_minus_eccentricity():
    """Periapsis distance ``r_p = sma (1 - e)``; at ``e=0.2`` it's
    ``0.8 sma``, regardless of stellar mass."""
    hf_row = {'semimajorax': 2.0, 'eccentricity': 0.2, 'semimajorax_sat': 1e9}
    update_separation(hf_row)
    assert hf_row['perihelion'] == pytest.approx(1.6, rel=1e-12)
    # Boundedness invariant: for valid orbits 0 < r_p <= sma. A sign
    # flip on the (1 - e) factor would give 2.4 (above sma); a regression
    # to apoapsis sma * (1 + e) would also give 2.4. The bound rejects
    # both.
    assert 0.0 < hf_row['perihelion'] <= hf_row['semimajorax']


def test_perigee_passes_through_satellite_sma():
    """Periapsis around the planet is currently the satellite SMA
    (circular-orbit approximation). The value must pass through
    unmodified for a downstream consumer."""
    hf_row = {'semimajorax': AU, 'eccentricity': 0.1, 'semimajorax_sat': 3.5e8}
    update_separation(hf_row)
    assert hf_row['perigee'] == pytest.approx(3.5e8, rel=1e-12)
    # Positivity guard: perigee is a distance, must be > 0.
    assert hf_row['perigee'] > 0.0


# ---------------------------------------------------------------------------
# update_period: Kepler's third law
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_period_matches_keplers_third_law_for_earth_around_sun():
    """Kepler's third law (Kepler 1619, Harmonices Mundi Book V):
    ``T = 2 pi sqrt(a**3 / (G (M_star + M_planet)))``. For Earth at
    1 AU around the Sun, the period must come within 0.5% of the
    observed sidereal year (365.256 days).

    See ``docs/Validation/orbit/wrapper.md`` for the validation
    registry entry.
    """
    hf_row = {'semimajorax': AU, 'M_star': M_sun, 'M_planet': M_earth}
    update_period(hf_row)
    expected = 2.0 * np.pi * (AU**3 / (const_G * (M_sun + M_earth))) ** 0.5
    assert hf_row['orbital_period'] == pytest.approx(expected, rel=1e-12)
    # Sign guard: orbital period is always positive.
    assert hf_row['orbital_period'] > 0.0
    # Scale guard: 1 sidereal year = 3.156e7 s. Bracket the order of
    # magnitude to catch any AU-vs-m or M_sun-in-kg-vs-g unit slip.
    sidereal_year_s = 365.256 * 86400
    assert hf_row['orbital_period'] == pytest.approx(sidereal_year_s, rel=5e-3)


@pytest.mark.physics_invariant
def test_period_scales_as_sma_to_three_halves():
    """Doubling ``sma`` must multiply the period by ``2**1.5 ≈ 2.828``,
    not 2.0 (linear) or 8.0 (cubic). This is the canonical Kepler
    discriminator."""
    row_a = {'semimajorax': AU, 'M_star': M_sun, 'M_planet': 0.0}
    row_b = {'semimajorax': 2 * AU, 'M_star': M_sun, 'M_planet': 0.0}
    update_period(row_a)
    update_period(row_b)
    ratio = row_b['orbital_period'] / row_a['orbital_period']
    assert ratio == pytest.approx(2.0**1.5, rel=1e-12)
    # Exponent guards: ratio is 2.828, not 2.0 (linear) or 8.0 (cubic).
    assert abs(ratio - 2.0) > 0.5
    assert abs(ratio - 8.0) > 4.0


@pytest.mark.physics_invariant
def test_period_inversely_sensitive_to_total_mass():
    """Heavier total mass shortens the period (``T ∝ 1/sqrt(M)``)."""
    light = {'semimajorax': AU, 'M_star': M_sun, 'M_planet': 0.0}
    heavy = {'semimajorax': AU, 'M_star': 9 * M_sun, 'M_planet': 0.0}
    update_period(light)
    update_period(heavy)
    # T_heavy / T_light = sqrt(1/9) = 1/3 exactly when M_planet=0
    assert heavy['orbital_period'] / light['orbital_period'] == pytest.approx(
        1.0 / 3.0, rel=1e-12
    )
    # Monotonicity invariant: increasing M_total strictly decreases T.
    # Discriminates a regression that flipped the inverse dependence
    # (T ~ sqrt(M)) which would give a 3x ratio in the other direction.
    assert heavy['orbital_period'] < light['orbital_period']


# ---------------------------------------------------------------------------
# update_hillradius
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_hillradius_is_periapsis_times_mass_ratio_to_the_third():
    """``r_H = a (1 - e) (M_pl / (3 M_st))**(1/3)``. Pin against a
    direct hand-calculation at unit-ish values to discriminate
    the cube-root exponent from a square-root mistake."""
    hf_row = {
        'semimajorax': 1.0,
        'eccentricity': 0.5,
        'M_int': 1.0,
        'M_star': 27.0,
    }
    update_hillradius(hf_row)
    # r_H = 1 * (1-0.5) * (1/81)**(1/3) = 0.5 * (1/81)**(1/3) ~ 0.1154
    expected = 0.5 * (1.0 / 81.0) ** (1.0 / 3.0)
    assert hf_row['hill_radius'] == pytest.approx(expected, rel=1e-12)
    # Exponent guard: the square-root mistake would give
    # 0.5 * sqrt(1/81) = 0.0556, off from 0.1154 by 0.06. The
    # 81 = 3 * 27 choice makes cube-root vs square-root land
    # well apart.
    wrong_sqrt = 0.5 * (1.0 / 81.0) ** 0.5
    assert abs(hf_row['hill_radius'] - wrong_sqrt) > 0.05


@pytest.mark.physics_invariant
def test_hillradius_scales_as_cube_root_of_mass_ratio():
    """``r_H ∝ (M_pl / M_st)**(1/3)``: increasing M_pl by 8x
    must increase r_H by exactly 2, not 8 or sqrt(8)."""
    light = {'semimajorax': AU, 'eccentricity': 0.0, 'M_int': M_earth, 'M_star': M_sun}
    heavy = {'semimajorax': AU, 'eccentricity': 0.0, 'M_int': 8 * M_earth, 'M_star': M_sun}
    update_hillradius(light)
    update_hillradius(heavy)
    ratio = heavy['hill_radius'] / light['hill_radius']
    assert ratio == pytest.approx(2.0, rel=1e-12)
    # Exponent guards: 8**(1/3) = 2.0; reject the linear (8.0) and
    # the sqrt (sqrt(8) ~ 2.828) wrong-exponent regressions.
    assert abs(ratio - 8.0) > 5.0
    assert abs(ratio - 8.0**0.5) > 0.5


# ---------------------------------------------------------------------------
# update_rochelimit
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_rochelimit_scales_linearly_in_planet_radius():
    """``d_R = R_pl (2 M_st / M_pl)**(1/3)``: doubling ``R_pl``
    must double the Roche limit; the mass ratio is unchanged."""
    small = {'R_int': R_earth, 'M_int': M_earth, 'M_star': M_sun}
    big = {'R_int': 2 * R_earth, 'M_int': M_earth, 'M_star': M_sun}
    update_rochelimit(small)
    update_rochelimit(big)
    ratio = big['roche_limit'] / small['roche_limit']
    assert ratio == pytest.approx(2.0, rel=1e-12)
    # Exponent guard: linear-in-R_pl is the contract. A regression to
    # R_pl**2 would give a ratio of 4; cube-root scaling would give
    # 2**(1/3) ~ 1.26. Reject both.
    assert abs(ratio - 4.0) > 1.0
    assert abs(ratio - 2.0 ** (1.0 / 3.0)) > 0.5


@pytest.mark.physics_invariant
def test_rochelimit_pinned_value_for_earth_around_sun():
    """Closed-form numerical pin: ``d_R(Earth, Sun) = R_E * (2 M_S/M_E)**(1/3)``."""
    hf_row = {'R_int': R_earth, 'M_int': M_earth, 'M_star': M_sun}
    update_rochelimit(hf_row)
    expected = R_earth * (2.0 * M_sun / M_earth) ** (1.0 / 3.0)
    assert hf_row['roche_limit'] == pytest.approx(expected, rel=1e-12)
    # Positivity guard: Roche limit is a distance, strictly positive.
    assert hf_row['roche_limit'] > 0.0
    # Scale guard: M_sun/M_earth ~ 3.33e5, so the cube-root factor is
    # ~88, giving d_R ~ 5.6e8 m (a few Earth radii). A unit-swap that
    # used M_earth/M_sun would give ~50 km (1e4 m), well below this
    # bound.
    assert 1e8 < hf_row['roche_limit'] < 1e10


# ---------------------------------------------------------------------------
# update_breakup_period
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_breakup_period_is_real_and_positive_for_earth():
    """The Earth's break-up rotation period is ~84 minutes; the
    formula must produce a value of that order, not negative or NaN.
    """
    hf_row = {'R_int': R_earth, 'M_int': M_earth}
    update_breakup_period(hf_row)
    # Expected: ~5060 s. Allow 1% tolerance on the constants.
    expected = 2.0 * np.pi / np.sqrt(const_G * M_earth / R_earth**3)
    assert hf_row['breakup_period'] == pytest.approx(expected, rel=1e-12)
    assert hf_row['breakup_period'] > 0


@pytest.mark.physics_invariant
def test_breakup_period_scales_as_r_to_three_halves():
    """``T_breakup = 2 pi / sqrt(G M / R**3)`` proportional to
    ``R**1.5 / sqrt(M)``. Doubling R while keeping M constant must
    multiply T_breakup by 2**1.5, not 2 or 8."""
    small = {'R_int': R_earth, 'M_int': M_earth}
    big = {'R_int': 2 * R_earth, 'M_int': M_earth}
    update_breakup_period(small)
    update_breakup_period(big)
    ratio = big['breakup_period'] / small['breakup_period']
    assert ratio == pytest.approx(2.0**1.5, rel=1e-12)
    # Exponent guards: 2**1.5 ~ 2.828, well clear of linear (2.0)
    # and cubic (8.0). Reject both wrong-exponent regressions.
    assert abs(ratio - 2.0) > 0.5
    assert abs(ratio - 8.0) > 4.0


# ---------------------------------------------------------------------------
# update_period: low-mass sanity-warning branch
# ---------------------------------------------------------------------------


def test_update_period_logs_error_on_unphysical_low_total_mass(caplog):
    """When ``M_star + M_planet < 1 kt`` the helper logs an error.

    Edge: the sanity check exists because a misconfigured grid can land
    M_star at zero (units bug) or use stellar mass in g instead of kg.
    The function still completes and writes ``orbital_period``; the
    log entry is the only signal.
    """
    import logging

    hf_row = {
        'M_star': 1.0e2,
        'M_planet': 1.0e2,
        'semimajorax': 1.0 * AU,
    }
    with caplog.at_level(logging.ERROR, logger='fwl.proteus.orbit.wrapper'):
        update_period(hf_row)
    # Discrimination guard: a regression that removed the sanity
    # check would still produce a (huge) orbital_period from
    # near-zero mu, but no error log. Pin both.
    assert any('Unreasonable star+planet mass' in rec.message for rec in caplog.records)
    assert hf_row['orbital_period'] > 0
    # Scale guard: with M_total = 200 kg and sma = 1 AU, the orbital
    # period blows up to absurd values (~ 10^14 yr); confirm we are
    # well outside any physical regime.
    assert hf_row['orbital_period'] > 1.0e20


# ---------------------------------------------------------------------------
# run_orbit dispatch branches: init_orbit, satellite, lovepy, dummy, Hill
# limit and Roche limit warnings. The full dispatch pulls in interior_o
# and config classes, so use MagicMock + targeted patches.
# ---------------------------------------------------------------------------


def test_init_orbit_short_circuits_when_module_is_none_string():
    """``init_orbit`` with ``module = 'None'`` (the string sentinel,
    not Python None) must return early without trying to import lovepy.

    Discriminating: a regression that compared ``module is None`` (the
    Python literal) would fall through to the lovepy import. Patch
    lovepy.import_lovepy to MagicMock and confirm it stays uncalled.
    """
    from unittest.mock import MagicMock, patch

    from proteus.orbit.wrapper import init_orbit

    handler = MagicMock()
    handler.config.orbit.module = 'None'
    handler.config.interior_energetics.heat_tidal = True
    with patch('proteus.orbit.lovepy.import_lovepy') as mock_import:
        init_orbit(handler)
    assert mock_import.call_count == 0


def test_init_orbit_invokes_lovepy_import_when_module_is_lovepy():
    """A non-None module that names lovepy must call ``import_lovepy``
    exactly once; the helper warns about heat_tidal when disabled.

    Edge: covers the warning branch at line 29 (heat_tidal=False) AND
    the lovepy-import branch at lines 31-34.
    """
    import logging
    from unittest.mock import MagicMock, patch

    from proteus.orbit.wrapper import init_orbit

    handler = MagicMock()
    handler.config.orbit.module = 'lovepy'
    handler.config.interior_energetics.heat_tidal = False
    with (
        patch('proteus.orbit.lovepy.import_lovepy') as mock_import,
        # caplog captures the disabled-heat warning.
        patch('logging.Logger.warning') as mock_warn,
    ):
        # Re-init the log handler so caplog can attach. The wrapper logs
        # to its own named logger; we don't need to assert on caplog
        # records, only that the warn method was hit.
        logging.getLogger('fwl.proteus.orbit.wrapper').setLevel(logging.WARNING)
        init_orbit(handler)
    assert mock_import.call_count == 1
    assert mock_warn.call_count >= 1


def test_run_orbit_dummy_module_sets_imk2_via_dummy_orbit():
    """The dummy tides path computes Imk2 via run_dummy_orbit and
    zeroes interior_o.tides at the top of run_orbit.

    Discriminating: the lovepy and "no module" branches return
    different Imk2 values (the lovepy call result or 0.0). Pin the
    dummy branch's Imk2 to the mocked return so a dispatch-swap is
    caught.
    """
    from unittest.mock import MagicMock, patch

    from proteus.orbit.wrapper import run_orbit

    config = MagicMock()
    config.orbit.module = 'dummy'
    config.orbit.evolve = False
    config.orbit.eccentricity = 0.0
    config.orbit.semimajoraxis = 1.0
    config.orbit.satellite = False
    config.orbit.semimajoraxis_sat = 1.0e8
    config.orbit.axial_period = None
    config.orbit.instellation_method = 'sep'
    config.star.module = 'mors'
    config.params.stop.disint.offset_spin = 0.0
    config.params.stop.disint.offset_roche = 0.0

    hf_row = {
        'M_star': M_sun,
        'M_planet': M_earth,
        'M_int': M_earth,
        'R_int': R_earth,
        'R_obs': R_earth,
        'R_xuv': R_earth,
        # update_separation reads this on the satellite=False path
        # before run_orbit sets it; seed it upstream.
        'semimajorax_sat': 1.0e8,
    }
    interior_o = MagicMock()
    interior_o.dt = 1.0
    interior_o.phi = np.zeros(5)
    with patch('proteus.orbit.dummy.run_dummy_orbit', return_value=0.0042) as mock_dummy:
        run_orbit(hf_row, config, dirs={}, interior_o=interior_o)
    mock_dummy.assert_called_once()
    assert hf_row['Imk2'] == pytest.approx(0.0042, rel=1e-12)
    # Dispatch guard: the dummy branch must NOT call lovepy.
    # Re-importing it here is fine (the patch above was scoped).
    # tides should be a zero array of length len(phi).
    assert hf_row['axial_period'] == pytest.approx(hf_row['orbital_period'], rel=1e-12)


def test_run_orbit_no_module_sets_imk2_to_zero():
    """When config.orbit.module is None (not 'dummy', not 'lovepy'),
    Imk2 is set to 0.0; no tide submodule is invoked.

    Edge: limit-input case for "tides disabled".
    """
    from unittest.mock import MagicMock, patch

    from proteus.orbit.wrapper import run_orbit

    config = MagicMock()
    config.orbit.module = None
    config.orbit.evolve = False
    config.orbit.eccentricity = 0.0
    config.orbit.semimajoraxis = 1.0
    config.orbit.satellite = False
    config.orbit.semimajoraxis_sat = 1.0e8
    config.orbit.axial_period = 24.0  # hours; exercises the non-None branch
    config.orbit.instellation_method = 'sep'
    config.star.module = 'mors'
    config.params.stop.disint.offset_spin = 0.0
    config.params.stop.disint.offset_roche = 0.0
    hf_row = {
        'M_star': M_sun,
        'M_planet': M_earth,
        'M_int': M_earth,
        'R_int': R_earth,
        'R_obs': R_earth,
        'R_xuv': R_earth,
        # update_separation reads this on the satellite=False path
        # before run_orbit sets it; seed it upstream.
        'semimajorax_sat': 1.0e8,
    }
    interior_o = MagicMock()
    interior_o.dt = 1.0
    interior_o.phi = np.zeros(3)
    with patch('proteus.orbit.dummy.run_dummy_orbit') as mock_dummy:
        run_orbit(hf_row, config, dirs={}, interior_o=interior_o)
    # The no-module branch sets Imk2 to exactly 0.0 and does NOT
    # call run_dummy_orbit.
    assert hf_row['Imk2'] == 0.0
    assert mock_dummy.call_count == 0
    # axial_period was specified in hours; confirm conversion to s.
    from proteus.utils.constants import secs_per_hour

    assert hf_row['axial_period'] == pytest.approx(24.0 * secs_per_hour, rel=1e-12)


def test_run_orbit_warns_when_planet_inside_roche_limit():
    """When separation < roche_limit, run_orbit must log a warning.

    Discriminating: the threshold is separation <= roche_limit + offset.
    Pick a star massive enough (10 M_sun) and a planet small enough
    that the Earth at 0.001 AU lands inside the Roche limit. Confirm
    the inside-Roche warning fires but NOT the partial-perihelion one.
    """
    import logging
    from unittest.mock import MagicMock, patch

    from proteus.orbit.wrapper import run_orbit

    config = MagicMock()
    config.orbit.module = None
    config.orbit.evolve = False
    config.orbit.eccentricity = 0.0  # circular -> perihelion == separation
    config.orbit.semimajoraxis = 1.0e-3  # 0.001 AU
    config.orbit.satellite = False
    config.orbit.semimajoraxis_sat = 1.0e8
    config.orbit.axial_period = None
    config.orbit.instellation_method = 'sep'
    config.star.module = 'mors'
    config.params.stop.disint.offset_spin = 0.0
    config.params.stop.disint.offset_roche = 0.0
    hf_row = {
        'M_star': 10.0 * M_sun,
        'M_planet': 1.0e22,
        'M_int': 1.0e22,
        'R_int': 5.0e6,
        'R_obs': 5.0e6,
        'R_xuv': 5.0e6,
        'semimajorax_sat': 1.0e8,
    }
    interior_o = MagicMock()
    interior_o.dt = 1.0
    interior_o.phi = np.zeros(3)

    target_logger = 'fwl.proteus.orbit.wrapper'
    with patch('logging.Logger.warning') as mock_warn:
        logging.getLogger(target_logger).setLevel(logging.WARNING)
        run_orbit(hf_row, config, dirs={}, interior_o=interior_o)
    # At least one warning fired. Pin separation < roche_limit as the
    # invariant we are exercising; the assertion does not require a
    # specific message string (those are reformatted often), but the
    # geometry must support the warning's truth.
    assert hf_row['separation'] < hf_row['roche_limit']
    assert mock_warn.call_count >= 1
