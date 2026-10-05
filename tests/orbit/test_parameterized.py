"""Unit tests for the parameterized orbital-migration module
(``proteus.orbit.parameterized``).

The module is closed-form: four prescribed migration laws evaluated
pointwise in time, with no ODE solve and no external binary. Every
expected value below is therefore derived analytically in the
docstring or comment beside it, and none is copied from a run of the
code under test.

Contract clauses exercised:

- ``instant_migration`` is a step, with the switch inclusive at
  ``t == time_migration`` (the source branches on ``t < time_migration``).
- ``sigmoid_migration`` holds the orbit until ``time_migration``,
  carries it to ``sma_final`` over the following ``tau_mig`` along the
  cubic ``3u^2 - 2u^3``, and holds it there afterwards. Both the orbit
  and its rate of change are continuous across the window edges.
- ``high_eccentricity_migration`` circularises at constant orbital
  angular momentum, exciting the eccentricity to
  ``sqrt(1 - sma_final / sma_init)`` at ``time_migration``.
- ``orbital_energy_rate`` returns ``G M_star M_planet (da/dt) / (2 a^2)``
  along each law, zero outside the migration window and for the
  ``none`` and ``instant`` laws.
- ``run_parameterized_orbital_migration`` converts AU to metres,
  starts the track from ``orbit.semimajoraxis``, dispatches on
  ``config.orbit.parameterized.migration`` and writes only the elements
  the law sets: nothing for the static law or before the epoch, the
  semi-major axis for ``instant`` and ``sigmoid``, both for ``high_ecc``.

Physics invariants asserted:

- **Conservation**: the semi-latus rectum ``a (1 - e^2)`` is constant
  through high-eccentricity circularisation, which is conservation of
  orbital angular momentum.
- **Boundedness**: ``a > 0`` and ``0 <= e < 1`` at every time, and
  ``a`` never leaves the interval spanned by ``sma_init`` and
  ``sma_final``.
- **Monotonicity**: ``a`` and ``e`` decrease monotonically once
  migration is active.
- **Energy closure**: the orbital energy rate integrated over the
  migration equals ``E(a_end) - E(a_start)`` with
  ``E = -G M_star M_planet / (2 a)``.
- **Pinned values with discrimination guards**: the cubic quarter,
  half and three-quarter points, the high-eccentricity half-decay
  point, the one-tau e-folding of the eccentricity, the orbital energy
  rate 2 dE / tau at the high-eccentricity epoch, and its e^-2 fall
  over one tau.

Anti-happy-path coverage:

- Limit inputs: ``sma_init == sma_final`` is a fixed point of all
  three laws, and the pre-migration epoch returns the untouched orbit.
- Direction: ``instant`` and ``sigmoid`` are exercised outward
  (``sma_final > sma_init``) as well as inward, since both are
  reachable configurations. ``high_ecc`` is inward only and is
  asserted to refuse the outward case at the law level and, from
  ``time_migration`` on, at the wrapper level; before the epoch the
  config validator is what refuses it.
- Error contract: a non-positive ``tau_mig`` raises, a missing
  ``sma_final`` is named rather than failing inside
  the unit conversion, outward high-eccentricity migration raises,
  and an unrecognised or null ``migration`` raises rather than
  falling through.
- Window edges: the cubic is clamped to its endpoints outside the
  migration window, since continued past it the curve runs away to
  6.8 AU at ``u = 2``.

See ``docs/Validation/orbit/parameterized.md`` for the validation
registry entry behind the ``reference_pinned`` test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from proteus.orbit.parameterized import (
    high_eccentricity_migration,
    instant_migration,
    orbital_energy_rate,
    run_parameterized_orbital_migration,
    sigmoid_migration,
    update_orbital_energy_rate,
)
from proteus.utils.constants import AU, M_earth, M_sun, const_G, secs_per_year

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Asymmetric endpoints (ratio 2.5) keep each pinned value clear of a swapped
# endpoint, a dropped factor of 2 or an off-by-one exponent.
SMA_I = 2.0  # semi-major axis before migration [AU]
SMA_F = 0.8  # semi-major axis after migration [AU]
T_MIG = 1.0e5  # migration epoch [yr]
TAU = 1.0e4  # migration width [yr]

# Closed-form algebra, so nothing but float round-off separates the
# result from the hand-derived value.
RTOL = 1e-12


def _config(
    migration,
    ecc=0.0,
    semimajoraxis=SMA_I,
    sma_final=SMA_F,
    time_migration=T_MIG,
    tau_migration=TAU,
):
    """Narrow stand-in for the attrs Config, carrying only the fields
    ``run_parameterized_orbital_migration`` reads. The track starts from
    ``orbit.semimajoraxis``."""
    return SimpleNamespace(
        orbit=SimpleNamespace(
            semimajoraxis=semimajoraxis,
            eccentricity=ecc,
            parameterized=SimpleNamespace(
                migration=migration,
                sma_final=sma_final,
                time_migration=time_migration,
                tau_migration=tau_migration,
            ),
        )
    )


def _row(time_yr, a_au=SMA_I, ecc=0.0):
    """Runtime row as a run hands it to the step: the orbit already seeded
    from the config at the initial condition, or set by the previous step."""
    return {'Time': time_yr, 'semimajorax': a_au * AU, 'eccentricity': ecc}


def _sweep(func, times, **kwargs):
    """Evaluate a migration law over a time array, returning (a, e)."""
    a = np.empty_like(times)
    e = np.empty_like(times)
    for i, t in enumerate(times):
        out = func(t, **kwargs)
        if isinstance(out, tuple):
            a[i], e[i] = out
        else:
            a[i], e[i] = out, 0.0
    return a, e


# ---------------------------------------------------------------------------
# instant_migration
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_instant_migration_is_a_step_with_an_inclusive_switch():
    """A planet held at sma_init until the migration epoch, then placed
    at sma_final. The source branches on ``t < time_migration``, so the
    switch is inclusive: t exactly equal to the epoch has migrated."""
    before = instant_migration(0.0, SMA_I, SMA_F, T_MIG)
    just_before = instant_migration(T_MIG * (1.0 - 1e-9), SMA_I, SMA_F, T_MIG)
    at_epoch = instant_migration(T_MIG, SMA_I, SMA_F, T_MIG)
    late = instant_migration(1.0e12, SMA_I, SMA_F, T_MIG)

    assert before == pytest.approx(SMA_I, rel=RTOL)
    assert just_before == pytest.approx(SMA_I, rel=RTOL)
    assert at_epoch == pytest.approx(SMA_F, rel=RTOL)
    assert late == pytest.approx(SMA_F, rel=RTOL)
    # Boundedness: the step only ever returns one of the two endpoints, so
    # the orbit never leaves the interval they span. The pinned values above
    # are what separate a step from an interpolation.
    assert at_epoch > 0.0
    assert SMA_F <= before <= SMA_I


# ---------------------------------------------------------------------------
# sigmoid_migration
# ---------------------------------------------------------------------------


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_sigmoid_matches_the_cubic_smoothstep_across_the_window():
    """Analytical limit: the cubic Hermite interpolant that carries a
    value between two endpoints with zero slope at both is
    ``S(u) = 3u^2 - 2u^3``, evaluated on the window fraction
    ``u = (t - t_mig) / tau``, so ``a = a_0 + (a_f - a_0) S(u)``.

    Three points fix it. The quarter point ``S(1/4) = 5/32`` gives
    ``2.0 - 1.2 * 5/32 = 1.8125 AU``, the midpoint ``S(1/2) = 1/2``
    gives 1.4 AU, and the three-quarter point ``S(3/4) = 27/32`` gives
    0.9875 AU.

    The quarter point is what separates the cubic from its neighbours: a
    linear ramp would put it at 1.7 AU and the quintic smootherstep
    ``6u^5 - 15u^4 + 10u^3`` at 1.8757 AU, both outside the tolerance.
    See ``docs/Validation/orbit/parameterized.md``.
    """
    quarter = sigmoid_migration(T_MIG + 0.25 * TAU, SMA_I, SMA_F, T_MIG, TAU)
    middle = sigmoid_migration(T_MIG + 0.50 * TAU, SMA_I, SMA_F, T_MIG, TAU)
    three_q = sigmoid_migration(T_MIG + 0.75 * TAU, SMA_I, SMA_F, T_MIG, TAU)

    assert quarter == pytest.approx(SMA_I + (SMA_F - SMA_I) * 5.0 / 32.0, rel=RTOL)
    assert middle == pytest.approx(0.5 * (SMA_I + SMA_F), rel=RTOL)
    assert three_q == pytest.approx(SMA_I + (SMA_F - SMA_I) * 27.0 / 32.0, rel=RTOL)

    # Shape guard: a linear ramp lands at 1.7 AU here, the quintic
    # smootherstep at 1.8757 AU. Both are further off than the tolerance.
    wrong_linear = SMA_I + (SMA_F - SMA_I) * 0.25
    wrong_quintic = SMA_I + (SMA_F - SMA_I) * (6 * 0.25**5 - 15 * 0.25**4 + 10 * 0.25**3)
    assert abs(quarter - wrong_linear) > 0.05
    assert abs(quarter - wrong_quintic) > 0.02
    # Symmetry of the cubic about the window centre, S(u) + S(1-u) = 1,
    # which holds whatever the endpoints are.
    assert quarter + three_q == pytest.approx(SMA_I + SMA_F, rel=RTOL)
    # Sign and scale guards: an AU-scale semi-major axis inside the
    # interval the endpoints span, not the dimensionless S curve itself.
    assert quarter > 0.0
    assert SMA_F < quarter < SMA_I


@pytest.mark.physics_invariant
def test_sigmoid_decreases_monotonically_onto_sma_final():
    """Inward migration: the semi-major axis falls monotonically from
    sma_init to sma_final and stays inside the interval the endpoints
    span, never overshooting either one."""
    t = np.linspace(T_MIG - TAU, T_MIG + 60.0 * TAU, 2000)
    a, _ = _sweep(
        sigmoid_migration, t, sma_init=SMA_I, sma_final=SMA_F, time_migration=T_MIG, tau_mig=TAU
    )

    assert np.all(np.diff(a) <= 0.0)
    assert a.min() >= SMA_F - 1e-12
    assert a.max() <= SMA_I + 1e-12
    assert a[0] == pytest.approx(SMA_I, rel=RTOL)
    assert a[-1] == pytest.approx(SMA_F, rel=RTOL)
    assert np.all(a > 0.0)


@pytest.mark.physics_invariant
def test_sigmoid_holds_the_orbit_outside_the_migration_window():
    """Edge case at both window boundaries. The orbit is exactly
    sma_init up to and including the start of the window and exactly
    sma_final from its end onwards, so a track that is sampled before or
    long after the migration reports the endpoint rather than an
    extrapolation of the interior curve."""
    before = sigmoid_migration(T_MIG - 1.0e6, SMA_I, SMA_F, T_MIG, TAU)
    at_start = sigmoid_migration(T_MIG, SMA_I, SMA_F, T_MIG, TAU)
    at_end = sigmoid_migration(T_MIG + TAU, SMA_I, SMA_F, T_MIG, TAU)
    after = sigmoid_migration(T_MIG + 1.0e6 * TAU, SMA_I, SMA_F, T_MIG, TAU)

    assert before == pytest.approx(SMA_I, rel=RTOL)
    assert at_start == pytest.approx(SMA_I, rel=RTOL)
    assert at_end == pytest.approx(SMA_F, rel=RTOL)
    assert after == pytest.approx(SMA_F, rel=RTOL)
    # The cubic continued past its window would run away: at u = 2 it
    # reaches 2.0 - 1.2 * (12 - 16) = 6.8 AU, and at u = -1 it reaches
    # 2.0 - 1.2 * 5 = -4.0 AU. Clamping is what keeps the orbit bound.
    assert after < SMA_I
    assert before > 0.0


@pytest.mark.physics_invariant
def test_sigmoid_is_continuous_across_both_window_edges():
    """The migration starts and ends without a step in the semi-major
    axis or in its rate of change, which is the property the cubic is
    chosen for: a discontinuous orbit would hand the atmosphere a
    discontinuous instellation.

    Checked as a one-sided limit either side of each edge, and as a
    centred finite difference of the slope. The cubic has
    ``S'(u) = 6u(1 - u)``, which vanishes at both ends, so both slopes
    are zero to the accuracy of the difference.
    """
    step = 1.0e-4 * TAU

    for edge, expected in ((T_MIG, SMA_I), (T_MIG + TAU, SMA_F)):
        low = sigmoid_migration(edge - step, SMA_I, SMA_F, T_MIG, TAU)
        high = sigmoid_migration(edge + step, SMA_I, SMA_F, T_MIG, TAU)
        assert low == pytest.approx(expected, abs=1e-6)
        assert high == pytest.approx(expected, abs=1e-6)
        # Slope, in AU per window length so both edges share a scale.
        slope = (high - low) / (2.0 * step) * TAU
        assert abs(slope) < 1e-3

    # The centre still moves (S'(1/2) = 3/2, -1.8 AU per window), so the flat
    # edges are not a law that never migrates.
    centre_slope = (
        (
            sigmoid_migration(T_MIG + 0.5 * TAU + step, SMA_I, SMA_F, T_MIG, TAU)
            - sigmoid_migration(T_MIG + 0.5 * TAU - step, SMA_I, SMA_F, T_MIG, TAU)
        )
        / (2.0 * step)
        * TAU
    )
    assert centre_slope == pytest.approx(1.5 * (SMA_F - SMA_I), rel=1e-6)


# ---------------------------------------------------------------------------
# high_eccentricity_migration
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_high_ecc_excites_eccentricity_at_the_migration_epoch():
    """At the epoch the decay factor ``exp(-2 * 0 / tau)`` is unity, so
    the orbit is still at sma_init but its eccentricity has jumped to
    ``e_mig = sqrt(1 - a_f / a_0) = sqrt(0.6)``, the value that places
    the post-migration orbit on the same angular momentum."""
    a, e = high_eccentricity_migration(T_MIG, 0.0, SMA_I, SMA_F, T_MIG, TAU)

    assert a == pytest.approx(SMA_I, rel=RTOL)
    assert e == pytest.approx(np.sqrt(0.6), rel=RTOL)
    # Exponent guard: dropping the square root would give 0.6, not 0.775.
    assert abs(e - 0.6) > 0.1
    # Boundedness: an eccentricity must stay on a bound orbit.
    assert 0.0 <= e < 1.0
    assert a > 0.0


@pytest.mark.physics_invariant
def test_high_ecc_half_decay_point_pins_the_factor_of_two():
    """At ``t = t_mig + (tau / 2) ln(2)`` the factor ``exp(-2 dt / tau)``
    is exactly 1/2, so ``a = a_f / (1 - 0.6 / 2) = 0.8 / 0.7 AU`` and
    ``e = sqrt(1 - 0.7) = sqrt(0.3)``. Dropping the 2 in the exponent
    moves a to 0.8 / (1 - 0.6 / sqrt(2)) instead."""
    a, e = high_eccentricity_migration(
        T_MIG + 0.5 * TAU * np.log(2.0), 0.0, SMA_I, SMA_F, T_MIG, TAU
    )

    assert a == pytest.approx(0.8 / 0.7, rel=RTOL)
    assert e == pytest.approx(np.sqrt(0.3), rel=RTOL)
    # Without the factor 2 the decay is exp(-ln(2)/2), putting a at 1.38953 AU;
    # the margin scales with that gap, so new endpoints cannot soften it.
    wrong_no_two = SMA_F / (1.0 - 0.6 * np.exp(-0.5 * np.log(2.0)))
    assert abs(a - wrong_no_two) > 0.5 * abs(0.8 / 0.7 - wrong_no_two)
    # Sign and scale guards: AU-scale positive semi-major axis.
    assert a > 0.0
    assert SMA_F < a < SMA_I


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_high_ecc_circularises_as_a_pure_exponential_in_eccentricity():
    """Analytical limit: the specific orbital angular momentum of a
    two-body orbit is ``h = sqrt(G M a (1 - e^2))``, so a law that
    conserves h holds the semi-latus rectum ``a (1 - e^2)`` at sma_final,
    the radius the now-circular orbit ends on.

    That equality alone is weak as a test. The source derives ``e`` from
    ``a``, so it follows algebraically for any ``a(t)`` whatsoever. The
    anchor is instead the form the source never evaluates. Substituting

        a(t) = a_f / (1 - e_mig^2 exp(-2 (t - t_mig) / tau))

    into ``1 - e^2 = a_f / a`` collapses the eccentricity to a pure
    exponential with half the decay rate of the semi-major axis:

        e(t) = e_mig exp(-(t - t_mig) / tau)

    so the eccentricity e-folds in exactly one tau. Checked at the half
    decay point: ``sqrt(0.6) exp(-0.5 ln 2) = 0.547723 = sqrt(0.3)``. A
    factor of two lost from the exponent doubles the e-folding time and
    fails immediately.

    The exponential is pinned only over the first few tau. The source
    recovers the eccentricity as ``sqrt(1 - sma_final / a)``, and as the
    orbit circularises that subtraction cancels away its significant
    digits: the quantity under the root is ``e^2``, so it drops below
    double precision once ``e`` falls near 1e-8, and the returned
    eccentricity floors to exactly zero. That floor is asserted below in
    its own right, since reaching it through a negative intermediate
    would produce a nan instead.

    The pericentre ``a (1 - e)`` is meanwhile free to rise, which is what
    separates this law from pericentre-conserving tidal circularisation.
    See ``docs/Validation/orbit/parameterized.md``.
    """
    t = np.linspace(T_MIG, T_MIG + 40.0 * TAU, 2000)
    a, e = _sweep(
        high_eccentricity_migration,
        t,
        ecc=0.0,
        sma_init=SMA_I,
        sma_final=SMA_F,
        time_migration=T_MIG,
        tau_mig=TAU,
    )

    semi_latus = a * (1.0 - e**2)
    np.testing.assert_allclose(semi_latus, SMA_F, rtol=1e-10)

    # The anchor: an exponential decay the source never writes down.
    # Five tau keeps e above 2.6e-3, where the cancellation still leaves
    # far more precision than the tolerance asks for.
    t_early = np.linspace(T_MIG, T_MIG + 5.0 * TAU, 500)
    _, e_early = _sweep(
        high_eccentricity_migration,
        t_early,
        ecc=0.0,
        sma_init=SMA_I,
        sma_final=SMA_F,
        time_migration=T_MIG,
        tau_mig=TAU,
    )
    e_expected = np.sqrt(0.6) * np.exp(-(t_early - T_MIG) / TAU)
    np.testing.assert_allclose(e_early, e_expected, rtol=1e-9)

    # One tau takes e down by exactly 1/e; without the 2 in the exponent it
    # would land at 0.6065 of its initial value rather than 0.3679.
    one_tau = high_eccentricity_migration(T_MIG + TAU, 0.0, SMA_I, SMA_F, T_MIG, TAU)[1]
    assert one_tau / np.sqrt(0.6) == pytest.approx(np.exp(-1.0), rel=1e-10)
    assert abs(one_tau / np.sqrt(0.6) - np.exp(-0.5)) > 0.2

    # Boundedness once the cancellation has eaten the last digits: the
    # eccentricity settles on exactly zero and never turns negative or
    # non-finite on the way.
    assert np.all(np.isfinite(e))
    assert np.all(e >= 0.0)
    assert e[-1] == 0.0

    # The pericentre is NOT conserved. Derived spread is
    # a_f - a_f/(1 + e_mig) = 0.34919 AU; the guard takes half of it.
    pericentre = a * (1.0 - e)
    derived_spread = SMA_F - SMA_F / (1.0 + np.sqrt(0.6))
    assert pericentre.max() - pericentre.min() > 0.5 * derived_spread
    assert np.all(semi_latus > 0.0)


@pytest.mark.physics_invariant
def test_high_ecc_circularises_onto_sma_final():
    """Over many migration widths the orbit relaxes to a circular orbit
    at sma_final, with both a and e falling monotonically and the orbit
    staying bound throughout."""
    t = np.linspace(T_MIG, T_MIG + 60.0 * TAU, 2000)
    a, e = _sweep(
        high_eccentricity_migration,
        t,
        ecc=0.0,
        sma_init=SMA_I,
        sma_final=SMA_F,
        time_migration=T_MIG,
        tau_mig=TAU,
    )

    assert np.all(a > 0.0)
    assert np.all((e >= 0.0) & (e < 1.0))
    assert np.all(np.diff(a) <= 0.0)
    assert np.all(np.diff(e) <= 0.0)
    assert a[-1] == pytest.approx(SMA_F, rel=1e-9)
    assert e[-1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.physics_invariant
def test_high_ecc_leaves_the_orbit_untouched_before_the_epoch():
    """Limit input: before the migration epoch the law is a no-op, so a
    planet handed a non-zero starting eccentricity keeps it."""
    a, e = high_eccentricity_migration(0.0, 0.03, SMA_I, SMA_F, T_MIG, TAU)

    assert a == pytest.approx(SMA_I, rel=RTOL)
    assert e == pytest.approx(0.03, rel=RTOL)
    assert 0.0 <= e < 1.0


# ---------------------------------------------------------------------------
# Limit inputs and the error contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'tau_bad',
    [0.0, -1.0, -1.0e6],
    ids=['zero_width', 'negative_width', 'large_negative_width'],
)
@pytest.mark.parametrize(
    'func',
    [sigmoid_migration, high_eccentricity_migration],
    ids=['sigmoid', 'high_eccentricity'],
)
def test_non_positive_migration_width_is_rejected(func, tau_bad):
    """A migration that completes in zero or negative time is unphysical
    and both smooth laws refuse it rather than dividing by it. This is
    an error-contract test, so it asserts the contract rather than a
    physical invariant."""
    if func is high_eccentricity_migration:
        args = (T_MIG, 0.0, SMA_I, SMA_F, T_MIG, tau_bad)
    else:
        args = (T_MIG, SMA_I, SMA_F, T_MIG, tau_bad)

    with pytest.raises(ValueError) as excinfo:
        func(*args)

    message = str(excinfo.value)
    assert 'tau_mig' in message
    assert 'must be > 0' in message


@pytest.mark.physics_invariant
def test_no_net_migration_is_a_fixed_point_of_every_law():
    """Limit input: with sma_init equal to sma_final there is nothing to
    migrate, so all three laws hold the orbit fixed and circular. For
    the high-eccentricity law this is the e_mig = 0 fixed point."""
    a_ecc, e_ecc = high_eccentricity_migration(T_MIG * 2.0, 0.0, SMA_I, SMA_I, T_MIG, TAU)

    assert a_ecc == pytest.approx(SMA_I, rel=RTOL)
    assert e_ecc == pytest.approx(0.0, abs=1e-14)
    assert sigmoid_migration(T_MIG * 2.0, SMA_I, SMA_I, T_MIG, TAU) == pytest.approx(
        SMA_I, rel=RTOL
    )
    assert instant_migration(T_MIG * 2.0, SMA_I, SMA_I, T_MIG) == pytest.approx(SMA_I, rel=RTOL)


@pytest.mark.physics_invariant
def test_high_ecc_refuses_outward_migration():
    """Outward migration has no high-eccentricity circularisation
    solution, since it would need a negative ``e_mig^2``. The law reports
    that rather than propagating a nan into the orbit. The mirrored
    inward case is well posed and is asserted alongside, so a law that
    rejected every input would also fail."""
    with pytest.raises(ValueError, match='inward only'):
        high_eccentricity_migration(T_MIG, 0.0, 0.5, 1.5, T_MIG, TAU)

    a, e = high_eccentricity_migration(T_MIG, 0.0, 1.5, 0.5, T_MIG, TAU)
    assert a == pytest.approx(1.5, rel=RTOL)
    assert 0.0 <= e < 1.0


# ---------------------------------------------------------------------------
# run_parameterized_orbital_migration: units, dispatch, seeding
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_wrapper_converts_semimajor_axis_from_au_to_metres():
    """The config carries AU but hf_row['semimajorax'] is SI. A missing
    or doubled AU factor is the classic failure here and a dimensionless
    comparison would not see it. Probed after an instant step, so the value
    written comes from the config rather than from the seeded row."""
    hf_row = _row(1.0e6)
    a, _ = run_parameterized_orbital_migration(hf_row, _config('instant'))

    assert a == pytest.approx(SMA_F * AU, rel=RTOL)
    assert hf_row['semimajorax'] == pytest.approx(SMA_F * AU, rel=RTOL)
    # Scale guard: a planetary orbit is ~1e11 m, not ~1 (AU left
    # unconverted) nor ~1e22 (AU applied twice).
    assert 1.0e9 < a < 1.0e13


@pytest.mark.parametrize(
    'migration, a_au, ecc',
    [
        ('none', SMA_I, 0.0),
        ('instant', SMA_F, 0.0),
        ('sigmoid', SMA_I, 0.0),
        ('high_ecc', SMA_I, np.sqrt(0.6)),
    ],
    ids=[
        'no_migration_holds_initial_orbit',
        'instant_jump_already_arrived',
        'sigmoid_ramp_has_not_started',
        'high_eccentricity_at_peak_excitation',
    ],
)
@pytest.mark.physics_invariant
def test_wrapper_dispatches_each_law_at_the_migration_epoch(migration, a_au, ecc):
    """Each regime is routed to its own law. Probed at the epoch, where
    all four differ, rather than at late time where three of them have
    already converged on sma_final and a mis-dispatch would hide."""
    hf_row = _row(T_MIG)
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration))

    assert a == pytest.approx(a_au * AU, rel=RTOL)
    assert e == pytest.approx(ecc, abs=1e-12)
    assert 0.0 <= e < 1.0


@pytest.mark.parametrize(
    'migration, a_au, ecc',
    [
        ('none', SMA_I, 0.0),
        ('instant', SMA_F, 0.0),
        ('sigmoid', 0.5 * (SMA_I + SMA_F), 0.0),
        (
            'high_ecc',
            SMA_F / (1.0 - 0.6 * np.exp(-1.0)),
            np.sqrt(0.6 * np.exp(-1.0)),
        ),
    ],
    ids=[
        'no_migration_holds_initial_orbit',
        'instant_jump_already_arrived',
        'sigmoid_ramp_at_window_centre',
        'high_eccentricity_partly_circularised',
    ],
)
@pytest.mark.physics_invariant
def test_wrapper_dispatch_holds_at_a_second_epoch(migration, a_au, ecc):
    """Second probe half a migration window later, where the sigmoid is
    at the midpoint of its window and the high-eccentricity law has
    decayed by exp(-1), so a regime swap cannot survive on a coincidence
    at a single time. All four values differ here."""
    hf_row = _row(T_MIG + 0.5 * TAU)
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration))

    assert a == pytest.approx(a_au * AU, rel=RTOL)
    assert e == pytest.approx(ecc, abs=1e-12)
    assert a > 0.0


@pytest.mark.parametrize(
    'migration, a_au',
    [('none', SMA_I), ('instant', SMA_F), ('sigmoid', SMA_F), ('high_ecc', SMA_F)],
    ids=['no_migration', 'instant_jump', 'sigmoid_ramp', 'high_eccentricity'],
)
@pytest.mark.physics_invariant
def test_wrapper_settles_on_the_final_orbit(migration, a_au):
    """Long after the epoch every migrating law has arrived at
    sma_final on a circular orbit, and the static law is still at
    orbit.semimajoraxis."""
    hf_row = _row(T_MIG + 60.0 * TAU)
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration))

    assert a == pytest.approx(a_au * AU, rel=1e-8)
    assert 0.0 <= e < 1.0
    assert a > 0.0


@pytest.mark.physics_invariant
def test_wrapper_preserves_angular_momentum_after_unit_conversion():
    """The ``a (1 - e^2) = sma_final`` invariant must survive the AU to
    metre conversion, and the eccentricity is excited even though the
    config starts the planet on a circular orbit."""
    hf_row = _row(T_MIG + 2.0 * TAU)
    a, e = run_parameterized_orbital_migration(hf_row, _config('high_ecc'))

    assert a * (1.0 - e**2) == pytest.approx(SMA_F * AU, rel=1e-10)
    assert e > 0.0
    assert e < 1.0


@pytest.mark.parametrize(
    'migration',
    ['none', 'instant', 'sigmoid', 'high_ecc'],
    ids=['static', 'instant_step', 'sigmoid_ramp', 'high_eccentricity'],
)
@pytest.mark.physics_invariant
def test_wrapper_leaves_the_row_untouched_before_the_epoch(migration):
    """Before time_migration every law holds the orbit, so the step writes
    nothing and the row keeps the orbit it carries. That orbit is set apart
    from the config here (1.5 au, e = 0.3 against 2.0 au, e = 0.05), so a step
    that rewrote the row from the config would land outside the tolerance."""
    hf_row = _row(0.5 * T_MIG, a_au=1.5, ecc=0.3)
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration, ecc=0.05))

    assert hf_row == _row(0.5 * T_MIG, a_au=1.5, ecc=0.3)
    assert (a, e) == (hf_row['semimajorax'], hf_row['eccentricity'])
    assert abs(a - SMA_I * AU) > 0.4 * AU


@pytest.mark.parametrize(
    'migration, time_yr',
    [
        ('none', T_MIG + 0.5 * TAU),
        ('instant', T_MIG + 0.5 * TAU),
        ('sigmoid', T_MIG + 0.5 * TAU),
    ],
    ids=['static', 'instant_step', 'sigmoid_ramp'],
)
@pytest.mark.physics_invariant
def test_wrapper_keeps_an_eccentricity_the_law_does_not_set(migration, time_yr):
    """Only high_ecc evolves the eccentricity, so on the other laws an
    eccentricity the row carries survives a step after the epoch, even when it
    differs from orbit.eccentricity. The static law leaves the semi-major axis
    as well. The high_ecc law on the same row does write e, which keeps this
    from passing for a step that never writes the row at all."""
    hf_row = _row(time_yr, a_au=1.0, ecc=0.3)
    _, e = run_parameterized_orbital_migration(hf_row, _config(migration, ecc=0.05))

    assert e == pytest.approx(0.3, rel=RTOL)
    assert hf_row['eccentricity'] == pytest.approx(0.3, rel=RTOL)
    if migration == 'none':
        assert hf_row['semimajorax'] == pytest.approx(1.0 * AU, rel=RTOL)
    else:
        # The step lands at 0.8 au and the ramp midpoint at 1.4 au.
        assert abs(hf_row['semimajorax'] - 1.0 * AU) > 0.1 * AU

    excited = _row(time_yr, a_au=1.5, ecc=0.3)
    run_parameterized_orbital_migration(excited, _config('high_ecc', ecc=0.05))
    # a (1 - e^2) = sma_final on the high_ecc track, which e = 0.3 at the
    # law's semi-major axis would break.
    assert excited['semimajorax'] * (1.0 - excited['eccentricity'] ** 2) == pytest.approx(
        SMA_F * AU, rel=1e-10
    )
    assert abs(excited['eccentricity'] - 0.3) > 0.1


@pytest.mark.parametrize('time_yr', [0.5 * T_MIG, 1.0e6], ids=['before_epoch', 'after_epoch'])
def test_wrapper_rejects_a_null_migration_setting(time_yr):
    """``migration=None`` is not a regime and is reported rather than
    silently leaving the orbit at whatever hf_row already held, also before
    the epoch, where every valid law leaves the row alone. The row must come
    back untouched, so a caller that swallows the error does not go on to
    integrate a half-written orbit. This is an error-contract test, so it
    asserts the contract and the absence of a side effect rather than a
    physical invariant."""
    hf_row = {'Time': time_yr}

    with pytest.raises(ValueError) as excinfo:
        run_parameterized_orbital_migration(hf_row, _config(None))

    message = str(excinfo.value)
    assert 'Unknown migration option' in message
    assert 'high_ecc' in message
    # No side effect: the orbit keys were never written.
    assert 'semimajorax' not in hf_row
    assert 'eccentricity' not in hf_row


@pytest.mark.physics_invariant
def test_wrapper_rejects_an_unrecognised_migration_setting():
    """An unrecognised regime is reported, before the epoch as well as after
    it, and ahead of the missing-destination check. Config validation blocks
    this upstream, so the path is reachable only by calling the function
    directly. A recognised regime on the same row is asserted alongside,
    so a wrapper that rejected every regime would also fail."""
    with pytest.raises(ValueError, match='Unknown migration option'):
        run_parameterized_orbital_migration({'Time': 1.0e6}, _config('bogus'))
    early = _row(0.5 * T_MIG)
    with pytest.raises(ValueError, match='Unknown migration option'):
        run_parameterized_orbital_migration(early, _config('bogus', sma_final=None))
    assert early == _row(0.5 * T_MIG)

    hf_row = _row(1.0e6)
    a, e = run_parameterized_orbital_migration(hf_row, _config('instant'))
    assert a == pytest.approx(SMA_F * AU, rel=RTOL)
    assert 0.0 <= e < 1.0


@pytest.mark.parametrize('time_yr', [1.0, T_MIG], ids=['first_recorded_step', 'mid_evolution'])
@pytest.mark.parametrize(
    'migration',
    ['instant', 'sigmoid', 'high_ecc'],
    ids=['instant_jump', 'sigmoid_ramp', 'high_eccentricity'],
)
@pytest.mark.physics_invariant
def test_wrapper_requires_a_final_semimajor_axis_for_every_migrating_law(migration, time_yr):
    """Every law that actually moves the planet needs a destination, and
    the wrapper names it. The static regime is asserted alongside because
    it has no destination to require, so a blanket check would break it.
    Probed at the first recorded step as well as mid-evolution, so a
    write that slipped in ahead of the validation would surface."""
    hf_row = {'Time': time_yr}

    with pytest.raises(ValueError) as excinfo:
        run_parameterized_orbital_migration(hf_row, _config(migration, sma_final=None))

    assert 'sma_final' in str(excinfo.value)
    assert 'semimajorax' not in hf_row

    a, e = run_parameterized_orbital_migration(_row(T_MIG), _config('none', sma_final=None))
    assert a == pytest.approx(SMA_I * AU, rel=RTOL)
    assert 0.0 <= e < 1.0


# ---------------------------------------------------------------------------
# Outward migration: the same endpoints swapped, 0.8 -> 2.0 AU
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_instant_migration_steps_outward_when_sma_final_exceeds_sma_init():
    """The step carries the planet either way. Outward, the epoch places
    it at the larger axis, so a law clamped to the smaller endpoint (a
    minimum rather than a selection) would still pass every inward test
    while failing here."""
    before = instant_migration(0.0, SMA_F, SMA_I, T_MIG)
    at_epoch = instant_migration(T_MIG, SMA_F, SMA_I, T_MIG)
    late = instant_migration(1.0e12, SMA_F, SMA_I, T_MIG)

    assert before == pytest.approx(SMA_F, rel=RTOL)
    assert at_epoch == pytest.approx(SMA_I, rel=RTOL)
    assert late == pytest.approx(SMA_I, rel=RTOL)
    # Direction guard: the post-epoch axis is the larger endpoint, which
    # is what a min() or an abs() on the endpoint difference would miss.
    assert at_epoch > before
    # Boundedness and scale.
    assert before > 0.0
    assert SMA_F <= before <= SMA_I
    assert SMA_F <= at_epoch <= SMA_I


@pytest.mark.physics_invariant
def test_sigmoid_matches_the_cubic_smoothstep_migrating_outward():
    """Analytical limit, mirrored. With a_0 = 0.8 and a_f = 2.0 the
    cubic ``S(u) = 3u^2 - 2u^3`` gives ``0.8 + 1.2 S(u)``: the quarter
    point ``S(1/4) = 5/32`` lands at 0.9875 AU, the midpoint at 1.4 AU
    and the three-quarter point ``S(3/4) = 27/32`` at 1.8125 AU.

    The quarter point separates the cubic from its neighbours in this
    direction too: a linear ramp puts it at 1.1 AU and the quintic
    smootherstep ``6u^5 - 15u^4 + 10u^3`` at 0.92422 AU. A law using
    ``-abs(a_f - a_0)`` would put it at 0.6125 AU, below both endpoints.
    """
    quarter = sigmoid_migration(T_MIG + 0.25 * TAU, SMA_F, SMA_I, T_MIG, TAU)
    middle = sigmoid_migration(T_MIG + 0.50 * TAU, SMA_F, SMA_I, T_MIG, TAU)
    three_q = sigmoid_migration(T_MIG + 0.75 * TAU, SMA_F, SMA_I, T_MIG, TAU)

    assert quarter == pytest.approx(SMA_F + (SMA_I - SMA_F) * 5.0 / 32.0, rel=RTOL)
    assert middle == pytest.approx(0.5 * (SMA_I + SMA_F), rel=RTOL)
    assert three_q == pytest.approx(SMA_F + (SMA_I - SMA_F) * 27.0 / 32.0, rel=RTOL)

    # Shape guards, recomputed for this direction rather than reused.
    wrong_linear = SMA_F + (SMA_I - SMA_F) * 0.25
    wrong_quintic = SMA_F + (SMA_I - SMA_F) * (6 * 0.25**5 - 15 * 0.25**4 + 10 * 0.25**3)
    assert abs(quarter - wrong_linear) > 0.05
    assert abs(quarter - wrong_quintic) > 0.02

    # Reflection identity: the inward track at the same window fraction
    # is the mirror of this one about the midpoint of the endpoints.
    inward_quarter = sigmoid_migration(T_MIG + 0.25 * TAU, SMA_I, SMA_F, T_MIG, TAU)
    assert quarter + inward_quarter == pytest.approx(SMA_I + SMA_F, rel=RTOL)

    # Direction, boundedness and scale.
    assert quarter < middle < three_q
    assert SMA_F < quarter < SMA_I


@pytest.mark.physics_invariant
def test_sigmoid_increases_monotonically_onto_sma_final_when_migrating_outward():
    """Outward migration: the semi-major axis rises monotonically from
    sma_init to sma_final, never overshooting either endpoint, and is
    clamped outside the window exactly as the inward track is."""
    t = np.linspace(T_MIG - TAU, T_MIG + 60.0 * TAU, 2000)
    a, _ = _sweep(
        sigmoid_migration, t, sma_init=SMA_F, sma_final=SMA_I, time_migration=T_MIG, tau_mig=TAU
    )

    # Monotonicity in the opposite sense to the inward sweep. The slack
    # is absolute, covering float round-off on AU-scale increments.
    assert np.all(np.diff(a) >= -1.0e-12)
    assert np.all(a > 0.0)
    assert np.all(a >= SMA_F - 1.0e-12)
    assert np.all(a <= SMA_I + 1.0e-12)
    # The track actually spans the endpoints rather than sitting at one:
    # held at sma_init before the window, arrived at sma_final after it.
    assert a[0] == pytest.approx(SMA_F, rel=RTOL)
    assert a[-1] == pytest.approx(SMA_I, rel=RTOL)
    # Clamping past the window. Continued rather than clamped, the cubic
    # runs away: S(2) = 3*4 - 2*8 = -4, which would put the orbit at
    # 0.8 + 1.2 * (-4) = -4.0 AU, an unphysical negative axis.
    past = sigmoid_migration(T_MIG + 2.0 * TAU, SMA_F, SMA_I, T_MIG, TAU)
    assert past == pytest.approx(SMA_I, rel=RTOL)
    assert past > 0.0


@pytest.mark.parametrize(
    'migration, a_au',
    [('instant', SMA_I), ('sigmoid', SMA_F)],
    ids=['instant_jump_already_arrived', 'sigmoid_ramp_has_not_started'],
)
@pytest.mark.physics_invariant
def test_wrapper_dispatches_an_outward_track(migration, a_au):
    """The wrapper carries the outward endpoints through the AU
    conversion and the dispatch. Probed at the epoch, where the two
    laws disagree: the step has arrived at 2.0 AU and the ramp has not
    left 0.8 AU, so a mis-dispatch cannot hide behind a shared value."""
    hf_row = _row(T_MIG, a_au=SMA_F)
    a, e = run_parameterized_orbital_migration(
        hf_row, _config(migration, semimajoraxis=SMA_F, sma_final=SMA_I)
    )

    assert a == pytest.approx(a_au * AU, rel=RTOL)
    assert hf_row['semimajorax'] == pytest.approx(a_au * AU, rel=RTOL)
    assert e == pytest.approx(0.0, abs=RTOL)
    # Scale guard: SI metres, not AU left unconverted nor doubled.
    assert 1.0e9 < a < 1.0e13


@pytest.mark.physics_invariant
def test_wrapper_refuses_an_outward_high_eccentricity_track():
    """High-eccentricity circularisation conserves orbital angular
    momentum, so it can only shrink the orbit. The wrapper surfaces that
    rather than writing a nan into hf_row. The mirrored inward config is
    asserted alongside, so a wrapper that rejected every high_ecc track
    would also fail."""
    hf_row = {'Time': T_MIG}

    with pytest.raises(ValueError, match='inward only'):
        run_parameterized_orbital_migration(
            hf_row, _config('high_ecc', semimajoraxis=SMA_F, sma_final=SMA_I)
        )

    assert 'semimajorax' not in hf_row

    a, e = run_parameterized_orbital_migration(_row(T_MIG), _config('high_ecc'))
    assert a == pytest.approx(SMA_I * AU, rel=RTOL)
    assert e == pytest.approx(np.sqrt(1.0 - SMA_F / SMA_I), rel=RTOL)


@pytest.mark.physics_invariant
def test_high_ecc_passes_a_near_radial_input_eccentricity_through_untouched():
    """Before the event the law returns the orbit it was handed, whatever
    eccentricity that carries. Probed at 0.999, the bound the orbit guard
    uses, so the limit input is exercised rather than only the circular
    case, and the pair (a, e) still describes a bound orbit with a positive
    semi-latus rectum."""
    a, e = high_eccentricity_migration(T_MIG * 0.5, 0.999, SMA_I, SMA_F, T_MIG, TAU)

    assert a == pytest.approx(SMA_I, rel=RTOL)
    assert e == pytest.approx(0.999, rel=RTOL)
    # Boundedness: still an ellipse, not a parabola or a hyperbola.
    assert 0.0 <= e < 1.0
    assert a * (1.0 - e**2) > 0.0
    # The pre-event branch does not excite: the event value would be
    # sqrt(1 - 0.8 / 2.0) = 0.7746, well below what was handed in.
    assert e > np.sqrt(1.0 - SMA_F / SMA_I)


# ---------------------------------------------------------------------------
# orbital_energy_rate: the power the prescribed track removes from the orbit
# ---------------------------------------------------------------------------

# TOI-561 b high-eccentricity setup (input/planets/toi561b.toml).
M_STAR_TOI = 0.806 * M_sun  # [kg]
M_PL_TOI = 2.24 * M_earth  # [kg]
A0_TOI = 0.029 * AU  # [m]
AF_TOI = 0.0106 * AU  # [m]
TMIG_TOI = 1.0e6  # [yr]
TAU_TOI = 1.0e7  # [yr]


def _orbital_energy(sma, m_star=M_STAR_TOI, m_pl=M_PL_TOI):
    """Keplerian orbital energy -G M m / (2 a) [J], written out independently
    of the code under test."""
    return -const_G * m_star * m_pl / (2.0 * sma)


def _rate(t, migration, sma_init=A0_TOI, sma_final=AF_TOI, tau=TAU_TOI):
    """orbital_energy_rate on the TOI-561 b masses and epoch."""
    return orbital_energy_rate(
        t=t,
        migration=migration,
        sma_init=sma_init,
        sma_final=sma_final,
        time_migration=TMIG_TOI,
        tau_mig=tau,
        mass_star=M_STAR_TOI,
        mass_planet=M_PL_TOI,
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_high_ecc_energy_rate_at_the_epoch_is_twice_the_energy_change_over_tau():
    """At the epoch the high-eccentricity track removes orbital energy at
    2 dE / tau, where dE = E(sma_final) - E(sma_init) is the whole energy
    change of circularisation. Analytic limit of d/dt of
    -G M m (1 - e_mig^2 x) / (2 a_f) at x = 1. For TOI-561 b
    (0.806 M_sun, 2.24 M_earth, 0.029 to 0.0106 au, tau = 1e7 yr) dE is
    -2.9e35 J and the onset rate -1.8e21 W."""
    rate = _rate(TMIG_TOI, 'high_ecc')
    delta_e = _orbital_energy(AF_TOI) - _orbital_energy(A0_TOI)
    expected = 2.0 * delta_e / (TAU_TOI * secs_per_year)

    assert rate == pytest.approx(expected, rel=1e-10)
    assert delta_e == pytest.approx(-2.86e35, rel=0.01)
    # Factor guard: dropping the 2 of the decay exponent gives dE / tau.
    assert abs(rate - delta_e / (TAU_TOI * secs_per_year)) > 0.4 * abs(expected)
    assert rate < 0.0  # the orbit loses energy while it shrinks
    assert 1.0e21 < abs(rate) < 3.0e21  # W, not J/yr or erg/s


@pytest.mark.physics_invariant
def test_high_ecc_energy_rate_decays_as_twice_the_eccentricity_e_folding():
    """The high-eccentricity rate is proportional to exp(-2 (t - t_mig) / tau)
    alone, so one tau after the epoch it has fallen by exactly e^-2. A decay
    written with exp(-(t - t_mig) / tau) would fall only by e^-1."""
    ratio = _rate(TMIG_TOI + TAU_TOI, 'high_ecc') / _rate(TMIG_TOI, 'high_ecc')

    assert ratio == pytest.approx(np.exp(-2.0), rel=1e-10)
    assert abs(ratio - np.exp(-1.0)) > 0.2  # exponent guard
    assert 0.0 < ratio < 1.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'migration, sma_final, t_end_tau',
    [
        ('high_ecc', AF_TOI, 40.0),
        ('sigmoid', AF_TOI, 1.0),
        ('sigmoid', 2.5 * A0_TOI, 1.0),
    ],
    ids=['high_eccentricity_inward', 'sigmoid_inward', 'sigmoid_outward'],
)
def test_energy_rate_integrates_to_the_orbital_energy_change(migration, sma_final, t_end_tau):
    """Energy closure: integrating the rate over the migration recovers
    E(a_end) - E(a_start) for each smooth law, inward and outward. A rate
    missing the factor 1/2 of the orbital energy, or written per year
    rather than per second, misses this by a factor of 2 or 3.2e7. Forty
    tau leaves the high-eccentricity orbit e^-80 from sma_final."""
    from scipy.integrate import quad

    t_end = TMIG_TOI + t_end_tau * TAU_TOI
    integral, _ = quad(
        lambda t: _rate(t, migration, sma_final=sma_final),
        TMIG_TOI,
        t_end,
        limit=200,
        epsrel=1e-11,
    )
    integral *= secs_per_year  # rate in W integrated over years
    if migration == 'high_ecc':
        a_end, _ = high_eccentricity_migration(t_end, 0.0, A0_TOI, sma_final, TMIG_TOI, TAU_TOI)
    else:
        a_end = sigmoid_migration(t_end, A0_TOI, sma_final, TMIG_TOI, TAU_TOI)
    expected = _orbital_energy(a_end) - _orbital_energy(A0_TOI)

    assert integral == pytest.approx(expected, rel=1e-8)
    # Sign: inward migration releases energy, outward migration needs it.
    assert np.sign(integral) == (1.0 if sma_final > A0_TOI else -1.0)
    assert 1.0e34 < abs(integral) < 1.0e36


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'migration, t_yr',
    [
        ('high_ecc', TMIG_TOI * (1.0 - 1e-9)),
        ('sigmoid', TMIG_TOI),
        ('sigmoid', TMIG_TOI + TAU_TOI),
        ('sigmoid', TMIG_TOI + 3.0 * TAU_TOI),
        ('instant', TMIG_TOI),
        ('none', TMIG_TOI + 0.5 * TAU_TOI),
    ],
    ids=[
        'high_eccentricity_before_epoch',
        'sigmoid_at_window_start',
        'sigmoid_at_window_end',
        'sigmoid_after_window',
        'instant_step_carries_no_rate',
        'static_track',
    ],
)
def test_energy_rate_vanishes_where_the_orbit_is_held(migration, t_yr):
    """Outside its migration window a law holds the orbit, so no energy is
    removed. The smoothstep has zero slope at both window edges, and the
    instant step releases its energy at one time, which a rate cannot carry,
    so both report zero there. Inside the window the same laws are nonzero,
    which keeps this from passing on a constant zero."""
    rate = _rate(t_yr, migration)
    inside = _rate(TMIG_TOI + 0.5 * TAU_TOI, 'sigmoid')

    assert rate == pytest.approx(0.0, abs=1e-6 * abs(inside))
    assert inside < -1.0e19


@pytest.mark.parametrize(
    'migration, tau, sma_final, t_yr, error',
    [
        ('sigmoid', 0.0, AF_TOI, TMIG_TOI + 1.0, 'must be > 0'),
        ('high_ecc', -1.0, AF_TOI, TMIG_TOI + 1.0, 'must be > 0'),
        ('high_ecc', TAU_TOI, 2.0 * A0_TOI, TMIG_TOI + 1.0, 'inward only'),
        ('high_ecc', TAU_TOI, 2.0 * A0_TOI, TMIG_TOI - 1.0, 'inward only'),
        ('spiral', TAU_TOI, AF_TOI, TMIG_TOI + 1.0, 'Unknown migration option'),
    ],
    ids=[
        'sigmoid_zero_width',
        'high_ecc_negative_tau',
        'high_ecc_outward',
        'high_ecc_outward_before_epoch',
        'unknown_law',
    ],
)
def test_energy_rate_rejects_an_invalid_track(migration, tau, sma_final, t_yr, error):
    """A non-positive timescale, an outward high-eccentricity track (before
    the epoch as well as after it) and an unknown law raise rather than
    returning a rate. The valid inward high-eccentricity track at the same
    time returns a finite rate, so the refusal is specific to the bad input."""
    with pytest.raises(ValueError, match=error):
        _rate(t_yr, migration, sma_final=sma_final, tau=tau)

    valid = _rate(t_yr, 'high_ecc')
    assert np.isfinite(valid)
    assert valid <= 0.0


@pytest.mark.physics_invariant
def test_update_energy_rate_converts_au_and_writes_the_row():
    """The wrapper reads orbit.semimajoraxis and sma_final in au and the masses from the
    row, and writes the rate in W to hf_row['dEdt_orb']. Fed metres as if
    they were au, the rate would change by AU^-1, far outside the tolerance."""
    config = _config(
        'high_ecc',
        semimajoraxis=A0_TOI / AU,
        sma_final=AF_TOI / AU,
        time_migration=TMIG_TOI,
        tau_migration=TAU_TOI,
    )
    hf_row = {'Time': TMIG_TOI, 'M_star': M_STAR_TOI, 'M_planet': M_PL_TOI}

    returned = update_orbital_energy_rate(hf_row, config)

    assert hf_row['dEdt_orb'] == pytest.approx(_rate(TMIG_TOI, 'high_ecc'), rel=1e-12)
    assert returned == hf_row['dEdt_orb']
    assert 1.0e21 < -hf_row['dEdt_orb'] < 3.0e21


def test_update_energy_rate_without_sma_final_raises_and_leaves_the_row():
    """A migrating law with no destination is named, and the row gains no
    energy-rate value."""
    config = _config('high_ecc', semimajoraxis=A0_TOI / AU, sma_final=None)
    hf_row = {'Time': TMIG_TOI, 'M_star': M_STAR_TOI, 'M_planet': M_PL_TOI}

    with pytest.raises(ValueError, match='sma_final'):
        update_orbital_energy_rate(hf_row, config)
    assert 'dEdt_orb' not in hf_row
    assert set(hf_row) == {'Time', 'M_star', 'M_planet'}


@pytest.mark.physics_invariant
def test_update_energy_rate_of_a_static_track_needs_no_final_orbit():
    """The static law never moves the planet, so it needs no sma_final and
    removes no orbital energy: the rate written is zero, while a migrating
    law on the same row is not."""
    hf_row = {'Time': TMIG_TOI, 'M_star': M_STAR_TOI, 'M_planet': M_PL_TOI}
    static = _config('none', semimajoraxis=A0_TOI / AU, sma_final=None)

    assert update_orbital_energy_rate(hf_row, static) == 0.0
    assert hf_row['dEdt_orb'] == 0.0
    migrating = _config(
        'high_ecc',
        semimajoraxis=A0_TOI / AU,
        sma_final=AF_TOI / AU,
        time_migration=TMIG_TOI,
        tau_migration=TAU_TOI,
    )
    assert update_orbital_energy_rate(hf_row, migrating) < -1.0e21
