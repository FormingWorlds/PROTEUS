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
- ``sigmoid_migration`` follows the logistic ``1 / (1 + exp(x))``
  centred on ``time_migration`` with width ``tau_mig``.
- ``high_eccentricity_migration`` circularises at constant orbital
  angular momentum, exciting the eccentricity to
  ``sqrt(1 - sma_final / sma_init)`` at ``time_migration``.
- ``run_parameterized_orbital_migration`` converts AU to metres,
  dispatches on ``config.orbit.parameterized.migration``, and seeds
  ``hf_row`` from the config on the first recorded step.

Physics invariants asserted:

- **Conservation**: the semi-latus rectum ``a (1 - e^2)`` is constant
  through high-eccentricity circularisation, which is conservation of
  orbital angular momentum.
- **Boundedness**: ``a > 0`` and ``0 <= e < 1`` at every time, and
  ``a`` never leaves the interval spanned by ``sma_init`` and
  ``sma_final``.
- **Monotonicity**: ``a`` and ``e`` decrease monotonically once
  migration is active.
- **Pinned values with discrimination guards**: the logistic centre
  and quarter point, and the high-eccentricity half-decay point.

Anti-happy-path coverage:

- Limit inputs: ``sma_init == sma_final`` is a fixed point of all
  three laws, and the pre-migration epoch returns the untouched orbit.
- Error contract: a non-positive ``tau_mig`` raises, and
  ``migration=None`` raises rather than falling through.
- Two documented gaps are pinned as strict xfails so that closing
  either one turns the corresponding test red: the discontinuity in
  ``sigmoid_migration`` at ``time_migration``, and the unguarded
  ``sqrt`` of a negative number when ``sma_final > sma_init``.

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
    run_parameterized_orbital_migration,
    sigmoid_migration,
)
from proteus.utils.constants import AU

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Asymmetric, non-unity values so that a swapped sma_init/sma_final, a
# dropped factor of 2 or an off-by-one exponent cannot pass by
# coincidence. A ratio of 2.5 between the endpoints keeps every pinned
# value well separated from its wrong-formula neighbour.
SMA_I = 2.0  # semi-major axis before migration [AU]
SMA_F = 0.8  # semi-major axis after migration [AU]
T_MIG = 1.0e5  # migration epoch [yr]
TAU = 1.0e4  # migration width [yr]

# Closed-form algebra, so nothing but float round-off separates the
# result from the hand-derived value.
RTOL = 1e-12


def _config(
    migration, ecc=0.0, sma_init=SMA_I, sma_final=SMA_F, time_migration=T_MIG, tau_migration=TAU
):
    """Narrow stand-in for the attrs Config, carrying only the fields
    ``run_parameterized_orbital_migration`` reads."""
    return SimpleNamespace(
        orbit=SimpleNamespace(
            eccentricity=ecc,
            parameterized=SimpleNamespace(
                migration=migration,
                sma_init=sma_init,
                sma_final=sma_final,
                time_migration=time_migration,
                tau_migration=tau_migration,
            ),
        )
    )


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
    # Boundedness: a step never produces a value off the endpoints, so a
    # regression that interpolated instead would be caught here.
    assert at_epoch > 0.0
    assert SMA_F <= before <= SMA_I


# ---------------------------------------------------------------------------
# sigmoid_migration
# ---------------------------------------------------------------------------


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_sigmoid_matches_the_logistic_centre_and_quarter_point():
    """Pin sigmoid_migration against the analytical logistic
    ``s(x) = 1 / (1 + exp(x))``, with ``x = (t - t_mig) / tau``, so that
    ``a = (a_0 - a_f) s + a_f``.

    Two points on the analytic curve, chosen because they separate the
    plausible wrong formulas:

    - At the centre ``t = t_mig``, ``s = 1/2``, giving the arithmetic
      mean ``a = 0.8 + 1.2 / 2 = 1.4 AU``.
    - At ``t = t_mig + tau ln(3)``, ``exp(x) = 3`` so ``s = 1/4``,
      giving ``a = 0.8 + 1.2 / 4 = 1.1 AU``.

    The second point is what fixes the width: halving tau in the
    exponent moves it to 0.92 AU, far outside tolerance. See
    ``docs/Validation/orbit/parameterized.md``.
    """
    centre = sigmoid_migration(T_MIG, SMA_I, SMA_F, T_MIG, TAU)
    quarter = sigmoid_migration(T_MIG + TAU * np.log(3.0), SMA_I, SMA_F, T_MIG, TAU)

    assert centre == pytest.approx(1.4, rel=RTOL)
    assert quarter == pytest.approx(1.1, rel=RTOL)
    # Width guard: tau -> tau/2 puts the quarter point at
    # 0.8 + 1.2 / (1 + 9) = 0.92 AU, so the gap discriminates.
    wrong_half_tau = SMA_F + (SMA_I - SMA_F) / (1.0 + 9.0)
    assert abs(quarter - wrong_half_tau) > 0.1
    # Sign guard: a semi-major axis is strictly positive under this law.
    assert quarter > 0.0
    # Scale guard: values are AU-scale, not metres (1e11) and not the
    # dimensionless logistic itself (<= 1).
    assert SMA_F < quarter < SMA_I


@pytest.mark.physics_invariant
def test_sigmoid_decreases_monotonically_onto_sma_final():
    """Inward migration: once the law is active the semi-major axis
    falls monotonically and settles on sma_final, never overshooting
    below it nor rising above the logistic centre value."""
    t = np.linspace(T_MIG, T_MIG + 60.0 * TAU, 2000)
    a, _ = _sweep(
        sigmoid_migration, t, sma_init=SMA_I, sma_final=SMA_F, time_migration=T_MIG, tau_mig=TAU
    )

    assert np.all(np.diff(a) <= 0.0)
    assert a.min() >= SMA_F - 1e-12
    assert a.max() <= 0.5 * (SMA_I + SMA_F) + 1e-12
    assert a[-1] == pytest.approx(SMA_F, rel=1e-9)
    assert np.all(a > 0.0)


@pytest.mark.xfail(
    strict=True,
    reason='sigmoid_migration returns sma_init for t < time_migration, so the '
    'semi-major axis steps by half the migration distance at the epoch '
    'instead of passing through it smoothly',
)
@pytest.mark.physics_invariant
def test_sigmoid_is_continuous_across_the_migration_epoch():
    """A logistic centred on t_mig is smooth and antisymmetric about its
    centre. Mirroring the 1.1 AU quarter point, the value at
    ``t = t_mig - tau ln(3)`` is ``0.8 + 1.2 * 3/4 = 1.7 AU``, and the
    pair either side of the centre sums to ``a_0 + a_f``."""
    offset = TAU * np.log(3.0)
    early = sigmoid_migration(T_MIG - offset, SMA_I, SMA_F, T_MIG, TAU)
    late = sigmoid_migration(T_MIG + offset, SMA_I, SMA_F, T_MIG, TAU)

    assert early == pytest.approx(1.7, rel=RTOL)
    # Logistic symmetry about the centre, independent of the pinned value.
    assert early + late == pytest.approx(SMA_I + SMA_F, rel=RTOL)


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
    # Exponent guard: without the factor 2 the decay factor is
    # exp(-ln(2)/2) = 0.70711, putting a at 1.38953 AU. The margin is a
    # fraction of the derived gap so retuning the endpoints cannot
    # silently soften it.
    wrong_no_two = SMA_F / (1.0 - 0.6 * np.exp(-0.5 * np.log(2.0)))
    assert abs(a - wrong_no_two) > 0.5 * abs(0.8 / 0.7 - wrong_no_two)
    # Sign and scale guards: AU-scale positive semi-major axis.
    assert a > 0.0
    assert SMA_F < a < SMA_I


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_high_ecc_conserves_orbital_angular_momentum():
    """Analytical limit: the specific orbital angular momentum of a
    two-body orbit is ``h = sqrt(G M a (1 - e^2))``, so a migration law
    that conserves h holds the semi-latus rectum ``a (1 - e^2)``
    constant at sma_final, the radius the now-circular orbit ends on.

    The source derives ``e`` from ``a``, so that equality is an identity
    of the implementation and holds for any ``a(t)`` at all. It checks
    that ``e(t)`` is the exact inverse of ``a(t)``, not that the time
    dependence is right. The independent pin on ``a(t)`` below is what
    anchors the law itself:

        a(t) = a_f / (1 - e_mig^2 exp(-2 (t - t_mig) / tau))

    The pericentre ``a (1 - e)`` is meanwhile free to rise, which is
    what separates this law from pericentre-conserving tidal
    circularisation. See ``docs/Validation/orbit/parameterized.md``.
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
    # Independent pin on the time dependence, written out from the
    # closed form rather than read back from the module.
    a_expected = SMA_F / (1.0 - 0.6 * np.exp(-2.0 * (t - T_MIG) / TAU))
    np.testing.assert_allclose(a, a_expected, rtol=1e-12)
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


@pytest.mark.xfail(
    strict=True,
    reason='run_parameterized_orbital_migration multiplies sma_init by AU without '
    'checking it, so a config that leaves the schema default of None in '
    'place fails with TypeError instead of naming the missing key',
)
@pytest.mark.physics_invariant
def test_wrapper_reports_a_missing_semimajor_axis():
    """``sma_init`` and ``sma_final`` default to None in the orbit schema,
    so selecting the parameterized model without setting them is a
    reachable configuration. It should name the missing key rather than
    failing inside an arithmetic expression. A fully specified config is
    asserted alongside as the positive control."""
    with pytest.raises(ValueError):
        run_parameterized_orbital_migration(
            {'Time': 1.0e6}, _config('none', sma_init=None), dt=0.0
        )

    a, e = run_parameterized_orbital_migration({'Time': 1.0e6}, _config('none'), dt=0.0)
    assert a == pytest.approx(SMA_I * AU, rel=RTOL)
    assert 0.0 <= e < 1.0


@pytest.mark.xfail(
    strict=True,
    reason='high_eccentricity_migration evaluates sqrt(1 - sma_final / sma_init), '
    'which is the square root of a negative number when the planet '
    'migrates outward, and returns nan instead of reporting it',
)
@pytest.mark.physics_invariant
def test_high_ecc_refuses_outward_migration():
    """Outward migration has no high-eccentricity circularisation
    solution, since it would need a negative ``e_mig^2``. The law should
    say so rather than propagating a nan into the orbit. The mirrored
    inward case is well posed and is asserted alongside, so that closing
    the gap turns this test green rather than moving the failure."""
    with pytest.raises(ValueError):
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
    comparison would not see it."""
    hf_row = {'Time': 1.0e6}
    a, _ = run_parameterized_orbital_migration(hf_row, _config('none'), dt=0.0)

    assert a == pytest.approx(SMA_I * AU, rel=RTOL)
    assert hf_row['semimajorax'] == pytest.approx(SMA_I * AU, rel=RTOL)
    # Scale guard: a planetary orbit is ~1e11 m, not ~1 (AU left
    # unconverted) nor ~1e22 (AU applied twice).
    assert 1.0e9 < a < 1.0e13


@pytest.mark.parametrize(
    'migration, a_au, ecc',
    [
        ('none', SMA_I, 0.0),
        ('instant', SMA_F, 0.0),
        ('sigmoid', 1.4, 0.0),
        ('high_ecc', SMA_I, np.sqrt(0.6)),
    ],
    ids=[
        'no_migration_holds_initial_orbit',
        'instant_jump_already_arrived',
        'sigmoid_ramp_at_logistic_centre',
        'high_eccentricity_at_peak_excitation',
    ],
)
@pytest.mark.physics_invariant
def test_wrapper_dispatches_each_law_at_the_migration_epoch(migration, a_au, ecc):
    """Each regime is routed to its own law. Probed at the epoch, where
    all four differ, rather than at late time where three of them have
    already converged on sma_final and a mis-dispatch would hide."""
    hf_row = {'Time': T_MIG}
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration), dt=0.0)

    assert a == pytest.approx(a_au * AU, rel=RTOL)
    assert e == pytest.approx(ecc, abs=1e-12)
    assert 0.0 <= e < 1.0


@pytest.mark.parametrize(
    'migration, a_au, ecc',
    [
        ('none', SMA_I, 0.0),
        ('instant', SMA_F, 0.0),
        ('sigmoid', 1.1, 0.0),
        ('high_ecc', 6.0 / 7.0, np.sqrt(1.0 / 15.0)),
    ],
    ids=[
        'no_migration_holds_initial_orbit',
        'instant_jump_already_arrived',
        'sigmoid_ramp_at_quarter_point',
        'high_eccentricity_partly_circularised',
    ],
)
@pytest.mark.physics_invariant
def test_wrapper_dispatch_holds_at_a_second_epoch(migration, a_au, ecc):
    """Second probe one ``tau ln(3)`` later, where the sigmoid is at its
    quarter point and the high-eccentricity law has decayed by 1/9, so a
    regime swap cannot survive on a coincidence at a single time."""
    hf_row = {'Time': T_MIG + TAU * np.log(3.0)}
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration), dt=0.0)

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
    sma_init."""
    hf_row = {'Time': T_MIG + 500.0 * TAU}
    a, e = run_parameterized_orbital_migration(hf_row, _config(migration), dt=0.0)

    assert a == pytest.approx(a_au * AU, rel=1e-8)
    assert 0.0 <= e < 1.0
    assert a > 0.0


@pytest.mark.physics_invariant
def test_wrapper_preserves_angular_momentum_after_unit_conversion():
    """The ``a (1 - e^2) = sma_final`` invariant must survive the AU to
    metre conversion, and the eccentricity is excited even though the
    config starts the planet on a circular orbit."""
    hf_row = {'Time': T_MIG + 2.0 * TAU}
    a, e = run_parameterized_orbital_migration(hf_row, _config('high_ecc'), dt=0.0)

    assert a * (1.0 - e**2) == pytest.approx(SMA_F * AU, rel=1e-10)
    assert e > 0.0
    assert 0.0 <= e < 1.0


@pytest.mark.physics_invariant
def test_wrapper_overwrites_a_stale_orbit_on_the_first_recorded_step():
    """At Time <= 1 yr, the first step PROTEUS records, a row carrying a
    stale orbit is replaced by the config values. Every valid regime
    also writes both keys on this path, so the seeding block itself is
    not observable through the return; this pins the row contract, not
    the block."""
    hf_row = {'Time': 1.0, 'semimajorax': 999.0, 'eccentricity': 0.9}
    a, e = run_parameterized_orbital_migration(hf_row, _config('none', ecc=0.05), dt=0.0)

    assert a == pytest.approx(SMA_I * AU, rel=RTOL)
    assert e == pytest.approx(0.05, rel=RTOL)
    assert hf_row['semimajorax'] == pytest.approx(SMA_I * AU, rel=RTOL)
    assert 0.0 <= e < 1.0


def test_wrapper_rejects_a_null_migration_setting():
    """``migration=None`` is not a regime and is reported rather than
    silently leaving the orbit at whatever hf_row already held. The row
    must come back untouched, so a caller that swallows the error does
    not go on to integrate a half-written orbit. This is an
    error-contract test, so it asserts the contract and the absence of a
    side effect rather than a physical invariant."""
    hf_row = {'Time': 1.0e6}

    with pytest.raises(ValueError) as excinfo:
        run_parameterized_orbital_migration(hf_row, _config(None), dt=0.0)

    message = str(excinfo.value)
    assert 'Unknown migration option' in message
    assert 'high_ecc' in message
    # No side effect: the orbit keys were never written.
    assert 'semimajorax' not in hf_row
    assert 'eccentricity' not in hf_row


@pytest.mark.xfail(
    strict=True,
    reason='run_parameterized_orbital_migration has no else branch, so an '
    'unrecognised migration string falls through every elif. With a bare '
    'row it raises KeyError on the return; with a pre-populated row it '
    'silently returns the previous orbit',
)
@pytest.mark.physics_invariant
def test_wrapper_rejects_an_unrecognised_migration_setting():
    """An unrecognised regime should be reported. Config validation
    blocks this upstream, so the gap is reachable only by calling the
    function directly. A recognised regime on the same row is asserted
    alongside, so that closing the gap turns this test green rather than
    moving the failure onto the second call."""
    with pytest.raises(ValueError):
        run_parameterized_orbital_migration({'Time': 1.0e6}, _config('bogus'), dt=0.0)

    hf_row = {'Time': 1.0e6}
    a, e = run_parameterized_orbital_migration(hf_row, _config('none'), dt=0.0)
    assert a == pytest.approx(SMA_I * AU, rel=RTOL)
    assert 0.0 <= e < 1.0
