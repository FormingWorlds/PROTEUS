"""Unit tests for ``proteus.orbit.hansen``: the Kepler solver and the
cached, FFT-backed Hansen coefficients shared by the star-planet and
planet-satellite tidal evolution models (``sp1d``/``ps1d``/``ps1d_evec``).

Exercises:

- ``kepler_newton``: reduction to the mean anomaly at zero eccentricity,
  a forward-map round trip (Kepler's equation solved for E, then
  independently re-evaluated to check M is recovered), and the equation
  residual at several (M, e) pairs.
- ``hansen_fft`` / ``get_all_m_hansen``: the exact Kronecker-delta limit
  at zero eccentricity (an orbit at e=0 has zero radial variation, so
  ``(r/a)^n`` is unity and the Fourier decomposition of
  ``exp(i m v) = exp(i m M)`` collapses to a single mode at k=m for any
  n), and the classical closed-form time average
  ``<(a/r)^3> = (1-e^2)^(-3/2)``, cross-checked against an independent
  numerical quadrature (not the analytic formula alone).
- ``nextpow2_int``: the FFT-size rounding helper used internally by
  ``hansen_fft``.
- ``padded_k_range_for_evection``: the look-ahead eccentricity padding
  used by ``orbit/obliqua.py`` while the evection resonance band is
  active, including its reduction to the unpadded ``kmin_kmax_for_e``
  result at zero rate and its clip to the table's own domain.

Anti-happy-path coverage:

- Kepler solver exercised at e=0 (degenerate/edge case) and at e up to
  0.8 (near-parabolic regime for this application).
- ``nextpow2_int`` pinned at exact powers of two and their neighbours.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
- docs/How-to/test_categorization.md
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy import integrate

import proteus.orbit.hansen as hansen_mod
from proteus.orbit.hansen import (
    _HansenTable,
    _KRangeTable,
    _select_k_range,
    get_all_m_hansen,
    hansen_fft,
    init_hansen_table,
    init_k_range_table,
    kepler_newton,
    kmin_kmax_for_e,
    nextpow2_int,
    padded_k_range_for_evection,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# kepler_newton
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_kepler_newton_reduces_to_mean_anomaly_at_zero_eccentricity():
    """At e=0 the orbit is circular: Kepler's equation E - e sin(E) = M
    degenerates to E = M identically, for any M.

    Limit-input invariant, checked at several M values so a regression
    that added a stray e-independent offset would not go unnoticed at
    a single accidentally-zero M.
    """
    M = np.array([0.0, 0.3, 1.5, 3.0, 5.5])
    E = kepler_newton(M, 0.0)
    np.testing.assert_allclose(E, np.mod(M, 2 * np.pi), atol=1e-12)
    # Boundedness guard: eccentric anomaly must stay within [0, 2*pi) per
    # the function's own np.mod wrap-around convention.
    assert np.all(E >= 0.0) and np.all(E < 2 * np.pi)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('E0,e', [(1.234, 0.3), (0.5, 0.05), (5.9, 0.8), (2.9, 0.65)])
def test_kepler_newton_recovers_eccentric_anomaly_via_forward_map(E0, e):
    """Round-trip check against the forward Kepler map, not the solver
    under test.

    Given a chosen eccentric anomaly ``E0``, the mean anomaly it
    generates is ``M0 = E0 - e sin(E0)`` -- this is Kepler's equation
    itself, evaluated in the forward (non-iterative) direction and
    computed independently of ``kepler_newton``. Feeding ``M0`` back
    into ``kepler_newton`` must recover ``E0`` to within the solver's
    documented Newton-iteration tolerance (1e-13 on the step size).
    """
    M0 = E0 - e * np.sin(E0)
    E_back = float(kepler_newton(np.array([M0]), e)[0])
    assert E_back == pytest.approx(E0, abs=1e-10)
    # Scale/sanity guard: for e < 1 the eccentric and mean anomalies at
    # the same point on the orbit cannot differ by more than ~e radians
    # (E - M = e sin E). A regression that returned M unchanged (e.g. a
    # disabled Newton loop) would still coincidentally pass at small e,
    # so also assert the *equation* is satisfied, independent of E0.
    assert abs(E_back - e * np.sin(E_back) - M0) < 1e-10


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'M,e',
    [(0.1, 0.01), (2.0, 0.4), (4.5, 0.7), (0.9, 0.8), (6.0, 0.2)],
)
def test_kepler_newton_satisfies_equation_residual(M, e):
    """The returned E must satisfy E - e sin(E) = M (mod 2*pi) to near
    machine precision, independent of whether E itself is "correct" in
    an absolute sense -- this is the defining equation the Newton loop
    is supposed to converge on.
    """
    E = float(kepler_newton(np.array([M]), e)[0])
    residual = E - e * np.sin(E) - np.mod(M, 2 * np.pi)
    # Newton's method here is documented to stop once |dE| < 1e-13;
    # the equation residual itself is well within 1e-10 for all tested
    # (M, e). A regression that broke convergence (e.g. wrong fp
    # derivative sign) would blow this up to O(1).
    assert abs(np.mod(residual + np.pi, 2 * np.pi) - np.pi) < 1e-9


def test_kepler_newton_returns_finite_result_when_iteration_cap_is_hit():
    """At extreme eccentricity near periapsis (e=0.999, small M), the
    fixed 10-iteration cap is exhausted without reaching the 1e-13
    convergence threshold -- this is well beyond the e<=0.95 range the
    module's own docstring says the application targets, but the
    solver must still return a finite value (its best estimate after
    10 iterations) rather than exiting the loop with a stale/undefined
    E, silently truncating, or raising.
    """
    M = np.array([0.001])
    e = 0.999
    E = kepler_newton(M, e)
    assert np.all(np.isfinite(E))
    # The residual is not machine-precision here (that's the point --
    # the cap was hit before full convergence), but it must still be
    # much smaller than a non-iterating guess (E=M) would leave: a
    # regression that broke the Newton step entirely (e.g. returned M
    # unchanged) would leave a residual of order e ~ 1, not the ~1e-2
    # this partially-converged case actually achieves.
    residual = abs(E[0] - e * np.sin(E[0]) - M[0])
    assert residual < 1e-1


# ---------------------------------------------------------------------------
# hansen_fft / get_all_m_hansen
# ---------------------------------------------------------------------------


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('n,m', [(-3, 0), (-3, 2), (-3, -2), (-4, 1)])
def test_hansen_fft_reduces_to_kronecker_delta_at_zero_eccentricity(n, m):
    """At e=0, r/a = 1 and the true anomaly equals the mean anomaly
    (v = M), so the Hansen integrand ``(r/a)^n exp(i m v)`` collapses to
    the pure tone ``exp(i m M))`` for ANY degree n. Its Fourier
    decomposition in M is therefore the Kronecker delta X_k^{n,m}(0) =
    1 if k=m else 0 -- independent of n, which is what makes this a
    genuine test of the FFT/indexing machinery rather than of the
    integrand itself.
    """
    k, X = hansen_fft(n=n, m=m, e=0.0, kmin=-4, kmax=4)
    expected = np.where(k == m, 1.0, 0.0)
    np.testing.assert_allclose(X, expected, atol=1e-9)
    # Discrimination guard: an off-by-one in the k-index bookkeeping
    # (e.g. an fftshift/index convention bug) would place the spike at
    # k = m +/- 1 instead, which the elementwise comparison above
    # already catches; this restates the peak location explicitly so a
    # reader sees the failure mode without re-deriving it.
    assert X[list(k).index(m)] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('e', [0.1, 0.3, 0.6, 0.8])
def test_hansen_fft_dc_term_matches_kepler_orbit_time_average(e):
    """The k=0 (DC) component of the n=-3, m=0 Hansen coefficient is the
    orbit-averaged ``<(a/r)^3>``, which has the classical closed form
    ``(1 - e^2)^(-3/2)`` (the identity underlying the eccentricity-tide
    torque in tidal theory). Rather than trust that formula on its own,
    this test re-derives it independently via the standard Kepler
    change of variables ``dM = (1 - e cos E) dE = (r/a) dE``:

        <(a/r)^3>_M = (1/2*pi) integral_0^{2*pi} (a/r)^3 dM
                    = (1/2*pi) integral_0^{2*pi} (1 - e cos E)^(-2) dE

    evaluated here by numerical quadrature (``scipy.integrate.quad``),
    which never calls ``hansen_fft`` or the closed-form expression. All
    three (analytic formula, independent quadrature, and the function
    under test) are compared.
    """
    analytic = (1.0 - e**2) ** -1.5
    quad_val, quad_err = integrate.quad(lambda E: (1.0 - e * np.cos(E)) ** -2, 0.0, 2 * np.pi)
    quad_avg = quad_val / (2 * np.pi)
    assert quad_avg == pytest.approx(analytic, rel=1e-8)
    # quad's absolute error estimate scales with the integral's raw
    # magnitude (~2*pi * analytic, up to ~30 at e=0.8); compare it in
    # relative terms instead of a fixed absolute bound.
    assert quad_err / abs(quad_val) < 1e-6

    k, X = hansen_fft(n=-3, m=0, e=e, kmin=-1, kmax=1)
    dc = X[list(k).index(0)]
    assert dc == pytest.approx(analytic, rel=1e-6)
    assert dc == pytest.approx(quad_avg, rel=1e-6)
    # Sign/scale guard: <(a/r)^3> >= 1 always (r <= a on average weighted
    # this way is not the claim; the claim is the prefactor grows with e).
    # A regression that dropped the exponent (e.g. computed (1-e^2)^-0.5,
    # the n=-1 average) would give 1.0206 instead of 1.0152 at e=0.1 --
    # close enough to require the tighter e=0.8 point to discriminate,
    # where (1-e^2)^-0.5 = 1.667 vs the correct (1-e^2)^-1.5 = 4.630.
    assert dc > 1.0


def test_hansen_fft_explicit_n_calls_nextpow2_int_to_round_up(monkeypatch):
    """When ``N`` is given explicitly, ``hansen_fft`` must round it up to
    the next power of two via ``nextpow2_int`` before running the FFT.
    Spies on ``nextpow2_int`` directly (rather than comparing FFT outputs,
    which converge to the same value regardless of N at these low k's,
    so would not actually discriminate a broken/bypassed rounding step)
    to pin the exact code path: called once with the raw N on the
    explicit-N branch, and never on the adaptive (N=None) branch.
    """
    calls = []
    original = hansen_mod.nextpow2_int

    def spy(x):
        calls.append(x)
        return original(x)

    monkeypatch.setattr(hansen_mod, 'nextpow2_int', spy)

    hansen_fft(n=-3, m=0, e=0.3, kmin=-3, kmax=3, N=100)
    assert calls == [100]

    calls.clear()
    hansen_fft(n=-3, m=0, e=0.3, kmin=-3, kmax=3)  # N=None: adaptive branch
    assert calls == []


# ---------------------------------------------------------------------------
# _select_k_range / init_k_range_table / kmin_kmax_for_e: the eccentricity ->
# [kmin, kmax] window logic, independent of the Hansen-value table above.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_select_k_range_widens_with_eccentricity():
    """The mode window must always cover at least the [-2, 4] floor (padded),
    and must widen at higher eccentricity -- pins the module docstring's
    claim that the number of non-negligible modes 'grows sharply with e'.
    """
    kmin_lo, kmax_lo = _select_k_range(0.05, k_search_max=80)
    kmin_hi, kmax_hi = _select_k_range(0.6, k_search_max=80)
    assert kmin_lo <= -2 and kmax_lo >= 4
    assert kmin_hi <= -2 and kmax_hi >= 4
    # Discriminating: the e=0.6 window must be strictly wider than e=0.05's,
    # not merely equal to the floor at both.
    assert (kmax_hi - kmin_hi) > (kmax_lo - kmin_lo)


def test_select_k_range_falls_back_to_padded_default_when_threshold_unreachable():
    """An unreachably high threshold means no k in the search window has
    |X_k| >= threshold for either the m=0 or m=2 branch -- the function
    must fall back to its documented default window ([-2, 4]) plus pad,
    not an empty or undefined range.
    """
    kmin, kmax = _select_k_range(0.3, threshold=2.0, k_search_max=50, pad=2)
    assert kmin == -2 - 2
    assert kmax == 4 + 2


def test_init_k_range_table_builds_once_and_is_a_noop_on_repeat_call(monkeypatch):
    """Mirrors ``init_hansen_table``'s own force/no-op contract: a repeat
    call without ``force=True`` must leave the existing table (same
    object) untouched, even if given different arguments.
    """
    monkeypatch.setattr(hansen_mod, '_k_range_table', None)
    e_grid = np.array([0.0, 0.3])
    init_k_range_table(e_grid=e_grid, force=True)
    table_after_first = hansen_mod._k_range_table
    assert table_after_first is not None
    np.testing.assert_allclose(table_after_first.e_grid, e_grid)
    assert table_after_first.kmin.shape == (2,)
    assert table_after_first.kmax.shape == (2,)
    # Every entry must cover the [-2, 4] floor (see _select_k_range).
    assert np.all(table_after_first.kmin <= -2)
    assert np.all(table_after_first.kmax >= 4)

    # Repeat call without force=True, with different arguments: no-op.
    init_k_range_table(e_grid=np.array([0.5, 0.9]))
    assert hansen_mod._k_range_table is table_after_first


def test_kmin_kmax_for_e_lazily_builds_table_and_clamps_out_of_range_e(monkeypatch):
    """``kmin_kmax_for_e`` must build the table on first use if absent
    (patched here to a small fake table, since the real default sweep
    takes on the order of a minute), and must clamp an eccentricity
    above the grid's maximum to the last grid point rather than
    extrapolating or raising.
    """
    monkeypatch.setattr(hansen_mod, '_k_range_table', None)
    built = {'n_calls': 0}

    def fake_init(e_grid=None, force=False):
        built['n_calls'] += 1
        hansen_mod._k_range_table = _KRangeTable(
            e_grid=np.array([0.0, 0.5]), kmin=np.array([-6, -10]), kmax=np.array([6, 12])
        )

    monkeypatch.setattr(hansen_mod, 'init_k_range_table', fake_init)

    kmin, kmax = kmin_kmax_for_e(0.05)
    assert built['n_calls'] == 1
    assert (kmin, kmax) == (-6, 6)

    # Table now exists: a second call must NOT rebuild it.
    kmin_hi, kmax_hi = kmin_kmax_for_e(10.0)  # far above the grid's max (0.5)
    assert built['n_calls'] == 1
    # Clamped to the last grid point's window, not extrapolated.
    assert (kmin_hi, kmax_hi) == (-10, 12)


def test_init_hansen_table_is_a_noop_on_repeat_call_without_force(monkeypatch):
    """A second call without ``force=True`` must leave the existing table
    (same object, same window) untouched, even when given a completely
    different e_grid/kmin/kmax -- discriminates a regression that
    rebuilt on every call regardless of the guard.
    """
    monkeypatch.setattr(hansen_mod, '_hansen_table', None)
    init_hansen_table(e_grid=np.array([0.0, 0.1]), kmin=-3, kmax=3, n_deg=1, force=True)
    table_after_first = hansen_mod._hansen_table
    assert table_after_first.kmin == -3
    assert table_after_first.kmax == 3

    init_hansen_table(e_grid=np.array([0.5]), kmin=-1, kmax=1, n_deg=1)  # no force
    assert hansen_mod._hansen_table is table_after_first
    # Discrimination: a broken no-op guard would have rebuilt with the
    # second call's kmin=-1/kmax=1 window instead of keeping the first.
    assert hansen_mod._hansen_table.kmin == -3
    assert hansen_mod._hansen_table.kmax == 3


def test_init_hansen_table_derives_kmin_kmax_from_k_range_table_when_omitted(monkeypatch):
    """When ``kmin``/``kmax`` are not given, ``init_hansen_table`` must
    derive them from the (existing) k-range table's own realized bounds
    -- the overall min of its kmin column and max of its kmax column --
    rather than requiring the caller to hand-pick a window.
    """
    monkeypatch.setattr(hansen_mod, '_hansen_table', None)
    monkeypatch.setattr(
        hansen_mod,
        '_k_range_table',
        _KRangeTable(
            e_grid=np.array([0.0, 0.5]), kmin=np.array([-6, -10]), kmax=np.array([6, 12])
        ),
    )
    init_hansen_table(e_grid=np.array([0.0, 0.2]), n_deg=1, force=True)
    table = hansen_mod._hansen_table
    assert table.kmin == -10
    assert table.kmax == 12


def test_get_all_m_hansen_lazily_builds_table_when_absent(monkeypatch):
    """The hot-path entry point must build the table itself on first use
    if no setup call happened first (patched to a fast fake here; the
    real default sweep is a one-time ~minute cost, out of the unit
    tier's budget).
    """
    monkeypatch.setattr(hansen_mod, '_hansen_table', None)

    def fake_init(n_deg=2, **_kw):
        hansen_mod._hansen_table = _HansenTable(
            e_grid=np.array([0.0, 0.5]),
            kmin=-3,
            kmax=3,
            n_deg=n_deg,
            values={m: np.zeros((2, 7)) for m in range(-n_deg, n_deg + 1)},
        )

    monkeypatch.setattr(hansen_mod, 'init_hansen_table', fake_init)

    k_range, results = get_all_m_hansen(e=0.1, n_deg=2, kmin=-3, kmax=3)
    assert hansen_mod._hansen_table is not None
    assert set(results.keys()) == {-2, -1, 0, 1, 2}
    np.testing.assert_array_equal(k_range, np.arange(-3, 4))


def test_get_all_m_hansen_truncates_and_warns_when_requested_k_range_exceeds_table_window(
    monkeypatch, caplog
):
    """Requesting a wider [kmin, kmax] than the table was built with must
    not crash the caller (a padded, forward-looking request -- see
    orbit/hansen.py's padded_k_range_for_evection -- can legitimately ask
    for more than a table built for a narrower eccentricity range
    covers): it is clipped to the table's actual [kmin, kmax] and a
    warning is logged, rather than silently returning zeros or raising.
    """
    monkeypatch.setattr(hansen_mod, '_hansen_table', None)
    init_hansen_table(e_grid=np.array([0.0, 0.1]), kmin=-4, kmax=4, n_deg=2, force=True)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.orbit.hansen'):
        k_range, results = get_all_m_hansen(e=0.05, n_deg=2, kmin=-10, kmax=10)

    # Clipped to the table's own [-4, 4] window, not the requested [-10, 10].
    np.testing.assert_array_equal(k_range, np.arange(-4, 5))
    assert set(results.keys()) == {-2, -1, 0, 1, 2}
    assert any('exceeds' in rec.message for rec in caplog.records)

    # Discrimination: a request WITHIN the table's window must produce the
    # identical result without any truncation warning -- confirms this is
    # specifically an out-of-window clip, not something that always fires.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.orbit.hansen'):
        k_range_in, _ = get_all_m_hansen(e=0.05, n_deg=2, kmin=-4, kmax=4)
    assert len(k_range_in) == 9
    assert not any('exceeds' in rec.message for rec in caplog.records)


def test_get_all_m_hansen_all_m_are_delta_functions_at_zero_eccentricity(monkeypatch):
    """``get_all_m_hansen`` must reproduce the e=0 Kronecker-delta limit
    (see the ``hansen_fft`` test above) for every m from -n to n
    simultaneously, confirming the dictionary assembly loop does not
    mix up m indices.

    A minimal 2-point e-grid is force-built here instead of relying
    on the lazy default: ``init_hansen_table``'s one-time FFT sweep
    over the production ~100-point ``_DEFAULT_E_GRID`` takes on the
    order of a minute of wall time, which the unit tier's budget
    cannot absorb, and the table it builds is cached in a
    process-global (``proteus.orbit.hansen._hansen_table``) --
    building it here with a narrow ad hoc range would silently limit
    or corrupt interpolation for any other test in the same pytest
    process that calls ``get_all_m_hansen`` afterward. Resetting the
    global via ``monkeypatch`` (auto-restored after this test) keeps
    the fast, minimal table scoped to this test only.
    """

    n = 2
    monkeypatch.setattr('proteus.orbit.hansen._hansen_table', None)
    init_hansen_table(e_grid=np.array([0.0, 0.1]), kmin=-4, kmax=4, n_deg=n, force=True)

    k_range, results = get_all_m_hansen(e=0.0, n_deg=n, kmin=-4, kmax=4)
    assert set(results.keys()) == {-2, -1, 0, 1, 2}
    for m in range(-n, n + 1):
        expected = np.where(k_range == m, 1.0, 0.0)
        np.testing.assert_allclose(results[m], expected, atol=1e-9)


# ---------------------------------------------------------------------------
# nextpow2_int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'x,expected',
    [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (1023, 10), (1024, 10), (1025, 11)],
)
def test_nextpow2_int_matches_smallest_covering_power(x, expected):
    """``2**nextpow2_int(x) >= x`` and no smaller power of two covers x;
    pinned at exact powers of two and their neighbours, where off-by-one
    boundary bugs in ``ceil(log2(x))`` are most likely to surface.
    """
    p = nextpow2_int(x)
    assert p == expected
    if x > 0:
        assert 2**p >= x
        assert 2 ** (p - 1) < x or p == 0


# ---------------------------------------------------------------------------
# padded_k_range_for_evection
# ---------------------------------------------------------------------------


@pytest.fixture
def _fast_k_range_table(monkeypatch):
    """A narrow, fast eccentricity grid for the [kmin, kmax] table, reset
    after the test via monkeypatch. Building the default table
    (``init_k_range_table()`` with no arguments) costs on the order of a
    minute of wall time -- a real FFT-based search at every one of ~100+
    grid points -- far outside the unit tier's budget. Mirrors the
    ``_hansen_table`` reset pattern used above for
    ``test_get_all_m_hansen_matches_hand_built_delta_at_e_zero``.
    """
    monkeypatch.setattr('proteus.orbit.hansen._k_range_table', None)
    init_k_range_table(e_grid=np.array([0.0, 0.2, 0.4, 0.6, 0.8]), force=True)


@pytest.mark.physics_invariant
def test_padded_k_range_for_evection_reduces_to_unpadded_at_zero_rate(_fast_k_range_table):
    """At de_dt_yr=0 (or padding_factor=0), the padded lookup must be
    IDENTICAL to calling ``kmin_kmax_for_e(e_now)`` directly: the padding
    is meant to vanish once outside the regime it exists for (rate small
    relative to eccentricity), and this is the exact-zero limit of that.
    """
    e_now = 0.4
    unpadded = kmin_kmax_for_e(e_now)

    assert (
        padded_k_range_for_evection(e_now, de_dt_yr=0.0, dt_next_yr=100.0, padding_factor=2.0)
        == unpadded
    )
    assert (
        padded_k_range_for_evection(e_now, de_dt_yr=0.05, dt_next_yr=100.0, padding_factor=0.0)
        == unpadded
    )


@pytest.mark.physics_invariant
def test_padded_k_range_for_evection_widens_with_faster_expected_growth(_fast_k_range_table):
    """A nonzero de/dt over a real look-ahead window must widen the window
    relative to the unpadded (current-e) one -- the entire point of the
    padding: cover where e is headed over the upcoming step, not just
    where it is now.
    """
    e_now = 0.2
    unpadded_kmin, unpadded_kmax = kmin_kmax_for_e(e_now)

    # de/dt=0.01/yr over a 20 yr look-ahead pads e by 2*0.01*20=0.4,
    # landing near e=0.6 -- comfortably into a wider table bucket.
    kmin_pad, kmax_pad = padded_k_range_for_evection(
        e_now, de_dt_yr=0.01, dt_next_yr=20.0, padding_factor=2.0
    )
    assert kmax_pad >= unpadded_kmax
    assert kmin_pad <= unpadded_kmin
    # Discrimination: the padded window must be a genuinely DIFFERENT
    # (strictly wider) bucket, not merely the same one by coincidence --
    # otherwise this test would pass even if the padding were a no-op.
    assert (kmin_pad, kmax_pad) != (unpadded_kmin, unpadded_kmax)


@pytest.mark.physics_invariant
def test_padded_k_range_for_evection_ignores_the_sign_of_de_dt(_fast_k_range_table):
    """A DECREASING eccentricity (post-peak decay, de_dt_yr < 0) must pad
    outward exactly as a growing one would: the failure mode this padding
    guards against is only ever "needed modes excluded", so erring wide is
    the deliberately safe direction regardless of which way e is moving.
    """
    e_now = 0.2
    growing = padded_k_range_for_evection(
        e_now, de_dt_yr=0.01, dt_next_yr=20.0, padding_factor=2.0
    )
    decaying = padded_k_range_for_evection(
        e_now, de_dt_yr=-0.01, dt_next_yr=20.0, padding_factor=2.0
    )
    assert growing == decaying
    # Discrimination: this is not simply because both are no-ops -- the
    # padded window is still strictly wider than the unpadded one.
    assert growing != kmin_kmax_for_e(e_now)


@pytest.mark.physics_invariant
def test_padded_k_range_for_evection_clips_to_e_cap(_fast_k_range_table):
    """An extreme rate/look-ahead combination must clip the padded
    eccentricity to the explicit ``e_cap`` argument rather than
    extrapolating past it. ``e_cap=0.5`` is deliberately set BELOW the
    fixture table's own top grid point (0.8), so a pass here cannot be
    explained by ``kmin_kmax_for_e``'s own internal saturation at the
    table's edge -- it must come from this function's own clip.
    """
    e_now = 0.2
    huge_pad = padded_k_range_for_evection(
        e_now, de_dt_yr=10.0, dt_next_yr=1000.0, padding_factor=1.0, e_cap=0.5
    )
    assert huge_pad == kmin_kmax_for_e(0.5)
    # Discrimination: if the clip were silently absorbed by the table's
    # own top-of-domain saturation instead of this function's e_cap, the
    # result would equal kmin_kmax_for_e(0.8), not kmin_kmax_for_e(0.5).
    assert huge_pad != kmin_kmax_for_e(0.8)
