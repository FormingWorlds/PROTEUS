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

import numpy as np
import pytest
from scipy import integrate

from proteus.orbit.hansen import (
    get_all_m_hansen,
    hansen_fft,
    init_hansen_table,
    kepler_newton,
    nextpow2_int,
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
