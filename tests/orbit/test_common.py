"""Unit tests for ``proteus.orbit.common``: the Kepler solver, Hansen
coefficients, planetary moment of inertia, and the ``Tides_t`` container
shared by the star-planet tidal evolution models (``sp0d``/``sp1d``).

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
- ``get_C_planet``: pinned against the exact uniform-density-sphere
  moment of inertia ``C = (2/5) M R^2``, and the SPIDER array-reversal
  branch, which is shown to silently flip the sign of the result when
  the reversal is skipped for surface-first input.
- ``Tides_t``: interaction lookup/registration contract.

Anti-happy-path coverage:

- Kepler solver exercised at e=0 (degenerate/edge case) and at e up to
  0.8 (near-parabolic regime for this application).
- ``Tides_t.get`` on an unregistered (primary, perturber) pair must
  raise ``KeyError`` rather than returning a default.
- ``get_C_planet``'s SPIDER-reversal branch is exercised both correctly
  (module='spider' on surface-first input) and incorrectly (surface-first
  input without the flag), pinning that the wrong branch does not merely
  drift but flips the sign of C_planet outright.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy import integrate

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import (
    Tides_t,
    get_all_m_hansen,
    get_C_planet,
    hansen_fft,
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
    process-global (``proteus.orbit.common._hansen_table``) --
    building it here with a narrow ad hoc range would silently limit
    or corrupt interpolation for any other test in the same pytest
    process that calls ``get_all_m_hansen`` afterward. Resetting the
    global via ``monkeypatch`` (auto-restored after this test) keeps
    the fast, minimal table scoped to this test only.
    """
    import proteus.orbit.common as common_mod

    n = 2
    monkeypatch.setattr(common_mod, '_hansen_table', None)
    common_mod.init_hansen_table(
        e_grid=np.array([0.0, 0.1]), kmin=-4, kmax=4, n_deg=n, force=True
    )

    k_range, results = get_all_m_hansen(e=0.0, n_deg=n, kmin=-4, kmax=4)
    assert set(results.keys()) == {-2, -1, 0, 1, 2}
    for m in range(-n, n + 1):
        expected = np.where(k_range == m, 1.0, 0.0)
        np.testing.assert_allclose(results[m], expected, atol=1e-9)


# ---------------------------------------------------------------------------
# get_C_planet
# ---------------------------------------------------------------------------


def _uniform_sphere_interior(R: float, rho0: float, nlev_b: int) -> tuple[Interior_t, float]:
    interior = Interior_t(nlev_b=nlev_b)
    interior.radius = np.linspace(0.0, R, nlev_b)
    interior.density = np.full(nlev_b - 1, rho0)
    M = (4.0 / 3.0) * np.pi * R**3 * rho0
    return interior, M


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_get_c_planet_matches_uniform_density_sphere_analytic_value():
    """For a spatially uniform density, the shell-sum in ``get_C_planet``
    reduces to the exact textbook moment of inertia of a solid sphere,
    ``C = (2/5) M R^2`` -- exactly, not just approximately, because the
    per-shell integral ``rho * (r1^5 - r0^5) / 5`` is exact for constant
    rho regardless of how finely the shells are spaced.
    """
    R, rho0 = 6.371e6, 5500.0
    interior, M = _uniform_sphere_interior(R, rho0, nlev_b=8)
    hf_row: dict = {'M_int': M, 'R_int': R}
    cfg = cast(Any, SimpleNamespace(interior_energetics=SimpleNamespace(module='aragog')))

    get_C_planet(hf_row, cfg, interior)

    expected = (2.0 / 5.0) * M * R**2
    assert hf_row['C_planet'] == pytest.approx(expected, rel=1e-12)
    # Discrimination guard: the classic wrong-prefactor bugs for a solid
    # sphere are 1/3 (thin shell) and 1/2 (disk); both are far outside a
    # 1e-6 relative window around 2/5.
    assert abs(hf_row['C_planet'] / (M * R**2) - 1.0 / 3.0) > 0.05
    assert abs(hf_row['C_planet'] / (M * R**2) - 1.0 / 2.0) > 0.1
    # Sanity/scale guard: C_factor for a uniform sphere is exactly 0.4,
    # comfortably inside the physically reasonable [0.2, 0.4] range for
    # real (centrally condensed) planets quoted in the source's own log
    # message.
    assert 0.2 < hf_row['C_planet'] / (M * R**2) <= 0.4


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_get_c_planet_spider_reversal_recovers_cmb_first_ordering():
    """SPIDER emits interior arrays surface-first; ``get_C_planet``
    reverses them when ``config.interior_energetics.module == 'spider'``
    so that index 0 lines up with the CMB, matching the ordering every
    other caller uses. This test pins that the reversal branch produces
    the SAME value as directly supplying CMB-first arrays, using a
    non-uniform (core-mantle-crust) density profile so that reversal
    order actually matters (a uniform profile would pass even with a
    silently broken reversal).
    """
    R = 6.371e6
    r_edges_cmb_first = np.array([0.0, 0.5 * R, 0.8 * R, R])
    rho_cmb_first = np.array([9000.0, 5000.0, 3000.0])  # core -> mantle -> crust

    r0, r1 = r_edges_cmb_first[:-1], r_edges_cmb_first[1:]
    expected_C = (8 * np.pi / 3.0) * np.sum(rho_cmb_first * (r1**5 - r0**5) / 5.0)
    M = np.sum(rho_cmb_first * (4.0 / 3.0 * np.pi * (r1**3 - r0**3)))

    def run(radius, density, module):
        interior = Interior_t(nlev_b=4)
        interior.radius = radius.copy()
        interior.density = density.copy()
        hf_row: dict = {'M_int': M, 'R_int': R}
        cfg = cast(Any, SimpleNamespace(interior_energetics=SimpleNamespace(module=module)))
        get_C_planet(hf_row, cfg, interior)
        return hf_row['C_planet']

    # Non-SPIDER caller supplying already CMB-first arrays: no reversal
    # needed, must match the independently hand-summed expected value.
    c_direct = run(r_edges_cmb_first, rho_cmb_first, module='aragog')
    assert c_direct == pytest.approx(expected_C, rel=1e-12)

    # SPIDER caller supplying surface-first arrays: the reversal branch
    # must recover the identical physical answer.
    c_spider = run(r_edges_cmb_first[::-1], rho_cmb_first[::-1], module='spider')
    assert c_spider == pytest.approx(expected_C, rel=1e-12)

    # Sign guard: a positive density profile must give a positive moment
    # of inertia under the correct (CMB-first) pairing.
    assert expected_C > 0.0


@pytest.mark.physics_invariant
def test_get_c_planet_without_reversal_flag_flips_sign_on_surface_first_input():
    """Discriminating negative case for the reversal branch above: if
    surface-first arrays are supplied WITHOUT setting
    ``module == 'spider'``, the shell pairing ``(r0, r1) = (r_edges[:-1],
    r_edges[1:])`` sees a descending radius array, so ``r1 < r0`` for
    every shell and ``(r1**5 - r0**5)`` is negative throughout. The
    result is not a small numerical drift -- it is the exact negative of
    the physically correct value, which is what makes this bug loud
    rather than silent, and is the reason a caller mismatching the
    module flag would be caught immediately rather than producing a
    plausible-looking wrong answer.
    """
    R = 6.371e6
    r_edges_cmb_first = np.array([0.0, 0.5 * R, 0.8 * R, R])
    rho_cmb_first = np.array([9000.0, 5000.0, 3000.0])
    r0, r1 = r_edges_cmb_first[:-1], r_edges_cmb_first[1:]
    expected_C = (8 * np.pi / 3.0) * np.sum(rho_cmb_first * (r1**5 - r0**5) / 5.0)
    M = np.sum(rho_cmb_first * (4.0 / 3.0 * np.pi * (r1**3 - r0**3)))

    interior = Interior_t(nlev_b=4)
    interior.radius = r_edges_cmb_first[::-1].copy()
    interior.density = rho_cmb_first[::-1].copy()
    hf_row: dict = {'M_int': M, 'R_int': R}
    cfg = cast(Any, SimpleNamespace(interior_energetics=SimpleNamespace(module='dummy')))

    get_C_planet(hf_row, cfg, interior)

    assert hf_row['C_planet'] == pytest.approx(-expected_C, rel=1e-12)
    assert hf_row['C_planet'] < 0.0


# ---------------------------------------------------------------------------
# Tides_t
# ---------------------------------------------------------------------------


def test_tides_t_get_raises_keyerror_for_unregistered_interaction():
    """Error-contract case: querying an interaction that was never
    ``add``-ed must raise ``KeyError`` rather than returning ``None`` or
    a default, since callers (``sp1d``) index straight into ``.nmk``
    without a None-check.
    """
    tides = Tides_t()
    with pytest.raises(KeyError):
        tides.get(primary='planet', perturber='star')
    # No side effect: a failed lookup must not have registered anything.
    assert tides.interactions == []


def test_tides_t_add_is_idempotent_for_the_same_pair():
    """Calling ``add`` twice for the same (primary, perturber) pair must
    return the SAME object both times, not create a duplicate
    interaction (``sp1d`` calls ``.get`` repeatedly assuming a single
    canonical entry per pair).
    """
    tides = Tides_t()
    first = tides.add(primary='planet', perturber='star')
    second = tides.add(primary='planet', perturber='star')
    assert first is second
    assert len(tides.interactions) == 1
    # A different perturber must create a genuinely distinct entry.
    third = tides.add(primary='planet', perturber='moon')
    assert third is not first
    assert len(tides.interactions) == 2


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
