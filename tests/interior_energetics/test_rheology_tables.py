"""Tests for ``proteus.interior_energetics.rheology_tables``.

Source under test: the self-consistent-scheme (SCS) solver for spherical grains with
spherical inclusions, the dimensionless table grid, the on-disk cache and the
interpolators, and the maximum packing fraction of Mader et al. (2013, eq. 49).

Invariants checked:
- the drained SCS shear modulus vanishes at an inclusion fraction of 1/2 for any solid
  Poisson ratio (analytical limit of the scheme);
- to first order in the inclusion fraction the SCS equals the Hashin-Shtrikman upper
  bound for empty spherical pores (dilute limit);
- the moduli stay inside their Hashin-Shtrikman bounds and fall monotonically with v;
- the pure-solid and pure-inclusion limits are recovered exactly;
- the cache regenerates a missing, stale or damaged file and serves a valid one.

See ``docs/How-to/testing.md`` and ``docs/Validation/interior_energetics/rheology_tables.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.interior_energetics import rheology_tables as rt

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# A small grid keeps table generation in the millisecond range.
SMALL = rt.ScsTableGrid(n_nu=3, n_kliq=3, n_gliq=3, n_v=41)


@pytest.fixture(autouse=True)
def _fresh_memo():
    """Each test starts with an empty in-memory table cache."""
    rt._load_cached.cache_clear()
    yield
    rt._load_cached.cache_clear()


def _hs_void_upper(v, mu_s):
    """HS upper bounds (kappa, mu) for empty spherical pores, kappa_sol = 1."""
    zeta = mu_s / 6.0 * (9.0 + 8.0 * mu_s) / (1.0 + 2.0 * mu_s)
    kappa = 4.0 * mu_s * (1.0 - v) / (4.0 * mu_s + 3.0 * v)
    return kappa, mu_s * (1.0 - v) * zeta / (zeta + v * mu_s)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('nu', [0.2, 0.33, 0.45], ids=['nu_low', 'nu_mantle', 'nu_high'])
def test_drained_shear_collapses_at_half_porosity(nu):
    """Dry spherical pores: the SCS shear modulus vanishes at v = 1/2 for any Poisson ratio.

    Analytical limit of Berryman's (1980) scheme: with kappa, mu -> 0 at fixed ratio the
    two SCS equations give kappa/mu = 4(1-v)/(3v) and zeta/mu = v/(1-v), which hold
    together only at v = 1/2, independent of the solid moduli.
    """
    mu_s = float(rt.mu_over_kappa(nu))
    v = np.array([0.0, 0.3, 0.45, 0.49, 0.495, 0.5, 0.505, 0.6])
    k, mu = rt.solve_scs(v, mu_s, 0.0, 0.0)
    mu = mu[0] / mu_s
    assert mu[4] > 1e-3  # still rigid just below 1/2
    assert mu[6] == 0.0 and mu[7] == 0.0  # collapsed above 1/2
    assert mu[5] < 1e-5  # the collapse sits at 1/2 itself
    assert k[0][-1] == 0.0  # empty pores: no bulk stiffness past the collapse
    # the HS upper bound, the most plausible stand-in, keeps about a third of its rigidity there
    assert _hs_void_upper(0.5, mu_s)[1] / mu_s > 0.1


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_dilute_pores_match_hashin_shtrikman_upper_bound():
    """At v = 1e-3 the SCS equals the HS upper bound for dilute spherical pores.

    Both are exact to first order in v (Eshelby dilute limit, Hashin and Shtrikman 1963),
    so they agree to O(v^2); the Voigt average (1 - v) mu_s does not.
    """
    mu_s, v = 0.6, 1e-3  # nu = 0.25
    k, mu = rt.solve_scs([0.0, v], mu_s, 0.0, 0.0)
    k_hs, mu_hs = _hs_void_upper(v, mu_s)
    assert k[0, 1] == pytest.approx(k_hs, rel=1e-5)
    assert mu[0, 1] == pytest.approx(mu_hs, rel=1e-5)
    voigt = (1.0 - v) * mu_s
    assert abs(voigt - mu_hs) / mu_hs > 1e-4  # first-order slope differs from Voigt


@pytest.mark.physics_invariant
def test_scs_limits_and_bounds_for_unrelaxed_melt():
    """Melt at (kappa_liq, G_inf): pure-solid and pure-melt limits, HS bounds, monotonic fall.

    kappa_liq = 0.49 and G_inf = 0.032 (in units of kappa_sol) are the 25 GPa values of
    the MgSiO3 construction; the melt is softer than the solid in both moduli.
    """
    mu_s, k_l, g_l = 0.384, 0.49, 0.032
    v = np.linspace(0.0, 1.0, 51)
    k, mu = (x[0] for x in rt.solve_scs(v, mu_s, k_l, g_l))
    assert (k[0], mu[0]) == (pytest.approx(1.0, abs=1e-12), pytest.approx(mu_s, abs=1e-12))
    assert (k[-1], mu[-1]) == (pytest.approx(k_l, rel=1e-9), pytest.approx(g_l, rel=1e-9))
    lo = g_l + (1 - v) / (
        1 / (mu_s - g_l) + 6 * (k_l + 2 * g_l) * v / (5 * g_l * (3 * k_l + 4 * g_l))
    )
    hi = mu_s + v / (
        1 / (g_l - mu_s) + 6 * (1 + 2 * mu_s) * (1 - v) / (5 * mu_s * (3 + 4 * mu_s))
    )
    assert np.all(mu >= lo * (1 - 1e-9)) and np.all(mu <= hi * (1 + 1e-9))
    assert np.all(np.diff(mu) < 0) and np.all(np.diff(k) < 0)


@pytest.mark.physics_invariant
def test_saturated_bulk_tends_to_reuss_after_collapse():
    """Inviscid melt: once rigidity is lost the bulk modulus is the Reuss average.

    The saturated SCS keeps rigidity beyond the drained collapse (melt-filled pores are
    stiffer than empty ones), and past its own collapse the aggregate is a suspension.
    """
    mu_s, k_l = 0.384, 0.49
    v = np.linspace(0.0, 1.0, 201)
    k, mu = (x[0] for x in rt.solve_scs(v, mu_s, k_l, 0.0))
    v_c = v[np.flatnonzero(mu > 0)[-1]]
    assert 0.5 < v_c < 0.8  # collapses later than the drained scheme
    reuss = 1.0 / (v / k_l + (1.0 - v))
    after = v > v_c
    np.testing.assert_allclose(k[after], reuss[after], rtol=1e-12)
    assert np.all(k[~after] >= reuss[~after] * (1 - 1e-12))  # Reuss is the lower bound


@pytest.mark.physics_invariant
def test_tables_reproduce_solver_at_nodes_and_bracket_between(tmp_path):
    """Interpolated moduli equal the solver at grid nodes and lie between neighbours off them."""
    tab = rt.load_tables(SMALL, tmp_path)
    ax = SMALL.axes()
    nu, kl, gl = ax['nu'][1], 10 ** ax['log_kliq'][1], 10 ** ax['log_gliq'][1]
    k_ref, mu_ref = rt.solve_scs(ax['v'], float(rt.mu_over_kappa(nu)), kl, gl)
    np.testing.assert_allclose(tab.unrelaxed_mu(ax['v'], nu, kl, gl), mu_ref[0], rtol=1e-12)
    v_mid = 0.5 * (ax['v'][10] + ax['v'][11])
    mid = float(tab.unrelaxed_mu(v_mid, nu, kl, gl))
    assert mu_ref[0][11] < mid < mu_ref[0][10]
    kd, mud = tab.drained(ax['v'][:5], nu)
    assert np.all(np.diff(kd) < 0) and np.all(np.diff(mud) < 0)


@pytest.mark.physics_invariant
def test_framework_moves_collapse_to_critical_fraction(tmp_path):
    """The stretched drained moduli vanish at v_crit and equal the SCS at v * v_c0 / v_crit."""
    tab = rt.load_tables(SMALL, tmp_path)
    nu, v_crit = 0.3, 0.25  # collapse moved from 1/2 to 1/4
    assert tab.drained_collapse == pytest.approx(0.5, abs=1e-3)
    k_f, mu_f = tab.framework(np.array([0.1, 0.25, 0.4]), v_crit, nu)
    v_c0 = np.interp(nu, SMALL.axes()['nu'], tab.drained_collapse)
    k_ref, mu_ref = tab.drained(0.1 * v_c0 / v_crit, nu)
    assert (k_f[0], mu_f[0]) == (pytest.approx(float(k_ref)), pytest.approx(float(mu_ref)))
    assert np.all(k_f[1:] == 0.0) and np.all(mu_f[1:] == 0.0)


def test_lookup_outside_table_range_raises(tmp_path):
    """A Poisson ratio, melt modulus or fraction outside the grid raises instead of extrapolating."""
    tab = rt.load_tables(SMALL, tmp_path)
    with pytest.raises(ValueError, match='nu outside'):
        tab.drained(0.2, 0.49)
    with pytest.raises(ValueError, match='G_inf/kappa_sol outside'):
        tab.unrelaxed_mu(0.2, 0.3, 0.5, 5.0)
    with pytest.raises(ValueError, match='v outside'):
        tab.saturated_kappa(1.2, 0.3, 0.5)


def test_grid_rejects_unphysical_axes():
    """Grid validation: Poisson ratio below 1/2, positive moduli ratios, enough nodes."""
    with pytest.raises(ValueError, match='nu_max < 0.5'):
        rt.ScsTableGrid(nu_max=0.5)
    with pytest.raises(ValueError, match='kliq'):
        rt.ScsTableGrid(kliq_min=0.0)
    with pytest.raises(ValueError, match='nodes'):
        rt.ScsTableGrid(n_v=2)
    assert rt.ScsTableGrid().hash_short() != rt.ScsTableGrid(n_v=101).hash_short()


def test_cache_generates_once_and_reuses_file(tmp_path, monkeypatch):
    """A cold cache builds and writes the tables; a later load reads the file without solving."""
    calls = []
    real = rt.build_tables
    monkeypatch.setattr(rt, 'build_tables', lambda g: calls.append(g) or real(g))
    rt.load_tables(SMALL, tmp_path)
    path = rt.table_path(SMALL, tmp_path)
    assert path.is_file() and len(calls) == 1
    rt._load_cached.cache_clear()
    tab = rt.load_tables(SMALL, tmp_path)
    assert len(calls) == 1  # served from disk
    assert tab.grid == SMALL
    assert not list(tmp_path.glob('*.tmp'))  # atomic write left no temporary file


@pytest.mark.parametrize('damage', ['stale_solver', 'truncated'], ids=['stale', 'damaged'])
def test_cache_regenerates_stale_or_damaged_file(tmp_path, monkeypatch, damage):
    """A file from another solver version, or a truncated file, is rebuilt rather than used."""
    path = rt.table_path(SMALL, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if damage == 'stale_solver':
        np.savez(path, grid_hash=SMALL.hash_short(), solver='0123456789ab')
    else:
        path.write_bytes(b'PK\x03\x04 truncated')
    calls = []
    real = rt.build_tables
    monkeypatch.setattr(rt, 'build_tables', lambda g: calls.append(g) or real(g))
    tab = rt.load_tables(SMALL, tmp_path)
    assert len(calls) == 1
    with np.load(path) as data:
        assert str(data['solver']) == rt.solver_identity()
    assert tab.drained_collapse == pytest.approx(0.5, abs=1e-3)


def test_cache_dir_prefers_env_over_fwl_data(tmp_path, monkeypatch):
    """``$PROTEUS_RHEOLOGY_CACHE_DIR`` overrides the FWL_DATA location, read at call time."""
    monkeypatch.delenv(rt.CACHE_ENV, raising=False)
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl'))
    assert rt.cache_dir() == tmp_path / 'fwl' / rt.CACHE_SUBDIR
    monkeypatch.setenv(rt.CACHE_ENV, str(tmp_path / 'shared'))
    assert rt.cache_dir() == tmp_path / 'shared'
    assert rt.table_path(SMALL).parent == tmp_path / 'shared'


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_max_packing_fraction_mader_eq49():
    """Mader et al. (2013, eq. 49): phi_m = phi_m1 exp[-(log10 r_p)^2 / (2 b^2)].

    r_p = 10, b = 1 gives phi_m1 exp(-1/2); a natural logarithm in place of log10 would
    give phi_m1 exp(-2.65), a factor 9 lower.
    """
    assert rt.max_packing_fraction(0.55, 1.0, 1.0) == pytest.approx(0.55, rel=1e-12)
    val = rt.max_packing_fraction(0.55, 1.0, 10.0)
    assert val == pytest.approx(0.55 * np.exp(-0.5), rel=1e-12)
    assert abs(val - 0.55 * np.exp(-(np.log(10.0) ** 2) / 2.0)) > 0.2
    assert rt.max_packing_fraction(0.55, 1.0, 0.1) == pytest.approx(val, rel=1e-12)  # symmetric
    with pytest.raises(ValueError, match='positive'):
        rt.max_packing_fraction(0.55, 1.0, 0.0)


@pytest.mark.physics_invariant
def test_poisson_ratio_round_trip():
    """mu/kappa from nu and nu from (kappa, mu) invert each other; nu = 1/4 gives mu/kappa = 3/5."""
    assert float(rt.mu_over_kappa(0.25)) == pytest.approx(0.6, rel=1e-12)
    nu = np.array([0.1, 0.3, 0.45])
    np.testing.assert_allclose(rt.poisson_ratio(1.0, rt.mu_over_kappa(nu)), nu, rtol=1e-12)
    assert float(rt.mu_over_kappa(0.5)) == pytest.approx(0.0, abs=1e-15)  # incompressible limit
