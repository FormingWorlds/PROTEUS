"""Tests for ``proteus.interior_energetics.rheology_mush``.

Source under test: the melt volume fraction, the Hashin-Shtrikman bounds, the framework
critical fraction from the config, the mush shear, drained bulk and grain bulk moduli, and
the update of the interior arrays.

Invariants checked:
- volume fraction: 0 and 1 at the end members, equal to the mass fraction for equal
  densities, and inverted by the mass-fraction formula;
- the unrelaxed shear modulus lies inside its HS bounds, falls monotonically with melt
  fraction, equals mu_sol for the solid and G_inf for the melt;
- the drained bulk modulus is kappa_sol for the solid, at most the empty-pore HS bound,
  and 0 from the framework collapse on.

See ``docs/How-to/testing.md``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from proteus.interior_energetics import rheology_mush as rm
from proteus.interior_energetics import rheology_tables as rt

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

SMALL = rt.ScsTableGrid(n_nu=6, n_kliq=4, n_gliq=4, n_v=101)
# 25 GPa-like end members: nu_sol = 0.32, kappa_liq / kappa_sol = 0.5, G_inf / kappa_sol = 0.033
EM = {
    'rho_sol': 4100.0,
    'rho_liq': 3300.0,
    'kappa_sol': 300e9,
    'mu_sol': 120e9,
    'kappa_liq': 150e9,
}
G_INF, V_CRIT = 1e10, 1.0 - 0.8 * 0.55


@pytest.fixture
def tables(tmp_path):
    """Small SCS tables built in a temporary cache."""
    rt._load_cached.cache_clear()
    yield rt.load_tables(SMALL, tmp_path)
    rt._load_cached.cache_clear()


def _em(n):
    return {k: np.full(n, v) for k, v in EM.items()}


@pytest.mark.physics_invariant
def test_volume_fraction_limits_and_inverse():
    """v(phi) is 0 and 1 at the end members, phi for equal densities, and inverts exactly."""
    phi = np.array([0.0, 0.2, 0.5, 1.0])
    v = rm.volume_fraction(phi, 4100.0, 3300.0)
    assert v[0] == pytest.approx(0.0) and v[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(rm.volume_fraction(phi, 3500.0, 3500.0), phi, rtol=1e-12)
    back = v * 3300.0 / (v * 3300.0 + (1 - v) * 4100.0)
    np.testing.assert_allclose(back, phi, rtol=1e-12)
    assert np.all(v[1:-1] > phi[1:-1])  # a lighter melt occupies more volume than mass


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_hs_shear_bounds_analytical_limits():
    """Hashin and Shtrikman (1963): equal shear moduli make both bounds that modulus, and the
    bounds lie inside the Voigt and Reuss averages (Hill 1952).

    At v = 0.5 with mu_liq = mu_sol / 12 the Reuss average (0.154 mu_sol) falls well below
    the HS lower bound, so a Reuss-for-HS substitution is excluded.
    """
    v = np.array([0.1, 0.5, 0.9])
    lo, hi = rm.hs_shear_bounds(v, 300e9, 120e9, 150e9, 120e9 * (1 - 1e-9))
    np.testing.assert_allclose(lo, 120e9, rtol=1e-8)
    np.testing.assert_allclose(hi, 120e9, rtol=1e-8)
    lo, hi = rm.hs_shear_bounds(v, 300e9, 120e9, 150e9, 10e9)
    voigt = (1 - v) * 120e9 + v * 10e9
    reuss = 1.0 / ((1 - v) / 120e9 + v / 10e9)
    assert np.all(lo > reuss) and np.all(hi < voigt)
    assert lo[1] / reuss[1] > 1.2  # the HS lower bound differs from Reuss by more than 20 %


@pytest.mark.physics_invariant
def test_hs_bounds_bracket_and_meet_at_end_members():
    """HS shear bounds: lower <= upper, both equal mu_sol at v = 0 and mu_liq at v = 1."""
    v = np.linspace(0.0, 1.0, 11)
    lo, hi = rm.hs_shear_bounds(v, 300e9, 120e9, 150e9, G_INF)
    assert np.all(lo <= hi * (1 + 1e-12))
    assert (lo[0], hi[0]) == (pytest.approx(120e9), pytest.approx(120e9))
    assert (lo[-1], hi[-1]) == (pytest.approx(G_INF), pytest.approx(G_INF))
    kd, mud = rm.hs_void_upper(np.array([0.0, 1.0]), 300e9, 120e9)
    assert (kd[0], mud[0]) == (pytest.approx(300e9), pytest.approx(120e9))
    assert (kd[1], mud[1]) == (pytest.approx(0.0, abs=1e-3), pytest.approx(0.0, abs=1e-3))


@pytest.mark.physics_invariant
def test_mush_moduli_limits_bounds_and_monotonic_fall(tables):
    """Unrelaxed shear inside its HS bounds and falling with phi; drained bulk 0 past v_crit."""
    phi = np.linspace(0.0, 1.0, 41)
    out = rm.mush_moduli(phi, _em(phi.size), V_CRIT, G_INF, tables)
    v = rm.volume_fraction(phi, EM['rho_sol'], EM['rho_liq'])
    lo, hi = rm.hs_shear_bounds(v, EM['kappa_sol'], EM['mu_sol'], EM['kappa_liq'], G_INF)
    assert np.all(out['shear'] >= lo * (1 - 1e-9)) and np.all(out['shear'] <= hi * (1 + 1e-9))
    assert np.all(np.diff(out['shear']) <= 1e-6 * EM['mu_sol'])
    assert out['shear'][0] == pytest.approx(EM['mu_sol'], rel=1e-9)
    assert out['shear'][-1] == pytest.approx(G_INF, rel=1e-9)
    assert out['bulkd'][0] == pytest.approx(EM['kappa_sol'], rel=1e-9)
    assert np.all(out['bulkd'][v >= V_CRIT] == 0.0)
    assert np.all(
        out['bulkd'] <= rm.hs_void_upper(v, EM['kappa_sol'], EM['mu_sol'])[0] * (1 + 1e-9)
    )
    np.testing.assert_allclose(out['bulk'], EM['kappa_sol'])  # grain modulus for Obliqua


@pytest.mark.physics_invariant
def test_shear_beyond_collapse_is_hs_lower_bound(tables):
    """Past the framework collapse the unrelaxed modulus is the HS lower bound, not 0.

    A suspension of crystals in melt keeps the melt's unrelaxed rigidity G_inf; a relaxed
    construction would return 0 there.
    """
    phi = np.array([0.7, 0.9])
    out = rm.mush_moduli(phi, _em(2), V_CRIT, G_INF, tables)
    v = rm.volume_fraction(phi, EM['rho_sol'], EM['rho_liq'])
    lo, _ = rm.hs_shear_bounds(v, EM['kappa_sol'], EM['mu_sol'], EM['kappa_liq'], G_INF)
    np.testing.assert_allclose(out['shear'], lo, rtol=1e-9)
    assert np.all(out['shear'] > G_INF)


def test_mush_moduli_rejects_missing_end_member(tables):
    """A NaN or non-positive end member raises instead of propagating."""
    em = _em(3)
    em['kappa_liq'][1] = np.nan
    with pytest.raises(ValueError, match='kappa_liq'):
        rm.mush_moduli(np.array([0.1, 0.2, 0.3]), em, V_CRIT, G_INF, tables)
    em = _em(3)
    em['mu_sol'][0] = 0.0
    with pytest.raises(ValueError, match='mu_sol'):
        rm.mush_moduli(np.array([0.1, 0.2, 0.3]), em, V_CRIT, G_INF, tables)


def test_critical_volume_fraction_from_config():
    """v_crit = 1 - y phi_m with phi_m = phi_m1 for equant crystals (0.56 for the defaults)."""
    from proteus.config._interior import Rheology

    cfg = types.SimpleNamespace(interior_energetics=types.SimpleNamespace(rheology=Rheology()))
    assert rm.critical_volume_fraction(cfg) == pytest.approx(1.0 - 0.8 * 0.55)
    cfg.interior_energetics.rheology = Rheology(framework_fraction=1.0, packing_fraction=0.6)
    assert rm.critical_volume_fraction(cfg) == pytest.approx(0.4)


def test_update_sets_interior_arrays(tables, monkeypatch):
    """update_mush_rheology fills shear, grain bulk and drained bulk from the layer state."""
    from proteus.config._interior import Rheology

    interior = types.SimpleNamespace(
        phi=np.array([0.0, 0.3, 1.0]), pres=np.full(3, 25e9), temp=np.full(3, 3000.0)
    )
    cfg = types.SimpleNamespace(interior_energetics=types.SimpleNamespace(rheology=Rheology()))
    monkeypatch.setattr(rm.rem, 'paleos_mgsio3_path', lambda: 'unused')
    monkeypatch.setattr(rm.rem, 'load_endmembers', lambda path: 'endmembers')
    monkeypatch.setattr(rm, 'layer_endmembers', lambda em, p, t: _em(len(p)))
    monkeypatch.setattr(rm.rt, 'load_tables', lambda: tables)
    rm.update_mush_rheology(interior, cfg)
    assert interior.shear[0] == pytest.approx(EM['mu_sol'], rel=1e-9)
    assert interior.shear[-1] == pytest.approx(G_INF, rel=1e-9)
    assert (
        interior.bulkd[0] == pytest.approx(EM['kappa_sol'], rel=1e-9)
        and interior.bulkd[-1] == 0.0
    )
    np.testing.assert_allclose(interior.bulk, EM['kappa_sol'])
