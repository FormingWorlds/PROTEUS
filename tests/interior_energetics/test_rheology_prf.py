"""Tests for ``proteus.interior_energetics.rheology_prf``.

Source under test: the Kervazo et al. (2021) melt-fraction fits and their update of the
interior arrays; the melt volume fraction, the Hashin-Shtrikman bounds, the bulk viscosity,
the undrained bulk modulus, and the Nur et al. (1998) critical-porosity model with its update
of the interior arrays.

Invariants checked:
- Kervazo fits: the Costa et al. (2009) form with the Table 3 critical crystal fraction
  phi_* (phist = 1 - phi_*), the melt reference values for pure melt, positivity and
  monotonic fall with melt fraction;
- volume fraction: 0 and 1 at the end members, equal to the mass fraction for equal
  densities, and inverted by the mass-fraction formula;
- Nur model: the modified Voigt formula of the paper, its Gassmann saturation, the HS
  bounds and the monotonic fall of the frame moduli, the suspension shear modulus at the HS
  lower bound of the melt;
- undrained modulus: the grain modulus without melt, the Reuss average for an empty frame,
  and the melt volume (not mass) fraction as porosity.

See ``docs/How-to/testing.md``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from proteus.interior_energetics import rheology_prf as rm

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# 25 GPa-like end members: nu_sol = 0.32, kappa_liq / kappa_sol = 0.5, G_inf / kappa_sol = 0.033
EM = {
    'rho_sol': 4100.0,
    'rho_liq': 3300.0,
    'kappa_sol': 300e9,
    'mu_sol': 120e9,
    'kappa_liq': 150e9,
    'mu_liq': 1e10,
}
G_INF = EM['mu_liq']


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
@pytest.mark.reference_pinned
def test_bulk_viscosity_prudhomme_bird_form():
    """zeta = (4/3) eta (1 - v) / v (Prud'homme and Bird 1978; Simpson et al. 2010, eq. 11b).

    At v = 0.25 this is 4 eta; Taylor's dilute form (4/3) eta / v gives 5.33 eta, 33 % more.
    The value vanishes for pure melt, stays finite in the solid and lies below the
    empty-pore HS bound (4/3) eta_sol (1 - v) / v for any eta below eta_sol.
    """
    eta = 1e18
    assert float(rm.bulk_viscosity(eta, 0.25)) == pytest.approx(4.0 * eta, rel=1e-12)
    assert abs(4.0 / 3.0 * eta / 0.25 - 4.0 * eta) / (4.0 * eta) > 0.3
    assert float(rm.bulk_viscosity(eta, 1.0)) == pytest.approx(0.0, abs=1e-30)
    solid = float(rm.bulk_viscosity(eta, 0.0))
    assert np.isfinite(solid) and solid > 1e14 * eta
    v = np.linspace(0.05, 0.95, 10)
    assert np.all(rm.bulk_viscosity(eta, v) <= 4.0 / 3.0 * 1e22 * (1 - v) / v)


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


# ============================================================================
# Kervazo et al. (2021) fits
# ============================================================================

# Kervazo et al. (2021, A&A 650, A72) Table 3: (dotl, delta, xi, gamma, phi_*), phi_* the
# critical crystal fraction
KERVAZO_TABLE3 = {
    'visc': (1.0, 25.7, 1.17e-9, 5.0, 0.569),
    'shear': (10.0, 2.10, 7.08e-7, 5.0, 0.597),
    'bulk': (1e9, 2.62, 0.102, 5.0, 0.712),
}


def _costa(crystal, dotl, delta, xi, gamma, phi_c):
    """Costa et al. (2009) relative law in the crystal fraction, written out independently."""
    from scipy.special import erf

    theta = crystal / phi_c
    f = (1.0 - xi) * erf(np.sqrt(np.pi) / (2.0 * (1.0 - xi)) * theta * (1.0 + theta**gamma))
    return dotl * (1.0 + theta**delta) / (1.0 - f) ** (2.5 * phi_c)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('which', ['visc', 'shear', 'bulk'])
def test_kervazo_table3_critical_crystal_fraction(which):
    """Kervazo et al. (2021) Table 3 enters as the critical crystal fraction phi_*.

    The law takes the melt fraction, so it must equal the Costa et al. (2009) form in the
    crystal fraction 1 - phi with critical fraction phi_*. For pure melt it returns the
    melt reference value (the analytical limit Theta = 0). Reading phi_* as a critical melt
    fraction instead moves the value at phi = 0.5 by a factor of at least 2.6 for every
    channel, far outside the 1e-10 tolerance.
    """
    dotl, delta, xi, gamma, phi_c = KERVAZO_TABLE3[which]
    par = {'visc': rm.par_visc, 'shear': rm.par_shear, 'bulk': rm.par_bulk}[which]
    assert par.phist == pytest.approx(1.0 - phi_c, rel=1e-12)
    phi = np.array([0.0, 0.1, 0.3, 0.5, 0.8])
    expected = _costa(1.0 - phi, dotl, delta, xi, gamma, phi_c)
    np.testing.assert_allclose(rm.eval_rheoparam(phi, which), expected, rtol=1e-10)
    assert rm.eval_rheoparam(1.0, which) == pytest.approx(dotl, rel=1e-12)
    swapped = _costa(0.5, dotl, delta, xi, gamma, 1.0 - phi_c)
    ratio = swapped / rm.eval_rheoparam(0.5, which)
    assert max(ratio, 1.0 / ratio) > 2.5


@pytest.mark.physics_invariant
def test_kervazo_solid_branch_and_monotonic_fall():
    """All channels are positive, fall with melt fraction and span many decades for eta.

    At phi = 0 every channel lies above its melt reference; eta(0) of about 1e19 Pa s
    excludes a regression that returns the melt value 1 Pa s.
    """
    phis = np.linspace(0.0, 1.0, 21)
    for which in ('visc', 'shear', 'bulk'):
        vals = rm.eval_rheoparam(phis, which)
        assert np.all(vals > 0.0)
        assert np.all(np.diff(vals) <= 0.0), f'{which} must not rise with melt fraction'
    assert rm.eval_rheoparam(0.0, 'visc') > 1e18
    assert rm.eval_rheoparam(0.0, 'shear') > 100.0 * rm.par_shear.dotl
    assert rm.eval_rheoparam(0.0, 'bulk') > rm.par_bulk.dotl


def test_kervazo_invalid_channel_raises():
    """An unknown or empty channel name raises; the valid channels stay finite."""
    with pytest.raises(ValueError, match='Invalid rheological parameter'):
        rm.eval_rheoparam(0.3, 'this_is_not_a_channel')
    with pytest.raises(ValueError):
        rm.eval_rheoparam(0.3, '')
    for which in ('visc', 'shear', 'bulk'):
        val = rm.eval_rheoparam(0.3, which)
        assert np.isfinite(val) and val > 0.0


@pytest.mark.physics_invariant
def test_update_rheology_kervazo_sets_interior_arrays():
    """The Kervazo profile sets shear and bulk per layer, bulk viscosity eta / phi, no bulkd.

    The viscosity array is left as given; in a solid layer (phi = 0) the bulk viscosity
    stays finite through the 1e-15 floor.
    """
    phi = np.array([0.0, 0.2, 0.6, 1.0])
    visc = np.array([1e21, 1e17, 1e3, 1.0])
    interior = types.SimpleNamespace(
        phi=phi, visc=visc.copy(), shear=np.zeros(4), bulk=np.zeros(4),
        bulkd=np.ones(4), bulk_liq=np.ones(4),
    )  # fmt: skip
    rm.update_rheology_kervazo(interior)
    assert interior.bulkd is None and interior.bulk_liq is None
    np.testing.assert_allclose(interior.shear, rm.eval_rheoparam(phi, 'shear'), rtol=1e-12)
    np.testing.assert_allclose(interior.bulk, rm.eval_rheoparam(phi, 'bulk'), rtol=1e-12)
    np.testing.assert_allclose(interior.visc, visc)
    np.testing.assert_allclose(interior.bulk_visc[1:], visc[1:] / phi[1:], rtol=1e-12)
    assert np.isfinite(interior.bulk_visc[0]) and interior.bulk_visc[0] > 1e14 * visc[0]
    assert interior.shear[-1] == pytest.approx(rm.par_shear.dotl, rel=1e-12)


@pytest.mark.physics_invariant
def test_update_viscosity_kervazo_follows_melt_fraction():
    """The dummy-interior viscosity falls from the solid value to the 1 Pa s melt value."""
    interior = types.SimpleNamespace(phi=np.array([0.0, 0.3, 1.0]), visc=np.zeros(3))
    rm.update_viscosity_kervazo(interior)
    np.testing.assert_allclose(
        interior.visc, rm.eval_rheoparam(interior.phi, 'visc'), rtol=1e-12
    )
    assert interior.visc[0] / interior.visc[1] > 1e3
    assert interior.visc[-1] == pytest.approx(1.0, rel=1e-12)


# ============================================================================
# Nur et al. (1998) critical-porosity model
# ============================================================================


def _nur_inputs(v, v_crit=0.45):
    """Melt mass fractions for the volume fractions ``v`` and the end members of ``EM``."""
    phi = v * EM['rho_liq'] / (v * EM['rho_liq'] + (1.0 - v) * EM['rho_sol'])
    return phi, _em(len(v))


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_nur_modified_voigt_matches_paper_formula():
    """Nur et al. (1998, p. 361): K_eff = (1 - v/v_c) K_s + (v/v_c) K_c, K_c = Reuss at v_c.

    At v = 0.15, v_c = 0.45 the bulk modulus is 2/3 K_s + 1/3 K_c; the plain Voigt
    average (1 - v) K_s + v K_l, which ignores the critical porosity, is 3 % higher. The
    shear modulus of a melt of vanishing rigidity is (1 - v/v_c) mu_s, to 1e-8 relative.
    """
    v = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 1.0])
    phi, em = _nur_inputs(v)
    em['mu_liq'] = np.full(v.size, 1e-3)
    out = rm.nur_effective_moduli(phi, em, 0.45)
    k_c = 1.0 / (0.55 / EM['kappa_sol'] + 0.45 / EM['kappa_liq'])
    expect_k = np.where(
        v <= 0.45,
        (1 - v / 0.45) * EM['kappa_sol'] + v / 0.45 * k_c,
        1.0 / ((1 - v) / EM['kappa_sol'] + v / EM['kappa_liq']),
    )
    np.testing.assert_allclose(out['bulk'], expect_k, rtol=1e-12)
    np.testing.assert_allclose(
        out['shear'][:4], np.maximum(1 - v[:4] / 0.45, 0.0) * EM['mu_sol'], rtol=1e-6, atol=1.0
    )
    voigt = (1 - v[1]) * EM['kappa_sol'] + v[1] * EM['kappa_liq']
    assert voigt / out['bulk'][1] > 1.03
    assert out['bulk'][3] == pytest.approx(k_c, rel=1e-12)  # continuous at v_c
    assert out['shear'][-1] < 1e-2 * EM['mu_sol']


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_gassmann_of_nur_frame_reproduces_saturated_modified_voigt():
    """Biot-Gassmann saturation of the dry Nur frame returns Nur's saturated curve exactly.

    With K_d = (1 - x) K_s, x = v/v_c, alpha = x and S = v/K_l + (alpha - v)/K_s, the
    Gassmann modulus K_d + alpha^2/S equals (1 - x) K_s + x K_c algebraically. A frame
    that is not the dry limit of the modified Voigt bound (for instance one capped below
    the HS bound at v_c = 0.3) would differ; at v_c = 0.3 the HS cap is inactive.
    """
    v = np.array([0.02, 0.1, 0.2, 0.29])
    phi, em = _nur_inputs(v)
    out = rm.nur_framework_moduli(phi, em, 0.3)
    alpha = 1.0 - out['bulkd'] / EM['kappa_sol']
    assert np.all(alpha - v >= 0.0)  # Biot storativity positive
    storage = v / EM['kappa_liq'] + (alpha - v) / EM['kappa_sol']
    k_sat = out['bulkd'] + alpha**2 / storage
    ref = rm.nur_effective_moduli(phi, em, 0.3)['bulk']
    np.testing.assert_allclose(k_sat, ref, rtol=1e-10)
    assert abs(k_sat[1] - ((1 - v[1]) * EM['kappa_sol'] + v[1] * EM['kappa_liq'])) > 1e9


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_undrained_bulk_modulus_limits_and_volume_porosity():
    """Gassmann (1951) limits: K_s without melt, Reuss for an empty frame, K_d below K_u <= K_s.

    The melt volume fraction enters, not the mass fraction: for v = 0.6 of a melt lighter than
    the solid the mass fraction is lower, and using it gives a stiffer modulus by several per
    cent, outside the 1e-12 tolerance. A frame as stiff as the grain stays at K_s.
    """
    ks, kl = EM['kappa_sol'], EM['kappa_liq']
    v = np.array([0.0, 0.2, 0.6, 1.0])
    # drained frame of the Nur law at v_c = 0.45 (zero beyond the collapse)
    kd = np.maximum(1 - v / 0.45, 0.0) * ks
    ku = rm.undrained_bulk_modulus(kd, ks, kl, v)
    reuss = 1.0 / ((1 - v) / ks + v / kl)
    assert ku[0] == pytest.approx(ks, rel=1e-12)
    np.testing.assert_allclose(ku[2:], reuss[2:], rtol=1e-12)  # empty frame: Reuss, kl at v = 1
    assert ku[3] == pytest.approx(kl, rel=1e-12)
    assert np.all(ku >= kd - 1.0) and np.all(ku <= ks * (1 + 1e-12))
    assert ku[1] > kd[1] + 1e9  # the melt stiffens the frame noticeably
    # mass fraction of the melt-rich layer, and the stiffer (wrong) modulus it would give
    phi = v[2] * EM['rho_liq'] / (v[2] * EM['rho_liq'] + (1 - v[2]) * EM['rho_sol'])
    assert phi < v[2]
    wrong = rm.undrained_bulk_modulus(0.0, ks, kl, phi)
    assert wrong / ku[2] - 1 > 1e-2
    # no undrained stiffening when the frame is the grain (alpha = 0), even at finite melt content
    assert rm.undrained_bulk_modulus(ks, ks, kl, 0.3) == pytest.approx(ks, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('v_crit', [0.3, 0.45, 0.6], ids=['vc_0.3', 'vc_0.45', 'vc_0.6'])
def test_nur_frame_inside_hashin_shtrikman_bounds_and_monotonic(v_crit):
    """Frame moduli respect the HS bounds (the scaled linear law alone exceeds the empty-pore
    upper bound by 3 % at v_c = 0.45 and 15 % at 0.6), fall with melt, and vanish past v_c.
    """
    v = np.linspace(0.0, 1.0, 101)
    phi, em = _nur_inputs(v)
    out = rm.nur_framework_moduli(phi, em, v_crit)
    k_hi, mu_hi_void = rm.hs_void_upper(v, EM['kappa_sol'], EM['mu_sol'])
    lo, hi = rm.hs_shear_bounds(v, EM['kappa_sol'], EM['mu_sol'], EM['kappa_liq'], EM['mu_liq'])
    assert np.all(out['bulkd'] <= k_hi * (1 + 1e-12)) and np.all(out['bulkd'] >= 0.0)
    assert np.all(out['shear'] >= lo * (1 - 1e-12)) and np.all(out['shear'] <= hi * (1 + 1e-12))
    assert np.all(np.diff(out['bulkd']) <= 1e-9 * EM['kappa_sol'])
    assert np.all(np.diff(out['shear']) <= 1e-9 * EM['mu_sol'])
    assert np.all(out['bulkd'][v >= v_crit] == 0.0)
    assert out['bulkd'][0] == pytest.approx(EM['kappa_sol'], rel=1e-12)
    assert out['shear'][0] == pytest.approx(EM['mu_sol'], rel=1e-12)
    uncapped = (1 - v / v_crit).clip(0) * EM['kappa_sol']
    assert (v_crit < 0.4) or np.any(
        uncapped > k_hi * (1 + 1e-6)
    )  # the cap is what holds the bound


@pytest.mark.physics_invariant
def test_nur_suspension_shear_is_hs_lower_bound_of_the_melt():
    """Beyond v_c the shear modulus is the HS lower bound of the melt (not Reuss, which is
    22 % lower at v = 0.7 and 32 % at v_c), is continuous at v_c, and follows the melt modulus.
    """
    v = np.array([0.449999, 0.450001, 0.7])
    phi, em = _nur_inputs(v)
    out = rm.nur_framework_moduli(phi, em, 0.45)
    lo, _ = rm.hs_shear_bounds(v, EM['kappa_sol'], EM['mu_sol'], EM['kappa_liq'], EM['mu_liq'])
    np.testing.assert_allclose(out['shear'][1:], lo[1:], rtol=1e-12)
    assert out['shear'][0] == pytest.approx(out['shear'][1], rel=1e-4)
    reuss = 1.0 / ((1 - v[2]) / EM['mu_sol'] + v[2] / EM['mu_liq'])
    assert out['shear'][2] > 1.25 * reuss  # Reuss is 22 % below the bound at v = 0.7
    em_soft = {k: np.array(x) for k, x in em.items()}
    em_soft['mu_liq'] = np.full(3, 10.0)
    soft = rm.nur_framework_moduli(phi, em_soft, 0.45)
    assert soft['shear'][2] < 1e-6 * out['shear'][2]  # Kervazo-like 10 Pa melt: soft suspension


def test_nur_rejects_bad_critical_fraction_and_end_members():
    """v_crit outside (0, 1) and a missing or non-positive end member raise ValueError."""
    phi, em = _nur_inputs(np.array([0.1, 0.2]))
    good = rm.nur_framework_moduli(
        phi, em, 0.45
    )  # the same inputs are accepted for a valid v_c
    assert np.all(good['bulkd'] > 0.0) and np.all(good['bulkd'] < EM['kappa_sol'])
    assert np.all(good['shear'] > EM['mu_liq'])
    for bad in (0.0, 1.0, np.nan, 1.5):
        with pytest.raises(ValueError, match='v_crit'):
            rm.nur_framework_moduli(phi, em, bad)
        with pytest.raises(ValueError, match='v_crit'):
            rm.nur_effective_moduli(phi, em, bad)
    em['mu_liq'][0] = 0.0
    with pytest.raises(ValueError, match='mu_liq'):
        rm.nur_framework_moduli(phi, em, 0.45)


def test_nur_critical_volume_fraction_needs_a_configured_value():
    """The critical fraction is required: no sandstone default is assumed for the mush."""
    from proteus.config._interior import Rheology

    def _cfg(**kw):
        rh = types.SimpleNamespace(**kw) if kw else Rheology()
        return types.SimpleNamespace(interior_energetics=types.SimpleNamespace(rheology=rh))

    assert rm.nur_critical_volume_fraction(_cfg(nur_critical_melt_volume_fraction=0.4)) == (
        pytest.approx(0.4)
    )
    assert rm.nur_critical_volume_fraction(_cfg(nur_critical_melt_volume_fraction=0.05)) == (
        pytest.approx(0.05)
    )  # cracked igneous rock, Nur et al. Table 1
    with pytest.raises(ValueError, match='nur_critical_melt_volume_fraction'):
        rm.nur_critical_volume_fraction(_cfg())
    with pytest.raises(ValueError, match='strictly between'):
        rm.nur_critical_volume_fraction(_cfg(nur_critical_melt_volume_fraction=1.0))


def test_update_rheology_nur_sets_interior_arrays(monkeypatch):
    """update_rheology_nur fills shear, grain and drained bulk, melt bulk and bulk viscosity.

    The melt shear modulus of the config (3e9 Pa, not the 1e10 Pa default) reaches the
    end-member loader, which keys its cache on it.
    """
    from proteus.config._interior import Rheology

    interior = types.SimpleNamespace(
        phi=np.array([0.0, 0.2, 0.9]),
        pres=np.full(3, 25e9),
        temp=np.full(3, 3000.0),
        visc=np.array([1e21, 1e15, 1e2]),
    )
    cfg = types.SimpleNamespace(
        interior_energetics=types.SimpleNamespace(
            rheology=Rheology(nur_critical_melt_volume_fraction=0.45, melt_shear_modulus=3e9)
        )
    )
    loaded = []
    monkeypatch.setattr(rm.rem, 'paleos_mgsio3_path', lambda: 'unused')
    monkeypatch.setattr(rm.rem, 'load_endmembers', lambda path, g: loaded.append(g) or 'em')
    monkeypatch.setattr(rm, 'layer_endmembers', lambda em, p, t: _em(len(p)))
    rm.update_rheology_nur(interior, cfg)
    assert loaded == [pytest.approx(3e9)]
    v = rm.volume_fraction(interior.phi, EM['rho_sol'], EM['rho_liq'])
    assert interior.bulkd[0] == pytest.approx(EM['kappa_sol'], rel=1e-12)
    assert interior.bulkd[1] < interior.bulkd[0] and interior.bulkd[2] == 0.0
    assert interior.shear[0] == pytest.approx(EM['mu_sol'], rel=1e-12)
    np.testing.assert_allclose(interior.bulk, EM['kappa_sol'])
    np.testing.assert_allclose(interior.bulk_liq, EM['kappa_liq'])
    np.testing.assert_allclose(
        interior.bulk_visc[1:], 4 / 3 * interior.visc[1:] * (1 - v[1:]) / v[1:], rtol=1e-12
    )
    # undrained modulus at the melt volume fraction: grain modulus without melt, Reuss for the
    # collapsed frame of the melt-rich layer
    np.testing.assert_allclose(
        interior.bulk_und,
        rm.undrained_bulk_modulus(interior.bulkd, interior.bulk, interior.bulk_liq, v),
        rtol=1e-12,
    )
    assert interior.bulk_und[0] == pytest.approx(EM['kappa_sol'], rel=1e-12)
    reuss = 1.0 / ((1 - v[2]) / EM['kappa_sol'] + v[2] / EM['kappa_liq'])
    assert interior.bulk_und[2] == pytest.approx(reuss, rel=1e-12)
