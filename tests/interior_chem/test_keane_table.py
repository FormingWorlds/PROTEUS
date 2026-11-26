"""Keane + Deng-thermal melt EOS and its in-memory table
(``src/proteus/interior_chem/keane_table.py``).

Invariants and contract clauses exercised here:

* Analytic identities of the fitted form: on the 3000 K isotherm the pressure
  is zero at V0 and the bulk modulus -V dP/dV there is K0; the thermal
  pressure at V0 is (a - b + c)/1000 (T - 3000 K).
* The liquid-volume solve inverts P(V, T) and falls with pressure; above
  T_b(1 bar) = 4206 K the melts have no liquid at 1 bar.
* The table: zero at 1 bar on every isotherm, the FPMD coverage of Deng et
  al. (2020) Fig. 1, and bilinear interpolation within 0.05 log10 fO2 of
  direct evaluation off the grid nodes.
* Error contract: an integration path outside the liquid region raises.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import R
from scipy.interpolate import RegularGridInterpolator

from proteus.interior_chem import keane_table as K

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_MELT_IDS = ['Fe2+ melt', 'Fe3+ melt']


@pytest.fixture(scope='module')
def table():
    """The table, built once for this module."""
    return K.build_table()


@pytest.fixture(scope='module')
def melts():
    """The two melts and their hottest-liquid temperature T_b(P)."""
    red, ox = K._Melt(K.PARAMS[K.RED]), K._Melt(K.PARAMS[K.OX])
    return red, ox, K._hottest_liquid_T([red, ox])


@pytest.mark.physics_invariant
@pytest.mark.parametrize('melt', [K.RED, K.OX], ids=_MELT_IDS)
def test_cold_curve_has_zero_pressure_and_bulk_modulus_k0_at_v0(melt):
    """At V0 on the 3000 K isotherm the Keane form gives P = 0 and
    -V dP/dV = K0 (central difference, step 1e-6 V0)."""
    params = K.PARAMS[melt]
    V0, K0 = params[0], params[1]
    h = 1e-6 * V0
    dPdV = (K.pressure(V0 + h, K.T_REF, params) - K.pressure(V0 - h, K.T_REF, params)) / (2 * h)

    assert K.pressure(V0, K.T_REF, params) == pytest.approx(0.0, abs=1e-12)
    assert -V0 * dPdV == pytest.approx(K0, rel=1e-6)
    # Guard: K0' in place of K0 (both stored next to each other) misses by > 10 GPa.
    assert abs(-V0 * dPdV - params[2]) > 10.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('melt', [K.RED, K.OX], ids=_MELT_IDS)
def test_thermal_pressure_at_v0_is_the_deng_polynomial(melt):
    """At x = V/V0 = 1 the thermal term is (a - b + c)/1000 per K away from
    3000 K, linear in T: 1.38 GPa (Fe2+) and 1.30 GPa (Fe3+) at 4000 K."""
    params = K.PARAMS[melt]
    a, b, c = params[4:]
    for T in (1500.0, 4000.0, 9000.0):
        expected = (a - b + c) / 1000.0 * (T - K.T_REF)
        assert K.pressure(params[0], T, params) == pytest.approx(expected, rel=1e-12)
    # Guard: the sign of b flipped would give > 100 GPa at 4000 K instead of about 1.3 GPa.
    assert abs((a + b + c) - (a - b + c)) > 100.0


@pytest.mark.physics_invariant
def test_liquid_volume_inverts_the_eos_and_falls_with_pressure(melts):
    """V(P, T) returns the dense root: P(V(P, T), T) = P to 1e-6 relative, and V
    decreases monotonically with pressure, on the 3000 K isotherm from 1 GPa and on
    the 12000 K isotherm from 50 GPa (below it that isotherm has no liquid) to 4 TPa."""
    red, ox, _ = melts
    for m in (red, ox):
        for T, P_lo in ((3000.0, 1.0), (12000.0, 50.0)):
            P = np.geomspace(P_lo, 4000.0, 60)
            V = m.volume(P, np.full_like(P, T))
            assert np.all(np.isfinite(V))
            np.testing.assert_allclose(K.pressure(V, T, m.params), P, rtol=1e-6)
            assert np.all(np.diff(V) < 0.0)


@pytest.mark.physics_invariant
def test_no_liquid_at_one_bar_above_the_hottest_liquid_temperature(melts):
    """The fit has a liquid at 1 bar only below T_b(1 bar) = 4206 K: at 10000 K
    the volume is NaN, at 3000 K it is finite and within 10 % of V0."""
    red, ox, t_b = melts

    assert float(t_b(K.P_MIN_GPA)) == pytest.approx(4206.0, abs=10.0)
    assert np.isnan(ox.volume(K.P_MIN_GPA, 10000.0))
    v = float(red.volume(K.P_MIN_GPA, 3000.0))
    assert abs(v / red.params[0] - 1.0) < 0.1


@pytest.mark.physics_invariant
def test_table_starts_at_zero_and_covers_the_stated_range(table):
    """The integral is taken from 1 bar, so the first pressure column is zero on
    every isotherm; the grid spans 1 bar - 4 TPa and 1000 - 20000 K at 100 points
    per decade (762 x 132)."""
    I = table['int_dV_dP_ox']

    assert I.shape == (132, 762)
    assert np.all(np.isfinite(I))
    np.testing.assert_allclose(I[:, 0], 0.0, atol=1e-9)
    assert 10 ** table['log10_P_Pa'][[0, -1]] == pytest.approx([1e5, 4e12], rel=1e-12)
    assert 10 ** table['log10_T_K'][[0, -1]] == pytest.approx([1000.0, 20000.0], rel=1e-12)


@pytest.mark.parametrize(
    'T, P_gpa, inside',
    [
        (3000.0, 70.0, True),
        (3000.0, 80.0, False),
        (3900.0, 90.0, True),
        (2000.0, 20.0, False),
        (5000.0, 10.0, False),
    ],
    ids=[
        '3000 K below 74 GPa',
        '3000 K above',
        '3900 K below 94 GPa',
        '2000 K above 17 GPa',
        'hotter than the FPMD',
    ],
)
def test_fpmd_mask_follows_the_deepest_fpmd_point_of_each_isotherm(table, T, P_gpa, inside):
    """in_fpmd marks 2000-4000 K up to the deepest Fe3+ FPMD pressure of the nearest
    Deng et al. (2020) isotherm (17, 74, 94 GPa)."""
    mask = K.interpolators(table)['in_fpmd']
    value = float(mask([[np.log10(T), np.log10(P_gpa * 1e9)]])[0])

    assert value in (0.0, 1.0)
    assert bool(value) is inside


@pytest.mark.physics_invariant
def test_bilinear_table_matches_direct_evaluation_off_the_grid(table, melts):
    """At 40 seeded random points off the nodes the interpolated integral is within
    0.01 in I/(R T ln10) of the direct one, i.e. 0.05 in log10 fO2 with the
    Hirschmann (2022) coefficient 0.193; the median is below 1e-3. Guard: the
    nearest node instead of bilinear interpolation has a median error 10 times larger."""
    red, ox, t_b = melts
    rng = np.random.default_rng(1)
    lt = rng.uniform(3.0, np.log10(2.0e4), 40)
    lp = rng.uniform(5.0, np.log10(4.0e12), 40)
    direct = np.array(
        [
            K._oxidation_integral(red, ox, t_b, 10**t, np.array([10**p / 1e9]))[0]
            for t, p in zip(lt, lp)
        ]
    )
    q = np.column_stack([lt, lp])
    scale = R * 10**lt * np.log(10.0)
    err = np.abs(K.interpolators(table)['interp'](q) - direct) / scale
    nearest = RegularGridInterpolator(
        (table['log10_T_K'], table['log10_P_Pa']), table['int_dV_dP_ox'], method='nearest'
    )
    err_nearest = np.abs(nearest(q) - direct) / scale

    assert err.max() < 0.01
    assert np.median(err) < 1e-3
    assert np.median(err_nearest) > 10.0 * np.median(err)


def test_path_outside_the_liquid_region_raises(monkeypatch, melts):
    """Without the margin below T_b(P) a 10000 K path crosses pressures where the
    melts have no liquid; the integral raises instead of returning NaN. With the
    margin the same isotherm integrates."""
    red, ox, t_b = melts
    P = np.array([1.0, 100.0])
    ok = K._oxidation_integral(red, ox, t_b, 10000.0, P)
    monkeypatch.setattr(K, 'T_B_MARGIN', -1.0e5)

    assert np.all(np.isfinite(ok))
    with pytest.raises(RuntimeError, match='undefined dV on the 10000 K'):
        K._oxidation_integral(red, ox, t_b, 10000.0, P)
