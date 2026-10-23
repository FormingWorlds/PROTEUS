"""Pressure term for Fe disproportionation (``src/proteus/interior_chem/eos_deng.py``).

Covers the integral of the reaction volume change for 3FeO = 2FeO1.5 + Fe,
built as in Schaefer et al. (2024) ``deltaGFeOFeO15_Deng.m``: the Deng et al.
(2020) silicate melt equation of state together with liquid FeO and liquid Fe,
each integrated directly on Schaefer's grids.

Invariants and contract clauses exercised here:

* Analytic identity of the fitted form: the fourth-order Birch-Murnaghan
  pressure returns the reference pressure exactly at the reference volume.
* Volume inversion as Schaefer's fsolve finds it: the compressed-branch root
  above P(V0, T), the expanded-branch root (V > V0) below it, and, where the
  fit has no root, the stationary point of the residual.
* Schaefer's integration conventions: the FeO term integrates from zero, so
  at 1 bar the integral is minus V_FeO times 1 bar, not zero.
* Published benchmark: the oxidation integral reproduces the Deng 12.5 mol%
  values of Zhang et al. (2024) Fig. S5 at their experimental conditions.
* Sign structure: positive near the surface, negative at depth.
* The validity contract: points above the temperature ceiling or beyond the
  pressure range Schaefer et al. exercised are reported as invalid.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.interior_chem import eos_deng
from proteus.interior_chem.eos_deng import (
    _B,
    _C,
    _P0,
    _T0,
    _V0,
    P_EXERCISED,
    T_CEILING,
    _bh,
    _bm4_pressure,
    _int_V_FeO,
    _solve_V,
    int_dV_dP,
    int_dV_dP_oxidation,
    p_at_V0,
    table_path,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('endmember', [0, 1], ids=['ferrous endmember', 'ferric endmember'])
def test_birch_murnaghan_returns_the_reference_pressure_at_the_reference_volume(endmember):
    """At the reference volume and reference temperature the fitted equation
    of state reproduces the reference pressure exactly.

    This is an analytic identity of the fourth-order Birch-Murnaghan form with
    the published coefficients, and it is what anchors the foot of the
    pressure integral. A transposed coefficient or a wrong strain definition
    would move it by orders of magnitude.
    """
    P = _bm4_pressure(_V0[endmember], _T0, endmember)

    assert P == pytest.approx(_P0, rel=1e-9)
    # Sign guard: compression is positive. A flipped strain sign lands at -1e-4.
    assert P > 0.0
    # Scale guard: 1e-4 GPa is 1 bar. A bar/GPa confusion would give 1e5.
    assert 1e-6 < P < 1e-2


@pytest.mark.physics_invariant
@pytest.mark.parametrize('endmember', [0, 1], ids=['ferrous endmember', 'ferric endmember'])
def test_thermal_pressure_fit_turns_upward_just_below_the_reference_volume(endmember):
    """The thermal-pressure coefficient decreases with expanding volume only
    up to V/V0 near 0.97, then rises.

    The published coefficients form a parabola in V/V0 with a positive leading
    term, so beyond its minimum the thermal pressure grows on expansion. This
    is why P(V) has a minimum on the expanded branch and why, hot enough, the
    equation has no root at low pressure.
    """
    minimum_at = _B[endmember] / (2.0 * _C[endmember])
    below = float(_bh(np.array(minimum_at - 0.1), endmember))
    at = float(_bh(np.array(minimum_at), endmember))
    above = float(_bh(np.array(minimum_at + 0.4), endmember))

    assert minimum_at == pytest.approx(0.97, abs=0.02)
    assert at < below
    assert at < above
    # The coefficient stays positive throughout, so the sign of the thermal
    # term is set by (T - T0) alone.
    assert at > 0.0


@pytest.mark.physics_invariant
def test_pressure_at_the_reference_volume_is_one_bar_at_the_reference_temperature():
    """P(V0, T) separates the compressed from the expanded branch. At T0 the
    thermal term vanishes, so it is the reference pressure; colder, it moves
    below 1 bar and the expanded branch is never reached.

    Limit input of P0 + BH(1)(T - T0), checked against a direct evaluation of
    the equation of state at V0.
    """
    for i in (0, 1):
        assert p_at_V0(_T0, i) == pytest.approx(_P0, rel=1e-9)
        assert float(p_at_V0(4000.0, i)) == pytest.approx(
            float(_bm4_pressure(_V0[i], 4000.0, i)), rel=1e-12
        )
    assert p_at_V0(2000.0, 0) < 0.0 < p_at_V0(4000.0, 0)
    # Scale guard: about 1.3 GPa at 4000 K, not a fraction of a bar nor 100 GPa.
    assert 0.5 < p_at_V0(4000.0, 0) < 3.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'T, P',
    [(2500.0, 30.0), (4500.0, 120.0)],
    ids=['cold mid-mantle', 'hot lowermost mantle'],
)
def test_compressed_branch_volume_solves_the_equation_of_state(T, P):
    """Above P(V0, T) the solved volume is the compressed-branch root: it
    satisfies the equation of state and lies below V0, and it shrinks as
    pressure rises.
    """
    for i in (0, 1):
        V = _solve_V(P, T, i)
        assert _bm4_pressure(V, T, i) == pytest.approx(P, abs=1e-9)
        assert 0.15 * _V0[i] < V < _V0[i]
        assert _solve_V(1.1 * P, T, i) < V


@pytest.mark.physics_invariant
def test_expanded_branch_root_is_used_below_the_reference_volume_pressure():
    """Above T0 and below P(V0, T) the liquid sits on the expanded branch,
    V > V0, where Schaefer's fsolve converges. The root solves the equation
    of state; a volume capped at V0 (the former splice) would not.
    """
    T, P = 3711.0, 0.3  # magma-ocean base at 80 GPa, first node of its integral
    for i in (0, 1):
        assert P < p_at_V0(T, i)
        V = _solve_V(P, T, i)
        assert _bm4_pressure(V, T, i) == pytest.approx(P, abs=1e-9)
        assert 1.01 * _V0[i] < V < 1.2 * _V0[i]
        # Discrimination: V0 itself misses the target pressure by ~0.6 GPa.
        assert abs(_bm4_pressure(_V0[i], T, i) - P) > 0.3


@pytest.mark.physics_invariant
def test_no_root_case_returns_the_residual_minimum_and_warns_once(monkeypatch, caplog):
    """Hot enough, P(V) beyond V0 never falls to 1 bar. Schaefer's fsolve
    then returns, unconverged, the stationary point of the squared residual,
    the minimum of P(V); that volume is used and the case is logged once.
    """
    import logging

    T, P = 4500.0, _P0
    for i in (0, 1):
        V = _solve_V(P, T, i)
        assert _bm4_pressure(V, T, i) > P + 0.3  # no root: the residual stays large
        h = 1e-4 * _V0[i]
        slope = (_bm4_pressure(V + h, T, i) - _bm4_pressure(V - h, T, i)) / (2 * h)
        curv = (
            _bm4_pressure(V + h, T, i) + _bm4_pressure(V - h, T, i) - 2 * _bm4_pressure(V, T, i)
        )
        assert abs(slope * _V0[i]) < 1e-5  # stationary in P(V)
        assert curv > 0.0  # a minimum, not a maximum
        assert _V0[i] < V < 1.6 * _V0[i]

    monkeypatch.setattr(eos_deng, '_NO_ROOT_WARNED', False)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_chem.eos_deng'):
        int_dV_dP(T, 50.0)
        int_dV_dP(T, 60.0)
    assert sum('no volume root' in r.message for r in caplog.records) == 1


@pytest.mark.physics_invariant
def test_integral_vanishes_at_one_bar_for_every_term():
    """All three terms (melt, liquid FeO, liquid Fe) integrate from 1 bar, as
    in Komabayashi (2014) Eq 1, so at P = 1 bar the total and the oxidation
    integral are zero. Guard: Schaefer's archived FeO integral, which starts
    at P = 0, would leave -V_FeO x 1 bar (about -1.5 J/mol) here.
    """
    T = 3000.0
    total, valid = int_dV_dP(T, _P0)
    ox, _ = int_dV_dP_oxidation(T, _P0)
    from_zero = -(13650.0 + 2.92 * (T - 1673.0)) * 1e-3 * _P0 * 1e3

    assert bool(valid)
    assert float(total) == pytest.approx(0.0, abs=1e-9)
    assert float(ox) == pytest.approx(0.0, abs=1e-9)
    assert abs(from_zero) > 1.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    'name, P, T, fig_s5',
    [
        ('DAC81', 38.0, 3941.0, 72.0e3),
        ('DAC93', 43.4, 3802.0, 82.5e3),
        ('DAC88', 71.1, 4394.0, 133.5e3),
    ],
    ids=['38 GPa', '43 GPa', '71 GPa'],
)
def test_oxidation_integral_reproduces_the_deng_values_of_zhang_2024(name, P, T, fig_s5):
    """int dV(FeO1.5 - FeO) dP matches the Deng et al. (2020) 12.5 mol% FeO*
    values of Zhang et al. (2024) Sci. Adv. 10, eadp1752, supplementary
    Fig. S5 (orange diamonds), at the pressures and time-averaged
    temperatures of their Table S1.

    The Fig. S5 values are read off the plot to about +-1.5 kJ/mol, hence a 3%
    tolerance. The 25 mol% FeO* model in the same figure sits 40-75 kJ/mol
    higher, and dropping the factor 1/2 doubles the value.
    """
    I, valid = int_dV_dP_oxidation(T, P)

    assert bool(valid), name
    assert float(I) == pytest.approx(fig_s5, rel=0.03)
    assert abs(2.0 * float(I) - fig_s5) > 0.5 * fig_s5  # missing factor 1/2
    assert 1.0e4 < float(I) < 1.0e6  # J/mol, not kJ/mol or cm3 GPa


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_reaction_volume_change_is_positive_near_the_surface_and_negative_at_depth():
    """Disproportionation is volumetrically unfavourable in a shallow melt and
    favourable in a deep one, so the pressure term changes sign.

    Anchor: Schaefer et al. (2024) JGR Planets 129, e2023JE008262, whose
    metal-saturation results have no metal forming in the 500 km magma ocean
    models but a large metal event at the base of the whole-mantle models.
    That contrast requires the integral to change sign with depth.
    """
    shallow, shallow_valid = int_dV_dP(2500.0, 2.0)
    deep, deep_valid = int_dV_dP(2500.0, 20.0)

    assert float(shallow) > 0.0
    assert float(deep) < 0.0
    assert bool(shallow_valid) and bool(deep_valid)
    # Scale guard: tens of kJ/mol, not tens of J/mol nor tens of MJ/mol. The
    # deep value must also dominate, since it is what drives saturation.
    assert 1.0e3 < abs(float(deep)) < 1.0e6
    assert abs(float(deep)) > abs(float(shallow))


@pytest.mark.parametrize(
    'T_kelvin, P_gpa, reason',
    [
        (T_CEILING + 100.0, 50.0, 'above the temperature ceiling'),
        (3000.0, P_EXERCISED + 50.0, 'beyond the calibrated pressure range'),
        (np.nan, 50.0, 'undefined temperature'),
    ],
)
def test_points_outside_the_equation_of_state_domain_are_reported_invalid(
    T_kelvin, P_gpa, reason
):
    """Conditions outside the envelope are flagged rather than extrapolated,
    and return a zero the caller can distinguish from a computed zero through
    the validity flag.
    """
    value, valid = int_dV_dP(T_kelvin, P_gpa)
    ox, ox_valid = int_dV_dP_oxidation(T_kelvin, P_gpa)

    assert not bool(valid), reason
    assert not bool(ox_valid), reason
    assert float(value) == pytest.approx(0.0, abs=1e-30)
    assert float(ox) == pytest.approx(0.0, abs=1e-30)


def test_band_above_the_deng_limit_is_evaluated_and_logged_once(monkeypatch, caplog):
    """Between the Deng modelling limit (5000 K) and the envelope ceiling
    (6500 K) the integral is computed, flagged valid and logged once as an
    extrapolation; just above the ceiling it is excluded.
    """
    import logging

    assert eos_deng.T_DENG_MAX < T_CEILING
    monkeypatch.setattr(eos_deng, '_EXTRAPOLATION_WARNED', False)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_chem.eos_deng'):
        inside, ok_in = int_dV_dP(6000.0, 100.0)
        int_dV_dP(6400.0, 100.0)
        outside, ok_out = int_dV_dP(T_CEILING + 10.0, 100.0)

    assert bool(ok_in) and not bool(ok_out)
    assert float(outside) == 0.0
    # Same sign and order as the 5000 K value at this depth: hundreds of kJ/mol.
    assert -1.0e6 < float(inside) < -1.0e5
    assert sum('modelling limit' in r.message for r in caplog.records) == 1


def test_array_input_keeps_its_shape_and_masks_only_the_invalid_points():
    """A mixed array of cells returns the same shape, real values where valid
    and zeros where invalid, matching the scalar evaluation point by point."""
    T = np.array([[3000.0, T_CEILING + 1.0], [2500.0, 4000.0]])
    P = np.array([[50.0, 50.0], [20.0, P_EXERCISED + 1.0]])
    value, valid = int_dV_dP(T, P)

    assert value.shape == T.shape and valid.shape == T.shape
    np.testing.assert_array_equal(valid, [[True, False], [True, False]])
    assert value[0, 1] == 0.0 and value[1, 1] == 0.0
    assert value[0, 0] == pytest.approx(float(int_dV_dP(3000.0, 50.0)[0]), rel=1e-12)
    assert abs(value[1, 0]) > 1.0e3


def test_conditions_inside_the_domain_are_reported_valid():
    """Magma-ocean conditions well inside both limits are accepted, so the
    validity mask is not rejecting everything."""
    value, valid = int_dV_dP(3000.0, 50.0)

    assert bool(valid)
    # A real, non-zero correction: tens to hundreds of kJ/mol.
    assert 1.0e3 < abs(float(value)) < 1.0e6


@pytest.mark.physics_invariant
def test_liquid_iron_oxide_volume_integral_is_linear_at_low_pressure():
    """At P much smaller than K0 = 128 GPa the liquid FeO volume is its 1-bar
    value V0 exp(alpha0 (T - 298)) (Komabayashi 2014 Table 2), so the integral
    from 1 bar is that volume times (P - 1 bar).

    Guard: the Lange & Carmichael (1987) 1-bar volume of Schaefer's archived
    code is 10 % larger at 2500 K, far outside the tolerance.
    """
    T = 2500.0
    V_cm3 = 13.16 * np.exp(4.7e-5 * (T - 298.0))
    small, smaller = float(_int_V_FeO(T, 0.02)), float(_int_V_FeO(T, 0.01))

    # 1 cm3/mol x 1 GPa = 1e3 J/mol; compression over 0.02 GPa is 2e-4.
    assert small == pytest.approx(V_cm3 * (0.02 - _P0) * 1e3, rel=1e-3)
    assert smaller == pytest.approx(V_cm3 * (0.01 - _P0) * 1e3, rel=1e-3)
    V_lc = (13650.0 + 2.92 * (T - 1673.0)) * 1e-3
    assert abs(smaller / (V_lc * (0.01 - _P0) * 1e3) - 1.0) > 0.05


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    'phase, P, T, V_table3',
    [
        ('Fe', 136.0, 4000.0, 5.18),
        ('FeO', 136.0, 4000.0, 8.63),
        ('Fe', 330.0, 5000.0, 4.21),
        ('FeO', 330.0, 5000.0, 6.59),
    ],
    ids=['Fe 136 GPa', 'FeO 136 GPa', 'Fe 330 GPa', 'FeO 330 GPa'],
)
def test_liquid_volumes_reproduce_komabayashi_2014_table3(phase, P, T, V_table3):
    """The liquid Fe and FeO volumes match Komabayashi (2014) JGR Solid Earth
    119, 4164, Table 3 (end-member columns, X_Fe = 1 and 0) to 0.2 %, the
    rounding of the tabulated two decimals. Guards: the Lange & Carmichael
    FeO volume of Schaefer's archived code is 14 % and 37 % too large at these
    points, and the Fe value without thermal expansion 2-6 % too small."""
    V = float(eos_deng._VT(P, T, phase))
    V_cold = float(eos_deng._V298(P, phase))

    assert V == pytest.approx(V_table3, rel=2e-3)
    assert abs(V_cold / V_table3 - 1.0) > 0.01
    if phase == 'FeO':
        V_lc = (13650.0 + 2.92 * (T - 1673.0)) * 1e-3 * (1.0 + 4.0 * P / 30.33) ** -0.25
        assert V_lc / V_table3 > 1.1


def test_extrapolation_beyond_the_komabayashi_range_is_logged_once(monkeypatch, caplog):
    """Above 350 GPa or 7000 K the FeO and Fe terms are computed (finite, the
    integral keeps its sign and size) and the extrapolation is logged once."""
    import logging

    monkeypatch.setattr(eos_deng, '_KOMA_EXTRAPOLATION_WARNED', False)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_chem.eos_deng'):
        inside = eos_deng._fe_feo_terms(np.array([4000.0]), np.array([100.0]))
        beyond = eos_deng._fe_feo_terms(np.array([10000.0]), np.array([1000.0]))
        eos_deng._fe_feo_terms(np.array([12000.0]), np.array([2000.0]))

    assert all(np.isfinite(v[0]) for v in inside + beyond)
    # int V_FeO dP: about 1.1 MJ/mol at 100 GPa and 6.7 MJ/mol at 1 TPa.
    assert 5.0e5 < float(inside[0][0]) < 2.0e6
    assert 4.0e6 < float(beyond[0][0]) < 1.0e7
    assert sum('database range' in r.message for r in caplog.records) == 1


@pytest.mark.physics_invariant
def test_liquid_iron_volume_is_the_reference_volume_at_zero_pressure():
    """The Vinet inversion is skipped at P ~ 0, where V = V298 exactly, joins
    it continuously, and solves the Vinet equation at depth."""
    assert eos_deng._V_Fe(0.0) == eos_deng._V298_FE
    assert eos_deng._V_Fe(1e-7) == eos_deng._V298_FE
    assert eos_deng._V_Fe(1e-3) == pytest.approx(eos_deng._V298_FE, rel=1e-4)
    V = eos_deng._V_Fe(10.0)
    assert V < eos_deng._V298_FE
    assert eos_deng._vinet_resid(V, 10.0) == pytest.approx(0.0, abs=1e-9)


# ── model = 'keane_table' (Keane + Deng-thermal lookup table) ─────────────────


def _pressure_term(I, T):
    """int dV dP / (R T ln10), the log10 units of the fO2 pressure term."""
    from scipy.constants import R

    return np.asarray(I) / (R * np.asarray(T) * np.log(10.0))


@pytest.mark.usefixtures('keane_table')
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    'T, P', [(3000.0, 50.0), (3000.0, 70.0), (4000.0, 80.0), (4000.0, 5.0)]
)
def test_keane_table_matches_deng_inside_the_fpmd_coverage(T, P):
    """Inside the Deng et al. (2020) FPMD coverage (Fig. 1: 3000 K to 74 GPa,
    4000 K to 94 GPa for the 12.5 mol% Fe3+ melt) both models fit the same
    points, so their oxidation integrals agree to within 0.15 in log10 units
    (cross-implementation check; the table fit residual is 0.08-0.09 GPa).
    Guard: doubling the table value, the volume change per formula unit of
    Fe2O3 instead of per Fe, misses by more than 1.
    """
    deng, ok_d = int_dV_dP_oxidation(T, P)
    keane, ok_k = int_dV_dP_oxidation(T, P, model='keane_table')

    assert bool(ok_d) and bool(ok_k)
    assert abs(float(_pressure_term(keane - deng, T))) < 0.15
    if P >= 50.0:
        assert abs(float(_pressure_term(2.0 * keane - deng, T))) > 1.0


@pytest.mark.usefixtures('keane_table')
@pytest.mark.physics_invariant
def test_keane_table_oxidation_integral_vanishes_at_one_bar_and_rises_with_pressure():
    """The oxidation integral starts from 1 bar, so it is zero there (and at
    lower pressure, where the 1-bar value is used); with dV > 0 near the
    surface it rises monotonically over the first 50 GPa at 3000 K."""
    at_1bar, ok = int_dV_dP_oxidation(3000.0, 1.0e-4, model='keane_table')
    below, ok_below = int_dV_dP_oxidation(3000.0, 1.0e-6, model='keane_table')
    P = np.linspace(0.1, 50.0, 50)
    I, ok_all = int_dV_dP_oxidation(3000.0, P, model='keane_table')

    assert bool(ok) and bool(ok_below) and np.all(ok_all)
    assert float(at_1bar) == pytest.approx(0.0, abs=1e-6)
    assert float(below) == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(I) > 0.0)


@pytest.mark.usefixtures('keane_table')
@pytest.mark.physics_invariant
def test_keane_table_disproportionation_term_differs_from_deng_only_through_the_melt_volumes():
    """int_dV_dP = 2 int dV_melt dP - int V_FeO dP + int V_Fe dP, with the FeO
    and Fe terms shared by both models, so the difference between the models
    is exactly twice the difference of their oxidation integrals."""
    T, P = 3000.0, 50.0
    full_d, _ = int_dV_dP(T, P)
    full_k, ok = int_dV_dP(T, P, model='keane_table')
    ox_d, _ = int_dV_dP_oxidation(T, P)
    ox_k, _ = int_dV_dP_oxidation(T, P, model='keane_table')

    assert bool(ok)
    assert float(full_k - full_d) == pytest.approx(2.0 * float(ox_k - ox_d), rel=1e-9, abs=1e-6)
    # Guard: a single (not doubled) melt term would leave half the difference.
    assert abs(float(full_k - full_d) - float(ox_k - ox_d)) > 1.0e3


@pytest.mark.usefixtures('keane_table')
def test_keane_table_covers_super_earth_conditions_outside_the_deng_envelope(
    monkeypatch, caplog
):
    """At 10000 K and 1 TPa the deng2020 model is outside its envelope, while
    the table returns a valid value of the expected size (hundreds of kJ/mol)
    and logs the extrapolation beyond the FPMD coverage once."""
    import logging

    monkeypatch.setattr(eos_deng, '_TABLE_EXTRAPOLATION_WARNED', False)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_chem.eos_deng'):
        deng, ok_d = int_dV_dP_oxidation(10000.0, 1000.0)
        keane, ok_k = int_dV_dP_oxidation(10000.0, 1000.0, model='keane_table')
        int_dV_dP_oxidation(12000.0, 2000.0, model='keane_table')

    assert not bool(ok_d) and float(deng) == 0.0
    assert bool(ok_k)
    assert 1.0e5 < float(keane) < 1.0e6
    assert sum('FPMD coverage' in r.message for r in caplog.records) == 1


@pytest.mark.usefixtures('keane_table')
@pytest.mark.parametrize(
    'T_kelvin, P_gpa, reason',
    [
        (500.0, 10.0, 'below the table temperature range'),
        (25000.0, 10.0, 'above the table temperature range'),
        (3000.0, 5000.0, 'beyond the table pressure range'),
        (np.nan, 10.0, 'undefined temperature'),
    ],
)
def test_keane_table_points_outside_the_table_are_reported_invalid(T_kelvin, P_gpa, reason):
    """Outside the tabulated range both integrals return zero flagged invalid,
    and the stated envelope matches the table range."""
    ox, ok_ox = int_dV_dP_oxidation(T_kelvin, P_gpa, model='keane_table')
    full, ok_full = int_dV_dP(T_kelvin, P_gpa, model='keane_table')

    assert not bool(ok_ox) and not bool(ok_full), reason
    assert float(ox) == 0.0 and float(full) == 0.0
    (t_lo, t_hi), (_, p_hi) = (
        eos_deng._keane_table()['T_range'],
        eos_deng._keane_table()['P_range'],
    )
    assert (round(t_lo), round(t_hi), round(p_hi)) == (1000, 20000, 4000)
    assert eos_deng.ENVELOPE['keane_table'] == '1000 K <= T <= 20000 K, P <= 4000 GPa'


@pytest.mark.parametrize('bad', ['Deng2020', 'keane', ''])
def test_unknown_melt_eos_model_is_rejected(bad):
    """A misspelt model raises before any evaluation, for both integrals."""
    with pytest.raises(ValueError, match='Unknown melt EOS model'):
        int_dV_dP(3000.0, 50.0, model=bad)
    with pytest.raises(ValueError, match='Unknown melt EOS model'):
        int_dV_dP_oxidation(3000.0, 50.0, model=bad)
    assert eos_deng.MODELS == ('deng2020', 'keane_table')


def test_missing_keane_table_raises_with_the_expected_location(monkeypatch, tmp_path):
    """The table is not part of the repository: without it in FWL_DATA, the
    keane_table model raises FileNotFoundError naming FWL_DATA/interior_chem,
    and the deng2020 model is unaffected."""
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr(eos_deng, '_TABLE', None)

    assert table_path() == tmp_path / 'interior_chem' / 'keane_deng_table.npz'
    with pytest.raises(FileNotFoundError, match='interior_chem'):
        int_dV_dP_oxidation(3000.0, 50.0, model='keane_table')
    value, valid = int_dV_dP_oxidation(3000.0, 50.0)
    assert bool(valid) and 1.0e4 < float(value) < 1.0e6
