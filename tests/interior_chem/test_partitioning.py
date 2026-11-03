"""Fe-Mg exchange partitioning (``src/proteus/interior_chem/partitioning.py``).

Invariants and references checked:

* K_D(P, T) reproduces the published fits: Dragulet & Stixrude (2024) for
  bridgmanite and Dragulet & Stixrude (2026) for post-perovskite, which
  agree at the 180 GPa triple point; the gas constant is per mole.
* K_D decreases with pressure in both phases, and the temperature clamp
  holds K_D at the edge of each fit.
* The solid composition tends to K_D x for a dilute melt and to 1 for a
  pure FeSiO3 melt.
* The per-cell split conserves Fe2+, keeps Mg >= 0 in both phases, returns
  solid and residual melt that satisfy K_D, and tends to the infinitesimal
  equilibrium D as the solid fraction vanishes.
* Many small batch steps converge to the fractional-crystallisation ODE
  (Dragulet & Stixrude 2024 Eq. 12, 2026 Eq. 9) in the (Mg,Fe)SiO3 basis.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from proteus.interior_chem.partitioning import (
    KD_T_RANGE,
    MU_FEO15,
    MU_FESIO3,
    MU_MGSIO3,
    P_PPV_MAX_GPA,
    P_TRIPLE_GPA,
    d_fe2_equilibrium,
    fe2_into_solid_kd,
    kd_fe_mg,
    melt_n_mg,
    solid_x_fe,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# 1 kg of melt with the tracker's 8 wt% ferrous FeO and f_0 = 0.1 (redox Step 1-2)
_N2 = 0.08 / 0.07184
_N3 = _N2 * 0.1 / 0.9


def _phase_x(n_fe2, m, n_fe3):
    """Fe2+ / (Fe2+ + Mg) of a phase of mass m [kg] in the (Mg,Fe)SiO3 basis."""
    n_mg = (m - n_fe2 * MU_FESIO3 - n_fe3 * MU_FEO15) / MU_MGSIO3
    return n_fe2 / (n_fe2 + n_mg), n_mg


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_bridgmanite_kd_matches_the_dragulet_stixrude_2024_fit():
    """Dragulet & Stixrude (2024, GRL, Fig. 4): K_D ~ 0.3 near 25 GPa and
    <= 0.1 above 100 GPa on the 4000 K curve. The value at 136 GPa and
    4000 K is the fit evaluated by hand: exp(-(-19660 + 136e9 * 0.53e-6)
    / (R 4000) - 12.04 / R) = 0.0486."""
    kd, clamped = kd_fe_mg([25.0, 100.0, 136.0], [4000.0, 4000.0, 4000.0])
    assert 0.25 < kd[0] < 0.33
    assert kd[1] <= 0.1
    assert kd[2] == pytest.approx(0.0486, rel=2e-3)
    assert not np.any(clamped)
    # k_B with per-mole parameters underflows exp(-dH / k_B T) to 0
    wrong = np.exp(-(-19.66e3 + 136e9 * 0.53e-6) / (1.380649e-23 * 4000.0))
    assert abs(kd[2] - wrong) > 0.5 * kd[2]
    # A flipped sign of dS multiplies K_D by exp(2 * 12.04 / R) = 18
    assert kd[2] * np.exp(-2 * 12.04 / 8.314) < 0.01


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_two_fits_agree_with_the_triple_point_values_of_the_2026_paper():
    """Dragulet & Stixrude (2026, Section 3.1): at the 180 GPa triple point
    K_D steps from 0.057 +- 0.007 (bridgmanite) to 0.086 +- 0.021
    (post-perovskite). The temperature at which the bridgmanite fit gives
    0.057 there, 6430 K, must give the post-perovskite value within its
    uncertainty."""
    T_tp = 6430.0
    kd_bdg, _ = kd_fe_mg(P_TRIPLE_GPA - 1e-9, T_tp)
    kd_ppv, _ = kd_fe_mg(P_TRIPLE_GPA, T_tp)
    assert float(kd_bdg) == pytest.approx(0.057, abs=0.002)
    assert float(kd_ppv) == pytest.approx(0.086, abs=0.021)
    # The step is upward, as in their Fig. 3b; a phase swap would reverse it
    assert float(kd_ppv) > float(kd_bdg) + 0.02


@pytest.mark.physics_invariant
def test_kd_falls_with_pressure_in_both_phases():
    """Iron grows more incompatible on compression (positive exchange volume
    in both fits), at fixed temperature inside each fitted range."""
    P_bdg = np.linspace(25.0, 175.0, 16)
    P_ppv = np.linspace(180.0, 600.0, 16)
    kd_bdg, _ = kd_fe_mg(P_bdg, 4500.0)
    kd_ppv, _ = kd_fe_mg(P_ppv, 8000.0)
    assert np.all(np.diff(kd_bdg) < 0.0)
    assert np.all(np.diff(kd_ppv) < 0.0)
    assert np.all((kd_bdg > 0.0) & (kd_bdg < 1.0))
    assert np.all((kd_ppv > 0.0) & (kd_ppv < 1.0))


@pytest.mark.physics_invariant
def test_temperature_and_pressure_outside_the_fits_take_the_edge_value():
    """Bridgmanite below 3000 K takes the 3000 K value, post-perovskite below
    6000 K the 6000 K value and above 603 GPa the 603 GPa value; each such
    cell is flagged, and cells inside the fit are not."""
    T_lo_bdg = KD_T_RANGE['bdg'][0]
    T_lo_ppv = KD_T_RANGE['ppv'][0]
    kd, clamped = kd_fe_mg(
        [100.0, 100.0, 300.0, 300.0, 800.0, 400.0],
        [2000.0, T_lo_bdg, 4000.0, T_lo_ppv, 8000.0, 8000.0],
    )
    edge, _ = kd_fe_mg(P_PPV_MAX_GPA, 8000.0)
    assert kd[0] == pytest.approx(kd[1], rel=1e-12)
    assert kd[2] == pytest.approx(kd[3], rel=1e-12)
    assert kd[4] == pytest.approx(float(edge), rel=1e-12)
    np.testing.assert_array_equal(clamped, [True, False, True, False, True, False])
    # Unclamped extrapolation to 2000 K would differ by more than 30%
    assert (
        abs(kd[0] - np.exp(-(-19.66e3 + 100e9 * 0.53e-6) / (8.314 * 2000) - 12.04 / 8.314))
        > 0.3 * kd[0]
    )


@pytest.mark.physics_invariant
def test_solid_composition_limits():
    """X_s = K_D x / (1 - (1 - K_D) x) tends to K_D x for a dilute melt, is 1
    for a pure FeSiO3 melt, is 0 without iron, and rises with x."""
    kd = 0.05
    assert solid_x_fe(kd, 1e-8) / 1e-8 == pytest.approx(kd, rel=1e-6)
    assert float(solid_x_fe(kd, 1.0)) == pytest.approx(1.0, rel=1e-12)
    assert float(solid_x_fe(kd, 0.0)) == 0.0
    x = np.linspace(0.0, 1.0, 21)
    assert np.all(np.diff(solid_x_fe(kd, x)) > 0.0)
    # The naive X_s = K_D x would leave a pure FeSiO3 melt with a 5% Fe solid
    assert float(solid_x_fe(kd, 1.0)) - kd * 1.0 > 0.9


@pytest.mark.physics_invariant
@pytest.mark.parametrize('kd', [0.03, 0.3], ids=['ppv_like', 'shallow_bdg_like'])
@pytest.mark.parametrize('s', [1e-3, 0.3, 0.9], ids=['trace_solid', 'third', 'most'])
def test_split_conserves_iron_and_returns_phases_at_the_exchange_coefficient(kd, s):
    """One cell of 1 kg crystallises a fraction s: Fe2+ moves into the solid
    without loss, Mg stays >= 0 in both phases, and the solid and residual
    melt it returns have (X_s (1 - X_l)) / (X_l (1 - X_s)) = K_D."""
    n3s = 0.6 * _N3 * s  # an arbitrary Fe3+ share of the solid
    n2s = float(fe2_into_solid_kd(_N2, _N3, n3s, 1.0, s, kd))
    assert 0.0 < n2s < _N2
    x_s, mg_s = _phase_x(n2s, s, n3s)
    x_l, mg_l = _phase_x(_N2 - n2s, 1.0 - s, _N3 - n3s)
    assert mg_s > 0.0 and mg_l > 0.0
    assert (x_s * (1 - x_l)) / (x_l * (1 - x_s)) == pytest.approx(kd, rel=1e-9)
    # Mg closes: the two phases hold the Mg of the cell before the step
    assert mg_s + mg_l == pytest.approx(float(melt_n_mg(1.0, _N2, _N3)), rel=1e-12)
    # Iron is incompatible, so the residual melt is richer than the solid
    assert x_l > x_s


@pytest.mark.physics_invariant
def test_split_edge_cases_none_all_and_no_iron():
    """s = 0 moves no Fe2+, s = 1 moves all of it, and a cell without Fe2+
    returns 0; vectorised over cells."""
    out = fe2_into_solid_kd(
        np.array([_N2, _N2, 0.0]),
        np.array([_N3, _N3, _N3]),
        np.array([0.0, _N3, 0.0]),
        np.ones(3),
        np.array([0.0, 1.0, 0.5]),
        np.full(3, 0.05),
    )
    np.testing.assert_allclose(out, [0.0, _N2, 0.0], rtol=0, atol=1e-15)
    assert out[1] > 1.0  # mol: all of the ~1.1 mol of Fe2+, not a share of it


@pytest.mark.physics_invariant
def test_vanishing_solid_tends_to_the_equilibrium_partition_coefficient():
    """For s -> 0 the residual melt keeps its composition, so the solved D
    equals d_fe2_equilibrium of the starting melt; at s = 0.5 the melt has
    enriched and D is clearly higher, because D rises with X_Fe."""
    kd = 0.05
    x0, _ = _phase_x(_N2, 1.0, _N3)
    d_eq = float(d_fe2_equilibrium(kd, x0, _N2, 0.0))

    def d_of(s):
        n2s = float(fe2_into_solid_kd(_N2, _N3, 0.0, 1.0, s, kd))
        return (n2s / s) / ((_N2 - n2s) / (1.0 - s))

    assert d_of(1e-7) == pytest.approx(d_eq, rel=1e-5)
    assert d_of(0.5) > 1.1 * d_eq
    # The FeO melt basis gives a ~7% lower D at this composition; the solver must not
    x_feo = _N2 / (_N2 + (1.0 - _N2 * 0.07184 - _N3 * MU_FEO15) / MU_MGSIO3)
    d_feo = float(d_fe2_equilibrium(kd, x_feo, _N2, 0.0))
    assert abs(d_feo - d_eq) > 0.04 * d_eq


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_small_batch_steps_converge_to_fractional_crystallisation():
    """Dragulet & Stixrude (2024) Eq. 12 / (2026) Eq. 9, written per formula
    unit: N dx/dN = x_s(x) - x, with the melt mass falling by the solid's
    molar mass per unit removed. 2000 batch steps of the tracker split must
    reproduce the ODE's melt X_Fe at 20% melt left; 5 steps must not, which
    shows the comparison resolves the step-size error."""
    kd, F = 0.05, 0.2
    n2_0 = 0.08 / 0.07184
    n_mg_0 = float(melt_n_mg(1.0, n2_0, 0.0))

    def rhs(u, y):
        # u: solid mass removed [kg]; N: melt formula units [mol]; x: melt X_Fe.
        # Removing du takes du/m_s units, x_s of them Fe: N dx = (x - x_s) du / m_s
        N, x = y
        xs = solid_x_fe(kd, x)
        m_s = xs * MU_FESIO3 + (1.0 - xs) * MU_MGSIO3
        return [-1.0 / m_s, (x - xs) / (N * m_s)]

    sol = solve_ivp(
        rhs,
        (0.0, 1.0 - F),
        [n2_0 + n_mg_0, n2_0 / (n2_0 + n_mg_0)],
        rtol=1e-10,
        atol=1e-12,
    )
    x_ode = sol.y[1, -1]

    def tracker(n_steps):
        n2, M = n2_0, 1.0
        s = 1.0 - F ** (1.0 / n_steps)
        for _ in range(n_steps):
            n2 -= float(fe2_into_solid_kd(n2, 0.0, 0.0, M, s, kd))
            M *= 1.0 - s
        return _phase_x(n2, M, 0.0)[0]

    assert tracker(2000) == pytest.approx(x_ode, rel=2e-3)
    assert abs(tracker(5) - x_ode) > 5e-3 * x_ode
    # Strong enrichment, not a trivially unchanged melt: x rises above 3x its start
    assert x_ode > 3.0 * n2_0 / (n2_0 + n_mg_0)


@pytest.mark.physics_invariant
def test_mass_without_magnesium_stays_in_the_residual_melt():
    """m_other_c, the mass metal separation leaves in the melt, holds no Mg:
    the solid and the residual melt with its Mg from (1 - s) M - m_other
    satisfy K_D, Fe2+ is conserved, and less melt Mg raises the melt X_Fe,
    so more Fe2+ enters the solid. m_other_c = 0 is the plain split."""
    n2, n3, n3s, M, s, kd = 0.4, 0.05, 0.02, 100.0, 0.3, 0.2
    base = float(fe2_into_solid_kd(n2, n3, n3s, M, s, kd))
    assert float(fe2_into_solid_kd(n2, n3, n3s, M, s, kd, 0.0)) == base
    m_other = 20.0
    n2s = float(fe2_into_solid_kd(n2, n3, n3s, M, s, kd, m_other))
    assert 0.0 < n2s < n2
    mg_s = (s * M - n2s * MU_FESIO3 - n3s * MU_FEO15) / MU_MGSIO3
    mg_l = ((1 - s) * M - m_other - (n2 - n2s) * MU_FESIO3 - (n3 - n3s) * MU_FEO15) / MU_MGSIO3
    x_s, x_l = n2s / (n2s + mg_s), (n2 - n2s) / (n2 - n2s + mg_l)
    assert (x_s / (1 - x_s)) / (x_l / (1 - x_l)) == pytest.approx(kd, rel=1e-9)
    # 20 kg less melt Mg (about 30% of it): X_l up, so the solid takes more Fe2+
    assert n2s > 1.05 * base
