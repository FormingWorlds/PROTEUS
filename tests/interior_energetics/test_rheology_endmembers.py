"""Tests for ``proteus.interior_energetics.rheology_endmembers``.

Source under test: the PALEOS unified-table reader, the per-phase adiabatic bulk modulus,
the SLB (2005) finite-strain shear law with its Debye thermal term, the melting-curve end
members, the cache and the (P, T) lookups.

A small synthetic table in the PALEOS format replaces the 48 MB production table. Its
density is ``ln rho = ln rho0 + P / K - beta s`` with ``s = c ln T - d P``, so the adiabatic
bulk modulus is exactly ``K`` per phase while the isothermal one is ``1 / (1/K + beta d)``.

Invariants checked:
- the adiabatic K_S equals the analytic K of each phase, including at rows next to a
  polymorph boundary (no difference across the boundary);
- the shear law returns G0 at the reference state and softens with temperature;
- the Debye energy follows its high- and low-temperature limits;
- melting-curve and (P, T) lookups return the solid and melt end members of each side.

See ``docs/How-to/testing.md`` and ``docs/Validation/interior_energetics/rheology_endmembers.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import R

from proteus.interior_energetics import rheology_endmembers as rem

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

K_BRG, K_PPV, K_LIQ = 250e9, 400e9, 120e9  # Pa
P_TR, T_MELT = 3e10, 3000.0  # brg -> ppv transition [Pa], melting temperature [K]
BETA, C_S, D_S = 1e-4, 1000.0, 1e-8  # ln rho per entropy unit; s = C ln T - D P


def _write_table(path):
    """Synthetic PALEOS-format table on a log-uniform (P, T) grid; returns the axes."""
    log_p = np.linspace(9.5, 11.0, 31)  # 3.2 GPa to 100 GPa
    log_t = np.linspace(3.0, 3.7, 15)  # 1000 K to 5000 K
    rows = []
    for p in 10.0**log_p:
        for t in 10.0**log_t:
            s = C_S * np.log(t) - D_S * p
            if t >= T_MELT:
                phase, k, rho0 = 'liquid', K_LIQ, 3000.0
            elif p < P_TR:
                phase, k, rho0 = 'solid-brg', K_BRG, 4000.0
            else:
                phase, k, rho0 = 'solid-ppv', K_PPV, 4300.0
            rho = rho0 * np.exp(p / k - BETA * (s - 7000.0))
            rows.append(f'{p:.10e} {t:.10e} {rho:.10e} 0 {s:.10e} 0 0 0 0 {phase}')
    path.write_text('# synthetic PALEOS-format table\n' + '\n'.join(rows) + '\n')
    return log_p, log_t


@pytest.fixture
def table(tmp_path):
    """Synthetic table file and a clean end-member memo."""
    rem.load_endmembers.cache_clear()
    path = tmp_path / 'paleos_synthetic.dat'
    _write_table(path)
    yield path
    rem.load_endmembers.cache_clear()


@pytest.mark.physics_invariant
def test_adiabatic_bulk_is_phase_k_and_not_isothermal(table):
    """K_S recovers each phase's K at fixed entropy, also next to the brg-ppv boundary.

    The isothermal modulus 1 / (1/K + beta d) is 20 % below K for brg; a difference
    across the boundary would mix the two phases' densities and miss both.
    """
    grid = rem.read_paleos_grid(table)
    ks = rem.adiabatic_bulk_grid(grid)
    brg, ppv = grid['code'] == rem.PHASES.index('brg'), grid['code'] == rem.PHASES.index('ppv')
    np.testing.assert_allclose(ks[brg & np.isfinite(ks)], K_BRG, rtol=1e-6)
    np.testing.assert_allclose(ks[ppv & np.isfinite(ks)], K_PPV, rtol=1e-6)
    k_t = 1.0 / (1.0 / K_BRG + BETA * D_S)
    assert abs(k_t - K_BRG) / K_BRG > 0.1  # isothermal modulus is excluded by the tolerance
    last_brg = np.flatnonzero(brg.any(axis=1))[-1]  # row whose upper neighbour is ppv
    edge = ks[last_brg][brg[last_brg]]
    # only the coldest node, whose entropy lies off the lower neighbour row, has no K_S
    assert np.isfinite(edge[1:]).all()
    assert edge[1:] == pytest.approx(K_BRG, rel=1e-6)


@pytest.mark.physics_invariant
def test_melting_curve_sides_carry_their_phase(table):
    """Solid side: highest solid node below T_melt; melt side: lowest liquid node above it."""
    em = rem.load_endmembers(table, table.parent)
    m = em.melting(np.array([1e10, 6e10]))
    assert np.all(m['T_melt'] < T_MELT) and np.all(m['T_melt'] > 0.8 * T_MELT)
    np.testing.assert_allclose(m['kappa_sol'], [K_BRG, K_PPV], rtol=1e-6)
    np.testing.assert_allclose(m['kappa_liq'], K_LIQ, rtol=1e-6)
    assert np.all(m['rho_liq'] < m['rho_sol'])  # this synthetic melt is lighter


@pytest.mark.physics_invariant
def test_solid_lookup_interpolates_and_falls_back_above_solid_range(table):
    """Bilinear (P, T) lookup inside the solid; melting-curve solid side above T_melt."""
    em = rem.load_endmembers(table, table.parent)
    inside = em.solid(1e10, 1500.0)
    assert inside['kappa'] == pytest.approx(K_BRG, rel=1e-6)
    assert 3800.0 < inside['rho'] < 4400.0
    hot = em.solid(1e10, 4500.0)  # above the solid range: solidus-side values
    side = em.melting(1e10)
    assert hot['rho'] == pytest.approx(side['rho_sol'], rel=1e-12)
    assert hot['mu'] == pytest.approx(side['mu_sol'], rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('phase', ['en', 'lpcen', 'hpcen', 'brg', 'ppv'])
def test_shear_law_returns_g0_at_reference_state(phase):
    """SLB (2005) eq. 33: at zero strain (rho = M / V0) and T = 300 K the modulus is G0.

    At a strain of 0.02 the (1+2f)^(5/2) prefactor is checked against the (1+2f)^(3/2)
    variant, which differs by 2 %.
    """
    v0, k0, k0p = rem.PALEOS_EOS[phase]
    g0, g0p = rem.SLB_SHEAR[phase][:2]
    rho0 = rem.M_MGSIO3 / v0
    assert float(rem.slb_shear_modulus(rho0, 300.0, phase)) == pytest.approx(g0, rel=1e-9)
    f = 0.02
    rho = rho0 * (1.0 + 2.0 * f) ** 1.5
    bracket = (
        g0
        + (3 * k0 * g0p - 5 * g0) * f
        + (6 * k0 * g0p - 24 * k0 - 14 * g0 + 4.5 * k0 * k0p) * f**2
    )
    val = float(rem.slb_shear_modulus(rho, 300.0, phase))
    assert val == pytest.approx((1 + 2 * f) ** 2.5 * bracket, rel=1e-9)
    assert abs(val - (1 + 2 * f) ** 1.5 * bracket) / val > 0.015


@pytest.mark.physics_invariant
def test_shear_law_softens_with_temperature():
    """At fixed density the thermal term lowers the shear modulus as T rises above 300 K."""
    rho = 4200.0
    g = rem.slb_shear_modulus(rho, np.array([300.0, 1500.0, 3000.0, 4500.0]), 'brg')
    assert np.all(np.diff(g) < 0)
    assert np.all(g > 0)
    assert g[0] / g[-1] < 1.5  # softening of tens of percent, not orders of magnitude


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_debye_energy_high_and_low_temperature_limits():
    """Debye energy: 3nRT[1 - 3x/8 + x^2/20] for x = theta/T << 1, and 3 pi^4 n R T^4 / (5 theta^3) for T << theta."""
    theta, n = 900.0, rem.N_ATOMS
    hot = 50.0 * theta
    x = theta / hot
    expect_hot = 3 * n * R * hot * (1 - 3 * x / 8 + x**2 / 20)
    assert float(rem.debye_energy(theta, hot)) == pytest.approx(expect_hot, rel=1e-6)
    cold = theta / 100.0
    expect_cold = 3 * np.pi**4 * n * R * cold**4 / (5 * theta**3)
    assert float(rem.debye_energy(theta, cold)) == pytest.approx(expect_cold, rel=1e-6)
    assert 3 * n * R * cold > 10 * expect_cold  # the classical limit is excluded at low T


def test_phase_labels_map_to_codes():
    """PALEOS labels map to the phase index; unknown labels map to -1."""
    assert rem.phase_code('solid-brg') == rem.PHASES.index('brg')
    assert rem.phase_code('liquid') == rem.LIQUID
    assert rem.phase_code('solid-majorite') == -1


def test_cache_built_once_and_missing_table_raises(table, monkeypatch):
    """The first load derives and caches the end members; a later load reads the cache."""
    calls = []
    real = rem.build_endmembers
    monkeypatch.setattr(rem, 'build_endmembers', lambda g: calls.append(1) or real(g))
    rem.load_endmembers(table, table.parent)
    assert rem.endmember_cache_path(table, table.parent).is_file() and len(calls) == 1
    rem.load_endmembers.cache_clear()
    rem.load_endmembers(table, table.parent)
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError, match='PALEOS MgSiO3'):
        rem.load_endmembers(table.parent / 'absent.dat', table.parent)
