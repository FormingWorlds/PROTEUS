"""
Unit tests for proteus.outgas.dummy module.

Parameterized volatile partitioning without thermodynamic solver.
Splits elemental budgets between atmosphere and melt using melt-fraction-
dependent partition coefficient: f_dissolved = Phi_global, f_atm = 1 - Phi_global.

Physics tested:
- Melt fraction partitioning (Phi_global -> f_dissolved/f_atm)
- Element to species conversion (H->H2O, C->CO2, N->N2, S->SO2)
- Thin-atmosphere pressure: P = m*g / (4*pi*R^2)
- VMR = P_species / P_total
- Mean molecular weight from VMRs
- Mass conservation across reservoirs
- Edge cases: fully molten, fully solid, zero inventory, zero gravity
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.outgas.dummy import calc_surface_pressures_dummy
from proteus.utils.constants import element_list, gas_list


def _make_hf_row(
    H_kg=1e20,
    C_kg=1e19,
    N_kg=1e18,
    S_kg=1e17,
    Phi_global=0.5,
    T_magma=3000.0,
    gravity=9.81,
    R_int=6.371e6,
):
    """Build a minimal hf_row for dummy outgas tests."""
    hf_row = {
        'T_magma': T_magma,
        'Phi_global': Phi_global,
        'gravity': gravity,
        'R_int': R_int,
        'H_kg_total': H_kg,
        'C_kg_total': C_kg,
        'N_kg_total': N_kg,
        'S_kg_total': S_kg,
    }
    # Zero out remaining elements
    for e in element_list:
        hf_row.setdefault(f'{e}_kg_total', 0.0)
    return hf_row


@pytest.mark.unit
def test_dummy_outgas_partitioning_half_molten():
    """Test that Phi_global=0.5 splits mass 50/50 between melt and atmosphere."""
    hf_row = _make_hf_row(H_kg=1e20, Phi_global=0.5)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    # H2O should have 50% in atmosphere, 50% dissolved
    assert hf_row['H2O_kg_atm'] > 0
    assert hf_row['H2O_kg_liquid'] > 0
    assert hf_row['H2O_kg_atm'] == pytest.approx(hf_row['H2O_kg_liquid'], rel=1e-10)


@pytest.mark.unit
def test_dummy_outgas_fully_molten():
    """Phi_global=1.0: all volatiles dissolved, zero atmosphere.

    Physics: fully molten magma ocean retains all volatiles.
    """
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, Phi_global=1.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)
    for s in gas_list:
        assert hf_row[f'{s}_kg_atm'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_fully_solid():
    """Phi_global=0.0: all volatiles outgassed, zero dissolved.

    Physics: fully solidified mantle, all volatiles in atmosphere.
    """
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, Phi_global=0.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    assert hf_row['P_surf'] > 0.0
    for s in ('H2O', 'CO2'):
        assert hf_row[f'{s}_kg_liquid'] == pytest.approx(0.0, abs=1e-30)
        assert hf_row[f'{s}_kg_atm'] > 0.0


@pytest.mark.unit
def test_dummy_outgas_species_mapping():
    """Test element-to-species mapping: H->H2O, C->CO2, N->N2, S->SO2."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    # All four species should have nonzero atmospheric mass
    assert hf_row['H2O_kg_atm'] > 0.0
    assert hf_row['CO2_kg_atm'] > 0.0
    assert hf_row['N2_kg_atm'] > 0.0
    assert hf_row['SO2_kg_atm'] > 0.0

    # Species not in the mapping should be zero
    assert hf_row['H2_kg_atm'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['CO_kg_atm'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['CH4_kg_atm'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_pressure_calculation():
    """Test thin-atmosphere pressure: P = m*g / (4*pi*R^2).

    Physics: for a thin atmosphere on a spherical planet.
    """
    R = 6.371e6
    g = 9.81
    area = 4.0 * np.pi * R**2
    # Only H, all outgassed (Phi=0)
    H_kg = 1e20
    mass_frac_H_in_H2O = 2 * 1.008 / 18.015
    H2O_kg = H_kg / mass_frac_H_in_H2O

    hf_row = _make_hf_row(H_kg=H_kg, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    expected_P = H2O_kg * g / area * 1e-5  # Pa -> bar
    assert hf_row['P_surf'] == pytest.approx(expected_P, rel=1e-6)
    assert hf_row['H2O_bar'] == pytest.approx(expected_P, rel=1e-6)


@pytest.mark.unit
def test_dummy_outgas_vmr_single_species():
    """Single-species atmosphere should have VMR=1.0 for that species."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    assert hf_row['H2O_vmr'] == pytest.approx(1.0, rel=1e-10)
    assert hf_row['CO2_vmr'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_vmr_sum_to_one():
    """VMRs must sum to 1.0 when P_surf > 0 (normalization check)."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.3)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    vmr_sum = sum(hf_row.get(f'{s}_vmr', 0.0) for s in gas_list)
    assert vmr_sum == pytest.approx(1.0, rel=1e-10)


@pytest.mark.unit
def test_dummy_outgas_mmw_single_species():
    """MMW should equal species molar mass for single-species atmosphere."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    # Pure H2O atmosphere: MMW = 18.015e-3 kg/mol
    assert hf_row['atm_kg_per_mol'] == pytest.approx(18.015e-3, rel=1e-3)


@pytest.mark.unit
def test_dummy_outgas_mass_conservation():
    """Total species mass = atm + liquid + solid for each species."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.4)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    for s in ('H2O', 'CO2', 'N2', 'SO2'):
        total = hf_row[f'{s}_kg_total']
        parts = (
            hf_row[f'{s}_kg_atm']
            + hf_row[f'{s}_kg_liquid']
            + hf_row[f'{s}_kg_solid']
        )
        assert total == pytest.approx(parts, rel=1e-10), f'{s} mass not conserved'


@pytest.mark.unit
def test_dummy_outgas_zero_inventory():
    """Zero elemental inventory should produce zero pressures and masses."""
    hf_row = _make_hf_row(H_kg=0, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.5)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)
    # M_atm is set by the wrapper, not by the dummy module itself
    assert 'M_atm' not in hf_row or hf_row['M_atm'] == 0.0


@pytest.mark.unit
def test_dummy_outgas_zero_gravity():
    """Zero gravity should produce zero pressures but nonzero masses."""
    hf_row = _make_hf_row(H_kg=1e20, gravity=0.0, Phi_global=0.0)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)
    # Masses should still be partitioned
    assert hf_row['H2O_kg_atm'] > 0.0


@pytest.mark.unit
def test_dummy_outgas_phi_clamped():
    """Phi_global > 1 or < 0 should be clamped to [0, 1]."""
    # Phi > 1 should be clamped to 1 (fully dissolved)
    hf_row = _make_hf_row(H_kg=1e20, Phi_global=1.5)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)
    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)

    # Phi < 0 should be clamped to 0 (fully outgassed)
    hf_row2 = _make_hf_row(H_kg=1e20, Phi_global=-0.5)
    calc_surface_pressures_dummy(dirs, config, hf_row2)
    assert hf_row2['P_surf'] > 0.0


@pytest.mark.unit
def test_dummy_outgas_element_reservoir_consistency():
    """Element reservoir masses should be consistent with species partitioning."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.3)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    # H reservoir: atm + liquid should equal total H mass
    H_atm = hf_row.get('H_kg_atm', 0.0)
    H_liquid = hf_row.get('H_kg_liquid', 0.0)
    H_total = hf_row.get('H_kg_total', 0.0)
    assert H_atm + H_liquid == pytest.approx(H_total, rel=1e-6)
    assert H_atm > 0.0
    assert H_liquid > 0.0


@pytest.mark.unit
def test_dummy_outgas_oxygen_total_set():
    """O_kg_total should be set as sum of O_kg_atm + O_kg_liquid."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, Phi_global=0.5)
    dirs = {'output': '/tmp/test'}
    config = MagicMock()

    calc_surface_pressures_dummy(dirs, config, hf_row)

    O_total = hf_row.get('O_kg_total', 0.0)
    O_atm = hf_row.get('O_kg_atm', 0.0)
    O_liquid = hf_row.get('O_kg_liquid', 0.0)
    assert O_total == pytest.approx(O_atm + O_liquid, rel=1e-10)
    assert O_total > 0.0  # Both H2O and CO2 contain oxygen
