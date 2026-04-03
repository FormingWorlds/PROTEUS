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
- Edge cases: fully molten, fully solid, zero inventory, zero gravity, clamping
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.outgas.common import expected_keys
from proteus.outgas.dummy import _ELEMENT_TO_SPECIES, _MMW, calc_surface_pressures_dummy
from proteus.utils.constants import element_list, gas_list

# Default planet parameters for tests
_R = 6.371e6
_g = 9.81
_area = 4.0 * np.pi * _R**2


def _make_hf_row(
    H_kg=1e20,
    C_kg=1e19,
    N_kg=1e18,
    S_kg=1e17,
    Phi_global=0.5,
    T_magma=3000.0,
    gravity=_g,
    R_int=_R,
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
    for e in element_list:
        hf_row.setdefault(f'{e}_kg_total', 0.0)
    return hf_row


def _run(hf_row):
    """Call dummy outgas with a mock config."""
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    calc_surface_pressures_dummy(dirs, config, hf_row)


def _expected_species_kg(element_kg, element):
    """Compute expected species kg from element kg using _ELEMENT_TO_SPECIES."""
    species, mass_frac = _ELEMENT_TO_SPECIES[element]
    return species, element_kg / mass_frac


# ---------------------------------------------------------------------------
# Key completeness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_sets_all_expected_keys():
    """All keys from expected_keys() except M_atm must be present after the call.

    M_atm is set by the wrapper, not by the dummy module.
    """
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.3)
    _run(hf_row)

    missing = [k for k in expected_keys() if k != 'M_atm' and k not in hf_row]
    assert missing == [], f'Missing keys after dummy outgas: {missing}'


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_partitioning_half_molten():
    """Phi_global=0.5 splits mass 50/50 between melt and atmosphere."""
    H_kg = 1e20
    hf_row = _make_hf_row(H_kg=H_kg, Phi_global=0.5)
    _run(hf_row)

    _, H2O_kg = _expected_species_kg(H_kg, 'H')
    expected_half = 0.5 * H2O_kg
    assert hf_row['H2O_kg_atm'] == pytest.approx(expected_half, rel=1e-10)
    assert hf_row['H2O_kg_liquid'] == pytest.approx(expected_half, rel=1e-10)


@pytest.mark.unit
def test_dummy_outgas_fully_molten():
    """Phi_global=1.0: all volatiles dissolved, zero atmosphere."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, Phi_global=1.0)
    _run(hf_row)

    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)
    for s in gas_list:
        assert hf_row[f'{s}_kg_atm'] == pytest.approx(0.0, abs=1e-30)
        assert hf_row[f'{s}_bar'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_fully_solid():
    """Phi_global=0.0: all volatiles outgassed, zero dissolved."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.0)
    _run(hf_row)

    assert hf_row['P_surf'] > 0.0
    for s in ('H2O', 'CO2', 'N2', 'SO2'):
        assert hf_row[f'{s}_kg_liquid'] == pytest.approx(0.0, abs=1e-30)
        assert hf_row[f'{s}_kg_atm'] > 0.0


# ---------------------------------------------------------------------------
# Species mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_species_mapping():
    """H->H2O, C->CO2, N->N2, S->SO2; unmapped species should be zero."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.0)
    _run(hf_row)

    for s in ('H2O', 'CO2', 'N2', 'SO2'):
        assert hf_row[f'{s}_kg_atm'] > 0.0

    unmapped = [s for s in gas_list if s not in ('H2O', 'CO2', 'N2', 'SO2')]
    for s in unmapped:
        assert hf_row[f'{s}_kg_atm'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_single_element_zero():
    """One element zero while others nonzero: only that species is absent."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=1e18, S_kg=1e17, Phi_global=0.0)
    _run(hf_row)

    assert hf_row['CO2_kg_atm'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['CO2_bar'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['H2O_kg_atm'] > 0.0
    assert hf_row['N2_kg_atm'] > 0.0
    assert hf_row['SO2_kg_atm'] > 0.0


# ---------------------------------------------------------------------------
# Pressure calculation (exact values for all four elements)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    'element,element_kg',
    [('H', 1e20), ('C', 1e19), ('N', 1e18), ('S', 1e17)],
)
def test_dummy_outgas_pressure_per_element(element, element_kg):
    """Verify P = m*g / (4*pi*R^2) for each element independently."""
    kwargs = {'H_kg': 0, 'C_kg': 0, 'N_kg': 0, 'S_kg': 0, 'Phi_global': 0.0}
    kwargs[f'{element[0]}_kg'] = element_kg
    hf_row = _make_hf_row(**kwargs)
    _run(hf_row)

    species, mass_frac = _ELEMENT_TO_SPECIES[element]
    species_kg = element_kg / mass_frac
    expected_P = species_kg * _g / _area * 1e-5
    assert hf_row['P_surf'] == pytest.approx(expected_P, rel=1e-8)
    assert hf_row[f'{species}_bar'] == pytest.approx(expected_P, rel=1e-8)


@pytest.mark.unit
def test_dummy_outgas_pressure_sum():
    """P_surf = sum of all partial pressures."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.3)
    _run(hf_row)

    p_sum = sum(hf_row[f'{s}_bar'] for s in gas_list)
    assert hf_row['P_surf'] == pytest.approx(p_sum, rel=1e-10)


# ---------------------------------------------------------------------------
# VMR
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_vmr_single_species():
    """Single-species atmosphere: VMR=1.0 for that species, 0 for others."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.0)
    _run(hf_row)

    assert hf_row['H2O_vmr'] == pytest.approx(1.0, abs=1e-10)
    for s in gas_list:
        if s != 'H2O':
            assert hf_row[f'{s}_vmr'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_vmr_sum_to_one():
    """VMRs must sum to 1.0 when P_surf > 0."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.3)
    _run(hf_row)

    vmr_sum = sum(hf_row[f'{s}_vmr'] for s in gas_list)
    assert vmr_sum == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Mean molecular weight
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_mmw_single_species():
    """Pure H2O atmosphere: MMW = 18.015e-3 kg/mol."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.0)
    _run(hf_row)

    assert hf_row['atm_kg_per_mol'] == pytest.approx(18.015e-3, rel=1e-8)


@pytest.mark.unit
def test_dummy_outgas_mmw_multi_species():
    """Multi-species atmosphere: MMW = sum(VMR_i * M_i), analytically verified."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=0, S_kg=0, Phi_global=0.0)
    _run(hf_row)

    # With H and C only (Phi=0): H2O + CO2
    _, H2O_kg = _expected_species_kg(1e20, 'H')
    _, CO2_kg = _expected_species_kg(1e19, 'C')
    P_H2O = H2O_kg * _g / _area * 1e-5
    P_CO2 = CO2_kg * _g / _area * 1e-5
    P_total = P_H2O + P_CO2
    vmr_H2O = P_H2O / P_total
    vmr_CO2 = P_CO2 / P_total
    expected_mmw = vmr_H2O * _MMW['H2O'] + vmr_CO2 * _MMW['CO2']
    assert hf_row['atm_kg_per_mol'] == pytest.approx(expected_mmw, rel=1e-8)


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_species_mass_conservation():
    """Total = atm + liquid + solid for each active species."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.4)
    _run(hf_row)

    for s in ('H2O', 'CO2', 'N2', 'SO2'):
        total = hf_row[f'{s}_kg_total']
        parts = hf_row[f'{s}_kg_atm'] + hf_row[f'{s}_kg_liquid'] + hf_row[f'{s}_kg_solid']
        assert total == pytest.approx(parts, rel=1e-10), f'{s} mass not conserved'

        # Also verify the total matches the element-derived value
        for elem, (species, mass_frac) in _ELEMENT_TO_SPECIES.items():
            if species == s:
                expected_total = float(hf_row.get(f'{elem}_kg_total', 0.0)) / mass_frac
                assert total == pytest.approx(expected_total, rel=1e-10), \
                    f'{s} total does not match element budget'


@pytest.mark.unit
def test_dummy_outgas_element_reservoir_consistency():
    """Element reservoir masses: atm + liquid should equal total for each element."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=1e18, S_kg=1e17, Phi_global=0.3)
    _run(hf_row)

    for elem in ('H', 'C', 'N', 'S'):
        e_atm = hf_row[f'{elem}_kg_atm']
        e_liquid = hf_row[f'{elem}_kg_liquid']
        e_total = hf_row[f'{elem}_kg_total']
        assert e_atm + e_liquid == pytest.approx(e_total, rel=1e-10), \
            f'{elem} element reservoir not conserved'
        assert e_atm > 0.0
        assert e_liquid > 0.0


@pytest.mark.unit
def test_dummy_outgas_mol_keys_set():
    """Molar amounts should be consistent with mass and molar mass."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.5)
    _run(hf_row)

    _, H2O_kg = _expected_species_kg(1e20, 'H')
    expected_mol = H2O_kg / _MMW['H2O']
    assert hf_row['H2O_mol_total'] == pytest.approx(expected_mol, rel=1e-8)
    assert hf_row['H2O_mol_atm'] == pytest.approx(0.5 * expected_mol, rel=1e-8)
    assert hf_row['H2O_mol_liquid'] == pytest.approx(0.5 * expected_mol, rel=1e-8)
    assert hf_row['H2O_mol_solid'] == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# Oxygen accounting
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_oxygen_total():
    """O_kg_total = O_kg_atm + O_kg_liquid, with correct physical value."""
    hf_row = _make_hf_row(H_kg=1e20, C_kg=1e19, N_kg=0, S_kg=0, Phi_global=0.5)
    _run(hf_row)

    O_total = hf_row['O_kg_total']
    O_atm = hf_row['O_kg_atm']
    O_liquid = hf_row['O_kg_liquid']
    assert O_total == pytest.approx(O_atm + O_liquid, rel=1e-10)

    # Verify physical value: O from H2O + O from CO2
    _, H2O_kg = _expected_species_kg(1e20, 'H')
    _, CO2_kg = _expected_species_kg(1e19, 'C')
    O_from_H2O = H2O_kg * (15.999 / 18.015)
    O_from_CO2 = CO2_kg * (2 * 15.999 / 44.009)
    assert O_total == pytest.approx(O_from_H2O + O_from_CO2, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dummy_outgas_zero_inventory():
    """Zero elemental inventory: zero pressures, zero species masses."""
    hf_row = _make_hf_row(H_kg=0, C_kg=0, N_kg=0, S_kg=0, Phi_global=0.5)
    _run(hf_row)

    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)
    for s in gas_list:
        assert hf_row[f'{s}_kg_atm'] == pytest.approx(0.0, abs=1e-30)
        assert hf_row[f'{s}_bar'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_zero_gravity():
    """Zero gravity: zero pressures but nonzero masses."""
    hf_row = _make_hf_row(H_kg=1e20, gravity=0.0, Phi_global=0.0)
    _run(hf_row)

    assert hf_row['P_surf'] == pytest.approx(0.0, abs=1e-30)
    _, H2O_kg = _expected_species_kg(1e20, 'H')
    assert hf_row['H2O_kg_atm'] == pytest.approx(H2O_kg, rel=1e-10)


@pytest.mark.unit
def test_dummy_outgas_phi_clamped_above():
    """Phi_global > 1 is clamped to 1: equivalent to fully molten."""
    hf_row_clamped = _make_hf_row(H_kg=1e20, Phi_global=1.5)
    hf_row_exact = _make_hf_row(H_kg=1e20, Phi_global=1.0)
    _run(hf_row_clamped)
    _run(hf_row_exact)

    assert hf_row_clamped['P_surf'] == pytest.approx(hf_row_exact['P_surf'], abs=1e-30)
    assert hf_row_clamped['H2O_kg_atm'] == pytest.approx(hf_row_exact['H2O_kg_atm'], abs=1e-30)


@pytest.mark.unit
def test_dummy_outgas_phi_clamped_below():
    """Phi_global < 0 is clamped to 0: equivalent to fully solid."""
    hf_row_clamped = _make_hf_row(H_kg=1e20, Phi_global=-0.5)
    hf_row_exact = _make_hf_row(H_kg=1e20, Phi_global=0.0)
    _run(hf_row_clamped)
    _run(hf_row_exact)

    assert hf_row_clamped['P_surf'] == pytest.approx(hf_row_exact['P_surf'], rel=1e-10)
    assert hf_row_clamped['H2O_kg_atm'] == pytest.approx(hf_row_exact['H2O_kg_atm'], rel=1e-10)
