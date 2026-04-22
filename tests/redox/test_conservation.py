"""
Conservation tests: species-resolved escape debit +
soft-assert_redox_conserved.

Plan v6 §2.6 + §3.1 step 7.
"""
from __future__ import annotations

import pytest

from proteus.redox.budget import RB_COEF
from proteus.redox.conservation import (
    _species_molar_mass,
    assert_redox_conserved,
    debit_escape,
)


@pytest.mark.unit
def test_species_molar_mass_parsing():
    """Verify the formula-parser."""
    from proteus.utils.constants import element_mmw
    assert _species_molar_mass('H2O', element_mmw) == pytest.approx(
        2 * element_mmw['H'] + element_mmw['O'], rel=1e-12,
    )
    assert _species_molar_mass('SiO2', element_mmw) == pytest.approx(
        element_mmw['Si'] + 2 * element_mmw['O'], rel=1e-12,
    )
    assert _species_molar_mass('NH3', element_mmw) == pytest.approx(
        element_mmw['N'] + 3 * element_mmw['H'], rel=1e-12,
    )


@pytest.mark.unit
def test_debit_escape_H2O_is_neutral():
    """H2O escape debits zero (H and O both at reference)."""
    row = {'R_budget_atm': 0.0, 'R_escaped_cum': 0.0}
    delta = debit_escape(row, {'H2O': 1.0e9})
    assert delta == 0.0
    assert row['R_budget_atm'] == 0.0


@pytest.mark.unit
def test_debit_escape_H2_oxidises_atmosphere():
    """H2 escape raises R_atm (atmosphere oxidises as H is reduced)."""
    row = {'R_budget_atm': 0.0, 'R_escaped_cum': 0.0}
    # 1 kg H2 = ~500 mol H2; RB_H2 = -2; ΔR = -(-2)·n = +2·n.
    delta = debit_escape(row, {'H2': 1.0})   # 1 kg
    assert delta > 0.0
    assert row['R_budget_atm'] > 0.0
    assert row['R_escaped_cum'] > 0.0


@pytest.mark.unit
def test_debit_escape_CO2_reduces_atmosphere():
    """CO2 escape lowers R_atm (atmosphere reduces)."""
    row = {'R_budget_atm': 0.0, 'R_escaped_cum': 0.0}
    delta = debit_escape(row, {'CO2': 1.0})
    assert delta < 0.0
    assert row['R_budget_atm'] < 0.0


@pytest.mark.unit
def test_debit_escape_unknown_species_skipped_with_warning(caplog):
    caplog.set_level('WARNING')
    row = {'R_budget_atm': 0.0, 'R_escaped_cum': 0.0}
    debit_escape(row, {'XenonHexafluoride': 1.0})
    assert row['R_budget_atm'] == 0.0
    assert any('unknown species' in r.message.lower() for r in caplog.records)


@pytest.mark.unit
def test_debit_escape_multi_species_summed():
    """Multi-species debit is linear sum of contributions."""
    row = {'R_budget_atm': 0.0, 'R_escaped_cum': 0.0}
    debit_escape(row, {'H2': 1.0, 'CO2': 1.0, 'H2O': 1.0e9})
    row2 = {'R_budget_atm': 0.0, 'R_escaped_cum': 0.0}
    debit_escape(row2, {'H2': 1.0})
    debit_escape(row2, {'CO2': 1.0})
    debit_escape(row2, {'H2O': 1.0e9})
    assert row['R_budget_atm'] == pytest.approx(row2['R_budget_atm'], rel=1e-12)


@pytest.mark.unit
def test_assert_redox_conserved_zero_flux_no_warning(caplog):
    """
    When no escape / dispro flux happens, two identical hf_rows should
    satisfy conservation within machine precision.
    """
    caplog.set_level('WARNING')
    row = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    row.update({f'{s}_mol_liquid': 0.0 for s in RB_COEF})
    row.update({
        'n_Fe3_melt': 0.0, 'n_Fe2_melt': 0.0,
        'n_Fe3_solid_total': 0.0, 'n_Fe2_solid_total': 0.0,
        'n_Fe0_solid_total': 0.0,
        'Fe_kg_core': 1.94e24, 'H_kg_core': 0.0,
        'O_kg_core': 0.0, 'Si_kg_core': 0.0,
        'R_budget_atm': 0.0, 'R_escaped_cum': 0.0,
    })
    import copy
    row_prev = copy.deepcopy(row)
    residual = assert_redox_conserved(row, row_prev, soft_tol=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-15)
    # No warnings should have been emitted.
    assert not any('soft-fail' in r.message.lower() for r in caplog.records)


@pytest.mark.unit
def test_assert_redox_conserved_escape_closure():
    """
    If R_atm falls by exactly ΔR_escape_cum between steps, residual is
    zero.
    """
    base = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    base.update({f'{s}_mol_liquid': 0.0 for s in RB_COEF})
    base.update({
        'n_Fe3_melt': 0.0, 'n_Fe2_melt': 0.0,
        'n_Fe3_solid_total': 0.0, 'n_Fe2_solid_total': 0.0,
        'n_Fe0_solid_total': 0.0,
        'Fe_kg_core': 1.94e24, 'H_kg_core': 0.0,
        'O_kg_core': 0.0, 'Si_kg_core': 0.0,
    })
    # Initial: 1e20 mol H2 in atm, no escape yet.
    row_prev = dict(base)
    row_prev['H2_mol_atm'] = 1.0e20
    row_prev['R_escaped_cum'] = 0.0

    # After: H2 drops by 1e18 mol (escape); R_escaped_cum goes +2e18.
    row_now = dict(base)
    row_now['H2_mol_atm'] = 0.99e20
    row_now['R_escaped_cum'] = +2e18    # ΔR_atm_escape = -(-2)·1e18

    residual = assert_redox_conserved(row_now, row_prev, soft_tol=1e-6)
    assert residual < 1e-12, f'expected closure; got residual={residual}'
