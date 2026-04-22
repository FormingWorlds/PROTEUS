"""
Redox-budget tests: REFERENCE_STATES self-consistency and
Hirschmann 2023 Table 1 numerical anchors.

Plan v6 §2.3 + §5.2.
"""
from __future__ import annotations

import pytest

from proteus.redox.budget import (
    DEFERRED_SPECIES,
    RB_COEF,
    REFERENCE_STATES,
    SPECIES_COMPOSITION,
    _compute_rb_coef,
    redox_budget_atm,
    redox_budget_core,
    redox_budget_mantle,
    redox_budget_species,
    redox_budget_total,
)

# ------------------------------------------------------------------
# Reference-state + convention locks
# ------------------------------------------------------------------


@pytest.mark.unit
def test_reference_states_evans_mantle_like():
    """Reference states match Evans mantle-like convention (plan §2.1)."""
    assert REFERENCE_STATES['Fe'] == 2      # Fe²⁺
    assert REFERENCE_STATES['C'] == 0       # C⁰
    assert REFERENCE_STATES['S'] == -2      # S²⁻
    assert REFERENCE_STATES['H'] == 1       # H¹⁺
    assert REFERENCE_STATES['O'] == -2      # O²⁻
    assert REFERENCE_STATES['N'] == 0       # N⁰ (self-chosen)
    assert REFERENCE_STATES['Si'] == 4      # Si⁴⁺
    assert REFERENCE_STATES['Mg'] == 2      # Mg²⁺
    assert REFERENCE_STATES['Cr'] == 3      # Cr³⁺


@pytest.mark.unit
def test_rb_coef_auto_generated_matches_hardcoded():
    """Every RB_COEF entry must equal the derived value from
    SPECIES_COMPOSITION."""
    for species in SPECIES_COMPOSITION:
        assert RB_COEF[species] == _compute_rb_coef(species), (
            f'{species}: RB_COEF = {RB_COEF[species]} but derivation gives '
            f'{_compute_rb_coef(species)}'
        )


# ------------------------------------------------------------------
# Hirschmann 2023 Table 1 numerical anchors
# ------------------------------------------------------------------


@pytest.mark.unit
def test_hirschmann23_table1_anchors():
    """Key RB factors from Hirschmann 2023 Table 1."""
    assert RB_COEF['CO2'] == 4        # C: 0→+4
    assert RB_COEF['O2'] == 4         # O: -2→0 per atom, 2 atoms → +4
    assert RB_COEF['Fe2O3'] == 2      # 2 Fe at +3 (vs +2 ref) + 3 O at ref
    assert RB_COEF['FeS2'] == 2       # Fe at ref + 2 S at -1 (v=+1 each)
    assert RB_COEF['FeO'] == 0        # all at reference


@pytest.mark.unit
def test_rb_atmospheric_species():
    """Plan v6 §2.3 atmospheric species (locked coefficients)."""
    assert RB_COEF['H2O'] == 0
    assert RB_COEF['H2'] == -2
    assert RB_COEF['CO'] == 2
    assert RB_COEF['CH4'] == -4
    assert RB_COEF['SO2'] == 6        # was WRONG (+4) in earlier plan revs
    assert RB_COEF['H2S'] == 0        # was WRONG (-2) in earlier plan revs
    assert RB_COEF['S2'] == 4
    assert RB_COEF['N2'] == 0
    assert RB_COEF['NH3'] == -3


@pytest.mark.unit
def test_rb_vapor_species():
    """SiO, SiO₂, MgO, FeO₂ — all supported via vap_list."""
    assert RB_COEF['SiO'] == -2       # Si at +2 vs +4 ref: v = -2
    assert RB_COEF['SiO2'] == 0       # Si at +4 = ref
    assert RB_COEF['MgO'] == 0        # Mg at +2 = ref
    assert RB_COEF['FeO2'] == 2       # Fe at +4 vs +2 ref: v = +2


# ------------------------------------------------------------------
# Deferred-species safety
# ------------------------------------------------------------------


@pytest.mark.unit
def test_deferred_species_warn_and_skip(caplog):
    """redox_budget_species must return 0 and warn for deferred species."""
    caplog.set_level('WARNING')
    for deferred in DEFERRED_SPECIES:
        assert redox_budget_species(deferred, 100.0) == 0.0
    assert any('deferred' in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_unknown_species_raises_keyerror():
    with pytest.raises(KeyError):
        redox_budget_species('HelloWorld', 1.0)


# ------------------------------------------------------------------
# Per-reservoir budget integrations
# ------------------------------------------------------------------


@pytest.mark.unit
def test_redox_budget_atm_pure_H2O_is_zero():
    """A pure-H2O atmosphere has R_atm = 0."""
    row = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    row['H2O_mol_atm'] = 1.0e20
    assert redox_budget_atm(row) == 0.0


@pytest.mark.unit
def test_redox_budget_atm_pure_H2_is_reducing():
    """A pure-H2 atmosphere has negative R_atm (reduced)."""
    row = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    row['H2_mol_atm'] = 1.0e20
    assert redox_budget_atm(row) == -2.0 * 1.0e20


@pytest.mark.unit
def test_redox_budget_atm_pure_CO2_is_oxidising():
    """A pure-CO2 atmosphere has positive R_atm (oxidised)."""
    row = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    row['CO2_mol_atm'] = 1.0e20
    assert redox_budget_atm(row) == 4.0 * 1.0e20


@pytest.mark.unit
def test_redox_budget_core_pure_iron_is_strongly_reducing():
    """Earth-like pure-Fe core has R_core ≈ -6.4e25 mol e-."""
    row = {
        'Fe_kg_core': 1.94e24,      # Earth core mass
        'H_kg_core': 0.0, 'O_kg_core': 0.0, 'Si_kg_core': 0.0,
    }
    r = redox_budget_core(row)
    # Expected: -2 · M_core / MW_Fe ≈ -6.95e25 mol e-.
    assert r < 0
    assert r == pytest.approx(-6.9e25, rel=0.05)


@pytest.mark.unit
def test_redox_budget_total_sums_reservoirs():
    """R_total = R_atm + R_mantle + R_core."""
    row = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    row.update({f'{s}_mol_liquid': 0.0 for s in RB_COEF})
    # Populate the redox-scaffolding keys to nonzero so each reservoir
    # contributes something.
    row['H2_mol_atm'] = 1.0e19
    row['n_Fe3_melt'] = 1.0e20
    row['n_Fe2_melt'] = 0.0
    row['n_Fe3_solid_total'] = 0.0
    row['n_Fe0_solid_total'] = 0.0
    row['Fe_kg_core'] = 1.94e24
    row['H_kg_core'] = 0.0
    row['O_kg_core'] = 0.0
    row['Si_kg_core'] = 0.0

    r_total = redox_budget_total(row)
    r_atm = redox_budget_atm(row)
    r_mantle = redox_budget_mantle(row)
    r_core = redox_budget_core(row)
    assert r_total == pytest.approx(r_atm + r_mantle + r_core, rel=1e-12)
    assert r_atm < 0              # pure H2
    assert r_mantle > 0           # pure Fe³⁺ in melt
    assert r_core < 0             # pure Fe⁰
