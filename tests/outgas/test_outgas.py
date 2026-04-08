"""
Unit tests for proteus.outgas.wrapper module

This module orchestrates volatile outgassing calculations, including:
- Elemental inventory calculations (total volatile masses)
- Desiccation detection (below mass threshold)
- Surface pressure calculations from outgassing models (CALLIOPE)
- Atmosphere mass aggregation from volatile reservoirs
- Output formatting and logging

Physics tested:
- Volatile conservation (element_list: H, O, C, N, S, Si, Mg, Fe, Na)
- Desiccation threshold logic (config.outgas.mass_thresh)
- Pressure-mass-VMR relationships (ideal gas approximation)
- Mean molecular weight calculation (atm_kg_per_mol)

Related documentation:
- docs/test_infrastructure.md: Testing standards and structure
- docs/test_categorization.md: Test classification criteria
- docs/test_building.md: Best practices for test implementation

Mocking strategy:
- Mock all CALLIOPE functions (calc_surface_pressures, calc_target_masses) to isolate wrapper logic
- Use real constants from proteus.utils.constants (element_list, gas_list)
- Mock logging to avoid console spam during tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proteus.outgas.wrapper import (
    calc_target_elemental_inventories,
    check_desiccation,
    run_desiccated,
    run_outgassing,
)
from proteus.utils.constants import element_list, gas_list


@pytest.mark.unit
def test_calc_target_elemental_inventories_calliope_enabled():
    """
    Test that calc_target_elemental_inventories calls CALLIOPE when module='calliope'.

    Physics: Total volatile masses (H_kg_total, C_kg_total, etc.) are calculated by
    CALLIOPE based on planet composition and partitioning between reservoirs.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'calliope'
    hf_row = {'Time': 0.0}

    with patch('proteus.outgas.calliope.calc_target_masses') as mock_calc:
        calc_target_elemental_inventories(dirs, config, hf_row)
        mock_calc.assert_called_once_with(dirs, config, hf_row)


@pytest.mark.unit
def test_calc_target_elemental_inventories_always_calls_calc_target_masses():
    """
    Test that calc_target_elemental_inventories always computes element budgets.

    Physics: Element budgets (H, C, N, S) are needed by ALL outgas modules,
    not just CALLIOPE, because they drive the mass balance. The function
    always calls calc_target_masses regardless of the outgas module.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'none'
    hf_row = {'Time': 0.0}

    with patch('proteus.outgas.calliope.calc_target_masses') as mock_calc:
        calc_target_elemental_inventories(dirs, config, hf_row)
        mock_calc.assert_called_once_with(dirs, config, hf_row)


@pytest.mark.unit
def test_check_desiccation_not_desiccated():
    """
    Test check_desiccation returns False when volatiles remain above threshold.

    Physics: Planet still has volatile inventory (H_kg_total > 1e16 kg).
    Typical for early Earth with ~1 ocean mass (1.4e21 kg H2O ≈ 1.6e20 kg H).
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16  # 10 petagrams threshold

    # Construct hf_row with all elements above threshold (except O, which is ignored)
    hf_row = {}
    for e in element_list:
        if e == 'O':
            hf_row[e + '_kg_total'] = 1e15  # Below threshold, but ignored
        else:
            hf_row[e + '_kg_total'] = 1e18  # Well above threshold

    result = check_desiccation(config, hf_row)
    assert result is False


@pytest.mark.unit
def test_check_desiccation_fully_desiccated():
    """
    Test check_desiccation returns True when all volatiles (except O) below threshold.

    Physics: Complete desiccation after long-term escape (Mars-like scenario).
    All H, C, N, S depleted to negligible amounts. With no escape baseline
    tracked (M_vol_initial absent or zero), the function falls back to the
    threshold-only behaviour.
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16  # 10 petagrams

    # All elements below threshold; no M_vol_initial baseline -> fallback path.
    hf_row = {}
    for e in element_list:
        hf_row[e + '_kg_total'] = 1e12  # 1 teragram << threshold

    result = check_desiccation(config, hf_row)
    assert result is True


@pytest.mark.unit
def test_check_desiccation_refused_when_loss_unexplained():
    """
    Test the escape-balance gate: when M_ele has collapsed but cumulative
    escape cannot account for it, check_desiccation must REFUSE to declare
    desiccation. This is the regression guard for the CHILI sweep R7/R21
    cascade where an AGNI failure wiped the atmosphere as a side effect and
    the old check_desiccation laundered the failure into a fake "solidified
    + desiccated" exit.

    Discriminating values: initial inventory = 1e21 kg, escaped = 1e15 kg,
    current = 1e10 kg. Loss = 1e21 - 1e10 ≈ 1e21 kg, which is 1e6x larger
    than the cumulative escape can explain. The 1.5x slack and 1e3 floor
    do not save it.
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    hf_row = {}
    for e in element_list:
        if e == 'O':
            hf_row[e + '_kg_total'] = 1e22
        else:
            hf_row[e + '_kg_total'] = 1e10  # well below threshold
    hf_row['M_vol_initial'] = 1e21  # 1 EO H equivalent
    hf_row['esc_kg_cumulative'] = 1e15  # only 0.1% of inventory could be lost

    result = check_desiccation(config, hf_row)
    assert result is False, (
        'Loss of ~1e21 kg with only 1e15 kg escape budget MUST be refused'
    )


@pytest.mark.unit
def test_check_desiccation_allowed_when_loss_matches_escape():
    """
    Test that legitimate desiccation (loss accounted for by escape) is still
    accepted by the gate. This is the positive case the gate must NOT block.

    Initial inventory = 1e16 kg, escaped = 1e16 kg over the run, current ~ 0.
    Lost = 1e16 ≈ esc_cum, ratio ~ 1.0, well within the 1.5x tolerance.
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    hf_row = {}
    for e in element_list:
        hf_row[e + '_kg_total'] = 1e10  # below threshold
    hf_row['M_vol_initial'] = 1e16
    hf_row['esc_kg_cumulative'] = 1.1e16  # slightly more than initial

    result = check_desiccation(config, hf_row)
    assert result is True, 'Loss matching cumulative escape must be allowed'


@pytest.mark.unit
def test_check_desiccation_gate_inactive_without_baseline():
    """
    Test that the gate falls back to threshold-only behaviour when
    M_vol_initial is missing or zero (e.g. resuming from an old CSV that
    predates the baseline tracking, or escape never ran).

    Without a baseline we cannot reason about whether the loss is real, so
    we trust the threshold check (preserving backward compatibility with
    runs whose helpfile schema has no M_vol_initial column).
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    hf_row = {}
    for e in element_list:
        hf_row[e + '_kg_total'] = 1e10  # below threshold

    # Case A: key absent
    assert check_desiccation(config, dict(hf_row)) is True

    # Case B: key present but zero
    case_b = dict(hf_row, M_vol_initial=0.0, esc_kg_cumulative=0.0)
    assert check_desiccation(config, case_b) is True

    # Case C: key present but NaN (e.g. cast from missing pandas column)
    import math
    case_c = dict(hf_row, M_vol_initial=math.nan, esc_kg_cumulative=0.0)
    assert check_desiccation(config, case_c) is True


@pytest.mark.unit
def test_check_desiccation_gate_with_one_t_floor():
    """
    Test the absolute floor (1e3 kg) on the gate. With M_vol_initial near
    the threshold and escape near zero, a tiny rounding-noise loss
    (~ a few hundred kg) must NOT trip the refusal.

    Edge case: cur_m_ele = M_vol_initial - 500 kg. lost = 500 kg, esc_cum = 0.
    Predicate: 500 > 1.5 * 0 + 1000  →  False  →  desiccation allowed.
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    hf_row = {}
    for e in element_list:
        if e == 'H':
            hf_row[e + '_kg_total'] = 1e10  # tiny but the only contributor
        else:
            hf_row[e + '_kg_total'] = 0.0
    # All elements are below threshold so the threshold check passes.
    hf_row['M_vol_initial'] = 1e10 + 500.0  # 500 kg of "rounding noise"
    hf_row['esc_kg_cumulative'] = 0.0

    # 500 kg loss is below the 1e3 absolute floor, so the gate must allow it.
    result = check_desiccation(config, hf_row)
    assert result is True


@pytest.mark.unit
def test_check_desiccation_oxygen_ignored():
    """
    Test that check_desiccation ignores oxygen (abundant in silicate rocks).

    Physics: Oxygen is not a volatile tracer (locked in mantle oxides MgO, FeO, SiO2).
    Only H, C, N, S, Na define volatile inventory for desiccation check.
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    # Only oxygen above threshold, all others below
    hf_row = {}
    for e in element_list:
        if e == 'O':
            hf_row[e + '_kg_total'] = 1e22  # Huge amount (mantle oxygen)
        else:
            hf_row[e + '_kg_total'] = 1e12  # Below threshold

    result = check_desiccation(config, hf_row)
    assert result is True  # Still desiccated (O doesn't count)


@pytest.mark.unit
def test_check_desiccation_single_element_prevents():
    """
    Test that a single element above threshold prevents desiccation.

    Physics: Even if H, C, N are lost, S remaining in SO2 atmosphere prevents desiccation
    (Venus-like scenario with sulfuric acid clouds).
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    # Only S above threshold
    hf_row = {}
    for e in element_list:
        if e == 'S':
            hf_row[e + '_kg_total'] = 1e18  # Sulfur-rich atmosphere
        else:
            hf_row[e + '_kg_total'] = 1e12  # Others depleted

    result = check_desiccation(config, hf_row)
    assert result is False  # Not desiccated (S remains)


@pytest.mark.unit
def test_run_outgassing_calliope_calculation():
    """
    Test run_outgassing calls CALLIOPE and aggregates atmosphere mass.

    Physics: CALLIOPE calculates equilibrium partitioning between melt and atmosphere
    (Henry's law, Redlich-Kwong EOS). Wrapper sums volatile masses to get M_atm.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'calliope'

    # Setup hf_row with gas masses and VMR (typical early Earth steam atmosphere)
    hf_row = {}
    for s in gas_list:
        hf_row[s + '_kg_atm'] = 1e19 if s == 'H2O' else 1e16  # H2O-dominated
        hf_row[s + '_vmr'] = 0.95 if s == 'H2O' else 0.01  # Volume mixing ratios
        hf_row[s + '_bar'] = 250.0 if s == 'H2O' else 5.0  # Partial pressures

    hf_row['P_surf'] = 300.0  # Total surface pressure (bar)
    hf_row['atm_kg_per_mol'] = 0.018  # Mean molecular weight (kg/mol) for steam

    with patch('proteus.outgas.calliope.calc_surface_pressures') as mock_calc:
        run_outgassing(dirs, config, hf_row)

        # Verify CALLIOPE was called
        mock_calc.assert_called_once_with(dirs, config, hf_row)

        # Verify atmosphere mass aggregation: M_atm = sum of all gas masses
        expected_M_atm = sum(hf_row[s + '_kg_atm'] for s in gas_list)
        assert hf_row['M_atm'] == pytest.approx(expected_M_atm, rel=1e-10)


@pytest.mark.unit
def test_run_outgassing_atmosphere_mass_conservation():
    """
    Test that M_atm is correctly summed from individual gas species.

    Physics: Mass conservation requires M_atm = Σ m_i (sum over all species).
    No creation/destruction of mass during pressure calculation.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'calliope'

    # Setup with known gas masses for all gases in gas_list
    hf_row = {}
    # gas_list has 15 elements, use decreasing masses
    gas_masses = [
        1e20,
        5e19,
        2e19,
        1e18,
        5e17,
        3e17,
        1e17,
        5e16,
        1e16,
        1e15,
        1e14,
        1e13,
        1e12,
        1e11,
        1e10,
    ]
    for i, s in enumerate(gas_list):
        hf_row[s + '_kg_atm'] = gas_masses[i]
        hf_row[s + '_vmr'] = 0.1 / len(gas_list)  # Equal fractional mixing
        hf_row[s + '_bar'] = 10.0 / len(gas_list)  # Equal partial pressure

    hf_row['P_surf'] = 100.0
    hf_row['atm_kg_per_mol'] = 0.029  # N2-like

    with patch('proteus.outgas.calliope.calc_surface_pressures'):
        run_outgassing(dirs, config, hf_row)

        # Check mass conservation with tolerance for floating point
        expected_M_atm = sum(gas_masses)
        assert hf_row['M_atm'] == pytest.approx(expected_M_atm, rel=1e-10)


@pytest.mark.unit
def test_run_outgassing_disabled_module():
    """
    Test run_outgassing with module='none' (disabled) - should still aggregate M_atm.

    Physics: Even without outgassing model, mass aggregation is required for other modules
    (atmosphere, escape) that use M_atm.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'none'  # Disabled

    # Setup with fixed gas masses
    hf_row = {}
    for s in gas_list:
        hf_row[s + '_kg_atm'] = 1e18  # Fixed 1 exagram per species
        hf_row[s + '_vmr'] = 1.0 / len(gas_list)  # Equal mixing
        hf_row[s + '_bar'] = 10.0

    hf_row['P_surf'] = 90.0
    hf_row['atm_kg_per_mol'] = 0.044  # CO2-like

    with patch('proteus.outgas.calliope.calc_surface_pressures') as mock_calc:
        run_outgassing(dirs, config, hf_row)

        # CALLIOPE should not be called
        mock_calc.assert_not_called()

        # M_atm should still be aggregated
        expected_M_atm = len(gas_list) * 1e18
        assert hf_row['M_atm'] == expected_M_atm


@pytest.mark.unit
def test_run_desiccated_zeros_outgassing_keys():
    """
    Test run_desiccated sets most outgassing keys to zero after desiccation.

    Physics: Desiccated planet has no atmosphere (P_surf = 0, all gas masses = 0).
    Must avoid divide-by-zero elsewhere (preserve atm_kg_per_mol, VMRs).
    """
    config = MagicMock()

    # Setup hf_row with outgassing data
    hf_row = {}
    for s in gas_list:
        hf_row[s + '_kg_atm'] = 1e18  # Non-zero before desiccation
        hf_row[s + '_vmr'] = 0.1  # VMR preserved to avoid divide-by-zero
        hf_row[s + '_bar'] = 10.0

    hf_row['P_surf'] = 90.0
    hf_row['atm_kg_per_mol'] = 0.029  # MMW preserved to avoid divide-by-zero
    hf_row['M_atm'] = 1e20

    run_desiccated(config, hf_row)

    # Check that pressure and masses are zeroed
    assert hf_row['P_surf'] == 0.0
    assert hf_row['M_atm'] == 0.0
    for s in gas_list:
        assert hf_row[s + '_kg_atm'] == 0.0
        assert hf_row[s + '_bar'] == 0.0

    # Check that excepted keys are NOT zeroed (prevent divide-by-zero downstream)
    assert hf_row['atm_kg_per_mol'] == 0.029  # Preserved
    for s in gas_list:
        assert hf_row[s + '_vmr'] == 0.1  # Preserved


@pytest.mark.unit
def test_run_desiccated_preserves_critical_keys():
    """
    Test that run_desiccated preserves atm_kg_per_mol and VMRs to avoid crashes.

    Physics: Other modules (atmosphere, escape) may use these in denominators.
    Setting to zero would cause division errors. Preserve last valid values.
    """
    config = MagicMock()

    # Setup with specific values for all gases in gas_list
    hf_row = {
        'atm_kg_per_mol': 0.018,  # H2O-dominated before desiccation
        'P_surf': 100.0,
        'M_atm': 1e20,
    }

    # Add VMR and mass for all gases in gas_list
    for s in gas_list:
        if s == 'H2O':
            hf_row[s + '_vmr'] = 0.9
        elif s == 'CO2':
            hf_row[s + '_vmr'] = 0.08
        elif s == 'N2':
            hf_row[s + '_vmr'] = 0.02
        else:
            hf_row[s + '_vmr'] = 0.0
        hf_row[s + '_kg_atm'] = 1e17
        hf_row[s + '_bar'] = 5.0

    original_mmw = hf_row['atm_kg_per_mol']
    original_vmrs = {s: hf_row[s + '_vmr'] for s in gas_list}

    run_desiccated(config, hf_row)

    # Verify preservation
    assert hf_row['atm_kg_per_mol'] == original_mmw
    for s in gas_list:
        assert hf_row[s + '_vmr'] == original_vmrs[s]


@pytest.mark.unit
def test_run_outgassing_zero_atmosphere_mass():
    """
    Test run_outgassing with all gas masses zero (edge case).

    Physics: M_atm = 0 represents bare rock surface (no atmosphere).
    Should not crash when summing zero masses.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'calliope'

    # All gases have zero mass
    hf_row = {}
    for s in gas_list:
        hf_row[s + '_kg_atm'] = 0.0
        hf_row[s + '_vmr'] = 0.0
        hf_row[s + '_bar'] = 0.0

    hf_row['P_surf'] = 0.0
    hf_row['atm_kg_per_mol'] = 0.029  # Arbitrary (not used when M_atm=0)

    with patch('proteus.outgas.calliope.calc_surface_pressures'):
        run_outgassing(dirs, config, hf_row)

        # M_atm should be exactly zero
        assert hf_row['M_atm'] == 0.0


@pytest.mark.unit
def test_check_desiccation_at_exact_threshold():
    """
    Test check_desiccation behavior when volatile mass at or below threshold (edge case).

    Physics: At threshold, planet is considered desiccated (>= not >).
    Only values strictly greater than threshold prevent desiccation.
    This is conservative (requires complete inventory loss for termination).
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    # All elements AT or below threshold (none strictly above)
    hf_row = {}
    for e in element_list:
        if e == 'O':
            hf_row[e + '_kg_total'] = 0.0  # Ignored (never checked)
        elif e == 'H':
            hf_row[e + '_kg_total'] = 1e16  # Exactly at threshold (not > threshold)
        else:
            hf_row[e + '_kg_total'] = 5e15  # Below threshold

    result = check_desiccation(config, hf_row)

    # Current implementation uses >, so nothing > threshold means desiccated
    assert result is True


@pytest.mark.unit
def test_run_outgassing_mixed_species_dominance():
    """
    Test run_outgassing with realistic mixed-species atmosphere (Earth-like).

    Physics: Modern Earth has N2 (78%), O2 (21%), Ar (1%), CO2 (0.04%).
    Test validates correct aggregation with diverse masses and VMRs.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'calliope'

    # Earth-like composition (simplified to available gases)
    # Assume: N2 dominant, O2 secondary, H2O/CO2 trace, rest negligible
    hf_row = {
        'N2_kg_atm': 3.86e18,  # 78% by volume
        'O2_kg_atm': 1.05e18,  # 21% by volume
        'H2O_kg_atm': 1.3e16,  # Trace amount
        'CO2_kg_atm': 3e15,  # 400 ppm
        'H2_kg_atm': 0.0,
        'CO_kg_atm': 0.0,
        'CH4_kg_atm': 0.0,
        'NH3_kg_atm': 0.0,
        'S2_kg_atm': 0.0,
        'SO2_kg_atm': 0.0,
        'H2S_kg_atm': 0.0,
        'SiO_kg_atm': 0.0,
        'SiO2_kg_atm': 0.0,
        'MgO_kg_atm': 0.0,
        'FeO2_kg_atm': 0.0,
    }

    for s in gas_list:
        if s in ['N2', 'O2']:
            hf_row[s + '_vmr'] = 0.5  # ~50% each for simplicity
            hf_row[s + '_bar'] = 0.5
        elif s == 'H2O':
            hf_row[s + '_vmr'] = 0.002
            hf_row[s + '_bar'] = 0.002
        elif s == 'CO2':
            hf_row[s + '_vmr'] = 0.0004
            hf_row[s + '_bar'] = 0.0004
        else:
            hf_row[s + '_vmr'] = 0.0
            hf_row[s + '_bar'] = 0.0

    hf_row['P_surf'] = 1.0  # 1 bar
    hf_row['atm_kg_per_mol'] = 0.029  # N2/O2-dominated

    with patch('proteus.outgas.calliope.calc_surface_pressures'):
        run_outgassing(dirs, config, hf_row)

        # Check mass conservation
        expected_M_atm = sum(hf_row[s + '_kg_atm'] for s in gas_list)
        assert hf_row['M_atm'] == pytest.approx(expected_M_atm, rel=1e-10)

        # M_atm should be dominated by N2+O2
        assert hf_row['M_atm'] > 4.9e18  # Mostly N2+O2 mass
