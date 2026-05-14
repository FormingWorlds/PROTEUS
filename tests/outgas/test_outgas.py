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
    _resolve_oxygen_budget,
    calc_target_elemental_inventories,
    check_desiccation,
    check_ic_oxygen_budget,
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
    # Issue #677: O_mode must be a real string under the new schema.
    # 'ic_chemistry' defers O_kg_total to CALLIOPE so the test remains
    # focused on the calc_target_masses dispatch behaviour.
    config.planet.elements.O_mode = 'ic_chemistry'
    config.planet.volatile_reservoir = 'mantle'
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
    config.planet.elements.O_mode = 'ic_chemistry'
    config.planet.volatile_reservoir = 'mantle'
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
    assert result is False, 'Loss of ~1e21 kg with only 1e15 kg escape budget MUST be refused'


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
def test_check_desiccation_oxygen_prevents_under_d1a():
    """
    Test that O above mass_thresh prevents desiccation (issue #677 fix).

    Pre-fix behaviour: O was ignored on the grounds that mantle FeO/MgO
    buffered an "infinite" atmospheric O reservoir, so an O-rich
    atmosphere was treated as desiccated when H/C/N/S vanished. Under
    D1A whole-planet O accounting, O is a tracked element and an
    atmosphere holding substantial O is not meaningfully desiccated.

    Discriminating: O = 1e22 kg vs threshold 1e16 kg (six orders of
    magnitude over). If the threshold loop silently dropped O the
    result would be True (the old behaviour).
    """
    config = MagicMock()
    config.outgas.mass_thresh = 1e16

    # Only oxygen above threshold, all others below
    hf_row = {}
    for e in element_list:
        if e == 'O':
            hf_row[e + '_kg_total'] = 1e22  # Above threshold
        else:
            hf_row[e + '_kg_total'] = 1e12  # Below threshold

    result = check_desiccation(config, hf_row)
    # Under D1A, O above threshold blocks desiccation.
    assert result is False, (
        'Issue #677 fix: O_kg_total > mass_thresh must prevent desiccation. '
        'If this assertion fails, check whether the O skip was re-introduced '
        'in check_desiccation per-element threshold loop.'
    )

    # Edge case: with O also below threshold, the gate now allows desiccation.
    hf_row['O_kg_total'] = 1e12  # Below threshold
    assert check_desiccation(config, hf_row) is True


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
    config.planet.fO2_source = 'user_constant'

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
    config.planet.fO2_source = 'user_constant'

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
    config.planet.fO2_source = 'user_constant'

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
    config.planet.fO2_source = 'user_constant'

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
    config.planet.fO2_source = 'user_constant'

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


# ============================================================================
# Regression: desiccation must zero the new element mass ratio columns
# ============================================================================


@pytest.mark.unit
def test_expected_keys_includes_element_mass_ratios():
    """The helpfile schema in outgas.common.expected_keys must include the
    same 36 unordered element-pair ratio columns that
    utils.coupler.GetHelpfileKeys registers, so run_desiccated zeros them
    instead of leaving stale ratios from the last outgassing step."""
    from proteus.outgas.common import expected_keys
    from proteus.utils.constants import element_list

    keys = expected_keys()
    ratio_keys = [k for k in keys if '/' in k and k.endswith('_atm')]

    # 9 elements -> C(9, 2) = 36 unordered pairs.
    n_elem = len(element_list)
    expected_count = n_elem * (n_elem - 1) // 2
    assert len(ratio_keys) == expected_count, (
        f'expected {expected_count} ratio keys, got {len(ratio_keys)}: {ratio_keys}'
    )

    # Each (e1, e2) appears exactly once, in either order.
    for e1 in element_list:
        for e2 in element_list:
            if e1 == e2:
                continue
            forward = f'{e2}/{e1}_atm'
            backward = f'{e1}/{e2}_atm'
            assert (forward in keys) ^ (backward in keys), (
                f'pair ({e1}, {e2}) must appear exactly once, not zero or twice'
            )


@pytest.mark.unit
def test_run_desiccated_zeros_element_mass_ratios():
    """Regression: when a planet desiccates, the {e2}/{e1}_atm helpfile
    columns must be reset to 0.0. Before this fix, the keys were absent
    from expected_keys() so run_desiccated left them at their last
    non-zero value, producing physically misleading helpfile rows."""
    from proteus.outgas.wrapper import run_desiccated

    # Pre-populate ratios with non-zero values, as if the previous
    # outgassing step had a CO2-rich atmosphere.
    hf_row = {
        'C/H_atm': 12.0,
        'O/H_atm': 8.0,
        'N/H_atm': 0.7,
        'S/H_atm': 0.05,
        'C/O_atm': 1.5,
        'atm_kg_per_mol': 0.044,
        'H2O_vmr': 0.0,
    }

    config = MagicMock()
    run_desiccated(config, hf_row)

    # All ratio columns must be zero after desiccation.
    for key in ('C/H_atm', 'O/H_atm', 'N/H_atm', 'S/H_atm', 'C/O_atm'):
        assert hf_row[key] == 0.0, f'{key} not zeroed after desiccation: {hf_row[key]}'

    # atm_kg_per_mol is in the excepted_keys list and must survive.
    assert hf_row['atm_kg_per_mol'] == pytest.approx(0.044)


# ============================================================================
# Issue #677 whole-planet oxygen accounting tests
# ============================================================================


@pytest.mark.unit
def test_resolve_oxygen_budget_ppmw_mode():
    """_resolve_oxygen_budget('ppmw') mirrors the H ppmw conversion.

    Physics: 1e5 ppmw = 10 percent by mass relative to the reservoir.
    Discriminating value: M_mantle = 4.04e24 kg (Earth mantle), so the
    expected O_kg is 4.04e23. Off-by-one in the 1e-6 factor would yield
    4.04e17 or 4.04e29 — both visibly wrong.
    """
    config = MagicMock()
    config.planet.elements.O_mode = 'ppmw'
    config.planet.elements.O_budget = 1e5  # ppmw
    config.planet.volatile_reservoir = 'mantle'
    hf_row = {'M_mantle': 4.04e24, 'M_int': 5.97e24}

    o_kg = _resolve_oxygen_budget(config, hf_row)
    assert o_kg == pytest.approx(4.04e23, rel=1e-10)


@pytest.mark.unit
def test_resolve_oxygen_budget_kg_mode_pass_through():
    """O_mode='kg' returns O_budget unchanged."""
    config = MagicMock()
    config.planet.elements.O_mode = 'kg'
    config.planet.elements.O_budget = 3.7e23
    config.planet.volatile_reservoir = 'mantle'
    hf_row = {'M_mantle': 4.04e24, 'M_int': 5.97e24}
    assert _resolve_oxygen_budget(config, hf_row) == pytest.approx(3.7e23)


@pytest.mark.unit
def test_resolve_oxygen_budget_FeO_mantle_wt_pct_mode():
    """FeO_mantle_wt_pct mode: O_kg = M_mantle * wt% * (M_O/M_FeO).

    For Earth mantle (4.04e24 kg) at 10 wt% FeO, the expected O_kg is
    4.04e24 * 0.10 * 0.2227 ≈ 8.998e22 kg. Discriminating: if the M_O/M_FeO
    factor were forgotten the result would be 4.04e23 kg (5x too large).
    """
    config = MagicMock()
    config.planet.elements.O_mode = 'FeO_mantle_wt_pct'
    config.planet.elements.O_budget = 10.0  # wt%
    config.planet.volatile_reservoir = 'mantle'
    hf_row = {'M_mantle': 4.04e24, 'M_int': 5.97e24}

    o_kg = _resolve_oxygen_budget(config, hf_row)
    # M_O = 15.999, M_FeO = 71.844, so M_O/M_FeO = 0.22269...
    expected = 4.04e24 * 0.10 * (15.999 / 71.844)
    assert o_kg == pytest.approx(expected, rel=1e-4)


@pytest.mark.unit
def test_resolve_oxygen_budget_ic_chemistry_mode_returns_none():
    """ic_chemistry mode defers to CALLIOPE; returns None."""
    config = MagicMock()
    config.planet.elements.O_mode = 'ic_chemistry'
    config.planet.elements.O_budget = 0.0
    config.planet.volatile_reservoir = 'mantle'
    hf_row = {'M_mantle': 4.04e24, 'M_int': 5.97e24}
    assert _resolve_oxygen_budget(config, hf_row) is None


@pytest.mark.unit
def test_resolve_oxygen_budget_unknown_mode_raises():
    """Unknown O_mode falls through to ValueError.

    Edge case: the in_() validator on Elements.O_mode normally prevents
    this, but the helper has a defensive fallback so that future mode
    additions surface clearly if a wrapper path is forgotten.
    """
    config = MagicMock()
    config.planet.elements.O_mode = 'not_a_real_mode'
    config.planet.elements.O_budget = 1.0
    config.planet.volatile_reservoir = 'mantle'
    hf_row = {'M_mantle': 4.04e24, 'M_int': 5.97e24}
    with pytest.raises(ValueError, match='Unknown O_mode'):
        _resolve_oxygen_budget(config, hf_row)


@pytest.mark.unit
def test_resolve_oxygen_budget_mantle_plus_core_reservoir():
    """volatile_reservoir='mantle+core' uses M_int (not M_mantle) for ppmw.

    Discriminating: M_int is 1.5x M_mantle for Earth-like core fraction,
    so the O_kg result differs by exactly that factor between the two
    reservoir choices.
    """
    config = MagicMock()
    config.planet.elements.O_mode = 'ppmw'
    config.planet.elements.O_budget = 1e5
    config.planet.volatile_reservoir = 'mantle+core'
    hf_row = {'M_mantle': 4.04e24, 'M_int': 5.97e24}
    # ppmw against M_int now, not M_mantle
    assert _resolve_oxygen_budget(config, hf_row) == pytest.approx(5.97e23, rel=1e-10)


@pytest.mark.unit
def test_check_ic_oxygen_budget_passes_within_tolerance():
    """IC consistency check accepts user budget within 50% of CALLIOPE's.

    Discriminating: user budget 1.0e23 vs chem 1.4e23 has rel divergence
    (1.4 - 1.0) / 1.4 = 0.286 < 0.5, so it passes. The sentinel must be
    flipped to -1 so subsequent calls do not re-fire.
    """
    config = MagicMock()
    config.planet.fO2_source = 'user_constant'
    hf_row = {
        'O_kg_user_ic': 1.0e23,
        'O_kg_total': 1.4e23,  # CALLIOPE's value
    }
    # Should not raise
    check_ic_oxygen_budget(config, hf_row, threshold_frac=0.5)
    # Sentinel flipped so re-call is a no-op
    assert hf_row['O_kg_user_ic'] == pytest.approx(-1.0)


@pytest.mark.unit
def test_check_ic_oxygen_budget_hard_fails_above_threshold():
    """IC consistency check raises ValueError when divergence > 50%.

    Discriminating: user 1e23, chem 1e24 has rel divergence
    (1e24 - 1e23) / 1e24 = 0.9 > 0.5, must hard-fail. Error message
    should name the divergence percentage and the threshold.
    """
    config = MagicMock()
    config.planet.fO2_source = 'user_constant'
    hf_row = {
        'O_kg_user_ic': 1.0e23,
        'O_kg_total': 1.0e24,
    }
    with pytest.raises(ValueError, match='IC oxygen budget mismatch'):
        check_ic_oxygen_budget(config, hf_row, threshold_frac=0.5)


@pytest.mark.unit
def test_check_ic_oxygen_budget_skipped_under_ic_chemistry_sentinel():
    """Sentinel value -1 (ic_chemistry mode) skips the check."""
    config = MagicMock()
    config.planet.fO2_source = 'user_constant'
    hf_row = {
        'O_kg_user_ic': -1.0,
        'O_kg_total': 1.0e30,  # absurdly large, but check is skipped
    }
    # Must not raise
    check_ic_oxygen_budget(config, hf_row, threshold_frac=0.5)
    # Sentinel preserved
    assert hf_row['O_kg_user_ic'] == pytest.approx(-1.0)


@pytest.mark.unit
def test_check_ic_oxygen_budget_skipped_when_chem_zero():
    """Degenerate case: chem O_kg_total = 0 → check skipped.

    Edge case: this can happen if CALLIOPE failed silently or if the
    atmosphere is fully desiccated. We don't want a false divergence
    alarm in those regimes.
    """
    config = MagicMock()
    config.planet.fO2_source = 'user_constant'
    hf_row = {
        'O_kg_user_ic': 1.0e23,
        'O_kg_total': 0.0,
    }
    check_ic_oxygen_budget(config, hf_row, threshold_frac=0.5)
    # Sentinel unchanged (check did nothing)
    assert hf_row['O_kg_user_ic'] == pytest.approx(1.0e23)


@pytest.mark.unit
def test_check_ic_oxygen_budget_skipped_under_path_C():
    """Under planet.fO2_source != 'user_constant' the user O is the
    authoritative input that drives chemistry, so the check is skipped.

    Discriminating: a divergence that would normally hard-fail under
    user_constant must pass cleanly under from_O_budget — and crucially
    must NOT flip the sentinel (the user O is still authoritative and
    re-readable for subsequent diagnostic purposes).
    """
    config = MagicMock()
    config.planet.fO2_source = 'from_O_budget'
    hf_row = {
        'O_kg_user_ic': 1.0e23,
        'O_kg_total': 1.0e24,  # 90% divergence — would fail under user_constant
    }
    check_ic_oxygen_budget(config, hf_row, threshold_frac=0.5)
    # Sentinel must remain intact; we did not run the check at all.
    assert hf_row['O_kg_user_ic'] == pytest.approx(1.0e23)


@pytest.mark.unit
def test_run_outgassing_rejects_from_O_budget_until_wrapper_lands():
    """planet.fO2_source = 'from_O_budget' is accepted by the config
    schema but the runtime path is not yet wired up. run_outgassing
    must refuse with NotImplementedError rather than silently fall
    through to the legacy buffered-fO2 chemistry, which would produce
    different output from what the user requested.

    Discriminating: the error message names the actual fO2_source
    value the user set so the failure is self-explanatory.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.outgas.module = 'calliope'
    config.planet.fO2_source = 'from_O_budget'
    hf_row = {}

    with pytest.raises(NotImplementedError, match='from_O_budget'):
        run_outgassing(dirs, config, hf_row)


@pytest.mark.unit
def test_config_load_rejects_missing_O_mode(tmp_path):
    """Configs without explicit O_mode fail at load (issue #677 hard cutover).

    Regression test for the migration discipline: a TOML that has
    [planet.elements] but no O_mode line must raise a ValueError with
    a clear migration hint. Discriminating: error message must mention
    the four valid modes so users can self-correct without external docs.
    """
    from proteus.config import read_config_object

    # Minimal valid config except for O_mode missing entirely.
    toml_text = """
config_version = "3.0"

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot = 1.0
    volatile_mode = "elements"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 0.5

[outgas]
    fO2_shift_IW = 4
"""
    cfg_path = tmp_path / 'missing_O_mode.toml'
    cfg_path.write_text(toml_text)

    with pytest.raises(ValueError, match='O_mode is required'):
        read_config_object(str(cfg_path))


def _minimal_valid_toml(extra_planet: str = '', extra_elements: str = '') -> str:
    """Build a minimal TOML that passes all other validators; only the
    planet/elements blocks are perturbed by the caller."""
    return f"""
config_version = "3.0"

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot = 1.0
    volatile_mode = "elements"
{extra_planet}
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 0.5
        O_mode   = "ic_chemistry"
        O_budget = 0.0
{extra_elements}

[outgas]
    fO2_shift_IW = 4
"""


def _write_toml(tmp_path, name: str, text: str):
    """Write a TOML string to a tmp_path file and return the path."""
    cfg_path = tmp_path / name
    cfg_path.write_text(text)
    return cfg_path


@pytest.mark.unit
def test_config_rejects_from_mantle_redox_reserved(tmp_path):
    """planet.fO2_source = 'from_mantle_redox' is a reserved enum value
    for the radial Fe3+/Fe2+ framework (issue #653). The runtime is not
    wired so the config-level validator must reject it at load time;
    otherwise users would silently fall through to legacy behaviour
    that doesn't match what they asked for.

    Discriminating: error message must mention issue #653 so users can
    find the upstream tracking.
    """
    from proteus.config import read_config_object

    cfg_path = _write_toml(
        tmp_path,
        'reserved.toml',
        _minimal_valid_toml(extra_planet='    fO2_source = "from_mantle_redox"\n'),
    )

    with pytest.raises(ValueError, match='issue #653'):
        read_config_object(str(cfg_path))


@pytest.mark.unit
def test_config_rejects_from_O_budget_with_ic_chemistry(tmp_path):
    """planet.fO2_source = 'from_O_budget' requires an authoritative O
    budget. O_mode = 'ic_chemistry' defers O to the chemistry solver,
    which contradicts Path C — there is nothing to invert against.

    Discriminating: error message names both fields so the user knows
    which one to change.
    """
    from proteus.config import read_config_object

    # Note _minimal_valid_toml's default O_mode is ic_chemistry
    cfg_path = _write_toml(
        tmp_path,
        'from_O_budget_ic_chem.toml',
        _minimal_valid_toml(extra_planet='    fO2_source = "from_O_budget"\n'),
    )

    with pytest.raises(ValueError, match='ic_chemistry'):
        read_config_object(str(cfg_path))


@pytest.mark.unit
def test_config_rejects_from_O_budget_with_gas_prs(tmp_path):
    """planet.fO2_source = 'from_O_budget' requires
    planet.volatile_mode = 'elements' so the O inventory is derivable
    from planet.elements.O_budget. Under volatile_mode = 'gas_prs' the
    user supplies partial pressures directly and the element budgets
    are inert, so Path C has no O target to invert against.

    Discriminating: error message must name volatile_mode and gas_prs
    so the user knows which field to change.
    """
    from proteus.config import read_config_object

    toml_text = """
config_version = "3.0"

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot = 1.0
    volatile_mode = "gas_prs"
    fO2_source = "from_O_budget"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 0.5
        O_mode   = "kg"
        O_budget = 1.0e21
    [planet.gas_prs]
        H2O = 50.0
        CO2 = 5.0
        N2  = 1.0
        S2  = 0.1

[outgas]
    fO2_shift_IW = 0
"""
    cfg_path = _write_toml(tmp_path, 'from_O_budget_gas_prs.toml', toml_text)

    with pytest.raises(ValueError, match='gas_prs'):
        read_config_object(str(cfg_path))


@pytest.mark.unit
@pytest.mark.parametrize(
    'o_mode,o_budget', [('ppmw', 500.0), ('kg', 1.0e21), ('FeO_mantle_wt_pct', 8.0)]
)
def test_config_accepts_from_O_budget_with_concrete_O_mode(tmp_path, o_mode, o_budget):
    """Every concrete O budget mode pairs with from_O_budget. Schema must
    accept all three (ppmw, kg, FeO_mantle_wt_pct) — the runtime gate
    in run_outgassing will refuse for now, but the validator must not.

    Discriminating: parametrize across all three modes so a regression
    that breaks one (e.g., kg-mode unit conversion blunder) is caught.
    """
    from proteus.config import read_config_object

    toml_text = f"""
config_version = "3.0"

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot = 1.0
    volatile_mode = "elements"
    fO2_source = "from_O_budget"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 0.5
        O_mode   = "{o_mode}"
        O_budget = {o_budget}

[outgas]
    fO2_shift_IW = 0
"""
    cfg_path = _write_toml(tmp_path, f'from_O_budget_{o_mode}.toml', toml_text)
    cfg = read_config_object(str(cfg_path))
    assert cfg.planet.fO2_source == 'from_O_budget'
    assert cfg.planet.elements.O_mode == o_mode
    assert cfg.planet.elements.O_budget == pytest.approx(o_budget)


@pytest.mark.unit
def test_config_fO2_source_default_is_user_constant(tmp_path):
    """Configs that omit fO2_source default to 'user_constant', preserving
    the legacy behaviour for the 108 already-migrated TOMLs without
    requiring a second migration pass.

    Strong-form assertion: also confirm the rest of the minimal config
    parsed correctly (mass_tot, O_mode), so a regression that breaks the
    TOML parse but leaves fO2_source readable does not pass silently.
    """
    from proteus.config import read_config_object

    cfg_path = _write_toml(
        tmp_path,
        'default_fO2_source.toml',
        _minimal_valid_toml(),
    )
    cfg = read_config_object(str(cfg_path))

    assert cfg.planet.fO2_source == 'user_constant'
    assert cfg.planet.elements.O_mode == 'ic_chemistry'
    assert cfg.planet.mass_tot == pytest.approx(1.0)


@pytest.mark.unit
def test_config_rejects_unknown_fO2_source_value(tmp_path):
    """The in_() validator rejects any value not in the three-element
    enum, so a typo (e.g., 'user_const') produces a clear error rather
    than silent fallback.

    Discriminating: ``match`` pins the error to the in_() validator
    output containing the rejected value, so the test cannot pass on a
    coincidental ValueError from a different validator (e.g., the
    O_mode check above it in the chain).
    """
    from proteus.config import read_config_object

    cfg_path = _write_toml(
        tmp_path,
        'typo.toml',
        _minimal_valid_toml(extra_planet='    fO2_source = "user_const"\n'),
    )

    with pytest.raises(ValueError, match='user_const'):
        read_config_object(str(cfg_path))


@pytest.mark.unit
def test_config_warns_when_fO2_shift_IW_set_with_from_O_budget(tmp_path):
    """When the user sets a non-default outgas.fO2_shift_IW alongside
    planet.fO2_source = 'from_O_budget', the buffer offset is *derived*
    at runtime, so the configured value is silently ignored. Emit a
    UserWarning so the misconfiguration surfaces.

    Discriminating: the warning message must name fO2_shift_IW and
    from_O_budget so the user knows which to remove.
    """
    from proteus.config import read_config_object

    toml_text = """
config_version = "3.0"

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot = 1.0
    volatile_mode = "elements"
    fO2_source = "from_O_budget"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 0.5
        O_mode   = "kg"
        O_budget = 1.0e21

[outgas]
    fO2_shift_IW = 4.0
"""
    cfg_path = _write_toml(tmp_path, 'redundant_fO2.toml', toml_text)

    with pytest.warns(UserWarning, match='from_O_budget'):
        cfg = read_config_object(str(cfg_path))
    # The config still loads — this is a warning, not an error.
    assert cfg.planet.fO2_source == 'from_O_budget'
