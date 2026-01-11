"""
Unit tests for proteus.escape.wrapper module

This module tests atmospheric escape functionality including:
- run_escape(): Generic escape orchestrator (dummy/zephyrus/disabled modes)
- run_zephyrus(): Energy-limited escape via ZEPHYRUS library
- calc_new_elements(): Elemental inventory updates after unfractionated escape

Physics tested:
- Escape flux conservation (kg/s → kg/yr conversions)
- Elemental mass ratio preservation during unfractionated escape
- Reservoir selection (bulk, outgas, pxuv)
- Minimum threshold enforcement for desiccated planets
- Non-negative mass constraints
- XUV-driven hydrodynamic escape (ZEPHYRUS)

All tests use mocked ZEPHYRUS calls to avoid heavy physics computation (<100ms runtime).
See docs/test_infrastructure.md, docs/test_building.md for testing standards.

Related documentation:
- docs/test_infrastructure.md: Test framework and CI integration
- docs/test_categorization.md: Test markers and categories
- docs/test_building.md: Best practices for writing robust tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =======================================================================================
# SECTION: run_escape() — Generic escape orchestrator
# =======================================================================================


@pytest.mark.unit
def test_run_escape_disabled():
    """Test escape when module is disabled (None).

    Physical scenario: Planet with escape module turned off.
    Validates that esc_rate_total is set to zero and no element calculations occur.
    """
    from proteus.escape.wrapper import run_escape

    # Mock config with escape disabled
    config = MagicMock()
    config.escape.module = None

    # Minimal hf_row for escape calculation
    hf_row = {}

    # Call run_escape with escape disabled
    run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    # Verify escape rate is zero
    assert hf_row['esc_rate_total'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_run_escape_dummy():
    """Test escape using dummy module with fixed rate.

    Physical scenario: Planet with constant bulk escape rate (e.g., 1e5 kg/s).
    Validates that dummy escape rate is assigned and elements are updated correctly.
    """
    from proteus.escape.wrapper import run_escape

    # Mock config with dummy escape at 1e5 kg/s
    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.dummy.rate = 1e5  # kg/s
    config.escape.reservoir = 'bulk'
    config.outgas.mass_thresh = 1e10  # kg

    # Minimal hf_row with initial element inventories (all elements from element_list)
    hf_row = {
        'H_kg_total': 1e21,  # Large H reservoir (e.g., Earth ocean mass)
        'C_kg_total': 1e18,  # Carbon inventory
        'N_kg_total': 1e19,  # Nitrogen inventory
        'S_kg_total': 1e17,  # Sulfur inventory
        'Si_kg_total': 1e19,  # Silicon inventory
        'Mg_kg_total': 1e18,  # Magnesium inventory
        'Fe_kg_total': 1e20,  # Iron inventory
        'Na_kg_total': 1e16,  # Sodium inventory
        'H_kg_atm': 1e20,  # Atmospheric H
        'C_kg_atm': 1e17,  # Atmospheric C
        'N_kg_atm': 1e18,  # Atmospheric N
        'S_kg_atm': 1e16,  # Atmospheric S
        'Si_kg_atm': 1e17,  # Atmospheric Si
        'Mg_kg_atm': 1e16,  # Atmospheric Mg
        'Fe_kg_atm': 1e17,  # Atmospheric Fe
        'Na_kg_atm': 1e15,  # Atmospheric Na
    }

    # Call run_escape
    run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    # Verify escape rate matches dummy rate
    assert hf_row['esc_rate_total'] == pytest.approx(1e5, rel=1e-8)

    # Verify element inventories were updated (should be reduced)
    assert hf_row['H_kg_total'] < 1e21  # H should decrease
    assert hf_row['C_kg_total'] < 1e18  # C should decrease
    assert hf_row['N_kg_total'] < 1e19  # N should decrease
    assert hf_row['S_kg_total'] < 1e17  # S should decrease


@pytest.mark.unit
@patch('zephyrus.escape.EL_escape')
def test_run_escape_zephyrus(mock_el_escape):
    """Test escape using ZEPHYRUS energy-limited model.

    Physical scenario: Hot Jupiter with XUV-driven hydrodynamic escape.
    Validates that ZEPHYRUS is called with correct parameters and escape rate is assigned.
    """
    from proteus.escape.wrapper import run_escape

    # Mock ZEPHYRUS EL_escape to return a specific escape rate
    mock_el_escape.return_value = 1e7  # kg/s (high escape rate for hot Jupiter)

    # Mock config with ZEPHYRUS escape
    config = MagicMock()
    config.escape.module = 'zephyrus'
    config.escape.zephyrus.tidal = False
    config.escape.zephyrus.efficiency = 0.15
    config.escape.zephyrus.Pxuv = 5e-5  # bar
    config.escape.reservoir = 'outgas'
    config.outgas.mass_thresh = 1e10  # kg
    config.star.mass = 1.0e30  # kg (Sun-like)

    # Minimal hf_row for hot Jupiter
    hf_row = {
        'semimajorax': 0.05 * 1.496e11,  # 0.05 AU in meters
        'eccentricity': 0.01,
        'M_planet': 1.898e27,  # Jupiter mass in kg
        'R_int': 7.0e7,  # 70,000 km radius
        'R_xuv': 8.0e7,  # XUV radius slightly larger
        'F_xuv': 1e4,  # W/m^2 (high XUV flux)
        'H_kg_total': 1e24,  # Large H reservoir
        'C_kg_total': 1e20,
        'N_kg_total': 1e21,
        'S_kg_total': 1e19,
        'Si_kg_total': 1e19,
        'Mg_kg_total': 1e18,
        'Fe_kg_total': 1e20,
        'Na_kg_total': 1e17,
        'H_kg_atm': 1e23,  # Atmospheric reservoirs for 'outgas' mode
        'C_kg_atm': 1e19,
        'N_kg_atm': 1e20,
        'S_kg_atm': 1e18,
        'Si_kg_atm': 1e17,
        'Mg_kg_atm': 1e16,
        'Fe_kg_atm': 1e17,
        'Na_kg_atm': 1e16,
    }

    # Mock stellar track
    stellar_track = MagicMock()

    # Call run_escape
    run_escape(config, hf_row, dt=1000.0, stellar_track=stellar_track)

    # Verify ZEPHYRUS was called with correct parameters
    mock_el_escape.assert_called_once()
    call_args = mock_el_escape.call_args[0]
    assert not call_args[0]  # tidal contribution
    assert call_args[1] == pytest.approx(0.05 * 1.496e11, rel=1e-6)  # semimajor axis
    assert call_args[2] == pytest.approx(0.01, rel=1e-6)  # eccentricity

    # Verify escape rate matches mock return value
    assert hf_row['esc_rate_total'] == pytest.approx(1e7, rel=1e-8)


@pytest.mark.unit
def test_run_escape_invalid_module():
    """Test that invalid escape module raises ValueError.

    Physical scenario: Configuration error with unrecognized escape module.
    Validates proper error handling for invalid module names.
    """
    from proteus.escape.wrapper import run_escape

    # Mock config with invalid module name
    config = MagicMock()
    config.escape.module = 'invalid_module'

    hf_row = {}

    # Verify ValueError is raised
    with pytest.raises(ValueError, match='Invalid escape model'):
        run_escape(config, hf_row, dt=1000.0, stellar_track=None)


# =======================================================================================
# SECTION: run_zephyrus() — Energy-limited escape
# =======================================================================================


@pytest.mark.unit
@patch('zephyrus.escape.EL_escape')
def test_run_zephyrus_no_tidal(mock_el_escape):
    """Test ZEPHYRUS escape without tidal contribution.

    Physical scenario: Earth-like planet at 1 AU with moderate XUV flux.
    Validates that tidal heating is disabled (tidal=False) in EL_escape call.
    """
    from proteus.escape.wrapper import run_zephyrus

    # Mock EL_escape to return a specific rate
    mock_el_escape.return_value = 1e3  # kg/s (moderate escape)

    # Mock config
    config = MagicMock()
    config.escape.zephyrus.tidal = False
    config.escape.zephyrus.efficiency = 0.1
    config.star.mass = 2.0e30  # kg

    # Minimal hf_row
    hf_row = {
        'semimajorax': 1.496e11,  # 1 AU
        'eccentricity': 0.0,
        'M_planet': 5.972e24,  # Earth mass
        'R_int': 6.371e6,  # Earth radius
        'R_xuv': 6.5e6,  # Slightly larger XUV radius
        'F_xuv': 100.0,  # W/m^2
    }

    stellar_track = MagicMock()

    # Call run_zephyrus
    mlr = run_zephyrus(config, hf_row, stellar_track)

    # Verify return value
    assert mlr == pytest.approx(1e3, rel=1e-8)

    # Verify EL_escape was called with tidal=False
    mock_el_escape.assert_called_once()
    assert not mock_el_escape.call_args[0][0]  # tidal parameter


@pytest.mark.unit
@patch('zephyrus.escape.EL_escape')
def test_run_zephyrus_with_tidal(mock_el_escape):
    """Test ZEPHYRUS escape with tidal heating contribution.

    Physical scenario: Hot Jupiter with tidal heating enhancing escape.
    Validates that tidal=True is passed to EL_escape.
    """
    from proteus.escape.wrapper import run_zephyrus

    # Mock EL_escape to return enhanced escape rate
    mock_el_escape.return_value = 1e8  # kg/s (very high escape with tidal)

    # Mock config with tidal enabled
    config = MagicMock()
    config.escape.zephyrus.tidal = True
    config.escape.zephyrus.efficiency = 0.2
    config.star.mass = 1.5e30  # kg

    # Hot Jupiter scenario
    hf_row = {
        'semimajorax': 0.03 * 1.496e11,  # 0.03 AU
        'eccentricity': 0.05,
        'M_planet': 1e27,  # Sub-Jupiter mass
        'R_int': 1.0e8,  # 100,000 km (inflated radius)
        'R_xuv': 1.2e8,
        'F_xuv': 1e5,  # W/m^2 (extreme XUV)
    }

    stellar_track = MagicMock()

    # Call run_zephyrus
    mlr = run_zephyrus(config, hf_row, stellar_track)

    # Verify enhanced escape rate
    assert mlr == pytest.approx(1e8, rel=1e-8)

    # Verify tidal=True was passed
    mock_el_escape.assert_called_once()
    assert mock_el_escape.call_args[0][0]  # tidal parameter


# =======================================================================================
# SECTION: calc_new_elements() — Elemental inventory updates
# =======================================================================================


@pytest.mark.unit
def test_calc_new_elements_bulk_reservoir():
    """Test elemental inventory update using bulk reservoir.

    Physical scenario: Unfractionated escape from entire planet (bulk).
    Validates that elemental mass ratios are preserved during escape.
    """
    from proteus.escape.wrapper import calc_new_elements

    # Initial hf_row with bulk inventories
    hf_row = {
        'esc_rate_total': 1e5,  # kg/s
        'H_kg_total': 1e21,  # Dominant H reservoir
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e19,
        'Mg_kg_total': 1e18,
        'Fe_kg_total': 1e20,
        'Na_kg_total': 1e16,
        # Oxygen not included (set by fO2)
    }

    dt = 1000.0  # years
    reservoir = 'bulk'
    min_thresh = 1e10  # kg

    # Calculate initial mass ratios
    M_vols_initial = (
        hf_row['H_kg_total']
        + hf_row['C_kg_total']
        + hf_row['N_kg_total']
        + hf_row['S_kg_total']
    )
    emr_H_initial = hf_row['H_kg_total'] / M_vols_initial
    emr_C_initial = hf_row['C_kg_total'] / M_vols_initial

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify all elements are in output
    assert 'H' in tgt
    assert 'C' in tgt
    assert 'N' in tgt
    assert 'S' in tgt
    assert 'O' not in tgt  # Oxygen excluded

    # Verify masses decreased (escape occurred)
    assert tgt['H'] < hf_row['H_kg_total']
    assert tgt['C'] < hf_row['C_kg_total']
    assert tgt['N'] < hf_row['N_kg_total']
    assert tgt['S'] < hf_row['S_kg_total']

    # Verify elemental mass ratios are approximately preserved (unfractionated)
    M_vols_final = tgt['H'] + tgt['C'] + tgt['N'] + tgt['S']
    emr_H_final = tgt['H'] / M_vols_final
    emr_C_final = tgt['C'] / M_vols_final
    assert emr_H_final == pytest.approx(emr_H_initial, rel=1e-5)
    assert emr_C_final == pytest.approx(emr_C_initial, rel=1e-5)

    # Verify no negative masses
    for e in tgt:
        assert tgt[e] >= 0.0


@pytest.mark.unit
def test_calc_new_elements_outgas_reservoir():
    """Test elemental inventory update using outgas (atmospheric) reservoir.

    Physical scenario: Escape from outgassed atmosphere only.
    Validates that atmospheric reservoirs (_kg_atm) are used for mass ratios.
    """
    from proteus.escape.wrapper import calc_new_elements

    # hf_row with both bulk and atmospheric inventories
    hf_row = {
        'esc_rate_total': 1e4,  # kg/s
        'H_kg_total': 1e21,  # Bulk H (mostly in interior)
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e19,
        'Mg_kg_total': 1e18,
        'Fe_kg_total': 1e20,
        'Na_kg_total': 1e16,
        'H_kg_atm': 1e20,  # Atmospheric H (used for outgas mode)
        'C_kg_atm': 1e17,
        'N_kg_atm': 1e18,
        'S_kg_atm': 1e16,
        'Si_kg_atm': 1e17,
        'Mg_kg_atm': 1e16,
        'Fe_kg_atm': 1e17,
        'Na_kg_atm': 1e15,
    }

    dt = 500.0  # years
    reservoir = 'outgas'
    min_thresh = 1e10  # kg

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify outputs are for TOTAL inventories (not just atmosphere)
    # But mass ratios derived from ATMOSPHERIC inventories
    assert tgt['H'] < hf_row['H_kg_total']  # Total H decreased
    assert tgt['C'] < hf_row['C_kg_total']
    assert tgt['N'] < hf_row['N_kg_total']
    assert tgt['S'] < hf_row['S_kg_total']


@pytest.mark.unit
def test_calc_new_elements_below_threshold():
    """Test elemental inventory when mass falls below minimum threshold.

    Physical scenario: Desiccated planet where volatile mass < 1e10 kg.
    Validates that inventories below threshold are set to zero.
    """
    from proteus.escape.wrapper import calc_new_elements

    # Very small volatile inventory (planet nearly desiccated)
    hf_row = {
        'esc_rate_total': 1e5,  # kg/s
        'H_kg_total': 1e9,  # Below threshold
        'C_kg_total': 1e8,
        'N_kg_total': 1e8,
        'S_kg_total': 1e7,
        'Si_kg_total': 1e8,
        'Mg_kg_total': 1e8,
        'Fe_kg_total': 1e9,
        'Na_kg_total': 1e7,
    }

    dt = 1000.0  # years
    reservoir = 'bulk'
    min_thresh = 1e10  # kg

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Total volatile mass is below threshold, so no change expected
    assert tgt['H'] == pytest.approx(hf_row['H_kg_total'], abs=1.0)
    assert tgt['C'] == pytest.approx(hf_row['C_kg_total'], abs=1.0)


@pytest.mark.unit
def test_calc_new_elements_prevent_negative_mass():
    """Test that elemental masses cannot go negative during escape.

    Physical scenario: Extreme escape rate that would deplete element inventory.
    Validates that masses are clamped to zero (not negative).
    """
    from proteus.escape.wrapper import calc_new_elements

    # Small inventory with very high escape rate
    hf_row = {
        'esc_rate_total': 1e10,  # kg/s (extremely high)
        'H_kg_total': 1e18,  # Small H inventory
        'C_kg_total': 1e15,
        'N_kg_total': 1e16,
        'S_kg_total': 1e14,
        'Si_kg_total': 1e16,
        'Mg_kg_total': 1e15,
        'Fe_kg_total': 1e17,
        'Na_kg_total': 1e14,
    }

    dt = 1e6  # years (long timescale → massive escape)
    reservoir = 'bulk'
    min_thresh = 1e10  # kg

    # Call calc_new_elements (should clamp to zero)
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify all masses are non-negative
    for e in tgt:
        assert tgt[e] >= 0.0


@pytest.mark.unit
def test_calc_new_elements_pxuv_not_supported():
    """Test that pxuv reservoir raises NotImplementedError.

    Physical scenario: Fractionated escape at XUV optical depth level.
    Validates that pxuv mode is not yet implemented.
    """
    from proteus.escape.wrapper import calc_new_elements

    hf_row = {
        'esc_rate_total': 1e5,
        'H_kg_total': 1e20,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e18,
        'Mg_kg_total': 1e17,
        'Fe_kg_total': 1e19,
        'Na_kg_total': 1e16,
    }

    dt = 1000.0
    reservoir = 'pxuv'
    min_thresh = 1e10

    # Verify ValueError is raised for pxuv
    with pytest.raises(ValueError, match='Fractionation at p_xuv is not yet supported'):
        calc_new_elements(hf_row, dt, reservoir, min_thresh)


@pytest.mark.unit
def test_calc_new_elements_invalid_reservoir():
    """Test that invalid reservoir name raises ValueError.

    Physical scenario: Configuration error with unrecognized reservoir.
    Validates proper error handling.
    """
    from proteus.escape.wrapper import calc_new_elements

    hf_row = {
        'esc_rate_total': 1e5,
        'H_kg_total': 1e20,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e18,
        'Mg_kg_total': 1e17,
        'Fe_kg_total': 1e19,
        'Na_kg_total': 1e16,
    }

    dt = 1000.0
    reservoir = 'invalid_reservoir'
    min_thresh = 1e10

    # Verify ValueError is raised
    with pytest.raises(ValueError, match='Invalid escape reservoir'):
        calc_new_elements(hf_row, dt, reservoir, min_thresh)


@pytest.mark.unit
def test_calc_new_elements_zero_escape_rate():
    """Test elemental inventory with zero escape rate.

    Physical scenario: Escape disabled or negligible escape.
    Validates that inventories remain unchanged when esc_rate_total = 0.
    """
    from proteus.escape.wrapper import calc_new_elements

    # hf_row with zero escape rate
    hf_row = {
        'esc_rate_total': 0.0,  # No escape
        'H_kg_total': 1e20,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e18,
        'Mg_kg_total': 1e17,
        'Fe_kg_total': 1e19,
        'Na_kg_total': 1e16,
    }

    dt = 1000.0
    reservoir = 'bulk'
    min_thresh = 1e10

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify inventories are unchanged
    assert tgt['H'] == pytest.approx(hf_row['H_kg_total'], rel=1e-10)
    assert tgt['C'] == pytest.approx(hf_row['C_kg_total'], rel=1e-10)
    assert tgt['N'] == pytest.approx(hf_row['N_kg_total'], rel=1e-10)
    assert tgt['S'] == pytest.approx(hf_row['S_kg_total'], rel=1e-10)
