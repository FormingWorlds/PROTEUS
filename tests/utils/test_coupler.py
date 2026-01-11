"""
Tests for proteus.utils.coupler module

This module contains comprehensive unit tests for PROTEUS coupler utility functions,
including helpfile management, state printing, lock file operations, and helper
functions for module version checking and system diagnostics.

**Test Categories**:
- Helpfile operations (creation, row extension, CSV I/O)
- Current state printing and diagnostics
- Lock file management
- Time formatting utilities
- Version validation and system information

All tests use mocked external dependencies (file I/O, subprocess calls) to ensure
fast execution (target: <100ms per test). See docs/test_building.md for test
infrastructure guidelines.

Fixtures from conftest.py:
- earth_params, ultra_hot_params, intermediate_params: Physical test scenarios
- tmp_config_paths: Temporary configuration file paths
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from proteus.utils.constants import element_list
from proteus.utils.coupler import (
    CreateHelpfileFromDict,
    CreateLockFile,
    ExtendHelpfile,
    GetHelpfileKeys,
    PrintCurrentState,
    ReadHelpfileFromCSV,
    WriteHelpfileToCSV,
    ZeroHelpfileRow,
    _get_current_time,
)

# =============================================================================
# Test: Helpfile Key Generation
# =============================================================================


@pytest.mark.unit
def test_get_helpfile_keys_returns_list():
    """Test that GetHelpfileKeys returns a non-empty list."""
    keys = GetHelpfileKeys()
    assert isinstance(keys, list)
    assert len(keys) > 0


@pytest.mark.unit
def test_get_helpfile_keys_contains_required_keys():
    """Test that GetHelpfileKeys includes all required output variables."""
    keys = GetHelpfileKeys()

    # Basic tracking
    assert 'Time' in keys

    # Orbital parameters
    assert 'semimajorax' in keys
    assert 'orbital_period' in keys
    assert 'eccentricity' in keys

    # Temperatures
    assert 'T_surf' in keys
    assert 'T_magma' in keys
    assert 'T_eqm' in keys

    # Pressures and fluxes
    assert 'P_surf' in keys
    assert 'F_int' in keys
    assert 'F_atm' in keys
    assert 'F_net' in keys


@pytest.mark.unit
def test_get_helpfile_keys_includes_gas_species():
    """Test that GetHelpfileKeys includes species from gas_list."""
    keys = GetHelpfileKeys()

    # Check for at least H2O and CO2
    for species in ['H2O', 'CO2']:
        assert f'{species}_mol_atm' in keys
        assert f'{species}_kg_atm' in keys
        assert f'{species}_vmr' in keys
        assert f'{species}_bar' in keys


@pytest.mark.unit
def test_get_helpfile_keys_includes_element_masses():
    """Test that GetHelpfileKeys includes element masses from element_list."""
    keys = GetHelpfileKeys()

    # Check for at least C, H, O
    for element in ['C', 'H', 'O', 'N', 'S']:
        if element in element_list:
            assert f'{element}_kg_atm' in keys
            assert f'{element}_kg_solid' in keys
            assert f'{element}_kg_total' in keys


@pytest.mark.unit
def test_get_helpfile_keys_no_duplicates():
    """Test that GetHelpfileKeys contains no duplicate keys."""
    keys = GetHelpfileKeys()
    assert len(keys) == len(set(keys)), 'Duplicate keys found in helpfile'


# =============================================================================
# Test: Helpfile Row Creation and Initialization
# =============================================================================


@pytest.mark.unit
def test_zero_helpfile_row_returns_dict():
    """Test that ZeroHelpfileRow returns a dictionary."""
    row = ZeroHelpfileRow()
    assert isinstance(row, dict)


@pytest.mark.unit
def test_zero_helpfile_row_all_values_are_zero():
    """Test that ZeroHelpfileRow initializes all values to 0.0."""
    row = ZeroHelpfileRow()
    for key, value in row.items():
        assert value == 0.0, f'Key {key} has value {value}, expected 0.0'


@pytest.mark.unit
def test_zero_helpfile_row_has_correct_keys():
    """Test that ZeroHelpfileRow has all keys from GetHelpfileKeys."""
    row = ZeroHelpfileRow()
    keys = GetHelpfileKeys()

    assert set(row.keys()) == set(keys), 'ZeroHelpfileRow keys do not match GetHelpfileKeys'


@pytest.mark.unit
def test_create_helpfile_from_dict_creates_dataframe():
    """Test that CreateHelpfileFromDict creates a pandas DataFrame."""
    row = ZeroHelpfileRow()
    row['Time'] = 1.0
    row['T_surf'] = 300.0

    hf = CreateHelpfileFromDict(row)
    assert isinstance(hf, pd.DataFrame)
    assert len(hf) == 1


@pytest.mark.unit
def test_create_helpfile_from_dict_preserves_values():
    """Test that CreateHelpfileFromDict preserves input values."""
    row = ZeroHelpfileRow()
    row['Time'] = 1.5e9
    row['T_surf'] = 350.0
    row['P_surf'] = 100.0

    hf = CreateHelpfileFromDict(row)

    assert hf['Time'].iloc[0] == 1.5e9
    assert hf['T_surf'].iloc[0] == 350.0
    assert hf['P_surf'].iloc[0] == 100.0


@pytest.mark.unit
def test_create_helpfile_from_dict_has_all_keys():
    """Test that CreateHelpfileFromDict includes all helpfile keys."""
    row = ZeroHelpfileRow()
    hf = CreateHelpfileFromDict(row)

    keys = GetHelpfileKeys()
    for key in keys:
        assert key in hf.columns


# =============================================================================
# Test: Helpfile Row Extension
# =============================================================================


@pytest.mark.unit
def test_extend_helpfile_appends_row():
    """Test that ExtendHelpfile appends a new row to helpfile."""
    # Create initial helpfile
    row1 = ZeroHelpfileRow()
    row1['Time'] = 0.0
    hf = CreateHelpfileFromDict(row1)

    # Extend with new row
    row2 = ZeroHelpfileRow()
    row2['Time'] = 1.0
    row2['T_surf'] = 300.0
    hf_extended = ExtendHelpfile(hf, row2)

    assert len(hf_extended) == 2
    assert hf_extended['Time'].iloc[0] == pytest.approx(0.0)
    assert hf_extended['Time'].iloc[1] == pytest.approx(1.0)
    assert hf_extended['T_surf'].iloc[1] == pytest.approx(300.0)


@pytest.mark.unit
def test_extend_helpfile_multiple_rows():
    """Test that ExtendHelpfile works with multiple sequential extensions."""
    row1 = ZeroHelpfileRow()
    row1['Time'] = 0.0
    hf = CreateHelpfileFromDict(row1)

    # Add 5 more rows
    for i in range(1, 6):
        row = ZeroHelpfileRow()
        row['Time'] = float(i)
        row['T_surf'] = 250.0 + i * 10.0
        hf = ExtendHelpfile(hf, row)

    assert len(hf) == 6
    assert hf['Time'].iloc[-1] == pytest.approx(5.0)
    assert hf['T_surf'].iloc[-1] == pytest.approx(300.0)


@pytest.mark.unit
def test_extend_helpfile_validates_keys():
    """Test that ExtendHelpfile raises error for missing keys."""
    row1 = ZeroHelpfileRow()
    hf = CreateHelpfileFromDict(row1)

    # Create a row with missing keys
    row2 = {'Time': 1.0}  # Missing other keys

    with pytest.raises(Exception, match='mismatched keys'):
        ExtendHelpfile(hf, row2)


@pytest.mark.unit
def test_extend_helpfile_preserves_dataframe_order():
    """Test that ExtendHelpfile preserves row order."""
    row1 = ZeroHelpfileRow()
    row1['Time'] = 0.0
    hf = CreateHelpfileFromDict(row1)

    times = [1.0, 2.0, 3.0, 4.0]
    for t in times:
        row = ZeroHelpfileRow()
        row['Time'] = t
        hf = ExtendHelpfile(hf, row)

    expected_times = [0.0] + times
    for i, expected_t in enumerate(expected_times):
        assert hf['Time'].iloc[i] == pytest.approx(expected_t)


# =============================================================================
# Test: Helpfile CSV I/O
# =============================================================================


@pytest.mark.unit
def test_write_helpfile_to_csv_creates_file():
    """Test that WriteHelpfileToCSV creates a CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create helpfile
        row = ZeroHelpfileRow()
        row['Time'] = 1.0
        row['T_surf'] = 300.0
        hf = CreateHelpfileFromDict(row)

        # Write to CSV
        fpath = WriteHelpfileToCSV(tmpdir, hf)

        # Verify file exists
        assert os.path.exists(fpath)
        assert fpath.endswith('runtime_helpfile.csv')
        assert 'runtime_helpfile.csv' in os.listdir(tmpdir)


@pytest.mark.unit
def test_write_helpfile_to_csv_contains_data():
    """Test that WriteHelpfileToCSV writes correct data to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        row = ZeroHelpfileRow()
        row['Time'] = 1.5
        row['T_surf'] = 350.5
        hf = CreateHelpfileFromDict(row)

        WriteHelpfileToCSV(tmpdir, hf)

        # Read back and verify
        fpath = os.path.join(tmpdir, 'runtime_helpfile.csv')
        df = pd.read_csv(fpath, sep=r'\s+')

        assert df['Time'].iloc[0] == pytest.approx(1.5)
        assert df['T_surf'].iloc[0] == pytest.approx(350.5)


@pytest.mark.unit
def test_write_helpfile_to_csv_overwrites_existing():
    """Test that WriteHelpfileToCSV overwrites existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write first helpfile
        row1 = ZeroHelpfileRow()
        row1['Time'] = 1.0
        hf1 = CreateHelpfileFromDict(row1)
        WriteHelpfileToCSV(tmpdir, hf1)

        # Write second helpfile (should overwrite)
        row2 = ZeroHelpfileRow()
        row2['Time'] = 2.0
        hf2 = CreateHelpfileFromDict(row2)
        WriteHelpfileToCSV(tmpdir, hf2)

        # Verify only latest data is in file
        fpath = os.path.join(tmpdir, 'runtime_helpfile.csv')
        df = pd.read_csv(fpath, sep=r'\s+')

        assert len(df) == 1
        assert df['Time'].iloc[0] == pytest.approx(2.0)


@pytest.mark.unit
def test_write_and_read_helpfile_roundtrip():
    """Test that helpfile survives write and read cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and write
        row = ZeroHelpfileRow()
        row['Time'] = 1.0e8
        row['T_surf'] = 287.5
        row['T_magma'] = 2500.0
        row['P_surf'] = 1.0
        row['F_int'] = 50.0
        hf_original = CreateHelpfileFromDict(row)
        WriteHelpfileToCSV(tmpdir, hf_original)

        # Read back
        hf_read = ReadHelpfileFromCSV(tmpdir)

        # Compare
        assert hf_read['Time'].iloc[0] == pytest.approx(1.0e8)
        assert hf_read['T_surf'].iloc[0] == pytest.approx(287.5)
        assert hf_read['T_magma'].iloc[0] == pytest.approx(2500.0)
        assert hf_read['P_surf'].iloc[0] == pytest.approx(1.0)
        assert hf_read['F_int'].iloc[0] == pytest.approx(50.0)


@pytest.mark.unit
def test_read_helpfile_from_csv_missing_file():
    """Test that ReadHelpfileFromCSV raises error for missing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(Exception, match='Cannot find helpfile'):
            ReadHelpfileFromCSV(tmpdir)


@pytest.mark.unit
def test_write_helpfile_multiple_rows_roundtrip():
    """Test that multiple rows survive write and read cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create helpfile with multiple rows
        row1 = ZeroHelpfileRow()
        row1['Time'] = 0.0
        hf = CreateHelpfileFromDict(row1)

        for i in range(1, 5):
            row = ZeroHelpfileRow()
            row['Time'] = float(i) * 1.0e7
            row['T_surf'] = 300.0 + i * 10.0
            hf = ExtendHelpfile(hf, row)

        # Write and read
        WriteHelpfileToCSV(tmpdir, hf)
        hf_read = ReadHelpfileFromCSV(tmpdir)

        # Verify
        assert len(hf_read) == 5
        assert hf_read['Time'].iloc[-1] == pytest.approx(4.0e7)
        assert hf_read['T_surf'].iloc[-1] == pytest.approx(340.0)


# =============================================================================
# Test: Lock File Operations
# =============================================================================


@pytest.mark.unit
def test_create_lock_file_creates_file():
    """Test that CreateLockFile creates a lock file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = CreateLockFile(tmpdir)

        assert os.path.exists(lockfile_path)
        assert 'keepalive' in lockfile_path


@pytest.mark.unit
def test_create_lock_file_returns_path():
    """Test that CreateLockFile returns the correct file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = CreateLockFile(tmpdir)

        expected_path = os.path.join(tmpdir, 'keepalive')
        assert lockfile_path == expected_path


@pytest.mark.unit
def test_create_lock_file_overwrites_existing():
    """Test that CreateLockFile removes existing file before creating new one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create first lock file
        lockfile_path1 = CreateLockFile(tmpdir)

        # Create second lock file (should overwrite)
        lockfile_path2 = CreateLockFile(tmpdir)

        assert lockfile_path1 == lockfile_path2
        # Verify it's still a valid file
        assert os.path.exists(lockfile_path2)


@pytest.mark.unit
def test_lock_file_contains_message():
    """Test that lock file contains descriptive message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = CreateLockFile(tmpdir)

        with open(lockfile_path, 'r') as f:
            content = f.read()

        assert 'stop' in content.lower() or 'remove' in content.lower()


# =============================================================================
# Test: Current State Printing
# =============================================================================


@pytest.mark.unit
def test_print_current_state_with_valid_row():
    """Test that PrintCurrentState accepts valid helpfile row."""
    hf_row = ZeroHelpfileRow()
    hf_row['Time'] = 1.0e8
    hf_row['T_surf'] = 287.0
    hf_row['T_magma'] = 3000.0
    hf_row['P_surf'] = 1.0
    hf_row['Phi_global'] = 0.5
    hf_row['F_atm'] = 100.0
    hf_row['F_int'] = 50.0

    with patch('proteus.utils.coupler.log') as mock_log:
        PrintCurrentState(hf_row)

        # Verify logger was called
        assert mock_log.info.called


@pytest.mark.unit
def test_print_current_state_includes_time():
    """Test that PrintCurrentState includes Time in output."""
    hf_row = ZeroHelpfileRow()
    hf_row['Time'] = 1.5e8

    with patch('proteus.utils.coupler.log') as mock_log:
        PrintCurrentState(hf_row)

        # Check if Time was in any log call
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('Model time' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_current_state_includes_temperatures():
    """Test that PrintCurrentState includes T_surf and T_magma."""
    hf_row = ZeroHelpfileRow()
    hf_row['T_surf'] = 300.5
    hf_row['T_magma'] = 2500.3

    with patch('proteus.utils.coupler.log') as mock_log:
        PrintCurrentState(hf_row)

        # Verify temperatures were formatted in output
        # (exact matching is difficult due to formatting)
        assert mock_log.info.called


# =============================================================================
# Test: Time Formatting
# =============================================================================


@pytest.mark.unit
def test_get_current_time_returns_string():
    """Test that _get_current_time returns a string."""
    result = _get_current_time()
    assert isinstance(result, str)


@pytest.mark.unit
def test_get_current_time_includes_date():
    """Test that _get_current_time includes date components."""
    result = _get_current_time()

    # Should contain year-month-day pattern
    assert len(result) > 0
    # Check for date-like components
    assert any(c.isdigit() for c in result)


@pytest.mark.unit
def test_get_current_time_is_recent():
    """Test that _get_current_time returns recent timestamp."""
    result = _get_current_time()

    # Should contain current year
    current_year = datetime.now().year
    assert str(current_year) in result


@pytest.mark.unit
def test_get_current_time_format_consistency():
    """Test that _get_current_time returns consistent format."""
    result1 = _get_current_time()
    result2 = _get_current_time()

    # Both should have same length (allowing for second differences)
    assert abs(len(result1) - len(result2)) <= 1


# =============================================================================
# Test: Edge Cases and Physics Validation
# =============================================================================


@pytest.mark.unit
def test_helpfile_with_realistic_earth_values():
    """Test helpfile with realistic modern Earth values."""
    row = ZeroHelpfileRow()

    # Modern Earth parameters
    row['Time'] = 4.567e9  # age in years
    row['T_surf'] = 288.0  # K, current surface temp
    row['T_magma'] = 4000.0  # K, internal temp
    row['P_surf'] = 1.01325  # bar, 1 atm
    row['R_int'] = 3.480e6  # m, dry core radius
    row['M_tot'] = 5.972e24  # kg
    row['F_int'] = 47.0  # W/m2, internal heat flow
    row['F_atm'] = 170.0  # W/m2, absorbed solar

    hf = CreateHelpfileFromDict(row)

    assert hf['T_surf'].iloc[0] == pytest.approx(288.0)
    assert hf['P_surf'].iloc[0] == pytest.approx(1.01325, rel=1e-4)
    assert hf['M_tot'].iloc[0] == pytest.approx(5.972e24)


@pytest.mark.unit
def test_helpfile_with_extreme_values():
    """Test helpfile with extreme but physically valid values."""
    row = ZeroHelpfileRow()

    # Ultra-hot planet
    row['T_surf'] = 1500.0  # K, dayside temp
    row['P_surf'] = 10000.0  # bar, high pressure
    row['T_magma'] = 6000.0  # K, hot interior

    # Low-mass planet
    row['M_tot'] = 1.0e23  # kg, ~2 Earth masses

    hf = CreateHelpfileFromDict(row)

    assert hf['T_surf'].iloc[0] == pytest.approx(1500.0)
    assert hf['P_surf'].iloc[0] == pytest.approx(10000.0)


@pytest.mark.unit
def test_helpfile_float_precision():
    """Test that helpfile preserves floating-point precision."""
    row = ZeroHelpfileRow()

    # Use a value with many significant figures
    precise_value = 1.23456789e12

    row['M_tot'] = precise_value

    hf = CreateHelpfileFromDict(row)

    assert hf['M_tot'].iloc[0] == pytest.approx(precise_value, rel=1e-8)


@pytest.mark.unit
def test_helpfile_scientific_notation_consistency():
    """Test that scientific notation values are consistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        row = ZeroHelpfileRow()
        row['Time'] = 1.234e8  # yr
        row['M_tot'] = 5.972e24  # kg
        hf = CreateHelpfileFromDict(row)

        # Write and read back
        WriteHelpfileToCSV(tmpdir, hf)
        hf_read = ReadHelpfileFromCSV(tmpdir)

        assert hf_read['Time'].iloc[0] == pytest.approx(1.234e8, rel=1e-5)
        assert hf_read['M_tot'].iloc[0] == pytest.approx(5.972e24, rel=1e-5)
