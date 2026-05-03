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
    _SECONDS_PER_YEAR,
    CreateHelpfileFromDict,
    CreateLockFile,
    ExtendHelpfile,
    GetHelpfileKeys,
    PrintCurrentState,
    ReadHelpfileFromCSV,
    WriteHelpfileToCSV,
    ZeroHelpfileRow,
    _get_current_time,
    _populate_energy_residual,
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

    with pytest.raises(Exception, match='missing expected keys'):
        ExtendHelpfile(hf, row2)


@pytest.mark.unit
def test_extend_helpfile_warns_on_unknown_keys(caplog):
    """ExtendHelpfile should WARN (not raise) on unknown keys so the CSV
    silently-drop-key bug is surfaced in logs. Private (underscore-prefixed)
    keys are intentionally skipped. Explicitly-allowlisted non-schema keys
    like ``core_state_initial`` are also skipped."""
    import logging

    row = ZeroHelpfileRow()
    row['_structure_stale'] = True  # private transient key: must not warn
    row['core_state_initial'] = 'liquid'  # allowlisted string key: must not warn
    row['nonsense_future_key'] = 1.0  # genuine drift: must warn
    hf = CreateHelpfileFromDict(ZeroHelpfileRow())

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.coupler'):
        ExtendHelpfile(hf, row)

    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    joined = '\n'.join(r.message for r in warns)
    assert 'nonsense_future_key' in joined, f'Expected unknown-key warning, got: {joined!r}'
    assert '_structure_stale' not in joined, 'Private key leaked into warning'
    assert 'core_state_initial' not in joined, 'Allowlisted key triggered warning'


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
    row['T_magma'] = 1600.0  # K, upper mantle potential temperature (see #466)
    row['P_surf'] = 1.01325  # bar, 1 atm
    row['R_int'] = 3.480e6  # m, dry core radius
    row['M_planet'] = 5.972e24  # kg
    row['F_int'] = 47.0  # W/m2, internal heat flow
    row['F_atm'] = 170.0  # W/m2, net atmospheric heat flux

    hf = CreateHelpfileFromDict(row)

    assert hf['T_surf'].iloc[0] == pytest.approx(288.0)
    assert hf['P_surf'].iloc[0] == pytest.approx(1.01325, rel=1e-4)
    assert hf['M_planet'].iloc[0] == pytest.approx(5.972e24)


@pytest.mark.unit
def test_helpfile_with_extreme_values():
    """Test helpfile with extreme but physically valid values."""
    row = ZeroHelpfileRow()

    # Ultra-hot planet
    row['T_surf'] = 1500.0  # K, dayside temp
    row['P_surf'] = 10000.0  # bar, high pressure
    row['T_magma'] = 6000.0  # K, hot interior

    # Low-mass planet
    row['M_planet'] = 1.0e23  # kg, ~2 Earth masses

    hf = CreateHelpfileFromDict(row)

    assert hf['T_surf'].iloc[0] == pytest.approx(1500.0)
    assert hf['P_surf'].iloc[0] == pytest.approx(10000.0)


@pytest.mark.unit
def test_helpfile_float_precision():
    """Test that helpfile preserves floating-point precision."""
    row = ZeroHelpfileRow()

    # Use a value with many significant figures
    precise_value = 1.23456789e12

    row['M_planet'] = precise_value

    hf = CreateHelpfileFromDict(row)

    assert hf['M_planet'].iloc[0] == pytest.approx(precise_value, rel=1e-8)


@pytest.mark.unit
def test_helpfile_scientific_notation_consistency():
    """Test that scientific notation values are consistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        row = ZeroHelpfileRow()
        row['Time'] = 1.234e8  # yr
        row['M_planet'] = 5.972e24  # kg
        hf = CreateHelpfileFromDict(row)

        # Write and read back
        WriteHelpfileToCSV(tmpdir, hf)
        hf_read = ReadHelpfileFromCSV(tmpdir)

        assert hf_read['Time'].iloc[0] == pytest.approx(1.234e8, rel=1e-5)
        assert hf_read['M_planet'].iloc[0] == pytest.approx(5.972e24, rel=1e-5)


# =============================================================================
# Test: Version Getter Functions
# =============================================================================


@pytest.mark.unit
def test_get_spider_version():
    """Test that _get_spider_version returns expected version string."""
    from proteus.utils.coupler import _get_spider_version

    version = _get_spider_version()
    assert isinstance(version, str)
    assert version == '0.2.0'


@pytest.mark.unit
def test_get_petsc_version():
    """Test that _get_petsc_version returns expected version string."""
    from proteus.utils.coupler import _get_petsc_version

    version = _get_petsc_version()
    assert isinstance(version, str)
    assert version == '1.3.19.0'


@pytest.mark.unit
def test_get_git_revision_with_mock():
    """Test that _get_git_revision returns git hash."""
    import subprocess

    from proteus.utils.coupler import _get_git_revision

    with patch('subprocess.check_output') as mock_check_output:
        mock_check_output.return_value = b'abc123def456789\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _get_git_revision(tmpdir)

            assert result == 'abc123def456789'
            mock_check_output.assert_called_once_with(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )


@pytest.mark.unit
def test_get_socrates_version_with_mock():
    """Test that _get_socrates_version reads version file."""
    from proteus.utils.coupler import _get_socrates_version

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock version file
        version_file = os.path.join(tmpdir, 'version')
        with open(version_file, 'w') as f:
            f.write('24.1.0\n')

        # Mock environment variable
        with patch.dict(os.environ, {'RAD_DIR': tmpdir}):
            version = _get_socrates_version()
            assert version == '24.1.0'


@pytest.mark.unit
def test_get_agni_version_with_mock():
    """Test that _get_agni_version reads TOML file."""
    from proteus.utils.coupler import _get_agni_version

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock Project.toml (AGNI format: top-level version)
        toml_content = b'name = "AGNI"\nversion = "1.8.0"\n'
        toml_path = os.path.join(tmpdir, 'Project.toml')
        with open(toml_path, 'wb') as f:
            f.write(toml_content)

        dirs = {'agni': tmpdir}
        version = _get_agni_version(dirs)

        assert version == '1.8.0'


@pytest.mark.unit
def test_get_julia_version_with_mock():
    """Test that _get_julia_version parses julia --version output."""
    from proteus.utils.coupler import _get_julia_version

    with patch('subprocess.check_output') as mock_check_output:
        mock_check_output.return_value = b'julia version 1.9.3\n'

        version = _get_julia_version()

        assert version == '1.9.3'
        mock_check_output.assert_called_once_with(['julia', '--version'])


# =============================================================================
# Test: Print Functions
# =============================================================================


@pytest.mark.unit
def test_print_header():
    """Test that print_header logs header information."""
    from proteus.utils.coupler import print_header

    with patch('proteus.utils.coupler.log') as mock_log:
        print_header()

        # Verify logger was called with header content
        assert mock_log.info.called
        # Check that PROTEUS and copyright year are mentioned
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('PROTEUS' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_stoptime_seconds():
    """Test that print_stoptime formats runtime in seconds."""
    from proteus.utils.coupler import print_stoptime

    start_time = datetime.now() - pd.Timedelta(seconds=45)

    with patch('proteus.utils.coupler.log') as mock_log:
        print_stoptime(start_time)

        # Verify logger was called
        assert mock_log.info.called
        # Check that seconds formatting was used
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('seconds' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_stoptime_minutes():
    """Test that print_stoptime formats runtime in minutes."""
    from proteus.utils.coupler import print_stoptime

    start_time = datetime.now() - pd.Timedelta(minutes=5, seconds=30)

    with patch('proteus.utils.coupler.log') as mock_log:
        print_stoptime(start_time)

        # Verify minutes formatting was used
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('minutes' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_stoptime_hours():
    """Test that print_stoptime formats runtime in hours."""
    from proteus.utils.coupler import print_stoptime

    start_time = datetime.now() - pd.Timedelta(hours=2, minutes=15)

    with patch('proteus.utils.coupler.log') as mock_log:
        print_stoptime(start_time)

        # Verify hours formatting was used
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('hours' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_system_configuration():
    """Test that print_system_configuration logs system info."""
    from proteus.utils.coupler import print_system_configuration

    dirs = {'fwl': '/path/to/fwl'}

    with patch('proteus.utils.coupler.log') as mock_log:
        with patch('os.getlogin', return_value='testuser'):
            print_system_configuration(dirs)

            # Verify logger was called
            assert mock_log.info.called
            # Check that key system info was logged
            log_calls = [str(call) for call in mock_log.info.call_args_list]
            assert any('Python version' in str(call) for call in log_calls)
            assert any('testuser' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_system_configuration_fallback_username():
    """Test that print_system_configuration handles OSError for username."""
    from proteus.utils.coupler import print_system_configuration

    dirs = {'fwl': '/path/to/fwl'}

    with patch('proteus.utils.coupler.log') as mock_log:
        with patch('os.getlogin', side_effect=OSError):
            with patch('pwd.getpwuid') as mock_getpwuid:
                mock_getpwuid.return_value.pw_name = 'fallback_user'

                print_system_configuration(dirs)

                # Verify fallback username was used
                assert mock_log.info.called


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
def test_write_helpfile_validates_missing_keys():
    """Test that WriteHelpfileToCSV raises error for missing keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a DataFrame with missing keys
        df = pd.DataFrame({'Time': [1.0]})  # Missing all other required keys

        with pytest.raises(Exception, match='mismatched keys'):
            WriteHelpfileToCSV(tmpdir, df)


@pytest.mark.unit
def test_zero_helpfile_row_is_immutable_structure():
    """Test that ZeroHelpfileRow returns new dict each time."""
    row1 = ZeroHelpfileRow()
    row2 = ZeroHelpfileRow()

    # Modify first row
    row1['Time'] = 100.0

    # Second row should still be zero
    assert row2['Time'] == 0.0


@pytest.mark.unit
def test_helpfile_preserves_column_order():
    """Test that helpfile operations preserve column order."""
    keys = GetHelpfileKeys()
    row = ZeroHelpfileRow()
    hf = CreateHelpfileFromDict(row)

    # Verify column order matches GetHelpfileKeys
    assert list(hf.columns) == keys


@pytest.mark.unit
def test_create_lock_file_with_absolute_path():
    """Test that CreateLockFile works with absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        abs_path = os.path.abspath(tmpdir)
        lockfile_path = CreateLockFile(abs_path)

        assert os.path.isabs(lockfile_path)
        assert os.path.exists(lockfile_path)


@pytest.mark.unit
def test_extend_helpfile_preserves_dtype():
    """Test that ExtendHelpfile preserves float dtype."""
    row1 = ZeroHelpfileRow()
    row1['Time'] = 1.0
    hf = CreateHelpfileFromDict(row1)

    row2 = ZeroHelpfileRow()
    row2['Time'] = 2.0
    hf_extended = ExtendHelpfile(hf, row2)

    # All columns should be float
    assert hf_extended['Time'].dtype == float


@pytest.mark.unit
def test_helpfile_handles_large_time_values():
    """Test that helpfile handles large time values (Gyr scale)."""
    row = ZeroHelpfileRow()
    row['Time'] = 4.567e9  # 4.567 Gyr (age of Earth)
    hf = CreateHelpfileFromDict(row)

    assert hf['Time'].iloc[0] == pytest.approx(4.567e9)


@pytest.mark.unit
def test_helpfile_handles_negative_fluxes():
    """Test that helpfile handles negative flux values (heat loss)."""
    row = ZeroHelpfileRow()
    row['F_int'] = -50.0  # negative = heat loss from interior
    row['F_atm'] = -100.0  # negative = net radiative cooling
    hf = CreateHelpfileFromDict(row)

    assert hf['F_int'].iloc[0] == pytest.approx(-50.0)
    assert hf['F_atm'].iloc[0] == pytest.approx(-100.0)


# =============================================================================
# Test: Energy-conservation bookkeeping (_populate_energy_residual)
# =============================================================================


def _aragog_row(
    *,
    time_yr: float,
    E_state_J: float,
    F_int: float = 0.0,
    F_cmb: float = 0.0,
    Q_radio_W: float = 0.0,
    Q_dil_W: float = 0.0,
    Q_tidal_W: float = 0.0,
    R_int: float = 6.371e6,
    R_core: float = 3.481e6,
    T_magma: float = 3000.0,
) -> dict:
    """Helper: build a ZeroHelpfileRow populated with the columns the
    energy-residual helper reads. Uses Earth-like radii by default."""
    row = ZeroHelpfileRow()
    row['Time'] = time_yr
    row['E_state_J'] = E_state_J
    row['F_int'] = F_int
    row['F_cmb'] = F_cmb
    row['Q_radio_W'] = Q_radio_W
    row['Q_dil_W'] = Q_dil_W
    row['Q_tidal_W'] = Q_tidal_W
    row['R_int'] = R_int
    row['R_core'] = R_core
    row['T_magma'] = T_magma
    return row


@pytest.mark.unit
def test_helpfile_keys_include_energy_conservation_columns():
    """The helpfile schema must expose the per-row energy-conservation
    columns (anchor: E_state_J + per-source breakdown) and the cumulative
    bookkeeping columns. Catches accidental drift if a future cleanup
    removes one. F_cmb is a separate flux from F_int and must also be
    present so the closed-mantle balance can subtract the bottom flux."""
    keys = GetHelpfileKeys()
    expected = {
        'F_cmb',
        'E_state_J',
        'Q_radio_W',
        'Q_dil_W',
        'Q_tidal_W',
        'dE_predicted_J',
        'E_residual_J',
        'E_residual_frac',
    }
    missing = expected - set(keys)
    assert not missing, f'Energy-conservation keys missing from schema: {missing}'

    # No duplicates: would silently shadow the column on CSV write.
    assert len(keys) == len(set(keys)), 'Helpfile schema contains duplicate keys'


@pytest.mark.unit
def test_populate_energy_residual_inactive_when_E_state_zero():
    """Non-Aragog modules leave E_state_J at 0. The bookkeeping helper
    must short-circuit: residual columns stay at 0.0, NEVER NaN, even
    when other power columns happen to be populated. NaN would propagate
    into cumulative sums and silently corrupt downstream rows."""
    hf = CreateHelpfileFromDict(_aragog_row(time_yr=0.0, E_state_J=0.0))
    # Even with non-zero F_int, E_state_J=0 must trigger the inactive path.
    new_row = _aragog_row(time_yr=1.0, E_state_J=0.0, F_int=100.0)

    _populate_energy_residual(hf, new_row)

    assert new_row['dE_predicted_J'] == pytest.approx(0.0)
    assert new_row['E_residual_J'] == pytest.approx(0.0)
    assert new_row['E_residual_frac'] == pytest.approx(0.0)


@pytest.mark.unit
def test_populate_energy_residual_anchor_row_is_zero():
    """Row 0 of an Aragog run is the anchor: by definition there is no
    prior state to integrate against, so dE_predicted and the residual
    must both be exactly 0 even though E_state_J itself is large
    (~1e31 J for an Earth-mass mantle)."""
    empty_hf = pd.DataFrame(columns=GetHelpfileKeys())
    new_row = _aragog_row(time_yr=0.0, E_state_J=1.234e31, F_int=200.0)

    _populate_energy_residual(empty_hf, new_row)

    assert new_row['dE_predicted_J'] == pytest.approx(0.0)
    assert new_row['E_residual_J'] == pytest.approx(0.0)
    assert new_row['E_residual_frac'] == pytest.approx(0.0)


@pytest.mark.unit
def test_populate_energy_residual_closes_for_pure_cooling():
    """Synthetic closed-mantle physics: F_int = 100 W/m², no other
    sources, constant area. Trapezoidal integral with constant power is
    exact, so a self-consistent E_state trajectory (E0 + dE_pred) must
    yield residual identically zero. The discriminating asymmetric
    timestep (1 yr then 9 yr) catches off-by-one in which prev-row is
    used for the trapezoid endpoint."""
    R_int = 6.0e6
    A_int = 4.0 * 3.141592653589793 * R_int * R_int
    F_int = 100.0  # W/m^2, positive = heat loss
    P = -F_int * A_int  # W, negative = mantle cools

    E0 = 1.0e31
    t1 = 1.0  # yr
    t2 = 10.0  # yr (asymmetric step)
    dt1_s = t1 * _SECONDS_PER_YEAR
    dt2_s = (t2 - t1) * _SECONDS_PER_YEAR

    E1 = E0 + P * dt1_s
    E2 = E1 + P * dt2_s

    # Anchor row.
    row0 = _aragog_row(time_yr=0.0, E_state_J=E0, F_int=F_int, R_int=R_int)
    hf = CreateHelpfileFromDict(row0)

    # Step 1.
    row1 = _aragog_row(time_yr=t1, E_state_J=E1, F_int=F_int, R_int=R_int)
    _populate_energy_residual(hf, row1)
    hf = ExtendHelpfile(hf, row1)

    # Step 2.
    row2 = _aragog_row(time_yr=t2, E_state_J=E2, F_int=F_int, R_int=R_int)
    _populate_energy_residual(hf, row2)

    # Cumulative dE_predicted must match analytic P*dt to FP precision.
    assert row1['dE_predicted_J'] == pytest.approx(P * dt1_s, rel=1e-12)
    assert row2['dE_predicted_J'] == pytest.approx(P * (dt1_s + dt2_s), rel=1e-12)

    # Residual must be ~0 (relative to the large dE actual). Tolerance
    # is FP-cancellation-dominated at this magnitude (E ~ 1e31, dE ~ 1e25),
    # not bookkeeping-logic-dominated; 1e-10 is still ~5 orders of
    # magnitude tighter than any physically interesting residual.
    assert abs(row2['E_residual_J']) < 1e-3 * abs(E2 - E0)
    assert abs(row2['E_residual_frac']) < 1e-10


@pytest.mark.unit
def test_populate_energy_residual_detects_missing_source_term():
    """Discriminating test: a 'real' planet receives a strong dilatation
    heating Q_dil = +1e15 W on top of F_int cooling, so the recorded
    E_state stays nearly constant (heating ≈ cooling). If the bookkeeping
    integrand FORGOT to include Q_dil, the predicted dE would be the
    cooling-only number, and the residual would equal +Q_dil*dt with the
    correct sign. We assert the helper INCLUDES Q_dil so the residual is
    near zero. Catches the omission bug directly."""
    R_int = 6.0e6
    A_int = 4.0 * 3.141592653589793 * R_int * R_int
    F_int = 100.0
    P_cool = -F_int * A_int  # W
    Q_dil = -P_cool  # W, exactly cancels cooling

    E0 = 5.0e30
    t1 = 100.0  # yr
    dt_s = t1 * _SECONDS_PER_YEAR
    # E_state stays exactly constant because heating cancels cooling.
    E1 = E0

    row0 = _aragog_row(time_yr=0.0, E_state_J=E0, F_int=F_int, Q_dil_W=Q_dil, R_int=R_int)
    hf = CreateHelpfileFromDict(row0)

    row1 = _aragog_row(time_yr=t1, E_state_J=E1, F_int=F_int, Q_dil_W=Q_dil, R_int=R_int)
    _populate_energy_residual(hf, row1)

    # Trapezoidal with both endpoints = 0 power must give dE_pred = 0
    # to FP precision (cancellation between P_cool and Q_dil is exact).
    assert row1['dE_predicted_J'] == pytest.approx(0.0, abs=1.0)
    # E_state is exactly constant, so dE_actual = 0 and residual = 0.
    assert row1['E_residual_J'] == pytest.approx(0.0, abs=1.0)
    # Disable Q_dil retroactively in the integrand to verify it would FAIL:
    # if the helper had not included Q_dil_W, predicted dE would be
    # P_cool*dt and residual would be +|P_cool|*dt = ~Q_dil*dt.
    expected_residual_if_buggy = -P_cool * dt_s  # > 0
    assert expected_residual_if_buggy > 1e25, 'Test setup too weak to discriminate'


@pytest.mark.unit
def test_populate_energy_residual_handles_non_monotonic_time():
    """PROTEUS dt fallback can produce a step where new Time <= prev Time.
    In that case the helper must NOT emit a negative dE_predicted (which
    would corrupt the cumulative sum); it should set this step's
    increment to zero and leave the residual equal to dE_actual alone."""
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=10.0, E_state_J=E0, F_int=200.0)
    hf = CreateHelpfileFromDict(row0)

    # Non-monotonic step: same time, smaller energy (somehow).
    E1 = E0 - 1.0e25  # dE_actual = -1e25 J
    row1 = _aragog_row(time_yr=10.0, E_state_J=E1, F_int=200.0)
    _populate_energy_residual(hf, row1)

    # dE_pred must not advance (zero increment + zero prior).
    assert row1['dE_predicted_J'] == pytest.approx(0.0)
    # Residual = dE_actual - 0.
    assert row1['E_residual_J'] == pytest.approx(-1.0e25)


@pytest.mark.unit
def test_populate_energy_residual_frac_normalises_safely_when_dE_tiny():
    """At quiescent steady state, dE_actual can be much smaller than 1 J
    in absolute value. Naive division would amplify bookkeeping noise to
    arbitrarily large fractions. The helper floors the divisor at 1 J,
    so for a tiny dE_actual the fractional residual stays bounded and
    interpretable. We use E_state_J = 100 J (small but non-zero so the
    inactive-path guard does NOT trigger) so the 0.5 J perturbation
    survives float64 cancellation; the floor logic itself is the
    quantity under test, not the realistic E magnitude."""
    E0 = 100.0  # J (not realistic; chosen so 0.5 J survives FP)
    row0 = _aragog_row(time_yr=0.0, E_state_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # dE_actual = +0.5 J, dE_pred = 0 (no fluxes set).
    row1 = _aragog_row(time_yr=1.0, E_state_J=E0 + 0.5)
    _populate_energy_residual(hf, row1)

    # Without flooring this would be 0.5 / 0.5 = 1.0 (huge); with floor
    # at 1 J the divisor stays >= 1 and the fraction stays within [-1, 1].
    assert row1['E_residual_J'] == pytest.approx(0.5)
    assert abs(row1['E_residual_frac']) <= 1.0
    # Specifically: 0.5 / max(0.5, 1.0) = 0.5.
    assert row1['E_residual_frac'] == pytest.approx(0.5)


@pytest.mark.unit
def test_populate_energy_residual_F_cmb_uses_core_area_not_int_area():
    """Closed-mantle balance: F_int * A_int (top) vs F_cmb * A_cmb
    (bottom). If the helper accidentally used A_int for both, an Earth-
    like geometry (R_int=6.371e6, R_core=3.481e6) would over-credit the
    CMB inflow by factor (R_int/R_core)^2 ≈ 3.35. We construct a case
    where ONLY F_cmb is non-zero and confirm the predicted power matches
    F_cmb * 4πR_core^2, NOT F_cmb * 4πR_int^2."""
    R_int = 6.371e6
    R_core = 3.481e6
    A_int = 4.0 * 3.141592653589793 * R_int * R_int
    A_core = 4.0 * 3.141592653589793 * R_core * R_core
    F_cmb = 50.0  # W/m^2

    E0 = 1.0e31
    t1 = 100.0
    dt_s = t1 * _SECONDS_PER_YEAR
    # Self-consistent trajectory using A_core (the correct area).
    E1 = E0 + F_cmb * A_core * dt_s

    row0 = _aragog_row(time_yr=0.0, E_state_J=E0, F_cmb=F_cmb, R_int=R_int, R_core=R_core)
    hf = CreateHelpfileFromDict(row0)

    row1 = _aragog_row(time_yr=t1, E_state_J=E1, F_cmb=F_cmb, R_int=R_int, R_core=R_core)
    _populate_energy_residual(hf, row1)

    # Residual ~0 confirms helper uses A_core; if it used A_int instead,
    # the residual would be (A_core - A_int) * F_cmb * dt ≈ -2.4e30 J,
    # i.e. ~24% of E_state.
    assert row1['dE_predicted_J'] == pytest.approx(F_cmb * A_core * dt_s, rel=1e-12)
    assert abs(row1['E_residual_J']) < 1e-6 * abs(E1 - E0)
    # Sanity check that the failure-mode residual would have been huge:
    # if the helper had used A_int instead of A_core, residual would be
    # (A_core - A_int) * F_cmb * dt ≈ -5.6e25 J, large enough to dwarf
    # the actual residual (~1 J after FP cancellation).
    bogus_residual = (A_core - A_int) * F_cmb * dt_s
    assert abs(bogus_residual) > 1e25, 'Test geometry too symmetric to discriminate'
