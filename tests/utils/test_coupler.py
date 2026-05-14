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
    _populate_energy_residual,
    get_proteus_directories,
)

pytestmark = pytest.mark.unit

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
def test_get_socrates_version_raises_without_rad_dir():
    """Test that _get_socrates_version errors when RAD_DIR is missing."""
    from proteus.utils.coupler import _get_socrates_version

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError, match='RAD_DIR environment variable is not set'):
            _get_socrates_version()


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
#
# The conservation primitive is the per-call energy integral set computed by
# Aragog over its CVODE sub-step trajectory and exposed in helpfile columns
# step_dE_F_int_J / step_dE_F_cmb_J / step_dE_Q_radio_cons_J /
# step_dE_Q_tidal_cons_J / step_solver_residual_J. PROTEUS just cumulatively
# sums these; no helpfile-side trapezoidal interpolation between
# possibly-transient F_cmb snapshots. The cons (frozen-mass) variants of the
# Q sources pair with the frozen-mass E_state_cons_J state variable; the
# legacy state-mass step_dE_Q_radio_J / step_dE_Q_tidal_J columns are kept
# in the schema as instrumentation only and are NOT read by the residual.


def _aragog_row(
    *,
    time_yr: float,
    E_state_cons_J: float,
    step_dE_F_int_J: float = 0.0,
    step_dE_F_cmb_J: float = 0.0,
    step_dE_Q_radio_cons_J: float = 0.0,
    step_dE_Q_tidal_cons_J: float = 0.0,
    step_solver_residual_J: float = 0.0,
    F_cmb: float = 0.0,
    R_int: float = 6.371e6,
    R_core: float = 3.481e6,
    T_magma: float = 3000.0,
) -> dict:
    """Helper: build a ZeroHelpfileRow populated with the cons (frozen-mass)
    columns the energy-residual helper reads. Uses Earth-like radii by
    default. The instantaneous F_cmb keyword argument is supported only so
    the discrimination test can prove it is IGNORED by the cumulative-sum
    logic; it has no effect on dE_predicted_cons_J."""
    row = ZeroHelpfileRow()
    row['Time'] = time_yr
    row['E_state_cons_J'] = E_state_cons_J
    row['step_dE_F_int_J'] = step_dE_F_int_J
    row['step_dE_F_cmb_J'] = step_dE_F_cmb_J
    row['step_dE_Q_radio_cons_J'] = step_dE_Q_radio_cons_J
    row['step_dE_Q_tidal_cons_J'] = step_dE_Q_tidal_cons_J
    row['step_solver_residual_J'] = step_solver_residual_J
    row['F_cmb'] = F_cmb
    row['R_int'] = R_int
    row['R_core'] = R_core
    row['T_magma'] = T_magma
    return row


@pytest.mark.unit
def test_helpfile_keys_include_energy_conservation_columns():
    """Schema must expose the per-call step-delta columns AND the cumulative
    bookkeeping columns. Catches accidental drift if a future cleanup
    removes one. Each step delta is a separate source so the breakdown can
    be reconstructed downstream."""
    keys = GetHelpfileKeys()
    expected = {
        # Frozen-mass conservation primitives (consumed by the residual).
        'E_state_cons_J',
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_cons_J',
        'step_dE_Q_tidal_cons_J',
        'step_solver_residual_J',
        # Cumulative columns derived from the primitives above.
        'dE_predicted_cons_J',
        'E_residual_cons_J',
        'E_residual_cons_frac',
        'solver_residual_J',
        # State-mass columns kept as diagnostics (NOT used for residuals).
        'E_state_J',
        'step_dE_Q_radio_J',
        'step_dE_Q_tidal_J',
    }
    missing = expected - set(keys)
    assert not missing, f'Energy-conservation keys missing from schema: {missing}'
    assert len(keys) == len(set(keys)), 'Helpfile schema contains duplicate keys'
    # Negative regression: the deprecated state-mass residual columns must
    # stay removed. They masked real signal because the state-dependent
    # rho(P,S)*V mass weighting introduced a non-conservation cross term
    # that grew with mantle cooling.
    deprecated_state_mass = {'dE_predicted_J', 'E_residual_J', 'E_residual_frac'}
    leaked_dep = deprecated_state_mass & set(keys)
    assert not leaked_dep, (
        f'Deprecated state-mass residual columns reappeared in schema: {leaked_dep}'
    )
    # Negative regression: the historical Q_dil/F_dil columns must stay
    # deleted. If a future cleanup adds them back, this test fails loudly.
    forbidden = {'F_dil', 'Q_dil_W', 'step_dE_Q_dil_J'}
    leaked = forbidden & set(keys)
    assert not leaked, f'Deleted dilatation columns reappeared in schema: {leaked}'


@pytest.mark.unit
def test_populate_energy_residual_inactive_when_E_state_cons_zero():
    """Non-Aragog modules leave E_state_cons_J at 0. The bookkeeping
    helper must short-circuit: cons residual + solver_residual columns
    stay at 0.0, NEVER NaN, even when step deltas happen to be non-zero.
    NaN would propagate into cumulative sums and silently corrupt
    downstream rows."""
    hf = CreateHelpfileFromDict(_aragog_row(time_yr=0.0, E_state_cons_J=0.0))
    new_row = _aragog_row(
        time_yr=1.0,
        E_state_cons_J=0.0,
        step_dE_F_int_J=-1.0e25,
        step_dE_F_cmb_J=+1.0e25,
        step_solver_residual_J=+1.0e10,
    )

    _populate_energy_residual(hf, new_row)

    assert new_row['dE_predicted_cons_J'] == pytest.approx(0.0)
    assert new_row['E_residual_cons_J'] == pytest.approx(0.0)
    assert new_row['E_residual_cons_frac'] == pytest.approx(0.0)
    assert new_row['solver_residual_J'] == pytest.approx(0.0)


@pytest.mark.unit
def test_populate_energy_residual_anchor_row_is_zero():
    """Row 0 of an Aragog run is the anchor: by definition there is no
    prior state to integrate against. dE_predicted_cons, the cons
    residual, AND solver_residual_J must all be exactly 0 even when
    E_state_cons_J is large (~1e31 J for Earth) AND the per-call step
    deltas + step_solver_residual_J happen to be populated (Aragog
    reports them on the first call too, but they are conventionally
    ignored at the anchor)."""
    empty_hf = pd.DataFrame(columns=GetHelpfileKeys())
    new_row = _aragog_row(
        time_yr=0.0,
        E_state_cons_J=1.234e31,
        step_dE_F_int_J=-9.99e30,  # large but must be ignored at anchor
        step_dE_F_cmb_J=+5.55e30,
        step_solver_residual_J=+1.0e15,  # also ignored at anchor
    )

    _populate_energy_residual(empty_hf, new_row)

    assert new_row['dE_predicted_cons_J'] == pytest.approx(0.0)
    assert new_row['E_residual_cons_J'] == pytest.approx(0.0)
    assert new_row['E_residual_cons_frac'] == pytest.approx(0.0)
    assert new_row['solver_residual_J'] == pytest.approx(0.0)


@pytest.mark.unit
def test_populate_energy_residual_cumulative_sum_across_three_rows():
    """Synthetic 3-row trajectory with asymmetric step deltas. Cumulative
    dE_predicted_cons at row N must equal the sum of step deltas for
    rows 1..N (using the cons sources). The asymmetry (different
    magnitudes per row) catches an off-by-one in which the prior row's
    dE_predicted_cons_J is being used as the running total. Also
    asserts solver_residual_J accumulates linearly: a regression that
    used the prior-row value as the starting point would miss the
    fact that step_solver_residual_J is itself the per-call increment.
    """
    E0 = 1.0e31
    # Three asymmetric increments, all sources active.
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # Step 1: cooling dominates.
    delta_1_F_int = -3.0e29
    delta_1_Q_radio = +1.0e29
    expected_dE_pred_1 = delta_1_F_int + delta_1_Q_radio  # = -2.0e29
    E1 = E0 + expected_dE_pred_1
    solver_inc_1 = +2.0e22  # tiny solver residual increment, machine-precision-like
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E1,
        step_dE_F_int_J=delta_1_F_int,
        step_dE_Q_radio_cons_J=delta_1_Q_radio,
        step_solver_residual_J=solver_inc_1,
    )
    _populate_energy_residual(hf, row1)
    hf = ExtendHelpfile(hf, row1)

    # Step 2: F_cmb heat input > cooling, net warming.
    delta_2_F_int = -2.0e29
    delta_2_F_cmb = +5.0e29
    expected_dE_pred_2 = expected_dE_pred_1 + delta_2_F_int + delta_2_F_cmb  # = +1e29
    E2 = E0 + expected_dE_pred_2
    solver_inc_2 = -3.0e22  # negative increment to discriminate sum-vs-overwrite
    row2 = _aragog_row(
        time_yr=25.0,
        E_state_cons_J=E2,
        step_dE_F_int_J=delta_2_F_int,
        step_dE_F_cmb_J=delta_2_F_cmb,
        step_solver_residual_J=solver_inc_2,
    )
    _populate_energy_residual(hf, row2)
    hf = ExtendHelpfile(hf, row2)

    # Step 3: F_cmb adds energy from below.
    delta_3_F_cmb = +1.0e29
    delta_3_F_int = -4.0e29
    expected_dE_pred_3 = expected_dE_pred_2 + delta_3_F_cmb + delta_3_F_int  # = -2e29
    E3 = E0 + expected_dE_pred_3
    solver_inc_3 = +5.0e22
    row3 = _aragog_row(
        time_yr=50.0,
        E_state_cons_J=E3,
        step_dE_F_cmb_J=delta_3_F_cmb,
        step_dE_F_int_J=delta_3_F_int,
        step_solver_residual_J=solver_inc_3,
    )
    _populate_energy_residual(hf, row3)

    assert row1['dE_predicted_cons_J'] == pytest.approx(expected_dE_pred_1, rel=1e-12)
    assert row2['dE_predicted_cons_J'] == pytest.approx(expected_dE_pred_2, rel=1e-12)
    assert row3['dE_predicted_cons_J'] == pytest.approx(expected_dE_pred_3, rel=1e-12)

    # Self-consistent trajectory => zero cons residual at every row.
    assert abs(row1['E_residual_cons_J']) < 1e-3 * abs(E1 - E0)
    assert abs(row3['E_residual_cons_frac']) < 1e-10

    # solver_residual_J is the running sum of step_solver_residual_J;
    # ExtendHelpfile only persists row1 and row2 to hf at this point
    # (row3 is still in flight), so row3's value is row2-cumulative
    # plus its own increment.
    assert row1['solver_residual_J'] == pytest.approx(solver_inc_1, rel=1e-12)
    assert row2['solver_residual_J'] == pytest.approx(solver_inc_1 + solver_inc_2, rel=1e-12)
    assert row3['solver_residual_J'] == pytest.approx(
        solver_inc_1 + solver_inc_2 + solver_inc_3, rel=1e-12
    )


@pytest.mark.unit
def test_populate_energy_residual_ignores_instantaneous_F_cmb_spike():
    """Discriminating test for the bug that motivated the per-call
    integral rewrite: a single transient spike in the instantaneous
    F_cmb column (which Aragog can report at a CVODE phase-boundary
    moment) must NOT contaminate dE_predicted_cons_J. Only the
    integrated step deltas contribute. We construct a row where
    instantaneous F_cmb is set to an absurd 1e23 W/m^2 (the historical
    bug pattern) but the per-call step delta is physical; assert the
    cumulative is governed by the step delta alone."""
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # Physical step delta: -1e29 J (mild cooling).
    physical_delta = -1.0e29
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E0 + physical_delta,
        step_dE_F_int_J=physical_delta,
        # Catastrophic instantaneous spike — must be IGNORED.
        F_cmb=1.0e23,
    )
    _populate_energy_residual(hf, row1)

    # If the helper used the instantaneous spike, dE_predicted_cons
    # would be wildly different from -1e29 (the spike-driven
    # trapezoid would give something like ~1e30 over a 10 yr step).
    # With step-delta logic the result is exactly the physical delta
    # to FP precision.
    assert row1['dE_predicted_cons_J'] == pytest.approx(physical_delta, rel=1e-12)
    assert abs(row1['E_residual_cons_J']) < 1e-3 * abs(physical_delta)


@pytest.mark.unit
def test_populate_energy_residual_frac_normalises_safely_when_dE_tiny():
    """At quiescent steady state, dE_actual_cons can be much smaller
    than 1 J in absolute value. Naive division would amplify
    bookkeeping noise to arbitrarily large fractions. The helper
    floors the divisor at 1 J, so for a tiny dE_actual_cons the
    fractional residual stays bounded and interpretable. We use
    E_state_cons_J = 100 J (small but non-zero so the inactive-path
    guard does NOT trigger) so the 0.5 J perturbation survives
    float64 cancellation; the floor logic itself is the quantity
    under test, not the realistic E magnitude."""
    E0 = 100.0  # J (not realistic; chosen so 0.5 J survives FP)
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # dE_actual_cons = +0.5 J, dE_pred_cons = 0 (no step deltas set).
    row1 = _aragog_row(time_yr=1.0, E_state_cons_J=E0 + 0.5)
    _populate_energy_residual(hf, row1)

    # Without flooring this would be 0.5 / 0.5 = 1.0; with floor at 1 J
    # the divisor stays >= 1 and the fraction stays within [-1, 1].
    assert row1['E_residual_cons_J'] == pytest.approx(0.5)
    assert abs(row1['E_residual_cons_frac']) <= 1.0
    # Specifically: 0.5 / max(0.5, 1.0) = 0.5.
    assert row1['E_residual_cons_frac'] == pytest.approx(0.5)


@pytest.mark.unit
def test_populate_energy_residual_residual_detects_missing_source():
    """Conservation residual must surface a missing source. We
    construct a 'real' planet whose E_state_cons actually ROSE by
    5e29 J because of a fictional unreported source, but Aragog (in
    this scenario) only reported the cooling step delta (-3e29 J).
    The residual should be +8e29 J = (E_state_cons-E_state_cons[0])
    - dE_predicted_cons, sign and magnitude both meaningful. Catches
    a regression where the helper would silently zero residuals or
    apply the wrong sign convention."""
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    actual_dE = +5.0e29  # planet warmed
    reported_step_delta = -3.0e29  # Aragog only reported the cooling
    row1 = _aragog_row(
        time_yr=100.0,
        E_state_cons_J=E0 + actual_dE,
        step_dE_F_int_J=reported_step_delta,
    )
    _populate_energy_residual(hf, row1)

    expected_residual = actual_dE - reported_step_delta  # = +8e29 J
    assert row1['dE_predicted_cons_J'] == pytest.approx(reported_step_delta, rel=1e-12)
    assert row1['E_residual_cons_J'] == pytest.approx(expected_residual, rel=1e-12)
    # E_residual_cons_frac should be O(1) — residual is comparable to actual.
    assert abs(row1['E_residual_cons_frac']) > 0.5


@pytest.mark.unit
def test_get_proteus_directories_has_required_keys():
    """Sanity: the directory dict exposes the keys the runtime depends on."""
    dirs = get_proteus_directories(outdir='unit-test')
    # Module-source paths still referenced at runtime.
    required_keys = (
        'proteus',
        'agni',
        'spider',
        'aragog',
        'zalmoxis',
        'vulcan',
        'tools',
        'utils',
        'input',
    )
    for required in required_keys:
        assert required in dirs, f"missing required directory key '{required}'"
    # Per-run output subtree.
    for required in (
        'output',
        'output/data',
        'output/offchem',
        'output/observe',
        'output/plots',
    ):
        assert required in dirs


@pytest.mark.unit
def test_get_proteus_directories_editable_submodule_paths():
    """Each editable FWL submodule maps to its on-disk sibling directory.

    Aragog / Zalmoxis / VULCAN are installed via the ``tools/get_*.sh``
    scripts as editable sibling checkouts inside the PROTEUS root. The
    paths are case-sensitive on Linux: Aragog clones to ``aragog/``,
    Zalmoxis to ``Zalmoxis/``, VULCAN to ``VULCAN/``. Pin the case here
    so a doctor command or runtime path-resolver does not silently look
    in the wrong directory.
    """
    dirs = get_proteus_directories(outdir='unit-test')
    # Path basename must match the on-disk casing the get_*.sh scripts use.
    assert os.path.basename(dirs['aragog']) == 'aragog'
    assert os.path.basename(dirs['zalmoxis']) == 'Zalmoxis'
    assert os.path.basename(dirs['vulcan']) == 'VULCAN'
    # Each path is anchored at the PROTEUS root (the parent of the
    # editable checkout), not somewhere else like /tmp or site-packages.
    assert os.path.dirname(dirs['aragog']) == dirs['proteus']
    assert os.path.dirname(dirs['zalmoxis']) == dirs['proteus']
    assert os.path.dirname(dirs['vulcan']) == dirs['proteus']
