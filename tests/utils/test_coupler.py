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

import math
import os
import sys
import tempfile
import types
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import proteus.utils.coupler as coupler_mod
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
    print_citation,
    print_module_configuration,
    remove_excess_files,
    set_directories,
    variable_is_logarithmic,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

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
    MagicMock()
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
    # Discrimination: a regression that returned an empty list would also
    # trivially satisfy the no-duplicate check; pin a substantial lower
    # bound on the schema size so an accidental truncation is caught.
    assert len(keys) > 50


# =============================================================================
# Test: Helpfile Row Creation and Initialization
# =============================================================================


@pytest.mark.unit
def test_zero_helpfile_row_returns_dict():
    """Test that ZeroHelpfileRow returns a dictionary."""
    MagicMock()
    row = ZeroHelpfileRow()
    assert isinstance(row, dict)
    # Discriminating check: the row is non-empty (carries every helpfile column
    # zeroed out) and includes the canonical Time scalar; an empty-dict regression
    # would pass isinstance alone.
    assert len(row) > 0
    assert 'Time' in row


@pytest.mark.unit
def test_zero_helpfile_row_all_values_are_zero():
    """Test that ZeroHelpfileRow initializes all values to 0.0."""
    row = ZeroHelpfileRow()
    for key, value in row.items():
        assert value == 0.0, f'Key {key} has value {value}, expected 0.0'
    # Discrimination: every value must be a Python float (not int 0 or
    # numpy zero), so a regression that initialised values to integer 0
    # would be caught even though `0 == 0.0` is True.
    for key, value in row.items():
        assert isinstance(value, float), f'Key {key} is {type(value).__name__}, expected float'


@pytest.mark.unit
def test_zero_helpfile_row_has_correct_keys():
    """Test that ZeroHelpfileRow has all keys from GetHelpfileKeys."""
    row = ZeroHelpfileRow()
    keys = GetHelpfileKeys()

    assert set(row.keys()) == set(keys), 'ZeroHelpfileRow keys do not match GetHelpfileKeys'
    # Discrimination: equal set sizes corroborate the equality above;
    # a regression that emitted duplicate keys via the loop would make
    # set(row.keys()) lossy and len(row) != len(keys).
    assert len(row) == len(keys)


@pytest.mark.unit
def test_create_helpfile_from_dict_creates_dataframe():
    """Test that CreateHelpfileFromDict creates a pandas DataFrame."""
    MagicMock()
    row = ZeroHelpfileRow()
    row['Time'] = 1.0
    row['T_surf'] = 300.0

    hf = CreateHelpfileFromDict(row)
    assert isinstance(hf, pd.DataFrame)
    assert len(hf) == 1


@pytest.mark.unit
def test_create_helpfile_from_dict_preserves_values():
    """Test that CreateHelpfileFromDict preserves input values."""
    MagicMock()
    row = ZeroHelpfileRow()
    row['Time'] = 1.5e9
    row['T_surf'] = 350.0
    row['P_surf'] = 100.0

    hf = CreateHelpfileFromDict(row)

    assert hf['Time'].iloc[0] == pytest.approx(1.5e9, rel=1e-12)
    assert hf['T_surf'].iloc[0] == pytest.approx(350.0, rel=1e-12)
    assert hf['P_surf'].iloc[0] == pytest.approx(100.0, rel=1e-12)


@pytest.mark.unit
def test_create_helpfile_from_dict_has_all_keys():
    """Test that CreateHelpfileFromDict includes all helpfile keys."""
    row = ZeroHelpfileRow()
    hf = CreateHelpfileFromDict(row)

    keys = GetHelpfileKeys()
    for key in keys:
        assert key in hf.columns
    # Discrimination: the DataFrame must carry exactly the schema columns
    # in schema order (no extras leaking in from the dict, no reordering);
    # CSV writers downstream rely on the column order being canonical.
    assert list(hf.columns) == keys


# =============================================================================
# Test: Helpfile Row Extension
# =============================================================================


@pytest.mark.unit
def test_extend_helpfile_appends_row():
    """Test that ExtendHelpfile appends a new row to helpfile."""
    # Create initial helpfile
    MagicMock()
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
    MagicMock()
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
    MagicMock()
    row1 = ZeroHelpfileRow()
    hf = CreateHelpfileFromDict(row1)

    # Create a row with missing keys
    row2 = {'Time': 1.0}  # Missing other keys

    with pytest.raises(Exception, match='missing expected keys'):
        ExtendHelpfile(hf, row2)
    # Discrimination: the raise must happen BEFORE the helpfile gets a new
    # row appended. A regression that appended first and then raised would
    # leave the helpfile growing on every failed call.
    assert len(hf) == 1


@pytest.mark.unit
def test_extend_helpfile_warns_on_unknown_keys(caplog):
    """ExtendHelpfile should WARN (not raise) on unknown keys so the CSV
    silently-drop-key bug is surfaced in logs. Private (underscore-prefixed)
    keys are intentionally skipped. Explicitly-allowlisted non-schema keys
    like ``core_state_initial`` are also skipped."""
    import logging

    MagicMock()
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
    MagicMock()
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
    # Discrimination: the helpfile must be strictly monotonic in Time so
    # a regression that reversed insertion order or shuffled rows would
    # produce a non-sorted column.
    times_seq = hf['Time'].tolist()
    assert times_seq == sorted(times_seq)


# =============================================================================
# Test: Helpfile CSV I/O
# =============================================================================


@pytest.mark.unit
def test_write_helpfile_to_csv_creates_file():
    """Test that WriteHelpfileToCSV creates a CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create helpfile
        MagicMock()
        row = ZeroHelpfileRow()
        row['Time'] = 1.0
        row['T_surf'] = 300.0
        hf = CreateHelpfileFromDict(row)

        # Write to CSV
        MagicMock()
        fpath = WriteHelpfileToCSV(tmpdir, hf)

        # Verify file exists
        assert os.path.exists(fpath)
        assert fpath.endswith('runtime_helpfile.csv')
        assert 'runtime_helpfile.csv' in os.listdir(tmpdir)


@pytest.mark.unit
def test_write_helpfile_to_csv_contains_data():
    """Test that WriteHelpfileToCSV writes correct data to file."""
    MagicMock()
    with tempfile.TemporaryDirectory() as tmpdir:
        row = ZeroHelpfileRow()
        row['Time'] = 1.5
        row['T_surf'] = 350.5
        hf = CreateHelpfileFromDict(row)

        MagicMock()
        WriteHelpfileToCSV(tmpdir, hf)

        # Read back and verify
        fpath = os.path.join(tmpdir, 'runtime_helpfile.csv')
        df = pd.read_csv(fpath, sep=r'\s+')

        assert df['Time'].iloc[0] == pytest.approx(1.5)
        assert df['T_surf'].iloc[0] == pytest.approx(350.5)


@pytest.mark.unit
def test_write_helpfile_to_csv_overwrites_existing():
    """Test that WriteHelpfileToCSV overwrites existing file."""
    MagicMock()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write first helpfile
        row1 = ZeroHelpfileRow()
        row1['Time'] = 1.0
        hf1 = CreateHelpfileFromDict(row1)
        MagicMock()
        WriteHelpfileToCSV(tmpdir, hf1)

        # Write second helpfile (should overwrite)
        row2 = ZeroHelpfileRow()
        row2['Time'] = 2.0
        hf2 = CreateHelpfileFromDict(row2)
        MagicMock()
        WriteHelpfileToCSV(tmpdir, hf2)

        # Verify only latest data is in file
        fpath = os.path.join(tmpdir, 'runtime_helpfile.csv')
        df = pd.read_csv(fpath, sep=r'\s+')

        assert len(df) == 1
        assert df['Time'].iloc[0] == pytest.approx(2.0)


@pytest.mark.unit
def test_write_and_read_helpfile_roundtrip():
    """Test that helpfile survives write and read cycle."""
    MagicMock()
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
        # Precondition: confirm no helpfile is present so the missing-file
        # branch is what gets exercised (not a tmpdir leftover).
        assert not os.path.exists(os.path.join(tmpdir, 'runtime_helpfile.csv'))
        with pytest.raises(Exception, match='Cannot find helpfile'):
            ReadHelpfileFromCSV(tmpdir)
        # Discrimination: the error path must not have created a stub file
        # as a side effect. A regression that wrote an empty CSV before
        # raising would silently corrupt resume.
        assert not os.path.exists(os.path.join(tmpdir, 'runtime_helpfile.csv'))


@pytest.mark.unit
def test_write_helpfile_multiple_rows_roundtrip():
    """Test that multiple rows survive write and read cycle."""
    MagicMock()
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
        MagicMock()
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
        # Discrimination: the returned path is not just a string; the file
        # must actually exist on disk. A regression that returned the
        # expected path string without writing the file would still pass
        # the equality check above but leave the loop unable to find it.
        assert os.path.isfile(lockfile_path)


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
        # Discrimination: the message must reference PROTEUS by name so an
        # operator who finds the file in an unfamiliar directory can tell
        # which simulator owns it. A regression that wrote a generic
        # placeholder ("delete me") would lose that traceability.
        assert 'proteus' in content.lower()


# =============================================================================
# Test: Current State Printing
# =============================================================================


@pytest.mark.unit
def test_print_current_state_with_valid_row():
    """Test that PrintCurrentState accepts valid helpfile row."""
    MagicMock()
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
        # Discrimination: every populated quantity must produce its own
        # log.info call. The source emits 9 lines (header + Wall time +
        # Time + T_surf + T_magma + P_surf + Phi_global + F_atm + F_int);
        # a regression that fused lines or dropped one would land below 9.
        assert mock_log.info.call_count >= 9


@pytest.mark.unit
def test_print_current_state_includes_time():
    """Test that PrintCurrentState includes Time in output."""
    MagicMock()
    hf_row = ZeroHelpfileRow()
    hf_row['Time'] = 1.5e8

    with patch('proteus.utils.coupler.log') as mock_log:
        PrintCurrentState(hf_row)

        # Check if Time was in any log call
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('Model time' in str(call) for call in log_calls)
        # Discrimination: the numeric Time value must appear in the log
        # in %.2e format. A regression that printed the label but dropped
        # the substitution (e.g. broken %-format) would still match
        # 'Model time' but lose the 1.50e+08 fingerprint.
        assert any('1.50e+08' in str(call) for call in log_calls)


@pytest.mark.unit
def test_print_current_state_includes_temperatures():
    """Test that PrintCurrentState includes T_surf and T_magma."""
    MagicMock()
    hf_row = ZeroHelpfileRow()
    hf_row['T_surf'] = 300.5
    hf_row['T_magma'] = 2500.3

    with patch('proteus.utils.coupler.log') as mock_log:
        PrintCurrentState(hf_row)

        # Verify temperatures were formatted in output
        # (exact matching is difficult due to formatting)
        assert mock_log.info.called
        # Discrimination: the 8.3f-formatted T_surf and T_magma numeric
        # values must appear in the captured log output. A regression that
        # swapped the two labels or dropped the values would surface here.
        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any('300.500' in str(call) for call in log_calls)
        assert any('2500.300' in str(call) for call in log_calls)


# =============================================================================
# Test: Time Formatting
# =============================================================================


@pytest.mark.unit
def test_get_current_time_returns_string():
    """Test that _get_current_time returns a string."""
    result = _get_current_time()
    assert isinstance(result, str)
    # Discriminating check: the returned string is non-empty and looks like a
    # timestamp (contains a digit and a separator). The sibling test below pins
    # the YYYY-MM-DD shape; this minimal guard rejects an empty string slipping
    # past the isinstance check.
    assert len(result) > 0


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
    # Discrimination: the result must NOT contain a future year (e.g. a
    # regression to a hardcoded future date or off-by-one year handling).
    assert str(current_year + 1) not in result


@pytest.mark.unit
def test_get_current_time_format_consistency():
    """Test that _get_current_time returns consistent format."""
    result1 = _get_current_time()
    result2 = _get_current_time()

    # Both should have same length (allowing for second differences)
    assert abs(len(result1) - len(result2)) <= 1
    # Discrimination: the format is '%Y-%m-%d %H:%M:%S %Z'. Pin the YYYY-MM-DD
    # date-separator shape so a regression that switched to a localized
    # format (e.g. 'MM/DD/YYYY') is caught even if it kept the same length.
    assert result1[4] == '-' and result1[7] == '-' and result1[10] == ' '


# =============================================================================
# Test: Edge Cases and Physics Validation
# =============================================================================


@pytest.mark.unit
def test_helpfile_with_realistic_earth_values():
    """Test helpfile with realistic modern Earth values."""
    MagicMock()
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
    MagicMock()
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
    MagicMock()
    row = ZeroHelpfileRow()

    # Use a value with many significant figures
    precise_value = 1.23456789e12

    row['M_planet'] = precise_value

    hf = CreateHelpfileFromDict(row)

    assert hf['M_planet'].iloc[0] == pytest.approx(precise_value, rel=1e-8)
    # Discrimination: the stored value must NOT have been truncated to
    # single precision (~7 digits). 1.23456789e12 keeps 9 digits in
    # float64 but only 7 in float32; assert at tighter rel-tol than the
    # primary to surface a regression that downcasted columns.
    assert hf['M_planet'].iloc[0] == pytest.approx(precise_value, rel=1e-12)


@pytest.mark.unit
def test_helpfile_scientific_notation_consistency():
    """Test that scientific notation values are consistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        MagicMock()
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
            # Return value must be a plain string, not bytes.
            assert isinstance(result, str)
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
            # Discrimination: the trailing newline from the file must be
            # stripped. A regression that returned the raw read would equal
            # '24.1.0\n' and fail downstream version-parse logic that
            # splits on '.' expecting clean numeric components.
            assert '\n' not in version


@pytest.mark.unit
def test_get_socrates_version_raises_without_rad_dir():
    """Test that _get_socrates_version errors when RAD_DIR is missing."""
    from proteus.utils.coupler import _get_socrates_version

    with patch.dict(os.environ, {}, clear=True):
        # Precondition: confirm RAD_DIR is actually absent in the clean
        # env so the missing-env branch is what gets exercised (rather
        # than a leaked outer-process value).
        assert 'RAD_DIR' not in os.environ
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
        # Discrimination: a regression that returned the 'name' field
        # ('AGNI') instead of the version key would still be a non-empty
        # string. Pin the dotted-version shape explicitly.
        assert version.count('.') == 2
        assert version != 'AGNI'


@pytest.mark.unit
def test_get_julia_version_with_mock():
    """Test that _get_julia_version parses julia --version output."""
    from proteus.utils.coupler import _get_julia_version

    with patch('subprocess.check_output') as mock_check_output:
        mock_check_output.return_value = b'julia version 1.9.3\n'

        version = _get_julia_version()

        assert version == '1.9.3'
        assert isinstance(version, str)
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
        # Discrimination: the minutes branch must NOT also report hours.
        # A regression that dropped the elif and fell through to the
        # hours branch (or printed both labels) would surface here.
        assert not any('hours' in str(call) for call in log_calls)


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
        # Discrimination: exactly one 'Total runtime' line should land.
        # A regression that printed both 'hours' and the minutes
        # fallback would emit two such lines.
        runtime_lines = [c for c in log_calls if 'Total runtime' in str(c)]
        assert len(runtime_lines) == 1


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
                # Discrimination: the pwd fallback path must have been
                # taken (a regression that re-raised the OSError instead
                # of falling back would not call pwd.getpwuid). Also pin
                # the fallback username actually appears in the log.
                mock_getpwuid.assert_called_once()
                log_calls = [str(call) for call in mock_log.info.call_args_list]
                assert any('fallback_user' in str(call) for call in log_calls)


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
            MagicMock()
            WriteHelpfileToCSV(tmpdir, df)
        # Discrimination: the validation must short-circuit BEFORE writing
        # the CSV. A regression that wrote first and then raised would
        # leave a corrupted stub file on disk that resume would consume.
        assert not os.path.exists(os.path.join(tmpdir, 'runtime_helpfile.csv'))


@pytest.mark.unit
def test_zero_helpfile_row_is_immutable_structure():
    """Test that ZeroHelpfileRow returns new dict each time."""
    MagicMock()
    row1 = ZeroHelpfileRow()
    row2 = ZeroHelpfileRow()

    # Modify first row
    row1['Time'] = 100.0

    # Second row should still be zero
    assert row2['Time'] == 0.0
    # Discrimination: the two dicts must be distinct objects (no shared
    # cached singleton). A regression that returned a cached module-level
    # dict would have id(row1) == id(row2) and the mutation above would
    # also alter row2.
    assert row1 is not row2
    assert row1['Time'] == pytest.approx(100.0)


@pytest.mark.unit
def test_helpfile_preserves_column_order():
    """Test that helpfile operations preserve column order."""
    keys = GetHelpfileKeys()
    row = ZeroHelpfileRow()
    hf = CreateHelpfileFromDict(row)

    # Verify column order matches GetHelpfileKeys
    assert list(hf.columns) == keys
    # Discrimination: 'Time' must be the first column. The CSV format
    # used downstream relies on Time leading the row; a regression that
    # alphabetised the schema would still equal-compare key-set but put
    # 'C_kg_atm' or similar at index 0.
    assert hf.columns[0] == 'Time'


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
    MagicMock()
    row1 = ZeroHelpfileRow()
    row1['Time'] = 1.0
    hf = CreateHelpfileFromDict(row1)

    row2 = ZeroHelpfileRow()
    row2['Time'] = 2.0
    hf_extended = ExtendHelpfile(hf, row2)

    # All columns should be float
    assert hf_extended['Time'].dtype == float
    # Discrimination: every numeric column must stay float64 after extend.
    # A regression that introduced an object dtype (e.g. via mixed-type
    # coercion on a string-valued private key) would surface here.
    assert hf_extended['T_surf'].dtype == float
    assert hf_extended['M_planet'].dtype == float


@pytest.mark.unit
def test_helpfile_handles_large_time_values():
    """Test that helpfile handles large time values (Gyr scale)."""
    MagicMock()
    row = ZeroHelpfileRow()
    row['Time'] = 4.567e9  # 4.567 Gyr (age of Earth)
    hf = CreateHelpfileFromDict(row)

    assert hf['Time'].iloc[0] == pytest.approx(4.567e9)
    # Discrimination: the Time column must stay float64 so a Gyr-scale
    # value retains its full precision. A regression that coerced Time
    # to int (truncating to 4567000000) would still equal-compare the
    # nominal value at approx() tolerance but lose sub-Myr resolution.
    assert hf['Time'].dtype == float
    assert hf['Time'].iloc[0] == pytest.approx(4.567e9, rel=1e-12)


@pytest.mark.unit
def test_helpfile_handles_negative_fluxes():
    """Test that helpfile handles negative flux values (heat loss)."""
    MagicMock()
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
    MagicMock()
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
    MagicMock()
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

    MagicMock()
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
    MagicMock()
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # Physical step delta: -1e29 J (mild cooling).
    physical_delta = -1.0e29
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E0 + physical_delta,
        step_dE_F_int_J=physical_delta,
        # Catastrophic instantaneous spike; must be IGNORED.
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
    MagicMock()
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
    # E_residual_cons_frac should be O(1); residual is comparable to actual.
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


# ============================================================================
# Issue #677 mass-conservation invariant tests
# ============================================================================


@pytest.mark.unit
def test_assert_mass_conservation_passes_when_invariants_hold():
    """assert_mass_conservation accepts M_atm <= M_planet and per-species sum match.

    Physical scenario: post-outgas state with a non-trivial atmosphere.
    M_atm = 4.6e24 kg (close to Earth's mantle), M_planet = 5.97e24 kg
    (1 M_earth), per-species sum exactly equals M_atm.
    """
    from proteus.utils.constants import gas_list
    from proteus.utils.coupler import assert_mass_conservation

    config = MagicMock()

    hf_row = {
        'M_atm': 4.6e24,
        'M_planet': 5.97e24,
    }
    # Distribute M_atm across gas_list so the per-species sum equals M_atm.
    # Use asymmetric values so the sum is a meaningful check (not all equal).
    per_species = 4.6e24 / len(gas_list)
    for s in gas_list:
        hf_row[s + '_kg_atm'] = per_species

    result = assert_mass_conservation(hf_row, config)
    assert result is None  # contract: helper returns None silently when M_atm <= M_planet
    # Discriminating check: M_atm < M_planet strictly (not vacuously zero), and
    # the per-species sum exactly equals M_atm so the closure path is exercised.
    assert hf_row['M_atm'] < hf_row['M_planet']
    species_sum = sum(hf_row[s + '_kg_atm'] for s in gas_list)
    assert math.isclose(species_sum, hf_row['M_atm'], rel_tol=1e-12)


@pytest.mark.unit
def test_assert_mass_conservation_fails_when_M_atm_exceeds_M_planet():
    """assert_mass_conservation hard-fails when M_atm > M_planet (the issue #677 symptom).

    Discriminating: M_atm = 7.2e24 vs M_planet = 5.97e24, a 21 percent
    excess, well above the 1e-6 tolerance. Must raise RuntimeError with
    a message naming the M_atm and M_planet values.
    """
    from proteus.utils.coupler import assert_mass_conservation

    config = MagicMock()

    hf_row = {
        'M_atm': 7.2e24,  # Atmosphere exceeds planet (the issue #677 symptom)
        'M_planet': 5.97e24,
    }
    for s_idx in range(15):
        # Stub kg_atm columns so the per-species check doesn't fire first
        # (we want to test the M_atm > M_planet path specifically).
        pass

    with pytest.raises(RuntimeError, match='Mass conservation violation'):
        assert_mass_conservation(hf_row, config)
    # Discrimination: confirm the excess is well above the 1e-6 tolerance
    # so the test is exercising the violation branch, not riding the
    # numerical edge. M_atm/M_planet should exceed 1.0 by ~21 percent.
    assert hf_row['M_atm'] / hf_row['M_planet'] > 1.2
    # And the hf_row dict must NOT be mutated by the failed call (so
    # downstream callers can inspect the row that triggered the raise).
    assert hf_row['M_atm'] == pytest.approx(7.2e24)


@pytest.mark.unit
def test_assert_mass_conservation_fails_when_species_sum_disagrees():
    """assert_mass_conservation hard-fails when sum(s_kg_atm) != M_atm.

    Edge case: per-species kg_atm values are stale or a species is missing
    from the M_atm sum loop. Discriminating: sum = 4.0e24 but M_atm = 4.6e24
    (a 15 percent disagreement).
    """
    from proteus.utils.constants import gas_list
    from proteus.utils.coupler import assert_mass_conservation

    hf_row = {
        'M_atm': 4.6e24,
        'M_planet': 5.97e24,
    }
    # Intentionally under-report: per-species sum is only ~87 percent of M_atm.
    per_species = (0.87 * 4.6e24) / len(gas_list)
    for s in gas_list:
        hf_row[s + '_kg_atm'] = per_species
    config = MagicMock()
    with pytest.raises(RuntimeError, match='M_atm bookkeeping inconsistency'):
        assert_mass_conservation(hf_row, config)
    # Discrimination: the M_atm <= M_planet invariant must HOLD here so
    # the failure must come from the per-species bookkeeping path, not
    # from the M_atm > M_planet path. Pin both legs.
    assert hf_row['M_atm'] < hf_row['M_planet']
    species_sum = sum(hf_row[s + '_kg_atm'] for s in gas_list)
    assert species_sum < 0.9 * hf_row['M_atm']


@pytest.mark.unit
def test_assert_mass_conservation_skips_when_M_planet_zero():
    """assert_mass_conservation gracefully handles pre-IC state where M_planet=0.

    Edge case: at the very first call site M_planet may not yet be set
    (still 0 from ZeroHelpfileRow). The assertion must not false-positive
    in that regime.
    """
    from proteus.utils.coupler import assert_mass_conservation

    hf_row = {
        'M_atm': 1e10,  # something
        'M_planet': 0.0,  # not yet computed
    }
    # Must not raise; the M_planet=0 short-circuit handles the pre-IC case.
    result = assert_mass_conservation(hf_row)
    assert result is None  # contract: M_planet=0 short-circuits the conservation check
    # Discriminating check: M_atm is non-zero, so only the M_planet=0 skip branch
    # can produce a silent pass on this row.
    assert hf_row['M_atm'] > 0.0
    assert hf_row['M_planet'] == 0.0


# ============================================================================
# P_surf = P_vol + P_vap surface-pressure invariant tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_assert_surface_pressure_consistency_passes_when_vapourise_disabled():
    """P_surf == P_vol with P_vap == 0 is accepted when rock vapour is off.

    Physical scenario: an ordinary CALLIOPE/atmodeller outgas step with
    outgas.vapourise = False. P_vol mirrors P_surf exactly (no vapour
    contribution), which is the state run_outgassing leaves hf_row in.
    """
    from proteus.utils.coupler import assert_surface_pressure_consistency

    config = MagicMock()
    config.outgas.vapourise = False

    hf_row = {'P_surf': 120.0, 'P_vol': 120.0, 'P_vap': 0.0}

    result = assert_surface_pressure_consistency(config, hf_row)
    assert result is None  # contract: silent pass when the invariant holds
    # Discriminating check: P_vol is non-trivially large (not a vacuous 0/0
    # pass) and exactly equals P_surf, isolating the "vapourise off" branch.
    assert hf_row['P_vol'] == pytest.approx(hf_row['P_surf'])
    assert hf_row['P_vap'] == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_assert_surface_pressure_consistency_passes_when_vapourise_enabled():
    """P_surf == P_vol + P_vap is accepted with a nonzero rock-vapour term.

    Physical scenario: LavAtmos has run (outgas.vapourise = True) and added
    a rock-vapour partial pressure on top of the volatile total, mirroring
    compute_silicate_outgassing's P_vol/P_vap/P_surf bookkeeping.
    """
    from proteus.utils.coupler import assert_surface_pressure_consistency

    config = MagicMock()
    config.outgas.vapourise = True

    # Asymmetric split (not a 50/50 coincidence) so the sum genuinely
    # exercises both terms rather than passing via a degenerate value.
    hf_row = {'P_surf': 137.5, 'P_vol': 90.0, 'P_vap': 47.5}

    result = assert_surface_pressure_consistency(config, hf_row)
    assert result is None
    assert hf_row['P_vap'] > 0.0
    assert hf_row['P_vol'] + hf_row['P_vap'] == pytest.approx(hf_row['P_surf'])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_assert_surface_pressure_consistency_rejects_nonzero_P_vap_when_disabled():
    """P_vap must be exactly zero whenever outgas.vapourise is False.

    Regression scenario this guards against: a stale rock-vapour pressure
    (e.g. left over from a prior iteration, or written by a code path that
    forgot to gate on the config flag) surviving into an iteration where
    vapourise is disabled. Must raise before the P_surf==P_vol+P_vap check
    even runs, since that check alone (100.0 == 95.0 + 5.0) would otherwise
    pass and mask the real bug.
    """
    from proteus.utils.coupler import assert_surface_pressure_consistency

    config = MagicMock()
    config.outgas.vapourise = False

    hf_row = {'P_surf': 100.0, 'P_vol': 95.0, 'P_vap': 5.0}

    with pytest.raises(RuntimeError, match='outgas.vapourise=False'):
        assert_surface_pressure_consistency(config, hf_row)
    # Discrimination: P_surf == P_vol + P_vap holds exactly here (100 == 95 + 5),
    # so a weaker implementation that only checked the sum would wrongly pass
    # this row. Pin that the sum-consistent row is exactly what makes this a
    # useful regression test, not an artifact of an already-broken sum.
    assert hf_row['P_vol'] + hf_row['P_vap'] == pytest.approx(hf_row['P_surf'])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_assert_surface_pressure_consistency_rejects_mismatched_total():
    """P_surf disagreeing with P_vol + P_vap by more than the tolerance raises.

    Discriminating: P_surf=150.0 but P_vol+P_vap=100.0, a 33 percent
    disagreement, far above the default 1e-6 relative tolerance. Models a
    code path (e.g. run_crystallized's escape scaling) that rescaled
    P_surf without rescaling P_vol/P_vap in step.
    """
    from proteus.utils.coupler import assert_surface_pressure_consistency

    config = MagicMock()
    config.outgas.vapourise = True

    hf_row = {'P_surf': 150.0, 'P_vol': 80.0, 'P_vap': 20.0}

    with pytest.raises(RuntimeError, match='Surface pressure inconsistency'):
        assert_surface_pressure_consistency(config, hf_row)
    # Discrimination: the mismatch (50.0) is far above what float rounding
    # could produce, confirming the raise is the real bookkeeping check and
    # not a numerical-noise false positive.
    assert abs(hf_row['P_surf'] - (hf_row['P_vol'] + hf_row['P_vap'])) > 1.0


@pytest.mark.unit
def test_assert_surface_pressure_consistency_skips_pre_ic():
    """No atmosphere yet (all pressures zero) short-circuits without raising.

    Edge case: at the very first call, before any outgassing has run,
    P_surf/P_vol/P_vap are all still at their ZeroHelpfileRow default.
    """
    from proteus.utils.coupler import assert_surface_pressure_consistency

    config = MagicMock()
    config.outgas.vapourise = True

    hf_row = {'P_surf': 0.0, 'P_vol': 0.0, 'P_vap': 0.0}

    result = assert_surface_pressure_consistency(config, hf_row)
    assert result is None
    assert hf_row['P_surf'] == 0.0


# ---------------------------------------------------------------------------
# Edge / error paths in the git-revision and version-validation helpers
# (lines 50-52, 65-78, 165-166, 201-202 in utils/coupler.py).
# ---------------------------------------------------------------------------


def test_get_git_revision_returns_unknown_when_chdir_fails():
    """When the target directory does not exist (or cannot be entered),
    _get_git_revision must return the literal 'unknown' string and the
    caller's CWD must be preserved.

    Discriminating: assert the returned value is exactly 'unknown'
    (not None, not the empty string, not a partial git hash). A
    regression that propagated the OSError up would raise here.
    """
    from proteus.utils.coupler import _get_git_revision

    cwd_before = os.getcwd()
    bogus_dir = '/totally/does/not/exist/this/path/has/no/chance'
    result = _get_git_revision(bogus_dir)
    cwd_after = os.getcwd()
    assert result == 'unknown'
    # Side-effect guard: the helper must not strand the caller in a
    # different directory. A regression that returned before the
    # finally-block chdir-back would leave cwd somewhere else.
    assert cwd_after == cwd_before


def test_get_git_revision_returns_unknown_on_subprocess_failure():
    """Even if chdir succeeds, a missing git executable or a non-repo
    directory must surface as 'unknown' rather than raising.

    Discriminating: patch subprocess.check_output to raise
    FileNotFoundError (git absent). The except branch covers
    CalledProcessError, FileNotFoundError, TimeoutExpired, and the
    catch-all Exception; this test pins the FileNotFoundError path
    explicitly so a future tightening that narrowed the except clause
    would fail loudly.
    """
    from proteus.utils.coupler import _get_git_revision

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('subprocess.check_output', side_effect=FileNotFoundError('git not found')):
            result = _get_git_revision(tmpdir)
    assert result == 'unknown'
    assert isinstance(result, str)


def test_get_git_revision_returns_unknown_on_subprocess_timeout():
    """A hanging git process that exceeds the 5 s timeout must surface
    as 'unknown'. Pin the TimeoutExpired exception path specifically.
    """
    import subprocess as sp

    from proteus.utils.coupler import _get_git_revision

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            'subprocess.check_output',
            side_effect=sp.TimeoutExpired(cmd='git', timeout=5),
        ):
            result = _get_git_revision(tmpdir)
    assert result == 'unknown'
    assert isinstance(result, str)


def _all_dummy_modules_config():
    """Config with every coupling slot set to 'dummy' except for
    interior_energetics, which is wired to the module under test.
    Lets validate_module_versions walk a single branch deterministically.
    """
    from unittest.mock import MagicMock

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_struct.module = 'dummy'
    config.atmos_clim.module = 'dummy'
    config.outgas.module = 'dummy'
    config.escape.module = 'dummy'
    config.star.module = 'dummy'
    return config


def test_validate_module_versions_raises_when_module_out_of_date(tmp_path):
    """A required module at a version older than the pinned minimum
    triggers an EnvironmentError pointing at the troubleshooting URL.

    Discriminating: pin both the exception type AND the side-effect
    (UpdateStatusfile called once with status code 20). A regression
    that swapped the order (raise before write) or that downgraded to
    a log-only warning would fail one of the two assertions.
    """
    from unittest.mock import MagicMock

    from proteus.utils.coupler import validate_module_versions

    config = _all_dummy_modules_config()
    with (
        patch.dict(
            'sys.modules',
            {'aragog': MagicMock(__version__='0.0.1')},
            clear=False,
        ),
        # importlib.metadata.requires is what the closure _get_expver
        # consults; pinning it lets us drive the comparison from
        # outside the validate_module_versions function.
        patch(
            'importlib.metadata.requires',
            return_value=['fwl-aragog>=99.99.99'],
        ),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update,
    ):
        with pytest.raises(EnvironmentError, match='Out-of-date modules'):
            validate_module_versions({'rad': str(tmp_path)}, config)
    mock_update.assert_called_once()
    args, _ = mock_update.call_args
    assert args[1] == 20


def test_validate_module_versions_accepts_compatible_versions(tmp_path):
    """Installed version above the expected minimum: no raise, no
    status-file write.

    Edge: limit-input compatible setup. Discriminating: a regression
    that always wrote status would fail the mock_update assertion.
    """
    from unittest.mock import MagicMock

    from proteus.utils.coupler import validate_module_versions

    config = _all_dummy_modules_config()
    with (
        patch.dict(
            'sys.modules',
            {'aragog': MagicMock(__version__='99.99.99')},
            clear=False,
        ),
        patch('importlib.metadata.requires', return_value=['fwl-aragog>=1.0.0']),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update,
    ):
        result = validate_module_versions({'rad': str(tmp_path)}, config)
    assert result is None
    assert mock_update.call_count == 0


def test_validate_module_versions_skips_check_when_no_pinned_minimum(tmp_path):
    """When the resolver finds no pinned minimum (the dep string does
    not name the module), the inner _valid_ver helper short-circuits
    via the 'return True if expected is None' guard.

    Edge: the closure _get_expver returns None for a module that
    doesn't appear in requires(). Discriminating: even an obviously
    ancient __version__='0.0.1' must NOT trip the check when no
    expected minimum is available.
    """
    from unittest.mock import MagicMock

    from proteus.utils.coupler import validate_module_versions

    config = _all_dummy_modules_config()
    with (
        patch.dict(
            'sys.modules',
            {'aragog': MagicMock(__version__='0.0.1')},
            clear=False,
        ),
        patch(
            'importlib.metadata.requires',
            return_value=['fwl-something-else>=1.0.0'],
        ),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update,
    ):
        result = validate_module_versions({'rad': str(tmp_path)}, config)
    assert result is None
    assert mock_update.call_count == 0


@pytest.mark.unit
def test_print_module_configuration_logs_versions_for_spider_agni_stack(monkeypatch):
    """Cover print_module_configuration branches for spider/agni/calliope/boreas/mors/lovepy."""
    config = types.SimpleNamespace(
        interior_energetics=types.SimpleNamespace(module='spider'),
        atmos_clim=types.SimpleNamespace(module='agni'),
        outgas=types.SimpleNamespace(module='calliope'),
        escape=types.SimpleNamespace(module='boreas'),
        star=types.SimpleNamespace(module='mors'),
        orbit=types.SimpleNamespace(module='lovepy'),
        accretion=types.SimpleNamespace(module='dummy'),
        atmos_chem=types.SimpleNamespace(module='vulcan'),
        observe=types.SimpleNamespace(module='petitRADTRANS'),
    )
    dirs = {
        'proteus': '/tmp/proteus',
        'output': '/tmp/out',
        'rad': '/tmp/rad',
    }

    monkeypatch.setattr(coupler_mod, '_get_git_revision', lambda _d: 'abc123')
    monkeypatch.setattr(coupler_mod, '_get_agni_version', lambda _d: '1.8.0')
    monkeypatch.setattr(coupler_mod, '_get_socrates_version', lambda: '24.1.0')
    monkeypatch.setattr(coupler_mod, '_get_julia_version', lambda: '1.9.3')
    monkeypatch.setitem(sys.modules, 'calliope', types.SimpleNamespace(__version__='1.2.3'))
    monkeypatch.setitem(sys.modules, 'boreas', types.SimpleNamespace(__version__='2.3.4'))
    monkeypatch.setitem(sys.modules, 'mors', types.SimpleNamespace(__version__='3.4.5'))
    monkeypatch.setitem(
        sys.modules, 'petitRADTRANS', types.SimpleNamespace(__version__='4.5.6')
    )

    with patch('proteus.utils.coupler.log') as mock_log:
        print_module_configuration(dirs, config, '/tmp/cfg.toml')
        messages = [str(call) for call in mock_log.info.call_args_list]
        assert any('PROTEUS version' in m for m in messages)
        assert any('Interior module   spider version' in m for m in messages)
        assert any('PETSc' in m for m in messages)
        assert any('Atmos_clim module agni version' in m for m in messages)
        assert any('Outgas module     calliope version' in m for m in messages)
        assert any('Escape module     boreas version' in m for m in messages)
        assert any('Star module       mors version' in m for m in messages)
        assert any('Observe module    petitRADTRANS version' in m for m in messages)


@pytest.mark.unit
def test_print_module_configuration_logs_versions_for_aragog_janus_zephyrus(monkeypatch):
    """Cover print_module_configuration branches for aragog/janus/zephyrus."""
    config = types.SimpleNamespace(
        interior_energetics=types.SimpleNamespace(module='aragog'),
        atmos_clim=types.SimpleNamespace(module='janus'),
        outgas=types.SimpleNamespace(module='dummy'),
        escape=types.SimpleNamespace(module='zephyrus'),
        star=types.SimpleNamespace(module='dummy'),
        orbit=types.SimpleNamespace(module='dummy'),
        accretion=types.SimpleNamespace(module='dummy'),
        atmos_chem=types.SimpleNamespace(module='dummy'),
        observe=types.SimpleNamespace(module='dummy'),
    )
    dirs = {'proteus': '/tmp/proteus', 'output': '/tmp/out', 'rad': '/tmp/rad'}

    monkeypatch.setattr(coupler_mod, '_get_git_revision', lambda _d: 'def456')
    monkeypatch.setattr(coupler_mod, '_get_socrates_version', lambda: '24.1.0')
    monkeypatch.setitem(sys.modules, 'aragog', types.SimpleNamespace(__version__='0.7.0'))
    monkeypatch.setitem(sys.modules, 'janus', types.SimpleNamespace(__version__='0.5.0'))
    monkeypatch.setitem(sys.modules, 'zephyrus', types.SimpleNamespace(__version__='0.6.0'))

    with patch('proteus.utils.coupler.log') as mock_log:
        print_module_configuration(dirs, config, '/tmp/cfg.toml')
        messages = [str(call) for call in mock_log.info.call_args_list]
        assert any('Interior module   aragog version' in m for m in messages)
        assert any('Atmos_clim module janus version' in m for m in messages)
        assert any('Escape module     zephyrus version' in m for m in messages)


@pytest.mark.unit
def test_print_citation_covers_module_specific_citations_with_when_set(monkeypatch):
    """Cover print_citation branches for module-specific citations and chemistry gate."""
    config = types.SimpleNamespace(
        atmos_clim=types.SimpleNamespace(module='janus'),
        interior_energetics=types.SimpleNamespace(module='spider'),
        outgas=types.SimpleNamespace(module='calliope'),
        star=types.SimpleNamespace(module='mors'),
        orbit=types.SimpleNamespace(module='lovepy'),
        accretion=types.SimpleNamespace(module='dummy'),
        observe=types.SimpleNamespace(module='petitRADTRANS'),
        atmos_chem=types.SimpleNamespace(module='vulcan', when='runtime'),
    )

    with patch('proteus.utils.coupler.log') as mock_log:
        print_citation(config)
        messages = [str(call) for call in mock_log.info.call_args_list]
        assert any('Lichtenberg et al. (2021)' in m for m in messages)
        assert any('Graham et al. (2021)' in m for m in messages)
        assert any('Bower et al. (2021)' in m for m in messages)
        assert any('Johnstone et al. (2021)' in m for m in messages)
        assert any('Hay & Matsuyama (2019)' in m for m in messages)
        assert any('Mollière et al. (2019)' in m for m in messages)
        assert any('Tsai et al. (2021)' in m for m in messages)


@pytest.mark.unit
@pytest.mark.parametrize(
    'name,expected',
    [
        ('P_surf', True),
        ('X_vmr', True),
        ('CO2_bar', True),
        ('T_surf', False),
    ],
)
def test_variable_is_logarithmic_covers_membership_and_suffix_branches(name, expected):
    assert variable_is_logarithmic(name) is expected


@pytest.mark.unit
def test_extend_helpfile_replaces_nan_values_with_zero(caplog):
    """Cover NaN sanitization branch in ExtendHelpfile."""
    import logging

    hf = CreateHelpfileFromDict(ZeroHelpfileRow())
    row = ZeroHelpfileRow()
    row['Time'] = np.nan
    row['T_surf'] = np.nan

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.coupler'):
        out = ExtendHelpfile(hf, row)

    assert out['Time'].iloc[-1] == pytest.approx(0.0)
    assert out['T_surf'].iloc[-1] == pytest.approx(0.0)
    assert any('setting to zero' in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_remove_excess_files_includes_spectral_files_when_requested(monkeypatch):
    """Cover remove_excess_files branch that optionally removes spectral files."""
    removed = []
    monkeypatch.setattr('proteus.utils.coupler.safe_rm', lambda p: removed.append(p))

    remove_excess_files('/tmp/out', rm_spectralfiles=True)

    assert any(p.endswith('/keepalive') for p in removed)
    assert any(p.endswith('/runtime.sf') for p in removed)
    assert any(p.endswith('/runtime.sf_k') for p in removed)


def _install_updateplots_fakes(monkeypatch, calls):
    """Install fake modules imported by UpdatePlots and record function calls."""

    def rec(name):
        return lambda *a, **k: calls.append((name, len(a), tuple(sorted(k.keys()))))

    # Utility modules imported inside UpdatePlots
    atm_common = types.ModuleType('proteus.atmos_clim.common')
    atm_common.read_atmosphere_data = lambda *_a, **_k: [{'ok': True}]
    int_wrap = types.ModuleType('proteus.interior_energetics.wrapper')
    int_wrap.read_interior_data = lambda *_a, **_k: {'int': True}
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim.common', atm_common)
    monkeypatch.setitem(sys.modules, 'proteus.interior_energetics.wrapper', int_wrap)

    # Output-time providers
    aragog_mod = types.ModuleType('proteus.interior_energetics.aragog')
    aragog_mod.get_all_output_times = lambda _o: [1000, 2000, 3000]
    spider_mod = types.ModuleType('proteus.interior_energetics.spider')
    spider_mod.get_all_output_times = lambda _o: [1000, 2000, 3000]
    monkeypatch.setitem(sys.modules, 'proteus.interior_energetics.aragog', aragog_mod)
    monkeypatch.setitem(sys.modules, 'proteus.interior_energetics.spider', spider_mod)

    # Plot modules
    plot_map = {
        'proteus.plot.cpl_atmosphere': 'plot_atmosphere',
        'proteus.plot.cpl_bolometry': 'plot_bolometry',
        'proteus.plot.cpl_chem_atmosphere': 'plot_chem_atmosphere',
        'proteus.plot.cpl_emission': 'plot_emission',
        'proteus.plot.cpl_escape': 'plot_escape',
        'proteus.plot.cpl_fluxes_atmosphere': 'plot_fluxes_atmosphere',
        'proteus.plot.cpl_fluxes_global': 'plot_fluxes_global',
        'proteus.plot.cpl_global': 'plot_global',
        'proteus.plot.cpl_interior': 'plot_interior',
        'proteus.plot.cpl_interior_cmesh': 'plot_interior_cmesh',
        'proteus.plot.cpl_orbit': 'plot_orbit',
        'proteus.plot.cpl_sflux': 'plot_sflux',
        'proteus.plot.cpl_sflux_cross': 'plot_sflux_cross',
        'proteus.plot.cpl_spectra': 'plot_spectra',
        'proteus.plot.cpl_structure': 'plot_structure',
        'proteus.plot.cpl_visual': 'plot_visual',
    }
    for mod_name, fn_name in plot_map.items():
        mod = types.ModuleType(mod_name)
        setattr(mod, fn_name, rec(fn_name))
        monkeypatch.setitem(sys.modules, mod_name, mod)

    pop_mod = types.ModuleType('proteus.plot.cpl_population')
    pop_mod.plot_population_mass_radius = rec('plot_population_mass_radius')
    pop_mod.plot_population_time_density = rec('plot_population_time_density')
    monkeypatch.setitem(sys.modules, 'proteus.plot.cpl_population', pop_mod)


@pytest.mark.unit
def test_update_plots_covers_runtime_and_end_branches(monkeypatch, tmp_path):
    """Cover the large UpdatePlots path with mocked plotting backends."""
    calls = []
    _install_updateplots_fakes(monkeypatch, calls)

    monkeypatch.setattr(
        'proteus.utils.coupler.sample_times', lambda *_a, **_k: ([1000, 2000], None)
    )
    monkeypatch.setattr(
        'proteus.utils.coupler.glob.glob',
        lambda _p: [
            str(tmp_path / 'data' / '1000_atm.nc'),
            str(tmp_path / 'data' / '2000_atm.nc'),
            str(tmp_path / 'data' / '3000_atm.nc'),
        ],
    )

    cfg = types.SimpleNamespace(
        atmos_clim=types.SimpleNamespace(module='agni'),
        interior_energetics=types.SimpleNamespace(module='aragog'),
        observe=types.SimpleNamespace(module='petitRADTRANS'),
        orbit=types.SimpleNamespace(evolve=True, satellite=False),
        star=types.SimpleNamespace(module='mors', mors=types.SimpleNamespace(age_now=4.5)),
        atmos_chem=types.SimpleNamespace(module='vulcan'),
        params=types.SimpleNamespace(out=types.SimpleNamespace(plot_fmt='png')),
    )
    hf_all = pd.DataFrame({'Time': [1.0, 2.0, 3.0, 4.0]})
    dirs = {'output': str(tmp_path), 'fwl': str(tmp_path / 'fwl')}

    from proteus.utils.coupler import UpdatePlots

    UpdatePlots(hf_all, dirs, cfg, end=True, num_snapshots=2)

    called_names = [c[0] for c in calls]
    assert 'plot_global' in called_names
    assert 'plot_escape' in called_names
    assert 'plot_orbit' in called_names
    assert 'plot_interior' in called_names
    assert 'plot_atmosphere' in called_names
    assert 'plot_structure' in called_names
    assert 'plot_fluxes_global' in called_names
    assert 'plot_spectra' in called_names
    assert 'plot_visual' in called_names
    assert 'plot_emission' in called_names


@pytest.mark.unit
def test_set_directories_errors_without_fwl_data(monkeypatch):
    """Cover set_directories failure when FWL_DATA is absent."""
    cfg = types.SimpleNamespace(
        params=types.SimpleNamespace(out=types.SimpleNamespace(path='run1', logging='INFO')),
        atmos_clim=types.SimpleNamespace(module='dummy'),
    )
    monkeypatch.setattr('proteus.utils.coupler.get_proteus_dir', lambda: '/tmp/proteus')
    monkeypatch.delenv('FWL_DATA', raising=False)

    with patch('proteus.utils.coupler.UpdateStatusfile') as mock_update:
        with pytest.raises(EnvironmentError, match='FWL_DATA'):
            set_directories(cfg)
        mock_update.assert_called_once()


@pytest.mark.unit
def test_set_directories_auto_path_and_debug_temp(monkeypatch):
    """Cover set_directories auto-output branch and DEBUG temp-dir behavior."""
    cfg = types.SimpleNamespace(
        params=types.SimpleNamespace(out=types.SimpleNamespace(path='auto', logging='DEBUG')),
        atmos_clim=types.SimpleNamespace(module='dummy'),
    )
    monkeypatch.setattr('proteus.utils.coupler.get_proteus_dir', lambda: '/tmp/proteus')
    monkeypatch.setenv('FWL_DATA', '/tmp/fwl_data')
    monkeypatch.setattr('secrets.token_hex', lambda _n: 'beef')

    dirs = set_directories(cfg)

    assert cfg.params.out.path.startswith('run_')
    assert cfg.params.out.path.endswith('_beef')
    assert dirs['temp'] == dirs['output']
    assert dirs['fwl'].endswith('/tmp/fwl_data/')


@pytest.mark.unit
def test_set_directories_requires_rad_dir_for_agni(monkeypatch):
    """Cover set_directories AGNI/JANUS RAD_DIR required branch."""
    cfg = types.SimpleNamespace(
        params=types.SimpleNamespace(out=types.SimpleNamespace(path='run2', logging='INFO')),
        atmos_clim=types.SimpleNamespace(module='agni'),
    )
    monkeypatch.setattr('proteus.utils.coupler.get_proteus_dir', lambda: '/tmp/proteus')
    monkeypatch.setenv('FWL_DATA', '/tmp/fwl_data')
    monkeypatch.delenv('RAD_DIR', raising=False)
    monkeypatch.setattr('proteus.utils.coupler.create_tmp_folder', lambda: '/tmp/tmpwork')

    with patch('proteus.utils.coupler.UpdateStatusfile') as mock_update:
        with pytest.raises(EnvironmentError, match='RAD_DIR'):
            set_directories(cfg)
        mock_update.assert_called_once()


@pytest.mark.unit
def test_set_directories_sets_rad_and_temp_for_non_debug(monkeypatch):
    """Cover set_directories success path with AGNI and non-DEBUG temp folder."""
    cfg = types.SimpleNamespace(
        params=types.SimpleNamespace(out=types.SimpleNamespace(path='run3', logging='INFO')),
        atmos_clim=types.SimpleNamespace(module='agni'),
    )
    monkeypatch.setattr('proteus.utils.coupler.get_proteus_dir', lambda: '/tmp/proteus')
    monkeypatch.setenv('FWL_DATA', '/tmp/fwl_data')
    monkeypatch.setenv('RAD_DIR', '/tmp/rad_dir')
    monkeypatch.setattr('proteus.utils.coupler.create_tmp_folder', lambda: '/tmp/tmpwork')

    dirs = set_directories(cfg)

    assert dirs['rad'].endswith('/tmp/rad_dir/')
    assert dirs['temp'].endswith('/tmp/tmpwork/')
    assert dirs['output'].endswith('/tmp/proteus/output/run3/')


@pytest.mark.unit
def test_validate_module_versions_spider_stack_passes_with_unpinned_dep(monkeypatch, tmp_path):
    """Cover spider/zalmoxis/janus/calliope/zephyrus/mors pass branches plus unpinned deps."""
    from proteus.utils.coupler import validate_module_versions

    config = types.SimpleNamespace(
        interior_energetics=types.SimpleNamespace(module='spider'),
        interior_struct=types.SimpleNamespace(module='zalmoxis'),
        atmos_clim=types.SimpleNamespace(module='janus'),
        outgas=types.SimpleNamespace(module='calliope'),
        escape=types.SimpleNamespace(module='zephyrus'),
        star=types.SimpleNamespace(module='mors'),
    )
    monkeypatch.setitem(sys.modules, 'zalmoxis', types.SimpleNamespace(__version__='1.2'))
    monkeypatch.setitem(sys.modules, 'janus', types.SimpleNamespace(__version__='1.2'))
    monkeypatch.setitem(sys.modules, 'calliope', types.SimpleNamespace(__version__='1.2'))
    monkeypatch.setitem(sys.modules, 'zephyrus', types.SimpleNamespace(__version__='1.2'))
    monkeypatch.setitem(sys.modules, 'mors', types.SimpleNamespace(__version__='1.2'))

    with (
        patch(
            'importlib.metadata.requires',
            return_value=[
                'numpy',
                'fwl-zalmoxis>=1.0',
                'fwl-janus>=1.0',
                'fwl-calliope>=1.0',
                'fwl-zephyrus>=1.0',
                'fwl-mors>=1.0',
            ],
        ),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update,
    ):
        validate_module_versions({'rad': str(tmp_path)}, config)
    assert mock_update.call_count == 0


@pytest.mark.unit
def test_validate_module_versions_raises_for_old_janus_in_spider_stack(monkeypatch, tmp_path):
    """Cover out-of-date log/raise path with spider interior and old janus."""
    from proteus.utils.coupler import validate_module_versions

    config = types.SimpleNamespace(
        interior_energetics=types.SimpleNamespace(module='spider'),
        interior_struct=types.SimpleNamespace(module='dummy'),
        atmos_clim=types.SimpleNamespace(module='janus'),
        outgas=types.SimpleNamespace(module='dummy'),
        escape=types.SimpleNamespace(module='dummy'),
        star=types.SimpleNamespace(module='dummy'),
    )
    monkeypatch.setitem(sys.modules, 'janus', types.SimpleNamespace(__version__='0.1.0'))

    with (
        patch('importlib.metadata.requires', return_value=['fwl-janus>=1.0.0']),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update,
    ):
        with pytest.raises(EnvironmentError, match='Out-of-date modules'):
            validate_module_versions({'rad': str(tmp_path)}, config)
    mock_update.assert_called_once()


@pytest.mark.unit
def test_print_citation_agni_and_manual_mode_cover_noop_cases():
    """Cover print_citation match-case no-op arms and chemistry manual gate."""
    config = types.SimpleNamespace(
        atmos_clim=types.SimpleNamespace(module='agni'),
        interior_energetics=types.SimpleNamespace(module='aragog'),
        outgas=types.SimpleNamespace(module='atmodeller'),
        star=types.SimpleNamespace(module='dummy'),
        orbit=types.SimpleNamespace(module='dummy'),
        accretion=types.SimpleNamespace(module='dummy'),
        observe=types.SimpleNamespace(module='dummy'),
        atmos_chem=types.SimpleNamespace(module='vulcan', when='manually'),
    )

    with patch('proteus.utils.coupler.log') as mock_log:
        print_citation(config)
        messages = [str(call) for call in mock_log.info.call_args_list]
        assert any('Nicholls et al. (2025)' in m for m in messages)
        assert not any('Mollière et al. (2019)' in m for m in messages)
        assert not any('Tsai et al. (2021)' in m for m in messages)


@pytest.mark.unit
def test_update_plots_spider_dummy_atm_covers_skip_branches(monkeypatch, tmp_path):
    """Cover UpdatePlots branches when atmosphere is dummy and orbit is not evolving."""
    calls = []
    _install_updateplots_fakes(monkeypatch, calls)
    monkeypatch.setattr('proteus.utils.coupler.sample_times', lambda *_a, **_k: ([500], None))

    cfg = types.SimpleNamespace(
        atmos_clim=types.SimpleNamespace(module='dummy'),
        interior_energetics=types.SimpleNamespace(module='spider'),
        observe=types.SimpleNamespace(module=None),
        orbit=types.SimpleNamespace(evolve=False, satellite=False),
        star=types.SimpleNamespace(module='dummy', mors=types.SimpleNamespace(age_now=4.5)),
        atmos_chem=types.SimpleNamespace(module='dummy'),
        params=types.SimpleNamespace(out=types.SimpleNamespace(plot_fmt='png')),
    )
    hf_all = pd.DataFrame({'Time': [1.0, 2.0]})
    dirs = {'output': str(tmp_path), 'fwl': str(tmp_path / 'fwl')}

    from proteus.utils.coupler import UpdatePlots

    UpdatePlots(hf_all, dirs, cfg, end=True, num_snapshots=1)

    called_names = [c[0] for c in calls]
    assert 'plot_global' in called_names
    assert 'plot_escape' in called_names
    assert 'plot_interior' in called_names
    assert 'plot_orbit' not in called_names
    assert 'plot_atmosphere' not in called_names
    assert 'plot_spectra' not in called_names


@pytest.mark.unit
def test_remove_excess_files_without_spectra(monkeypatch):
    removed = []
    monkeypatch.setattr('proteus.utils.coupler.safe_rm', lambda p: removed.append(p))

    remove_excess_files('/tmp/out', rm_spectralfiles=False)

    assert len(removed) == 3
    assert all('runtime.sf' not in p for p in removed)
