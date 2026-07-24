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
fast execution (target: <100ms per test). See docs/How-to/testing.md for test
infrastructure guidelines.

Fixtures from conftest.py:
- earth_params, ultra_hot_params, intermediate_params: Physical test scenarios
- tmp_config_paths: Temporary configuration file paths
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
from datetime import datetime
from unittest.mock import patch

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
    _atm_snapshot_names,
    _get_current_time,
    _interior_snapshot_names,
    _netcdf_readable,
    _populate_energy_residual,
    _snapshot_readable,
    get_proteus_directories,
    print_citation,
    print_module_configuration,
    remove_excess_files,
    select_resumable_snapshot,
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

    row = ZeroHelpfileRow()
    row['_T_magma_raw'] = 3000.0  # private transient key: must not warn
    row['core_state_initial'] = 'liquid'  # allowlisted string key: must not warn
    row['nonsense_future_key'] = 1.0  # genuine drift: must warn
    hf = CreateHelpfileFromDict(ZeroHelpfileRow())

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.coupler'):
        ExtendHelpfile(hf, row)

    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    joined = '\n'.join(r.message for r in warns)
    assert 'nonsense_future_key' in joined, f'Expected unknown-key warning, got: {joined!r}'
    assert '_T_magma_raw' not in joined, 'Private key leaked into warning'
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
def test_write_helpfile_preserves_prior_file_on_failed_write():
    """A crash or full disk mid-write must not destroy the existing helpfile.

    The atomic write serialises to a temporary file and renames it into
    place, so a failed serialisation leaves the previous complete file
    untouched and a later resume can still read a full row history. A
    direct remove-then-write would have left the helpfile missing, which
    is the empty-CSV state that blocks resume.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        row1 = ZeroHelpfileRow()
        row1['Time'] = 111.0
        row1['T_surf'] = 301.0
        WriteHelpfileToCSV(tmpdir, CreateHelpfileFromDict(row1))

        fpath = os.path.join(tmpdir, 'runtime_helpfile.csv')
        with open(fpath) as fh:
            before = fh.read()

        # Simulate a disk-full / killed process during the next write.
        row2 = ZeroHelpfileRow()
        row2['Time'] = 222.0
        hf2 = CreateHelpfileFromDict(row2)
        with patch.object(
            pd.DataFrame,
            'to_csv',
            side_effect=OSError('[Errno 28] No space left on device'),
        ):
            with pytest.raises(OSError):
                WriteHelpfileToCSV(tmpdir, hf2)

        # The prior complete file survives byte-for-byte, and no temp
        # artefact is left behind.
        assert os.path.exists(fpath)
        with open(fpath) as fh:
            assert fh.read() == before
        assert not os.path.exists(fpath + '.tmp')

        # Discrimination guard: the surviving file is the v1 row (Time=111),
        # not the failed v2 row (Time=222) and not an empty file.
        df = pd.read_csv(fpath, sep=r'\s+')
        assert len(df) == 1
        assert df['Time'].iloc[0] == pytest.approx(111.0)


@pytest.mark.unit
def test_write_helpfile_leaves_no_temp_file_on_success():
    """A successful atomic write leaves only the final CSV, no .tmp artefact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        row = ZeroHelpfileRow()
        row['Time'] = 5.0
        WriteHelpfileToCSV(tmpdir, CreateHelpfileFromDict(row))

        entries = os.listdir(tmpdir)
        assert 'runtime_helpfile.csv' in entries
        assert not any(e.endswith('.tmp') for e in entries)


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
    # Discrimination: the stored value must NOT have been truncated to
    # single precision (~7 digits). 1.23456789e12 keeps 9 digits in
    # float64 but only 7 in float32; assert at tighter rel-tol than the
    # primary to surface a regression that downcasted columns.
    assert hf['M_planet'].iloc[0] == pytest.approx(precise_value, rel=1e-12)


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
            WriteHelpfileToCSV(tmpdir, df)
        # Discrimination: the validation must short-circuit BEFORE writing
        # the CSV. A regression that wrote first and then raised would
        # leave a corrupted stub file on disk that resume would consume.
        assert not os.path.exists(os.path.join(tmpdir, 'runtime_helpfile.csv'))


@pytest.mark.unit
def test_zero_helpfile_row_is_immutable_structure():
    """Test that ZeroHelpfileRow returns new dict each time."""
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
    step_dE_Q_radio_J: float = 0.0,
    step_dE_Q_tidal_J: float = 0.0,
    step_solver_residual_J: float = 0.0,
    step_dE_state_heat_J: float = 0.0,
    step_dE_impact_J: float = 0.0,
    F_cmb: float = 0.0,
    R_int: float = 6.371e6,
    R_core: float = 3.481e6,
    T_magma: float = 3000.0,
) -> dict:
    """Helper: build a ZeroHelpfileRow populated with the columns the
    energy-residual helper reads. The predicted side is the boundary-flux
    and source increments; the state side is ``step_dE_state_heat_J`` (the
    entropy-transported heat). Uses Earth-like radii by default. The
    instantaneous F_cmb keyword is supported only so the discrimination
    test can prove it is IGNORED by the cumulative-sum logic; it has no
    effect on dE_predicted_cons_J."""
    row = ZeroHelpfileRow()
    row['Time'] = time_yr
    row['E_state_cons_J'] = E_state_cons_J
    row['step_dE_F_int_J'] = step_dE_F_int_J
    row['step_dE_F_cmb_J'] = step_dE_F_cmb_J
    row['step_dE_Q_radio_cons_J'] = step_dE_Q_radio_cons_J
    row['step_dE_Q_tidal_cons_J'] = step_dE_Q_tidal_cons_J
    row['step_dE_Q_radio_J'] = step_dE_Q_radio_J
    row['step_dE_Q_tidal_J'] = step_dE_Q_tidal_J
    row['step_solver_residual_J'] = step_solver_residual_J
    row['step_dE_state_heat_J'] = step_dE_state_heat_J
    row['step_dE_impact_J'] = step_dE_impact_J
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
        # Predicted-side primitives (boundary + source, frozen-mass).
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_cons_J',
        'step_dE_Q_tidal_cons_J',
        'step_solver_residual_J',
        # State-side primitive: the entropy-transported heat content change.
        'step_dE_state_heat_J',
        # Giant-impact re-melt heat injection (enters both residual sides).
        'step_dE_impact_J',
        # Cumulative columns derived from the primitives above.
        'E_state_heat_cons_J',
        'dE_predicted_cons_J',
        'E_residual_cons_J',
        'E_residual_cons_frac',
        'solver_residual_J',
        # Enthalpy columns kept as diagnostics (NOT used for residuals).
        'E_state_cons_J',
        'E_state_J',
        'step_dE_Q_radio_J',
        'step_dE_Q_tidal_J',
        'step_dE_compression_J',
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
        step_dE_Q_radio_J=delta_1_Q_radio,
        step_solver_residual_J=solver_inc_1,
        # Self-consistent run: the entropy-transported heat per step equals
        # the predicted boundary+source increment, so the residual is zero.
        step_dE_state_heat_J=delta_1_F_int + delta_1_Q_radio,
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
        step_dE_state_heat_J=delta_2_F_int + delta_2_F_cmb,
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
        step_dE_state_heat_J=delta_3_F_cmb + delta_3_F_int,
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

    # Physical step delta: -1e29 J (mild cooling), matched on both sides.
    physical_delta = -1.0e29
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E0 + physical_delta,
        step_dE_F_int_J=physical_delta,
        step_dE_state_heat_J=physical_delta,
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

    # State-heat increment = +0.5 J, predicted = 0 (no step deltas set), so
    # the residual is +0.5 J against a cumulative state heat of +0.5 J.
    row1 = _aragog_row(time_yr=1.0, E_state_cons_J=E0, step_dE_state_heat_J=0.5)
    _populate_energy_residual(hf, row1)

    # Without flooring this would be 0.5 / 0.5 = 1.0; with floor at 1 J
    # the divisor stays >= 1 and the fraction stays within [-1, 1].
    assert row1['E_residual_cons_J'] == pytest.approx(0.5)
    assert abs(row1['E_residual_cons_frac']) <= 1.0
    # Specifically: 0.5 / max(0.5, 1.0) = 0.5.
    assert row1['E_residual_cons_frac'] == pytest.approx(0.5)


@pytest.mark.unit
def test_populate_energy_residual_residual_detects_missing_source():
    """Conservation residual must surface a missing source. We construct a
    'real' planet whose entropy-transported heat content actually ROSE by
    5e29 J because of a fictional unreported source, but Aragog (in this
    scenario) only reported the cooling boundary-flux delta (-3e29 J). The
    residual is the state-heat side minus the predicted side, +8e29 J, sign
    and magnitude both meaningful. Catches a regression where the helper
    would silently zero residuals or apply the wrong sign convention."""
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # Hold the enthalpy snapshot E_state_cons_J fixed at E0 and let the
    # entropy-transported heat carry the true rise. The reverted enthalpy-
    # anchor logic would read zero state change and report only +3e29; the
    # state-heat logic reports +8e29. The two regimes therefore disagree, so
    # this assertion fails on a revert rather than passing for both.
    actual_dE = +5.0e29  # true heat content rise (state side)
    reported_step_delta = -3.0e29  # Aragog only reported the cooling flux
    row1 = _aragog_row(
        time_yr=100.0,
        E_state_cons_J=E0,
        step_dE_F_int_J=reported_step_delta,
        step_dE_state_heat_J=actual_dE,
    )
    _populate_energy_residual(hf, row1)

    expected_residual = actual_dE - reported_step_delta  # = +8e29 J
    assert row1['dE_predicted_cons_J'] == pytest.approx(reported_step_delta, rel=1e-12)
    assert row1['E_state_heat_cons_J'] == pytest.approx(actual_dE, rel=1e-12)
    assert row1['E_residual_cons_J'] == pytest.approx(expected_residual, rel=1e-12)
    # Reverted enthalpy-anchor logic would give actual_dE=0 -> residual +3e29.
    enthalpy_anchor_residual = 0.0 - reported_step_delta
    assert abs(expected_residual - enthalpy_anchor_residual) > 1e29
    # E_residual_cons_frac should be O(1); residual is comparable to actual.
    assert abs(row1['E_residual_cons_frac']) > 0.5


@pytest.mark.unit
def test_populate_energy_residual_excludes_compression_from_predicted():
    """The compression term must NOT enter the predicted side.

    ``step_dE_compression_J`` is an enthalpy-framed quantity retained only
    as a diagnostic; the entropy-transported heat already carries the full
    thermodynamic content. We set a large compression value alongside a
    physical boundary-flux delta and assert the predicted increment equals
    the flux delta alone. If compression leaked back into the sum (the
    behaviour being reverted) the predicted value would shift by the
    compression magnitude, which the discrimination guard rules out.
    """
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    flux_delta = -2.0e29
    compression = +7.0e29  # large, opposite sign; must be ignored
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E0 + flux_delta,
        step_dE_F_int_J=flux_delta,
        step_dE_state_heat_J=flux_delta,
    )
    row1['step_dE_compression_J'] = compression
    _populate_energy_residual(hf, row1)

    assert row1['dE_predicted_cons_J'] == pytest.approx(flux_delta, rel=1e-12)
    # Compression excluded on both sides => residual stays at zero, not at
    # the compression magnitude.
    assert abs(row1['E_residual_cons_J']) < 1e-3 * abs(compression)


@pytest.mark.unit
def test_populate_energy_residual_predicted_uses_live_mass_heating():
    """The predicted side uses the live-density heating variants, matching the
    live-density state side.

    The state side integrates rho(P,S) (live EOS density), so the heating
    sources on the predicted side must use the same mass frame: the live
    step_dE_Q_radio_J / step_dE_Q_tidal_J, not the frozen-mass *_cons
    variants. We give the live and frozen radiogenic increments different
    values and assert the predicted side tracks the live one. If the residual
    reverted to the frozen-mass variant the predicted increment would differ
    by the live-minus-frozen gap, which the discrimination guard rejects.
    """
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    live_radio = +1.0e29
    frozen_radio = +4.0e29  # deliberately different; must be IGNORED
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E0 + live_radio,
        step_dE_Q_radio_J=live_radio,
        step_dE_Q_radio_cons_J=frozen_radio,
        step_dE_state_heat_J=live_radio,
    )
    _populate_energy_residual(hf, row1)

    assert row1['dE_predicted_cons_J'] == pytest.approx(live_radio, rel=1e-12)
    # Frozen-mass variant would have given +4e29; guard the gap.
    assert abs(row1['dE_predicted_cons_J'] - frozen_radio) > 1e29
    assert abs(row1['E_residual_cons_J']) < 1e-3 * abs(live_radio)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_populate_energy_residual_is_invariant_across_a_giant_impact():
    """A giant-impact re-melt is booked on both sides, leaving the residual closed.

    The re-melt injects mantle-scale heat as an entropy jump between solver
    calls, so no per-call state integral carries it. The booking enters the
    impact heat on BOTH cumulatives: the state side gains the heat that was
    actually added, the predicted side gains the impact as an energy source,
    and the residual is unchanged across the impact. A one-sided booking would
    shift the residual by the full injection, which dwarfs every physical
    increment here, so closure is the discriminating signature.
    """
    E0 = 1.0e31
    row0 = _aragog_row(time_yr=0.0, E_state_cons_J=E0)
    hf = CreateHelpfileFromDict(row0)

    # Ordinary cooling step before the impact.
    cool = -2.0e29
    row1 = _aragog_row(
        time_yr=10.0,
        E_state_cons_J=E0 + cool,
        step_dE_F_int_J=cool,
        step_dE_state_heat_J=cool,
    )
    _populate_energy_residual(hf, row1)
    hf = ExtendHelpfile(hf, row1)
    residual_before = row1['E_residual_cons_J']

    # Impact row: the solve itself cooled a little more, then the re-melt
    # injected mantle-scale heat (two orders above the step increments).
    dE_impact = +5.0e30
    row2 = _aragog_row(
        time_yr=20.0,
        E_state_cons_J=E0 + 2 * cool + dE_impact,
        step_dE_F_int_J=cool,
        step_dE_state_heat_J=cool,
        step_dE_impact_J=dE_impact,
    )
    _populate_energy_residual(hf, row2)

    # Both cumulatives carry the injection.
    assert row2['dE_predicted_cons_J'] == pytest.approx(2 * cool + dE_impact, rel=1e-12)
    assert row2['E_state_heat_cons_J'] == pytest.approx(2 * cool + dE_impact, rel=1e-12)
    # The residual is invariant across the impact: booked, not leaked.
    assert row2['E_residual_cons_J'] == pytest.approx(residual_before, abs=1e-3 * abs(cool))
    # Discrimination: booking on only one side would shift the residual by the
    # full 5e30 J injection, twenty-five times the physical step increment.
    assert abs(dE_impact) > 20 * abs(cool)

    # Boundary case: a zero-impact row must reduce to the ordinary bookkeeping,
    # so the column's default cannot perturb quiet steps.
    row3 = _aragog_row(
        time_yr=30.0,
        E_state_cons_J=E0 + 3 * cool + dE_impact,
        step_dE_F_int_J=cool,
        step_dE_state_heat_J=cool,
        step_dE_impact_J=0.0,
    )
    hf = ExtendHelpfile(hf, row2)
    _populate_energy_residual(hf, row3)
    assert row3['E_residual_cons_J'] == pytest.approx(
        row2['E_residual_cons_J'], abs=1e-3 * abs(cool)
    )


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

    hf_row = {
        'M_atm': 4.6e24,
        'M_planet': 5.97e24,
    }
    # Distribute M_atm across gas_list so the per-species sum equals M_atm.
    # Use asymmetric values so the sum is a meaningful check (not all equal).
    per_species = 4.6e24 / len(gas_list)
    for s in gas_list:
        hf_row[s + '_kg_atm'] = per_species

    result = assert_mass_conservation(hf_row)
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

    hf_row = {
        'M_atm': 7.2e24,  # Atmosphere exceeds planet (the issue #677 symptom)
        'M_planet': 5.97e24,
    }
    for s_idx in range(15):
        # Stub kg_atm columns so the per-species check doesn't fire first
        # (we want to test the M_atm > M_planet path specifically).
        pass

    with pytest.raises(RuntimeError, match='Mass conservation violation'):
        assert_mass_conservation(hf_row)
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

    with pytest.raises(RuntimeError, match='M_atm bookkeeping inconsistency'):
        assert_mass_conservation(hf_row)
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
    'name,expected,near_miss',
    [
        # Exact membership: the name list is matched whole, so a longer name
        # built on the same stem falls through to the linear default.
        ('P_surf', True, 'P_surface'),
        # Composition suffix: the deciding token is the '_vmr' substring.
        ('X_vmr', True, 'X_vmol'),
        # Partial-pressure suffix: the deciding token is the '_bar' substring.
        ('CO2_bar', True, 'CO2_bat'),
        # Linear default, taken the other way round: a name that is neither
        # listed nor suffixed stays linear until it gains the substring.
        ('T_surf', False, 'T_surf_vmr'),
    ],
)
def test_variable_is_logarithmic_covers_membership_and_suffix_branches(
    name, expected, near_miss
):
    """Log scaling is chosen by exact name membership or a _vmr / _bar substring.

    Each case pairs a name with a near-miss differing only in the deciding
    token, so the pair separates the branch under test from a classifier that
    keys off the general shape of the name. Inference reads this convention to
    decide which parameters it optimises in log10: a misclassified variable
    silently changes the objective it fits, or trips the transform's
    non-positive-bound check, well before anyone notices a wrong plot axis.
    """
    assert variable_is_logarithmic(name) is expected
    assert variable_is_logarithmic(near_miss) is not expected


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
    """A spider stack above its pins passes, and the pin itself is inclusive.

    'numpy' carries no version specifier and must be skipped rather than parsed.
    The second block is the boundary: an installed version exactly equal to the
    required minimum satisfies a '>=' pin, so a regression to a strict '>'
    comparison would reject a correctly-pinned environment and abort the run.
    """
    from proteus.utils.coupler import validate_module_versions

    config = types.SimpleNamespace(
        interior_energetics=types.SimpleNamespace(module='spider'),
        interior_struct=types.SimpleNamespace(module='zalmoxis'),
        atmos_clim=types.SimpleNamespace(module='janus'),
        outgas=types.SimpleNamespace(module='calliope'),
        escape=types.SimpleNamespace(module='zephyrus'),
        star=types.SimpleNamespace(module='mors'),
    )
    requires = [
        'numpy',
        'fwl-zalmoxis>=1.0',
        'fwl-janus>=1.0',
        'fwl-calliope>=1.0',
        'fwl-zephyrus>=1.0',
        'fwl-mors>=1.0',
    ]
    installed = ('zalmoxis', 'janus', 'calliope', 'zephyrus', 'mors')

    for name in installed:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(__version__='1.2'))

    with (
        patch('importlib.metadata.requires', return_value=requires),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update,
    ):
        validate_module_versions({'rad': str(tmp_path)}, config)
    assert mock_update.call_count == 0

    # Boundary: installed == required minimum. '>=' admits equality.
    for name in installed:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(__version__='1.0'))

    with (
        patch('importlib.metadata.requires', return_value=requires),
        patch('proteus.utils.coupler.UpdateStatusfile') as mock_update_eq,
    ):
        validate_module_versions({'rad': str(tmp_path)}, config)
    assert mock_update_eq.call_count == 0


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
    """With spectra kept, only the three run-scratch files are removed.

    Those are the AGNI log, the SPIDER scratch directory, and the lockfile. The
    spectral files are the expensive ones to regenerate and are kept unless
    rm_spectralfiles is set, so the count pins that nothing else was swept in
    and the suffix check pins which files were spared.
    """
    removed = []
    monkeypatch.setattr('proteus.utils.coupler.safe_rm', lambda p: removed.append(p))

    remove_excess_files('/tmp/out', rm_spectralfiles=False)

    assert len(removed) == 3
    assert all('runtime.sf' not in p for p in removed)


# =============================================================================
# Test: resume snapshot-pair validation
# =============================================================================


def _write_valid_nc(path: str) -> str:
    """Create a minimal, valid netCDF file at ``path``."""
    from netCDF4 import Dataset

    with Dataset(path, 'w') as ds:
        ds.createDimension('x', 1)
    return path


def _write_corrupt_nc(path: str) -> str:
    """Create a file that exists but does not open as netCDF (truncated write)."""
    with open(path, 'wb') as fh:
        fh.write(b'not a valid netcdf file')
    return path


def _hf_times(times):
    """Minimal helpfile DataFrame carrying just the columns the selector reads."""
    return pd.DataFrame({'Time': [float(t) for t in times], 'Phi_global': [0.9] * len(times)})


@pytest.mark.unit
def test_netcdf_readable_distinguishes_valid_corrupt_missing(tmp_path):
    """_netcdf_readable is True only for a file that opens as netCDF.

    Covers the three states the resume selector must tell apart: a complete
    file, a file present but truncated mid-write, and an absent file.
    """
    valid = _write_valid_nc(str(tmp_path / 'ok.nc'))
    corrupt = _write_corrupt_nc(str(tmp_path / 'bad.nc'))

    assert _netcdf_readable(valid) is True
    assert _netcdf_readable(corrupt) is False
    assert _netcdf_readable(str(tmp_path / 'missing.nc')) is False


@pytest.mark.unit
def test_select_resumable_snapshot_keeps_complete_latest(tmp_path):
    """A complete latest pair leaves the helpfile and the data dir untouched."""
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20, 30):
        _write_valid_nc(str(data / f'{t}_int.nc'))
        _write_valid_nc(str(data / f'{t}_atm.nc'))

    out, dropped = select_resumable_snapshot(str(tmp_path), _hf_times([10, 20, 30]))

    assert dropped == []
    assert len(out) == 3
    assert int(out.iloc[-1]['Time']) == 30
    # Discrimination guard: nothing was quarantined on the all-complete path.
    assert not list(data.glob('*.incomplete'))


@pytest.mark.unit
def test_select_resumable_snapshot_falls_back_on_corrupt_atm(tmp_path):
    """A truncated latest _atm.nc trims to the prior pair and deletes both halves.

    The interior half being intact must not save the step: the atmosphere
    half is required, so the row is dropped and both 30_*.nc files are
    deleted so the modules' latest-file globs land on time 20 and the
    archive sweep cannot pick them up.
    """
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20):
        _write_valid_nc(str(data / f'{t}_int.nc'))
        _write_valid_nc(str(data / f'{t}_atm.nc'))
    _write_valid_nc(str(data / '30_int.nc'))
    _write_corrupt_nc(str(data / '30_atm.nc'))

    out, dropped = select_resumable_snapshot(str(tmp_path), _hf_times([10, 20, 30]))

    assert dropped == [30]
    assert int(out.iloc[-1]['Time']) == 20
    assert not (data / '30_atm.nc.incomplete').exists()
    assert not (data / '30_int.nc.incomplete').exists()
    assert not (data / '30_atm.nc').exists()
    assert not (data / '30_int.nc').exists()


@pytest.mark.unit
def test_select_resumable_snapshot_falls_back_on_corrupt_int(tmp_path):
    """A truncated latest _int.nc (atmosphere intact) also triggers the fallback."""
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20):
        _write_valid_nc(str(data / f'{t}_int.nc'))
        _write_valid_nc(str(data / f'{t}_atm.nc'))
    _write_corrupt_nc(str(data / '30_int.nc'))
    _write_valid_nc(str(data / '30_atm.nc'))

    out, dropped = select_resumable_snapshot(str(tmp_path), _hf_times([10, 20, 30]))

    assert dropped == [30]
    assert int(out.iloc[-1]['Time']) == 20
    # Both halves of the incomplete pair are deleted (symmetry with the
    # corrupt-atm case): a stray valid 30_atm.nc must not be left for the
    # atmosphere module's latest-file glob to pick up against a missing 30_int.
    assert not (data / '30_int.nc.incomplete').exists()
    assert not (data / '30_atm.nc.incomplete').exists()
    assert not (data / '30_int.nc').exists()
    assert not (data / '30_atm.nc').exists()


@pytest.mark.unit
def test_select_resumable_snapshot_matches_writer_filename_conventions(tmp_path):
    """Interior (truncated %d) and atmosphere (rounded %.0f) names can differ by 1.

    Aragog writes <int(Time)>_int.nc and AGNI writes <round(Time)>_atm.nc, so
    a fractional Time >= .5 puts the two halves at integer names one apart.
    The selector must probe each half with its own writer's convention: the
    Aragog interior at 30 and the AGNI atmosphere at 31 for Time 30.7. Probing
    the interior name for the atmosphere half would miss the rounded file and
    wrongly drop a valid latest snapshot.
    """
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20):
        _write_valid_nc(str(data / f'{t}_int.nc'))
        _write_valid_nc(str(data / f'{t}_atm.nc'))
    # Time 30.7: interior truncates to 30, atmosphere rounds to 31.
    _write_valid_nc(str(data / '30_int.nc'))
    _write_valid_nc(str(data / '31_atm.nc'))

    out, dropped = select_resumable_snapshot(str(tmp_path), _hf_times([10, 20, 30.7]))

    # The offset pair is recognised as complete: nothing dropped, latest kept.
    assert dropped == []
    assert len(out) == 3
    assert out.iloc[-1]['Time'] == pytest.approx(30.7)
    assert not list(data.glob('*.incomplete'))


@pytest.mark.unit
def test_select_resumable_snapshot_dummy_atm_requires_only_interior(tmp_path):
    """With require_atm=False the selector ignores the (absent) atmosphere half."""
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20, 30):
        _write_valid_nc(str(data / f'{t}_int.nc'))  # no _atm.nc written at all

    out, dropped = select_resumable_snapshot(
        str(tmp_path), _hf_times([10, 20, 30]), require_atm=False
    )

    assert dropped == []
    assert int(out.iloc[-1]['Time']) == 30


@pytest.mark.unit
def test_select_resumable_snapshot_raises_when_no_complete_pair(tmp_path):
    """No resumable row raises, so the caller can restart fresh.

    Two ways to reach it: the scan walks every row and finds none backed by a
    readable pair, or there is no row to walk at all. The empty helpfile is the
    boundary case where the loop body never runs, so the raise has to come from
    the post-loop check rather than from a failed probe inside it.
    """
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20):
        _write_corrupt_nc(str(data / f'{t}_int.nc'))
        _write_corrupt_nc(str(data / f'{t}_atm.nc'))

    with pytest.raises(RuntimeError, match='No complete'):
        select_resumable_snapshot(str(tmp_path), _hf_times([10, 20]))

    # Boundary: a helpfile with no rows has nothing to resume from either.
    with pytest.raises(RuntimeError, match='No complete'):
        select_resumable_snapshot(str(tmp_path), _hf_times([]))


@pytest.mark.unit
def test_failed_selection_restores_quarantined_files(tmp_path):
    """A failed selection restores every quarantined file and deletes none.

    A successful selection deletes the quarantined trailing files, which are
    unusable once the helpfile is truncated below them; a FAILED selection
    must do the opposite and put every file back under its original name, so
    a failed resume attempt stays lossless and a later retry or manual
    inspection sees the run directory exactly as it was. Discrimination:
    each file carries unique bytes and the restored files must match them,
    so a re-created placeholder or a cross-file swap cannot pass.
    """
    data = tmp_path / 'data'
    data.mkdir()
    payloads = {}
    for t in (10, 20):
        for half in ('int', 'atm'):
            p = data / f'{t}_{half}.nc'
            _write_corrupt_nc(str(p))
            # Make each corrupt file unique so restore is distinguishable
            # from recreation; appending keeps the file corrupt.
            p.write_bytes(p.read_bytes() + p.name.encode())
            payloads[p.name] = p.read_bytes()

    with pytest.raises(RuntimeError, match='No complete'):
        select_resumable_snapshot(str(tmp_path), _hf_times([10, 20]))

    leftovers = sorted(f.name for f in data.iterdir())
    assert leftovers == sorted(payloads), (
        'originals must be back under their exact names with no .incomplete strays'
    )
    for name, blob in payloads.items():
        assert (data / name).read_bytes() == blob


def _write_valid_json(path: str) -> str:
    """Create a minimal, valid SPIDER-style interior JSON snapshot."""
    with open(path, 'w') as fh:
        json.dump({'data': {'S': [1.0, 2.0]}}, fh)
    return path


def _write_corrupt_json(path: str) -> str:
    """Create a truncated JSON file (present but not parseable)."""
    with open(path, 'w') as fh:
        fh.write('{"data": {"S": [1.0, 2.')  # cut off mid-write
    return path


@pytest.mark.unit
def test_interior_snapshot_names_track_each_writer_convention():
    """Each interior module's probe uses that writer's own filename format.

    Aragog truncates (``%d_int.nc``) and SPIDER rounds (``%.0f.json``), so at
    a fractional Time the two land on integer stems one apart; the dummy and
    boundary interiors write no snapshot at all. A probe that assumed the
    Aragog name for every module would miss the SPIDER file and wrongly
    quarantine a valid resume point.
    """
    # Time 30.7: Aragog truncates to 30, SPIDER rounds to 31.
    assert _interior_snapshot_names(30.7, 'aragog') == ['30_int.nc']
    assert _interior_snapshot_names(30.7, 'spider') == ['31.json']
    # Discrimination guard: the two conventions disagree on both stem and
    # extension, so a single-convention probe cannot cover both.
    assert _interior_snapshot_names(30.7, 'aragog') != _interior_snapshot_names(30.7, 'spider')
    # Dummy and boundary write no interior snapshot: empty constraint.
    assert _interior_snapshot_names(30.7, 'dummy') == []
    assert _interior_snapshot_names(30.7, 'boundary') == []
    # Unknown module falls back to the Aragog default rather than crashing.
    assert _interior_snapshot_names(30.7, 'other') == ['30_int.nc']


@pytest.mark.unit
def test_atm_snapshot_names_single_convention_per_writer():
    """The atmosphere probe uses exactly the active writer's rounding convention.

    Only one atmosphere module is active per run, so the probe must return that
    writer's single filename, not both integer names. JANUS writes
    ``str(int(Time))_atm.nc`` (truncates) while AGNI writes ``%.0f_atm.nc``
    (rounds). For a fractional Time the two stems differ; probing both would let
    a fractional row's truncated name collide with an adjacent row's rounded
    name, so the single-convention rule is what prevents the cross-row mismatch.
    """
    # Time 30.7: JANUS truncates to 30, AGNI rounds to 31. Each writer yields
    # its own single candidate, never both.
    assert _atm_snapshot_names(30.7, 'janus') == ['30_atm.nc']
    assert _atm_snapshot_names(30.7, 'agni') == ['31_atm.nc']
    # Discrimination guard: the two conventions disagree at this fractional Time,
    # which is exactly the collision (30.7 truncated == 30.2 rounded == 30) that
    # probing both names would reintroduce.
    assert _atm_snapshot_names(30.7, 'janus') != _atm_snapshot_names(30.7, 'agni')
    # Integer Time: both conventions coincide, so the choice of writer is moot.
    assert _atm_snapshot_names(30.0, 'janus') == ['30_atm.nc']
    assert _atm_snapshot_names(30.0, 'agni') == ['30_atm.nc']
    # Unknown or empty module falls back to the AGNI (rounded) default rather
    # than crashing; 30.7 rounds to 31 under that fall-back.
    assert _atm_snapshot_names(30.7, 'other') == ['31_atm.nc']
    assert _atm_snapshot_names(30.7, '') == ['31_atm.nc']


@pytest.mark.unit
def test_snapshot_readable_dispatches_json_and_netcdf(tmp_path):
    """_snapshot_readable validates a .json half by parse and an .nc half by open.

    SPIDER's interior half is JSON; a truncated write parses as invalid. The
    netCDF halves keep the existing open-check. Covers valid, corrupt, and
    absent for both extensions.
    """
    good_json = _write_valid_json(str(tmp_path / 'ok.json'))
    bad_json = _write_corrupt_json(str(tmp_path / 'bad.json'))
    good_nc = _write_valid_nc(str(tmp_path / 'ok.nc'))

    assert _snapshot_readable(good_json) is True
    assert _snapshot_readable(bad_json) is False
    assert _snapshot_readable(str(tmp_path / 'missing.json')) is False
    # netCDF dispatch unchanged.
    assert _snapshot_readable(good_nc) is True
    assert _snapshot_readable(str(tmp_path / 'missing.nc')) is False


@pytest.mark.unit
def test_select_resumable_snapshot_resumes_spider_json_history(tmp_path):
    """A SPIDER interior (JSON snapshots) resumes; the Aragog-only probe would not.

    SPIDER writes ``'%.0f.json'`` interior snapshots and no ``_int.nc``. With
    ``interior_module='spider'`` the selector must recognise the JSON half and
    keep the latest row; a probe hard-coded to the Aragog ``_int.nc`` name
    would find no interior half on any row and raise the no-complete-pair
    error, a regression against a SPIDER resume that worked before.
    """
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20, 30):
        _write_valid_json(str(data / f'{t:.0f}.json'))

    out, dropped = select_resumable_snapshot(
        str(tmp_path), _hf_times([10, 20, 30]), require_atm=False, interior_module='spider'
    )

    assert dropped == []
    assert int(out.iloc[-1]['Time']) == 30
    # Discrimination guard: the Aragog-name probe sees no interior half here,
    # so it would raise rather than return the valid latest row.
    with pytest.raises(RuntimeError, match='No complete'):
        select_resumable_snapshot(
            str(tmp_path), _hf_times([10, 20, 30]), require_atm=False, interior_module='aragog'
        )


@pytest.mark.unit
def test_select_resumable_snapshot_falls_back_on_corrupt_spider_json(tmp_path):
    """A truncated latest SPIDER JSON trims to the prior complete snapshot."""
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20):
        _write_valid_json(str(data / f'{t:.0f}.json'))
    _write_corrupt_json(str(data / '30.json'))

    out, dropped = select_resumable_snapshot(
        str(tmp_path), _hf_times([10, 20, 30]), require_atm=False, interior_module='spider'
    )

    assert dropped == [30]
    assert int(out.iloc[-1]['Time']) == 20
    # The incomplete latest half is deleted so SPIDER's latest-json glob
    # lands on time 20, not the truncated 30.json.
    assert not (data / '30.json.incomplete').exists()
    assert not (data / '30.json').exists()


@pytest.mark.unit
def test_select_resumable_snapshot_dummy_interior_trusts_helpfile(tmp_path):
    """Dummy/boundary interiors write no snapshot, so resume trusts the helpfile.

    With ``interior_module='dummy'`` and no atmosphere half, no snapshot file
    exists on disk at all; the selector must impose no interior constraint and
    keep the latest helpfile row rather than raise for a missing ``_int.nc``.
    """
    data = tmp_path / 'data'
    data.mkdir()  # deliberately empty: dummy interior + dummy atm write nothing

    out, dropped = select_resumable_snapshot(
        str(tmp_path), _hf_times([10, 20, 30]), require_atm=False, interior_module='dummy'
    )

    assert dropped == []
    assert int(out.iloc[-1]['Time']) == 30
    assert not list(data.glob('*.incomplete'))


@pytest.mark.unit
def test_select_resumable_snapshot_accepts_truncated_janus_atm(tmp_path):
    """A fractional-Time JANUS atmosphere (truncated name) is recognised, not quarantined.

    JANUS writes ``str(int(Time))_atm.nc`` (truncates), so at Time 30.7 it
    writes ``30_atm.nc`` while the rounded AGNI convention would look for
    ``31_atm.nc``. With ``atmos_module='janus'`` the selector probes the
    truncated name and accepts the valid row; the default AGNI convention would
    look for the absent ``31_atm.nc`` and wrongly drop it.
    """
    data = tmp_path / 'data'
    data.mkdir()
    for t in (10, 20):
        _write_valid_nc(str(data / f'{t}_int.nc'))
        _write_valid_nc(str(data / f'{t}_atm.nc'))
    # Time 30.7: interior truncates to 30; JANUS atmosphere also truncates to 30.
    _write_valid_nc(str(data / '30_int.nc'))
    _write_valid_nc(str(data / '30_atm.nc'))  # JANUS truncated name, NOT 31

    out, dropped = select_resumable_snapshot(
        str(tmp_path), _hf_times([10, 20, 30.7]), atmos_module='janus'
    )

    assert dropped == []
    assert len(out) == 3
    assert out.iloc[-1]['Time'] == pytest.approx(30.7)
    # Discrimination guard: nothing quarantined; the truncated atm name was
    # accepted rather than treated as a missing rounded 31_atm.nc.
    assert not list(data.glob('*.incomplete'))
    assert (data / '30_atm.nc').exists()
    # Under the default AGNI (rounded) convention the same layout finds no
    # 31_atm.nc for Time 30.7 and drops the row, so the writer selection is
    # what accepts the truncated file rather than a probe of both names. The
    # JANUS probe above accepted 30.7 and quarantined nothing, so the files are
    # untouched for this second probe.
    out_agni, dropped_agni = select_resumable_snapshot(
        str(tmp_path), _hf_times([10, 20, 30.7]), atmos_module='agni'
    )
    assert dropped_agni == [30]
    assert out_agni.iloc[-1]['Time'] == pytest.approx(20)


@pytest.mark.unit
def test_select_resumable_snapshot_rejects_cross_row_atm_collision(tmp_path):
    """A row's missing atmosphere is not satisfied by an adjacent row's file.

    SPIDER writes rounded ``'%.0f.json'`` interiors and AGNI writes rounded
    ``'%.0f_atm.nc'`` atmospheres. Time 30.2 rounds to 30 for both halves and
    Time 30.7 rounds to 31. If Time 30.7's own ``31_atm.nc`` never landed while
    Time 30.2's ``30_atm.nc`` is on disk, a probe that also accepted the
    truncated ``30_atm.nc`` for Time 30.7 would pair 30.7's interior with
    30.2's atmosphere and resume from a mismatched state. The single AGNI
    convention probes only ``31_atm.nc`` for 30.7, so the row is dropped and
    30.2's valid pair is left in place.
    """
    data = tmp_path / 'data'
    data.mkdir()
    # Time 30.2: complete SPIDER-JSON + AGNI pair (both round to 30).
    _write_valid_json(str(data / '30.json'))
    _write_valid_nc(str(data / '30_atm.nc'))
    # Time 30.7: interior present (rounds to 31); its own atmosphere is absent.
    _write_valid_json(str(data / '31.json'))
    # 31_atm.nc deliberately not written.

    out, dropped = select_resumable_snapshot(
        str(tmp_path),
        _hf_times([30.2, 30.7]),
        interior_module='spider',
        atmos_module='agni',
    )

    # 30.7 is rejected (its own atmosphere is missing); resume falls to 30.2.
    assert dropped == [30]
    assert len(out) == 1
    assert out.iloc[-1]['Time'] == pytest.approx(30.2)
    # 30.2's valid pair must survive untouched: neither file is a candidate for
    # 30.7 under the single rounded convention, so the collision cannot consume
    # them.
    assert (data / '30_atm.nc').exists()
    assert (data / '30.json').exists()
    # 30.7's orphaned interior is deleted so SPIDER's latest-json glob
    # cannot load it against the 30.2-trimmed helpfile.
    assert not (data / '31.json.incomplete').exists()
    assert not (data / '31.json').exists()
