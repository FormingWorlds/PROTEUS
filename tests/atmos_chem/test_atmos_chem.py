"""
Unit tests for proteus.atmos_chem.wrapper module

This module orchestrates atmospheric chemistry post-processing:
- run_chemistry(): runs the backend at the snapshots set by atmos_chem.times
- Result DataFrame generation and postprocessing

Physics tested:
- Module selection and dispatch to VULCAN (vulcan/none)
- Error handling for invalid modules
- Result file reading and DataFrame creation
- The config reaches the VULCAN backend unchanged, with one call per snapshot

Related documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Mocking strategy:
- Mock VULCAN offline runs to avoid heavy computation
- Mock file I/O (read_result) to isolate wrapper logic
- Use MagicMock for config objects
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock vulcan modules before importing wrapper to prevent import errors
# The vulcan.py module tries to import the VULCAN package which isn't available in CI
# We need to keep the mock in sys.modules throughout the test execution
mock_vulcan_package = MagicMock()
mock_vulcan_module = MagicMock()
mock_run_vulcan = MagicMock()
mock_vulcan_module.run_vulcan = mock_run_vulcan

# Put mocks in sys.modules before any imports (persist throughout test execution)
sys.modules['vulcan'] = mock_vulcan_package
sys.modules['proteus.atmos_chem.vulcan'] = mock_vulcan_module

# Imports must come after mocks are set up
from proteus.atmos_chem.common import chem_csv_name, read_result  # noqa: E402
from proteus.atmos_chem.wrapper import run_chemistry  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_read_result_disabled_module():
    """
    Test read_result returns None when module is None or 'none'.

    Physics: Disabled chemistry module should not produce output files.
    Function gracefully returns None instead of crashing.
    """
    outdir = '/tmp/test/output'

    # Test module=None
    result_none = read_result(outdir, None)
    assert result_none is None

    # Test module='none'
    result_disabled = read_result(outdir, 'none')
    assert result_disabled is None


@pytest.mark.unit
def test_read_result_file_not_found(tmp_path):
    """
    Test read_result returns None when output file doesn't exist.

    Physics: First PROTEUS run might not have chemistry output if module
    was disabled or offline calculation hasn't completed yet.
    """
    outdir = str(tmp_path / 'output')  # Non-existent directory
    module = 'vulcan'

    result = read_result(outdir, module)
    assert result is None  # missing-file branch must yield None silently
    # Discriminating check: the directory genuinely does not exist on disk, so
    # only the missing-file branch can have produced the None return.
    import os

    assert not os.path.exists(outdir)


@pytest.mark.unit
def test_read_result_successful(tmp_path):
    """
    Test read_result successfully reads CSV chemistry output.

    Physics: Completed chemistry calculation produces CSV with species
    abundances, VMRs, and other chemical composition data.
    """
    # Create fake chemistry output file
    outdir = str(tmp_path / 'output')
    offchem_dir = tmp_path / 'output' / 'offchem'
    offchem_dir.mkdir(parents=True)

    # Write realistic VULCAN CSV (tab-delimited, columns = physical quantities + species)
    csv_file = offchem_dir / 'vulcan.csv'
    csv_content = (
        'tmp\tp\tz\tKzz\tH2\tH2O\tCO2\tCO\tCH4\tN2\n'
        '1.850e+02\t1.000e+00\t3.501e+05\t6.197e+11\t2.711e-01\t8.895e-03\t1.679e-02\t6.966e-01\t6.604e-03\t0.000e+00\n'
        '1.866e+02\t1.334e+00\t3.483e+05\t5.522e+11\t2.711e-01\t8.895e-03\t1.679e-02\t6.966e-01\t6.604e-03\t0.000e+00\n'
        '1.882e+02\t1.779e+00\t3.465e+05\t4.920e+11\t2.711e-01\t8.895e-03\t1.679e-02\t6.966e-01\t6.604e-03\t0.000e+00\n'
    )
    csv_file.write_text(csv_content)

    # Read result
    result = read_result(outdir, 'vulcan')

    # Verify DataFrame structure matches real VULCAN output
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # 3 atmospheric levels
    assert 'tmp' in result.columns
    assert 'H2O' in result.columns
    assert 'CO2' in result.columns


@pytest.mark.unit
def test_read_result_custom_filename(tmp_path):
    """
    Test read_result with explicit filename override (per-snapshot files).

    Physics: Chemistry writes one output file per snapshot, like vulcan_5000p000.csv.
    The filename parameter allows reading these instead of the default vulcan.csv.
    """
    outdir = str(tmp_path / 'output')
    offchem_dir = tmp_path / 'output' / 'offchem'
    offchem_dir.mkdir(parents=True)

    # Write per-snapshot file (sub-year naming convention)
    csv_file = offchem_dir / 'vulcan_5000p000.csv'
    csv_content = (
        'tmp\tp\tz\tH2O\tCO2\n300.0\t1.0\t5e4\t1e-3\t0.96\n400.0\t10.0\t3e4\t1e-3\t0.96\n'
    )
    csv_file.write_text(csv_content)

    # Default filename (vulcan.csv) should NOT find the per-snapshot file
    result_default = read_result(outdir, 'vulcan')
    assert result_default is None

    # Explicit filename should find the per-snapshot file
    result_custom = read_result(outdir, 'vulcan', filename='vulcan_5000p000.csv')
    assert isinstance(result_custom, pd.DataFrame)
    assert len(result_custom) == 2
    assert 'CO2' in result_custom.columns


@pytest.mark.unit
def test_read_result_preserves_whitespace_format(tmp_path):
    """
    Test read_result correctly parses whitespace-delimited CSV.

    Physics: Chemistry output uses flexible whitespace delimiters to handle
    various data formats from different offline solvers.
    """
    outdir = str(tmp_path / 'output')
    offchem_dir = tmp_path / 'output' / 'offchem'
    offchem_dir.mkdir(parents=True)

    # Write VULCAN CSV with variable whitespace (tabs + trailing whitespace)
    csv_file = offchem_dir / 'vulcan.csv'
    csv_content = (
        'tmp\tp\tz\tKzz\tH2\tH2O\tCO\tCO2\n'
        '1.850e+02\t1.000e+00\t3.501e+05\t6.197e+11\t2.711e-01\t8.895e-03\t6.966e-01\t1.679e-02\n'
        '2.180e+02\t2.382e+02\t3.129e+05\t6.901e+10\t2.711e-01\t8.895e-03\t6.973e-01\t1.679e-02\n'
        '5.391e+02\t1.010e+05\t2.522e+05\t1.663e+10\t2.711e-01\t8.895e-03\t6.855e-01\t1.679e-02\n'
    )
    csv_file.write_text(csv_content)

    result = read_result(outdir, 'vulcan')

    # Verify parsing (whitespace regex splits correctly)
    assert len(result) == 3
    assert 'tmp' in result.columns
    assert 'CO' in result.columns
    # Check that pressure column has correct first value
    assert result['p'].iloc[0] == pytest.approx(1.0, rel=1e-3)


def _vulcan_config(times='end') -> MagicMock:
    """Minimal mock config selecting the VULCAN backend."""
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.times = times
    return config


def _dirs(tmp_path) -> dict:
    """Output directories as PROTEUS sets them up."""
    return {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}


def _write_vulcan_csv(dirs, hf_row):
    """Write the CSV run_vulcan would write for this row: a 3-level column
    with the surface temperature of the row at the bottom (last row)."""
    fpath = os.path.join(dirs['output/offchem'], chem_csv_name('vulcan', hf_row['Time']))
    pd.DataFrame(
        {
            'tmp': [185.0, 400.0, hf_row['T_surf']],
            'p': [1.0, 1.0e3, 1.0e7],  # Pa, increasing downwards
            'H2O': [8.9e-3, 8.9e-3, 8.9e-3],
            'CO2': [0.9911, 0.9911, 0.9911],
        }
    ).to_csv(fpath, sep='\t', index=False)
    return True


@pytest.mark.unit
def test_run_chemistry_disabled_module(tmp_path):
    """
    Test run_chemistry when no module is specified (disabled mode).

    Physics: When atmos_chem.module=None, no chemistry calculations occur.
    Function logs warning and returns None without creating output.
    """
    config = _vulcan_config()
    config.atmos_chem.module = None
    hf_all = pd.DataFrame({'Time': [0.0, 100.0], 'T_surf': [3000.0, 2000.0]})

    result = run_chemistry(_dirs(tmp_path), config, hf_all)

    assert result is None
    assert not (tmp_path / 'offchem').exists()


@pytest.mark.unit
def test_run_chemistry_invalid_module(tmp_path):
    """
    Test run_chemistry raises error for invalid module name.

    Physics: Invalid module should not proceed - require explicit
    module selection (vulcan, dummy or None).
    """
    config = _vulcan_config()
    config.atmos_chem.module = 'invalid_module'
    hf_all = pd.DataFrame({'Time': [0.0], 'T_surf': [3000.0]})

    # The offending module value must appear in the raised message
    with pytest.raises(ValueError, match='invalid_module'):
        run_chemistry(_dirs(tmp_path), config, hf_all)
    assert not (tmp_path / 'offchem').exists()


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('proteus.atmos_chem.vulcan.run_vulcan')
def test_run_chemistry_vulcan_end_returns_final_state(patched_run_vulcan, tmp_path):
    """
    With module='vulcan' and times='end', run_vulcan runs once, on the final
    helpfile row, with the config passed through unchanged and a new, empty
    network-compilation state.

    Physics: the returned profile is the final state (surface 1500 K, not
    the 3000 K of the first row), with positive temperatures and pressures
    increasing downwards.
    """
    patched_run_vulcan.side_effect = lambda dirs, config, hf_row, **_: _write_vulcan_csv(
        dirs, hf_row
    )
    config = _vulcan_config(times='end')
    dirs = _dirs(tmp_path)
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 50.0, 1000.0],
            'T_surf': [3000.0, 2200.0, 1500.0],
            'P_surf': [300.0, 200.0, 100.0],  # bar
        }
    )

    result = run_chemistry(dirs, config, hf_all)

    patched_run_vulcan.assert_called_once()
    args, kwargs = patched_run_vulcan.call_args
    assert args[0] is dirs and args[1] is config
    assert args[2]['Time'] == pytest.approx(1000.0) and kwargs == {'network_state': {}}
    assert result['tmp'].iloc[-1] == pytest.approx(1500.0)
    assert np.all(result['tmp'] > 0.0) and np.all(np.diff(result['p']) > 0.0)


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.run_vulcan')
def test_run_chemistry_vulcan_all_writes_one_csv_per_snapshot(patched_run_vulcan, tmp_path):
    """
    With times='all' and no atmosphere snapshots on disk, run_vulcan runs at
    each distinct helpfile time; each run leaves its own sub-year named CSV
    and 'vulcan.csv' holds the latest one.
    """
    patched_run_vulcan.side_effect = lambda dirs, config, hf_row, **_: _write_vulcan_csv(
        dirs, hf_row
    )
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 0.0, 12.32],
            'T_surf': [3100.0, 3000.0, 2500.0],
            'P_surf': [300.0, 300.0, 100.0],  # bar
        }
    )

    result = run_chemistry(_dirs(tmp_path), _vulcan_config(times='all'), hf_all)

    assert patched_run_vulcan.call_count == 2
    # The repeated init rows at t = 0 give one snapshot, from the last of them
    assert patched_run_vulcan.call_args_list[0].args[2]['T_surf'] == pytest.approx(3000.0)
    names = sorted(f.name for f in (tmp_path / 'offchem').iterdir())
    assert names == ['vulcan.csv', 'vulcan_0p000.csv', 'vulcan_12p320.csv']
    assert result['tmp'].iloc[-1] == pytest.approx(2500.0)


@pytest.mark.unit
@patch('proteus.atmos_chem.wrapper.read_result')
@patch('proteus.atmos_chem.vulcan.run_vulcan')
def test_run_chemistry_raises_when_backend_fails(
    patched_run_vulcan, patched_read_result, tmp_path
):
    """run_chemistry stops with RuntimeError when run_vulcan returns False.

    A False return from the backend signals "no output NetCDF / unrecognised
    network / wrong atmos module". The wrapper raises rather than reading a
    file that was never written, and writes no 'vulcan.csv' copy.
    """
    patched_run_vulcan.return_value = False
    hf_all = pd.DataFrame({'Time': [1000.0], 'T_surf': [1500.0], 'P_surf': [100.0]})

    with pytest.raises(RuntimeError, match='vulcan chemistry failed'):
        run_chemistry(_dirs(tmp_path), _vulcan_config(), hf_all)

    patched_run_vulcan.assert_called_once()
    patched_read_result.assert_not_called()
    assert not (tmp_path / 'offchem' / 'vulcan.csv').exists()
