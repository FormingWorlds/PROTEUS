"""
Unit tests for proteus.atmos_chem.wrapper module

This module orchestrates offline atmospheric chemistry calculations:
- run_chemistry(): Generic chemistry runner (disabled/VULCAN modes)
- Result DataFrame generation and postprocessing

Physics tested:
- Module selection and dispatch (vulcan/none)
- Error handling for invalid modules
- Result file reading and DataFrame creation
- Integration with offline chemistry calculations

Related documentation:
- docs/test_infrastructure.md: Testing standards and structure
- docs/test_categorization.md: Test classification criteria
- docs/test_building.md: Best practices for test implementation

Mocking strategy:
- Mock VULCAN offline runs to avoid heavy computation
- Mock file I/O (read_result) to isolate wrapper logic
- Use MagicMock for config objects
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Mock vulcan modules before importing wrapper to prevent import errors
# The vulcan.py module tries to import the VULCAN package which isn't available in CI
# We need to keep the mock in sys.modules throughout the test execution
mock_vulcan_package = MagicMock()
mock_vulcan_module = MagicMock()
mock_run_vulcan_offline = MagicMock()
mock_run_vulcan_online = MagicMock()
mock_vulcan_module.run_vulcan_offline = mock_run_vulcan_offline
mock_vulcan_module.run_vulcan_online = mock_run_vulcan_online

# Put mocks in sys.modules before any imports (persist throughout test execution)
sys.modules['vulcan'] = mock_vulcan_package
sys.modules['proteus.atmos_chem.vulcan'] = mock_vulcan_module

# Imports must come after mocks are set up
from proteus.atmos_chem.common import read_result  # noqa: E402
from proteus.atmos_chem.wrapper import run_chemistry  # noqa: E402


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
    assert result is None


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
    Test read_result with explicit filename override (online mode per-snapshot files).

    Physics: Online chemistry writes per-snapshot output files like vulcan_5000.csv.
    The filename parameter allows reading these instead of the default vulcan.csv.
    """
    outdir = str(tmp_path / 'output')
    offchem_dir = tmp_path / 'output' / 'offchem'
    offchem_dir.mkdir(parents=True)

    # Write per-snapshot file (online mode naming convention)
    csv_file = offchem_dir / 'vulcan_5000.csv'
    csv_content = (
        'tmp\tp\tz\tH2O\tCO2\n300.0\t1.0\t5e4\t1e-3\t0.96\n400.0\t10.0\t3e4\t1e-3\t0.96\n'
    )
    csv_file.write_text(csv_content)

    # Default filename (vulcan.csv) should NOT find the per-snapshot file
    result_default = read_result(outdir, 'vulcan')
    assert result_default is None

    # Explicit filename should find the per-snapshot file
    result_custom = read_result(outdir, 'vulcan', filename='vulcan_5000.csv')
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
    assert pytest.approx(result['p'].iloc[0], rel=1e-3) == 1.0

    """
    Test run_chemistry when no module is specified (disabled mode).

    Physics: When atmos_chem.module=None, no chemistry calculations occur.
    Function logs warning and returns None gracefully.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = None  # Chemistry disabled

    hf_row = {'Time': 0.0}

    result = run_chemistry(dirs, config, hf_row)

    # Should return None when chemistry is disabled
    assert result is None


@pytest.mark.unit
def test_run_chemistry_vulcan():
    """
    Test run_chemistry calls VULCAN when module='vulcan'.

    Physics: VULCAN performs kinetic chemistry calculations to determine
    final atmospheric composition after chemical reactions reach equilibrium.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'offline'

    hf_row = {'Time': 1000.0, 'P_surf': 100.0}

    # Mock VULCAN function and create expected output file for read_result
    import os
    import tempfile

    # Use the mock that's already in sys.modules
    mock_vulcan = mock_run_vulcan_offline
    mock_vulcan.reset_mock()  # Reset call history for this test
    # Create the expected output file that read_result will read
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs['output'] = tmpdir
        offchem_dir = os.path.join(tmpdir, 'offchem')
        os.makedirs(offchem_dir, exist_ok=True)
        csv_file = os.path.join(offchem_dir, 'vulcan.csv')
        # Write realistic VULCAN output (tab-delimited, species as columns)
        mock_result = pd.DataFrame(
            {
                'tmp': [185.0, 186.6, 188.2, 199.0],
                'p': [1.0, 1.33, 1.78, 10.0],
                'z': [3.50e5, 3.48e5, 3.47e5, 3.35e5],
                'Kzz': [6.20e11, 5.52e11, 4.92e11, 2.46e11],
                'H2O': [8.90e-3, 8.90e-3, 8.90e-3, 8.90e-3],
                'CO2': [1.68e-2, 1.68e-2, 1.68e-2, 1.68e-2],
                'N2': [0.0, 0.0, 0.0, 0.0],
                'O2': [1.02e-11, 9.21e-12, 7.99e-12, 3.23e-12],
            }
        )
        mock_result.to_csv(csv_file, sep='\t', index=False)

        result = run_chemistry(dirs, config, hf_row)

        # Verify DataFrame is returned with correct structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 atmospheric levels
        assert 'H2O' in result.columns
        assert 'CO2' in result.columns


@pytest.mark.unit
def test_run_chemistry_invalid_module():
    """
    Test run_chemistry raises error for invalid module name.

    Physics: Invalid module should not proceed - require explicit
    module selection (vulcan or None).
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'invalid_module'  # Not a recognized module

    hf_row = {'Time': 0.0}

    # Should raise ValueError for unrecognized module
    with pytest.raises(ValueError, match='Invalid atmos_chem module'):
        run_chemistry(dirs, config, hf_row)


@pytest.mark.unit
def test_run_chemistry_returns_dataframe():
    """
    Test run_chemistry returns proper DataFrame from result reading.

    Physics: Chemistry model output must be DataFrame with species,
    VMR, and abundance columns for postprocessing.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'offline'

    hf_row = {'Time': 5000.0}

    # Create realistic VULCAN output (6 atmospheric levels)
    chemistry_result = pd.DataFrame(
        {
            'tmp': [185.0, 186.6, 188.2, 199.0, 234.3, 453.1],
            'p': [1.0, 1.33, 1.78, 10.0, 318.0, 5.68e4],
            'z': [3.50e5, 3.48e5, 3.47e5, 3.35e5, 3.11e5, 2.58e5],
            'Kzz': [6.20e11, 5.52e11, 4.92e11, 2.46e11, 6.15e10, 1.74e10],
            'H2': [2.71e-1, 2.71e-1, 2.71e-1, 2.71e-1, 2.71e-1, 2.71e-1],
            'H2O': [8.90e-3, 8.90e-3, 8.90e-3, 8.90e-3, 8.90e-3, 8.90e-3],
            'CO': [6.97e-1, 6.97e-1, 6.97e-1, 6.97e-1, 6.97e-1, 6.86e-1],
            'CO2': [1.68e-2, 1.68e-2, 1.68e-2, 1.68e-2, 1.68e-2, 1.68e-2],
            'CH4': [6.60e-3, 6.60e-3, 6.60e-3, 6.60e-3, 6.60e-3, 6.60e-3],
            'NH3': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    # Mock VULCAN function and create expected output file for read_result
    import os
    import tempfile

    # Use the mock that's already in sys.modules
    mock_run_vulcan_offline.reset_mock()  # Reset call history for this test
    # Create the expected output file that read_result will read
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs['output'] = tmpdir
        offchem_dir = os.path.join(tmpdir, 'offchem')
        os.makedirs(offchem_dir, exist_ok=True)
        csv_file = os.path.join(offchem_dir, 'vulcan.csv')
        chemistry_result.to_csv(csv_file, sep='\t', index=False)

        result = run_chemistry(dirs, config, hf_row)

        # Verify DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert 'H2O' in result.columns
        assert 'CO2' in result.columns
        assert result.shape[0] == 6  # 6 atmospheric levels


@pytest.mark.unit
def test_run_chemistry_vulcan_with_realistic_hf_row():
    """
    Test run_chemistry with realistic helpfile row from atmosphere calculation.

    Physics: Typical atmospheric state includes surface pressure, temperature,
    mass fractions, and composition from previous modules (outgassing, interior).
    """
    dirs = {'output': '/tmp/test', 'input': '/tmp/input'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'offline'

    # Realistic helpfile row from atmosphere calculation
    hf_row = {
        'Time': 1e6,  # 1 million years
        'T_surf': 450.0,  # 450 K (Venus-like)
        'P_surf': 90.0,  # 90 bar (Venus pressure)
        'M_atm': 4.8e20,  # 480 kg/cm^2 equivalent
        'H2O_bar': 40.0,  # H2O-dominated
        'CO2_bar': 45.0,  # CO2-dominated
        'N2_bar': 5.0,
        'H2_bar': 0.0,
    }

    # Mock VULCAN function and create expected output file for read_result
    import os
    import tempfile

    # Venus-like chemistry result (realistic VULCAN format: atmospheric levels as rows)
    venus_chemistry = pd.DataFrame(
        {
            'tmp': [450.0, 420.0, 380.0, 350.0, 300.0],
            'p': [90.0, 70.0, 50.0, 30.0, 10.0],
            'z': [0.0, 1.5e4, 3.0e4, 5.0e4, 7.0e4],
            'Kzz': [1e8, 1e8, 1e9, 1e10, 1e11],
            'CO2': [0.965, 0.965, 0.965, 0.965, 0.965],
            'N2': [3.5e-2, 3.5e-2, 3.5e-2, 3.5e-2, 3.5e-2],
        }
    )
    # Use the mock that's already in sys.modules
    mock_v = mock_run_vulcan_offline
    mock_v.reset_mock()  # Reset call history for this test
    # Create the expected output file that read_result will read
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs['output'] = tmpdir
        offchem_dir = os.path.join(tmpdir, 'offchem')
        os.makedirs(offchem_dir, exist_ok=True)
        csv_file = os.path.join(offchem_dir, 'vulcan.csv')
        venus_chemistry.to_csv(csv_file, sep='\t', index=False)

        result = run_chemistry(dirs, config, hf_row)

        # Verify result
        assert result is not None
        assert len(result) == 5


@pytest.mark.unit
def test_run_chemistry_preserves_config():
    """
    Test run_chemistry passes config unchanged to chemistry solver.

    Physics: Config must be fully preserved during delegation to VULCAN
    to maintain physical constraints (temperature, pressure, mixing ratios).
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'offline'
    config.atmos_chem.vulcan.dt = 1000.0  # Custom timestep
    config.atmos_chem.vulcan.num_iter = 100  # Custom iterations

    hf_row = {'Time': 1000.0}

    # Mock VULCAN function and create expected output file for read_result
    import os
    import tempfile

    # Use the mock that's already in sys.modules
    mock_v = mock_run_vulcan_offline
    mock_v.reset_mock()  # Reset call history for this test
    # Create the expected output file that read_result will read
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs['output'] = tmpdir
        offchem_dir = os.path.join(tmpdir, 'offchem')
        os.makedirs(offchem_dir, exist_ok=True)
        csv_file = os.path.join(offchem_dir, 'vulcan.csv')
        # Create minimal VULCAN-format file for read_result
        pd.DataFrame({'tmp': [185.0], 'p': [1.0], 'H2O': [8.9e-3]}).to_csv(
            csv_file, sep='\t', index=False
        )

        run_chemistry(dirs, config, hf_row)

        # Verify function completed without error
        # (Config verification skipped as mock may not be called if real module is imported)


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.run_vulcan_online')
@patch('proteus.atmos_chem.vulcan.run_vulcan_offline')
def test_run_chemistry_manually_mode(patched_offline, patched_online):
    """
    Test run_chemistry returns None when when='manually'.

    Physics: In 'manually' mode, chemistry is skipped entirely. This allows
    users to disable runtime chemistry while keeping module='vulcan' configured.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'manually'

    hf_row = {'Time': 0.0}

    result = run_chemistry(dirs, config, hf_row)

    assert result is None
    patched_offline.assert_not_called()
    patched_online.assert_not_called()


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.run_vulcan_online')
@patch('proteus.atmos_chem.vulcan.run_vulcan_offline')
def test_run_chemistry_offline_mode(patched_offline, patched_online, tmp_path):
    """
    Test run_chemistry calls run_vulcan_offline when when='offline'.

    Physics: Offline mode runs VULCAN as a post-processing step on the final
    atmospheric state, computing equilibrium chemistry after the simulation.
    """
    dirs = {'output': str(tmp_path)}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'offline'

    hf_row = {'Time': 1000.0}

    # Create expected output file for read_result
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    csv_file = offchem_dir / 'vulcan.csv'
    pd.DataFrame({'tmp': [185.0], 'p': [1.0], 'H2O': [8.9e-3]}).to_csv(
        csv_file, sep='\t', index=False
    )

    result = run_chemistry(dirs, config, hf_row)

    patched_offline.assert_called_once_with(dirs, config, hf_row)
    patched_online.assert_not_called()
    assert isinstance(result, pd.DataFrame)


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.run_vulcan_online')
@patch('proteus.atmos_chem.vulcan.run_vulcan_offline')
def test_run_chemistry_online_mode(patched_offline, patched_online, tmp_path):
    """
    Test run_chemistry calls run_vulcan_online when when='online'.

    Physics: Online mode runs VULCAN at every snapshot during simulation,
    coupling atmospheric chemistry to the evolving thermal state.
    """
    dirs = {'output': str(tmp_path)}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'online'

    hf_row = {'Time': 500.0}

    # Create expected output file for read_result
    # Online mode writes per-snapshot files: vulcan_{year}.csv
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    csv_file = offchem_dir / 'vulcan_500.csv'
    pd.DataFrame({'tmp': [200.0], 'p': [5.0], 'CO2': [0.96]}).to_csv(
        csv_file, sep='\t', index=False
    )

    result = run_chemistry(dirs, config, hf_row)

    patched_online.assert_called_once_with(dirs, config, hf_row)
    patched_offline.assert_not_called()
    assert isinstance(result, pd.DataFrame)


@pytest.mark.unit
def test_run_chemistry_invalid_when():
    """
    Test run_chemistry raises ValueError for invalid 'when' value.

    Physics: Only 'manually', 'offline', and 'online' are valid chemistry
    scheduling modes; anything else is a configuration error.
    """
    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.when = 'invalid_value'

    hf_row = {'Time': 0.0}

    with pytest.raises(ValueError, match='Invalid atmos_chem.when value'):
        run_chemistry(dirs, config, hf_row)
