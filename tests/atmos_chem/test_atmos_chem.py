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

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from proteus.atmos_chem.common import read_result
from proteus.atmos_chem.wrapper import run_chemistry


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

    # Write fake CSV (whitespace-delimited as per source code)
    csv_file = offchem_dir / 'vulcan.csv'
    csv_content = """species vmr abundance
    H2O 0.75 1e22
    CO2 0.24 1e21
    N2 0.01 1e20"""
    csv_file.write_text(csv_content)

    # Read result
    result = read_result(outdir, 'vulcan')

    # Verify DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # 3 species
    assert list(result.columns) == ['species', 'vmr', 'abundance']
    assert result['species'].tolist() == ['H2O', 'CO2', 'N2']


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

    # Write CSV with variable whitespace (tabs, multiple spaces)
    csv_file = offchem_dir / 'vulcan.csv'
    csv_content = """species    vmr    vmr_uncertainty    source
    CO        1e-6    1e-7    oxidation
    CO2       0.24    0.01    outgassing
    H2O       0.75    0.05    outgassing"""
    csv_file.write_text(csv_content)

    result = read_result(outdir, 'vulcan')

    # Verify parsing (whitespace regex splits correctly)
    assert len(result) == 3
    assert 'vmr_uncertainty' in result.columns
    assert 'source' in result.columns
    # Check that species 'CO' has source 'oxidation' (row-based access)
    co_row = result[result['species'] == 'CO']
    assert co_row['source'].values[0] == 'oxidation'

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
    # Import vulcan module to ensure it's available for patching
    import proteus.atmos_chem.vulcan  # noqa: F401

    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'

    hf_row = {'Time': 1000.0, 'P_surf': 100.0}

    # Mock VULCAN function and result reading
    with patch('proteus.atmos_chem.vulcan.run_vulcan_offline') as mock_vulcan:
        with patch('proteus.atmos_chem.wrapper.read_result') as mock_read:
            # Setup mock result (typical chemistry output)
            mock_result = pd.DataFrame(
                {
                    'species': ['H2O', 'CO2', 'N2', 'O2'],
                    'vmr': [0.8, 0.15, 0.03, 0.02],
                    'abundance': [1e20, 1e19, 5e17, 3e17],
                }
            )
            mock_read.return_value = mock_result

            result = run_chemistry(dirs, config, hf_row)

            # Verify VULCAN was called
            mock_vulcan.assert_called_once_with(dirs, config, hf_row)

            # Verify result reading
            mock_read.assert_called_once_with(dirs['output'], 'vulcan')

            # Verify DataFrame is returned
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 4  # 4 species
            assert list(result['species']) == ['H2O', 'CO2', 'N2', 'O2']


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
    # Import vulcan module to ensure it's available for patching
    import proteus.atmos_chem.vulcan  # noqa: F401

    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'

    hf_row = {'Time': 5000.0}

    # Create realistic chemistry result
    chemistry_result = pd.DataFrame(
        {
            'species': ['H2', 'H2O', 'CO', 'CO2', 'CH4', 'NH3'],
            'vmr': [1e-8, 0.75, 1e-6, 0.24, 1e-10, 1e-12],
            'column_density': [1e15, 1e22, 1e16, 1e21, 1e12, 1e10],
        }
    )

    with patch('proteus.atmos_chem.vulcan.run_vulcan_offline'):
        with patch('proteus.atmos_chem.wrapper.read_result') as mock_read:
            mock_read.return_value = chemistry_result

            result = run_chemistry(dirs, config, hf_row)

            # Verify DataFrame structure
            assert isinstance(result, pd.DataFrame)
            assert 'species' in result.columns
            assert 'vmr' in result.columns
            assert result.shape[0] == 6  # 6 species


@pytest.mark.unit
def test_run_chemistry_vulcan_with_realistic_hf_row():
    """
    Test run_chemistry with realistic helpfile row from atmosphere calculation.

    Physics: Typical atmospheric state includes surface pressure, temperature,
    mass fractions, and composition from previous modules (outgassing, interior).
    """
    # Import vulcan module to ensure it's available for patching
    import proteus.atmos_chem.vulcan  # noqa: F401

    dirs = {'output': '/tmp/test', 'input': '/tmp/input'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'

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

    with patch('proteus.atmos_chem.vulcan.run_vulcan_offline') as mock_v:
        with patch('proteus.atmos_chem.wrapper.read_result') as mock_r:
            # Venus-like chemistry result
            venus_chemistry = pd.DataFrame(
                {
                    'species': ['CO2', 'H2SO4_aerosol', 'HCl', 'N2', 'Ar'],
                    'vmr': [0.965, 0.03, 0.004, 0.0009, 0.0001],
                }
            )
            mock_r.return_value = venus_chemistry

            result = run_chemistry(dirs, config, hf_row)

            # Verify function was called with correct arguments
            assert mock_v.called
            assert mock_r.called

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
    # Import vulcan module to ensure it's available for patching
    import proteus.atmos_chem.vulcan  # noqa: F401

    dirs = {'output': '/tmp/test'}
    config = MagicMock()
    config.atmos_chem.module = 'vulcan'
    config.atmos_chem.vulcan.dt = 1000.0  # Custom timestep
    config.atmos_chem.vulcan.num_iter = 100  # Custom iterations

    hf_row = {'Time': 1000.0}

    with patch('proteus.atmos_chem.vulcan.run_vulcan_offline') as mock_v:
        with patch('proteus.atmos_chem.wrapper.read_result'):
            run_chemistry(dirs, config, hf_row)

            # Verify config was passed exactly
            call_args = mock_v.call_args
            assert call_args[0][1] is config  # Second positional argument is config
