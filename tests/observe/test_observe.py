"""
Unit tests for proteus.observe.common module

This module provides utilities for reading transit and eclipse spectra:
- get_transit_fpath(): Construct filepath to transit spectrum file
- get_eclipse_fpath(): Construct filepath to eclipse spectrum file
- read_transit(): Load transit spectrum data from CSV
- read_eclipse(): Load eclipse spectrum data from CSV

Physics tested:
- File path construction for observation output files
- CSV reading with whitespace-delimited format
- Error handling for missing spectrum files
- Data structure validation (column presence, data types)

Related documentation:
- docs/test_infrastructure.md: Testing standards and structure
- docs/test_categorization.md: Test classification criteria
- docs/test_building.md: Best practices for test implementation

Mocking strategy:
- Use tmp_path fixture for temporary file creation
- Mock file I/O with actual CSV creation in temp directories
"""

from __future__ import annotations

import pandas as pd
import pytest

from proteus.observe.common import (
    OBS_SOURCES,
    get_eclipse_fpath,
    get_transit_fpath,
    read_eclipse,
    read_transit,
)


@pytest.mark.unit
def test_get_transit_fpath_outgas():
    """
    Test get_transit_fpath constructs correct filepath for outgas source.

    Physics: Transit spectra from volatile outgassing calculations stored
    with 'outgas' label in filename for traceability.
    """
    outdir = '/tmp/output'
    source = 'outgas'
    stage = 'initial'

    fpath = get_transit_fpath(outdir, source, stage)

    assert fpath == '/tmp/output/observe/transit_outgas_initial.csv'
    assert 'transit' in fpath
    assert source in fpath
    assert stage in fpath


@pytest.mark.unit
def test_get_transit_fpath_profile():
    """
    Test get_transit_fpath constructs correct filepath for profile source.

    Physics: Transit spectra from atmospheric profiles (e.g., JANUS output)
    stored with 'profile' label in filename.
    """
    outdir = '/tmp/output'
    source = 'profile'
    stage = 'final'

    fpath = get_transit_fpath(outdir, source, stage)

    assert fpath == '/tmp/output/observe/transit_profile_final.csv'
    assert 'profile' in fpath


@pytest.mark.unit
def test_get_eclipse_fpath_offchem():
    """
    Test get_eclipse_fpath constructs correct filepath for offchem source.

    Physics: Eclipse spectra from offline chemistry calculations stored
    with 'offchem' label in filename.
    """
    outdir = '/tmp/output'
    source = 'offchem'
    stage = 'composition'

    fpath = get_eclipse_fpath(outdir, source, stage)

    assert fpath == '/tmp/output/observe/eclipse_offchem_composition.csv'
    assert 'eclipse' in fpath
    assert 'offchem' in fpath


@pytest.mark.unit
def test_read_transit_success(tmp_path):
    """
    Test read_transit successfully reads transit spectrum CSV.

    Physics: Transit spectra contain wavelength vs. transmission data
    used for atmospheric characterization and exoplanet detection.
    Earth-like example: transmission features from H2O and O2.
    """
    # Create fake transit spectrum file
    outdir = str(tmp_path / 'output')
    observe_dir = tmp_path / 'output' / 'observe'
    observe_dir.mkdir(parents=True)

    csv_file = observe_dir / 'transit_outgas_initial.csv'
    # Match real Platon output format: tab-delimited, columns are
    # Wavelength/um, None/ppm, then per-gas contributions in ppm.
    csv_content = 'Wavelength/um\tNone/ppm\tH2O/ppm\n'
    csv_content += '5.00000000e-01\t1.20000000e+02\t1.10000000e+02\n'
    csv_content += '1.00000000e+00\t1.50000000e+02\t1.40000000e+02\n'
    csv_content += '2.00000000e+00\t1.80000000e+02\t1.70000000e+02\n'
    csv_content += '5.00000000e+00\t2.00000000e+02\t1.90000000e+02\n'
    csv_file.write_text(csv_content)

    # Read transit spectrum
    result = read_transit(outdir, 'outgas', 'initial')

    # Verify DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4  # 4 wavelength points
    assert 'Wavelength/um' in result.columns
    assert 'None/ppm' in result.columns
    assert 'H2O/ppm' in result.columns
    assert pytest.approx(result['Wavelength/um'].iloc[0], rel=1e-5) == 0.5


@pytest.mark.unit
def test_read_eclipse_success(tmp_path):
    """
    Test read_eclipse successfully reads eclipse spectrum CSV.

    Physics: Eclipse spectra contain wavelength vs. brightness data
    capturing thermal emission and reflected light from exoplanet.
    Venus-like example: strong thermal continuum in infrared.
    """
    # Create fake eclipse spectrum file
    outdir = str(tmp_path / 'output')
    observe_dir = tmp_path / 'output' / 'observe'
    observe_dir.mkdir(parents=True)

    csv_file = observe_dir / 'eclipse_profile_final.csv'
    # Match real Platon output format: tab-delimited, columns are
    # Wavelength/um, None/ppm, then per-gas contributions in ppm.
    csv_content = 'Wavelength/um\tNone/ppm\tCO2/ppm\n'
    csv_content += '1.00000000e+00\t1.50000000e+01\t1.40000000e+01\n'
    csv_content += '2.00000000e+00\t2.50000000e+01\t2.30000000e+01\n'
    csv_content += '5.00000000e+00\t3.50000000e+01\t3.20000000e+01\n'
    csv_content += '1.00000000e+01\t3.00000000e+01\t2.80000000e+01\n'
    csv_file.write_text(csv_content)

    # Read eclipse spectrum
    result = read_eclipse(outdir, 'profile', 'final')

    # Verify DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert 'Wavelength/um' in result.columns
    assert 'None/ppm' in result.columns
    assert 'CO2/ppm' in result.columns


@pytest.mark.unit
def test_read_transit_file_not_found(tmp_path):
    """
    Test read_transit raises FileNotFoundError when file doesn't exist.

    Physics: If observation module is disabled or hasn't been run yet,
    transit spectrum files will not exist in output directory.
    """
    outdir = str(tmp_path / 'empty_output')  # Non-existent directory

    with pytest.raises(FileNotFoundError, match='Transit spectrum file'):
        read_transit(outdir, 'outgas', 'initial')


@pytest.mark.unit
def test_read_eclipse_file_not_found(tmp_path):
    """
    Test read_eclipse raises FileNotFoundError when file doesn't exist.

    Physics: Eclipse spectra depend on successful observation calculation;
    missing file indicates module not run or calculation failed.
    """
    outdir = str(tmp_path / 'empty_output')  # Non-existent directory

    with pytest.raises(FileNotFoundError, match='Eclipse spectrum file'):
        read_eclipse(outdir, 'offchem', 'composition')


@pytest.mark.unit
def test_obs_sources_constant():
    """
    Test OBS_SOURCES constant contains expected observation source types.

    Physics: Three standard sources of atmospheric/spectral data:
    - outgas: Volatile composition from outgassing module
    - profile: Atmospheric structure from climate/convection modules
    - offchem: Chemical composition from offline chemistry module
    """
    assert OBS_SOURCES == ('outgas', 'profile', 'offchem')
    assert len(OBS_SOURCES) == 3
    assert 'outgas' in OBS_SOURCES
    assert 'profile' in OBS_SOURCES
    assert 'offchem' in OBS_SOURCES


# ============================================================================
# Test proteus.observe.wrapper module
# ============================================================================


@pytest.mark.unit
def test_calc_synthetic_spectra_dummy_atmos_skips_profile():
    """
    Test calc_synthetic_spectra skips 'profile' source when atmosphere is dummy.

    Physics: Dummy atmospheric model has no detailed structure, so
    'profile' based spectra cannot be computed. Only outgas and offchem
    are available.
    """
    from unittest.mock import MagicMock, patch

    from proteus.observe.wrapper import calc_synthetic_spectra

    # Create mock config with dummy atmos_clim but valid other modules
    config = MagicMock()
    config.observe.synthesis = 'platon'
    config.atmos_clim.module = 'dummy'  # Use dummy atmospheric model
    config.atmos_chem.module = 'vulcan'  # Chemistry enabled

    hf_row = {'Time': 0.0}
    outdir = '/tmp/output'

    with (
        patch('proteus.observe.platon.transit_depth') as mock_transit,
        patch('proteus.observe.platon.eclipse_depth') as mock_eclipse,
    ):
        calc_synthetic_spectra(hf_row, outdir, config)

        # Should process 'outgas' and 'offchem' but skip 'profile'
        # Two sources processed: outgas and offchem
        # Each gets transit and eclipse calculations
        assert mock_transit.call_count == 2  # outgas, offchem
        assert mock_eclipse.call_count == 2  # outgas, offchem


@pytest.mark.unit
def test_calc_synthetic_spectra_no_chemistry_skips_offchem():
    """
    Test calc_synthetic_spectra skips 'offchem' source when chemistry is disabled.

    Physics: Offline chemistry module is disabled (None), so 'offchem'
    based spectra cannot be computed. Only outgas and profile sources
    are processed.
    """
    from unittest.mock import MagicMock, patch

    from proteus.observe.wrapper import calc_synthetic_spectra

    # Create mock config with chemistry disabled
    config = MagicMock()
    config.observe.synthesis = 'platon'
    config.atmos_clim.module = 'janus'  # Detailed atmospheric model
    config.atmos_chem.module = None  # Chemistry disabled

    hf_row = {'Time': 0.0}
    outdir = '/tmp/output'

    with (
        patch('proteus.observe.platon.transit_depth') as mock_transit,
        patch('proteus.observe.platon.eclipse_depth') as mock_eclipse,
    ):
        calc_synthetic_spectra(hf_row, outdir, config)

        # Should process 'outgas' and 'profile' but skip 'offchem'
        assert mock_transit.call_count == 2  # outgas, profile
        assert mock_eclipse.call_count == 2  # outgas, profile


@pytest.mark.unit
def test_calc_synthetic_spectra_all_sources_available():
    """
    Test calc_synthetic_spectra processes all sources when all modules enabled.

    Physics: All three sources (outgas, profile, offchem) available when
    detailed atmosphere model and offline chemistry are both enabled.
    """
    from unittest.mock import MagicMock, patch

    from proteus.observe.wrapper import calc_synthetic_spectra

    # Create mock config with all modules enabled
    config = MagicMock()
    config.observe.synthesis = 'platon'
    config.atmos_clim.module = 'janus'  # Detailed model
    config.atmos_chem.module = 'vulcan'  # Chemistry enabled

    hf_row = {'Time': 100.0}
    outdir = '/tmp/output'

    with (
        patch('proteus.observe.platon.transit_depth') as mock_transit,
        patch('proteus.observe.platon.eclipse_depth') as mock_eclipse,
    ):
        calc_synthetic_spectra(hf_row, outdir, config)

        # All three sources: outgas, profile, offchem
        assert mock_transit.call_count == 3
        assert mock_eclipse.call_count == 3


@pytest.mark.unit
def test_calc_synthetic_spectra_invalid_synthesis_module():
    """
    Test calc_synthetic_spectra raises ValueError for invalid synthesis module.

    Physics: Only 'platon' is currently supported for synthetic spectrum
    generation. Invalid module names should fail gracefully.
    """
    from unittest.mock import MagicMock

    from proteus.observe.wrapper import calc_synthetic_spectra

    config = MagicMock()
    config.observe.synthesis = 'unknown_module'  # Invalid
    config.atmos_clim.module = 'janus'
    config.atmos_chem.module = 'vulcan'

    hf_row = {'Time': 0.0}
    outdir = '/tmp/output'

    with pytest.raises(ValueError, match='Unknown synthesis module'):
        calc_synthetic_spectra(hf_row, outdir, config)


@pytest.mark.unit
def test_run_observe_calls_synthetic_spectra():
    """
    Test run_observe orchestrates synthetic spectrum calculation.

    Physics: run_observe is the main entry point for computing observation
    predictions. It delegates to calc_synthetic_spectra for spectrum generation.
    """
    from unittest.mock import MagicMock, patch

    from proteus.observe.wrapper import run_observe

    config = MagicMock()
    config.observe.synthesis = 'platon'
    config.atmos_clim.module = 'janus'
    config.atmos_chem.module = 'vulcan'

    hf_row = {'Time': 50.0}
    outdir = '/tmp/output'

    with patch('proteus.observe.wrapper.calc_synthetic_spectra') as mock_calc:
        run_observe(hf_row, outdir, config)

        # Verify calc_synthetic_spectra was called with correct arguments
        mock_calc.assert_called_once_with(hf_row, outdir, config)
