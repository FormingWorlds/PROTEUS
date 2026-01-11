"""
Unit tests for proteus.atmos_clim.common module.

This module tests the shared utility functions used by all atmosphere-climate modules
(JANUS, AGNI, etc.). It verifies:
- Robust NetCDF data ingestion (reading profiles, handling flags)
- Physical state conversions (pressure <-> radius)
- Configuration helpers (spectral file paths)
- Interpolation of lookup tables (Albedo)

See also:
- docs/test_infrastructure.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.atmos_clim.common import (
    Albedo_t,
    get_radius_from_pressure,
    get_spfile_name_and_bands,
    get_spfile_path,
    ncdf_flag_to_bool,
    read_atmosphere_data,
    read_ncdf_profile,
)


@pytest.mark.unit
def test_ncdf_flag_to_bool():
    """
    Test NetCDF flag conversion.

    NetCDF files often store booleans as character bytes ('y'/'n' or 'Y'/'N').
    This test verifies that the converter correctly handles these byte strings
    and maps them to Python booleans.
    """
    # The function expects var[0] to have .tobytes(), which numpy char arrays do
    y = np.array([b'y'], dtype='S1')
    Y = np.array([b'Y'], dtype='S1')
    n = np.array([b'n'], dtype='S1')
    N = np.array([b'N'], dtype='S1')

    assert ncdf_flag_to_bool(y) is True
    assert ncdf_flag_to_bool(Y) is True
    assert ncdf_flag_to_bool(n) is False
    assert ncdf_flag_to_bool(N) is False

    # Ensure it fails safely on invalid input
    with pytest.raises(ValueError):
        ncdf_flag_to_bool(np.array([b'x'], dtype='S1'))


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile(mock_ds, mock_isfile):
    """
    Test reading NetCDF profile data with mocked file I/O.

    Verifies that:
    1. Arrays are correctly properly read from the dataset variables.
    2. Interleaved arrays (layer centers and edges) are handled if necessary.
    3. Radius/Height conversion logic works (AGNI vs JANUS formats).
    4. Metadata flags are preserved.
    """
    # Setup mocks
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    # Mock variables directly
    # p: pressure (Pa)
    # pl: pressure at layer edges
    # tmp: temperature (K)
    # r/z: radius/height (m)
    ds_instance.variables = {
        'p': np.array([100.0]),
        'pl': np.array([110.0, 90.0]),
        'tmp': np.array([300.0]),
        'tmpl': np.array([310.0, 290.0]),
        'r': np.array([6.4e6]),
        'rl': np.array([6.3e6, 6.5e6]),
        'planet_radius': [6.0e6],
        'transparent': np.array([b'y'], dtype='S1'),
    }

    # Run function
    result = read_ncdf_profile('dummy.nc')

    # Verify values are correctly extracted
    assert result['p'][0] == 110.0  # Should match first element of pl
    assert result['p'][1] == 100.0  # Should match first element of p
    assert result['p'][2] == 90.0  # Should match second element of pl
    assert result['t'][1] == 300.0  # Temperature

    # The function converts all outputs to float arrays, even booleans
    assert result['transparent'] == 1.0

    # Verify AGNI-style radius/height logic (default path in function)
    # r = z + rp => z = r - rp
    # rp = 6.0e6
    # r[0] = 6.4e6 => z[0]Approx 4.0e5
    assert result['r'][1] == 6.4e6
    assert result['z'][1] == pytest.approx(4.0e5)

    mock_ds.assert_called_with('dummy.nc')


@pytest.mark.unit
@patch('proteus.atmos_clim.common.read_ncdf_profile')
def test_read_atmosphere_data(mock_read):
    """
    Test wrapper for reading multiple profiles for different timesteps.

    Ensures that the function iterates correctly over the requested times
    and aggregates the results.
    """
    mock_read.return_value = {'t': [300.0]}

    times = [0, 100]
    result = read_atmosphere_data('output_dir', times)

    assert len(result) == 2
    assert mock_read.call_count == 2


@pytest.mark.unit
def test_get_radius_from_pressure():
    """
    Test finding geometric radius corresponding to a pressure level.

    This functionality is critical for coupling where we need to find
    the planetary radius at a specific reference pressure (e.g. surface).
    """
    p_arr = np.array([100.0, 10.0, 1.0])
    r_arr = np.array([10.0, 20.0, 30.0])

    # Exact match: Target 10 Pa => expect 20 m
    p_close, r_close = get_radius_from_pressure(p_arr, r_arr, 10.0)
    assert p_close == 10.0
    assert r_close == 20.0

    # Nearest neighbor: Target 50 Pa
    # In linear space: |100-50|=50, |10-50|=40. So 10 Pa is closer.
    p_close, r_close = get_radius_from_pressure(p_arr, r_arr, 50.0)
    assert p_close == 10.0


@pytest.mark.unit
def test_spfile_helpers():
    """
    Test spectral file configuration helpers.

    Verifies that the correct file paths are constructed based on the
    atmosphere module configuration (e.g. 'Dayspring' band set).
    """
    # Mock config object
    mock_conf = MagicMock()
    mock_conf.atmos_clim.module = 'janus'
    mock_conf.atmos_clim.janus.spectral_bands = '16'
    mock_conf.atmos_clim.janus.spectral_group = 'Dayspring'

    # Test get_spfile_name_and_bands
    group, bands = get_spfile_name_and_bands(mock_conf)
    assert group == 'Dayspring'
    assert bands == '16'

    # Test get_spfile_path construction
    # Expected: <fwl_dir>/spectral_files/<group>/<bands>/<group>.sf
    path = get_spfile_path('/fwl/data', mock_conf)
    assert path == '/fwl/data/spectral_files/Dayspring/16/Dayspring.sf'


@pytest.mark.unit
@patch('proteus.atmos_clim.common.pd.read_csv')
@patch('proteus.atmos_clim.common.os.path.isfile')
def test_albedo_t(mock_isfile, mock_read_csv):
    """
    Test Albedo_t lookup table class.

    Verifies:
    1. CSV reading logic (mocked).
    2. Interpolation of albedo vs temperature.
    3. Clamping behavior outside the data range.
    """
    mock_isfile.return_value = True

    # Mock DataFrame response
    # mock_df = MagicMock() - Responding to ruff F841
    # mimic dict access data['tmp']
    data_dict = {'tmp': np.array([100.0, 300.0, 1000.0]), 'albedo': np.array([0.5, 0.3, 0.1])}
    mock_read_csv.return_value = data_dict

    # Initialize class
    alb = Albedo_t('dummy.csv')
    assert alb.ok

    # Test evaluation (interpolation)
    # At 100K -> 0.5 (exact)
    assert alb.evaluate(100.0) == pytest.approx(0.5)
    # At 1000K -> 0.1 (exact)
    assert alb.evaluate(1000.0) == pytest.approx(0.1)
    # At 300K -> 0.3 (exact)
    assert alb.evaluate(300.0) == pytest.approx(0.3)

    # Test clamping behavior (physics safety check)
    # Below min temp -> stay at min albedo val (0.5), don't extrapolate
    assert alb.evaluate(50.0) == pytest.approx(0.5)
    # Above max temp -> stay at max albedo val (0.1)
    assert alb.evaluate(2000.0) == pytest.approx(0.1)
