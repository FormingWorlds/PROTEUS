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
    get_oarr_from_parr,
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
        'gases': np.array([[b'H', b'2', b'O'], [b'C', b'O', b'2']], dtype='S1'),
        'x_gases': np.array([0.1, 0.9]),
        'aerosols': np.array([[b's', b'o', b'o', b't'], [b's', b'u', b'l', b'f']], dtype='S1'),
        'aer_mmr': np.array([[1e-6, 2e-6]]),
        'cloud_mmr': np.array([1e-5]),
    }

    # Run function
    result = read_ncdf_profile(
        'dummy.nc', extra_keys=['gases', 'x_gases', 'aerosols', 'aer_mmr', 'cloud_mmr']
    )

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
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_without_combining_edges(mock_ds, mock_isfile):
    """Read centre and edge arrays separately when ``combine_edges`` is false.

    This covers the branch added for callers that need native NetCDF layering
    (N centre levels and N+1 edge levels as separate arrays).
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    # Use JANUS-style height variables to ensure this branch also works with z/zl input.
    ds_instance.variables = {
        'p': np.array([100.0, 80.0]),
        'pl': np.array([110.0, 90.0, 70.0]),
        'tmp': np.array([500.0, 450.0]),
        'tmpl': np.array([520.0, 470.0, 430.0]),
        'z': np.array([1.0e4, 2.0e4]),
        'zl': np.array([0.0, 1.5e4, 2.5e4]),
        'planet_radius': [6.0e6],
        'solved': np.array([b'n'], dtype='S1'),
    }

    result = read_ncdf_profile('dummy.nc', combine_edges=False)

    np.testing.assert_allclose(result['p'], np.array([100.0, 80.0]))
    np.testing.assert_allclose(result['pl'], np.array([110.0, 90.0, 70.0]))
    np.testing.assert_allclose(result['t'], np.array([500.0, 450.0]))
    np.testing.assert_allclose(result['tmpl'], np.array([520.0, 470.0, 430.0]))

    # JANUS path: r = z + planet_radius and rl = zl + planet_radius.
    np.testing.assert_allclose(result['r'], np.array([6.01e6, 6.02e6]))
    np.testing.assert_allclose(result['rl'], np.array([6.0e6, 6.015e6, 6.025e6]))

    assert result['solved'] == 0.0
    assert result['transparent'] == 0.0
    assert result['converged'] == 0.0


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_with_aerosols(mock_ds, mock_isfile):
    """
    Test reading NetCDF profile data with aerosol mass mixing ratios.

    Physical scenario: AGNI can output aerosol profiles (e.g., sulfate, silicate)
    in the atmosphere. These are stored as aer_mmr(nlev_c, naeros) arrays with
    corresponding names in the 'aerosols' variable.
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    # Mock basic atmospheric profile with two aerosol species
    ds_instance.variables = {
        'p': np.array([100.0]),
        'pl': np.array([110.0, 90.0]),
        'tmp': np.array([300.0]),
        'tmpl': np.array([310.0, 290.0]),
        'r': np.array([6.4e6]),
        'rl': np.array([6.3e6, 6.5e6]),
        'planet_radius': [6.0e6],
        'transparent': np.array([b'y'], dtype='S1'),
        # Aerosol data: 2 species (Sulfate, Silicate) at 1 level each
        'aerosols': np.array(
            [
                [
                    b'S',
                    b'u',
                    b'l',
                    b'f',
                    b'a',
                    b't',
                    b'e',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                ],
                [
                    b'S',
                    b'i',
                    b'l',
                    b'i',
                    b'c',
                    b'a',
                    b't',
                    b'e',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                    b' ',
                ],
            ],
            dtype='S1',
        ),
        'aer_mmr': np.array([[1e-6, 2e-6]]),  # 1 level, 2 aerosol species
    }

    # Read with aerosol data
    result = read_ncdf_profile('dummy.nc', extra_keys=['aer_mmr', 'aerosols'])

    # Verify basic profile data
    assert 'p' in result
    assert 'transparent' in result

    # Verify aerosol list was read (stored as numpy array)
    assert 'aerosols' in result
    assert len(result['aerosols']) == 2
    assert 'Silicate' in result['aerosols']
    assert 'Sulfate' in result['aerosols']

    # Verify individual aerosol MMRs were extracted
    assert 'Sulfate_mmr' in result
    assert 'Silicate_mmr' in result
    np.testing.assert_allclose(result['Sulfate_mmr'], np.array([1e-6]))
    np.testing.assert_allclose(result['Silicate_mmr'], np.array([2e-6]))


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_gases_list(mock_ds, mock_isfile):
    """
    Test reading list of gas species names without VMRs.

    Physical scenario: When reading just the gas species present in the
    atmosphere without their mixing ratios (useful for metadata queries).
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    # Mock gas names only
    ds_instance.variables = {
        'p': np.array([100.0]),
        'pl': np.array([110.0, 90.0]),
        'tmp': np.array([300.0]),
        'tmpl': np.array([310.0, 290.0]),
        'r': np.array([6.4e6]),
        'rl': np.array([6.3e6, 6.5e6]),
        'planet_radius': [6.0e6],
        'solved': np.array([b'y'], dtype='S1'),
        # Gas species names (3 gases: H2O, CO2, N2)
        'gases': np.array(
            [
                [b'H', b'2', b'O', b' ', b' ', b' '],
                [b'C', b'O', b'2', b' ', b' ', b' '],
                [b'N', b'2', b' ', b' ', b' ', b' '],
            ],
            dtype='S1',
        ),
    }

    # Read with gases key
    result = read_ncdf_profile('dummy.nc', extra_keys=['gases'])

    # Verify gas list was read and parsed correctly
    assert 'gases' in result
    assert len(result['gases']) == 3
    assert 'H2O' in result['gases']
    assert 'CO2' in result['gases']
    assert 'N2' in result['gases']


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_aerosols_list_only(mock_ds, mock_isfile):
    """
    Test reading list of aerosol species names without MMRs.

    Physical scenario: When checking which aerosol types are available
    in the simulation output without loading full profiles.
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    ds_instance.variables = {
        'p': np.array([100.0]),
        'pl': np.array([110.0, 90.0]),
        'tmp': np.array([300.0]),
        'tmpl': np.array([310.0, 290.0]),
        'r': np.array([6.4e6]),
        'rl': np.array([6.3e6, 6.5e6]),
        'planet_radius': [6.0e6],
        'solved': np.array([b'n'], dtype='S1'),
        # Aerosol species names only
        'aerosols': np.array(
            [
                [b'S', b'u', b'l', b'f', b'a', b't', b'e', b' '],
                [b'H', b'a', b'z', b'e', b' ', b' ', b' ', b' '],
            ],
            dtype='S1',
        ),
    }

    # Read with aerosols key
    result = read_ncdf_profile('dummy.nc', extra_keys=['aerosols'])

    # Verify aerosol list was read
    assert 'aerosols' in result
    assert len(result['aerosols']) == 2
    assert 'Sulfate' in result['aerosols']
    assert 'Haze' in result['aerosols']


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_no_aerosols_in_file(mock_ds, mock_isfile):
    """
    Test reading profile when aerosols key is requested but not present.

    Physical scenario: Attempting to read aerosol data from a simulation
    run without aerosols_enabled=True. Should handle gracefully.
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    # No aerosol variables in dataset
    ds_instance.variables = {
        'p': np.array([100.0]),
        'pl': np.array([110.0, 90.0]),
        'tmp': np.array([300.0]),
        'tmpl': np.array([310.0, 290.0]),
        'r': np.array([6.4e6]),
        'rl': np.array([6.3e6, 6.5e6]),
        'planet_radius': [6.0e6],
        'solved': np.array([b'y'], dtype='S1'),
    }

    # Read with aerosols/aer_mmr keys (should handle missing gracefully)
    result = read_ncdf_profile('dummy.nc', extra_keys=['aerosols', 'aer_mmr'])

    # Verify profile data is still read
    assert 'p' in result
    assert 'solved' in result

    # Missing keys should not be in result
    assert 'aerosols' not in result
    assert 'aer_mmr' not in result


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_with_clouds(mock_ds, mock_isfile):
    """
    Test reading NetCDF profile data with cloud properties.

    Physical scenario: AGNI outputs cloud mass mixing ratio, cloud area fraction,
    and cloud particle size when cloud_enabled=True.
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    ds_instance.variables = {
        'p': np.array([100.0, 200.0]),
        'pl': np.array([110.0, 150.0, 190.0]),
        'tmp': np.array([300.0, 280.0]),
        'tmpl': np.array([310.0, 290.0, 270.0]),
        'r': np.array([6.4e6, 6.3e6]),
        'rl': np.array([6.5e6, 6.35e6, 6.2e6]),
        'planet_radius': [6.0e6],
        'solved': np.array([b'y'], dtype='S1'),
        # Cloud data
        'cloud_mmr': np.array([1e-5, 2e-5]),
        'cloud_area': np.array([0.5, 0.8]),
        'cloud_size': np.array([1e-5, 1.2e-5]),
    }

    result = read_ncdf_profile('dummy.nc', extra_keys=['cloud_mmr', 'cloud_area', 'cloud_size'])

    # Verify cloud data was read
    assert 'cloud_mmr' in result
    assert 'cloud_area' in result
    assert 'cloud_size' in result

    np.testing.assert_allclose(result['cloud_mmr'], np.array([1e-5, 2e-5]))
    np.testing.assert_allclose(result['cloud_area'], np.array([0.5, 0.8]))
    np.testing.assert_allclose(result['cloud_size'], np.array([1e-5, 1.2e-5]))


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
def test_get_oarr_from_parr():
    """
    Test generic lookup of a value in one array at the nearest pressure level.

    get_oarr_from_parr is the generalised replacement for the older
    get_radius_from_pressure. It finds the entry in o_arr whose
    corresponding pressure in p_arr is closest to p_tgt.
    """
    p_arr = np.array([100.0, 10.0, 1.0])
    o_arr = np.array([10.0, 20.0, 30.0])

    # Exact match
    p_close, o_close = get_oarr_from_parr(p_arr, o_arr, 10.0)
    assert p_close == 10.0
    assert o_close == 20.0

    # Nearest neighbor
    p_close, o_close = get_oarr_from_parr(p_arr, o_arr, 50.0)
    assert p_close == 10.0


@pytest.mark.unit
def test_get_radius_from_pressure():
    """
    Test backwards-compatible wrapper around get_oarr_from_parr.

    get_radius_from_pressure delegates to get_oarr_from_parr but is kept
    so that older call-sites continue to work.
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
