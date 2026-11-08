"""
Unit tests for proteus.atmos_clim.common module.

This module tests the shared utility functions used by all atmosphere-climate modules
(JANUS, AGNI, etc.). It verifies:
- Robust NetCDF data ingestion (reading profiles, handling flags)
- Physical state conversions (pressure <-> radius)
- Configuration helpers (spectral file paths)

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('hypothesis')
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.atmos_clim.common import (  # noqa: E402
    clip_radius_to_hill,
    find_latest_atmosphere_time,
    get_oarr_from_parr,
    get_radius_from_pressure,
    get_spfile_name_and_bands,
    get_spfile_path,
    lid_boundary,
    lid_flux_cap,
    lid_flux_slope,
    lid_open_flux,
    list_atmosphere_times,
    ncdf_flag_to_bool,
    read_atmosphere_data,
    read_ncdf_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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


@pytest.mark.unit
def test_ncdf_flag_to_bool_logs_and_returns_none_on_invalid_input(caplog):
    """An unparseable flag byte must fail safely, not raise.

    Contract clause: a malformed NetCDF flag is reported via the logger and
    the caller decides the fallback (see `read_ncdf_profile`, which treats a
    `None` return as `False`), rather than aborting the whole profile read.

    Discrimination guard: a regression that silently swallowed the bad input
    and returned e.g. `False` (indistinguishable from a legitimate 'n') would
    still pass a bare `is None` check on its own, but not alongside the error
    log assertion below, which pins the caller-visible signal too.
    """
    with caplog.at_level('ERROR', logger='fwl.proteus.atmos_clim.common'):
        result = ncdf_flag_to_bool(np.array([b'x'], dtype='S1'))

    assert result is None
    assert any('Could not parse' in rec.message for rec in caplog.records)


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
        'gravity': np.array([9.8]),
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
    assert result['p'][0] == pytest.approx(110.0, rel=1e-12)  # first element of pl
    assert result['p'][1] == pytest.approx(100.0, rel=1e-12)  # first element of p
    assert result['p'][2] == pytest.approx(90.0, rel=1e-12)  # second element of pl
    assert result['t'][1] == pytest.approx(300.0, rel=1e-12)  # Temperature
    assert result['g'] == pytest.approx(np.array([9.8, 9.8, 9.8]), rel=1e-12)

    # The function converts all outputs to float arrays, even booleans
    assert result['transparent'] == pytest.approx(1.0, rel=1e-12)

    # Verify AGNI-style radius/height logic (default path in function)
    # r = z + rp => z = r - rp
    # rp = 6.0e6
    # r[0] = 6.4e6 => z[0]Approx 4.0e5
    assert result['r'][1] == pytest.approx(6.4e6, rel=1e-12)
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
        'gravity': np.array([9.8, 9.6]),
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
    np.testing.assert_allclose(result['g'], np.array([9.8, 9.6]))
    np.testing.assert_allclose(result['t'], np.array([500.0, 450.0]))
    np.testing.assert_allclose(result['tmpl'], np.array([520.0, 470.0, 430.0]))

    # JANUS path: r = z + planet_radius and rl = zl + planet_radius.
    np.testing.assert_allclose(result['r'], np.array([6.01e6, 6.02e6]))
    np.testing.assert_allclose(result['rl'], np.array([6.0e6, 6.015e6, 6.025e6]))

    assert result['solved'] == pytest.approx(0.0, abs=1e-12)
    assert result['transparent'] == pytest.approx(0.0, abs=1e-12)
    assert result['converged'] == pytest.approx(0.0, abs=1e-12)


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
        'gravity': np.array([9.8]),
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
        'gravity': np.array([9.8]),
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
        'gravity': np.array([9.8]),
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
        'gravity': np.array([9.8]),
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
        'gravity': np.array([9.8, 9.7]),
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
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_missing_gravity_falls_back_to_zeros(mock_ds, mock_isfile, caplog):
    """A NetCDF file with no `gravity` variable must not abort the read.

    Contract clause: an older or hand-built NetCDF file may predate the
    `gravity` output field. Reading it must fail safely (log + zero-filled
    fallback) rather than raising `KeyError`, so a single malformed archive
    does not stop a batch read of many profiles.

    Discrimination guard: the zero fallback is checked against every level,
    not just the first, so a regression that only zeroed `g[0]` would still
    be caught.
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    ds_instance.variables = {
        'p': np.array([100.0, 200.0]),
        'pl': np.array([110.0, 150.0, 190.0]),
        # 'gravity' deliberately absent.
        'tmp': np.array([300.0, 280.0]),
        'tmpl': np.array([310.0, 290.0, 270.0]),
        'r': np.array([6.4e6, 6.3e6]),
        'rl': np.array([6.5e6, 6.35e6, 6.2e6]),
        'planet_radius': [6.0e6],
    }

    with caplog.at_level('ERROR', logger='fwl.proteus.atmos_clim.common'):
        result = read_ncdf_profile('dummy.nc', combine_edges=False)

    assert any('gravity' in rec.message for rec in caplog.records)
    # Every level is zero, not just the first (rules out a partial fallback).
    np.testing.assert_array_equal(result['g'], np.zeros(2))


@pytest.mark.unit
@patch('proteus.atmos_clim.common.os.path.isfile')
@patch('netCDF4.Dataset')
def test_read_ncdf_profile_invalid_flag_falls_back_to_false(mock_ds, mock_isfile, caplog):
    """An unparseable metadata flag in the file must read back as False.

    Contract clause: `ncdf_flag_to_bool` now returns `None` (logged, not
    raised) on a byte it cannot parse; `read_ncdf_profile` must catch that
    `None` and coerce it to `False` rather than propagating a non-boolean
    value into the profile dict.

    Discrimination guard: the two untouched flags ('solved', 'converged',
    absent from this file) still take the plain "not found" default of
    False, so a regression that defaulted every flag to False regardless of
    the parse result would not be distinguishable without also asserting the
    error was actually logged for the malformed one.
    """
    mock_isfile.return_value = True

    ds_instance = MagicMock()
    mock_ds.return_value = ds_instance

    ds_instance.variables = {
        'p': np.array([100.0]),
        'pl': np.array([110.0, 90.0]),
        'gravity': np.array([9.8]),
        'tmp': np.array([300.0]),
        'tmpl': np.array([310.0, 290.0]),
        'r': np.array([6.4e6]),
        'rl': np.array([6.3e6, 6.5e6]),
        'planet_radius': [6.0e6],
        # Malformed byte: neither 'y' nor 'n'.
        'transparent': np.array([b'x'], dtype='S1'),
    }

    with caplog.at_level('ERROR', logger='fwl.proteus.atmos_clim.common'):
        result = read_ncdf_profile('dummy.nc')

    # Every value is coerced to a float array at the end of the read, so the
    # fallback surfaces as 0.0 rather than the Python singleton `False`.
    assert float(result['transparent']) == pytest.approx(0.0)
    assert any('Could not parse' in rec.message for rec in caplog.records)
    # Untouched flags still take the plain "not found" default.
    assert float(result['solved']) == pytest.approx(0.0)
    assert float(result['converged']) == pytest.approx(0.0)


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
    assert p_close == pytest.approx(10.0, rel=1e-12)
    assert o_close == pytest.approx(20.0, rel=1e-12)

    # Nearest neighbor
    p_close, o_close = get_oarr_from_parr(p_arr, o_arr, 50.0)
    assert p_close == pytest.approx(10.0, rel=1e-12)


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
    assert p_close == pytest.approx(10.0, rel=1e-12)
    assert r_close == pytest.approx(20.0, rel=1e-12)

    # Nearest neighbor: Target 50 Pa
    # In linear space: |100-50|=50, |10-50|=40. So 10 Pa is closer.
    p_close, r_close = get_radius_from_pressure(p_arr, r_arr, 50.0)
    assert p_close == pytest.approx(10.0, rel=1e-12)


@pytest.mark.unit
def test_spfile_helpers(tmp_path):
    """
    Test spectral file configuration helpers.

    Verifies that the correct file paths are constructed based on the
    atmosphere module configuration (e.g. 'Dayspring' band set).
    """
    # Mock config object
    mock_conf = MagicMock()
    mock_conf.atmos_clim.module = 'janus'
    mock_conf.atmos_clim.spectral_bands = '16'
    mock_conf.atmos_clim.spectral_group = 'Dayspring'

    # Test get_spfile_name_and_bands
    group, bands = get_spfile_name_and_bands(mock_conf)
    assert group == 'Dayspring'
    assert bands == '16'

    # Test get_spfile_path construction
    # Expected: <fwl_dir>/atmos_clim/spectral_files/<group>/<bands>/r<record>/<group>.sf
    path = get_spfile_path(str(tmp_path), mock_conf)
    assert path == str(
        tmp_path
        / 'atmos_clim'
        / 'spectral_files'
        / 'dayspring'
        / '16'
        / 'r15799318'
        / 'Dayspring.sf'
    )


@pytest.mark.unit
def test_get_spfile_path_reads_an_undeclared_group_from_the_local_tree(tmp_path, caplog):
    """A group that no manifest declares is read from spectral_files/<group>/<bands>,
    with an INFO line, also before the file exists."""
    conf = MagicMock()
    conf.atmos_clim.spectral_group = 'MyGroup'
    conf.atmos_clim.spectral_bands = '48'
    local = tmp_path / 'spectral_files' / 'MyGroup' / '48' / 'MyGroup.sf'
    with caplog.at_level('INFO', logger='fwl.proteus.atmos_clim.common'):
        assert get_spfile_path(str(tmp_path), conf) == str(local)
    assert 'is in no manifest' in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    'group, bands', [('MyGroup', '48'), ('Dayspring', '64'), ('Dayspring', '48')]
)
def test_require_spfile_path_stops_with_status_20_and_the_fetch_command(tmp_path, group, bands):
    """A missing spectral file writes status 20 and, for a declared group and band count,
    names `proteus get spectral` and `fwl-io relocate`; a present one is returned."""
    from proteus.atmos_clim.common import require_spfile_path

    conf = MagicMock()
    conf.atmos_clim.spectral_group = group
    conf.atmos_clim.spectral_bands = bands
    dirs = {'fwl': str(tmp_path), 'output': str(tmp_path / 'out')}
    fix = (
        '`proteus get spectral`'
        if bands == '48' and group == 'Dayspring'
        else 'No manifest declares this group and band count'
    )
    with (
        patch('proteus.atmos_clim.common.UpdateStatusfile') as status,
        pytest.raises(FileNotFoundError, match=fix) as exc,
    ):
        require_spfile_path(dirs, conf)
    status.assert_called_once_with(dirs, 20)
    assert ('`fwl-io relocate`' in str(exc.value)) == (fix == '`proteus get spectral`')
    path = Path(get_spfile_path(str(tmp_path), conf))
    path.parent.mkdir(parents=True)
    path.write_text('sf')
    with patch('proteus.atmos_clim.common.UpdateStatusfile') as status:
        assert require_spfile_path(dirs, conf) == str(path)
    status.assert_not_called()


# ---------------------------------------------------------------------------
# Coverage for previously-untested error branches: missing NetCDF file,
# archived-data warning.
# ---------------------------------------------------------------------------


def test_read_ncdf_profile_returns_none_when_file_missing(caplog, tmp_path):
    """read_ncdf_profile must log an error and return None when the
    NetCDF file is absent. The main loop relies on this contract to
    gate downstream reads.

    Discriminating: a regression that raised FileNotFoundError instead
    of returning None would crash the loop. Pin both the return value
    and the error log message.
    """
    import logging

    from proteus.atmos_clim.common import read_ncdf_profile

    nc_fpath = str(tmp_path / 'does_not_exist.nc')
    with caplog.at_level(logging.ERROR, logger='fwl.proteus.atmos_clim.common'):
        result = read_ncdf_profile(nc_fpath)
    assert result is None
    assert any('Could not find NetCDF file' in rec.message for rec in caplog.records)


def test_read_atmosphere_data_returns_none_when_any_profile_missing(
    caplog, tmp_path, monkeypatch
):
    """When at least one timestep NetCDF is unreadable, the helper
    logs a warning and returns None. The 'extract archived data'
    hint should also fire when a data.tar exists in the output
    folder, pointing the user at the recovery path.

    Discriminating: a regression that returned the partial list
    (with None entries) would fail any `is None` check at the call
    site. Pin both the return value and the archived-data warning.
    """
    import logging

    from proteus.atmos_clim import common
    from proteus.atmos_clim.common import read_atmosphere_data

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'data.tar').write_bytes(b'fake-archive-bytes')
    monkeypatch.setattr(common, 'read_ncdf_profile', lambda *_a, **_k: None)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.atmos_clim.common'):
        result = read_atmosphere_data(str(tmp_path), times=[0.0, 1.0])
    assert result is None
    messages = [r.message for r in caplog.records]
    assert any('NetCDF files could not be found' in m for m in messages)
    assert any('extract archived data' in m for m in messages)


def test_find_latest_atmosphere_time_returns_max(tmp_path):
    """The latest snapshot is the maximum parsed time, not the first globbed
    or the file count.

    Files are created out of chronological order and with a count (3) that
    differs from every time key, so a regression returning the glob-order
    first element, the minimum, or len(files) would all disagree with the
    correct maximum (5000).
    """
    data = tmp_path / 'data'
    data.mkdir()
    for t in (999, 5000, 100):
        (data / f'{t}_atm.nc').write_text('x')
    # An unrelated file must be ignored by the *_atm.nc glob.
    (data / '5000.sflux').write_text('x')

    latest = find_latest_atmosphere_time(str(tmp_path))
    assert latest == pytest.approx(5000.0, rel=1e-12)
    # Discrimination: not the count of files, not the minimum.
    assert latest != pytest.approx(3.0)
    assert latest != pytest.approx(100.0)


def test_find_latest_atmosphere_time_empty_returns_none(tmp_path):
    """With no atmosphere NetCDF files the helper returns None rather than
    raising, so callers can degrade gracefully.

    The data directory contains a non-matching file to confirm the glob is
    specific to the ``*_atm.nc`` pattern.
    """
    data = tmp_path / 'data'
    data.mkdir()
    (data / '1000.sflux').write_text('x')

    assert find_latest_atmosphere_time(str(tmp_path)) is None
    # Also handles a missing data directory without raising.
    assert find_latest_atmosphere_time(str(tmp_path / 'nonexistent')) is None


def test_list_atmosphere_times_ignores_other_and_quarantined_files(tmp_path):
    """Only data/*_atm.nc files count: interior and chemistry snapshots, stellar
    spectra and quarantined '.incomplete' files are left out, and the times
    come back ascending."""
    data = tmp_path / 'data'
    data.mkdir()
    for name in (
        '30p500_atm.nc',
        '0p250_atm.nc',
        '40p000_atm.nc.incomplete',
        '20p000_int.nc',
        '20p000_chem.nc',
        '20.sflux',
    ):
        (data / name).write_text('')

    times = list_atmosphere_times(str(tmp_path))

    assert times == pytest.approx([0.25, 30.5])
    assert find_latest_atmosphere_time(str(tmp_path)) == pytest.approx(times[-1])


# ---------------------------------------------------------------------------
# clip_radius_to_hill: the XUV level never sizes escape beyond the Hill radius
# ---------------------------------------------------------------------------


def _clip_config(enabled: bool = True, frac: float = 1.0):
    """Escape-config namespace carrying only what the clip reads."""
    from types import SimpleNamespace

    return SimpleNamespace(escape=SimpleNamespace(hill_clamp=enabled, hill_clamp_frac=frac))


@pytest.mark.physics_invariant
def test_clip_radius_to_hill_bounds_the_radius():
    """A radius beyond the Hill radius comes back at frac * R_Hill, and one
    inside comes back untouched, so the escape cross-section is bounded.

    The energy-limited rate goes as the radius cubed: the unclipped input at
    6x the Hill radius would inflate the rate 216-fold, so the discriminating
    check is that the clipped output removes that factor entirely.
    """
    hf_row = {'hill_radius': 1.0e8, 'R_int': 6.4e6}

    clipped = clip_radius_to_hill(_clip_config(), hf_row, 6.0e8)
    assert clipped == pytest.approx(1.0e8, rel=1e-12)
    assert (6.0e8 / clipped) ** 3 == pytest.approx(216.0, rel=1e-9)

    inside = clip_radius_to_hill(_clip_config(), hf_row, 7.0e7)
    assert inside == pytest.approx(7.0e7, rel=1e-12)

    # The fraction scales the limit, not the radius.
    half = clip_radius_to_hill(_clip_config(frac=0.5), hf_row, 6.0e8)
    assert half == pytest.approx(5.0e7, rel=1e-12)


@pytest.mark.physics_invariant
def test_clip_radius_to_hill_never_goes_below_the_solid_body():
    """The limit floors at R_int: the solid body is bound by definition, so a
    Hill radius inside the planet must not shrink the level below the surface.
    """
    hf_row = {'hill_radius': 3.0e6, 'R_int': 6.4e6}  # Hill inside the planet
    clipped = clip_radius_to_hill(_clip_config(), hf_row, 1.0e7)
    assert clipped == pytest.approx(6.4e6, rel=1e-12)
    # Discrimination: the naive frac * R_Hill limit is a factor 2.1 smaller.
    assert clipped != pytest.approx(3.0e6, rel=1e-1)


def test_clip_radius_to_hill_skips_when_disabled_or_unset():
    """Disabled config or a Hill radius that is zero (before the first orbit
    update) or non-finite leaves the radius untouched rather than clipping
    against a value that does not exist.
    """
    r = 6.0e8
    assert clip_radius_to_hill(_clip_config(enabled=False), {'hill_radius': 1.0e8}, r) == r
    assert clip_radius_to_hill(_clip_config(), {'hill_radius': 0.0}, r) == r
    assert clip_radius_to_hill(_clip_config(), {'hill_radius': float('nan')}, r) == r
    assert clip_radius_to_hill(_clip_config(), {}, r) == r


@pytest.mark.physics_invariant
@given(
    w=st.floats(1e-6, 1.0),
    G_lid=st.floats(1e-3, 1e5),
    d=st.floats(1e-4, 1.0),
    k=st.floats(0.1, 10.0),
    T_node=st.floats(300.0, 4000.0),
    dT_b=st.floats(0.0, 500.0),
    dT_s=st.floats(0.0, 3000.0),
)
@settings(max_examples=300, deadline=None)
def test_the_lid_never_loses_more_than_the_open_surface(w, G_lid, d, k, T_node, dT_b, dT_s):
    """Rule 3 of the lid coupling: with T_b <= T_node and T_surf <= T_node the call flux of
    lid_boundary, k / d_call (T_melt - T_surf), is at most the flux of the open surface,
    G_skin (T_node - T_surf), for every weight and lid conductance."""
    T_b, T_s = T_node - dT_b, T_node - dT_s
    row = dict(
        boundary_layer_thickness=d,
        boundary_layer_conductivity=k,
        T_magma=T_node,
        lid_weight=w,
        lid_conductance=G_lid,
        lid_T_base=T_b,
        lid_T_node=T_node,
    )
    d_call, k_call, T_melt = lid_boundary(row)
    F_call, F_open = k_call / d_call * (T_melt - T_s), k / d * (T_node - T_s)
    tol = 1e-12 * (abs(F_open) + k / d * T_node)
    assert F_call <= F_open + tol
    # the lid part alone loses no more than the open surface would at that weight
    G_part = w * min(G_lid, k / d)
    assert G_part * (T_b - T_s) <= w * F_open + tol


def _row(w, G=4.0, T_b=1400.0, T_node=1500.0, **kw):
    return dict(
        lid_weight=w,
        lid_conductance=G,
        lid_T_base=T_b,
        lid_T_node=T_node,
        T_magma=1450.0,
        boundary_layer_thickness=kw.pop('d', 0.01),
        boundary_layer_conductivity=kw.pop('k', 2.0),
        **kw,
    )


@pytest.mark.physics_invariant
def test_boundary_is_the_skin_without_a_lid_and_a_parallel_sum_with_one():
    """Without a lid the call takes the skin and T_magma; with one the parallel sum of the two parts."""
    no_lid = {
        'T_magma': 1450.0,
        'boundary_layer_thickness': 0.01,
        'boundary_layer_conductivity': 2.0,
    }
    assert lid_boundary(no_lid) == (0.01, 2.0, 1450.0)
    assert lid_boundary(_row(0.0)) == (0.01, 2.0, 1450.0)
    assert lid_boundary(_row(1.0)) == (2.0 / 4.0, 2.0, 1400.0)
    d, k, T = lid_boundary(_row(0.5))
    G = k / d
    assert G == pytest.approx(0.5 * 4.0 + 0.5 * 200.0, rel=1e-14)
    assert G * T == pytest.approx(0.5 * 4.0 * 1400.0 + 0.5 * 200.0 * 1500.0, rel=1e-14)


def test_call_caps_the_lid_conductance_at_the_skin():
    """In the call a lid conducts at most like the skin, so a thin lid never takes more than
    the open surface would."""
    d, k, T = lid_boundary(_row(1.0, G=400.0))
    assert k / d == pytest.approx(200.0, rel=1e-14)
    assert T == pytest.approx(1400.0, rel=1e-14)
    d, k, T = lid_boundary(_row(0.5, G=400.0))
    assert k / d == pytest.approx(200.0, rel=1e-14)
    assert (k / d) * T == pytest.approx(0.5 * 200.0 * 1400.0 + 0.5 * 200.0 * 1500.0, rel=1e-14)
    # at the skin value and below it the lid conductance passes unchanged
    assert lid_boundary(_row(1.0, G=200.0))[0] == pytest.approx(0.01, rel=1e-14)
    assert lid_boundary(_row(1.0, G=199.0))[0] == pytest.approx(2.0 / 199.0, rel=1e-14)


def test_boundary_reads_the_skin_of_the_columns():
    """A skin from the columns, as from condensate layers, sets the open branch."""
    d, k, T = lid_boundary(_row(0.5, d=0.1, k=1.0))
    assert k / d == pytest.approx(0.5 * 4.0 + 0.5 * 10.0, rel=1e-14)
    assert (k / d) * T == pytest.approx(0.5 * 4.0 * 1400.0 + 0.5 * 10.0 * 1500.0, rel=1e-14)
    row = _balanced(_row(0.5, d=0.1, k=1.0, T_surf=900.0))
    assert lid_open_flux(row) == pytest.approx(10.0 * 600.0, rel=1e-12)


def test_a_zero_skin_passes_without_a_lid_and_is_refused_with_one():
    """Without a lid the zero columns of a fixed surface pass unchanged; a live lid refuses."""
    assert lid_boundary(_row(0.0, d=0.0, k=0.0)) == (0.0, 0.0, 1450.0)
    assert lid_open_flux(_row(0.0, d=0.0, k=0.0, F_atm=42.0)) == pytest.approx(42.0, rel=1e-15)
    with pytest.raises(ValueError, match='needs a skin'):
        lid_boundary(_row(0.5, d=0.0, k=0.0))
    with pytest.raises(ValueError, match='needs a skin'):
        lid_open_flux(_row(0.5, d=0.0, k=0.0, T_surf=900.0))
    with pytest.raises(ValueError, match='needs a skin'):
        lid_boundary(_row(0.5, d=0.0, k=2.0))


def _balanced(row, gap=1.0):
    """Row with F_atm at ``gap`` times the parallel skin balance of its call."""
    d, k, T_melt = lid_boundary(row)
    row['F_atm'] = gap * k / d * (T_melt - row['T_surf'])
    return row


@pytest.mark.physics_invariant
def test_the_two_parts_sum_to_the_balance_of_the_atmosphere():
    """Where the atmosphere closes the skin balance, the open part is the skin flux and the
    weighted parts sum to F_atm."""
    row = _balanced(_row(0.3, T_surf=900.0))
    F_lid = 4.0 * (1400.0 - 900.0)
    F_open = lid_open_flux(row)
    assert F_open == pytest.approx(200.0 * (1500.0 - 900.0), rel=1e-14)
    assert 0.3 * F_lid + 0.7 * F_open == pytest.approx(row['F_atm'], rel=1e-12)


@pytest.mark.physics_invariant
def test_open_part_takes_the_gap_of_the_atmosphere():
    """Off the skin balance the open part is the skin flux scaled by F_atm over the balance."""
    row = _balanced(_row(0.3, T_surf=900.0), gap=0.95)
    S, L = 200.0 * 600.0, 4.0 * 500.0
    assert lid_open_flux(row) == pytest.approx(0.95 * S, rel=1e-14)
    assert 0.3 * 0.95 * L + 0.7 * lid_open_flux(row) == pytest.approx(row['F_atm'], rel=1e-12)


@pytest.mark.parametrize('gap', [0.95, 1.0, 1.05])
@pytest.mark.parametrize('w', [1e-12, 1e-9])
def test_open_part_goes_to_f_atm_as_the_weight_goes_to_zero(w, gap):
    """The boundary of the interior is continuous at the onset: F_atm at w = 0 and as w -> 0."""
    row = _balanced(_row(w, T_surf=1300.0), gap=gap)
    assert lid_open_flux(row) == pytest.approx(row['F_atm'], rel=1e-8)
    assert lid_open_flux(dict(row, lid_weight=0.0)) == row['F_atm']


def test_open_part_without_a_positive_balance_is_the_skin_flux(caplog, monkeypatch):
    """A surface at or above the melt side of the call gives no scale; the skin flux stays,
    with one warning per process."""
    import proteus.atmos_clim.common as common

    monkeypatch.setattr(common, '_OPEN_FLUX_WARNED', [False])
    row = _row(0.99, T_surf=1450.0, F_atm=7.0)
    d, k, T_melt = lid_boundary(row)
    assert k / d * (T_melt - 1450.0) < 0.0
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.common'):
        assert lid_open_flux(row) == pytest.approx(200.0 * 50.0, rel=1e-14)
        assert lid_open_flux(_row(0.5, T_surf=1500.0, F_atm=7.0)) == 0.0
        assert lid_open_flux(_balanced(_row(0.5, T_surf=900.0))) == pytest.approx(200.0 * 600.0)
    warned = [r for r in caplog.records if 'unscaled' in r.getMessage()]
    assert len(warned) == 1
    assert 'T_surf 1450.00 K' in warned[0].getMessage()


@pytest.mark.physics_invariant
@given(
    w=st.floats(1e-6, 1.0),
    G_lid=st.floats(1e-3, 1e5),
    d=st.floats(1e-4, 1.0),
    k=st.floats(0.1, 10.0),
    T_node=st.floats(300.0, 4000.0),
    dT_b=st.floats(0.0, 250.0),
    dT_s=st.floats(1.0, 3000.0),
    gap=st.floats(0.5, 1.5),
)
@settings(max_examples=300, deadline=None)
def test_the_open_part_and_the_cap_miss_f_atm_by_the_unscaled_cap(
    w, G_lid, d, k, T_node, dT_b, dT_s, gap
):
    """Only the open part is scaled by r = F_atm over the balance of the call, so the cap C of the
    lid part and the open part sum to F_atm - w C (r - 1)."""
    T_b = T_node - dT_b
    T_s = T_b - min(dT_s, T_b - 1.0)
    row = _balanced(_row(w, G=G_lid, T_b=T_b, T_node=T_node, d=d, k=k, T_surf=T_s), gap=gap)
    S = k / d * (T_node - T_s)
    L = min(G_lid, k / d) * (T_b - T_s)
    F_open = lid_open_flux(row)
    assert F_open == pytest.approx(gap * S, rel=1e-9)
    C = lid_flux_cap(row)
    assert C == pytest.approx(L, rel=1e-12)
    assert w * C + (1.0 - w) * F_open == pytest.approx(
        row['F_atm'] - w * C * (gap - 1.0), rel=1e-9, abs=1e-9 * row['F_atm']
    )


def test_open_flux_without_a_lid_is_the_flux_of_the_atmosphere():
    """Without a lid the open-surface flux is the F_atm of the row."""
    assert lid_open_flux({'F_atm': 42.0}) == pytest.approx(42.0, rel=1e-15)
    assert lid_open_flux(_row(0.0, F_atm=42.0)) == pytest.approx(42.0, rel=1e-15)


def test_cap_is_the_flux_the_call_gave_the_lid():
    """The cap is the lid branch of the atmosphere balance: G_call (T_b - T_surf)."""
    assert lid_flux_cap(_row(1.0, T_surf=900.0)) == pytest.approx(4.0 * 500.0, rel=1e-14)
    assert lid_flux_cap(_row(0.4, T_surf=900.0)) == pytest.approx(4.0 * 500.0, rel=1e-14)
    # a lid above the skin conductance gets the capped conductance of the call
    assert lid_flux_cap(_row(1.0, G=400.0, T_surf=900.0)) == pytest.approx(200.0 * 500.0)
    # no lid, or a surface at or above the base: no cap
    assert lid_flux_cap(_row(0.0, T_surf=900.0)) is None
    assert lid_flux_cap(_row(1.0, G=0.0, T_surf=900.0)) is None
    assert lid_flux_cap(_row(1.0, T_surf=1400.0)) is None
    with pytest.raises(ValueError, match='needs a skin'):
        lid_flux_cap(_row(0.5, d=0.0, k=0.0, T_surf=900.0))


def _rows(T, F):
    return pd.DataFrame(
        {'Time': [10.0 * (i + 1) for i in range(len(T))], 'T_surf': T, 'F_atm': F}
    )


def test_flux_slope_is_the_slope_of_the_last_two_rows():
    """F' of the lid cap: (F_atm - F_atm before) / (T_surf - T_surf before), with hf_all holding
    the current row or only the rows before it."""
    hf_all = _rows([700.0, 701.0], [15000.0, 15087.0])
    row = {'Time': 20.0, 'T_surf': 701.0, 'F_atm': 15087.0}
    assert lid_flux_slope(hf_all, row) == pytest.approx(87.0, rel=1e-12)
    assert lid_flux_slope(hf_all.iloc[:1], row) == pytest.approx(87.0, rel=1e-12)


@pytest.mark.parametrize(
    ('T', 'F'),
    [
        ([700.0, 700.005], [15000.0, 15087.0]),  # surface step below 0.01 K
        ([700.0, 701.0], [15087.0, 15000.0]),  # flux falls as the surface warms
        ([700.0, 701.0], [15000.0, float('nan')]),
    ],
)
def test_flux_slope_is_zero_without_a_usable_slope(T, F):
    """A surface step below 0.01 K, a slope of 0 or below, or a non-finite value give F' = 0."""
    hf_all = _rows(T, F)
    row = {'Time': 20.0, 'T_surf': T[1], 'F_atm': F[1]}
    assert lid_flux_slope(hf_all, row) == 0.0


def test_flux_slope_is_zero_without_a_row_before():
    """No row before the current one gives F' = 0."""
    row = {'Time': 10.0, 'T_surf': 700.0, 'F_atm': 15000.0}
    assert lid_flux_slope(None, row) == 0.0
    assert lid_flux_slope(_rows([700.0], [15000.0]), row) == 0.0


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_call_flux_is_the_parallel_sum_of_the_lid_and_the_open_surface(validation_plot):
    """Analytical limit: the lid part, weight w, and the open part, weight 1 - w, conduct in
    parallel to one surface, F = w min(G_lid, G_skin) (T_b - T_surf) + (1 - w) G_skin
    (T_node - T_surf). The one skin of lid_boundary, k / d_call (T_melt - T_surf), equals it at
    every T_surf to 1e-12, for G_lid of 1/4, 1 and 2 times G_skin. At w = 1 a skin in series
    with the lid gives 0.8 of it at G_lid = G_skin / 4 and 0.5 at G_lid = G_skin; a lid without
    the cap at G_skin gives 2 times it at G_lid = 2 G_skin."""
    T_b, T_node, d, k = 1400.0, 1500.0, 0.01, 2.0
    G_skin = k / d
    T_surf = np.linspace(300.0, 1390.0, 110)
    weights = (0.0, 0.3, 0.7, 1.0)
    ref, got = {}, {}
    for G_lid in (0.25 * G_skin, G_skin, 2.0 * G_skin):
        for w in weights:
            row = _row(w, G=G_lid, T_b=T_b, T_node=T_node, d=d, k=k)
            # the call passes T_magma unchanged at w = 0; Aragog writes T_node there
            row['T_magma'] = w * T_b + (1.0 - w) * T_node
            d_call, k_call, T_melt = lid_boundary(row)
            F = k_call / d_call * (T_melt - T_surf)
            F_ref = w * min(G_lid, G_skin) * (T_b - T_surf) + (1.0 - w) * G_skin * (
                T_node - T_surf
            )
            np.testing.assert_allclose(F, F_ref, rtol=1e-12)
            ref[G_lid, w], got[G_lid, w] = F_ref, F
    series = {
        G: 1.0 / (1.0 / G + 1.0 / G_skin) * (T_b - T_surf) for G in (0.25 * G_skin, G_skin)
    }
    np.testing.assert_allclose(series[0.25 * G_skin] / ref[0.25 * G_skin, 1.0], 0.8, rtol=1e-12)
    np.testing.assert_allclose(series[G_skin] / ref[G_skin, 1.0], 0.5, rtol=1e-12)
    uncapped = 2.0 * G_skin * (T_b - T_surf)
    np.testing.assert_allclose(uncapped / ref[2.0 * G_skin, 1.0], 2.0, rtol=1e-12)
    # the lines are straight in T_surf, so their end points draw them exactly
    G_plot, x, ends = 0.25 * G_skin, slice(None, None, 20), [0, -1]

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, w in enumerate(weights):
            ax.plot(
                T_surf[ends],
                ref[G_plot, w][ends] / 1e3,
                color=f'C{i}',
                label=f'w = {w:g}',
            )
            ax.plot(T_surf[x], got[G_plot, w][x] / 1e3, 'o', color=f'C{i}', ms=5)
        ax.plot(
            T_surf[ends],
            series[G_plot][ends] / 1e3,
            '--',
            color='C4',
            label='series, w = 1',
        )
        ink = plt.rcParams['text.color']
        ax.plot([], [], 'o', color=ink, ms=5, label='code')
        ax.set_xlabel('Surface temperature $T_\\mathrm{surf}$ [K]')
        ax.set_ylabel('$F$ [kW m$^{-2}$]')
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_parallel_sum')


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_weighted_parts_sum_to_f_atm_less_the_unscaled_cap(validation_plot):
    """Analytical limit: with r = F_atm over the skin balance of the call, the lid part at its
    cap C = min(G_lid, G_skin) (T_b - T_surf) and the open part sum to w C + (1 - w) F_open =
    F_atm - w C (r - 1), so to F_atm where the atmosphere closes its balance (r = 1). C and the
    balance are computed here from their formulas. Checked to 1e-12 for w from 0 to 1 at
    r = 0.95, 1 and 1.06, with G_lid = G_skin / 2 and 2 G_skin. At w = 0.5 and G_lid = G_skin / 2
    an open part equal to F_atm misses F_atm by 21 % at r = 1; an open part without the scale
    misses F_atm by 5.7 % and the closed form by 4.0 % at r = 1.06 (the closed form is 1.7 %
    below F_atm) and is not told apart at r = 1."""
    T_b, T_node, T_surf, G_skin = 1400.0, 1500.0, 900.0, 200.0
    w_grid = np.linspace(0.0, 1.0, 21)
    gaps = (0.95, 1.0, 1.06)
    ratio, closed = {}, {}
    for G_lid in (0.5 * G_skin, 2.0 * G_skin):
        G_cap = min(G_lid, G_skin)
        C_ref = G_cap * (T_b - T_surf)
        S = G_skin * (T_node - T_surf)
        for r in gaps:
            got, ref = [], []
            for w in w_grid:
                F_atm = r * (w * C_ref + (1.0 - w) * S)
                row = _row(w, G=G_lid, T_b=T_b, T_node=T_node, T_surf=T_surf, F_atm=F_atm)
                C = lid_flux_cap(row) or 0.0
                if w > 0.0:
                    assert C == pytest.approx(C_ref, rel=1e-14)
                total = w * C + (1.0 - w) * lid_open_flux(row)
                got.append(total / F_atm)
                ref.append(1.0 - w * C_ref * (r - 1.0) / F_atm)
            ratio[G_lid, r], closed[G_lid, r] = np.array(got), np.array(ref)
            np.testing.assert_allclose(ratio[G_lid, r], closed[G_lid, r], rtol=1e-12)
        np.testing.assert_allclose(ratio[G_lid, 1.0], 1.0, rtol=1e-12)
    C_h, S = 100.0 * 500.0, G_skin * 600.0
    F1, F6 = 0.5 * C_h + 0.5 * S, 1.06 * (0.5 * C_h + 0.5 * S)
    assert 1.0 - (0.5 * C_h + 0.5 * F1) / F1 == pytest.approx(0.206, abs=0.001)
    unscaled = (0.5 * C_h + 0.5 * S) / F6
    assert 1.0 - unscaled == pytest.approx(0.057, abs=0.001)
    assert closed[100.0, 1.06][10] - unscaled == pytest.approx(0.040, abs=0.001)
    assert 1.0 - ratio[100.0, 1.06][10] == pytest.approx(0.017, abs=0.001)
    ratio = {r: ratio[100.0, r] for r in gaps}
    closed = {r: closed[100.0, r] for r in gaps}

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, r in enumerate(gaps):
            ax.plot(w_grid, closed[r], color=f'C{i}', label=f'r = {r:g}')
            ax.plot(w_grid[::2], ratio[r][::2], 'o', color=f'C{i}', ms=5)
        ink = plt.rcParams['text.color']
        ax.plot([], [], 'o', color=ink, ms=5, label='code')
        ax.set_xlabel('Lid weight $w$')
        ax.set_ylabel('sum / $F_\\mathrm{atm}$')
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_open_part_sum')
