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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.atmos_clim.common import (
    clip_radius_to_hill,
    find_latest_atmosphere_time,
    get_oarr_from_parr,
    get_radius_from_pressure,
    get_spfile_name_and_bands,
    get_spfile_path,
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
def test_spfile_helpers():
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
    # Expected: <fwl_dir>/spectral_files/<group>/<bands>/<group>.sf
    path = get_spfile_path('/fwl/data', mock_conf)
    assert path == '/fwl/data/spectral_files/Dayspring/16/Dayspring.sf'


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
