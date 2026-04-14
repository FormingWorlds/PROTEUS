"""
Unit tests for proteus.atmos_clim.agni module.

This module tests the AGNI atmosphere interface including:
- Aerosol discovery (_determine_aerosols)
- Condensate species determination (_determine_condensates)
- AGNI atmosphere initialization (init_agni_atmos)

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from proteus.atmos_clim.agni import _determine_aerosols, _determine_condensates


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_success(mock_isdir, mock_listdir):
    """
    Test aerosol discovery when scattering data directory exists.

    Physical scenario: Scattering data for aerosols (e.g., sulfate, silicate)
    is available in FWL_DATA/scattering/scattering/*.mon files.
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = [
        'Sulfate.mon',
        'Silicate.mon',
        'Haze.mon',
        'other_file.txt',  # Should be ignored
        'readme.md',  # Should be ignored
    ]

    dirs = {'fwl': '/fake/fwl/path'}
    aerosols = _determine_aerosols(dirs)

    # Verify correct aerosols found and sorted
    assert len(aerosols) == 3
    assert aerosols == ['Haze', 'Silicate', 'Sulfate']  # alphabetically sorted

    # Verify correct directory was checked
    mock_isdir.assert_called_once_with('/fake/fwl/path/scattering/scattering')


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_missing_directory(mock_isdir):
    """
    Test aerosol discovery when scattering directory doesn't exist.

    Physical scenario: FWL_DATA not properly downloaded or scattering
    data not installed. Should return empty list and warn.
    """
    mock_isdir.return_value = False

    dirs = {'fwl': '/nonexistent/path'}
    aerosols = _determine_aerosols(dirs)

    # Should return empty list without crashing
    assert aerosols == []
    mock_isdir.assert_called_once()


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_empty_directory(mock_isdir, mock_listdir):
    """
    Test aerosol discovery when directory exists but has no .mon files.

    Physical scenario: Scattering directory present but empty or only
    contains non-aerosol files.
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = ['readme.txt', 'config.yaml']

    dirs = {'fwl': '/path/to/fwl'}
    aerosols = _determine_aerosols(dirs)

    # Should return empty list
    assert aerosols == []


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_single_species(mock_isdir, mock_listdir):
    """
    Test aerosol discovery with only one aerosol type.

    Physical scenario: Limited scattering data with only one aerosol species
    available (e.g., only sulfate aerosols).
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = ['Sulfate.mon']

    dirs = {'fwl': '/path/to/fwl'}
    aerosols = _determine_aerosols(dirs)

    assert len(aerosols) == 1
    assert aerosols == ['Sulfate']


@pytest.mark.unit
def test_determine_condensates():
    """
    Test condensate species determination from volatile list.

    Physical scenario: Given a list of volatile species, filter out
    those that are always dry (H2, N2, CO) to get condensable
    species like H2O, CO2, He, CH4, etc.
    """
    # Test with mixed list of dry and condensable species
    vol_list = ['H2O', 'CO2', 'N2', 'CH4', 'He', 'H2', 'CO']
    condensates = _determine_condensates(vol_list)

    # N2, H2, CO should be filtered out (always dry)
    assert 'H2O' in condensates
    assert 'CO2' in condensates
    assert 'CH4' in condensates
    assert 'He' in condensates  # He is condensable in AGNI
    assert 'N2' not in condensates
    assert 'H2' not in condensates
    assert 'CO' not in condensates


@pytest.mark.unit
def test_determine_condensates_all_dry():
    """
    Test condensate determination with only dry species.

    Physical scenario: Hydrogen-nitrogen-CO dominated atmosphere with no
    condensable species.
    """
    vol_list = ['H2', 'N2', 'CO']
    condensates = _determine_condensates(vol_list)

    # Should return empty list
    assert condensates == []


@pytest.mark.unit
def test_determine_condensates_all_condensable():
    """
    Test condensate determination with all condensable species.

    Physical scenario: Rocky planet atmosphere with water, CO2, and other
    condensable volatiles but no hydrogen/helium.
    """
    vol_list = ['H2O', 'CO2', 'NH3', 'CH4', 'SO2']
    condensates = _determine_condensates(vol_list)

    # All should remain
    assert len(condensates) == len(vol_list)
    assert set(condensates) == set(vol_list)


@pytest.mark.unit
def test_determine_condensates_empty_list():
    """
    Test condensate determination with empty volatile list.

    Physical scenario: Edge case where no volatiles are specified.
    """
    condensates = _determine_condensates([])
    assert condensates == []
