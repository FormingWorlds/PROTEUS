"""Unit tests for ``proteus.plot.cpl_chem_atmosphere``.

Covers the atmospheric chemistry plotting functions including aerosol and cloud
profiles. All matplotlib rendering, file I/O, NetCDF reads, and filesystem
operations are mocked for fast unit testing (<100 ms).

Testing standards:
  - docs/test_infrastructure.md
  - docs/test_categorization.md
  - docs/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.plot.cpl_chem_atmosphere import plot_chem_atmosphere


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
def test_plot_chem_atmosphere_no_files(mock_glob):
    """
    plot_chem_atmosphere returns early when no NetCDF files found.

    Physical scenario: simulation hasn't run yet or output was deleted.
    """
    mock_glob.return_value = []

    # Should return without error
    result = plot_chem_atmosphere('output_dir', 'vulcan', plot_format='png')
    assert result is None


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_basic(
    mock_join, mock_glob, mock_read, mock_subplots
):
    """
    Test basic plot generation with gas species only.

    Physical scenario: plotting atmospheric composition with major volatiles
    (H2O, CO2, N2, etc.) from AGNI NetCDF output.
    """
    # Mock file discovery
    mock_glob.return_value = ['output_dir/data/100_atm.nc']
    mock_join.return_value = 'output_dir/plots/plot_chem_atmosphere.png'

    # Mock NetCDF read with typical gas species
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6, 1e5]),  # Pa
        'tmpl': np.array([3000, 2000, 1000, 500]),  # K
        'H2O_vmr': np.array([1e-3, 1e-4, 1e-5]),
        'CO2_vmr': np.array([1e-2, 1e-3, 1e-4]),
        'N2_vmr': np.array([0.78, 0.78, 0.78]),
    }

    # Mock matplotlib
    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))

    # Mock twiny for temperature axis
    ax1_temp_mock = MagicMock()
    ax2_temp_mock = MagicMock()
    ax1_mock.twiny.return_value = ax1_temp_mock
    ax2_mock.twiny.return_value = ax2_temp_mock

    # Mock get_legend_handles_labels to return handles for the gas species
    # Need 3 handles for H2O, CO2, N2 (all have nonzero VMR)
    mock_handles = [MagicMock(), MagicMock(), MagicMock()]
    mock_labels = ['H2O', 'CO2', 'N2']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    # Run function
    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify NetCDF was read with correct extra keys
    mock_read.assert_called_once()
    call_kwargs = mock_read.call_args[1]
    assert 'pl' in call_kwargs['extra_keys']
    assert 'tmpl' in call_kwargs['extra_keys']
    assert 'x_gas' in call_kwargs['extra_keys']
    assert 'cloud_mmr' in call_kwargs['extra_keys']
    assert 'aer_mmr' in call_kwargs['extra_keys']

    # Verify two panels were created
    mock_subplots.assert_called_once()
    args, kwargs = mock_subplots.call_args
    assert args[0] == 2  # 2 rows
    assert args[1] == 1  # 1 column

    # Verify temperature axes were created
    ax1_mock.twiny.assert_called_once()
    ax2_mock.twiny.assert_called_once()

    # Verify plot was saved
    fig_mock.savefig.assert_called_once()


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_with_clouds(
    mock_join, mock_glob, mock_read, mock_subplots
):
    """
    Test plot generation including cloud profiles.

    Physical scenario: water clouds form in the atmosphere when cloud_enabled=True.
    The plot should show cloud mass mixing ratio and area fraction in panel 2.
    """
    mock_glob.return_value = ['output_dir/data/500_atm.nc']
    mock_join.return_value = 'output_dir/plots/plot_chem_atmosphere.png'

    # Mock NetCDF with cloud data
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6, 1e5]),
        'tmpl': np.array([3000, 2000, 1000, 500]),
        'H2O_vmr': np.array([1e-3, 1e-4, 1e-5]),
        'cloud_mmr': np.array([1e-6, 5e-6, 1e-7]),  # Non-zero clouds
        'cloud_area': np.array([0.0, 0.5, 0.1]),
    }

    # Mock matplotlib
    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))
    ax1_mock.twiny.return_value = MagicMock()
    ax2_mock.twiny.return_value = MagicMock()

    # Mock get_legend_handles_labels - only H2O has nonzero VMR
    mock_handles = [MagicMock()]
    mock_labels = ['H2O']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify second panel (ax2) was used for cloud plotting
    assert ax2_mock.plot.called
    assert ax2_mock.set_xlabel.called
    assert ax2_mock.set_ylabel.called


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_with_aerosols(
    mock_join, mock_glob, mock_read, mock_subplots
):
    """
    Test plot generation including aerosol profiles with species names.

    Physical scenario: aerosols (e.g., sulfate, silicate from volcanic outgassing)
    are present when aerosols_enabled=True. Should appear in panel 2 with proper
    labels for each aerosol species.
    """
    mock_glob.return_value = ['output_dir/data/1000_atm.nc']
    mock_join.return_value = 'output_dir/plots/plot_chem_atmosphere.png'

    # Mock NetCDF with multiple aerosol species
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6, 1e5]),
        'tmpl': np.array([3000, 2000, 1000, 500]),
        'H2O_vmr': np.array([1e-3, 1e-4, 1e-5]),
        'cloud_mmr': np.array([0.0, 0.0, 0.0]),
        'aerosols': ['Sulfate', 'Silicate'],  # Two aerosol species
        'Sulfate_mmr': np.array([1e-8, 5e-8, 1e-9]),  # Individual aerosol MMRs
        'Silicate_mmr': np.array([2e-9, 3e-9, 1e-10]),
    }

    # Mock matplotlib
    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))
    ax1_mock.twiny.return_value = MagicMock()
    ax2_mock.twiny.return_value = MagicMock()

    # Mock get_legend_handles_labels - only H2O has nonzero VMR
    mock_handles = [MagicMock()]
    mock_labels = ['H2O']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify aerosol plotting on second panel
    assert ax2_mock.plot.called
    assert ax2_mock.legend.called

    # Verify both aerosol species were plotted (2 species + maybe temperature)
    # Each aerosol calls plot once
    assert ax2_mock.plot.call_count >= 2


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_result')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_with_offchem(
    mock_join, mock_glob, mock_read, mock_read_result, mock_subplots
):
    """
    Test plot with offline chemistry results.

    Physical scenario: VULCAN or other chemistry module computes detailed
    kinetics; results are overlaid as solid lines on top of equilibrium
    chemistry (dashed lines).
    """
    mock_glob.return_value = ['output_dir/data/100_atm.nc']
    mock_join.return_value = 'output_dir/plots/plot_chem_atmosphere.png'

    # Mock NetCDF read
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6]),
        'tmpl': np.array([3000, 2000, 1000]),
        'H2O_vmr': np.array([1e-3, 1e-4]),
        'cloud_mmr': np.array([0.0, 0.0]),
        'cloud_area': np.array([0.0, 0.0]),
    }

    # Mock offline chemistry data
    import pandas as pd

    mock_read_result.return_value = pd.DataFrame(
        {
            'H2O': [1e-2, 1e-3],
            'CO2': [1e-3, 1e-4],
            'tmp': [3000, 2000],
            'p': [1e8, 1e7],
            'z': [0, 1e5],
            'Kzz': [1e5, 1e6],
        }
    )

    # Mock matplotlib
    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))
    ax1_mock.twiny.return_value = MagicMock()
    ax2_mock.twiny.return_value = MagicMock()

    # Mock get_legend_handles_labels - H2O and CO2 have nonzero VMR
    mock_handles = [MagicMock(), MagicMock()]
    mock_labels = ['H2O', 'CO2']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    plot_chem_atmosphere('output_dir', 'vulcan', plot_format='png', plot_offchem=True)

    # Verify offline chem was read
    mock_read_result.assert_called_once()

    # Verify plot was saved
    fig_mock.savefig.assert_called_once()


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
def test_plot_chem_atmosphere_temperature_overlay(
    mock_glob, mock_read, mock_subplots
):
    """
    Test that temperature profiles are added to both panels.

    Physical scenario: temperature provides critical context for interpreting
    gas abundances (condensation fronts) and cloud/aerosol distributions.
    """
    mock_glob.return_value = ['output_dir/data/100_atm.nc']

    # Mock NetCDF with temperature gradient
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6, 1e5]),
        'tmpl': np.array([3500, 2000, 800, 300]),  # Strong gradient
        'H2O_vmr': np.array([1e-3, 1e-4, 1e-5]),
        'cloud_mmr': np.array([0.0, 0.0, 0.0]),
        'cloud_area': np.array([0.0, 0.0, 0.0]),
    }

    # Mock matplotlib
    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))

    ax1_temp_mock = MagicMock()
    ax2_temp_mock = MagicMock()
    ax1_mock.twiny.return_value = ax1_temp_mock
    ax2_mock.twiny.return_value = ax2_temp_mock

    # Mock get_legend_handles_labels - only H2O has nonzero VMR
    mock_handles = [MagicMock()]
    mock_labels = ['H2O']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify temperature was plotted on both panels
    ax1_temp_mock.plot.assert_called_once()
    ax2_temp_mock.plot.assert_called_once()

    # Verify temperature axis was labeled on panel 1
    ax1_temp_mock.set_xlabel.assert_called_once()
    # Panel 2 temperature axis has no label (xticks hidden)


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
def test_plot_chem_atmosphere_time_annotation(
    mock_glob, mock_read, mock_subplots
):
    """
    Test that simulation time is annotated on the plot.

    Physical scenario: tracking atmospheric evolution over time requires
    timestamping each plot.
    """
    # Filename encodes time: 5000_atm.nc = 5000 years
    mock_glob.return_value = ['output_dir/data/5000_atm.nc']

    mock_read.return_value = {
        'pl': np.array([1e8, 1e7]),
        'tmpl': np.array([3000, 1000]),
        'H2O_vmr': np.array([1e-3]),
        'cloud_mmr': np.array([0.0]),
        'cloud_area': np.array([0.0]),
    }

    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))
    ax1_mock.twiny.return_value = MagicMock()
    ax2_mock.twiny.return_value = MagicMock()

    # Mock get_legend_handles_labels - only H2O has nonzero VMR
    mock_handles = [MagicMock()]
    mock_labels = ['H2O']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify time annotation was added to ax1 (not fig)
    ax1_mock.text.assert_called_once()
    call_args = ax1_mock.text.call_args[0]
    # Should contain the year value (5000)
    assert '5.00e+03' in call_args[2] or '5000' in str(call_args)


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
def test_plot_chem_atmosphere_clouds_and_aerosols(
    mock_glob, mock_read, mock_subplots
):
    """
    Test plotting with both clouds and aerosols present.

    Physical scenario: Atmosphere with both water clouds and aerosol particles
    (e.g., sulfate aerosols in a Venus-like atmosphere with H2SO4 clouds).
    """
    mock_glob.return_value = ['output_dir/data/200_atm.nc']

    # Mock with both cloud and aerosol data
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6, 1e5]),
        'tmpl': np.array([3000, 2000, 1000, 500]),
        'H2O_vmr': np.array([1e-3, 1e-4, 1e-5]),
        'CO2_vmr': np.array([1e-2, 1e-3, 1e-4]),
        'cloud_mmr': np.array([1e-6, 5e-6, 1e-7]),  # Cloud present
        'aerosols': ['Sulfate'],
        'Sulfate_mmr': np.array([1e-8, 2e-8, 5e-9]),  # Aerosol present
    }

    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))
    ax1_mock.twiny.return_value = MagicMock()
    ax2_mock.twiny.return_value = MagicMock()

    # Mock get_legend_handles_labels - H2O and CO2 have nonzero VMR
    mock_handles = [MagicMock(), MagicMock()]
    mock_labels = ['H2O', 'CO2']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify both panels were plotted
    assert ax1_mock.plot.called  # Gas species
    assert ax2_mock.plot.called  # Clouds + aerosols

    # Panel 2 should have plots for cloud + aerosol (at least 2)
    assert ax2_mock.plot.call_count >= 2


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.glob.glob')
def test_plot_chem_atmosphere_no_aerosols_key(
    mock_glob, mock_read, mock_subplots
):
    """
    Test plotting when aerosols key is missing (backward compatibility).

    Physical scenario: Reading output from older simulation without aerosol
    support. Should handle gracefully and plot other data.
    """
    mock_glob.return_value = ['output_dir/data/100_atm.nc']

    # Mock without aerosols key (older output format)
    mock_read.return_value = {
        'pl': np.array([1e8, 1e7, 1e6]),
        'tmpl': np.array([3000, 2000, 1000]),
        'H2O_vmr': np.array([1e-3, 1e-4]),
        'cloud_mmr': np.array([0.0, 0.0]),
        # No 'aerosols' key at all
    }

    fig_mock = MagicMock()
    ax1_mock = MagicMock()
    ax2_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))
    ax1_mock.twiny.return_value = MagicMock()
    ax2_mock.twiny.return_value = MagicMock()

    # Mock get_legend_handles_labels - only H2O has nonzero VMR
    mock_handles = [MagicMock()]
    mock_labels = ['H2O']
    ax1_mock.get_legend_handles_labels.return_value = (mock_handles, mock_labels)

    # Should not crash
    plot_chem_atmosphere('output_dir', None, plot_format='png', plot_offchem=False)

    # Verify gas panel was still plotted
    assert ax1_mock.plot.called
    fig_mock.savefig.assert_called_once()
