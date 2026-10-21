"""Unit tests for ``proteus.plot.cpl_chem_atmosphere``.

Covers the atmospheric chemistry plotting functions including aerosol and cloud
profiles. All matplotlib rendering, file I/O, NetCDF reads, and filesystem
operations are mocked for fast unit testing (<100 ms).

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.plot.cpl_chem_atmosphere import _aerosol_gas, plot_chem_atmosphere
from proteus.utils.plot import _preset_colours, latexify

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
def test_plot_chem_atmosphere_no_files(mock_atm_times):
    """
    plot_chem_atmosphere returns early when no NetCDF files found.

    Physical scenario: simulation hasn't run yet or output was deleted.
    """
    mock_atm_times.return_value = []

    # Should return without error
    result = plot_chem_atmosphere('output_dir', 'vulcan', plot_format='png')
    assert result is None  # empty-file-list branch must yield None silently
    # Discriminating check: the snapshot lookup was actually attempted; a regression
    # that short-circuited before checking for files would have left the mock
    # uncalled.
    mock_atm_times.assert_called()


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_basic(mock_join, mock_atm_times, mock_read, mock_subplots):
    """
    Test basic plot generation with gas species only.

    Physical scenario: plotting atmospheric composition with major volatiles
    (H2O, CO2, N2, etc.) from AGNI NetCDF output.
    """
    # Mock file discovery
    mock_atm_times.return_value = [100.0]
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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_with_clouds(mock_join, mock_atm_times, mock_read, mock_subplots):
    """
    Test plot generation including cloud profiles.

    Physical scenario: water clouds form in the atmosphere when cloud_enabled=True.
    The plot should show cloud mass mixing ratio and area fraction in panel 2.
    """
    mock_atm_times.return_value = [500.0]
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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_with_aerosols(
    mock_join, mock_atm_times, mock_read, mock_subplots
):
    """
    Test plot generation including aerosol profiles with species names.

    Physical scenario: aerosols (e.g., sulfate, silicate from volcanic outgassing)
    are present when aerosols_enabled=True. Should appear in panel 2 with proper
    labels for each aerosol species.
    """
    mock_atm_times.return_value = [1000.0]
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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
@patch('proteus.plot.cpl_chem_atmosphere.os.path.join')
def test_plot_chem_atmosphere_with_offchem(
    mock_join, mock_atm_times, mock_read, mock_read_result, mock_subplots
):
    """
    Test plot with offline chemistry results.

    Physical scenario: VULCAN or other chemistry module computes detailed
    kinetics; results are overlaid as solid lines on top of equilibrium
    chemistry (dashed lines).
    """
    mock_atm_times.return_value = [100.0]
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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
def test_plot_chem_atmosphere_temperature_overlay(mock_atm_times, mock_read, mock_subplots):
    """
    Test that temperature profiles are added to both panels.

    Physical scenario: temperature provides critical context for interpreting
    gas abundances (condensation fronts) and cloud/aerosol distributions.
    """
    mock_atm_times.return_value = [100.0]

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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
def test_plot_chem_atmosphere_time_annotation(mock_atm_times, mock_read, mock_subplots):
    """
    Test that simulation time is annotated on the plot.

    Physical scenario: tracking atmospheric evolution over time requires
    timestamping each plot.
    """
    # Filename encodes time: 5000_atm.nc = 5000 years
    mock_atm_times.return_value = [5000.0]

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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
def test_plot_chem_atmosphere_clouds_and_aerosols(mock_atm_times, mock_read, mock_subplots):
    """
    Test plotting with both clouds and aerosols present.

    Physical scenario: Atmosphere with both water clouds and aerosol particles
    (e.g., sulfate aerosols in a Venus-like atmosphere with H2SO4 clouds).
    """
    mock_atm_times.return_value = [200.0]

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
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
def test_plot_chem_atmosphere_no_aerosols_key(mock_atm_times, mock_read, mock_subplots):
    """
    Test plotting when aerosols key is missing (backward compatibility).

    Physical scenario: Reading output from older simulation without aerosol
    support. Should handle gracefully and plot other data.
    """
    mock_atm_times.return_value = [100.0]

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


@pytest.mark.unit
@patch('proteus.plot.cpl_chem_atmosphere.plt.subplots')
@patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile')
@patch('proteus.plot.cpl_chem_atmosphere.list_atmosphere_times')
def test_agni_aerosols_take_the_colour_and_name_of_their_gas(
    mock_atm_times, mock_read, mock_subplots, caplog
):
    """An aerosol named as a gas, in any case, is drawn as that gas, with no fall-back."""
    gases = ('Fe', 'FeO', 'MgO', 'NH3', 'SiO', 'SiO2')
    profile = {
        'pl': np.array([1e8, 1e7, 1e6, 1e5]),
        'tmpl': np.array([3000, 2000, 1000, 500]),
        'aerosols': np.array([g.lower() for g in gases[:-1]] + ['SiO2', 'dustdiv1']),
    }
    for name in profile['aerosols']:
        profile[name + '_mmr'] = np.array([1e-8, 5e-8, 1e-9])
    for g in gases:
        profile[g + '_vmr'] = np.array([1e-3, 1e-4, 1e-5])
    mock_atm_times.return_value = [1000.0]
    mock_read.return_value = profile
    ax1, ax2 = MagicMock(), MagicMock()
    mock_subplots.return_value = (MagicMock(), (ax1, ax2))
    ax1.get_legend_handles_labels.return_value = ([MagicMock()], ['Fe'])

    with caplog.at_level('WARNING'):
        plot_chem_atmosphere(
            'output_dir', None, plot_format='png', plot_gases=['Fe'], plot_offchem=False
        )

    assert len(ax2.plot.call_args_list) == 7
    drawn = {c.kwargs['label']: c.kwargs['color'] for c in ax2.plot.call_args_list}
    want = {latexify(g): _preset_colours[g] for g in gases}
    assert drawn == want | {'dustdiv1': _preset_colours['dustdiv1']}
    assert 'SiO$_2$' in drawn
    assert 'fallback colour' not in caplog.text


@pytest.mark.unit
def test_an_aerosol_matches_one_gas_apart_from_case_or_none():
    """Two gases that differ only in case, or no gas, give no match."""
    assert _aerosol_gas('co', ['CO', 'Co', 'H2O']) is None
    assert _aerosol_gas('co', ['CO', 'H2O']) == 'CO'
    assert _aerosol_gas('Feo', ['CO', 'FeO']) == 'FeO'
    assert _aerosol_gas('soot', ['CO', 'H2O']) is None
    assert _aerosol_gas('soot', []) is None


# ---------------------------------------------------------------------------
# Snapshot selection: final chemistry snapshot by default, nearest with `time`
# ---------------------------------------------------------------------------

# Atmosphere snapshots at 10, 20 and 30.5 yr; chemistry only at 10 and 20 yr, so
# the latest atmosphere snapshot (30.5 yr) has no chemistry partner.
_ATM_TOKENS = ('10p000', '20p000', '30p500')
_CHEM_TOKENS = ('10p000', '20p000')


def _profile():
    """A three-level AGNI profile, as read_ncdf_profile returns it [Pa, K]."""
    return {
        'pl': np.array([1e3, 1e5, 1e7]),
        'tmpl': np.array([300.0, 900.0, 2000.0]),
        'H2O_vmr': np.array([1e-3, 1e-4]),
    }


def _snapshot_dir(tmp_path, chem_tokens=_CHEM_TOKENS, legacy=False):
    """Create empty atmosphere snapshots and chemistry CSVs under tmp_path.

    Each chemistry CSV holds an H2O mixing ratio that encodes its time, so the
    file read can be identified from its content.
    """
    (tmp_path / 'data').mkdir()
    (tmp_path / 'offchem').mkdir()
    (tmp_path / 'plots').mkdir()
    for token in _ATM_TOKENS:
        (tmp_path / 'data' / f'{token}_atm.nc').write_text('')
    names = [f'vulcan_{token}.csv' for token in chem_tokens]
    if legacy:
        names = ['vulcan.csv']
    for name in names:
        (tmp_path / 'offchem' / name).write_text(
            'tmp p z Kzz H2O\n300 1e3 1e4 1e5 1e-6\n2000 1e7 0 1e5 1e-6\n'
        )
    return str(tmp_path)


def _run_plot(outdir, **kwargs):
    """Run plot_chem_atmosphere with the profile reader and the figure mocked.

    Returns the path given to read_ncdf_profile, the filename given to
    read_result (None for the default name), and the time annotation text.
    """
    from proteus.atmos_chem.common import read_result

    fig, ax1, ax2 = MagicMock(), MagicMock(), MagicMock()
    ax1.get_legend_handles_labels.return_value = ([MagicMock()], ['H2O'])
    with (
        patch('proteus.plot.cpl_chem_atmosphere.plt.subplots', return_value=(fig, (ax1, ax2))),
        patch(
            'proteus.plot.cpl_chem_atmosphere.read_ncdf_profile', return_value=_profile()
        ) as mock_read,
        patch('proteus.plot.cpl_chem_atmosphere.read_result', wraps=read_result) as mock_rr,
    ):
        plot_chem_atmosphere(outdir, 'vulcan', plot_format='png', **kwargs)
    nc_path = mock_read.call_args[0][0]
    rr_name = mock_rr.call_args.kwargs.get('filename') if mock_rr.called else 'not called'
    annotation = ax1.text.call_args[0][2]
    return nc_path, rr_name, annotation


@pytest.mark.unit
def test_plot_chem_atmosphere_defaults_to_final_chemistry_snapshot(tmp_path):
    """Without `time`, the final chemistry snapshot (20 yr) is drawn with the
    atmosphere snapshot at the same time, not the latest one (30.5 yr)."""
    nc_path, rr_name, annotation = _run_plot(_snapshot_dir(tmp_path))
    assert rr_name == 'vulcan_20p000.csv'
    assert nc_path.endswith('20p000_atm.nc')
    assert '2.00e+01' in annotation


@pytest.mark.unit
@pytest.mark.parametrize(
    ('time', 'token'),
    [(14.0, '10p000'), (16.0, '20p000'), (1.0e6, '20p000')],
    ids=['nearer_earlier_snapshot', 'nearer_later_snapshot', 'beyond_last_snapshot'],
)
def test_plot_chem_atmosphere_time_picks_nearest_chemistry_snapshot(tmp_path, time, token):
    """A requested time between two chemistry snapshots draws the nearer one,
    and the atmosphere snapshot read is the one at that chemistry time."""
    nc_path, rr_name, _ = _run_plot(_snapshot_dir(tmp_path), time=time)
    assert rr_name == f'vulcan_{token}.csv'
    assert nc_path.endswith(f'{token}_atm.nc')
    # The 30.5 yr atmosphere snapshot has no chemistry and is never chosen
    assert '30p500' not in nc_path


@pytest.mark.unit
def test_plot_chem_atmosphere_without_chemistry_uses_latest_atmosphere(tmp_path):
    """With chemistry off, the latest atmosphere snapshot is drawn, or the one
    nearest `time` when given, and no chemistry file is read."""
    outdir = _snapshot_dir(tmp_path)
    nc_path, rr_name, annotation = _run_plot(outdir, plot_offchem=False)
    assert nc_path.endswith('30p500_atm.nc')
    assert rr_name == 'not called'
    assert '3.05e+01' in annotation

    # 12 yr lies nearest the 10 yr atmosphere snapshot
    nc_path, _, _ = _run_plot(outdir, plot_offchem=False, time=12.0)
    assert nc_path.endswith('10p000_atm.nc')


@pytest.mark.unit
def test_plot_chem_atmosphere_falls_back_to_legacy_fixed_name(tmp_path):
    """Output holding only the fixed-name vulcan.csv is read through the default
    name, drawn with the latest atmosphere snapshot."""
    nc_path, rr_name, _ = _run_plot(_snapshot_dir(tmp_path, legacy=True))
    assert rr_name is None
    assert nc_path.endswith('30p500_atm.nc')


@pytest.mark.unit
def test_plot_chem_atmosphere_returns_early_without_atmosphere_snapshots(tmp_path):
    """Chemistry CSVs without any atmosphere snapshot give no plot and no read."""
    outdir = _snapshot_dir(tmp_path)
    for token in _ATM_TOKENS:
        (tmp_path / 'data' / f'{token}_atm.nc').unlink()
    with patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile') as mock_read:
        result = plot_chem_atmosphere(outdir, 'vulcan', plot_format='png')
    assert result is None
    mock_read.assert_not_called()


@pytest.mark.unit
def test_plot_chem_atmosphere_draws_chemistry_at_its_own_pressures(tmp_path):
    """The chemistry grid is not the climate grid: VULCAN layers are log-spaced
    while AGNI levels crowd towards the surface. Each chemistry profile is drawn
    against the pressures of its own CSV [Pa to bar], not the AGNI levels.

    AGNI levels at 10, 1e3, 3e3 and 1e7 Pa and chemistry layers at 20, 2e4 and
    9e6 Pa share no value, so drawing on the wrong grid shows in every point.
    """
    outdir = _snapshot_dir(tmp_path, chem_tokens=('20p000',))
    (tmp_path / 'offchem' / 'vulcan_20p000.csv').write_text(
        'tmp p z Kzz H2O\n300 2e1 2e5 1e5 1e-3\n900 2e4 1e5 1e5 1e-2\n2000 9e6 0 1e5 1e-1\n'
    )
    profile = {
        'pl': np.array([1e1, 1e3, 3e3, 1e7]),
        'tmpl': np.array([300.0, 500.0, 900.0, 2000.0]),
        'H2O_vmr': np.array([1e-4, 1e-4, 1e-4]),
    }
    fig, ax1, ax2 = MagicMock(), MagicMock(), MagicMock()
    ax1.get_legend_handles_labels.return_value = ([MagicMock()], ['H2O'])
    with (
        patch('proteus.plot.cpl_chem_atmosphere.plt.subplots', return_value=(fig, (ax1, ax2))),
        patch('proteus.plot.cpl_chem_atmosphere.read_ncdf_profile', return_value=profile),
    ):
        plot_chem_atmosphere(outdir, 'vulcan', plot_format='png', plot_gases=['H2O'])

    # Solid lines with data; the legend entry is an empty solid line
    solid = [
        c for c in ax1.plot.call_args_list if c.kwargs.get('ls') == 'solid' and len(c.args[0])
    ]
    assert len(solid) == 1
    x, y = (np.asarray(a, dtype=float) for a in solid[0].args)
    np.testing.assert_allclose(x, [1e-3, 1e-2, 1e-1])
    np.testing.assert_allclose(y, [2e-4, 2e-1, 9e1], rtol=1e-12)
    # Discrimination: not the AGNI level pressures, cut or not to the profile length
    assert not np.allclose(y, profile['pl'][: len(y)] * 1e-5)
