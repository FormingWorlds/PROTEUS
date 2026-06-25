"""Unit tests for ``proteus.plot.cpl_spectra``.

Covers ``plot_spectra`` (both-files-missing skip, single-file load
fallback, full-path two-panel plot) and ``plot_spectra_entry``.
Matplotlib + the read_transit / read_eclipse helpers are mocked.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import proteus.plot.cpl_spectra as spectra_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum_df(n: int = 100, label: str = 'all') -> pd.DataFrame:
    """Build a synthetic transit / eclipse spectrum DataFrame.

    Has the canonical 'Wavelength/um' column plus one or more flux
    columns (the function plots one line per non-Wavelength column).
    Label 'None' marks the full-atmosphere line (drawn in black,
    zorder=6); other labels are coloured.
    """
    wl = np.logspace(np.log10(0.5), np.log10(20.0), n)
    df = pd.DataFrame({'Wavelength/um': wl})
    df[f'depth_{label}/ppm'] = 100 + 50 * np.sin(2 * np.pi * np.log(wl))
    return df


def _install_mock_plt(monkeypatch):
    """Patch the module's ``plt`` binding; return mocks for the 2-panel
    subplot layout (fig, (axt, axb)).
    """
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_axt = MagicMock()
    mock_axb = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [mock_axt, mock_axb])
    monkeypatch.setattr(spectra_mod, 'plt', mock_plt)
    return mock_plt, mock_fig, mock_axt, mock_axb


# ---------------------------------------------------------------------------
# plot_spectra: missing-file behaviour
# ---------------------------------------------------------------------------


def test_plot_spectra_returns_early_when_both_files_absent(monkeypatch, tmp_path):
    """When both transit and eclipse readers raise FileNotFoundError, the
    function logs warnings for each and returns without creating a
    figure. Discrimination: subplots must not be called.
    """
    mock_plt, _, _, _ = _install_mock_plt(monkeypatch)
    monkeypatch.setattr(spectra_mod, 'read_transit', MagicMock(side_effect=FileNotFoundError()))
    monkeypatch.setattr(spectra_mod, 'read_eclipse', MagicMock(side_effect=FileNotFoundError()))

    result = spectra_mod.plot_spectra(output_dir=str(tmp_path))

    assert result is None
    assert mock_plt.subplots.call_count == 0


# ---------------------------------------------------------------------------
# plot_spectra: partial-data path (only transit)
# ---------------------------------------------------------------------------


def test_plot_spectra_proceeds_with_only_transit_available(monkeypatch, tmp_path):
    """When only the transit file is present (eclipse raises FileNotFound),
    the function still produces the figure with the transit panel
    populated. Discrimination: at least one plot line per transit-column
    is drawn on the top axis.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    transit_df = _make_spectrum_df(label='all')
    monkeypatch.setattr(spectra_mod, 'read_transit', MagicMock(return_value=transit_df))
    monkeypatch.setattr(spectra_mod, 'read_eclipse', MagicMock(side_effect=FileNotFoundError()))

    mock_plt, mock_fig, mock_axt, mock_axb = _install_mock_plt(monkeypatch)

    spectra_mod.plot_spectra(output_dir=str(tmp_path), plot_format='pdf')

    assert mock_plt.subplots.call_count == 1
    # Discrimination: transit panel has one line (one non-wavelength column);
    # eclipse panel has none because df_eclipse is None.
    assert mock_axt.plot.call_count == 1
    assert mock_axb.plot.call_count == 0
    assert mock_fig.savefig.call_count == 1


def test_plot_spectra_full_path_both_panels_populated(monkeypatch, tmp_path):
    """When both transit and eclipse data are present, both panels get
    one plot line per non-Wavelength column. With one flux column each,
    that gives 1+1 = 2 total plot calls across the two axes.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    transit_df = _make_spectrum_df(label='all')
    eclipse_df = _make_spectrum_df(label='no_H2O')
    monkeypatch.setattr(spectra_mod, 'read_transit', MagicMock(return_value=transit_df))
    monkeypatch.setattr(spectra_mod, 'read_eclipse', MagicMock(return_value=eclipse_df))

    mock_plt, mock_fig, mock_axt, mock_axb = _install_mock_plt(monkeypatch)

    spectra_mod.plot_spectra(
        output_dir=str(tmp_path),
        plot_format='pdf',
        wlmin=0.5,
        wlmax=20.0,
        source='profile',
    )

    # Discrimination: both panels get exactly one line each
    assert mock_axt.plot.call_count == 1
    assert mock_axb.plot.call_count == 1
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_spectra_profile.pdf')


def test_plot_spectra_uses_source_in_output_filename(monkeypatch, tmp_path):
    """Saved plot filename should include the selected atmospheric source.

    New naming contract is plot_spectra_{source}.{ext}.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    transit_df = _make_spectrum_df(label='all')
    eclipse_df = _make_spectrum_df(label='all')
    monkeypatch.setattr(spectra_mod, 'read_transit', MagicMock(return_value=transit_df))
    monkeypatch.setattr(spectra_mod, 'read_eclipse', MagicMock(return_value=eclipse_df))

    mock_plt, mock_fig, _, _ = _install_mock_plt(monkeypatch)

    spectra_mod.plot_spectra(output_dir=str(tmp_path), plot_format='png', source='offchem')

    assert mock_plt.subplots.call_count == 1
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_spectra_offchem.png')


def test_plot_spectra_full_path_label_none_special_styling(monkeypatch, tmp_path):
    """Columns whose extracted label is 'None' are drawn in black with
    zorder=6 (the full-atmosphere reference). Other labels use coloured
    lines with zorder=4. Discrimination: with one 'None' column and one
    other column, exactly one plot call uses color='black'.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    # Build a transit df with both a 'None' column (reference) and a 'CO2' column
    wl = np.logspace(np.log10(0.5), np.log10(20.0), 50)
    df = pd.DataFrame({'Wavelength/um': wl})
    df['depth_None/ppm'] = 100 + 50 * np.sin(2 * np.pi * np.log(wl))
    df['depth_CO2/ppm'] = 80 + 40 * np.sin(2 * np.pi * np.log(wl))
    monkeypatch.setattr(spectra_mod, 'read_transit', MagicMock(return_value=df))
    monkeypatch.setattr(spectra_mod, 'read_eclipse', MagicMock(side_effect=FileNotFoundError()))

    mock_plt, mock_fig, mock_axt, _ = _install_mock_plt(monkeypatch)

    spectra_mod.plot_spectra(output_dir=str(tmp_path))

    # Discrimination: one of the two plot calls must use color='black'
    # (the 'None' reference), the other a non-black color.
    colors = [call.kwargs.get('color') for call in mock_axt.plot.call_args_list]
    assert colors.count('black') == 1
    assert any(c != 'black' for c in colors)


def test_plot_spectra_removed_species_columns_overlay(monkeypatch, tmp_path):
    """Removed-species columns (<SPECIES>_removed/ppm) should be plotted as
    separate overlays with species-specific labels/colors (not collapsed to
    a generic 'removed' label).
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()

    wl = np.logspace(np.log10(0.5), np.log10(20.0), 50)
    df = pd.DataFrame({'Wavelength/um': wl})
    df['None/ppm'] = 100 + 50 * np.sin(2 * np.pi * np.log(wl))
    df['H2O_removed/ppm'] = 95 + 45 * np.sin(2 * np.pi * np.log(wl))
    df['CO2_removed/ppm'] = 90 + 40 * np.sin(2 * np.pi * np.log(wl))

    monkeypatch.setattr(spectra_mod, 'read_transit', MagicMock(return_value=df))
    monkeypatch.setattr(spectra_mod, 'read_eclipse', MagicMock(side_effect=FileNotFoundError()))

    _, _, mock_axt, _ = _install_mock_plt(monkeypatch)

    spectra_mod.plot_spectra(output_dir=str(tmp_path))

    # Baseline + two removed overlays.
    assert mock_axt.plot.call_count == 3

    labels = [call.kwargs.get('label') for call in mock_axt.plot.call_args_list]
    assert 'H$_2$O' in labels
    assert 'CO$_2$' in labels
    assert 'removed' not in labels


# ---------------------------------------------------------------------------
# plot_spectra_entry: wrapper
# ---------------------------------------------------------------------------


def test_plot_spectra_entry_forwards_handler_attributes(monkeypatch):
    """The entry wrapper extracts output dir + plot_fmt from the handler
    and forwards them to plot_spectra. A regression that dropped the
    plot_format would default to 'pdf' instead of the configured value.
    """
    captured = {}

    def fake_plot(output_dir, plot_format='pdf', wlmin=0.5, wlmax=20.0, source='profile'):
        captured['output_dir'] = output_dir
        captured['plot_format'] = plot_format

    monkeypatch.setattr(spectra_mod, 'plot_spectra', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/some/output/dir'}
    handler.config.params.out.plot_fmt = 'png'

    spectra_mod.plot_spectra_entry(handler)

    assert captured['output_dir'] == '/some/output/dir'
    assert captured['plot_format'] == 'png'
