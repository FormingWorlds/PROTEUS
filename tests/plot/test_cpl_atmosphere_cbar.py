"""Unit tests for ``proteus.plot.cpl_atmosphere_cbar``.

Covers ``plot_atmosphere_cbar`` (early-return guard plus full plot path
with colour bar) and the ``plot_atmosphere_cbar_entry`` wrapper.
Matplotlib + axes_grid1 are mocked at the source-binding attributes.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_atmosphere_cbar as atmos_cbar_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_profile(n_lev: int = 30) -> dict:
    """One atmospheric profile snapshot."""
    return {
        'z': np.linspace(0, 1.0e6, n_lev),
        'p': np.logspace(7, 0, n_lev),
        't': np.linspace(2500.0, 200.0, n_lev),
        'transparent': False,
    }


def _install_mock_plt(monkeypatch):
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.cm.ScalarMappable.return_value = MagicMock(
        to_rgba=MagicMock(return_value=(0.1, 0.2, 0.3, 1.0))
    )
    monkeypatch.setattr(atmos_cbar_mod, 'plt', mock_plt)

    monkeypatch.setattr(atmos_cbar_mod, 'mpl', MagicMock())
    monkeypatch.setattr(atmos_cbar_mod, 'make_axes_locatable', MagicMock())
    return mock_plt, mock_fig, mock_ax


def test_plot_atmosphere_cbar_returns_early_when_single_time(monkeypatch, tmp_path):
    """A single-snapshot times list (len < 2) is skipped with a warning.
    A regression that proceeded would have called subplots and saved a
    figure. Discrimination: neither subplots NOR savefig may have run.
    """
    mock_plt, mock_fig, _ = _install_mock_plt(monkeypatch)

    atmos_cbar_mod.plot_atmosphere_cbar(
        output_dir=str(tmp_path),
        times=[1e3],
        profiles=[_make_profile()],
    )

    assert mock_plt.subplots.call_count == 0
    assert mock_fig.savefig.call_count == 0


def test_plot_atmosphere_cbar_full_path_creates_figure_and_colourbar(monkeypatch, tmp_path):
    """With 3 snapshots, the function plots three T-P curves, attaches
    a vertical colourbar via make_axes_locatable, and saves the figure.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    times = [1e3, 1e5, 1e7]
    profiles = [_make_profile() for _ in times]

    atmos_cbar_mod.plot_atmosphere_cbar(
        output_dir=str(tmp_path),
        times=times,
        profiles=profiles,
        plot_format='pdf',
    )

    assert mock_plt.subplots.call_count == 1
    # Discrimination: 3 profile lines, one per snapshot
    assert mock_ax.plot.call_count == 3
    # Colourbar attached via fig.colorbar
    assert mock_fig.colorbar.call_count == 1
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_atmosphere_cbar.pdf')


def test_plot_atmosphere_cbar_entry_passes_nsamp_to_sample_output(monkeypatch):
    """The entry wrapper calls sample_output with nsamp=40 (the
    higher-resolution sampling for the colourbar plot). A regression
    that dropped nsamp would silently default and produce a different
    snapshot set.
    """
    captured_kwargs = {}

    def fake_sample_output(handler, **kwargs):
        captured_kwargs.update(kwargs)
        return [1e4, 1e5, 1e6], None

    monkeypatch.setattr(atmos_cbar_mod, 'sample_output', fake_sample_output)
    monkeypatch.setattr(
        atmos_cbar_mod,
        'read_atmosphere_data',
        lambda output_dir, times: [_make_profile() for _ in times],
    )

    captured_call = {}

    def fake_plot(output_dir, times, profiles, plot_format='pdf'):
        captured_call['output_dir'] = output_dir
        captured_call['times'] = times
        captured_call['plot_format'] = plot_format

    monkeypatch.setattr(atmos_cbar_mod, 'plot_atmosphere_cbar', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/expected/out'}
    handler.config.params.out.plot_fmt = 'png'

    atmos_cbar_mod.plot_atmosphere_cbar_entry(handler)

    # Discrimination: nsamp must be exactly 40 (the colourbar plot is
    # higher-resolution than the default atmosphere plot which uses the
    # default nsamp). A regression that called sample_output without
    # nsamp would not have the key.
    assert captured_kwargs.get('nsamp') == 40
    assert captured_kwargs.get('extension') == '_atm.nc'
    assert captured_call['output_dir'] == '/expected/out'
    assert captured_call['plot_format'] == 'png'
