"""Unit tests for ``proteus.plot.cpl_bolometry``.

Covers ``plot_bolometry`` (early-return guard plus full plot path) and
the ``plot_bolometry_entry`` wrapper. Matplotlib is mocked at the
source-binding attribute.

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

import proteus.plot.cpl_bolometry as bolometry_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_hf_all(n: int = 5, t_end: float = 1e8) -> pd.DataFrame:
    """Build a helpfile DataFrame with the four bolometric columns."""
    return pd.DataFrame(
        {
            'Time': np.logspace(2, np.log10(t_end), n),
            'albedo_pl': np.linspace(0.05, 0.30, n),
            'bond_albedo': np.linspace(0.10, 0.35, n),
            'transit_depth': np.linspace(1e-4, 5e-4, n),
            'eclipse_depth': np.linspace(1e-5, 5e-5, n),
        }
    )


def _install_mock_plt(monkeypatch):
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_axt = MagicMock()
    mock_axl = MagicMock()
    mock_axr = MagicMock()
    mock_axl.twinx.return_value = mock_axr
    mock_plt.subplots.return_value = (mock_fig, (mock_axt, mock_axl))
    monkeypatch.setattr(bolometry_mod, 'plt', mock_plt)
    return mock_plt, mock_fig, mock_axt, mock_axl, mock_axr


def test_plot_bolometry_returns_early_for_short_runs(monkeypatch, tmp_path):
    """A helpfile whose maximum Time is < 2 yr triggers the early-return.
    Discrimination: subplots is not called.
    """
    mock_plt, _, _, _, _ = _install_mock_plt(monkeypatch)
    hf = pd.DataFrame(
        {
            'Time': [0.5, 1.5],
            'albedo_pl': [0.1, 0.2],
            'bond_albedo': [0.1, 0.2],
            'transit_depth': [1e-4, 2e-4],
            'eclipse_depth': [1e-5, 2e-5],
        }
    )
    bolometry_mod.plot_bolometry(hf, str(tmp_path))

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


def test_plot_bolometry_full_path_two_panels_with_twin_axis(monkeypatch, tmp_path):
    """The full plot path creates two stacked axes with the lower one
    having a twin y-axis for eclipse_depth. Discrimination: albedo (2
    plot calls on axt), transit_depth (1 on axl), eclipse_depth (1 on
    axr). Total 4 plot calls split 2+1+1.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, mock_axt, mock_axl, mock_axr = _install_mock_plt(monkeypatch)

    hf = _make_hf_all(n=10)
    bolometry_mod.plot_bolometry(hf, str(tmp_path), plot_format='pdf')

    # Discrimination: axt has 2 plots (albedo_pl, bond_albedo)
    assert mock_axt.plot.call_count == 2
    # axl has 1 plot (transit_depth)
    assert mock_axl.plot.call_count == 1
    # axr (twin) has 1 plot (eclipse_depth)
    assert mock_axr.plot.call_count == 1
    # Figure saved
    assert mock_fig.savefig.call_count == 1


def test_plot_bolometry_entry_forwards_to_plot_bolometry(monkeypatch, tmp_path):
    """The entry wrapper reads the helpfile and forwards to plot_bolometry
    with output_dir + plot_format. Discrimination: a regression that
    dropped plot_format would default to 'pdf' instead of the
    configured value.
    """
    captured = {}

    def fake_plot(hf_all, output_dir, plot_format='pdf', t0=100.0):
        captured['output_dir'] = output_dir
        captured['plot_format'] = plot_format
        captured['rows'] = len(hf_all)

    monkeypatch.setattr(bolometry_mod, 'plot_bolometry', fake_plot)
    monkeypatch.setattr(pd, 'read_csv', lambda *_a, **_kw: _make_hf_all(n=7))

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path)}
    handler.config.params.out.plot_fmt = 'svg'

    bolometry_mod.plot_bolometry_entry(handler)

    assert captured['output_dir'] == str(tmp_path)
    assert captured['plot_format'] == 'svg'
    assert captured['rows'] == 7
