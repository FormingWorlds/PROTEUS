"""Unit tests for ``proteus.plot.cpl_escape``.

Covers ``plot_escape`` (early-return guards for short runs plus full
plot path through 3 panels) and the ``plot_escape_entry`` wrapper.
Matplotlib is mocked at the source-binding attribute.

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

import proteus.plot.cpl_escape as escape_mod
from proteus.utils.constants import element_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_hf_all(n: int = 6, t_end: float = 1e8) -> pd.DataFrame:
    """Build a helpfile DataFrame with every column plot_escape reads.

    Includes per-element <e>_kg_total, <e>_kg_atm, esc_rate_<e> for
    every element in element_list, plus M_atm, R_xuv, R_int, P_surf.
    """
    times = np.logspace(2, np.log10(t_end), n)
    df = pd.DataFrame({'Time': times})
    for e in element_list:
        df[f'{e}_kg_total'] = np.linspace(1e22, 1e21, n)  # decreasing
        df[f'{e}_kg_atm'] = np.linspace(1e20, 1e19, n)
        df[f'esc_rate_{e}'] = np.linspace(1e10, 1e8, n)  # kg/s
    df['esc_rate_total'] = sum(df[f'esc_rate_{e}'] for e in element_list)
    df['M_atm'] = df['H_kg_atm'] + df['O_kg_atm']
    df['R_xuv'] = np.linspace(7.0e6, 6.5e6, n)
    df['R_int'] = np.full(n, 6.371e6)
    df['P_surf'] = np.linspace(100.0, 30.0, n)  # bar
    return df


def _install_mock_plt(monkeypatch):
    """Patch plt; return (mock_plt, mock_fig, [ax0, ax1, ax2], mock_axr)."""
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax0 = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_axr = MagicMock()
    mock_ax2.twinx.return_value = mock_axr
    # subplots returns an array-like of 3 axes
    mock_plt.subplots.return_value = (mock_fig, [mock_ax0, mock_ax1, mock_ax2])
    monkeypatch.setattr(escape_mod, 'plt', mock_plt)
    return mock_plt, mock_fig, [mock_ax0, mock_ax1, mock_ax2], mock_axr


def test_plot_escape_returns_early_when_too_few_rows(monkeypatch, tmp_path):
    """Helpfile with fewer than 3 rows triggers the early-return.
    plot_escape uses iloc[2:] to crop initial rows, so it needs at
    least 3 rows. Discrimination: subplots not called.
    """
    mock_plt, _, _, _ = _install_mock_plt(monkeypatch)
    hf = _make_hf_all(n=2)  # only 2 rows

    escape_mod.plot_escape(hf, str(tmp_path))

    assert mock_plt.subplots.call_count == 0


def test_plot_escape_returns_early_when_max_time_below_two(monkeypatch, tmp_path):
    """Helpfile whose max Time < 2 yr triggers early-return regardless
    of row count. Discrimination: subplots not called.
    """
    mock_plt, _, _, _ = _install_mock_plt(monkeypatch)
    hf = _make_hf_all(n=10)
    hf['Time'] = np.linspace(0.1, 1.5, 10)  # all < 2 yr

    escape_mod.plot_escape(hf, str(tmp_path))

    assert mock_plt.subplots.call_count == 0


def test_plot_escape_full_path_plots_per_element_and_summary_panels(monkeypatch, tmp_path):
    """The full path plots per-element inventory + atmospheric inventory
    + escape rate (3 lines per element) on the top two axes, then the
    summary total + atmosphere mass on the top axis, and Rxuv + Psurf
    on the bottom axis. With 9 elements:
      ax0: 9 _kg_total + 9 _kg_atm + 1 total + 1 M_atm = 20 plot calls
      ax1: 9 esc_rate + 1 total (k-line) = 10 plot calls
      ax2: 1 R_xuv = 1 plot call
      axr: 1 P_surf = 1 plot call
    Discrimination: a regression that dropped one element from the
    loop would change these counts.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, axes, mock_axr = _install_mock_plt(monkeypatch)
    ax0, ax1, ax2 = axes

    hf = _make_hf_all(n=10)
    escape_mod.plot_escape(hf, str(tmp_path), plot_format='pdf')

    n_elem = len(element_list)
    # ax0: 2 lines per element + Total + M_atm
    assert ax0.plot.call_count == 2 * n_elem + 2
    # ax1: 1 line per element + 1 total
    assert ax1.plot.call_count == n_elem + 1
    # ax2: R_xuv
    assert ax2.plot.call_count == 1
    # axr: P_surf
    assert mock_axr.plot.call_count == 1
    # Saved exactly once
    assert mock_fig.savefig.call_count == 1


def test_plot_escape_entry_forwards_to_plot_escape(monkeypatch, tmp_path):
    """The entry wrapper reads runtime_helpfile.csv and forwards path +
    plot_fmt to plot_escape. Discrimination: a regression that dropped
    plot_format would default to 'pdf' instead of the configured value.
    """
    captured = {}

    def fake_plot(hf_all, output_dir, plot_format='pdf'):
        captured['output_dir'] = output_dir
        captured['plot_format'] = plot_format

    monkeypatch.setattr(escape_mod, 'plot_escape', fake_plot)
    monkeypatch.setattr(pd, 'read_csv', lambda *_a, **_kw: _make_hf_all(n=5))

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path)}
    handler.config.params.out.plot_fmt = 'png'

    escape_mod.plot_escape_entry(handler)

    assert captured['output_dir'] == str(tmp_path)
    assert captured['plot_format'] == 'png'
