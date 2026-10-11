"""Unit tests for ``proteus.plot.cpl_fluxes_global``.

Covers ``plot_fluxes_global`` (early-return guard for short runs plus
full plot path) and the ``plot_fluxes_global_entry`` wrapper.
Matplotlib is mocked at the source-binding attribute.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb, to_rgba

import proteus.plot.cpl_fluxes_global as fluxes_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_hf_all(n: int = 5, t_end: float = 1e8) -> pd.DataFrame:
    """Build a helpfile DataFrame with the seven flux + albedo columns
    plot_fluxes_global reads.
    """
    times = np.logspace(2, np.log10(t_end), n)
    return pd.DataFrame(
        {
            'Time': times,
            'F_atm': np.linspace(1e4, 1e2, n),
            'F_ins': np.full(n, 1361.0),  # Solar constant
            'F_olr': np.linspace(300.0, 200.0, n),
            'F_sct': np.linspace(100.0, 50.0, n),
            'F_int': np.linspace(1e4, 1e2, n),
            'F_tidal': np.linspace(10.0, 1.0, n),
            'F_radio': np.linspace(5.0, 0.5, n),
            'albedo_pl': np.linspace(0.05, 0.30, n),
        }
    )


def _make_config(s0_factor: float = 0.375, zenith_deg: float = 48.2, plot_fmt: str = 'pdf'):
    cfg = MagicMock()
    cfg.orbit.s0_factor = s0_factor
    cfg.orbit.zenith_angle = zenith_deg
    cfg.params.out.plot_fmt = plot_fmt
    return cfg


def _install_mock_plt(monkeypatch):
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    monkeypatch.setattr(fluxes_mod, 'plt', mock_plt)
    return mock_plt, mock_fig, mock_ax


def test_plot_fluxes_global_returns_early_for_too_few_samples(monkeypatch, tmp_path):
    """A helpfile with <3 rows past t0 triggers early-return.
    Discrimination: subplots not called.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    # Only 2 samples past t0=100
    hf = _make_hf_all(n=4)
    hf['Time'] = [10.0, 50.0, 200.0, 500.0]  # only 2 samples > 100

    fluxes_mod.plot_fluxes_global(hf, str(tmp_path), config=_make_config())

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


def test_plot_fluxes_global_full_path_plots_seven_flux_components(monkeypatch, tmp_path):
    """The full path plots 7 flux components plus a horizontal line
    (S-N limit) on the same axis. Discrimination: 7 ax.plot calls
    (Tidal, Radio, F_int, F_atm, F_olr, F_upw, F_asf) plus 1 axhline.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    hf = _make_hf_all(n=8)  # all > 100 yr
    fluxes_mod.plot_fluxes_global(hf, str(tmp_path), config=_make_config())

    # 7 flux components
    assert mock_ax.plot.call_count == 7
    # 1 horizontal line (S-N limit)
    assert mock_ax.axhline.call_count == 1
    # Saved once
    assert mock_fig.savefig.call_count == 1


def test_plot_fluxes_global_uses_config_plot_format_in_filename(monkeypatch, tmp_path):
    """The saved file extension is the configured plot_fmt from config.
    A regression that hardcoded 'pdf' would lose user preference.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, _ = _install_mock_plt(monkeypatch)

    hf = _make_hf_all(n=5)
    cfg = _make_config(plot_fmt='png')
    fluxes_mod.plot_fluxes_global(hf, str(tmp_path), config=cfg)

    saved_path = mock_fig.savefig.call_args.args[0]
    # Discrimination: must end in 'png', not 'pdf'
    assert saved_path.endswith('plot_fluxes_global.png')
    assert 'plot_fluxes_global' in saved_path
    assert not saved_path.endswith('.pdf')


def test_plot_fluxes_global_entry_forwards_handler_attributes(monkeypatch, tmp_path):
    """The entry wrapper reads runtime_helpfile.csv and forwards path +
    config to plot_fluxes_global.
    """
    captured = {}

    def fake_plot(hf_all, output_dir, config, t0=100.0):
        captured['output_dir'] = output_dir
        captured['config'] = config
        captured['rows'] = len(hf_all)

    monkeypatch.setattr(fluxes_mod, 'plot_fluxes_global', fake_plot)
    monkeypatch.setattr(pd, 'read_csv', lambda *_a, **_kw: _make_hf_all(n=5))

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path)}
    handler.config = _make_config()

    fluxes_mod.plot_fluxes_global_entry(handler)

    assert captured['output_dir'] == str(tmp_path)
    assert captured['config'] is handler.config
    assert captured['rows'] == 5


# A dark style; 0x44 / 0xFF of the way from its text colour (224) to its face colour (32).
DARK_STYLE = {'text.color': '#e0e0e0', 'axes.facecolor': '#202020'}
DARK_STYLE_DARK_GREY = (172.8 / 255,) * 3

_STYLES = pytest.mark.parametrize(
    'rc, text, face, dark_grey',
    [
        ({}, 'black', 'white', '#444444'),
        (DARK_STYLE, '#e0e0e0', '#202020', DARK_STYLE_DARK_GREY),
    ],
    ids=['default_style', 'dark_style'],
)


@_STYLES
def test_plot_fluxes_global_takes_its_neutral_colours_from_the_style(
    draw_real, tmp_path, rc, text, face, dark_grey
):
    """On real axes the runaway limit line and the grid have the text colour of
    the style, the net atmosphere flux its dark grey, and the legend frame its
    axes face colour: black, '#444444' and white in the default style."""
    (tmp_path / 'plots').mkdir()

    (fig,) = draw_real(
        fluxes_mod.plot_fluxes_global,
        _make_hf_all(),
        str(tmp_path),
        config=_make_config(),
        rc=rc,
    )

    (ax,) = fig.axes
    line = {ln.get_label(): ln for ln in ax.lines}
    assert to_rgb(line['S-N limit'].get_color()) == pytest.approx(to_rgb(text))
    assert to_rgb(line['Net (atm.)'].get_color()) == pytest.approx(to_rgb(dark_grey))
    assert to_rgb(ax.xaxis.get_gridlines()[0].get_color()) == pytest.approx(to_rgb(text))
    assert ax.get_legend().get_frame().get_facecolor() == pytest.approx(to_rgba(face, 0.99))
    assert to_rgb(line['Net (int.)'].get_color()) == pytest.approx(to_rgb('#ff7f0e'))
