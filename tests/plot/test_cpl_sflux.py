"""Unit tests for ``proteus.plot.cpl_sflux``.

Covers ``planck_function``, ``plot_sflux`` (no-data guard plus the
multi-time and single-time branches plus modern-spectrum overlay) and
``plot_sflux_entry``. Matplotlib and mpl are mocked at the
source-binding attributes.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_sflux as sflux_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# planck_function: closed-form Planck radiance
# ---------------------------------------------------------------------------


def test_planck_function_returns_positive_at_solar_peak_wavelength():
    """Planck function at 500 nm and 5772 K (solar peak) must be positive
    with a value in the expected magnitude range. Discrimination: sign
    + order-of-magnitude guards rule out scale-bug regressions.
    """
    val = sflux_mod.planck_function(500.0, 5772.0)
    assert val > 0
    # Discrimination: scale guard ~1e6 erg/s/cm^2/nm for solar peak
    assert 1e4 < val < 1e8


# ---------------------------------------------------------------------------
# plot_sflux: no-data guard
# ---------------------------------------------------------------------------


def _install_mock_plt(monkeypatch):
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    monkeypatch.setattr(sflux_mod, 'plt', mock_plt)
    monkeypatch.setattr(sflux_mod, 'mpl', MagicMock())
    monkeypatch.setattr(sflux_mod, 'make_axes_locatable', MagicMock())
    return mock_plt, mock_fig, mock_ax


def _write_sflux(path, wavelengths, fluxes, skiprows_header: int = 1, delimiter: str = '\t'):
    """Write a .sflux file with skiprows_header header rows. Historical
    spectra are tab-delimited (np.loadtxt(..., delimiter='\\t')); the
    modern spectrum (-1.sflux) is whitespace-delimited via default
    np.loadtxt.
    """
    lines = ['header'] * skiprows_header
    for w, f in zip(wavelengths, fluxes):
        lines.append(f'{w}{delimiter}{f}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_plot_sflux_returns_early_when_no_files(monkeypatch, tmp_path):
    """With no .sflux files in output_dir/data/, the function logs a
    warning and returns. Discrimination: subplots not called.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    (tmp_path / 'data').mkdir()

    sflux_mod.plot_sflux(output_dir=str(tmp_path))

    assert mock_plt.subplots.call_count == 0


def test_plot_sflux_single_file_branch_uses_blue_line_with_year_label(monkeypatch, tmp_path):
    """When exactly one .sflux file is found, the justone=True branch is
    taken: no colourbar, the line is plotted in 'tab:blue' with a year
    label. Discrimination: only ONE ax.plot call (the historical), since
    plt_modern=False here.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()
    wl = np.array([100.0, 500.0, 1000.0])
    _write_sflux(tmp_path / 'data' / '1000.sflux', wl, np.full_like(wl, 1e-3))

    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    sflux_mod.plot_sflux(output_dir=str(tmp_path), plt_modern=False, plot_format='pdf')

    # Discrimination: exactly 1 line (the single historical spectrum)
    assert mock_ax.plot.call_count == 1
    # In single-time branch, the color used is 'tab:blue'
    plot_color = mock_ax.plot.call_args.kwargs.get('color')
    assert plot_color == 'tab:blue'
    assert mock_fig.savefig.call_count == 1


def test_plot_sflux_multi_file_branch_plots_one_line_per_history_file(monkeypatch, tmp_path):
    """With multiple .sflux files AND plt_modern=False (no overlay), the
    function plots one line per historical spectrum. Discrimination:
    with 3 historical files, ax.plot is called exactly 3 times.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()
    wl = np.array([100.0, 500.0, 1000.0])
    # 3 historical spectra
    for time in (100, 10000, 1000000):
        _write_sflux(tmp_path / 'data' / f'{time}.sflux', wl, np.full_like(wl, 1e-4 * time))

    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    sflux_mod.plot_sflux(output_dir=str(tmp_path), plt_modern=False, plot_format='pdf')

    # Discrimination: exactly 3 lines, one per historical spectrum
    assert mock_ax.plot.call_count == 3
    assert mock_fig.savefig.call_count == 1


def test_plot_sflux_warns_when_plt_modern_but_modern_file_absent(monkeypatch, tmp_path, caplog):
    """plt_modern=True but no -1.sflux present: the function logs a
    warning but still produces the figure with N historical lines only.
    Discrimination: ax.plot call count == N (no modern overlay), AND
    a 'Could not find' warning is logged.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()
    wl = np.array([100.0, 500.0, 1000.0])
    for time in (100, 10000):
        _write_sflux(tmp_path / 'data' / f'{time}.sflux', wl, np.full_like(wl, 1e-4))

    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    with caplog.at_level('WARNING'):
        sflux_mod.plot_sflux(output_dir=str(tmp_path), plt_modern=True, plot_format='pdf')

    # Discrimination: no modern overlay; only 2 historical lines
    assert mock_ax.plot.call_count == 2
    assert any('Could not find' in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# plot_sflux_entry: wrapper
# ---------------------------------------------------------------------------


def test_plot_sflux_entry_sets_plt_modern_from_mors_module(monkeypatch):
    """When star.module == 'mors', plt_modern is True. Discrimination:
    plt_modern flag must be exactly True (not 'mors' string).
    """
    captured = {}

    def fake_plot(output_dir, wl_max=6000.0, plt_modern=True, plot_format='pdf'):
        captured['plt_modern'] = plt_modern
        captured['plot_format'] = plot_format

    monkeypatch.setattr(sflux_mod, 'plot_sflux', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/tmp/out'}
    handler.config.star.module = 'mors'
    handler.config.params.out.plot_fmt = 'svg'

    sflux_mod.plot_sflux_entry(handler)

    assert captured['plt_modern'] is True
    assert captured['plot_format'] == 'svg'


def test_plot_sflux_entry_sets_plt_modern_false_for_dummy_module(monkeypatch):
    """For star.module != 'mors', plt_modern is False so the modern
    spectrum overlay is suppressed (no -1.sflux file is expected).
    """
    captured = {}

    def fake_plot(output_dir, wl_max=6000.0, plt_modern=True, plot_format='pdf'):
        captured['plt_modern'] = plt_modern

    monkeypatch.setattr(sflux_mod, 'plot_sflux', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/tmp/out'}
    handler.config.star.module = 'dummy'
    handler.config.params.out.plot_fmt = 'pdf'

    sflux_mod.plot_sflux_entry(handler)

    assert captured['plt_modern'] is False
