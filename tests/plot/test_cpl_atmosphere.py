"""Unit tests for ``proteus.plot.cpl_atmosphere``.

Covers ``plot_atmosphere`` (early-return guards plus full plot path with
transparent / opaque profile styling) and the
``plot_atmosphere_entry`` wrapper. Matplotlib + mpl are mocked at the
source-binding attributes so tests run in milliseconds.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_atmosphere as atmos_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(n_lev: int = 30, transparent: bool = False) -> dict:
    """One atmospheric profile snapshot.

    ``z`` is altitude in metres above the surface; ``p`` in Pa; ``t`` in K.
    Profiles span a realistic 1 Pa to 1e7 Pa pressure range with surface T
    of 2500 K decreasing to 200 K at TOA, matching a typical magma-ocean
    cooling phase.
    """
    return {
        'z': np.linspace(0, 1.0e6, n_lev),
        'p': np.logspace(7, 0, n_lev),  # surface 1e7 Pa -> TOA 1 Pa
        't': np.linspace(2500.0, 200.0, n_lev),
        'transparent': transparent,
    }


def _install_mock_plt(monkeypatch):
    """Patch the module's ``plt`` and ``mpl`` bindings. Returns
    (mock_plt, mock_fig, mock_ax0, mock_ax1) for the 2-panel layout.
    """
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax0 = MagicMock()
    mock_ax1 = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax0, mock_ax1))
    mock_plt.cm.ScalarMappable.return_value = MagicMock(
        to_rgba=MagicMock(return_value=(0.1, 0.2, 0.3, 1.0))
    )
    monkeypatch.setattr(atmos_mod, 'plt', mock_plt)

    mock_mpl = MagicMock()
    monkeypatch.setattr(atmos_mod, 'mpl', mock_mpl)
    return mock_plt, mock_fig, mock_ax0, mock_ax1


# ---------------------------------------------------------------------------
# plot_atmosphere: early-return guards
# ---------------------------------------------------------------------------


def test_plot_atmosphere_returns_early_for_empty_times(monkeypatch, tmp_path):
    """Empty times list short-circuits the function. A regression that
    proceeded would attempt to index an empty profile list and raise.
    """
    mock_plt, _, _, _ = _install_mock_plt(monkeypatch)

    result = atmos_mod.plot_atmosphere(
        output_dir=str(tmp_path),
        times=[],
        profiles=[],
    )

    assert result is None
    assert mock_plt.subplots.call_count == 0


def test_plot_atmosphere_returns_early_when_max_time_below_minimum(monkeypatch, tmp_path):
    """When the latest snapshot time is < 2 yr, plotting is skipped. A
    regression that ignored the time guard would have called subplots
    and produced a figure on disk. Discrimination: no figure must be
    saved either (subplots could be skipped while savefig still ran via
    a stale figure handle).
    """
    mock_plt, _, _, _ = _install_mock_plt(monkeypatch)

    atmos_mod.plot_atmosphere(
        output_dir=str(tmp_path),
        times=[0.5, 1.5],
        profiles=[_make_profile(), _make_profile()],
    )

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


# ---------------------------------------------------------------------------
# plot_atmosphere: full plot path
# ---------------------------------------------------------------------------


def test_plot_atmosphere_two_panel_layout_with_opaque_profiles(monkeypatch, tmp_path):
    """The standard layout is 2 subplots (T-z above T-p), with one line
    per snapshot per panel. With 3 snapshots and all-opaque profiles
    each axis sees 3 plot calls; total across the two panels is 6.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()

    mock_plt, mock_fig, mock_ax0, mock_ax1 = _install_mock_plt(monkeypatch)

    times = [1e3, 1e5, 1e7]
    profiles = [_make_profile(transparent=False) for _ in times]

    atmos_mod.plot_atmosphere(
        output_dir=str(tmp_path),
        times=times,
        profiles=profiles,
        plot_format='pdf',
    )

    assert mock_plt.subplots.call_count == 1
    # Discrimination: 3 lines per panel for 3 snapshots
    assert mock_ax0.plot.call_count == 3
    assert mock_ax1.plot.call_count == 3
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_atmosphere.pdf')


def test_plot_atmosphere_uses_dotted_linestyle_for_transparent_profile(monkeypatch, tmp_path):
    """The transparent-flag on a profile flips the linestyle from solid
    to dotted. With one transparent and one opaque snapshot, exactly one
    plot call on each axis must use ls='dotted'.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()

    mock_plt, mock_fig, mock_ax0, mock_ax1 = _install_mock_plt(monkeypatch)

    times = [1e3, 1e5]
    profiles = [_make_profile(transparent=True), _make_profile(transparent=False)]

    atmos_mod.plot_atmosphere(
        output_dir=str(tmp_path),
        times=times,
        profiles=profiles,
        plot_format='pdf',
    )

    # Extract the ls= kwarg from each ax0.plot call. Discrimination: one
    # of them must be 'dotted' (the transparent profile) and one 'solid'.
    linestyles = [call.kwargs.get('ls') for call in mock_ax0.plot.call_args_list]
    assert linestyles.count('dotted') == 1
    assert linestyles.count('solid') == 1


# ---------------------------------------------------------------------------
# plot_atmosphere_entry: wrapper
# ---------------------------------------------------------------------------


def test_plot_atmosphere_entry_forwards_handler_attributes(monkeypatch):
    """The entry wrapper picks snapshot times via sample_output, reads
    atmospheric profiles, and forwards them to plot_atmosphere. A
    regression that dropped plot_format would default to 'pdf' instead
    of the handler-configured value.
    """
    captured = {}

    def fake_plot(output_dir, times, profiles, plot_format='pdf'):
        captured['output_dir'] = output_dir
        captured['times'] = times
        captured['profiles'] = profiles
        captured['plot_format'] = plot_format

    monkeypatch.setattr(atmos_mod, 'plot_atmosphere', fake_plot)
    monkeypatch.setattr(atmos_mod, 'sample_output', lambda *a, **kw: ([1e4, 1e6], None))
    monkeypatch.setattr(
        atmos_mod,
        'read_atmosphere_data',
        lambda output_dir, times: [_make_profile() for _ in times],
    )

    handler = MagicMock()
    handler.directories = {'output': '/expected/output/dir'}
    handler.config.params.out.plot_fmt = 'svg'

    atmos_mod.plot_atmosphere_entry(handler)

    assert captured['output_dir'] == '/expected/output/dir'
    assert captured['plot_format'] == 'svg'
    assert list(captured['times']) == [1e4, 1e6]
    assert len(captured['profiles']) == 2
