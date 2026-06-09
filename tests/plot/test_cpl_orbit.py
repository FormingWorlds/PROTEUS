"""Unit tests for ``proteus.plot.cpl_orbit``.

Covers ``plot_orbit``, ``plot_orbit_system``, and the ``plot_orbit_entry``
wrapper. Heavy matplotlib operations are mocked at the source-binding
attribute (``cpl_orbit.plt``) so tests run in milliseconds.

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

import proteus.plot.cpl_orbit as orbit_mod
from proteus.utils.constants import AU, secs_per_hour

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_hf_all(n: int = 5, t_start: float = 1e2, t_end: float = 1e8) -> pd.DataFrame:
    """Build a minimal runtime helpfile DataFrame for orbit plotting.

    Provides every column ``plot_orbit`` and ``plot_orbit_system`` read.
    Eccentricity is bounded in [0, 1) so the b = a * sqrt(1 - e^2) algebra
    in ``plot_orbit_system`` is real-valued.
    """
    return pd.DataFrame(
        {
            'Time': np.logspace(np.log10(t_start), np.log10(t_end), n),
            'semimajorax': np.linspace(1.5e11, 1.6e11, n),
            'eccentricity': np.linspace(0.01, 0.05, n),
            'semimajorax_sat': np.linspace(3.8e8, 4.0e8, n),
            'axial_period': np.linspace(24 * 3600, 30 * 3600, n),
            'roche_limit': np.full(n, 1.0e10),
        }
    )


def _install_mock_plt(monkeypatch):
    """Patch the module's ``plt`` binding with a MagicMock and return it."""
    mock_plt = MagicMock()
    # subplots called with two return forms in this module: (2, 1, ...) -> array of axes
    # and (1, 1, ...) -> single axes. Provide a context-flexible default that returns
    # a tuple (fig, axes) where axes can be indexed.
    monkeypatch.setattr(orbit_mod, 'plt', mock_plt)
    return mock_plt


# ---------------------------------------------------------------------------
# plot_orbit
# ---------------------------------------------------------------------------


def test_plot_orbit_returns_early_when_time_below_t0(tmp_path, monkeypatch):
    """``plot_orbit`` must skip without raising when the simulation has not
    yet advanced past ``t0`` years. The mocked plt.subplots must NOT be
    invoked: silent-skip is the contract for early-iteration plot calls.
    """
    mock_plt = _install_mock_plt(monkeypatch)
    hf_all = pd.DataFrame(
        {
            'Time': np.array([1.0, 10.0, 50.0]),
            'semimajorax': np.full(3, 1.5e11),
            'eccentricity': np.full(3, 0.0),
            'semimajorax_sat': np.full(3, 3.8e8),
            'axial_period': np.full(3, 24 * 3600),
            'roche_limit': np.full(3, 1.0e10),
        }
    )

    result = orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='png', t0=100.0)

    assert result is None
    # Discriminating check: the early-return path must not touch matplotlib.
    # A regression that drops the t0 guard would call subplots and save.
    assert not mock_plt.subplots.called


def test_plot_orbit_draws_and_saves_with_sufficient_time(tmp_path, monkeypatch):
    """When the helpfile contains times above t0, ``plot_orbit`` produces a
    2-row figure, applies a log x-scale to both axes, and saves the figure.
    """
    mock_fig = MagicMock()
    ax_t = MagicMock()
    ax_b = MagicMock()
    # twinx returns an independent axis for the right-side series
    ax_t.twinx.return_value = MagicMock()
    ax_b.twinx.return_value = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, [ax_t, ax_b])

    hf_all = _make_hf_all(n=6, t_start=1e2, t_end=1e8)
    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='png', t0=100.0)

    # The figure was saved once with the expected target path.
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args[0][0]
    assert saved_path.endswith('plot_orbit.png')
    # Both x-axes set log scale on the bottom plot (shared x). A regression
    # that swaps the scale to linear would leave this call missing.
    ax_b.set_xscale.assert_called_with('log')
    ax_t.set_xscale.assert_called_with('log')


def test_plot_orbit_passes_correct_units_to_axes(tmp_path, monkeypatch):
    """``plot_orbit`` divides ``semimajorax`` by AU and ``axial_period`` by
    ``secs_per_hour`` before plotting. The plotted y-values therefore land
    in the expected human-unit range, which discriminates against a
    forgotten unit conversion (raw SI would be ~1e11 not ~1).
    """
    mock_fig = MagicMock()
    ax_t = MagicMock()
    ax_b = MagicMock()
    ax_tr = MagicMock()
    ax_br = MagicMock()
    ax_t.twinx.return_value = ax_tr
    ax_b.twinx.return_value = ax_br
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, [ax_t, ax_b])

    hf_all = _make_hf_all(n=4, t_start=1e3, t_end=1e7)
    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='pdf', t0=100.0)

    # ax_t.plot is called with (time, semimajorax/AU); extract the second positional
    # argument and verify the magnitude lies in plausible AU range, not raw metres.
    y_planet = ax_t.plot.call_args[0][1]
    assert np.amax(y_planet) == pytest.approx(1.6e11 / AU, rel=1e-9)
    # Scale guard: AU-scaled values are O(1) for an Earth-like orbit. A
    # forgotten /AU would land at O(1e11).
    assert 0.5 < np.amax(y_planet) < 5.0

    # ax_br.plot receives (time, axial_period/secs_per_hour). Verify hour scaling.
    y_period = ax_br.plot.call_args[0][1]
    assert np.amax(y_period) == pytest.approx((30 * 3600) / secs_per_hour, rel=1e-9)
    # Scale guard: 24-30 h spans an Earth-day. A forgotten /secs_per_hour
    # would land at ~1e5.
    assert 10.0 < np.amax(y_period) < 50.0


def test_plot_orbit_yaxis_lower_bound_above_zero_when_min_eccentricity_positive(
    tmp_path, monkeypatch
):
    """For the right-axis eccentricity panel, ``plot_orbit`` sets
    ``ymin = amin(e) / yext``. With strictly positive eccentricities the
    lower y-bound must therefore be strictly positive: a regression that
    flipped the divide to a multiply would push ymin above amin(e).
    """
    mock_fig = MagicMock()
    ax_t = MagicMock()
    ax_b = MagicMock()
    ax_tr = MagicMock()
    ax_t.twinx.return_value = ax_tr
    ax_b.twinx.return_value = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, [ax_t, ax_b])

    hf_all = _make_hf_all(n=5, t_start=1e3, t_end=1e6)
    # Force a strictly positive minimum eccentricity.
    hf_all['eccentricity'] = np.linspace(0.10, 0.20, 5)

    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='png', t0=100.0)

    ymin_called, _ymax_called = ax_tr.set_ylim.call_args[0]
    # ymin = 0.10 / 1.05 ~ 0.0952 with positive sign.
    assert ymin_called == pytest.approx(0.10 / 1.05, rel=1e-9)
    # Sign guard: a sign flip ( - amin / yext ) lands at ~-0.0952; the actual
    # ymin must be strictly positive.
    assert ymin_called > 0


# ---------------------------------------------------------------------------
# plot_orbit_system
# ---------------------------------------------------------------------------


def test_plot_orbit_system_returns_early_when_below_t0(tmp_path, monkeypatch):
    """``plot_orbit_system`` skips when the maximum simulated time has not
    yet exceeded ``t0 + 1`` years. plt.subplots must NOT be invoked.
    """
    mock_plt = _install_mock_plt(monkeypatch)
    hf_all = pd.DataFrame(
        {
            'Time': np.array([1.0, 100.0, 500.0]),
            'semimajorax': np.full(3, 1.5e11),
            'eccentricity': np.full(3, 0.0),
            'semimajorax_sat': np.full(3, 3.8e8),
            'axial_period': np.full(3, 24 * 3600),
            'roche_limit': np.full(3, 1.0e10),
        }
    )

    result = orbit_mod.plot_orbit_system(hf_all, str(tmp_path), plot_format='png', t0=1e3)

    assert result is None
    # Same discrimination as the plot_orbit early-return: t0 guard must hold.
    assert not mock_plt.subplots.called


def test_plot_orbit_system_draws_planet_satellite_and_roche(tmp_path, monkeypatch):
    """With times that span well past ``t0`` and a small set of orbital
    snapshots, the system plot loops over every row and calls ax.plot for
    each (planet orbit + satellite orbit), plus the Roche-limit dashed
    line and two dummy legend entries.
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    # The ScalarMappable colourbar path is touched too; allow it to return a stub.
    mock_plt.cm.ScalarMappable.return_value = MagicMock()
    # make_axes_locatable expects a real Axes; replace with a stub returning a
    # MagicMock for .append_axes.
    monkeypatch.setattr(orbit_mod, 'make_axes_locatable', lambda _ax: MagicMock())

    n = 4
    hf_all = _make_hf_all(n=n, t_start=1e3, t_end=1e6)
    orbit_mod.plot_orbit_system(hf_all, str(tmp_path), plot_format='png', t0=1e3)

    # Per-row: planet ellipse plot + satellite plot = 2 ax.plot calls.
    # Plus one Roche-limit dashed line and two dummy-label plot([], []) calls.
    expected_plot_calls = 2 * n + 1 + 2
    assert mock_ax.plot.call_count == expected_plot_calls
    # Distinguishing guard: a regression that skipped the satellite ring would
    # land at n+3 calls (~7), not 2n+3 (~11).
    assert mock_ax.plot.call_count > n + 3

    mock_fig.savefig.assert_called_once()
    fpath = mock_fig.savefig.call_args[0][0]
    assert fpath.endswith('plot_orbit_system.png')


def test_plot_orbit_system_roche_radius_scaled_to_AU(tmp_path, monkeypatch):
    """The Roche-limit dashed ring is plotted at ``roche_limit / AU``. The
    x-component magnitude must therefore land at metres / AU, not at raw
    metres.
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.cm.ScalarMappable.return_value = MagicMock()
    monkeypatch.setattr(orbit_mod, 'make_axes_locatable', lambda _ax: MagicMock())

    hf_all = _make_hf_all(n=3, t_start=1e3, t_end=1e6)
    # Roche limit large enough to be the visible feature; AU-converted value ~6.68e-2 AU.
    hf_all['roche_limit'] = np.full(3, 1.0e10)

    orbit_mod.plot_orbit_system(hf_all, str(tmp_path), plot_format='png', t0=1e3)

    # The Roche-limit plot call is the (2 * nrows + 1)-th plot call when counted
    # over the planet (n) + sat (n) loop. Easier: scan all plot calls for the
    # one with the 'dashed' linestyle.
    dashed_calls = [c for c in mock_ax.plot.call_args_list if c.kwargs.get('ls') == 'dashed']
    assert len(dashed_calls) == 1
    x_vals = dashed_calls[0][0][0]
    expected = 1.0e10 / AU
    assert np.amax(x_vals) == pytest.approx(expected, rel=1e-9)
    # Scale guard: AU-scaled magnitude is ~6.7e-2; a forgotten /AU would
    # land at ~1e10.
    assert 1e-4 < np.amax(x_vals) < 1.0


# ---------------------------------------------------------------------------
# plot_orbit_entry
# ---------------------------------------------------------------------------


def test_plot_orbit_entry_reads_helpfile_and_calls_both_plots(monkeypatch, tmp_path):
    """The entry wrapper reads ``runtime_helpfile.csv`` from the run output
    directory and dispatches to both ``plot_orbit`` and
    ``plot_orbit_system`` with the configured plot format.
    """
    fake_hf = _make_hf_all(n=4, t_start=1e3, t_end=1e6)

    captured_calls = []

    def fake_plot_orbit(hf_all, output_dir, plot_format='pdf', t0=100.0):
        captured_calls.append(('plot_orbit', hf_all, output_dir, plot_format))

    def fake_plot_orbit_system(hf_all, output_dir, plot_format='pdf', t0=1e3):
        captured_calls.append(('plot_orbit_system', hf_all, output_dir, plot_format))

    monkeypatch.setattr(orbit_mod.pd, 'read_csv', lambda *a, **kw: fake_hf)
    monkeypatch.setattr(orbit_mod, 'plot_orbit', fake_plot_orbit)
    monkeypatch.setattr(orbit_mod, 'plot_orbit_system', fake_plot_orbit_system)

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path)}
    handler.config.params.out.plot_fmt = 'png'

    orbit_mod.plot_orbit_entry(handler)

    # Both downstream functions were invoked exactly once.
    names = [c[0] for c in captured_calls]
    assert names == ['plot_orbit', 'plot_orbit_system']
    # Both received the configured format. A regression hardcoding 'pdf'
    # would fail this check.
    assert all(c[3] == 'png' for c in captured_calls)
