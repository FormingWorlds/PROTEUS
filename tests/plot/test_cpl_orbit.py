"""Unit tests for ``proteus.plot.cpl_orbit``.

Covers ``plot_orbit``, ``plot_orbit_system``, and the ``plot_orbit_entry``
wrapper. Heavy matplotlib operations are mocked at the source-binding
attribute (``cpl_orbit.plt``) so tests run in milliseconds.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
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
            'orbital_period': np.linspace(3e7, 3.2e7, n),
            'axial_period': np.linspace(24 * 3600, 30 * 3600, n),
            'semimajorax_sat': np.linspace(3.8e8, 4.0e8, n),
            'eccentricity_sat': np.linspace(0.02, 0.06, n),
            'orbital_period_sat': np.linspace(2.3e6, 2.4e6, n),
            'axial_period_sat': np.linspace(2.3e6, 2.4e6, n),
            'roche_limit': np.full(n, 1.0e10),
        }
    )


def _make_axs_3x2() -> np.ndarray:
    """Build a (3, 2) object array of MagicMock axes matching plot_orbit's
    real ``plt.subplots(3, 2, ...)`` grid, with ``twinx()`` wired on every
    mock so the timescales-panel right axis is reachable."""
    axs = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            ax = MagicMock()
            ax.twinx.return_value = MagicMock()
            axs[i, j] = ax
    return axs


def _install_mock_plt(monkeypatch):
    """Patch the module's ``plt`` binding with a MagicMock and return it."""
    mock_plt = MagicMock()
    # subplots called with two return forms in this module: (2, 1, ...) -> array of axes
    # and (1, 1, ...) -> single axes. Provide a context-flexible default that returns
    # a tuple (fig, axes) where axes can be indexed.
    monkeypatch.setattr(orbit_mod, 'plt', mock_plt)
    return mock_plt


def _make_axs_4x1() -> np.ndarray:
    """Build a (4,) object array of MagicMock axes matching plot_evection's
    real ``plt.subplots(4, 1, ...)`` grid."""
    axs = np.empty(4, dtype=object)
    for i in range(4):
        axs[i] = MagicMock()
    return axs


def _make_evection_hf_all(n: int = 6, t_start: float = 1e2, t_end: float = 6e4) -> pd.DataFrame:
    """Build a minimal runtime helpfile DataFrame for ``plot_evection``.

    Semimajor axis, eccentricity and spin values are picked so
    ``_solve_e_stationary`` (the real, un-mocked root-finder) has a
    genuine solution for at least some rows -- this exercises the real
    numerics rather than a stub, matching how the driver is actually used.
    """
    return pd.DataFrame(
        {
            'Time': np.logspace(np.log10(t_start), np.log10(t_end), n),
            'semimajorax_sat': np.linspace(7.0, 12.0, n) * 6.371e6,
            'eccentricity_sat': np.linspace(0.05, 0.4, n),
            'axial_period': np.linspace(2 * 3600, 5 * 3600, n),
            'evection_angle': np.linspace(0.0, 3.0, n),
            'plan_sat_am': np.linspace(3.5e34, 3.6e34, n),
        }
    )


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
    """When the helpfile contains times above t0, ``plot_orbit`` produces
    the 3-row (semi-major axis, eccentricity, timescales) by 2-column
    (planet, satellite) grid, applies a log x-scale to every panel via
    ``axs.flat``, and saves the figure.
    """
    mock_fig = MagicMock()
    axs = _make_axs_3x2()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_hf_all(n=6, t_start=1e2, t_end=1e8)
    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='png', t0=100.0)

    # The figure was saved once with the expected target path.
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args[0][0]
    assert saved_path.endswith('plot_orbit.png')
    # Every panel in the grid gets a log x-scale (the shared-x loop over
    # axs.flat). A regression that only touched a subset of panels, or
    # swapped the scale to linear, would leave one of these missing.
    for i in range(3):
        for j in range(2):
            axs[i, j].set_xscale.assert_called_with('log')


def test_plot_orbit_passes_correct_units_to_axes(tmp_path, monkeypatch):
    """``plot_orbit`` divides ``semimajorax`` by AU and ``axial_period`` by
    ``secs_per_hour`` before plotting. The plotted y-values therefore land
    in the expected human-unit range, which discriminates against a
    forgotten unit conversion (raw SI would be ~1e11 not ~1).
    """
    mock_fig = MagicMock()
    axs = _make_axs_3x2()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_hf_all(n=4, t_start=1e3, t_end=1e7)
    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='pdf', t0=100.0)

    # Panel [0, 0] (planet semi-major axis) is called with (time,
    # semimajorax/AU); extract the second positional argument and verify
    # the magnitude lies in plausible AU range, not raw metres.
    y_planet = axs[0, 0].plot.call_args[0][1]
    assert np.amax(y_planet) == pytest.approx(1.6e11 / AU, rel=1e-9)
    # Scale guard: AU-scaled values are O(1) for an Earth-like orbit. A
    # forgotten /AU would land at O(1e11).
    assert 0.5 < np.amax(y_planet) < 5.0

    # Panel [2, 0]'s twinx (planet axial-spin-period right axis) receives
    # (time, axial_period/secs_per_hour). Verify hour scaling.
    ax_spin = axs[2, 0].twinx.return_value
    y_period = ax_spin.plot.call_args[0][1]
    assert np.amax(y_period) == pytest.approx((30 * 3600) / secs_per_hour, rel=1e-9)
    # Scale guard: 24-30 h spans an Earth-day. A forgotten /secs_per_hour
    # would land at ~1e5.
    assert 10.0 < np.amax(y_period) < 50.0


def test_plot_orbit_yaxis_lower_bound_above_zero_when_min_eccentricity_positive(
    tmp_path, monkeypatch
):
    """For the planet-eccentricity panel [1, 0], ``plot_orbit`` sets
    ``ymin = amin(e) / yext``. With strictly positive eccentricities the
    lower y-bound must therefore be strictly positive: a regression that
    flipped the divide to a multiply would push ymin above amin(e).
    """
    mock_fig = MagicMock()
    axs = _make_axs_3x2()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_hf_all(n=5, t_start=1e3, t_end=1e6)
    # Force a strictly positive minimum eccentricity.
    hf_all['eccentricity'] = np.linspace(0.10, 0.20, 5)

    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='png', t0=100.0)

    ymin_called, _ymax_called = axs[1, 0].set_ylim.call_args[0]
    # ymin = 0.10 / 1.05 ~ 0.0952 with positive sign.
    assert ymin_called == pytest.approx(0.10 / 1.05, rel=1e-9)
    # Sign guard: a sign flip ( - amin / yext ) lands at ~-0.0952; the actual
    # ymin must be strictly positive.
    assert ymin_called > 0


def test_plot_orbit_notates_blank_satellite_panels_when_no_satellite_data(
    tmp_path, monkeypatch
):
    """When the helpfile has no ``semimajorax_sat`` column (no satellite
    simulated), the right-hand column's 3 panels must be notated
    ('No Satellite Data') rather than raising a KeyError trying to plot
    columns that don't exist."""
    mock_fig = MagicMock()
    axs = _make_axs_3x2()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = pd.DataFrame(
        {
            'Time': np.logspace(2, 8, 5),
            'semimajorax': np.linspace(1.5e11, 1.6e11, 5),
            'eccentricity': np.linspace(0.01, 0.05, 5),
            'orbital_period': np.linspace(3e7, 3.2e7, 5),
            'axial_period': np.linspace(24 * 3600, 30 * 3600, 5),
        }
    )
    orbit_mod.plot_orbit(hf_all, str(tmp_path), plot_format='png', t0=100.0)

    for row in range(3):
        axs[row, 1].text.assert_called_once()
        args, kwargs = axs[row, 1].text.call_args
        assert args[2] == 'No Satellite Data'
    # Discrimination: the left (planet) column must still be drawn
    # normally, not also skipped.
    assert axs[0, 0].plot.call_count == 1


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
# plot_evection
# ---------------------------------------------------------------------------


def test_plot_evection_returns_early_when_time_below_t0(monkeypatch):
    """Must skip without raising, and without touching matplotlib, when
    the simulation has not yet advanced past ``t0`` years -- mirrors
    ``plot_orbit``'s equivalent guard."""
    mock_plt = _install_mock_plt(monkeypatch)
    hf_all = pd.DataFrame(
        {
            'Time': np.array([1.0, 5.0, 20.0]),
            'semimajorax_sat': np.full(3, 8.0 * 6.371e6),
            'eccentricity_sat': np.full(3, 0.1),
            'axial_period': np.full(3, 3 * 3600.0),
            'evection_angle': np.full(3, 0.5),
            'plan_sat_am': np.full(3, 3.5e34),
        }
    )

    result = orbit_mod.plot_evection(hf_all, '/tmp/out', plot_format='png', t0=100.0)

    assert result is None
    assert not mock_plt.subplots.called


def test_plot_evection_draws_a_four_panel_figure_and_saves(monkeypatch):
    """With sufficient time coverage, must build a 4-row shared-x figure
    and save it under the expected filename."""
    mock_fig = MagicMock()
    axs = _make_axs_4x1()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_evection_hf_all()
    orbit_mod.plot_evection(hf_all, '/tmp/out', plot_format='png', t0=100.0)

    mock_plt.subplots.assert_called_once()
    args, kwargs = mock_plt.subplots.call_args
    assert args[:2] == (4, 1)
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args[0][0]
    assert saved_path.endswith('plot_evection.png')


def test_plot_evection_uses_fine_phi_trace_when_provided(monkeypatch):
    """Panel (c) must plot the supplied fine (t, phi) trace, mod 2*pi,
    in preference to the coarse ``evection_angle`` helpfile column --
    the fine trace is the real per-substep resonance-angle data, the
    coarse column is a potentially-aliased fallback (see the module's
    own log message for the un-mocked branch).
    """
    mock_fig = MagicMock()
    axs = _make_axs_4x1()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_evection_hf_all()
    fine_t = np.array([1e2, 2e2, 3e2])
    fine_phi = np.array([0.1, 7.0, -1.0])  # includes values outside [0, 2*pi)

    orbit_mod.plot_evection(
        hf_all, '/tmp/out', plot_format='png', t0=100.0, fine_phi=fine_phi, fine_t=fine_t
    )

    panel_c_call = axs[2].plot.call_args
    plotted_t, plotted_y = panel_c_call[0]
    np.testing.assert_allclose(plotted_t, fine_t)
    np.testing.assert_allclose(plotted_y, np.mod(fine_phi, 2 * np.pi))
    # Discrimination: must NOT match the coarse evection_angle column
    # (the fallback this branch is supposed to override).
    coarse = np.mod(hf_all['evection_angle'].to_numpy(), 2 * np.pi)
    assert not np.allclose(plotted_y, coarse[: len(plotted_y)])


def test_plot_evection_falls_back_to_coarse_evection_angle_without_fine_trace(monkeypatch):
    """Without a fine trace, panel (c) must fall back to the coarse
    ``evection_angle`` helpfile column, plotted against the full
    ``Time`` array (not some other subset)."""
    mock_fig = MagicMock()
    axs = _make_axs_4x1()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_evection_hf_all()
    orbit_mod.plot_evection(hf_all, '/tmp/out', plot_format='png', t0=100.0)

    plotted_t, plotted_y = axs[2].plot.call_args[0]
    np.testing.assert_allclose(plotted_t, hf_all['Time'].to_numpy())
    assert len(plotted_y) == len(hf_all)


def test_plot_evection_wraps_coarse_angle_when_it_has_circulated(monkeypatch):
    """When the coarse ``evection_angle`` column spans more than a full
    turn (genuine circulation, not bounded libration), panel (c) must
    wrap it into ``[0, 2*pi)`` -- otherwise the raw, unwrapped angle
    would blow past the panel's fixed ``[-0.1, 2*pi+0.1]`` y-limits."""
    mock_fig = MagicMock()
    axs = _make_axs_4x1()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_evection_hf_all()
    # ptp = 20.0 > 2*pi: genuine circulation, not pure libration.
    hf_all['evection_angle'] = np.linspace(0.0, 20.0, len(hf_all))
    orbit_mod.plot_evection(hf_all, '/tmp/out', plot_format='png', t0=100.0)

    _plotted_t, plotted_y = axs[2].plot.call_args[0]
    assert np.all(plotted_y >= 0.0) and np.all(plotted_y < 2 * np.pi)
    # Discrimination: the raw (unwrapped) column reaches 20.0, well
    # outside [0, 2*pi) -- confirms wrapping actually happened, not
    # that the input already happened to be in range.
    assert np.amax(hf_all['evection_angle'].to_numpy()) > 2 * np.pi


@pytest.mark.parametrize('xscale', ['log', 'linear'])
def test_plot_evection_xscale_applies_to_every_panel(monkeypatch, xscale):
    """Both the log and linear x-scale branches must apply their scale
    (and their own distinct axis-limit convention) to every panel via
    ``axs.flat``, not just a subset."""
    mock_fig = MagicMock()
    axs = _make_axs_4x1()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_evection_hf_all()
    orbit_mod.plot_evection(hf_all, '/tmp/out', plot_format='png', t0=100.0, xscale=xscale)

    for ax in axs:
        ax.set_xscale.assert_called_with(xscale)
    # Discrimination: the two branches use distinct xlim conventions
    # (log: [t0, t_max]; linear: [0.0, t_max]) -- confirms the right
    # branch actually ran, not just that SOME xscale string was passed.
    _, xlim_kwargs = axs[0].set_xlim.call_args
    assert xlim_kwargs['left'] == (100.0 if xscale == 'log' else 0.0)


def test_plot_evection_filter_toggle_t_draws_vlines_on_all_time_panels(monkeypatch):
    """When ``filter_toggle_t`` is given, a vertical marker line must be
    drawn on panels (a), (b), (c) and (d) -- confirms the toggle
    annotation is wired to every time-series panel, not just the one
    it was first added to."""
    mock_fig = MagicMock()
    axs = _make_axs_4x1()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    hf_all = _make_evection_hf_all()
    orbit_mod.plot_evection(
        hf_all, '/tmp/out', plot_format='png', t0=100.0, filter_toggle_t=2.5e4
    )

    for ax in axs:
        assert ax.axvline.called
    # Discrimination: panel (c) additionally gets a labeled legend entry
    # for the toggle marker (the other three panels get the bare vline
    # only) -- confirms the annotation isn't a blanket, undifferentiated
    # call across all four axes.
    axs[2].legend.assert_called_with(loc='upper right', fontsize=9, framealpha=0.9)


# ---------------------------------------------------------------------------
# plot_lovenumber
# ---------------------------------------------------------------------------


def _make_lovenumber_ds(real_vals, imag_vals, *, n=2, m=0, k=1, sigma=0.5):
    """A single-mode (n, m, k) tidal-response snapshot, matching the
    dict-of-arrays interface ``plot_Lovenumber`` reads (``ds['n'][:]``,
    ``ds['knms_total']`` as a (2, n_modes) real/imag pair)."""
    return {
        'n': np.array([n]),
        'm': np.array([m]),
        'k': np.array([k]),
        'sigma_range': np.array([sigma]),
        'knms_total': np.array([[real_vals], [imag_vals]]),
    }


def test_plot_lovenumber_returns_early_for_no_times(monkeypatch):
    """Empty or ``None`` times must short-circuit without touching
    matplotlib."""
    mock_plt = _install_mock_plt(monkeypatch)
    assert orbit_mod.plot_lovenumber('/tmp/out', [], []) is None
    assert orbit_mod.plot_lovenumber('/tmp/out', None, []) is None
    assert not mock_plt.subplots.called


def test_plot_lovenumber_returns_early_when_max_time_below_threshold(monkeypatch):
    """Times all below the 2 yr minimum-data threshold must also
    short-circuit without touching matplotlib."""
    mock_plt = _install_mock_plt(monkeypatch)
    data = [_make_lovenumber_ds(1e-2, 2e-3)]
    result = orbit_mod.plot_lovenumber('/tmp/out', [1.0], data)
    assert result is None
    assert not mock_plt.subplots.called


def test_plot_lovenumber_returns_early_when_no_nonzero_love_numbers(monkeypatch, caplog):
    """If every recorded Love number is exactly zero (real and imaginary),
    there is nothing to plot -- must warn and return rather than
    building a figure with degenerate (all -inf) colour bounds."""
    import logging

    mock_plt = _install_mock_plt(monkeypatch)
    data = [_make_lovenumber_ds(0.0, 0.0), _make_lovenumber_ds(0.0, 0.0)]
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.plot.cpl_orbit'):
        result = orbit_mod.plot_lovenumber('/tmp/out', [1000, 2000], data)
    assert result is None
    assert not mock_plt.subplots.called
    assert any('No valid non-zero Love numbers' in rec.message for rec in caplog.records)


def test_plot_lovenumber_draws_and_saves_with_valid_data(monkeypatch):
    """With genuine non-zero Love-number data across two snapshots,
    must build the 2-panel (real, imaginary) scatter figure and save
    it under the expected filename."""
    mock_fig = MagicMock()
    axs = np.empty(2, dtype=object)
    axs[0] = MagicMock()
    axs[1] = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)
    mock_plt.get_cmap.return_value = MagicMock()

    data = [_make_lovenumber_ds(1e-2, 2e-3), _make_lovenumber_ds(1.5e-2, 3e-3)]
    orbit_mod.plot_lovenumber('/tmp/out', [1000, 2000], data, plot_format='png')

    assert axs[0].scatter.call_count == 1
    assert axs[1].scatter.call_count == 1
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args[0][0]
    assert saved_path.endswith('plot_lovenumber.png')


def test_plot_lovenumber_flags_potentially_unbound_points(monkeypatch):
    """A Love number with Re(k) > 1.5 or Im(k) > 1 signals a numerically
    unbound resonance response rather than a physically plausible value;
    the plot must ring that point on both panels in addition to the
    ordinary colour-coded scatter, and add exactly one legend entry
    explaining the marker."""
    mock_fig = MagicMock()
    axs = np.empty(2, dtype=object)
    axs[0] = MagicMock()
    axs[1] = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    # Same mode across two snapshots: the first stays within both
    # thresholds, the second breaches only the real-part threshold
    # (2.0 > 1.5) while its imaginary part (1e-3) stays well inside its
    # own threshold -- so the two conditions must be OR-ed, not AND-ed.
    data = [
        _make_lovenumber_ds(1e-2, 2e-3, n=2, m=0, k=1),
        _make_lovenumber_ds(2.0, 1e-3, n=2, m=0, k=1),
    ]
    orbit_mod.plot_lovenumber('/tmp/out', [1000, 2000], data, plot_format='png')

    # One colour-coded scatter plus one ring-marker scatter per panel.
    assert axs[0].scatter.call_count == 2
    assert axs[1].scatter.call_count == 2
    ring_call = axs[0].scatter.call_args_list[-1]
    assert ring_call.kwargs['edgecolors'] == 'red'
    assert ring_call.kwargs['facecolors'] == 'none'
    # Only the breaching sample (index 1) should be ringed, not both.
    assert len(ring_call.args[0]) == 1
    mock_fig.legend.assert_called_once()


def test_plot_lovenumber_does_not_flag_values_within_bounds(monkeypatch):
    """Values that stay strictly inside both unbound thresholds must not
    trigger the extra ring-marker scatter call, so the indicator does not
    fire on ordinary, well-behaved Love numbers near the boundary."""
    mock_fig = MagicMock()
    axs = np.empty(2, dtype=object)
    axs[0] = MagicMock()
    axs[1] = MagicMock()
    mock_plt = _install_mock_plt(monkeypatch)
    mock_plt.subplots.return_value = (mock_fig, axs)

    # Just under each threshold (1.4 < 1.5, 0.9 < 1) -- a discriminating
    # near-boundary case, not a value trivially far from either cutoff.
    data = [_make_lovenumber_ds(1.0, 0.5), _make_lovenumber_ds(1.4, 0.9)]
    orbit_mod.plot_lovenumber('/tmp/out', [1000, 2000], data, plot_format='png')

    assert axs[0].scatter.call_count == 1
    assert axs[1].scatter.call_count == 1
    mock_fig.legend.assert_called_once()


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


def test_plot_orbit_entry_dispatches_to_plot_evection_for_ps1d_evec(monkeypatch, tmp_path):
    """When ``planet_satellite_model == 'ps1d_evec'``, the entry wrapper
    must also call ``plot_evection`` -- with ``fine_t``/``fine_phi`` left
    ``None`` when no fine-evection CSV exists yet for this run (an early
    call, before the resonance band has produced any fine samples)."""
    fake_hf = _make_hf_all(n=4, t_start=1e3, t_end=1e6)
    captured = {}

    monkeypatch.setattr(orbit_mod.pd, 'read_csv', lambda *a, **kw: fake_hf)
    monkeypatch.setattr(orbit_mod, 'plot_orbit', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'plot_orbit_system', lambda *a, **kw: None)
    monkeypatch.setattr(
        orbit_mod,
        'plot_evection',
        lambda *a, **kw: captured.update(kw),
    )

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path), 'output/data': str(tmp_path / 'data')}
    handler.config.params.out.plot_fmt = 'png'
    handler.config.orbit.planet_satellite_model = 'ps1d_evec'
    handler.config.orbit.module = 'dummy'

    orbit_mod.plot_orbit_entry(handler)

    assert captured['fine_t'] is None
    assert captured['fine_phi'] is None
    assert captured['t0'] == pytest.approx(1e1)
    assert captured['xscale'] == 'linear'


def test_plot_orbit_entry_loads_fine_evection_data_when_present(monkeypatch, tmp_path):
    """When ``fine_evection_data.csv`` already exists for this run, the
    entry wrapper must load it and pass the real (t, phi) trace through
    to ``plot_evection`` as ``fine_t``/``fine_phi``, not leave them
    ``None``."""
    fake_hf = _make_hf_all(n=4, t_start=1e3, t_end=1e6)
    captured = {}

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    fine_path = data_dir / 'fine_evection_data.csv'
    t_vals = np.array([1.0e2, 2.0e2, 3.0e2])
    phi_vals = np.array([0.1, 0.5, 1.0])
    with open(fine_path, 'w') as f:
        f.write('t_abs_yr,phi\n')
        for t, phi in zip(t_vals, phi_vals):
            f.write(f'{t},{phi}\n')

    monkeypatch.setattr(orbit_mod.pd, 'read_csv', lambda *a, **kw: fake_hf)
    monkeypatch.setattr(orbit_mod, 'plot_orbit', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'plot_orbit_system', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'plot_evection', lambda *a, **kw: captured.update(kw))

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path), 'output/data': str(data_dir)}
    handler.config.params.out.plot_fmt = 'png'
    handler.config.orbit.planet_satellite_model = 'ps1d_evec'
    handler.config.orbit.module = 'dummy'

    orbit_mod.plot_orbit_entry(handler)

    np.testing.assert_allclose(captured['fine_t'], t_vals)
    np.testing.assert_allclose(captured['fine_phi'], phi_vals)


def test_plot_orbit_entry_warns_and_continues_when_fine_evection_data_is_malformed(
    monkeypatch, tmp_path, caplog
):
    """A present but unparseable ``fine_evection_data.csv`` must log a
    warning and fall back to ``fine_t=fine_phi=None``, not crash the
    whole plotting entry point over one malformed diagnostic file."""
    import logging

    fake_hf = _make_hf_all(n=4, t_start=1e3, t_end=1e6)
    captured = {}

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    fine_path = data_dir / 'fine_evection_data.csv'
    with open(fine_path, 'w') as f:
        f.write('not,valid,columns\nfor,this,loader\nextra\n')

    monkeypatch.setattr(orbit_mod.pd, 'read_csv', lambda *a, **kw: fake_hf)
    monkeypatch.setattr(orbit_mod, 'plot_orbit', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'plot_orbit_system', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'plot_evection', lambda *a, **kw: captured.update(kw))

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path), 'output/data': str(data_dir)}
    handler.config.params.out.plot_fmt = 'png'
    handler.config.orbit.planet_satellite_model = 'ps1d_evec'
    handler.config.orbit.module = 'dummy'

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.plot.cpl_orbit'):
        orbit_mod.plot_orbit_entry(handler)

    assert captured['fine_t'] is None
    assert captured['fine_phi'] is None
    assert any('Failed to load fine evection data' in rec.message for rec in caplog.records)


def test_plot_orbit_entry_dispatches_to_plot_lovenumber_for_obliqua(monkeypatch, tmp_path):
    """When ``config.orbit.module == 'obliqua'``, the entry wrapper must
    sample the available snapshot times, load their tidal data via
    ``read_tides_data``, and dispatch to ``plot_lovenumber`` with that
    data threaded through."""
    fake_hf = _make_hf_all(n=4, t_start=1e3, t_end=1e6)
    captured = {}

    monkeypatch.setattr(orbit_mod.pd, 'read_csv', lambda *a, **kw: fake_hf)
    monkeypatch.setattr(orbit_mod, 'plot_orbit', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'plot_orbit_system', lambda *a, **kw: None)
    monkeypatch.setattr(orbit_mod, 'sample_output', lambda *a, **kw: ([1000, 2000], None))
    monkeypatch.setattr(
        orbit_mod, 'read_tides_data', lambda *a, **kw: captured.setdefault('data', a) or []
    )
    monkeypatch.setattr(
        orbit_mod,
        'plot_lovenumber',
        lambda *a, **kw: captured.update(kw),
    )

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path), 'output/data': str(tmp_path / 'data')}
    handler.config.params.out.plot_fmt = 'png'
    handler.config.orbit.planet_satellite_model = None
    handler.config.orbit.module = 'obliqua'

    orbit_mod.plot_orbit_entry(handler)

    assert captured['times'] == [1000, 2000]
    assert captured['plot_format'] == 'png'
    # read_tides_data was called with ('obliqua', [1000, 2000]) as the
    # model/times pair (output_dir is the first positional argument).
    assert captured['data'][1] == 'obliqua'
    assert captured['data'][2] == [1000, 2000]
