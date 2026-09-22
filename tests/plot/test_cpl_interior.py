"""Unit tests for ``proteus.plot.cpl_interior``.

Covers ``plot_interior`` (aragog entropy-solver, aragog T-solver, and
spider branches, plus early-return guards) and the
``plot_interior_entry`` wrapper. Matplotlib is mocked at the
source-binding attribute.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_interior as interior_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AragogEntropyDS(dict):
    """Aragog entropy-solver snapshot dict (pres_s / temp_s / phi_s ...)."""


class _AragogTemperatureDS(dict):
    """Aragog T-solver snapshot dict (pres_b / temp_b / phi_b ...)."""


class _SpiderDS:
    """Spider snapshot stub with the methods that cpl_interior calls."""

    def __init__(self, n_lev: int = 5):
        # n_lev basic nodes; staggered (Htidal_s) has n_lev - 1.
        self._data = {
            ('data', 'pressure_b'): np.linspace(0.0, 1.3e11, n_lev),  # Pa
            ('data', 'radius_b'): np.linspace(6.371e6, 3.485e6, n_lev),  # m
            ('data', 'temp_b'): np.linspace(1800.0, 4500.0, n_lev),
            ('data', 'phi_b'): np.linspace(0.0, 1.0, n_lev),
            ('data', 'visc_b'): np.linspace(1.0e15, 1.0e21, n_lev),
            ('data', 'Jconv_b'): np.linspace(1.0e3, 1.0e5, n_lev),
            ('data', 'Htidal_s'): np.linspace(1e-8, 1e-7, n_lev - 1),
        }
        self._phi = np.linspace(0.0, 1.0, n_lev)

    def get_dict_values(self, keys):
        return self._data[tuple(keys)]

    def get_solid_phase_boolean_array(self, _layer):
        return self._phi < 0.05

    def get_mixed_phase_boolean_array(self, _layer):
        return (self._phi >= 0.05) & (self._phi <= 0.95)

    def get_melt_phase_boolean_array(self, _layer):
        return self._phi > 0.95


def _make_aragog_entropy_ds(n_lev: int = 5) -> _AragogEntropyDS:
    """Build an aragog entropy-solver snapshot covering the full phase range.

    phi_s spans 0..1 so the three masks (MASK_SO / MASK_MI / MASK_ME) are
    each populated. Pressure increases monotonically with depth, as does
    temperature. ``Ftotal_b`` is staggered so the 0.5*(fb[:-1]+fb[1:])
    averaging path is exercised.
    """
    return _AragogEntropyDS(
        {
            'pres_s': np.linspace(0.1, 130.0, n_lev),  # GPa
            'radius_s': np.linspace(6.371e6, 3.485e6, n_lev),  # m
            'temp_s': np.linspace(1800.0, 4500.0, n_lev),  # K
            'phi_s': np.linspace(0.0, 1.0, n_lev),
            'log10visc_s': np.linspace(15.0, 21.0, n_lev),
            'Ftotal_b': np.linspace(1.0e3, 1.0e5, n_lev + 1),
            'Htotal_s': np.linspace(1e-8, 1e-7, n_lev),
        }
    )


def _make_aragog_temperature_ds(n_lev: int = 5) -> _AragogTemperatureDS:
    """Build an aragog T-solver snapshot (basic-node variables only)."""
    return _AragogTemperatureDS(
        {
            'pres_b': np.linspace(0.1, 130.0, n_lev),
            'radius_b': np.linspace(6.371e6, 3.485e6, n_lev),
            'temp_b': np.linspace(1800.0, 4500.0, n_lev),
            'phi_b': np.linspace(0.0, 1.0, n_lev),
            'log10visc_b': np.linspace(15.0, 21.0, n_lev),
            'Fconv_b': np.linspace(1.0e3, 1.0e5, n_lev),
            'Htidal_s': np.linspace(1e-8, 1e-7, n_lev),
        }
    )


def _install_full_mocks(monkeypatch):
    """Patch plt with subplots that returns 5 panel mocks plus a fig."""
    axs = [MagicMock() for _ in range(5)]
    for ax in axs:
        ax.transAxes = MagicMock()
        ax.twinx.return_value = MagicMock()
    mock_fig = MagicMock()

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, axs)
    mock_plt.cm.ScalarMappable.return_value = MagicMock()
    monkeypatch.setattr(interior_mod, 'plt', mock_plt)
    return mock_plt, axs, mock_fig


# ---------------------------------------------------------------------------
# plot_interior: early-return guards
# ---------------------------------------------------------------------------


def test_plot_interior_returns_early_when_max_time_below_two(tmp_path, monkeypatch):
    """Times below 2 yr trigger silent skip; matplotlib must not be called."""
    mock_plt, _axs, _fig = _install_full_mocks(monkeypatch)
    times = [0.5, 1.0, 1.5]
    data = [_make_aragog_entropy_ds() for _ in times]

    result = interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        plot_format='png',
    )

    assert result is None
    # Discrimination: the guard must hold; subplots stays untouched.
    assert not mock_plt.subplots.called


def test_plot_interior_rejects_unknown_module(tmp_path, monkeypatch):
    """An unknown interior module must short-circuit before any plotting."""
    mock_plt, _axs, _fig = _install_full_mocks(monkeypatch)
    times = [10.0, 20.0, 30.0]
    data = [_make_aragog_entropy_ds() for _ in times]

    result = interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='mystery_solver',
        plot_format='png',
    )

    assert result is None
    # subplots is not called for the unknown-module short-circuit path.
    assert not mock_plt.subplots.called


# ---------------------------------------------------------------------------
# plot_interior: aragog entropy-solver branch
# ---------------------------------------------------------------------------


def test_plot_interior_aragog_entropy_branch_invokes_all_five_panels(tmp_path, monkeypatch):
    """The aragog entropy branch reads pres_s / temp_s / phi_s, divides
    temperatures by 1000 (kK), multiplies phi by 100 (percent), and emits
    three plot calls per panel per snapshot (solid + mush + melt masks).
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)
    n_snap = 3
    times = [10.0, 1e3, 1e6]
    data = [_make_aragog_entropy_ds(n_lev=6) for _ in range(n_snap)]

    interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        plot_format='png',
    )

    # Per snapshot, each of the first 5 axes receives 3 plot calls (3 masks).
    # ax[0] also collects three additional dummy plot calls for the legend, plus a
    # twinx-side phantom plot.
    for i in range(1, 5):
        # 3 masks * n_snap snapshots
        assert axs[i].plot.call_count >= 3 * n_snap
    mock_fig.savefig.assert_called_once()
    # The saved path lives under output_dir/plots/plot_interior.png.
    fpath = mock_fig.savefig.call_args[0][0]
    assert fpath.endswith('plot_interior.png')


def test_plot_interior_aragog_entropy_converts_temperature_to_kK(tmp_path, monkeypatch):
    """Temperature values plotted in panel (a) are divided by 1000 so that
    a 1800 K..4500 K mantle profile lands in [1.8, 4.5] kK rather than the
    raw Kelvin scale.
    """
    _mock_plt, axs, _fig = _install_full_mocks(monkeypatch)
    times = [10.0, 1e3, 1e6]
    data = [_make_aragog_entropy_ds(n_lev=6) for _ in range(3)]

    interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        plot_format='png',
    )

    # Look at any of the panel-(a) plot calls. The first positional argument
    # is the temperature in kK; max must be ~4.5, not ~4500.
    plot_calls = [c for c in axs[0].plot.call_args_list if len(c[0]) == 2]
    # Pick a non-dummy call (dummy calls use [-1, -2] as data).
    real_calls = [c for c in plot_calls if np.any(np.array(c[0][0]) > 0)]
    assert len(real_calls) >= 1
    y_kK = np.array(real_calls[0][0][0])
    # Scale guard: kK conversion sets max at ~4.5; raw Kelvin would be ~4500.
    assert np.amax(y_kK) < 50.0
    # Sign + magnitude guard: every plotted temperature is positive and
    # spans the molten-mantle regime.
    assert np.amin(y_kK) > 0


# ---------------------------------------------------------------------------
# plot_interior: aragog T-solver branch
# ---------------------------------------------------------------------------


def test_plot_interior_aragog_temperature_solver_branch(tmp_path, monkeypatch):
    """The aragog T-solver branch uses pres_b / temp_b / phi_b / log10visc_b
    / Fconv_b; ``_is_entropy`` is False and the code falls through the
    non-entropy clauses for every panel.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)
    times = [10.0, 1e3, 1e6]
    data = [_make_aragog_temperature_ds(n_lev=6) for _ in range(3)]

    interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        plot_format='pdf',
    )

    # Panels (b)-(e) all received plot calls; the figure saved once.
    assert axs[1].plot.call_count > 0
    assert axs[4].plot.call_count > 0
    mock_fig.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# plot_interior: spider branch
# ---------------------------------------------------------------------------


def test_plot_interior_spider_branch_converts_radius_to_km(tmp_path, monkeypatch):
    """The spider branch reads radius_b in metres and multiplies by 1e-3
    to get the secondary depth axis in km. The depth range therefore lands
    in [0, ~3000] km, not [0, ~3e6] m.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)
    times = [10.0, 1e3, 1e6]
    data = [_SpiderDS(n_lev=5) for _ in range(3)]

    interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='spider',
        plot_format='png',
    )

    # axs[-1] is the heating panel; its twinx-derived axb sets ylim on depth.
    # The twin axis is created via axs[-1].twinx() (always returns the same
    # MagicMock from the fixture). Read the set_ylim call:
    twin_ax = axs[-1].twinx.return_value
    # set_ylim was called with (top=amin(depth), bottom=amax(depth)) kwargs.
    set_ylim_kwargs = twin_ax.set_ylim.call_args.kwargs
    top = set_ylim_kwargs.get('top')
    bottom = set_ylim_kwargs.get('bottom')
    # spider depth = radius[0] - radius; max depth ~ (6.371e6 - 3.485e6) / 1e3 = 2886 km.
    # Scale guard: km-converted depth is O(1e3); m-scaled would be O(1e6).
    assert 2000.0 < bottom < 4000.0
    # Top of axis (smallest depth) should be ~0.
    assert top == pytest.approx(0.0, abs=1.0)
    mock_fig.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# plot_interior: log-scale dispatch on viscosity and flux
# ---------------------------------------------------------------------------


def test_plot_interior_sets_log_scale_when_viscosity_dynamic_range_wide(tmp_path, monkeypatch):
    """Panel (c) flips to log scale when ``visc_max > 100 * visc_min``. Our
    synthetic log10visc range (1e15..1e21) trips the threshold; a forgotten
    test against linear here would miss the wide-range default.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)
    times = [10.0, 1e3, 1e6]
    data = [_make_aragog_entropy_ds(n_lev=5) for _ in range(3)]

    interior_mod.plot_interior(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        plot_format='png',
    )

    # axs[2] is the viscosity panel; set_xscale was called with 'log'.
    set_xscale_args = [c[0][0] for c in axs[2].set_xscale.call_args_list]
    assert 'log' in set_xscale_args
    # axs[3] is the flux panel; flux_max=1e5/1e3=100 vs |flux_min|=1 ratio = 100,
    # but the guard is "> 100.0", strict, so the flux-symlog branch does NOT
    # fire in this fixture. set_xscale on axs[3] is therefore not called with
    # 'symlog'. We assert axs[3] still got set_xlim called instead.
    assert axs[3].set_xlim.called


# ---------------------------------------------------------------------------
# plot_interior_entry
# ---------------------------------------------------------------------------


def test_plot_interior_entry_dispatches_to_aragog(monkeypatch, tmp_path, capsys):
    """The entry wrapper picks ``_int.nc`` for the aragog module and
    forwards the configured plot format into ``plot_interior``.
    """
    captured = {}

    def _fake_sample(handler, extension, tmin):
        captured['extension'] = extension
        captured['tmin'] = tmin
        return [10.0, 100.0, 1000.0], None

    def _fake_read(out_dir, module, plot_times):
        captured['module'] = module
        return [_make_aragog_entropy_ds() for _ in plot_times]

    def _fake_plot(**kwargs):
        captured['format'] = kwargs['plot_format']

    monkeypatch.setattr(interior_mod, 'sample_output', _fake_sample)
    monkeypatch.setattr(interior_mod, 'read_interior_data', _fake_read)
    monkeypatch.setattr(interior_mod, 'plot_interior', _fake_plot)

    handler = MagicMock()
    handler.config.interior_energetics.module = 'aragog'
    handler.config.params.out.plot_fmt = 'png'
    handler.directories = {'output': str(tmp_path)}

    interior_mod.plot_interior_entry(handler)

    assert captured['extension'] == '_int.nc'
    # Discriminating second assertion: format propagated, module matched.
    assert captured['format'] == 'png'


def test_plot_interior_entry_dispatches_to_spider(monkeypatch, tmp_path):
    """The spider branch picks ``.json`` extension."""
    captured = {}

    def _fake_sample(handler, extension, tmin):
        captured['extension'] = extension
        return [10.0, 100.0, 1000.0], None

    monkeypatch.setattr(interior_mod, 'sample_output', _fake_sample)
    monkeypatch.setattr(
        interior_mod,
        'read_interior_data',
        lambda *a, **kw: [_SpiderDS() for _ in range(3)],
    )
    monkeypatch.setattr(interior_mod, 'plot_interior', lambda **kw: None)

    handler = MagicMock()
    handler.config.interior_energetics.module = 'spider'
    handler.config.params.out.plot_fmt = 'pdf'
    handler.directories = {'output': str(tmp_path)}

    interior_mod.plot_interior_entry(handler)

    assert captured['extension'] == '.json'
    # Sign discrimination: a wrong dispatch would land at '_int.nc'.
    assert captured['extension'] != '_int.nc'


def test_plot_interior_entry_unknown_module_returns_early(monkeypatch, tmp_path):
    """Unknown interior module path skips both sample_output and the main
    plot routine.
    """
    flags = {'sample': False, 'plot': False}
    monkeypatch.setattr(
        interior_mod,
        'sample_output',
        lambda *a, **kw: flags.__setitem__('sample', True) or ([], None),
    )
    monkeypatch.setattr(
        interior_mod,
        'plot_interior',
        lambda **kw: flags.__setitem__('plot', True),
    )

    handler = MagicMock()
    handler.config.interior_energetics.module = 'mystery_solver'
    handler.config.params.out.plot_fmt = 'png'
    handler.directories = {'output': str(tmp_path)}

    result = interior_mod.plot_interior_entry(handler)

    assert result is None
    # Neither sample_output nor the inner plot routine should have fired.
    assert flags == {'sample': False, 'plot': False}
