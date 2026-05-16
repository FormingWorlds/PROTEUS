"""Unit tests for ``proteus.plot.cpl_interior_cmesh``.

Covers ``plot_interior_cmesh`` (aragog T-solver branch, aragog
entropy-solver branch, spider branch, contour vs pcolormesh) and the
``plot_interior_cmesh_entry`` wrapper. Matplotlib is mocked at the
source-binding attribute; numerical data structures are small synthetic
arrays.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_interior_cmesh as cmesh_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AragogEntropyDS(dict):
    """Mimic the aragog entropy-solver snapshot dict.

    Keys match ``pres_s`` / ``temp_s`` / ``phi_s`` / ``log10visc_s`` /
    ``Ftotal_b``; values are 1D arrays so the production code's broadcasting
    semantics line up with real data.
    """


class _AragogTemperatureDS(dict):
    """Mimic the aragog T-solver snapshot dict (pres_b / temp_b / etc)."""


class _SpiderDS:
    """Mimic the ``spider_coupler_utils.spider`` snapshot interface.

    ``get_dict_values`` is the single getter that ``cpl_interior_cmesh``
    calls for every quantity. The shape (n_levels,) is consistent across
    pressure, temperature, melt fraction, viscosity, and flux.
    """

    def __init__(self, pressure_pa, temp_b, phi_b, visc_b, jconv_b):
        self._data = {
            ('data', 'pressure_b'): pressure_pa,
            ('data', 'temp_b'): temp_b,
            ('data', 'phi_b'): phi_b,
            ('data', 'visc_b'): visc_b,
            ('data', 'Jconv_b'): jconv_b,
        }

    def get_dict_values(self, keys):
        return self._data[tuple(keys)]


def _make_aragog_entropy_dataset(n_lev: int = 4):
    """Build a minimal aragog entropy snapshot with discriminating values."""
    return _AragogEntropyDS(
        {
            'pres_s': np.linspace(0.0, 130.0, n_lev),  # GPa
            'temp_s': np.linspace(1800.0, 4500.0, n_lev),  # K
            'phi_s': np.linspace(0.0, 1.0, n_lev),  # 0..1
            'log10visc_s': np.linspace(15.0, 21.0, n_lev),  # log10 Pa s
            # Ftotal_b is staggered (n_lev + 1 values); the routine averages
            # adjacent pairs down to n_lev.
            'Ftotal_b': np.linspace(1.0e3, 1.0e5, n_lev + 1),  # W/m^2
        }
    )


def _make_aragog_temperature_dataset(n_lev: int = 4):
    """Build a minimal aragog T-solver snapshot (no entropy keys)."""
    return _AragogTemperatureDS(
        {
            'pres_b': np.linspace(0.0, 130.0, n_lev),
            'temp_b': np.linspace(1800.0, 4500.0, n_lev),
            'phi_b': np.linspace(0.0, 1.0, n_lev),
            'log10visc_b': np.linspace(15.0, 21.0, n_lev),
            'Fconv_b': np.linspace(1.0e3, 1.0e5, n_lev),
        }
    )


def _make_spider_dataset(n_lev: int = 4):
    """Build a minimal spider snapshot wrapped behind ``get_dict_values``."""
    return _SpiderDS(
        pressure_pa=np.linspace(0.0, 1.3e11, n_lev),  # Pa, converted to GPa via *1e-9
        temp_b=np.linspace(1800.0, 4500.0, n_lev),
        phi_b=np.linspace(0.0, 1.0, n_lev),
        visc_b=np.linspace(1.0e15, 1.0e21, n_lev),
        jconv_b=np.linspace(1.0e3, 1.0e5, n_lev),
    )


def _install_full_mocks(monkeypatch, n_panels: int = 4):
    """Patch ``plt``, ``make_axes_locatable``, and the cmcrameri cmaps used
    by the routine. Returns (mock_plt, axs, mock_fig) tuple for assertion.
    """
    axs = [MagicMock() for _ in range(n_panels)]
    for ax in axs:
        ax.transAxes = MagicMock()
    mock_fig = MagicMock()

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, axs)
    monkeypatch.setattr(cmesh_mod, 'plt', mock_plt)

    # make_axes_locatable returns a divider whose append_axes returns an Axes;
    # the colorbar then uses that Axes. None of those are exercised numerically.
    monkeypatch.setattr(
        cmesh_mod, 'make_axes_locatable', lambda _ax: MagicMock()
    )

    return mock_plt, axs, mock_fig


# ---------------------------------------------------------------------------
# plot_interior_cmesh: early-return branch
# ---------------------------------------------------------------------------


def test_plot_interior_cmesh_returns_early_with_under_three_samples(
    tmp_path, monkeypatch
):
    """The routine demands at least 3 time samples to make a useful colourmesh.
    A 2-sample input must skip without touching matplotlib.
    """
    mock_plt, _axs, _fig = _install_full_mocks(monkeypatch)

    result = cmesh_mod.plot_interior_cmesh(
        output_dir=str(tmp_path),
        times=[1.0, 2.0],
        data=[_make_aragog_entropy_dataset(), _make_aragog_entropy_dataset()],
        module='aragog',
        plot_format='png',
    )

    assert result is None
    # Discrimination: the early-return path must not invoke subplots.
    # A regression that flipped the < 3 guard to < 2 would call subplots.
    assert not mock_plt.subplots.called


# ---------------------------------------------------------------------------
# plot_interior_cmesh: aragog entropy-solver branch
# ---------------------------------------------------------------------------


def test_plot_interior_cmesh_aragog_entropy_contour_branch(tmp_path, monkeypatch):
    """With aragog entropy-solver outputs and ``use_contour=True`` (the
    default) the routine averages the staggered ``Ftotal_b`` array down to
    the basic-node count and emits one contourf call per panel.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)

    n_samples = 5
    data = [_make_aragog_entropy_dataset(n_lev=4) for _ in range(n_samples)]
    times = list(range(n_samples))

    cmesh_mod.plot_interior_cmesh(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        use_contour=True,
        plot_format='pdf',
    )

    # Four panels, each receives exactly one contourf call. A regression that
    # accidentally fell through to pcolormesh would leave contourf at 0.
    contourf_per_panel = [ax.contourf.call_count for ax in axs]
    assert contourf_per_panel == [1, 1, 1, 1]
    pcolormesh_per_panel = [ax.pcolormesh.call_count for ax in axs]
    assert pcolormesh_per_panel == [0, 0, 0, 0]

    mock_fig.savefig.assert_called_once()
    fpath = mock_fig.savefig.call_args[0][0]
    assert fpath.endswith('plot_interior_cmesh.pdf')


def test_plot_interior_cmesh_aragog_entropy_pcolormesh_branch(tmp_path, monkeypatch):
    """With ``use_contour=False`` every panel renders via pcolormesh; the
    fourth (flux) panel uses ``extend='min'``.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)

    n_samples = 4
    data = [_make_aragog_entropy_dataset(n_lev=3) for _ in range(n_samples)]
    times = list(range(n_samples))

    cmesh_mod.plot_interior_cmesh(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        use_contour=False,
        plot_format='png',
    )

    assert [ax.pcolormesh.call_count for ax in axs] == [1, 1, 1, 1]
    assert [ax.contourf.call_count for ax in axs] == [0, 0, 0, 0]

    # The fourth pcolormesh call must include extend='min' for the log-scaled
    # convective flux. A regression dropping this leaves the under-flow black.
    flux_kwargs = axs[3].pcolormesh.call_args.kwargs
    assert flux_kwargs.get('extend') == 'min'
    mock_fig.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# plot_interior_cmesh: aragog T-solver branch
# ---------------------------------------------------------------------------


def test_plot_interior_cmesh_aragog_temperature_solver_branch(tmp_path, monkeypatch):
    """The T-solver branch uses ``pres_b`` / ``temp_b`` / ``phi_b`` /
    ``log10visc_b`` / ``Fconv_b`` and skips the staggered-flux averaging.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)

    n_samples = 3
    data = [_make_aragog_temperature_dataset(n_lev=4) for _ in range(n_samples)]
    times = [1.0, 5.0, 10.0]

    cmesh_mod.plot_interior_cmesh(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='aragog',
        use_contour=True,
        plot_format='png',
    )

    # Every panel drew at least once and the figure saved.
    assert all(ax.contourf.call_count == 1 for ax in axs)
    mock_fig.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# plot_interior_cmesh: spider branch
# ---------------------------------------------------------------------------


def test_plot_interior_cmesh_spider_pressure_converted_to_GPa(tmp_path, monkeypatch):
    """The spider branch reads ``pressure_b`` in Pa and multiplies by 1e-9 to
    obtain the GPa y-axis. A regression that dropped the conversion would
    leave the y-ticks at ~1e11 instead of ~130.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)

    n_samples = 3
    data = [_make_spider_dataset(n_lev=4) for _ in range(n_samples)]
    times = [10.0, 20.0, 30.0]

    cmesh_mod.plot_interior_cmesh(
        output_dir=str(tmp_path),
        times=times,
        data=data,
        module='spider',
        use_contour=True,
        plot_format='png',
    )

    # The y-axis ylim is set on every panel to (y_max_rounded, y_min_rounded).
    # Read back the (top, bottom) tuple passed to ax.set_ylim.
    set_ylim_args = axs[0].set_ylim.call_args[0][0]
    top, bottom = set_ylim_args
    # bottom is at the surface (0 GPa), top at ~130 GPa for our synthetic grid.
    # The bottom value should be 0 (rounded from 0.0).
    assert bottom == pytest.approx(0.0, abs=1e-9)
    # Scale guard: top should be O(100) GPa, not O(1e11) Pa.
    assert 50.0 < top < 200.0
    mock_fig.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# plot_interior_cmesh: viscosity log/linear norm dispatch
# ---------------------------------------------------------------------------


def test_plot_interior_cmesh_viscosity_logs_when_dynamic_range_exceeds_100x(
    tmp_path, monkeypatch
):
    """The viscosity colormap uses ``LogNorm`` when ``max > 100 * min`` and
    ``Normalize`` otherwise. Our synthetic log10visc range (1e15..1e21)
    spans 6 orders of magnitude; the dispatch must select the log norm.
    """
    _mock_plt, axs, mock_fig = _install_full_mocks(monkeypatch)

    captured_norms = []
    original_pcolormesh = axs[2].pcolormesh

    def _record_norm(*args, **kwargs):
        captured_norms.append(kwargs.get('norm'))
        return original_pcolormesh(*args, **kwargs)

    axs[2].pcolormesh = _record_norm

    n_samples = 3
    data = [_make_aragog_entropy_dataset(n_lev=4) for _ in range(n_samples)]
    cmesh_mod.plot_interior_cmesh(
        output_dir=str(tmp_path),
        times=list(range(n_samples)),
        data=data,
        module='aragog',
        use_contour=False,
        plot_format='png',
    )

    # Exactly one viscosity pcolormesh call was made on axs[2]; its norm is
    # the LogNorm chosen by the dynamic-range branch.
    assert len(captured_norms) == 1
    norm_cls_name = type(captured_norms[0]).__name__
    assert norm_cls_name == 'LogNorm'
    mock_fig.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# plot_interior_cmesh_entry
# ---------------------------------------------------------------------------


def test_plot_interior_cmesh_entry_dispatches_to_aragog(monkeypatch, tmp_path):
    """The entry wrapper looks at ``handler.config.interior_energetics.module``
    and dispatches to ``read_interior_data`` with the correct extension
    (``_int.nc`` for aragog, ``.json`` for spider).
    """
    captured = {}

    def _fake_sample_output(handler, extension, tmin, nsamp):
        captured['extension'] = extension
        captured['tmin'] = tmin
        return [1e3, 1e4, 1e5], None

    def _fake_read_interior_data(out_dir, module, plot_times):
        captured['module'] = module
        captured['n_times'] = len(plot_times)
        return [_make_aragog_entropy_dataset() for _ in plot_times]

    def _fake_plot_interior_cmesh(**kwargs):
        captured['called_main'] = True
        captured['format'] = kwargs.get('plot_format')

    monkeypatch.setattr(cmesh_mod, 'sample_output', _fake_sample_output)
    monkeypatch.setattr(cmesh_mod, 'read_interior_data', _fake_read_interior_data)
    monkeypatch.setattr(cmesh_mod, 'plot_interior_cmesh', _fake_plot_interior_cmesh)

    handler = MagicMock()
    handler.config.interior_energetics.module = 'aragog'
    handler.config.params.out.plot_fmt = 'png'
    handler.directories = {'output': str(tmp_path)}

    cmesh_mod.plot_interior_cmesh_entry(handler)

    assert captured['extension'] == '_int.nc'
    assert captured['module'] == 'aragog'
    # Discriminating second assertion: the plot_format propagated through.
    assert captured['format'] == 'png'


def test_plot_interior_cmesh_entry_dispatches_to_spider(monkeypatch, tmp_path):
    """The spider branch of the entry wrapper picks ``.json`` extension."""
    captured = {}

    def _fake_sample_output(handler, extension, tmin, nsamp):
        captured['extension'] = extension
        return [1e3, 1e4, 1e5], None

    monkeypatch.setattr(cmesh_mod, 'sample_output', _fake_sample_output)
    monkeypatch.setattr(
        cmesh_mod,
        'read_interior_data',
        lambda *a, **kw: [_make_spider_dataset() for _ in range(3)],
    )
    monkeypatch.setattr(cmesh_mod, 'plot_interior_cmesh', lambda **kw: None)

    handler = MagicMock()
    handler.config.interior_energetics.module = 'spider'
    handler.config.params.out.plot_fmt = 'pdf'
    handler.directories = {'output': str(tmp_path)}

    cmesh_mod.plot_interior_cmesh_entry(handler)

    assert captured['extension'] == '.json'
    # Discriminating sign guard: the wrong dispatch would land at '_int.nc'.
    assert captured['extension'] != '_int.nc'


def test_plot_interior_cmesh_entry_unknown_module_returns_early(monkeypatch, tmp_path):
    """An unknown interior module name must produce a warning and skip the
    plot without invoking ``read_interior_data`` or the inner plot routine.
    """
    flags = {'sample': False, 'read': False, 'plot': False}

    def _fail_sample(*a, **kw):
        flags['sample'] = True
        return [], None

    monkeypatch.setattr(cmesh_mod, 'sample_output', _fail_sample)
    monkeypatch.setattr(
        cmesh_mod,
        'read_interior_data',
        lambda *a, **kw: flags.__setitem__('read', True) or [],
    )
    monkeypatch.setattr(
        cmesh_mod,
        'plot_interior_cmesh',
        lambda **kw: flags.__setitem__('plot', True),
    )

    handler = MagicMock()
    handler.config.interior_energetics.module = 'mystery_solver'
    handler.config.params.out.plot_fmt = 'png'
    handler.directories = {'output': str(tmp_path)}

    result = cmesh_mod.plot_interior_cmesh_entry(handler)

    assert result is None
    # None of the downstream stages must fire on the unknown-module branch.
    assert flags == {'sample': False, 'read': False, 'plot': False}
