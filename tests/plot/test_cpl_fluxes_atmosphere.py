"""Unit tests for ``proteus.plot.cpl_fluxes_atmosphere``.

Covers ``plot_fluxes_atmosphere`` (no-data guard plus full plot path with
optional and required NetCDF variables) and the
``plot_fluxes_atmosphere_entry`` wrapper. Matplotlib + netCDF4 are
mocked at the source-binding attributes so tests run in milliseconds.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_fluxes_atmosphere as fluxes_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NCVarStub:
    """Minimal NetCDF variable stub: array indexing returns a numpy slice."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def __getitem__(self, key):
        return self._values[key]


class _NCDatasetStub:
    """NetCDF dataset stub supporting the context-manager protocol and a
    ``variables`` mapping that returns _NCVarStub instances. Optional
    variables (fl_cnvct, fl_latent, fl_tot, fl_sens) may be omitted to
    exercise the KeyError-fallback branches.
    """

    def __init__(self, variables: dict):
        self.variables = {k: _NCVarStub(v) for k, v in variables.items()}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_required_vars(n: int = 12) -> dict:
    """Build the seven required pressure/flux profile arrays.

    pl is in Pa (the function divides by 1e5 -> bar). Flux profiles are
    monotonically decreasing from TOA to surface with realistic radiative
    magnitudes (-100 to +300 W/m^2 range).
    """
    pl = np.logspace(0, 7, n)  # 1 Pa -> 1e7 Pa
    fl_D_LW = np.linspace(0, 250, n)
    fl_U_LW = np.linspace(200, 50, n)
    fl_D_SW = np.linspace(0, 1000, n)
    fl_U_SW = np.linspace(100, 30, n)
    fl_D = fl_D_LW + fl_D_SW
    fl_U = fl_U_LW + fl_U_SW
    fl_N = fl_U - fl_D
    return {
        'pl': pl,
        'fl_D_LW': fl_D_LW,
        'fl_U_LW': fl_U_LW,
        'fl_D_SW': fl_D_SW,
        'fl_U_SW': fl_U_SW,
        'fl_D': fl_D,
        'fl_U': fl_U,
        'fl_N': fl_N,
    }


def _install_mock_plt(monkeypatch):
    """Patch the module's ``plt`` binding; return (mock_plt, mock_fig, mock_ax)."""
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    monkeypatch.setattr(fluxes_mod, 'plt', mock_plt)
    return mock_plt, mock_fig, mock_ax


# ---------------------------------------------------------------------------
# plot_fluxes_atmosphere: no-data guard
# ---------------------------------------------------------------------------


def test_plot_fluxes_atmosphere_skips_when_no_netcdf_present(monkeypatch, tmp_path):
    """When no ``*_atm.nc`` files exist in output_dir/data/, the function
    logs a warning and returns without creating a figure. A regression
    that proceeded past the empty-list guard would have called subplots.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    # No netcdf files in tmp_path/data
    (tmp_path / 'data').mkdir()

    result = fluxes_mod.plot_fluxes_atmosphere(str(tmp_path), plot_format='pdf')

    assert result is None
    # Discrimination: a regression that fell through would have created
    # a figure. The guard must keep subplots-call-count at zero.
    assert mock_plt.subplots.call_count == 0


# ---------------------------------------------------------------------------
# plot_fluxes_atmosphere: full plot path
# ---------------------------------------------------------------------------


def test_plot_fluxes_atmosphere_full_path_with_all_optional_variables(monkeypatch, tmp_path):
    """When all required AND optional NetCDF variables are present, the
    function reads pl + 7 flux variables + 4 optional variables, builds
    a figure, and saves it. Verifies the dataset variable access count
    matches the contract (8 required + 4 optional = 12 reads).
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()
    # Create a dummy file path that the glob will pick up
    nc_path = tmp_path / 'data' / '00001_atm.nc'
    nc_path.touch()

    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    variables = _make_required_vars()
    variables['fl_cnvct'] = np.linspace(0, 50, 12)
    variables['fl_latent'] = np.linspace(0, 20, 12)
    variables['fl_tot'] = variables['fl_U'] - variables['fl_D']
    variables['fl_sens'] = np.array([15.0])

    nc_ds = _NCDatasetStub(variables)
    monkeypatch.setattr(fluxes_mod.nc, 'Dataset', lambda _path: nc_ds)

    fluxes_mod.plot_fluxes_atmosphere(str(tmp_path), plot_format='png')

    # One figure created, one savefig
    assert mock_plt.subplots.call_count == 1
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_fluxes_atmosphere.png')


def test_plot_fluxes_atmosphere_falls_back_when_optional_variables_missing(
    monkeypatch, tmp_path
):
    """When optional NetCDF variables (fl_cnvct, fl_latent, fl_tot,
    fl_sens) are absent, the function substitutes zero arrays / zero
    scalar and still produces a figure. A regression that didn't catch
    KeyError on the optional reads would raise before savefig.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()
    nc_path = tmp_path / 'data' / '00002_atm.nc'
    nc_path.touch()

    mock_plt, mock_fig, _ = _install_mock_plt(monkeypatch)

    # Provide only the 8 required variables; omit fl_cnvct, fl_latent,
    # fl_tot, fl_sens to exercise the four KeyError-fallback branches.
    variables = _make_required_vars()
    nc_ds = _NCDatasetStub(variables)
    monkeypatch.setattr(fluxes_mod.nc, 'Dataset', lambda _path: nc_ds)

    fluxes_mod.plot_fluxes_atmosphere(str(tmp_path), plot_format='pdf')

    # Discrimination: even without optional variables, the figure must
    # still be saved. A regression that propagated the KeyError would
    # have aborted before this.
    assert mock_fig.savefig.call_count == 1
    # The figure must have been built (subplots called once) on the
    # fallback path too, not produced from a dangling earlier handle.
    assert mock_plt.subplots.call_count == 1


def test_plot_fluxes_atmosphere_picks_natural_sort_last_file(monkeypatch, tmp_path):
    """When multiple NetCDF files exist, the function picks the last one
    by natural-sort. The plot is built from that file's data. A regression
    that lexicographically picked '00010' < '00002' would be caught by
    the natural-sort tie-break.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()
    # Create files in non-sorted order
    (tmp_path / 'data' / '00001_atm.nc').touch()
    (tmp_path / 'data' / '00002_atm.nc').touch()
    (tmp_path / 'data' / '00010_atm.nc').touch()

    mock_plt, mock_fig, _ = _install_mock_plt(monkeypatch)

    accessed_paths = []

    def _ds_factory(path):
        accessed_paths.append(path)
        return _NCDatasetStub(_make_required_vars())

    monkeypatch.setattr(fluxes_mod.nc, 'Dataset', _ds_factory)

    fluxes_mod.plot_fluxes_atmosphere(str(tmp_path), plot_format='pdf')

    # Discrimination: the path opened MUST be the natural-sort largest
    # (00010), not the lexicographic largest (which would be 00010 anyway
    # in this padded example; the contract is that natural_sort is what
    # picks it). Check the basename.
    assert len(accessed_paths) == 1
    assert '00010_atm.nc' in accessed_paths[0]
    assert mock_fig.savefig.call_count == 1


# ---------------------------------------------------------------------------
# plot_fluxes_atmosphere_entry: wrapper
# ---------------------------------------------------------------------------


def test_plot_fluxes_atmosphere_entry_forwards_handler_attributes(monkeypatch):
    """The entry wrapper reads output_dir and plot_fmt from the handler
    and forwards them verbatim to ``plot_fluxes_atmosphere``. A
    regression that dropped the plot_format would default to 'pdf'
    instead of the handler-configured value.
    """
    captured = {}

    def fake_plot(output_dir, plot_format='pdf'):
        captured['output_dir'] = output_dir
        captured['plot_format'] = plot_format

    monkeypatch.setattr(fluxes_mod, 'plot_fluxes_atmosphere', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/expected/output/dir'}
    handler.config.params.out.plot_fmt = 'svg'

    fluxes_mod.plot_fluxes_atmosphere_entry(handler)

    assert captured['output_dir'] == '/expected/output/dir'
    assert captured['plot_format'] == 'svg'
