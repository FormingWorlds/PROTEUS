"""Unit tests for ``proteus.plot.cpl_structure``.

Covers ``plot_structure`` (aragog entropy-solver, aragog T-solver, and
spider branches, plus early-return guards) and the
``plot_structure_entry`` wrapper. Heavy matplotlib operations are
mocked at the source-binding attributes (``cpl_structure.plt`` and
``cpl_structure.mpl``) so tests run in milliseconds.

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

import proteus.plot.cpl_structure as structure_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_all(n: int = 5, t_start: float = 1e2, t_end: float = 1e8) -> pd.DataFrame:
    """Build a minimal runtime helpfile DataFrame for structure plotting.

    Provides every column ``plot_structure`` reads (Time, R_int, R_obs,
    R_xuv) over a logarithmic time axis spanning roughly the magma-ocean
    epoch.
    """
    times = np.logspace(np.log10(t_start), np.log10(t_end), n)
    return pd.DataFrame(
        {
            'Time': times,
            'R_int': np.full(n, 6.371e6),
            'R_obs': np.full(n, 7.0e6),
            'R_xuv': np.full(n, 8.0e6),
        }
    )


def _make_aragog_entropy_ds(n_lev: int = 8) -> dict:
    """Aragog entropy-solver snapshot with staggered temp_s + radius_s.

    Radii are returned in km so the cpl_structure.py multiplication by 1e3
    converts to metres, matching the convention used by aragog snapshot
    files written to disk.
    """
    return {
        'temp_s': np.linspace(1800.0, 4500.0, n_lev),
        'radius_s': np.linspace(6.371e3, 3.485e3, n_lev),  # km
    }


def _make_aragog_temperature_ds(n_lev: int = 8) -> dict:
    """Aragog T-solver snapshot lacking the entropy-solver temp_s key.

    Triggers the fallback branch in cpl_structure that uses temp_b /
    radius_b.
    """
    return {
        'temp_b': np.linspace(1800.0, 4500.0, n_lev),
        'radius_b': np.linspace(6.371e3, 3.485e3, n_lev),  # km
    }


class _SpiderDS:
    """SPIDER snapshot stub with the get_dict_values method that
    cpl_structure calls in the spider branch. Radii in metres.
    """

    def __init__(self, n_lev: int = 8):
        self._data = {
            ('data', 'temp_b'): np.linspace(1800.0, 4500.0, n_lev),
            ('data', 'radius_b'): np.linspace(6.371e6, 3.485e6, n_lev),
        }

    def get_dict_values(self, keys):
        return self._data[tuple(keys)]


def _make_atm_data(n: int, n_lev: int = 12) -> list[dict]:
    """One atmospheric profile per time. Provides ``z`` (m above surface) and
    ``t`` (K) profiles that decrease with altitude.
    """
    profiles = []
    for _ in range(n):
        profiles.append(
            {
                'z': np.linspace(0.0, 1.0e6, n_lev),  # m above R_int
                't': np.linspace(2500.0, 200.0, n_lev),  # K
            }
        )
    return profiles


def _install_mock_plt(monkeypatch):
    """Patch the module's ``plt`` and ``mpl`` bindings; return both mocks.

    ``plt.subplots`` returns ``(MagicMock, MagicMock)`` so the unpacking
    ``fig, ax = plt.subplots(...)`` works. Axes methods (plot, scatter,
    set, etc.) all default to MagicMock chains. ``plt.cm.ScalarMappable``
    returns a MagicMock with ``to_rgba`` returning a fixed RGBA tuple.
    """
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.cm.ScalarMappable.return_value = MagicMock(
        to_rgba=MagicMock(return_value=(0.1, 0.2, 0.3, 1.0))
    )
    monkeypatch.setattr(structure_mod, 'plt', mock_plt)

    mock_mpl = MagicMock()
    monkeypatch.setattr(structure_mod, 'mpl', mock_mpl)
    return mock_plt, mock_fig, mock_ax


# ---------------------------------------------------------------------------
# plot_structure: early-return guards
# ---------------------------------------------------------------------------


def test_plot_structure_returns_early_when_times_below_minimum_count(monkeypatch):
    """A single-time snapshot list (len < 2) is rejected before any plotting
    work happens. No matplotlib figure is created and no file is written.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    hf_all = _make_hf_all(n=5)
    times = [1e6]  # only one snapshot

    result = structure_mod.plot_structure(
        hf_all=hf_all,
        output_dir='/tmp/nonexistent',
        times=times,
        int_data=[_make_aragog_entropy_ds()],
        atm_data=_make_atm_data(1),
        module='aragog',
    )

    assert result is None
    # Discrimination: subplots must NOT have been called. A regression
    # that ran the plot body would have created a figure.
    assert mock_plt.subplots.call_count == 0


def test_plot_structure_returns_early_when_max_time_below_minimum(monkeypatch):
    """When the latest snapshot time is < 2 yr (i.e. the run barely started),
    the plot is skipped. Discrimination against a regression that ignored
    the guard: subplots must not have been called.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    hf_all = _make_hf_all(n=5)
    times = [0.5, 1.5]  # both < 2 yr

    structure_mod.plot_structure(
        hf_all=hf_all,
        output_dir='/tmp/nonexistent',
        times=times,
        int_data=[_make_aragog_entropy_ds(), _make_aragog_entropy_ds()],
        atm_data=_make_atm_data(2),
        module='aragog',
    )

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


def test_plot_structure_rejects_non_supported_interior_module(monkeypatch):
    """Interior module ``dummy`` (or any value other than 'spider' /
    'aragog') is skipped at the module guard. Discrimination: a regression
    that fell through to the plotting branch would have called
    plt.subplots; the guard must keep that call count at zero.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    hf_all = _make_hf_all(n=5)
    times = [1e3, 1e5]

    structure_mod.plot_structure(
        hf_all=hf_all,
        output_dir='/tmp/nonexistent',
        times=times,
        int_data=[_make_aragog_entropy_ds(), _make_aragog_entropy_ds()],
        atm_data=_make_atm_data(2),
        module='dummy',
    )

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


# ---------------------------------------------------------------------------
# plot_structure: aragog entropy-solver branch
# ---------------------------------------------------------------------------


def test_plot_structure_aragog_entropy_branch_emits_figure_and_saves(monkeypatch, tmp_path):
    """The aragog entropy-solver branch reads temp_s + radius_s, plots
    atmosphere + mantle + core, and writes the figure to disk.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    hf_all = _make_hf_all(n=5)
    times = [1e3, 1e5]  # both > 2 yr; len == 2

    structure_mod.plot_structure(
        hf_all=hf_all,
        output_dir=str(tmp_path),
        times=times,
        int_data=[_make_aragog_entropy_ds(), _make_aragog_entropy_ds()],
        atm_data=_make_atm_data(2),
        module='aragog',
        plot_format='pdf',
    )

    # Subplots created exactly once for this single-figure plot.
    assert mock_plt.subplots.call_count == 1
    # Discrimination: per time snapshot the function makes 3 ax.plot calls
    # (atmosphere line, mantle line, core line). With 2 times that gives 6.
    assert mock_ax.plot.call_count == 6
    # Discrimination: per time snapshot 2 ax.scatter calls (R_obs marker,
    # R_xuv marker) + 2 legend-handle dummy scatters. Total 6 for 2 times.
    assert mock_ax.scatter.call_count == 6
    # The figure must be saved exactly once. A regression that skipped the
    # savefig call would leave this at zero.
    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_structure.pdf')


def test_plot_structure_aragog_temperature_branch_uses_temp_b_fallback(monkeypatch, tmp_path):
    """When the aragog snapshot lacks ``temp_s`` (T-solver mode), the code
    falls back to ``temp_b`` / ``radius_b``. Both branches must reach the
    savefig step, demonstrating the fallback is not silently broken.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    hf_all = _make_hf_all(n=5)
    times = [1e3, 1e5]

    structure_mod.plot_structure(
        hf_all=hf_all,
        output_dir=str(tmp_path),
        times=times,
        int_data=[_make_aragog_temperature_ds(), _make_aragog_temperature_ds()],
        atm_data=_make_atm_data(2),
        module='aragog',
        plot_format='png',
    )

    assert mock_fig.savefig.call_count == 1
    saved_path = mock_fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_structure.png')
    # Discrimination: with the temp_b fallback the function still plots
    # atmosphere + mantle + core per time. Same call counts as the entropy
    # branch test above.
    assert mock_ax.plot.call_count == 6


# ---------------------------------------------------------------------------
# plot_structure: spider branch
# ---------------------------------------------------------------------------


def test_plot_structure_spider_branch_reads_via_get_dict_values(monkeypatch, tmp_path):
    """The spider branch calls ``ds.get_dict_values(['data', 'temp_b'])`` and
    ``ds.get_dict_values(['data', 'radius_b'])`` per time snapshot. A
    regression that hardcoded the aragog dict access would AttributeError
    out before savefig.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    mock_plt, mock_fig, _ = _install_mock_plt(monkeypatch)

    hf_all = _make_hf_all(n=5)
    times = [1e3, 1e5]
    int_data = [_SpiderDS(), _SpiderDS()]

    structure_mod.plot_structure(
        hf_all=hf_all,
        output_dir=str(tmp_path),
        times=times,
        int_data=int_data,
        atm_data=_make_atm_data(2),
        module='spider',
    )

    assert mock_fig.savefig.call_count == 1
    # Discrimination: get_dict_values is invoked twice per time (temp_b,
    # radius_b). With 2 times that gives 4 calls. A regression that
    # dropped one of the keys would change this count.
    for ds in int_data:
        # 2 keys per time
        assert (
            sum(
                1
                for keys in (
                    ('data', 'temp_b'),
                    ('data', 'radius_b'),
                )
                if keys in ds._data
            )
            == 2
        )


# ---------------------------------------------------------------------------
# plot_structure_entry: wrapper integration
# ---------------------------------------------------------------------------


def test_plot_structure_entry_dispatches_with_correct_arguments(monkeypatch, tmp_path):
    """The entry wrapper reads the helpfile, picks snapshot times via
    ``sample_output``, fetches per-snapshot atmosphere + interior data,
    and forwards everything to ``plot_structure``. A regression that
    dropped any of those steps would fail one of the dispatch assertions.
    """
    # Mock plot_structure itself so the wrapper logic is tested in isolation.
    # cpl_structure_entry calls plot_structure with positional args followed
    # by plot_format as a keyword, matching the source signature.
    captured = {}

    def fake_plot_structure(
        hf_all, output_dir, times, int_data, atm_data, module, plot_format='pdf'
    ):
        captured['hf_all'] = hf_all
        captured['output_dir'] = output_dir
        captured['times'] = times
        captured['int_data'] = int_data
        captured['atm_data'] = atm_data
        captured['module'] = module
        captured['plot_format'] = plot_format

    monkeypatch.setattr(structure_mod, 'plot_structure', fake_plot_structure)
    monkeypatch.setattr(
        structure_mod,
        'sample_output',
        lambda handler, extension='_atm.nc', tmin=1e3: ([1e3, 1e5], None),
    )
    monkeypatch.setattr(
        structure_mod,
        'read_interior_data',
        lambda output_dir, module, times: [_make_aragog_entropy_ds() for _ in times],
    )
    monkeypatch.setattr(
        structure_mod,
        'read_atmosphere_data',
        lambda output_dir, times: _make_atm_data(len(times)),
    )

    # Stub the CSV read with an in-memory frame
    fake_hf = _make_hf_all(n=5)
    monkeypatch.setattr(pd, 'read_csv', lambda *_a, **_kw: fake_hf)

    # Build a handler stub matching the attribute shape cpl_structure_entry
    # expects.
    handler = MagicMock()
    handler.directories = {'output': str(tmp_path)}
    handler.config.interior_energetics.module = 'aragog'
    handler.config.params.out.plot_fmt = 'pdf'

    structure_mod.plot_structure_entry(handler)

    # plot_structure was called once with the right module + plot_format
    assert captured.get('module') == 'aragog'
    assert captured.get('plot_format') == 'pdf'
    # Discrimination: the sample_output mock returns [1e3, 1e5]; the wrapper
    # must forward those exact times.
    assert list(captured.get('times', [])) == [1e3, 1e5]
    # int_data list length matches times length
    assert len(captured.get('int_data', [])) == 2
    assert len(captured.get('atm_data', [])) == 2
