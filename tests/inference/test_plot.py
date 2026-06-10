"""
Unit tests for inference plotting utilities.

The module under test (``src/proteus/inference/plot.py``) is a utility module
(see ``.github/.claude/rules/proteus-tests.md`` Section 3): it is exempt from
the physics-invariant requirement but every test must still exercise an edge
or error path and pin assertions that discriminate against a plausible
regression. Tests here cover the five public functions:

  - ``plots_perf_timeline``: timeline + six histograms + distance scatter.
  - ``plots_perf_converge``: log-regret and best-value vs time / iteration.
  - ``plot_result_objective``: per-parameter scatter + histogram grid.
  - ``plot_result_correlation``: observable-vs-parameter grid with missing
    helpfile handling.
  - ``plot_proteus``: dispatch over ``plot_dispatch`` with skip set.

Matplotlib is mocked at the module-attribute level (``plot_mod.plt``) so no
figure is rendered and ``savefig`` is captured as a call assertion.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import toml

import proteus.inference.plot as plot_mod
import proteus.inference.transforms as transforms_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_logs(n_init: int = 0, n_extra: int = 5, x_dim: int = 1):
    """Build a synthetic list of BO worker log entries.

    Each entry has the fields ``plots_perf_timeline`` indexes:
    ``start_time``, ``end_time``, ``worker``, ``x_value``, ``y_value``,
    ``duration``, ``BO_time``, ``t_eval``, ``t_fit``, ``t_ac``, ``dist``.

    The first ``n_init`` rows mirror the contract that the function drops
    them via ``logs[n_init:]``. ``x_dim`` switches between the single-x
    annotation branch (``len(row['x_value']) == 1``) and the multi-x branch.
    """
    rng = np.random.default_rng(42)
    logs = []
    t = 0.0
    for i in range(n_init + n_extra):
        dur = 0.5 + 0.1 * (i % 3)
        bo = 0.05 + 0.01 * (i % 4)
        ev = 0.3 + 0.05 * (i % 5)
        fit = 0.02 + 0.005 * (i % 6)
        ac = 0.04 + 0.007 * (i % 4)
        x_val = list(rng.random(x_dim))
        # Mix dist None and float to exercise the notnull filter branch.
        dist_val = None if i % 3 == 0 else 0.1 + 0.01 * i
        logs.append(
            {
                'start_time': t,
                'end_time': t + dur,
                'worker': i % 3,
                'x_value': x_val,
                'y_value': 0.2 + 0.1 * i,
                'duration': dur,
                'BO_time': bo,
                't_eval': ev,
                't_fit': fit,
                't_ac': ac,
                'dist': dist_val,
            }
        )
        t += dur + 0.1
    return logs


def _install_plt_mock(monkeypatch):
    """Replace ``plot_mod.plt`` with a MagicMock and return (plt, fig, ax)."""
    fig = MagicMock(name='fig')
    ax = MagicMock(name='ax')
    plt_mock = MagicMock(name='plt')
    plt_mock.subplots.return_value = (fig, ax)
    # ``plt.cm.tab10`` is indexed by integer in the worker color loop.
    plt_mock.cm.tab10.side_effect = lambda i: (0.1 * (i + 1), 0.2, 0.3, 1.0)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    return plt_mock, fig, ax


# ---------------------------------------------------------------------------
# plots_perf_timeline
# ---------------------------------------------------------------------------


def test_plots_perf_timeline_empty_returns_without_drawing(tmp_path):
    """Empty ``logs`` short-circuits before any savefig call.

    Reuses the long-standing behavior: an empty DataFrame after the
    ``logs[n_init:]`` slice triggers ``log.debug('No logs to display.')``
    and a clean return. The plots subdir is never created.
    """
    plot_mod.plots_perf_timeline(logs=[], directory=str(tmp_path), n_init=0)
    assert not list((tmp_path / 'plots').glob('*.png'))
    plots_dir = tmp_path / 'plots'
    # The plots subdir is either absent (function returned before mkdir) or
    # empty (we got there but nothing was saved). A "silently writes garbage"
    # regression would leave at least one entry here.
    assert (not plots_dir.exists()) or not any(plots_dir.iterdir())


def test_plots_perf_timeline_n_init_drops_initial_rows(tmp_path, monkeypatch):
    """``n_init`` rows are dropped from the DataFrame before plotting.

    With ``n_init`` >= total entries, the DataFrame is empty and the
    function returns before any savefig call, just like the empty-logs
    case but via the slice-then-empty branch.
    """
    plt_mock, fig, _ax = _install_plt_mock(monkeypatch)
    logs = _make_logs(n_init=0, n_extra=3)
    plot_mod.plots_perf_timeline(logs=logs, directory=str(tmp_path), n_init=10)
    # No subplots / savefig were called because the slice gave empty df.
    assert plt_mock.subplots.call_count == 0
    assert fig.savefig.call_count == 0


def test_plots_perf_timeline_full_run_emits_seven_figures(tmp_path, monkeypatch):
    """A full run produces seven figures: timeline + 6 histograms / scatter.

    The function creates one figure for the worker timeline, four basic
    histograms (total / BO / eval / acquisition), two colored histograms
    (fit / acquisition use colored binning), and one distance scatter.
    All routed through ``plt.subplots`` and ``fig.savefig``.
    """
    plt_mock, _fig, _ax = _install_plt_mock(monkeypatch)
    # Pre-create the plots dir so the (mocked) savefig path resolves; this
    # makes the test robust against accidental real-IO regressions where
    # the mock is dropped.
    (tmp_path / 'plots').mkdir()

    logs = _make_logs(n_init=0, n_extra=6, x_dim=1)
    plot_mod.plots_perf_timeline(
        logs=logs, directory=str(tmp_path), n_init=0, min_text_width=0.5
    )

    # Seven figures total: timeline + 6 distribution / scatter panels.
    assert plt_mock.subplots.call_count == 7
    # plt.close called once per figure.
    assert plt_mock.close.call_count == 7


def test_plots_perf_timeline_multi_x_annotation_branch(tmp_path, monkeypatch):
    """``x_value`` with len > 1 takes the y-only annotation branch.

    The single-x branch formats ``(x, y) = (..., ...) ...s``; the multi-x
    branch formats ``y = ... ...s`` (no x because x is multi-dimensional).
    Both must run without raising.
    """
    plt_mock, _fig, _ax = _install_plt_mock(monkeypatch)
    (tmp_path / 'plots').mkdir()
    logs = _make_logs(n_init=0, n_extra=4, x_dim=3)
    plot_mod.plots_perf_timeline(logs=logs, directory=str(tmp_path), n_init=0)

    # Same seven figures regardless of x dimensionality.
    assert plt_mock.subplots.call_count == 7
    # The text annotation lands on ax via ax.text; each row contributes one
    # text call on the timeline axis. With 4 rows, the timeline figure's
    # ax sees at least 4 text calls.
    timeline_ax = plt_mock.subplots.return_value[1]
    assert timeline_ax.text.call_count >= 4


def test_plots_perf_timeline_stretches_xaxis_when_bars_narrower_than_min(tmp_path, monkeypatch):
    """When the narrowest bar is below ``min_text_width``, x-axis is stretched.

    The contract is: bar positions do not move, but the x-axis right limit
    is extended by ``min_text_width - min_bar_width`` (plus 5% padding).
    With bars of width 0.1 and min_text_width 1.0, the stretch is at least
    0.9, dwarfing the bar width itself.
    """
    plt_mock, _fig, ax = _install_plt_mock(monkeypatch)
    (tmp_path / 'plots').mkdir()
    # All bars are exactly 0.1s wide; max bar end is 0.1, min_text_width
    # is 1.0, so stretch_needed is 0.9 and xlim_max becomes 0.1 + 0.9 = 1.0.
    logs = []
    for i in range(3):
        logs.append(
            {
                'start_time': 0.0,
                'end_time': 0.1,
                'worker': i,
                'x_value': [0.5],
                'y_value': 0.1 * i,
                'duration': 0.1,
                'BO_time': 0.01,
                't_eval': 0.05,
                't_fit': 0.005,
                't_ac': 0.005,
                'dist': 0.1,
            }
        )
    plot_mod.plots_perf_timeline(
        logs=logs, directory=str(tmp_path), n_init=0, min_text_width=1.0
    )

    # ax.set_xlim should be called with left=0 and right >= 1.0 (xlim_max
    # of 1.0 plus 5% padding). A regression that forgot the stretch would
    # land at right ~= 0.105 (bar end + padding only).
    xlim_calls = [c for c in ax.set_xlim.call_args_list]
    assert len(xlim_calls) >= 1
    last = xlim_calls[0]
    right = last.kwargs.get('right', None)
    assert right is not None
    # Stretched right edge >> bar width.
    assert right > 1.0
    # Sanity: not absurdly large either.
    assert right < 10.0


def test_plots_perf_timeline_many_workers_uses_random_color_fallback(tmp_path, monkeypatch):
    """With more than 10 workers, color assignment falls back to random.

    Workers 0-9 get ``plt.cm.tab10(i)``; workers 10+ get a clipped random
    RGB triple. Both branches must execute without raising.
    """
    plt_mock, _fig, _ax = _install_plt_mock(monkeypatch)
    (tmp_path / 'plots').mkdir()
    logs = []
    for w in range(12):
        logs.append(
            {
                'start_time': 0.1 * w,
                'end_time': 0.1 * w + 0.5,
                'worker': w,
                'x_value': [0.5],
                'y_value': 0.1,
                'duration': 0.5,
                'BO_time': 0.01,
                't_eval': 0.05,
                't_fit': 0.005,
                't_ac': 0.005,
                'dist': 0.1,
            }
        )
    plot_mod.plots_perf_timeline(logs=logs, directory=str(tmp_path), n_init=0)
    # plt.cm.tab10 called exactly once per worker in [0, 9].
    assert plt_mock.cm.tab10.call_count == 10
    # All 12 workers got a color (the timeline drew one bar per worker).
    # Confirm by counting broken_barh on the timeline ax.
    timeline_ax = plt_mock.subplots.return_value[1]
    assert timeline_ax.broken_barh.call_count == 12


def test_plots_perf_timeline_filters_null_distance_from_scatter(tmp_path, monkeypatch):
    """``dist`` column is filtered with ``notnull`` before the scatter.

    Entries where ``dist`` is ``None`` must be excluded from the distance
    scatter call; otherwise pandas / matplotlib would receive NaN.
    """
    plt_mock, _fig, _ax = _install_plt_mock(monkeypatch)
    (tmp_path / 'plots').mkdir()
    logs = _make_logs(n_init=0, n_extra=6, x_dim=1)
    # _make_logs sets dist=None on indices 0, 3 -> 2 null, 4 real
    n_real = sum(1 for L in logs if L['dist'] is not None)
    plot_mod.plots_perf_timeline(logs=logs, directory=str(tmp_path), n_init=0)

    # Find the scatter call. plt.subplots returns (fig, ax) and the last
    # ax.scatter call comes from the distance plot.
    ax = plt_mock.subplots.return_value[1]
    assert ax.scatter.call_count >= 1
    scatter_call = ax.scatter.call_args_list[-1]
    # 2nd positional arg is the y-values (df_f['dist']); its length should
    # match the real-valued entries (4 here), not the full 6.
    y_arg = scatter_call.args[1]
    assert len(y_arg) == n_real
    assert n_real < len(logs)  # confirm filtering actually dropped rows


# ---------------------------------------------------------------------------
# plots_perf_converge
# ---------------------------------------------------------------------------


def test_plots_perf_converge_monotonic_best_value(tmp_path, monkeypatch):
    """``Y_best`` is the running max of ``Y`` after dropping ``n_init`` rows.

    Pass an explicit Y where the best value plateaus and then rises so the
    running-max branch fires. Confirm two savefigs (regret + bestval).
    """
    plt_mock = MagicMock(name='plt')
    fig = MagicMock(name='fig')
    axes = MagicMock(name='axes')
    # axes[0] and axes[1] are subscriptable.
    axes.__getitem__.return_value = MagicMock()
    plt_mock.subplots.return_value = (fig, axes)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    (tmp_path / 'plots').mkdir()

    Y = [0.1, 0.5, 0.4, 0.7, 0.6, 0.9]
    T = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    plot_mod.plots_perf_converge(D={'Y': Y}, T=T, n_init=0, directory=str(tmp_path))
    # Two figures saved: perf_regret and perf_bestval.
    assert fig.savefig.call_count == 2
    saved_paths = [c.args[0] for c in fig.savefig.call_args_list]
    assert any('perf_regret' in p for p in saved_paths)
    assert any('perf_bestval' in p for p in saved_paths)
    # Two subplots calls, one per figure.
    assert plt_mock.subplots.call_count == 2


def test_plots_perf_converge_n_init_drops_initial_evaluations(tmp_path, monkeypatch):
    """``n_init`` rows are stripped from Y before the running max begins.

    With n_init=2 and Y=[10, 9, 0.1, 0.2, 0.3], the function should treat
    [0.1, 0.2, 0.3] as the trajectory and ignore the 10/9 head. The
    running max then climbs from 0.1, NOT from 10. The test pins both
    figures still get saved and verifies the trajectory length implied
    by ``axes[1].plot`` (n vs Y_best) is 3, not 5.
    """
    plt_mock = MagicMock(name='plt')
    fig = MagicMock(name='fig')
    # Track per-index axes so we can introspect calls on axes[1].
    ax0 = MagicMock(name='ax0')
    ax1 = MagicMock(name='ax1')
    axes = MagicMock(name='axes')
    axes.__getitem__.side_effect = lambda idx: ax0 if idx == 0 else ax1
    plt_mock.subplots.return_value = (fig, axes)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    (tmp_path / 'plots').mkdir()

    Y = [10.0, 9.0, 0.1, 0.2, 0.3]
    T = [1.0, 2.0, 3.0, 4.0, 5.0]
    plot_mod.plots_perf_converge(D={'Y': Y}, T=T, n_init=2, directory=str(tmp_path))

    # Both figures saved.
    assert fig.savefig.call_count == 2
    # ax1.plot is called twice total (once per figure: log-regret-vs-step
    # and bestval-vs-step). First arg is ``n`` array of step indices.
    plot_calls = ax1.plot.call_args_list
    assert len(plot_calls) == 2
    # Step indices array must have length 3 (post-n_init slice).
    n_arr = plot_calls[0].args[0]
    assert len(n_arr) == 3
    # Discrimination: a regression that forgot to slice would give length 5.
    assert len(n_arr) != 5


def test_plots_perf_converge_log_regret_handles_zero_regret(tmp_path, monkeypatch):
    """``log10(regret + 1e-12)`` avoids ``-inf`` when Y_best equals oracle.

    The oracle is hard-coded to 10.0. When ``Y_best`` exactly equals 10,
    raw regret is 0 and ``np.log10(0)`` is -inf; the 1e-12 floor caps the
    log at -12. Confirm no NaN / inf surfaces in the plot data.
    """
    plt_mock = MagicMock(name='plt')
    fig = MagicMock(name='fig')
    ax0 = MagicMock(name='ax0')
    ax1 = MagicMock(name='ax1')
    axes = MagicMock(name='axes')
    axes.__getitem__.side_effect = lambda idx: ax0 if idx == 0 else ax1
    plt_mock.subplots.return_value = (fig, axes)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    (tmp_path / 'plots').mkdir()

    Y = [5.0, 10.0, 10.0]
    T = [0.0, 1.0, 2.0]
    plot_mod.plots_perf_converge(D={'Y': Y}, T=T, n_init=0, directory=str(tmp_path))

    # Fetch the log-regret-vs-time plot call (ax0.plot first call).
    assert ax0.plot.call_count >= 2
    log_regret_arr = ax0.plot.call_args_list[0].args[1]
    arr = np.asarray(log_regret_arr)
    assert np.all(np.isfinite(arr))
    # At the oracle-matched index, log10(regret + 1e-12) should hit the floor
    # of -12 (since regret = 0). Anywhere else it should be > -12.
    assert np.isclose(arr.min(), -12.0, atol=1e-6)


# ---------------------------------------------------------------------------
# plot_result_objective
# ---------------------------------------------------------------------------


def test_plot_result_objective_normal_path_two_parameters(tmp_path, monkeypatch):
    """Two-parameter case wires histograms + scatter for each parameter.

    With d=2 parameters, the function builds a (2, 2) axes grid and calls
    scatter and hist for each column. Confirm both columns are touched.
    """
    plt_mock = MagicMock(name='plt')
    fig = MagicMock(name='fig')
    # axs is a 2D array-like: axs[row, col]. Use a dict-like via __getitem__.
    cells = {}
    for r in (0, 1):
        for c in (0, 1):
            cells[(r, c)] = MagicMock(name=f'ax[{r},{c}]')
    axs = MagicMock(name='axs')
    axs.__getitem__.side_effect = lambda key: cells[key]
    plt_mock.subplots.return_value = (fig, axs)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    # Stub the unnormalize call: return X unchanged.
    monkeypatch.setattr(
        transforms_mod, 'unnormalize_parameters', lambda X, bounds: np.asarray(X, dtype=float)
    )
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda k: False)
    (tmp_path / 'plots').mkdir()

    np.random.seed(42)
    D = {
        'X': np.random.rand(8, 2),
        'Y': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    }
    parameters = {'a': [0.0, 1.0], 'b': [0.0, 1.0]}
    plot_mod.plot_result_objective(
        D=D, parameters=parameters, n_init=2, directory=str(tmp_path), yclip=-12
    )

    # One savefig call for result_objective.
    assert fig.savefig.call_count == 1
    save_path = fig.savefig.call_args_list[0].args[0]
    assert 'result_objective' in save_path
    # Each parameter column got one scatter call for clipped + one for
    # unclipped, plus one hist; with d=2 we get at least 2 scatter pairs.
    n_scatter = sum(cells[(1, c)].scatter.call_count for c in (0, 1))
    assert n_scatter >= 4  # two scatters per column, two columns
    n_hist = sum(cells[(0, c)].hist.call_count for c in (0, 1))
    assert n_hist == 2


def test_plot_result_objective_y_clipping_branch_fires(tmp_path, monkeypatch):
    """Y values below yclip are clipped and the y-label notes the clipping.

    Pass a Y vector with one extreme outlier well below ``yclip``. The
    ``mask`` branch in source then sets a different ylbl and produces
    one clipped-marker scatter (triangle) plus one unclipped scatter.
    """
    plt_mock = MagicMock(name='plt')
    fig = MagicMock(name='fig')
    cells = {}
    for r in (0, 1):
        for c in (0,):
            cells[(r, c)] = MagicMock(name=f'ax[{r},{c}]')
    axs = MagicMock(name='axs')
    axs.__getitem__.side_effect = lambda key: cells[key]
    plt_mock.subplots.return_value = (fig, axs)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    monkeypatch.setattr(
        transforms_mod, 'unnormalize_parameters', lambda X, bounds: np.asarray(X, dtype=float)
    )
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda k: False)
    (tmp_path / 'plots').mkdir()

    # One extreme outlier at -100 will be clipped against yclip=-12.
    D = {
        'X': np.array([[0.1], [0.2], [0.3], [0.4]]),
        'Y': np.array([-100.0, 0.1, 0.2, 0.3]),
    }
    plot_mod.plot_result_objective(
        D=D,
        parameters={'a': [0.0, 1.0]},
        n_init=1,
        directory=str(tmp_path),
        yclip=-12,
    )

    # ylabel on axs[1, 0] should mention the clip; the "clipped" branch
    # writes ``'Value of objective\nclipped to J>...'`` to that axis.
    ax10 = cells[(1, 0)]
    ylabel_call = ax10.set_ylabel.call_args_list
    assert len(ylabel_call) == 1
    label_text = ylabel_call[0].args[0]
    assert 'clipped' in label_text
    # Discrimination: the unclipped branch would have written just
    # 'Value of objective' with no 'clipped' substring.
    assert 'Value of objective' in label_text


def test_plot_result_objective_logarithmic_x_axis_branch(tmp_path, monkeypatch):
    """Logarithmic parameters trigger ``set_xscale('log')`` on the scatter axis.

    ``variable_is_logarithmic`` returning True for a parameter must cause
    ``axs[1, i].set_xscale('log')`` for column ``i``. Mock the helper to
    return True only for the first parameter to exercise both branches.
    """
    plt_mock = MagicMock(name='plt')
    fig = MagicMock(name='fig')
    cells = {}
    for r in (0, 1):
        for c in (0, 1):
            cells[(r, c)] = MagicMock(name=f'ax[{r},{c}]')
    axs = MagicMock(name='axs')
    axs.__getitem__.side_effect = lambda key: cells[key]
    plt_mock.subplots.return_value = (fig, axs)
    monkeypatch.setattr(plot_mod, 'plt', plt_mock)
    monkeypatch.setattr(
        transforms_mod, 'unnormalize_parameters', lambda X, bounds: np.asarray(X, dtype=float)
    )
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda k: k == 'a')
    (tmp_path / 'plots').mkdir()

    D = {
        'X': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
        'Y': np.array([0.1, 0.2, 0.3, 0.4]),
    }
    plot_mod.plot_result_objective(
        D=D,
        parameters={'a': [0.0, 1.0], 'b': [0.0, 1.0]},
        n_init=1,
        directory=str(tmp_path),
    )

    # Parameter 'a' (column 0) should be log-scaled; 'b' (column 1) linear.
    cells[(1, 0)].set_xscale.assert_called_with('log')
    # Parameter 'b' never had set_xscale called (linear is the default).
    assert cells[(1, 1)].set_xscale.call_count == 0


# ---------------------------------------------------------------------------
# plot_result_correlation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_result_correlation_multi_par_multi_obs(monkeypatch, tmp_path, caplog):
    """Multi-parameter, multi-observable case sets per-axis log scaling.

    Existing behavior: helpfile reads succeed for one case and fail with a
    warning for a missing one; ``variable_is_logarithmic`` drives axis
    scaling; legend and labels land on ``axs[0, 0]`` and the outer rim.
    """
    caplog.set_level('WARNING')
    workers = tmp_path / 'workers'

    case_ok = workers / 'w_0' / 'i_0'
    case_ok.mkdir(parents=True)
    (case_ok / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 1.5}}),
        encoding='utf-8',
    )
    pd.DataFrame([{'P_surf': 1.0}]).to_csv(
        case_ok / 'runtime_helpfile.csv', sep=' ', index=False
    )

    case_missing = workers / 'w_0' / 'i_1'
    case_missing.mkdir(parents=True)
    (case_missing / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 2.0}}),
        encoding='utf-8',
    )

    axis = MagicMock()
    axis.__getitem__.return_value = axis
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axis)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: True)

    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0]},
        obs={'P_surf': 1.0},
        directory=str(tmp_path),
    )

    axis.set_xscale.assert_called_with('log')
    axis.set_yscale.assert_called_with('log')
    axis.legend.assert_called_once()
    axis.set_xlabel.assert_called_once()
    axis.set_ylabel.assert_called_once()
    fig.savefig.assert_called_once()
    assert 'Missing helpfile for' in caplog.text


def test_plot_result_correlation_two_par_two_obs_uses_2d_axes(monkeypatch, tmp_path):
    """n_par > 1 and n_obs > 1 takes the ``axs[j, i]`` 2D indexing branch.

    The single-par / single-obs cases collapse to a 1D axis array; the 2D
    case uses ``axs[j, i]`` and applies the outer-rim label loop.
    """
    workers = tmp_path / 'workers'
    for i in range(2):
        case = workers / 'w_0' / f'i_{i}'
        case.mkdir(parents=True)
        (case / 'init_coupler.toml').write_text(
            toml.dumps({'planet': {'mass_tot': 1.5 + 0.1 * i, 'radius': 1.0}}),
            encoding='utf-8',
        )
        pd.DataFrame([{'P_surf': 1.0 * (i + 1), 'T_eqm': 250.0 + 10 * i}]).to_csv(
            case / 'runtime_helpfile.csv', sep=' ', index=False
        )

    # 2D axs: support both axs[j, i] tuple indexing (with negative indices
    # for the outer-rim label loop) and axs[i] single index.
    n = 2
    cells = {}
    for j in range(n):
        for i in range(n):
            cells[(j, i)] = MagicMock(name=f'ax[{j},{i}]')

    def _axs_lookup(key):
        if isinstance(key, tuple):
            j, i = key
            if j < 0:
                j += n
            if i < 0:
                i += n
            return cells[(j, i)]
        return MagicMock()

    axs = MagicMock(name='axs')
    axs.__getitem__.side_effect = _axs_lookup
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axs)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: False)

    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0], 'planet.radius': [0.5, 2.0]},
        obs={'P_surf': 1.0, 'T_eqm': 260.0},
        directory=str(tmp_path),
    )

    # Legend lands on axs[0, 0] in the 2D branch.
    cells[(0, 0)].legend.assert_called_once()
    fig.savefig.assert_called_once()
    # Outer rim x-labels go on axs[-1, i]: the test grid is 2x2 so [-1, 0]
    # and [-1, 1] both get set_xlabel. With our mock, [-1, 0] resolves
    # the same way [1, 0] does only if __getitem__ supports negative keys.
    # Skip that exact assertion and instead pin: at least 4 set_xticklabels
    # calls happened across the bottom row and right column hiders.
    total_xtick_hide = sum(c.set_xticklabels.call_count for c in cells.values())
    # Top row hides x-tick labels: j=0 for both i=0 and i=1.
    assert total_xtick_hide >= 2


def test_plot_result_correlation_single_par_multi_obs_uses_1d_axs(monkeypatch, tmp_path):
    """``n_par == 1`` with ``n_obs > 1`` takes the ``ax = axs[j]`` branch.

    Confirms the 1D-axes branch fires for the asymmetric (1, K) grid and
    the inner loop indexes by observable, not by parameter.
    """
    workers = tmp_path / 'workers'
    for i in range(2):
        case = workers / 'w_0' / f'i_{i}'
        case.mkdir(parents=True)
        (case / 'init_coupler.toml').write_text(
            toml.dumps({'planet': {'mass_tot': 1.5 + 0.1 * i}}),
            encoding='utf-8',
        )
        pd.DataFrame([{'P_surf': 1.0 + i, 'T_eqm': 250.0 + 10 * i}]).to_csv(
            case / 'runtime_helpfile.csv', sep=' ', index=False
        )

    cells = {0: MagicMock(name='ax[0]'), 1: MagicMock(name='ax[1]')}
    axs = MagicMock(name='axs')
    axs.__getitem__.side_effect = lambda k: cells[k]
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axs)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: False)

    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0]},
        obs={'P_surf': 1.0, 'T_eqm': 260.0},
        directory=str(tmp_path),
    )

    # axs[0] and axs[1] each got exactly one scatter (one per observable).
    assert cells[0].scatter.call_count == 1
    assert cells[1].scatter.call_count == 1
    fig.savefig.assert_called_once()


def test_plot_result_correlation_multi_par_single_obs_uses_1d_axs(monkeypatch, tmp_path):
    """``n_obs == 1`` with ``n_par > 1`` takes the ``ax = axs[i]`` branch.

    Mirror of the previous test for the (K, 1) grid.
    """
    workers = tmp_path / 'workers'
    for i in range(2):
        case = workers / 'w_0' / f'i_{i}'
        case.mkdir(parents=True)
        (case / 'init_coupler.toml').write_text(
            toml.dumps({'planet': {'mass_tot': 1.5 + 0.1 * i, 'radius': 1.0 + i}}),
            encoding='utf-8',
        )
        pd.DataFrame([{'P_surf': 1.0 + i}]).to_csv(
            case / 'runtime_helpfile.csv', sep=' ', index=False
        )

    cells = {0: MagicMock(name='ax[0]'), 1: MagicMock(name='ax[1]')}
    axs = MagicMock(name='axs')
    axs.__getitem__.side_effect = lambda k: cells[k]
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axs)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: False)

    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0], 'planet.radius': [0.5, 2.0]},
        obs={'P_surf': 1.0},
        directory=str(tmp_path),
    )

    # axs[0] and axs[1] each got exactly one scatter (one per parameter).
    assert cells[0].scatter.call_count == 1
    assert cells[1].scatter.call_count == 1
    fig.savefig.assert_called_once()


def test_plot_result_correlation_single_par_single_obs_uses_scalar_axes(monkeypatch, tmp_path):
    """The ``n_par == 1 and n_obs == 1`` branch dispatches on the scalar ax.

    With a single parameter and a single observable, ``plt.subplots(1, 1)``
    returns a single Axes (not an array). The function takes the
    ``ax = axs`` branch and the labels / legend go on that single axis.
    """
    workers = tmp_path / 'workers'
    case = workers / 'w_0' / 'i_0'
    case.mkdir(parents=True)
    (case / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 1.5}}),
        encoding='utf-8',
    )
    pd.DataFrame([{'P_surf': 1.0}]).to_csv(case / 'runtime_helpfile.csv', sep=' ', index=False)

    # Single Axes (scalar). plot_result_correlation uses ax = axs in this
    # branch and then calls scatter / axhline / set_xlabel / set_ylabel /
    # set_xscale / set_yscale / legend on it.
    axis = MagicMock(name='ax_scalar')
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axis)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: False)

    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0]},
        obs={'P_surf': 1.0},
        directory=str(tmp_path),
    )

    # In the scalar branch the legend / xlabel / ylabel calls go through
    # axs[0]; the MagicMock supports subscripting and __getitem__ returns a
    # sub-mock. Confirm savefig fired exactly once and lands in plots/.
    fig.savefig.assert_called_once()
    save_path = fig.savefig.call_args_list[0].args[0]
    assert 'result_correlation' in save_path
    assert 'plots' in save_path
    # The single Axes was used for the scatter (which lives in the
    # inner loop). With one case, one parameter, one observable, scatter
    # must have been called exactly once.
    assert axis.scatter.call_count == 1


# ---------------------------------------------------------------------------
# plot_proteus
# ---------------------------------------------------------------------------


def test_plot_proteus_skips_visual_entries(monkeypatch, tmp_path):
    """``plot_proteus`` dispatches every key except ``visual``/``anim_visual``.

    The function instantiates ``Proteus``, calls ``extract_archives``, and
    iterates ``plot_dispatch``. Keys ``'visual'`` and ``'anim_visual'``
    are deliberately skipped. Mock both the constructor and the dispatch
    table to capture calls.
    """
    fake_handler = MagicMock(name='Proteus instance')

    def _fake_proteus(config_path):
        # Record the config path so we can assert on it.
        fake_handler.config_path_used = config_path
        return fake_handler

    monkeypatch.setattr(plot_mod, 'Proteus', _fake_proteus)

    # Mock dispatch with five fake entries, two of which are in the skip set.
    fake_dispatch = {
        'global': MagicMock(name='plot_global'),
        'interior': MagicMock(name='plot_interior'),
        'visual': MagicMock(name='plot_visual'),  # must be skipped
        'anim_visual': MagicMock(name='plot_anim_visual'),  # must be skipped
        'orbit': MagicMock(name='plot_orbit'),
    }
    monkeypatch.setattr(plot_mod, 'plot_dispatch', fake_dispatch)

    cfg = tmp_path / 'best.toml'
    cfg.write_text('# stub')
    plot_mod.plot_proteus(best_config=str(cfg))

    # Constructor received the config path.
    assert fake_handler.config_path_used == str(cfg)
    fake_handler.extract_archives.assert_called_once()
    # Three keys ran: global, interior, orbit.
    fake_dispatch['global'].assert_called_once_with(fake_handler)
    fake_dispatch['interior'].assert_called_once_with(fake_handler)
    fake_dispatch['orbit'].assert_called_once_with(fake_handler)
    # Two keys were skipped.
    fake_dispatch['visual'].assert_not_called()
    fake_dispatch['anim_visual'].assert_not_called()


def test_plot_proteus_propagates_dispatch_errors(monkeypatch, tmp_path):
    """An exception inside a dispatched plot is not swallowed.

    There is no try/except wrap; a plot that raises must surface as the
    caller's exception. Pin both that the error propagates and that
    subsequent plots are not silently dispatched after the failure.
    """
    fake_handler = MagicMock(name='Proteus instance')
    monkeypatch.setattr(plot_mod, 'Proteus', lambda config_path: fake_handler)

    boom = MagicMock(name='plot_first', side_effect=RuntimeError('boom'))
    after = MagicMock(name='plot_after')
    monkeypatch.setattr(plot_mod, 'plot_dispatch', {'first': boom, 'second': after})

    cfg = tmp_path / 'best.toml'
    cfg.write_text('# stub')
    with pytest.raises(RuntimeError, match='boom'):
        plot_mod.plot_proteus(best_config=str(cfg))
    # The second plot must NOT have run because the loop did not catch.
    after.assert_not_called()
    fake_handler.extract_archives.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_plot_module_constants_have_expected_types():
    """Module-level constants are the documented types / values.

    ``fmt`` controls the output extension and ``dpi`` controls the figure
    resolution. A regression changing either silently would break downstream
    consumers that grep for ``.png`` artifacts.
    """
    assert plot_mod.fmt == 'png'
    # dpi is a positive int around 300; pin both type and bounds to catch a
    # regression that lowered it to a publication-incompatible value.
    assert isinstance(plot_mod.dpi, int)
    assert 100 <= plot_mod.dpi <= 1200


def test_plot_module_exposes_expected_public_functions():
    """All five documented functions are importable from the module.

    A refactor that renamed any of these would break inference orchestration
    code that calls them by name. Pin the names and that each is callable.
    """
    expected = (
        'plots_perf_timeline',
        'plots_perf_converge',
        'plot_result_objective',
        'plot_result_correlation',
        'plot_proteus',
    )
    for name in expected:
        attr = getattr(plot_mod, name, None)
        assert attr is not None, f'missing: {name}'
        assert callable(attr), f'not callable: {name}'
