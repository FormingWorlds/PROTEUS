"""
Unit tests for inference plotting utilities.

The module under test (``src/proteus/inference/plot.py``) is a utility module
(see ``tests/AGENTS.md``, "Physics modules"): it is exempt from
the physics-invariant requirement but every test must still exercise an edge
or error path and pin assertions that discriminate against a plausible
regression. Tests here cover the six public functions:

  - ``plots_perf_timeline``: timeline + six histograms + distance scatter.
  - ``plots_perf_converge``: log-regret and best-value vs time / iteration.
  - ``plot_result_objective``: per-parameter scatter + histogram grid.
  - ``plot_result_correlation``: observable-vs-parameter grid with missing
    helpfile handling.
  - ``plot_result_observables``: best-fit vs target comparison, including the
    failure-table exclusion, the named-best-case override and its fallback,
    and the two panel builders.
  - ``plot_proteus``: dispatch over ``plot_dispatch`` with skip set.

Matplotlib is mocked at the module-attribute level (``plot_mod.plt``) so no
figure is rendered and ``savefig`` is captured as a call assertion.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import toml

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.plot as plot_mod  # noqa: E402
import proteus.inference.transforms as transforms_mod  # noqa: E402

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
    case_empty = workers / 'w_0' / 'i_2'
    case_empty.mkdir(parents=True)
    (case_empty / 'init_coupler.toml').write_text('[planet]\nmass_tot = 2.5\n')
    (case_empty / 'runtime_helpfile.csv').write_text('P_surf\n', encoding='utf-8')
    # The header-only case pins the min_rows=1 skip; this one tests the OSError half.
    case_locked = workers / 'w_0' / 'i_3'
    case_locked.mkdir(parents=True)
    (case_locked / 'init_coupler.toml').write_text('[planet]\nmass_tot = 2.8\n')
    (case_locked / 'runtime_helpfile.csv').write_text('P_surf\n3.0\n', encoding='utf-8')
    real_read = plot_mod.read_helpfile_table

    def read_or_deny(path, **kwargs):
        if 'i_3' in str(path):
            raise PermissionError(f'Permission denied: {path}')
        return real_read(path, **kwargs)

    monkeypatch.setattr(plot_mod, 'read_helpfile_table', read_or_deny)

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
    assert 'Unreadable helpfile for' in caplog.text
    assert 'Permission denied' in caplog.text
    # Only the readable case is plotted, one point per panel.
    assert {len(c.args[0]) for c in axis.scatter.call_args_list} == {1}


@pytest.mark.unit
def test_plot_result_correlation_ignores_stray_console_log_file(monkeypatch, tmp_path):
    """A worker's console-log capture file must not be treated as a case dir.

    Regression for a crash where a stray file such as ``i_0_console.log``,
    sitting beside the real ``i_0`` case directory in a worker folder, matched
    the ``i_*`` glob used to find cases. ``toml.load`` then received a file
    path, not a directory, and raised ``NotADirectoryError`` when the code
    appended ``init_coupler.toml`` to it.
    """
    workers = tmp_path / 'workers'
    case_ok = workers / 'w_-1' / 'i_0'
    case_ok.mkdir(parents=True)
    (case_ok / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 1.5}}),
        encoding='utf-8',
    )
    pd.DataFrame([{'P_surf': 1.0}]).to_csv(
        case_ok / 'runtime_helpfile.csv', sep=' ', index=False
    )

    # Sibling capture file that matches the `i_*` glob but is not a case dir.
    (workers / 'w_-1' / 'i_0_console.log').write_text('log output\n', encoding='utf-8')

    axis = MagicMock()
    axis.__getitem__.return_value = axis
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axis)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: False)

    # Must not raise NotADirectoryError from treating the log file as a case.
    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0]},
        obs={'P_surf': 1.0},
        directory=str(tmp_path),
    )

    fig.savefig.assert_called_once()


def test_plot_result_correlation_skips_a_case_that_died_during_start_up(
    monkeypatch, tmp_path, caplog
):
    """A case that died before writing its resolved config has a folder but no
    ``init_coupler.toml``. It is skipped with a warning naming it, and the
    finished cases are still plotted. A study in which every case died still
    produces a figure, with no points on it, rather than failing on an empty
    array.
    """
    workers = tmp_path / 'workers'
    case_ok = workers / 'w_0' / 'i_0'
    case_ok.mkdir(parents=True)
    (case_ok / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 1.5}}),
        encoding='utf-8',
    )
    pd.DataFrame([{'P_surf': 2.0}]).to_csv(
        case_ok / 'runtime_helpfile.csv', sep=' ', index=False
    )
    # What `run_proteus` leaves for a child that exits before PROTEUS writes
    # anything of its own: the folder and the config it was handed.
    case_dead = workers / 'w_1' / 'i_0'
    case_dead.mkdir(parents=True)
    (case_dead / 'input.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 2.5}}), encoding='utf-8'
    )

    axis = MagicMock()
    axis.__getitem__.return_value = axis
    fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axis)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: False)

    def _plot():
        plot_mod.plot_result_correlation(
            pars={'planet.mass_tot': [0.7, 3.0]},
            obs={'P_surf': 1.0},
            directory=str(tmp_path),
        )

    with caplog.at_level('WARNING'):
        _plot()

    fig.savefig.assert_called_once()
    # Only the finished case is plotted: its parameter against its observable.
    xx, yy = axis.scatter.call_args.args[:2]
    assert list(xx) == pytest.approx([1.5])
    assert list(yy) == pytest.approx([2.0])
    warnings = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
    assert warnings == [f'Missing init_coupler.toml for {case_dead}']

    # Edge case: no case produced output. The figure is still written, with
    # empty data of the right width rather than an IndexError on X[:, 0].
    (case_ok / 'init_coupler.toml').unlink()
    axis.reset_mock()
    fig.reset_mock()
    _plot()
    fig.savefig.assert_called_once()
    xx, yy = axis.scatter.call_args.args[:2]
    assert (len(xx), len(yy)) == (0, 0)


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
    # The mock does not resolve negative keys like axs[-1, i], so instead of
    # the outer-rim labels, pin the set_xticklabels calls across the grid.
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
    """All six documented functions are importable from the module.

    A refactor that renamed any of these would break inference orchestration
    code that calls them by name. Pin the names and that each is callable.
    """
    expected = (
        'plots_perf_timeline',
        'plots_perf_converge',
        'plot_result_objective',
        'plot_result_correlation',
        'plot_result_observables',
        'plot_proteus',
    )
    for name in expected:
        attr = getattr(plot_mod, name, None)
        assert attr is not None, f'missing: {name}'
        assert callable(attr), f'not callable: {name}'


# ---------------------------------------------------------------------------
# plot_result_observables
# ---------------------------------------------------------------------------


def _write_case(root, worker, iter, row):
    """Create one case directory holding a single-row helpfile.

    ``row`` maps helpfile column names to their final-timestep value; a
    ``P_surf`` entry is added when absent because ``get_obs`` reads that
    column to decide whether the atmosphere escaped.
    """
    case = root / 'workers' / f'w_{worker}' / f'i_{iter}'
    case.mkdir(parents=True)
    row = dict(row)
    row.setdefault('P_surf', 1.0e5)
    pd.DataFrame([row]).to_csv(case / 'runtime_helpfile.csv', sep=' ', index=False)
    return case


def _two_axes(monkeypatch):
    """Point ``plot_mod.plt`` at a figure with two independent mock axes."""
    ax_ratio, ax_resid = MagicMock(name='ax_ratio'), MagicMock(name='ax_resid')
    fig = MagicMock(name='fig')
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, [ax_ratio, ax_resid])
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    return mock_plt, fig, ax_ratio, ax_resid


def _diamond_call(ax):
    """Return the scatter call that draws the best-fit diamond marker."""
    calls = [c for c in ax.scatter.call_args_list if c.kwargs.get('marker') == 'D']
    assert len(calls) == 1, f'expected one best-fit marker, got {len(calls)}'
    return calls[0]


@pytest.mark.unit
def test_collect_case_observables_pins_objective_and_skips_missing_helpfile(tmp_path, caplog):
    """The recomputed objective matches its closed form; a case with no
    helpfile is warned about and dropped rather than crashing the figure.

    Targets 2.0 and 5.0 against simulated 2.4 and 5.5 give normalised
    residuals of -0.2 and -0.1, so the squared distance is 0.05 and the
    objective is -log10(0.05 + 1e-10). The targets are deliberately not 1.0
    so that a regression to an un-normalised residual is distinguishable.
    """
    caplog.set_level('WARNING')
    obs = {'R_obs': 2.0, 'T_obs': 5.0}
    _write_case(tmp_path, 0, 0, {'R_obs': 2.4, 'T_obs': 5.5})
    (tmp_path / 'workers' / 'w_0' / 'i_1').mkdir(parents=True)

    df = plot_mod._collect_case_observables(tmp_path, obs)

    assert list(df['case']) == ['w0_i0']
    assert not bool(df.loc[0, 'excluded'])
    expected = -np.log10(0.05 + 1e-10)
    assert df.loc[0, 'J'] == pytest.approx(expected, rel=1e-9)
    # Discrimination guard: an un-normalised squared residual would give
    # -log10(0.4**2 + 0.5**2), which is 0.6 away, far beyond the tolerance.
    wrong = -np.log10(0.4**2 + 0.5**2 + 1e-10)
    assert abs(expected - wrong) > 1e-3
    assert 'Missing helpfile for' in caplog.text


@pytest.mark.unit
def test_collect_case_observables_marks_recorded_failures_as_excluded(tmp_path):
    """Cases listed in the failure table are flagged, scored ones are not.

    A run that failed or completed on an excluded status carries the failure
    score rather than a fit quality, so it must not be read as a candidate
    best fit even though its helpfile parses.
    """
    obs = {'R_obs': 1.0}
    _write_case(tmp_path, 0, 0, {'R_obs': 1.02})
    _write_case(tmp_path, 0, 1, {'R_obs': 1.00})
    pd.DataFrame([{'worker': 0, 'iter': 1, 'category': 'excluded'}]).to_csv(
        tmp_path / 'failures.csv', index=False
    )

    df = plot_mod._collect_case_observables(tmp_path, obs).set_index('case')

    assert not bool(df.loc['w0_i0', 'excluded'])
    assert bool(df.loc['w0_i1', 'excluded'])
    # The excluded case is the closer one, so a regression that ignored the
    # failure table would change which row wins on J.
    assert df.loc['w0_i1', 'J'] > df.loc['w0_i0', 'J']


@pytest.mark.unit
def test_plot_result_observables_highlights_the_reported_best_case(monkeypatch, tmp_path):
    """The named case is highlighted even when another scores better.

    The figure and the results summary must name the same run, so a supplied
    ``best_config`` overrides the highest recomputed objective.
    """
    (tmp_path / 'plots').mkdir()
    named = _write_case(tmp_path, 0, 0, {'R_obs': 1.20})
    _write_case(tmp_path, 0, 1, {'R_obs': 1.01})
    _mock_plt, fig, ax_ratio, _ax_resid = _two_axes(monkeypatch)

    plot_mod.plot_result_observables(
        obs={'R_obs': 1.0},
        directory=str(tmp_path),
        best_config=str(named / 'init_coupler.toml'),
    )

    call = _diamond_call(ax_ratio)
    assert 'w0_i0' in call.kwargs['label']
    # The diamond sits at the named case's ratio, not the better-scoring one.
    assert float(np.asarray(call.args[0])[0]) == pytest.approx(1.20, rel=1e-9)
    assert abs(1.20 - 1.01) > 1e-6
    saved = fig.savefig.call_args.args[0]
    assert str(saved).endswith('result_observables.png')


@pytest.mark.unit
def test_plot_result_observables_falls_back_to_best_j_for_an_unknown_case(
    monkeypatch, tmp_path, caplog
):
    """A ``best_config`` that names no scored case warns and falls back.

    Error contract: an unparsable or unmatched path must not abort the figure,
    because the comparison is still informative without the named highlight.
    """
    caplog.set_level('WARNING')
    (tmp_path / 'plots').mkdir()
    _write_case(tmp_path, 0, 0, {'R_obs': 1.20})
    _write_case(tmp_path, 0, 1, {'R_obs': 1.01})
    _mock_plt, fig, ax_ratio, _ax_resid = _two_axes(monkeypatch)

    plot_mod.plot_result_observables(
        obs={'R_obs': 1.0},
        directory=str(tmp_path),
        best_config='/does/not/exist/init_coupler.toml',
    )

    assert 'using the best J' in caplog.text
    call = _diamond_call(ax_ratio)
    assert 'w0_i1' in call.kwargs['label']
    assert float(np.asarray(call.args[0])[0]) == pytest.approx(1.01, rel=1e-9)
    fig.savefig.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    'setup, expected',
    [
        ('none', 'No case produced observables'),
        ('all_excluded', 'Every case carries the failure score'),
    ],
    ids=['no_cases', 'all_cases_excluded'],
)
def test_plot_result_observables_skips_when_nothing_is_scored(
    monkeypatch, tmp_path, caplog, setup, expected
):
    """With no scored case the function warns and draws nothing.

    Edge case: a study whose every evaluation failed has no fit to show, and
    picking a best row from an empty frame would raise instead.
    """
    caplog.set_level('WARNING')
    (tmp_path / 'plots').mkdir()
    (tmp_path / 'workers').mkdir()
    if setup == 'all_excluded':
        _write_case(tmp_path, 0, 0, {'R_obs': 1.0})
        pd.DataFrame([{'worker': 0, 'iter': 0, 'category': 'failure'}]).to_csv(
            tmp_path / 'failures.csv', index=False
        )
    mock_plt, _fig, _ax_ratio, _ax_resid = _two_axes(monkeypatch)

    plot_mod.plot_result_observables(obs={'R_obs': 1.0}, directory=str(tmp_path))

    assert expected in caplog.text
    assert mock_plt.subplots.call_count == 0
    assert not list((tmp_path / 'plots').glob('*.png'))


@pytest.mark.unit
def test_panel_ratio_parks_out_of_range_cases_on_the_axis_edges():
    """Cases beyond the ratio axis are drawn on its edges, one legend each.

    A row whose cases all collapse to ~1e-22, or run far above target, would
    otherwise be silently empty. Both edge markers point outwards and each
    legend label is emitted only once across the rows.
    """
    lo, hi = plot_mod.RATIO_CLIP
    obs = {'R_obs': 1.0, 'T_obs': 1.0}
    ok = pd.DataFrame(
        {
            'R_obs': [1e-6, 1.0],  # one below the axis, one inside
            'T_obs': [1e6, 1.0],  # one above the axis, one inside
            'case': ['w0_i0', 'w0_i1'],
        }
    )
    ax = MagicMock()

    plot_mod._panel_ratio(ax, ok, obs, np.array([1.0, 1.0]), 'w0_i1')

    edges = {
        c.kwargs['marker']: float(np.asarray(c.args[0])[0])
        for c in ax.scatter.call_args_list
        if c.kwargs.get('marker') in ('<', '>')
    }
    assert set(edges) == {'<', '>'}
    assert edges['<'] == pytest.approx(lo * 1.12, rel=1e-9)
    assert edges['>'] == pytest.approx(hi / 1.12, rel=1e-9)
    # Each marker style contributes exactly one legend entry.
    labels = [c.kwargs.get('label') for c in ax.scatter.call_args_list if c.kwargs.get('label')]
    assert len(labels) == len(set(labels))
    ax.set_xscale.assert_called_once_with('log')
    ax.set_xlim.assert_called_once_with(lo, hi)


@pytest.mark.unit
def test_panel_residual_keeps_limits_ordered_for_a_single_observable():
    """A lone observable still yields a strictly increasing x-range.

    Edge case: the residual spread is zero with one bar, so the span used to
    scale the limits and the label offsets has to fall back to a finite value
    or matplotlib receives left == right.
    """
    ax = MagicMock()

    plot_mod._panel_residual(ax, ['R_obs'], np.array([4.0]), best_J=1.25)

    left, right = ax.set_xlim.call_args.args
    assert right > left
    # The single bar must sit strictly inside the range, not on its edge.
    assert left < 4.0 < right
    assert ax.barh.call_count == 1
    assert 'J = +1.2500' in ax.set_title.call_args.args[0]


# ---------------------------------------------------------------------------
# plot_result_parameters
# ---------------------------------------------------------------------------

# One linear parameter whose range straddles zero and one log-scaled parameter,
# so the two placement rules give clearly different positions.
_PARS = {'outgas.fO2_shift_IW': [-4.0, 4.0], 'planet.elements.H_budget': [1.0e3, 2.0e4]}
_TRUTH = {'outgas.fO2_shift_IW': 0.0, 'planet.elements.H_budget': 5.0e3}


def _write_param_case(root, worker, iter, r_obs, fo2, h_budget):
    """Create a case with a helpfile and the config it ran with."""
    case = _write_case(root, worker, iter, {'R_obs': r_obs})
    (case / 'init_coupler.toml').write_text(
        f'[outgas]\nfO2_shift_IW = {fo2}\n[planet.elements]\nH_budget = {h_budget}\n'
    )
    return case


@pytest.mark.unit
def test_position_in_range_uses_log10_for_log_scaled_parameters():
    """A log-scaled value is placed by its log10 within the range, matching
    how the optimiser normalises it; values beyond the range fall outside
    [0, 1] instead of being clipped.
    """
    # 1e4 in [1e3, 2e4]: log placement is 1 / log10(20) = 0.7686.
    got = float(plot_mod._position_in_range(1.0e4, 'planet.elements.H_budget', [1.0e3, 2.0e4]))
    assert got == pytest.approx(1.0 / np.log10(20.0), rel=1e-12)
    # Discrimination guard: linear placement would give 9e3 / 1.9e4 = 0.4737.
    assert abs(got - 9.0e3 / 1.9e4) > 0.2
    lin = plot_mod._position_in_range([-4.0, 0.0, 6.0], 'outgas.fO2_shift_IW', [-4.0, 4.0])
    np.testing.assert_allclose(lin, [0.0, 0.5, 1.25], rtol=0, atol=1e-12)
    # Edge case: a non-positive value has no log position and comes back non-finite.
    assert not np.isfinite(
        plot_mod._position_in_range(0.0, 'planet.elements.H_budget', [1e3, 2e4])
    )


@pytest.mark.unit
def test_plot_result_parameters_places_truth_and_named_best_fit(monkeypatch, tmp_path):
    """The truth and the named best case are drawn at their range positions,
    and the error bars are best minus truth as a percentage of the range.

    The named case scores worse than the other one, so a regression that
    picked the highest objective instead would move the diamond.
    """
    (tmp_path / 'plots').mkdir()
    named = _write_param_case(tmp_path, 0, 0, 1.20, 2.0, 1.0e4)
    _write_param_case(tmp_path, 0, 1, 1.01, -1.0, 2.0e3)
    _mock_plt, fig, ax_pos, ax_err = _two_axes(monkeypatch)

    plot_mod.plot_result_parameters(
        _PARS, _TRUTH, {'R_obs': 1.0}, str(tmp_path), str(named / 'init_coupler.toml')
    )

    best_x = np.asarray(_diamond_call(ax_pos).args[0], dtype=float)
    np.testing.assert_allclose(best_x, [0.75, 1.0 / np.log10(20.0)], rtol=1e-12)
    truth_calls = [c for c in ax_pos.scatter.call_args_list if c.kwargs.get('marker') == '|']
    assert len(truth_calls) == 1
    truth_x = np.asarray(truth_calls[0].args[0], dtype=float)
    np.testing.assert_allclose(truth_x, [0.5, np.log10(5.0) / np.log10(20.0)], rtol=1e-12)

    pct = np.asarray(ax_err.barh.call_args.args[1], dtype=float)
    expected = 100.0 * (best_x - truth_x)
    np.testing.assert_allclose(pct, expected, rtol=1e-12)
    # Sign guard: the named best fit lies above the truth in both parameters.
    assert (pct > 0).all()
    # Discrimination guard: a linear error for H_budget would be 5e3 / 1.9e4 = 26.3 %,
    # not the 23.1 % of the log placement.
    assert abs(pct[1] - 100.0 * 5.0e3 / 1.9e4) > 1.0
    assert str(fig.savefig.call_args.args[0]).endswith('result_parameters.png')


@pytest.mark.unit
def test_plot_result_parameters_widens_axis_for_truth_outside_range(monkeypatch, tmp_path):
    """A true value beyond the sampled range stays visible: the axis widens
    past [0, 1] just far enough to include it.
    """
    (tmp_path / 'plots').mkdir()
    _write_param_case(tmp_path, 0, 0, 1.01, 0.0, 5.0e3)
    _mock_plt, _fig, ax_pos, _ax_err = _two_axes(monkeypatch)
    # fO2 shift of +6 lies at position 1.25 of the [-4, 4] range.
    truth = {'outgas.fO2_shift_IW': 6.0, 'planet.elements.H_budget': 5.0e3}

    plot_mod.plot_result_parameters(_PARS, truth, {'R_obs': 1.0}, str(tmp_path))

    left, right = ax_pos.set_xlim.call_args.args
    assert right == pytest.approx(1.25 + 0.04, rel=1e-12)
    assert left == pytest.approx(-0.04, rel=1e-12)


@pytest.mark.unit
def test_plot_result_parameters_skips_when_nothing_is_scored(monkeypatch, tmp_path, caplog):
    """Every case excluded leaves nothing to compare, so the function warns
    and draws nothing rather than picking a best row from an empty frame.
    """
    caplog.set_level('WARNING')
    (tmp_path / 'plots').mkdir()
    _write_param_case(tmp_path, 0, 0, 1.0, 0.0, 5.0e3)
    pd.DataFrame([{'worker': 0, 'iter': 0, 'category': 'failure'}]).to_csv(
        tmp_path / 'failures.csv', index=False
    )
    mock_plt, _fig, _ax_pos, _ax_err = _two_axes(monkeypatch)

    plot_mod.plot_result_parameters(_PARS, _TRUTH, {'R_obs': 1.0}, str(tmp_path))

    assert 'No scored case to compare with the true parameters' in caplog.text
    assert mock_plt.subplots.call_count == 0
    assert not list((tmp_path / 'plots').glob('*.png'))
