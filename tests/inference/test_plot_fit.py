"""
Unit tests for the best-fit plots in ``src/proteus/inference/plot_fit.py``.

A utility module (see ``tests/AGENTS.md``, "Physics modules"), so exempt from
the physics-invariant requirement. Covers the case scoring that feeds both
plots, ``plot_result_observables`` with the failure-table exclusion and the
named-best-case override and fallback, its two panel builders, and
``plot_result_parameters``.

Matplotlib is mocked at the module-attribute level (``plot_fit_mod.plt``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.plot_fit as plot_fit_mod  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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
    """Point ``plot_fit_mod.plt`` at a figure with two independent mock axes."""
    ax_ratio, ax_resid = MagicMock(name='ax_ratio'), MagicMock(name='ax_resid')
    fig = MagicMock(name='fig')
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, [ax_ratio, ax_resid])
    monkeypatch.setattr(plot_fit_mod, 'plt', mock_plt)
    return mock_plt, fig, ax_ratio, ax_resid


def _diamond_call(ax):
    """Return the scatter call that draws the best-fit diamond marker."""
    calls = [c for c in ax.scatter.call_args_list if c.kwargs.get('marker') == 'D']
    assert len(calls) == 1, f'expected one best-fit marker, got {len(calls)}'
    return calls[0]


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

    df = plot_fit_mod.collect_case_observables(tmp_path, obs)

    assert list(df['case']) == ['w0_i0']
    assert not bool(df.loc[0, 'excluded'])
    expected = -np.log10(0.05 + 1e-10)
    assert df.loc[0, 'J'] == pytest.approx(expected, rel=1e-9)
    # Discrimination guard: an un-normalised squared residual would give
    # -log10(0.4**2 + 0.5**2), which is 0.6 away, far beyond the tolerance.
    wrong = -np.log10(0.4**2 + 0.5**2 + 1e-10)
    assert abs(expected - wrong) > 1e-3
    assert 'Missing helpfile for' in caplog.text


@pytest.mark.physics_invariant
def test_collect_case_observables_scores_with_observable_correlation(tmp_path):
    """The plotted J uses the optimiser's correlation: u = (2, 2), rho = 0.6 gives chi2 5, not 8."""
    obs = {'R_obs': 6.0e6, 'T_obs': 400.0}
    sigma = {'R_obs': 5.0e5, 'T_obs': 25.0}
    _write_case(tmp_path, 0, 0, {'R_obs': 7.0e6, 'T_obs': 450.0})

    df = plot_fit_mod.collect_case_observables(tmp_path, obs, sigma, {'T_obs': {'R_obs': 0.6}})

    assert df.loc[0, 'J'] == pytest.approx(-np.log10(5.0 + 1e-10), rel=1e-9)
    independent = plot_fit_mod.collect_case_observables(tmp_path, obs, sigma)
    assert independent.loc[0, 'J'] == pytest.approx(-np.log10(8.0 + 1e-10), rel=1e-9)


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

    df = plot_fit_mod.collect_case_observables(tmp_path, obs).set_index('case')

    assert not bool(df.loc['w0_i0', 'excluded'])
    assert bool(df.loc['w0_i1', 'excluded'])
    # The excluded case is the closer one, so a regression that ignored the
    # failure table would change which row wins on J.
    assert df.loc['w0_i1', 'J'] > df.loc['w0_i0', 'J']


def test_plot_result_observables_highlights_the_reported_best_case(monkeypatch, tmp_path):
    """The named case is highlighted even when another scores better.

    The figure and the results summary must name the same run, so a supplied
    ``best_config`` overrides the highest recomputed objective. The plot draws
    from the frame it is given, with the helpfiles already gone.
    """
    (tmp_path / 'plots').mkdir()
    named = _write_case(tmp_path, 0, 0, {'R_obs': 1.20})
    _write_case(tmp_path, 0, 1, {'R_obs': 1.01})
    _mock_plt, fig, ax_ratio, _ax_resid = _two_axes(monkeypatch)
    cases = plot_fit_mod.collect_case_observables(tmp_path, {'R_obs': 1.0})
    # The plot draws from the collected frame, so the helpfiles are not read again.
    for hf in tmp_path.glob('workers/w_*/i_*/runtime_helpfile.csv'):
        hf.unlink()

    plot_fit_mod.plot_result_observables(
        cases, {'R_obs': 1.0}, str(tmp_path), str(named / 'init_coupler.toml')
    )

    call = _diamond_call(ax_ratio)
    assert 'w0_i0' in call.kwargs['label']
    # The diamond sits at the named case's ratio, not the better-scoring one.
    assert float(np.asarray(call.args[0])[0]) == pytest.approx(1.20, rel=1e-9)
    assert abs(1.20 - 1.01) > 1e-6
    saved = fig.savefig.call_args.args[0]
    assert str(saved).endswith('result_observables.png')


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

    plot_fit_mod.plot_result_observables(
        plot_fit_mod.collect_case_observables(tmp_path, {'R_obs': 1.0}),
        {'R_obs': 1.0},
        str(tmp_path),
        '/does/not/exist/init_coupler.toml',
    )

    assert 'using the best J' in caplog.text
    call = _diamond_call(ax_ratio)
    assert 'w0_i1' in call.kwargs['label']
    assert float(np.asarray(call.args[0])[0]) == pytest.approx(1.01, rel=1e-9)
    fig.savefig.assert_called_once()


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

    cases = plot_fit_mod.collect_case_observables(tmp_path, {'R_obs': 1.0})
    plot_fit_mod.plot_result_observables(cases, {'R_obs': 1.0}, str(tmp_path))

    assert expected in caplog.text
    assert mock_plt.subplots.call_count == 0
    assert not list((tmp_path / 'plots').glob('*.png'))


def test_panel_ratio_parks_out_of_range_cases_on_the_axis_edges():
    """Cases beyond the ratio axis are drawn on its edges, one legend each.

    A row whose cases all collapse to ~1e-22, or run far above target, would
    otherwise be silently empty. Both edge markers point outwards and each
    legend label is emitted only once across the rows.
    """
    lo, hi = plot_fit_mod.RATIO_CLIP
    obs = {'R_obs': 1.0, 'T_obs': 1.0}
    ok = pd.DataFrame(
        {
            'R_obs': [1e-6, 1.0],  # one below the axis, one inside
            'T_obs': [1e6, 1.0],  # one above the axis, one inside
            'case': ['w0_i0', 'w0_i1'],
        }
    )
    ax = MagicMock()

    plot_fit_mod._panel_ratio(ax, ok, obs, np.array([1.0, 1.0]), 'w0_i1')

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


def test_panel_residual_keeps_limits_ordered_for_a_single_observable():
    """A lone observable still yields a strictly increasing x-range.

    Edge case: the residual spread is zero with one bar, so the span used to
    scale the limits and the label offsets has to fall back to a finite value
    or matplotlib receives left == right.
    """
    ax = MagicMock()

    plot_fit_mod._panel_residual(ax, ['R_obs'], np.array([4.0]), best_J=1.25)

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


def test_position_in_range_uses_log10_for_log_scaled_parameters():
    """A log-scaled value is placed by its log10 within the range, matching
    how the optimiser normalises it; values beyond the range fall outside
    [0, 1] instead of being clipped.
    """
    # 1e4 in [1e3, 2e4]: log placement is 1 / log10(20) = 0.7686.
    got = float(
        plot_fit_mod._position_in_range(1.0e4, 'planet.elements.H_budget', [1.0e3, 2.0e4])
    )
    assert got == pytest.approx(1.0 / np.log10(20.0), rel=1e-12)
    # Discrimination guard: linear placement would give 9e3 / 1.9e4 = 0.4737.
    assert abs(got - 9.0e3 / 1.9e4) > 0.2
    lin = plot_fit_mod._position_in_range([-4.0, 0.0, 6.0], 'outgas.fO2_shift_IW', [-4.0, 4.0])
    np.testing.assert_allclose(lin, [0.0, 0.5, 1.25], rtol=0, atol=1e-12)
    # Edge case: a non-positive value has no log position and comes back non-finite.
    assert not np.isfinite(
        plot_fit_mod._position_in_range(0.0, 'planet.elements.H_budget', [1e3, 2e4])
    )


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

    cases = plot_fit_mod.collect_case_observables(tmp_path, {'R_obs': 1.0})
    plot_fit_mod.plot_result_parameters(
        cases, _PARS, _TRUTH, str(tmp_path), str(named / 'init_coupler.toml')
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


def test_plot_result_parameters_widens_axis_for_truth_outside_range(monkeypatch, tmp_path):
    """A true value beyond the sampled range stays visible: the axis widens
    past [0, 1] just far enough to include it.
    """
    (tmp_path / 'plots').mkdir()
    _write_param_case(tmp_path, 0, 0, 1.01, 0.0, 5.0e3)
    _mock_plt, _fig, ax_pos, _ax_err = _two_axes(monkeypatch)
    # fO2 shift of +6 lies at position 1.25 of the [-4, 4] range.
    truth = {'outgas.fO2_shift_IW': 6.0, 'planet.elements.H_budget': 5.0e3}

    cases = plot_fit_mod.collect_case_observables(tmp_path, {'R_obs': 1.0})
    plot_fit_mod.plot_result_parameters(cases, _PARS, truth, str(tmp_path))

    left, right = ax_pos.set_xlim.call_args.args
    assert right == pytest.approx(1.25 + 0.04, rel=1e-12)
    assert left == pytest.approx(-0.04, rel=1e-12)


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

    cases = plot_fit_mod.collect_case_observables(tmp_path, {'R_obs': 1.0})
    plot_fit_mod.plot_result_parameters(cases, _PARS, _TRUTH, str(tmp_path))

    assert 'No scored case to compare with the true parameters' in caplog.text
    assert mock_plt.subplots.call_count == 0
    assert not list((tmp_path / 'plots').glob('*.png'))
