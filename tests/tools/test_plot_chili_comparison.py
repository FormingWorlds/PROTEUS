"""
Unit tests for helpers of the CHILI comparison plot script.

These tests cover the run label, the smoothing helper and the figures of
``tools/plot_chili_comparison.py`` that take data of the current run, on
small tables in a temporary folder.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture(scope='module')
def tool():
    """The co-located ``plot_chili_comparison.py``, loaded as a module."""
    script_path = Path(__file__).resolve().parents[2] / 'tools' / 'plot_chili_comparison.py'
    spec = importlib.util.spec_from_file_location('plot_chili_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_with_log(path: Path, sha: str) -> Path:
    path.mkdir()
    (path / 'proteus_00.log').write_text(
        '[ INFO  ] PROTEUS version   26.7.14.post1.dev148+gd0c7f0806\n'
        f'[ INFO  ] PROTEUS git hash  {sha}\n'
    )
    return path


@pytest.fixture
def figures(tool, monkeypatch):
    """Figures that the script hands to ``_save``, by name, kept open for the test."""
    drawn = {}
    monkeypatch.setattr(tool, '_save', lambda fig, out, name: drawn.__setitem__(name, fig))
    yield drawn
    for fig in drawn.values():
        tool.plt.close(fig)


def _comparison_data(root: Path) -> Path:
    """A comparison folder with one model file for Earth and one for Venus."""
    (root / 'moai').mkdir(parents=True)
    table = pd.DataFrame(
        {
            't(yr)': [0.0, 1e3, 1e5],
            'T_surf(K)': [3000.0, 2500.0, 1700.0],
            'phi(vol_frac)': [0.99, 0.95, 0.05],
            'fO2_melt(bar)': [1e-2, 1e-4, 1e-8],
            'p_H2O(bar)': [1.0, 5.0, 70.0],
            'p_O2(bar)': [1e-2, 0.25, 1e-8],
        }
    )
    for planet in ('earth', 'venus'):
        table.to_csv(root / 'moai' / f'evolution-moai-{planet}-data.csv')
    return root


def _run_table() -> pd.DataFrame:
    """Rows of a run in the columns of a helpfile that the figures read."""
    return pd.DataFrame(
        {
            'Time': [1.0, 1e4, 1.13e7],
            'Phi_global': [1.0, 0.95, 0.05],
            'T_surf': [3300.0, 2900.0, 1800.0],
            'T_magma': [4200.0, 3100.0, 1810.0],
            'H2O_bar': [0.5, 4.0, 300.0],
            'O2_bar': [2.0e4, 128.0, 2.0e-5],
        }
    )


def test_run_label_names_the_commits_of_the_run_logs_in_argument_order(tool, tmp_path):
    """The label gives the commit of each run log once, in the order of the
    arguments, and takes it from the hash line, not from the version line."""
    a = _run_with_log(tmp_path / 'a', 'd0c7f08062bdf08631b61225bad460c6d4e381eb')
    b = _run_with_log(tmp_path / 'b', '81e1514466a590a9fd97acea99150ea5d978a751')
    c = _run_with_log(tmp_path / 'c', '0a1b2c3d4e5f60718293a4b5c6d7e8f901234567')

    assert tool._get_proteus_sha(a, None) == 'd0c7f080'
    assert tool._get_proteus_sha(a, a) == 'd0c7f080'
    assert tool._get_proteus_sha(a, b, c) == 'd0c7f080/81e15144/0a1b2c3d'
    assert tool._get_proteus_sha(c, a, b, a) == '0a1b2c3d/d0c7f080/81e15144'
    assert tool._get_proteus_sha(b, c, a) == '81e15144/0a1b2c3d/d0c7f080'
    with open(b / 'proteus_00.log', 'a') as log:
        log.write('[ INFO  ] PROTEUS git hash  0a1b2c3d4e5f60718293a4b5c6d7e8f901234567\n')
    assert tool._get_proteus_sha(b) == '81e15144/0a1b2c3d'


@pytest.mark.parametrize(
    'hash_line',
    [b'', b'[ INFO  ] PROTEUS git hash  unknown\n', b'[ INFO  ] PROTEUS git hash  D0C7F080\n'],
)
def test_run_label_is_unknown_for_a_log_without_a_commit(
    tool, tmp_path, monkeypatch, capsys, hash_line
):
    """A log without a usable hash line gives the label ``unknown`` and a warning,
    never the commit of the checkout; bytes that are not UTF-8 do not stop it."""

    def no_git(*args, **kwargs):
        raise AssertionError('the label must not ask git for the checkout commit')

    monkeypatch.setattr(tool.subprocess, 'run', no_git)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'proteus_00.log').write_bytes(
        b'[ INFO  ] PROTEUS version   26.7.14.post1.dev148+gd0c7f0806 \xff\xfe\n' + hash_line
    )
    (tmp_path / 'empty').mkdir()
    good = _run_with_log(tmp_path / 'good', 'd0c7f08062bdf08631b61225bad460c6d4e381eb')

    assert tool._get_proteus_sha(run) == 'unknown'
    assert 'no commit in' in capsys.readouterr().err
    assert tool._get_proteus_sha(good, run) == 'd0c7f080/unknown'
    assert tool._get_proteus_sha(tmp_path / 'empty', good, run) == 'unknown/d0c7f080'
    assert 'no commit in' in capsys.readouterr().err
    assert tool._get_proteus_sha(tmp_path / 'not_a_folder', None, good) == 'd0c7f080'
    assert tool._get_proteus_sha(good / 'proteus_00.log', good) == 'd0c7f080'
    assert tool._get_proteus_sha() == 'unknown'
    assert capsys.readouterr().err == ''


def test_figure_1_spans_20_myr_and_names_the_run_commit(tool, tmp_path, figures):
    """Figure 1 reaches 20 Myr, so a run that ends at 11.3 Myr is drawn in full;
    its legend names the commit of the run and its y label names both bases."""
    style = tool._new_style('d0c7f080')

    tool.plot_fig1(
        _comparison_data(tmp_path / 'data'), _run_table(), _run_table(), style, tmp_path
    )

    ax = figures['chili_fig1_melt_fraction'].axes[0]
    assert ax.get_xlim() == pytest.approx((1e-3, 2e1))
    assert ax.get_xlim()[1] > _run_table()['Time'].iloc[-1] / 1e6
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ['MOAI', 'PROTEUS (d0c7f080) Earth', 'PROTEUS (d0c7f080) Venus']
    assert ax.get_ylabel() == 'Melt fraction\n(by volume; current runs by mass)'


def test_composition_bars_hold_o2_of_the_models_and_of_the_run(tool, tmp_path):
    """The composition panel at 95% melt carries the O2 partial pressure of a
    model file and of the current run, beside the other gases."""
    names, gas, _, is_run = tool._collect_atm_panel(
        _comparison_data(tmp_path / 'data'), _run_table(), 'earth', 0.95, tool._new_style('x')
    )

    assert (names, is_run) == (['MOAI', 'PROTEUS (x)'], [False, True])
    assert gas['O2'] == pytest.approx([0.25, 128.0])
    assert gas['H2O'] == pytest.approx([5.0, 4.0])
    assert 'O2' in tool.GAS_LABELS and 'O2' in tool.GAS_COLORS


def test_figure_3_names_each_panel_in_its_title_and_draws_o2(tool, tmp_path, figures):
    """Figure 3 carries the panel label in the title of each panel, clear of the
    data, and its 95% panel has an O2 bar segment of the current run."""
    tool.plot_fig3(
        _comparison_data(tmp_path / 'data'), _run_table(), tool._new_style('x'), tmp_path
    )

    panel_a, panel_b = figures['chili_fig3_atm_composition'].axes[:2]
    assert '(a)' in panel_a.get_title() and '95%' in panel_a.get_title()
    assert '(b)' in panel_b.get_title() and '5%' in panel_b.get_title()
    assert [text.get_text() for text in panel_a.texts + panel_b.texts] == []
    heights = [patch.get_height() for patch in panel_a.patches]
    assert 128.0 in heights and 0.25 in heights


def test_figure_6_draws_the_o2_fugacity_of_the_run_at_its_magma_temperature(
    tool, tmp_path, figures
):
    """The curve of the current run in Figure 6 is log10 of its O2 partial
    pressure against T_magma, with the marker on its row at 5% melt."""
    run = _run_table()

    tool.plot_fig6(_comparison_data(tmp_path / 'data'), run, tool._new_style('x'), tmp_path)

    ax_abs, ax_rel = figures['chili_fig6_fO2_vs_T'].axes
    curve = [line for line in ax_abs.lines if line.get_label() == 'PROTEUS (x)'][0]
    marker = [line for line in ax_abs.lines if line.get_markersize() == 10][0]
    assert curve.get_xdata() == pytest.approx(run['T_magma'].values)
    assert curve.get_ydata() == pytest.approx(np.log10(run['O2_bar'].values))
    assert curve.get_xdata() != pytest.approx(run['T_surf'].values)
    assert marker.get_xdata() == pytest.approx([1810.0])
    assert marker.get_ydata() == pytest.approx([np.log10(2.0e-5)])
    relative = [line for line in ax_rel.lines if line.get_label() == 'PROTEUS (x)'][0]
    expected = np.log10(run['O2_bar'].values) - tool._iw_oneill02(run['T_magma'].values)
    assert relative.get_ydata() == pytest.approx(expected)
    marker_rel = [line for line in ax_rel.lines if line.get_markersize() == 10][0]
    assert marker_rel.get_xdata() == pytest.approx([1810.0])
    assert marker_rel.get_ydata() == pytest.approx([expected[-1]])
    assert ax_rel.get_ylim()[0] == pytest.approx(-0.5)
    assert ax_rel.get_legend() is None and ax_abs.get_legend() is not None


def test_figure_6_has_no_marker_for_a_run_that_ends_above_6_percent_melt(
    tool, tmp_path, figures
):
    """A run that ends at 6.8% melt, as the Nominal Venus run does, gets a curve
    and no circular marker; a row without O2 is left out of the curve."""
    run = _run_table()
    run['Phi_global'] = [1.0, 0.95, 0.068]
    run.loc[0, 'O2_bar'] = 0.0

    tool.plot_fig6(_comparison_data(tmp_path / 'data'), run, tool._new_style('x'), tmp_path)

    for ax in figures['chili_fig6_fO2_vs_T'].axes:
        assert [line for line in ax.lines if line.get_markersize() == 10] == []
        curve = [line for line in ax.lines if line.get_label() == 'PROTEUS (x)'][0]
        assert curve.get_xdata() == pytest.approx([3100.0, 1810.0])


def test_main_labels_the_figures_with_the_commits_of_all_run_folders(
    tool, tmp_path, monkeypatch
):
    """``main`` builds the label from the Earth run, the Venus run and the grid
    runs, in this order, and hands it to every figure function."""
    earth = _run_with_log(tmp_path / 'earth', 'd0c7f08062bdf08631b61225bad460c6d4e381eb')
    venus = _run_with_log(tmp_path / 'venus', '81e1514466a590a9fd97acea99150ea5d978a751')
    (tmp_path / 'grid').mkdir()
    for name, sha in (
        ('Hhigh_Clow', '3' * 40),
        ('Hmid_Clow', '2' * 40),
        ('Hlow_Chigh', '1' * 40),
        ('Hlow_Clow', '0' * 40),
    ):
        _run_with_log(tmp_path / 'grid' / f'chili_grid_earth_{name}', sha)
    _run_with_log(tmp_path / 'grid' / 'chili_grid_earth_Hlow_Clow_old', 'f' * 40)
    seen = {}
    for name in [n for n in dir(tool) if n.startswith('plot_')]:
        monkeypatch.setattr(
            tool, name, lambda *args, _name=name, **kwargs: seen.__setitem__(_name, args)
        )
    monkeypatch.setattr(tool, 'ensure_chili_repo', lambda path: path)
    argv = ['plot', '--proteus-earth', str(earth), '--proteus-venus', str(venus)]
    argv += ['--output', str(tmp_path / 'out')]

    for grid, label in (
        (
            ['--grid-dir', str(tmp_path / 'grid')],
            'd0c7f080/81e15144/00000000/11111111/22222222/33333333',
        ),
        ([], 'd0c7f080/81e15144'),
    ):
        seen.clear()
        monkeypatch.setattr(tool.sys, 'argv', argv + grid)

        tool.main()

        assert ('plot_grid_timescales' in seen) == bool(grid)
        assert {'plot_fig1', 'plot_fig6', 'plot_psurf'} <= set(seen)
        for name, args in seen.items():
            style = args[3] if name == 'plot_fig1' else args[2]
            assert style['label'] == f'PROTEUS ({label})', name


def test_smooth_uses_a_centred_mean_of_9_points_by_default(tool):
    """The default window is 9 points, centred; near an end and for an array
    shorter than the window the mean covers the points that exist."""
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0])

    out = tool._smooth(y)

    assert out[5] == pytest.approx(np.mean(y[1:10]))
    assert out[1] == pytest.approx(np.mean(y[0:6]))
    assert tool._smooth(y[:5])[2] == pytest.approx(np.mean(y[:5]))


def test_smooth_returns_short_and_non_positive_input_unchanged(tool):
    """Fewer than 3 points are returned as they are, and so is a column with a
    value at or below zero when the mean is taken in log space."""
    assert tool._smooth([3.0, 7.0]) == pytest.approx([3.0, 7.0])
    with_zero = np.array([4.0, 0.0, 9.0, 25.0, 1.0])
    assert tool._smooth(with_zero, log=True) == pytest.approx(with_zero)
    assert tool._smooth(with_zero)[2] != pytest.approx(with_zero[2])


@pytest.mark.parametrize('log', [False, True])
def test_smooth_accepts_a_helpfile_column_and_keeps_the_end_points(tool, log):
    """A column of a helpfile (a read-only array in pandas 3) is smoothed without
    an error; the first and last value stay exact and the middle is averaged."""
    column = pd.DataFrame({'y': [1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 8.0]})['y'].values

    out = tool._smooth(column, window=3, log=log)

    assert (out[0], out[-1]) == (column[0], column[-1])
    window = [100.0, 1.0, 100.0]
    middle = 10 ** np.mean(np.log10(window)) if log else np.mean(window)
    assert out[2] == pytest.approx(middle)
    assert out[2] != pytest.approx(column[2])
