"""
Unit tests for the run label, the smoothing helper, the colour helpers, the
command line and ``make_figure`` of ``tools/plot_chili_comparison.py``, on
small tables in a temporary folder.
"""

from __future__ import annotations

import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import pytest

from ._plot_chili_helpers import (
    DARK_BY_HAND,
    _comparison_data,
    _grid_case,
    _hex,
    _main_arguments,
    _png_width,
    _repo,
    _run_folder,
    _run_table,
    _run_with_log,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('close_figures'),
]


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


def test_composition_bars_hold_o2_of_the_models_and_of_the_run(tool, tmp_path):
    """The composition panel at 95% melt carries the O2 partial pressure of a
    model file and of the current run, beside the other gases."""
    names, gas, _, is_run = tool._collect_atm_panel(
        _comparison_data(tmp_path / 'data'), _run_table(), 'earth', 0.95, tool._new_style('x')
    )

    assert (names, is_run) == (['MOAI', 'PROTEUS (x)'], [False, True])
    assert gas['O2'] == pytest.approx([0.25, 128.0])
    assert gas['H2O'] == pytest.approx([5.0, 4.0])
    assert 'O2' in tool.GAS_LABELS and tool._colour('O2') == '#00dcc6'


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


def test_bar_of_a_colour_close_to_the_surface_gets_a_grey_edge(tool):
    """SO2 is close to a dark surface (contrast 1.3), so its bars get a mid grey
    edge there and a surface-coloured edge on a light surface; the bar of the
    current run has the text colour in both cases."""
    assert tool._contrast('#000000', '#ffffff') == pytest.approx(21.0)
    assert tool._contrast('#333333', '#ffffff') == pytest.approx(12.63, abs=0.005)
    assert tool._contrast('#bbbbbb', '#F2F5F7') == pytest.approx(1.75, abs=0.005)
    assert tool._contrast('#00008B', '#05070B') == pytest.approx(1.32, abs=0.005)
    assert tool._contrast('#05070B', '#00008B') == tool._contrast('#00008B', '#05070B')

    light = [_hex(c) for c in tool._bar_edges('#00008B', [False, True])]
    matplotlib.rcParams.update(DARK_BY_HAND)
    dark = [_hex(c) for c in tool._bar_edges('#00008B', [False, True])]
    water = [_hex(c) for c in tool._bar_edges('#027FB1', [False, True])]

    assert light == ['#ffffff', '#000000']
    assert dark == ['#777a7e', '#e9eef2']
    assert water == ['#05070b', '#e9eef2']


def test_style_option_calls_proteus_mpl_and_the_default_call_does_not_need_it(
    tool, tmp_path, monkeypatch, capsys
):
    """``--style dark`` hands the variant to ``proteus_mpl.use``; without the
    package it stops with exit status 2 and names the package; a call without
    ``--style`` draws without the package."""
    used = []
    monkeypatch.setitem(sys.modules, 'proteus_mpl', types.SimpleNamespace(use=used.append))
    seen = _main_arguments(tool, tmp_path, monkeypatch, '--style', 'dark')
    tool.main()
    assert used == ['dark'] and len(seen) == 10

    monkeypatch.setitem(sys.modules, 'proteus_mpl', None)
    with pytest.raises(SystemExit) as stop:
        tool.main()
    assert stop.value.code == 2
    assert 'pip install proteus-mpl' in capsys.readouterr().err
    assert len(seen) == 10

    _main_arguments(tool, tmp_path, monkeypatch)
    tool.main()
    assert used == ['dark']
    monkeypatch.setitem(sys.modules, 'proteus_mpl', types.SimpleNamespace(use=used.append))
    monkeypatch.setattr(tool.sys, 'argv', ['plot', '--style', 'white'])
    with pytest.raises(SystemExit):
        tool.main()
    assert used == ['dark']


def test_main_writes_a_png_and_a_pdf_for_each_figure_with_data(tool, tmp_path, monkeypatch):
    """``main`` saves each figure that a function returns as PNG and as PDF and
    closes it; a function without a figure leaves no file. Figure 5 without
    a model and without a run gives no figure."""
    _main_arguments(tool, tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'plot_psurf', lambda *args: tool.plt.figure())

    tool.main()

    written = sorted(path.name for path in (tmp_path / 'out').iterdir())
    assert written == ['chili_psurf_vs_time.pdf', 'chili_psurf_vs_time.png']
    assert tool.plt.get_fignums() == []
    (tmp_path / 'no_models').mkdir()
    assert tool.plot_fig5(tmp_path / 'no_models', None, tool._new_style('x')) is None
    assert tool.plt.get_fignums() == []


def test_make_figure_writes_the_png_at_the_dpi_of_the_style_and_labels_it_with_the_run_it_reads(
    tool, tmp_path
):
    """``make_figure`` writes the PNG to the given path, in a folder that it
    makes, at ``savefig.dpi`` of the active style; the label holds the commit
    of the run folder that the figure reads, not of another folder in
    ``run_dirs``."""
    earth = _run_folder(tmp_path / 'earth')
    other = _run_with_log(tmp_path / 'other', '81e1514466a590a9fd97acea99150ea5d978a751')
    out = tmp_path / 'new_folder' / 'figure.png'
    matplotlib.rcParams.update(DARK_BY_HAND | {'savefig.dpi': 50})

    fig = tool.make_figure(
        [str(other), earth],
        out,
        fig_id='chili_psurf_vs_time',
        chili_repo=str(_repo(tmp_path)),
        earth=str(earth),
    )

    assert _png_width(out) == 350
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    assert labels == ['MOAI', 'PROTEUS (d0c7f080)']
    assert fig.axes[0].lines[1].get_ydata() == pytest.approx([25.0, 140.0, 310.0])
    assert _hex(fig.get_facecolor()) == '#05070b'
    assert len(fig.legends) == 0 and fig.axes[0].get_legend() is not None


def test_make_figure_reads_the_grid_runs_and_names_their_commits(tool, tmp_path):
    """The grid figure reads the grid run folders with a helpfile; the label
    holds their commits in the order of the grid, not of ``run_dirs``, and a
    grid run that ``run_dirs`` does not list is refused."""
    data, grid = _grid_case(tmp_path)
    repo = tmp_path / 'repo'
    (repo / 'intercomparison').mkdir(parents=True)
    data.rename(repo / 'intercomparison' / 'outputs')
    runs = [
        _run_folder(grid / 'chili_grid_earth_Hhigh_Clow', '3' * 40),
        _run_folder(grid / 'chili_grid_earth_Hlow_Chigh', '1' * 40),
        grid / 'chili_grid_earth_Hlow_Clow',
    ]
    out = tmp_path / 'grid.png'

    fig = tool.make_figure(
        runs, out, fig_id='chili_grid_solidification', chili_repo=repo, grid_dir=str(grid)
    )

    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert 'PROTEUS (d0c7f080/11111111/33333333)' in legend and out.exists()
    with pytest.raises(ValueError, match='run_dirs does not hold these run folders'):
        tool.make_figure(
            [tmp_path], out, fig_id='chili_grid_solidification', chili_repo=repo, grid_dir=grid
        )


def test_make_figure_refuses_a_figure_without_the_run_that_it_reads(
    tool, tmp_path, monkeypatch
):
    """A wrong figure name, a run folder that the figure reads and that is not
    given or holds no helpfile, and a wrong ``run_dirs`` raise an error with
    the reason, before the comparison data is fetched, and write no file."""

    def no_clone(*args, **kwargs):
        raise AssertionError('a refused figure must not fetch the comparison data')

    monkeypatch.setattr(tool.subprocess, 'run', no_clone)
    missing = tmp_path / 'no_checkout'
    out = tmp_path / 'figure.png'
    earth = _run_folder(tmp_path / 'earth')
    empty = _run_with_log(tmp_path / 'empty', '81e1514466a590a9fd97acea99150ea5d978a751')

    def make(fig_id, runs=(), **folders):
        return tool.make_figure(list(runs), out, fig_id=fig_id, chili_repo=missing, **folders)

    with pytest.raises(ValueError, match="unknown figure 'chili_fig10'; known: chili_fig1_"):
        make('chili_fig10')
    needs = {
        'earth': [
            'fig1_melt_fraction',
            'fig2_milestones',
            'fig3_atm_composition',
            'fig9_geodynamics',
        ],
        'venus': ['fig5_venus_atm', 'fig6_fO2_vs_T', 'fig7_volatiles'],
        'grid_dir': ['grid_solidification'],
    }
    for name, stems in needs.items():
        for stem in stems:
            with pytest.raises(ValueError, match=f"figure 'chili_{stem}' needs {name}"):
                make(f'chili_{stem}', venus=earth if name == 'earth' else None)
    for no_grid in (None, ''):
        with pytest.raises(ValueError, match="figure 'chili_fig2_milestones' needs grid_dir"):
            make('chili_fig2_milestones', [earth], earth=earth, grid_dir=no_grid)
    with pytest.raises(ValueError, match='no run data in .*empty for figure'):
        make('chili_psurf_vs_time', [empty], earth=empty)
    with pytest.raises(ValueError, match='no run data in .*empty for figure'):
        make('chili_grid_solidification', grid_dir=empty)
    with pytest.raises(ValueError, match='run_dirs does not hold these run folders'):
        make('chili_psurf_vs_time', [empty], earth=earth)
    for wrong in (None, str(earth), earth):
        with pytest.raises(TypeError, match='run_dirs must be a list of run folders'):
            tool.make_figure(
                wrong, out, fig_id='chili_psurf_vs_time', chili_repo=missing, earth=earth
            )
    assert not out.exists() and not missing.exists()


def test_make_figure_closes_a_figure_that_it_cannot_draw_or_write(tool, tmp_path, monkeypatch):
    """A figure function that fails after it opened a figure, and a file that
    cannot be written, give ``ValueError`` with the cause and leave no open
    figure."""
    earth = _run_folder(tmp_path / 'earth')
    repo = _repo(tmp_path)
    (tmp_path / 'a_file').write_text('')

    def make(out):
        return tool.make_figure(
            [earth], out, fig_id='chili_psurf_vs_time', chili_repo=repo, earth=earth
        )

    with pytest.raises(ValueError, match="figure 'chili_psurf_vs_time' could not be made"):
        make(tmp_path / 'a_file' / 'figure.png')
    assert tool.plt.get_fignums() == []

    def broken(*args):
        tool.plt.figure()
        raise RuntimeError('no bars to draw')

    monkeypatch.setattr(tool, 'plot_psurf', broken)
    with pytest.raises(ValueError, match='could not be made: no bars to draw'):
        make(tmp_path / 'figure.png')
    assert tool.plt.get_fignums() == [] and not (tmp_path / 'figure.png').exists()


def test_interior_profiles_are_read_for_figure_9_only(tool, tmp_path, monkeypatch):
    """``make_figure`` reads the interior profiles of the Earth run for Figure 9
    and for no other figure; the grid runs are read without them."""
    data, grid = _grid_case(tmp_path)
    repo = tmp_path / 'repo'
    (repo / 'intercomparison').mkdir(parents=True)
    data.rename(repo / 'intercomparison' / 'outputs')
    earth = _run_folder(tmp_path / 'earth')
    runs = [earth, grid / 'chili_grid_earth_Hlow_Clow']
    read = []
    empty = {'Phi_global': np.array([]), 'R_rheo': np.array([]), 'visc_avg': np.array([])}
    monkeypatch.setattr(
        tool, '_extract_profiles', lambda folder, table: read.append(folder.name) or empty
    )

    for stem in ('psurf_vs_time', 'fig2_milestones', 'fig9_geodynamics'):
        tool.make_figure(
            runs,
            tmp_path / f'{stem}.png',
            fig_id=f'chili_{stem}',
            chili_repo=repo,
            earth=earth,
            grid_dir=grid if stem == 'fig2_milestones' else None,
        )
        assert read == (['earth'] if stem == 'fig9_geodynamics' else []), stem
    assert tool._load_proteus_helpfile(earth).attrs['_profiles'] is empty
