"""
Unit tests for helpers of the CHILI comparison plot script.

These tests cover the run label, the smoothing helper, the colours and the
figures of ``tools/plot_chili_comparison.py``, on small tables in a temporary
folder, under a dark style set by hand and under both proteus-mpl styles.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
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


# Text, surface and line colours of a dark style, none of them pure black or white.
DARK_BY_HAND = {
    'text.color': '#E9EEF2',
    'axes.labelcolor': '#E9EEF2',
    'axes.titlecolor': '#E9EEF2',
    'axes.edgecolor': '#E9EEF2',
    'xtick.color': '#9FB0BE',
    'ytick.color': '#9FB0BE',
    'axes.facecolor': '#05070B',
    'figure.facecolor': '#05070B',
    'hatch.color': '#E9EEF2',
}

# Model colours of the CHILI analysis code and the gas colours of the page.
MODEL_COLOURS = {
    'GOOEY': '#bbbbbb',
    'NEONGOOEY': '#ee6677',
    'PROTEUS CHILI': '#eb7d0f',
    'PACMAN': '#4477aa',
    'LINCS': '#228833',
    'MOAI': '#aa3377',
    'PlanAtMO': '#ccbb44',
}
MODEL_KEYS = ('gooey', 'neongooey', 'proteus', 'pacman', 'lincs', 'moai', 'planatmo')
GAS_COLOURS = {
    'H2O': '#027fb1',
    'CO2': '#d24901',
    'CO': '#d1ac02',
    'H2': '#008c01',
    'CH4': '#c720dd',
    'N2': '#870036',
    'S2': '#ff8fa1',
    'SO2': '#00008b',
    'H2S': '#640deb',
    'O2': '#00dcc6',
}


@pytest.fixture(autouse=True)
def close_figures(tool):
    """Each test starts from the default style and leaves no open figure."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        yield
    tool.plt.close('all')


@pytest.fixture(params=['dark_by_hand', 'light', 'dark'])
def style(request):
    """A style active for the test: dark colours set by hand, or a proteus-mpl style."""
    if request.param == 'dark_by_hand':
        matplotlib.rcParams.update(DARK_BY_HAND)
    else:
        pytest.importorskip('proteus_mpl').use(request.param)
    return request.param


def _comparison_data(root: Path, models=('moai',)) -> Path:
    """A comparison folder with a file for Earth and for Venus of each model."""
    table = pd.DataFrame(
        {
            't(yr)': [0.0, 1e3, 1e5],
            'T_surf(K)': [3000.0, 2500.0, 1700.0],
            'phi(vol_frac)': [0.99, 0.95, 0.05],
            'fO2_melt(bar)': [1e-2, 1e-4, 1e-8],
            'p_H2O(bar)': [1.0, 5.0, 70.0],
            'p_O2(bar)': [1e-2, 0.25, 1e-8],
            'p_SO2(bar)': [0.0, 2.0, 3.0],
            'p_surf(bar)': [20.0, 30.0, 80.0],
            'flux_OLR(W/m2)': [1e5, 1e4, 300.0],
            'R_solid(m)': [3.5e6, 4.0e6, 6.3e6],
            'viscosity(Pa.s)': [1e2, 1e3, 1e20],
            'massH_atm(kg)': [1e20, 2e20, 3e20],
            'massH_melt(kg)': [6e20, 5e20, 1e20],
            'massH_solid(kg)': [0.0, 0.0, 3e20],
            'massC_atm(kg)': [1e20, 2e20, 4e20],
            'massC_melt(kg)': [4e20, 3e20, 5e19],
            'massC_solid(kg)': [0.0, 0.0, 5e19],
        }
    )
    for model in models:
        (root / model).mkdir(parents=True)
        for planet in ('earth', 'venus'):
            table.to_csv(root / model / f'evolution-{model}-{planet}-data.csv')
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
            'SO2_bar': [0.0, 1.5, 2.5],
            'P_surf': [25.0, 140.0, 310.0],
            'F_olr': [2e5, 2e4, 280.0],
            'R_core': [3.48e6, 3.48e6, 3.48e6],
            'H_kg_atm': [1e20, 2e20, 3e20],
            'H_kg_liquid': [6e20, 5e20, 1e20],
            'H_kg_solid': [0.0, 0.0, 3e20],
            'H_kg_total': [7e20, 7e20, 7e20],
            'C_kg_atm': [1e20, 2e20, 4e20],
            'C_kg_liquid': [4e20, 3e20, 5e19],
            'C_kg_solid': [0.0, 0.0, 5e19],
            'C_kg_total': [5e20, 5e20, 5e20],
        }
    )


def _run_folder(path: Path, sha: str = 'd0c7f08062bdf08631b61225bad460c6d4e381eb') -> Path:
    """A run folder with a log and a helpfile of ``_run_table``."""
    _run_with_log(path, sha)
    _run_table().to_csv(path / 'runtime_helpfile.csv', sep=' ', index=False)
    return path


def _grid_case(tmp_path: Path) -> tuple[Path, Path]:
    """Comparison data with a grid file of the submitted run and of GOOEY, and a
    grid folder with one run."""
    data = _comparison_data(tmp_path / 'data', models=MODEL_KEYS)
    table = pd.read_csv(data / 'proteus' / 'evolution-proteus-earth-data.csv', index_col=0)
    table.to_csv(data / 'proteus' / 'evolution-proteus-earth-grid-Hlow-Clow-data.csv')
    table.to_csv(data / 'gooey' / 'evolution-gooey-earth-grid-Hlow-data.csv')
    (tmp_path / 'grid').mkdir()
    _run_folder(tmp_path / 'grid' / 'chili_grid_earth_Hlow_Clow')
    return data, tmp_path / 'grid'


def _hex(colour) -> str:
    return matplotlib.colors.to_hex(colour)


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


def test_figure_1_spans_20_myr_and_names_the_run_commit(tool, tmp_path):
    """Figure 1 reaches 20 Myr, so a run that ends at 11.3 Myr is drawn in full;
    its legend names the commit of the run and its y label names both bases."""
    style = tool._new_style('d0c7f080')

    fig = tool.plot_fig1(_comparison_data(tmp_path / 'data'), _run_table(), _run_table(), style)

    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((1e-3, 2e1))
    assert ax.get_xlim()[1] > _run_table()['Time'].iloc[-1] / 1e6
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
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
    assert 'O2' in tool.GAS_LABELS and tool._colour('O2') == '#00dcc6'


def test_figure_3_names_each_panel_in_its_title_and_draws_o2(tool, tmp_path):
    """Figure 3 carries the panel label in the title of each panel, clear of the
    data, and its 95% panel has an O2 bar segment of the current run."""
    fig = tool.plot_fig3(
        _comparison_data(tmp_path / 'data'), _run_table(), tool._new_style('x')
    )

    panel_a, panel_b = fig.axes[:2]
    assert '(a)' in panel_a.get_title() and '95%' in panel_a.get_title()
    assert '(b)' in panel_b.get_title() and '5%' in panel_b.get_title()
    assert [text.get_text() for text in panel_a.texts + panel_b.texts] == []
    heights = [patch.get_height() for patch in panel_a.patches]
    assert 128.0 in heights and 0.25 in heights


def test_figure_6_draws_the_o2_fugacity_of_the_run_at_its_magma_temperature(tool, tmp_path):
    """The curve of the current run in Figure 6 is log10 of its O2 partial
    pressure against T_magma, with the marker on its row at 5% melt."""
    run = _run_table()

    fig = tool.plot_fig6(_comparison_data(tmp_path / 'data'), run, tool._new_style('x'))

    ax_abs, ax_rel = fig.axes
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
    assert (ax_rel.get_legend(), ax_abs.get_legend(), len(fig.legends)) == (None, None, 1)
    for label, buffer in (
        ('Fischer+11', tool._iw_fischer11),
        ("O'Neill+02", tool._iw_oneill02),
    ):
        line = [ln for ln in ax_abs.lines if ln.get_label() == label][0]
        assert line.get_ydata() == pytest.approx(buffer(line.get_xdata()))
        assert line.get_linestyle() == {'Fischer+11': ':', "O'Neill+02": '--'}[label]
        assert _hex(line.get_color()) == '#000000'
    assert tool._iw_fischer11(2000.0) != pytest.approx(tool._iw_oneill02(2000.0), abs=0.05)


def test_figure_6_has_no_marker_for_a_run_that_ends_above_6_percent_melt(tool, tmp_path):
    """A run that ends at 6.8% melt, as the Nominal Venus run does, gets a curve
    and no circular marker; a row without O2 is left out of the curve."""
    run = _run_table()
    run['Phi_global'] = [1.0, 0.95, 0.068]
    run.loc[0, 'O2_bar'] = 0.0

    fig = tool.plot_fig6(_comparison_data(tmp_path / 'data'), run, tool._new_style('x'))

    for ax in fig.axes:
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


# Black, white and the greys 'grey', '0.25' and '0.4' as literal colours.
LITERAL_NEUTRALS = {'#000000', '#ffffff', '#808080', '#404040', '#666666'}


def _colours(fig) -> set[str]:
    """Hex colours of every visible text, text box, line, marker and patch of a figure."""
    found = set()
    for artist in fig.findobj(lambda a: a is not fig.patch and a.get_visible()):
        for getter in ('get_color', 'get_markeredgecolor', 'get_markerfacecolor'):
            if isinstance(artist, (matplotlib.text.Text, matplotlib.lines.Line2D)):
                value = getattr(artist, getter, lambda: 'none')()
                if matplotlib.colors.to_rgba(value)[3] > 0:
                    found.add(_hex(value))
        box = artist.get_bbox_patch() if isinstance(artist, matplotlib.text.Text) else None
        for patch in (artist, box):
            if isinstance(patch, matplotlib.patches.Patch):
                for value in (patch.get_facecolor(), patch.get_edgecolor()):
                    if value[3] > 0:
                        found.add(_hex(value))
    return found


def _all_figures(tool, tmp_path):
    """The 11 figures of the page on small tables, by file stem."""
    data, grid = _grid_case(tmp_path)
    earth = tool._load_proteus_helpfile(_run_folder(tmp_path / 'earth'))
    venus = earth.assign(Time=earth['Time'] * 2, T_magma=earth['T_magma'] + 100)
    calls = tool._figures(data, earth, venus, tool._new_style('d0c7f080'), grid)
    return {name: draw() for name, draw in calls.items()}


def test_each_file_stem_draws_its_own_figure_from_its_own_run(tool, tmp_path):
    """Each file stem gives the figure of its name: the panel count and an axis
    label identify it; Figures 6 and 7 draw the Venus run, the surface pressure
    figure and Figure 8 the Earth run, and Figure 2 the grid runs; the pale grid
    line of GOOEY is drawn less transparent than its carbon level asks."""
    figures = _all_figures(tool, tmp_path)

    seen = {
        name[6:]: (len(fig.axes), fig.axes[0].get_ylabel(), fig.axes[-1].get_xlabel())
        for name, fig in figures.items()
    }
    assert seen == {
        'fig1_melt_fraction': (
            1,
            'Melt fraction\n(by volume; current runs by mass)',
            'Time (Myr)',
        ),
        'fig2_milestones': (3, '', 'Earth simulated time [Myr]'),
        'fig3_atm_composition': (4, 'Partial pressure [bar]', ''),
        'fig4_mass_budgets': (3, r'$10^{20}$ kg', ''),
        'fig5_venus_atm': (2, 'Partial pressure [bar]', ''),
        'fig6_fO2_vs_T': (
            2,
            r'$\log_{10}(f\mathrm{O}_2 / \mathrm{bar})$',
            'Degassing temperature [K]',
        ),
        'fig7_volatiles': (2, 'Hydrogen', 'Simulated time [Myr]'),
        'fig8_olr': (2, r'Outgoing LW radiation [W/m$^2$]', r'$T_\mathrm{surf}$ [K]'),
        'fig9_geodynamics': (3, r'$T_\mathrm{surf}$ [K]', 'Melt fraction [vol%]'),
        'psurf_vs_time': (1, 'Surface pressure (bar)', 'Time (Myr)'),
        'grid_solidification': (
            1,
            'Solidification time (Myr)',
            r'H inventory ($\times 10^{20}$ kg)',
        ),
    }
    assert 'Earth' in figures['chili_fig3_atm_composition'].axes[0].get_title()
    assert 'Venus' in figures['chili_fig5_venus_atm'].axes[0].get_title()

    def run_line(stem, panel=0):
        lines = figures[f'chili_{stem}'].axes[panel].lines
        return [ln for ln in lines if ln.get_label() == 'PROTEUS (d0c7f080)'][0]

    assert run_line('fig6_fO2_vs_T').get_xdata() == pytest.approx([4300.0, 3200.0, 1910.0])
    assert run_line('fig7_volatiles').get_xdata() == pytest.approx([2e-6, 2e-2, 22.6])
    assert run_line('psurf_vs_time').get_xdata() == pytest.approx([1e-6, 1e-2, 11.3])
    assert run_line('fig8_olr', 1).get_xdata()[[0, -1]] == pytest.approx([3300.0, 1800.0])
    panel = figures['chili_fig2_milestones'].axes[2]
    grid_run = [ln for ln in panel.lines if ln.get_markersize() == 9]
    assert [list(ln.get_xdata()) for ln in grid_run] == [[11.3]]
    alphas = {
        _hex(ln.get_color()): ln.get_alpha() for ln in panel.lines if ln.get_markersize() == 6
    }
    assert alphas == {'#bbbbbb': 0.8, '#eb7d0f': 0.45}


def test_composition_bars_stack_from_zero_and_each_star_sits_on_its_bar(tool, tmp_path):
    """Gas segments stack in the order given, a negative partial pressure counts
    as zero, and the star of each model is at the x position of its bar on a
    temperature axis from 1500 to 4000 K in Figure 5."""
    fig, ax = tool.plt.subplots()
    gas = {'H2O': [-1.0, 2.0], 'CO2': [3.0, 4.0]}

    tool._stack_bars(ax, np.arange(2), gas, ['H2O', 'CO2'], [False, True])

    assert [p.get_height() for p in ax.patches] == [0.0, 2.0, 3.0, 4.0]
    assert [p.get_y() for p in ax.patches] == [0.0, 0.0, 0.0, 2.0]
    assert [p.get_linewidth() for p in ax.patches] == [0.5, 2.0, 0.5, 2.0]

    venus = tool.plot_fig5(
        _comparison_data(tmp_path / 'data'), _run_table(), tool._new_style('x')
    )
    bars, twin = venus.axes
    assert [p.get_y() for p in bars.patches] == pytest.approx([0, 0, 70, 300, 73, 302.5])
    stars = [(ln.get_xdata()[0], ln.get_ydata()[0], ln.get_markersize()) for ln in twin.lines]
    assert stars == [(0, 1700.0, 12), (1, 1800.0, 16)]
    assert twin.get_ylim() == (1500, 4000)


def test_every_figure_takes_its_neutral_colours_from_the_active_style(tool, tmp_path, style):
    """Under a style without pure black, white and grey, no figure draws a text,
    a text box, a line, a marker or a patch in such a literal colour; each
    figure has a constrained layout; the reference lines of Figure 8 take the
    text colour of the style."""
    figures = _all_figures(tool, tmp_path)

    assert len(figures) == 11
    for name, fig in figures.items():
        assert not LITERAL_NEUTRALS & _colours(fig), name
        assert type(fig.get_layout_engine()).__name__ == 'ConstrainedLayoutEngine', name
    text_colour = _hex(matplotlib.rcParams['text.color'])
    reference = [
        ln for ln in figures['chili_fig8_olr'].axes[0].lines if ln.get_linestyle() == '--'
    ]
    assert text_colour in {_hex(ln.get_color()) for ln in reference}
    assert text_colour != '#000000'


def test_default_style_keeps_black_and_white_neutrals(tool, tmp_path):
    """Without a style the neutral colours are black, mid grey and white, so
    the reference lines of Figure 8 are black, and the bars have white edges
    for the models and black edges for the current run (3 gases in Figure 5,
    H and C in Figure 4)."""
    figures = _all_figures(tool, tmp_path)

    lines = figures['chili_fig8_olr'].axes[1].lines
    assert '#000000' in {_hex(ln.get_color()) for ln in lines if ln.get_linestyle() == '-.'}
    for stem, of_run in (('fig5_venus_atm', 3), ('fig4_mass_budgets', 2)):
        edges = [_hex(p.get_edgecolor()) for p in figures[f'chili_{stem}'].axes[0].patches]
        assert set(edges) == {'#ffffff', '#000000'}, stem
        assert edges.count('#000000') == of_run, stem


def test_model_lines_take_the_colours_of_the_chili_analysis_code(tool, tmp_path, style):
    """Each model line of the surface pressure figure has its colour of the
    CHILI table under every style, and no model shares the colour of another."""
    data = _comparison_data(tmp_path / 'data', models=MODEL_KEYS)

    fig = tool.plot_psurf(data, _run_table(), tool._new_style('x'))

    drawn = {ln.get_label(): _hex(ln.get_color()) for ln in fig.axes[0].lines}
    assert {k: v for k, v in drawn.items() if k in MODEL_COLOURS} == MODEL_COLOURS
    assert len(set(MODEL_COLOURS.values())) == 7
    assert drawn['PROTEUS (x)'] == '#eb7d0f' != drawn['MOAI']


def test_gas_colours_follow_the_chili_code_for_h2s_and_o2_and_proteus_for_the_rest(
    tool, tmp_path
):
    """The 10 gases have the colours of the page; the bars of Figure 5 are
    drawn in them; H and C of Figure 4 take the element colours, each with its
    letter in the neutral colour of more contrast; a bar below a tenth of the
    axis height has no letter."""
    assert {gas: tool._colour(gas).lower() for gas in tool.GAS_SPECIES} == GAS_COLOURS
    assert (tool._colour('H'), tool._colour('C')) == ('#11B587', '#D20F01')

    fig = tool.plot_fig5(
        _comparison_data(tmp_path / 'data'), _run_table(), tool._new_style('x')
    )

    faces = {_hex(p.get_facecolor()) for p in fig.axes[0].patches}
    assert faces == {GAS_COLOURS['H2O'], GAS_COLOURS['O2'], GAS_COLOURS['SO2']}
    budget = tool.plot_fig4(
        _comparison_data(tmp_path / 'more'), _run_table(), tool._new_style('x')
    )
    assert {_hex(p.get_facecolor()) for p in budget.axes[0].patches} == {'#11b587', '#d20f01'}
    texts = budget.axes[0].texts
    letters = {t.get_text(): _hex(t.get_color()) for t in texts if len(t.get_text()) == 1}
    assert letters == {'H': '#000000', 'C': '#ffffff'}
    small = pd.read_csv(
        tmp_path / 'more' / 'moai' / 'evolution-moai-earth-data.csv', index_col=0
    )
    small['massH_atm(kg)'] = 3e19
    small.to_csv(tmp_path / 'more' / 'moai' / 'evolution-moai-earth-data.csv')
    budget = tool.plot_fig4(tmp_path / 'more', None, tool._new_style('x'))
    assert [t.get_text() for t in budget.axes[0].texts if len(t.get_text()) == 1] == ['C']


def test_submitted_and_current_run_share_the_colour_and_differ_in_line(tool, tmp_path, style):
    """The submitted run is thin and dashed, the current run thick and solid,
    both in the PROTEUS colour; in Figure 2 the current run has edged markers,
    and in Figure 7 an edged circle and a cross in the PROTEUS colour."""
    data, grid = _grid_case(tmp_path)
    run = tool._load_proteus_helpfile(_run_folder(tmp_path / 'earth'))
    ns = tool._new_style('x')

    olr = tool.plot_fig8(data, run, ns).axes[0]
    lines = {ln.get_label(): ln for ln in olr.lines}
    submitted, current = lines['PROTEUS CHILI'], lines['PROTEUS (x)']
    assert _hex(submitted.get_color()) == _hex(current.get_color()) == '#eb7d0f'
    assert (submitted.get_linestyle(), current.get_linestyle()) == ('--', '-')
    assert (submitted.get_linewidth(), current.get_linewidth()) == (1.5, 2.5)

    retained = tool.plot_fig7(data, run, ns).axes[0]
    marks = {ln.get_marker(): ln for ln in retained.lines if ln.get_markersize() == 10}
    text_colour = _hex(matplotlib.rcParams['text.color'])
    assert (_hex(marks['x'].get_markeredgecolor()), marks['x'].get_markeredgewidth()) == (
        '#eb7d0f',
        2.5,
    )
    assert (_hex(marks['o'].get_markeredgecolor()), _hex(marks['o'].get_color())) == (
        text_colour,
        '#eb7d0f',
    )

    panel = tool.plot_fig2(data, run, ns, grid_dir=grid).axes[0]
    grid_lines = [
        ln for ln in panel.lines if ln.get_marker() == 'o' and _hex(ln.get_color()) == '#eb7d0f'
    ]
    styles = sorted(
        (ln.get_linestyle(), ln.get_linewidth(), ln.get_markersize()) for ln in grid_lines
    )
    assert styles == [('-', 3.0, 9.0), ('--', 1.5, 6.0)]
    edged = [ln for ln in grid_lines if ln.get_linestyle() == '-'][0]
    assert _hex(edged.get_markeredgecolor()) == _hex(matplotlib.rcParams['text.color'])


def test_grid_figure_draws_both_runs_in_the_proteus_colour(tool, tmp_path, style):
    """The grid figure draws the submitted grid dashed with open markers and the
    current grid solid with edged markers, both in the PROTEUS colour."""
    data, grid = _grid_case(tmp_path)

    fig = tool.plot_grid_timescales(data, grid, tool._new_style('x'))

    submitted, current = fig.axes[0].lines
    surface = _hex(matplotlib.rcParams['axes.facecolor'])
    assert _hex(submitted.get_color()) == _hex(current.get_color()) == '#eb7d0f'
    assert (submitted.get_linestyle(), _hex(submitted.get_markerfacecolor())) == ('--', surface)
    assert (current.get_linestyle(), _hex(current.get_markerfacecolor())) == ('-', '#eb7d0f')
    assert _hex(current.get_markeredgecolor()) == _hex(matplotlib.rcParams['text.color'])
    assert submitted.get_alpha() == current.get_alpha() == tool.C_ALPHAS['Clow']
    assert (submitted.get_xdata(), submitted.get_ydata()) == (
        pytest.approx([1.6]),
        pytest.approx([0.1]),
    )
    assert (current.get_xdata(), current.get_ydata()) == (
        pytest.approx([1.6]),
        pytest.approx([11.3]),
    )
    assert 'PROTEUS CHILI' in [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def test_bar_of_a_colour_close_to_the_surface_gets_a_grey_edge(tool):
    """SO2 is close to a dark surface (contrast 1.3), so its bars get a mid grey
    edge there and a surface-coloured edge on a light surface; the bar of the
    current run has the text colour in both cases. A colour that fades into
    the surface at a low opacity is drawn at 0.8."""
    assert tool._contrast('#000000', '#ffffff') == pytest.approx(21.0)
    assert tool._contrast('#333333', '#ffffff') == pytest.approx(12.63, abs=0.005)
    assert tool._contrast('#bbbbbb', '#F2F5F7') == pytest.approx(1.75, abs=0.005)
    assert tool._contrast('#00008B', '#05070B') == pytest.approx(1.32, abs=0.005)
    assert tool._contrast('#05070B', '#00008B') == tool._contrast('#00008B', '#05070B')
    assert tool._visible_alpha('#bbbbbb', 0.45) == 0.8
    assert tool._visible_alpha('#bbbbbb', 1.0) == 1.0
    assert tool._visible_alpha('#4477aa', 0.45) == 0.45

    light = [_hex(c) for c in tool._bar_edges('#00008B', [False, True])]
    matplotlib.rcParams.update(DARK_BY_HAND)
    dark = [_hex(c) for c in tool._bar_edges('#00008B', [False, True])]
    water = [_hex(c) for c in tool._bar_edges('#027FB1', [False, True])]

    assert light == ['#ffffff', '#000000']
    assert dark == ['#777a7e', '#e9eef2']
    assert (tool._visible_alpha('#aa3377', 0.45), tool._visible_alpha('#bbbbbb', 0.45)) == (
        0.8,
        0.45,
    )
    assert water == ['#05070b', '#e9eef2']


def test_twin_axis_of_a_composition_panel_shows_its_spine_and_no_grid(tool, tmp_path, style):
    """The temperature axis of Figure 5 keeps its right spine under a style
    that hides right spines, and draws no grid over the bars."""
    fig = tool.plot_fig5(
        _comparison_data(tmp_path / 'data'), _run_table(), tool._new_style('x')
    )

    twin = fig.axes[1]
    fig.canvas.draw()
    assert twin.spines['right'].get_visible()
    assert not any(line.get_visible() for line in twin.get_ygridlines())
    assert twin.get_ylabel() == r'$T_\mathrm{surf}$ [K]'


def test_legends_and_panel_labels_lie_outside_the_axes(tool, tmp_path, style):
    """Figures 1, 2, 6, 8 and 9 carry their legend below the panels, clear of
    every axes box, and Figures 8 and 9 name each panel in its title."""
    figures = _all_figures(tool, tmp_path)

    for stem in (
        'fig1_melt_fraction',
        'fig2_milestones',
        'fig6_fO2_vs_T',
        'fig8_olr',
        'fig9_geodynamics',
    ):
        fig = figures[f'chili_{stem}']
        fig.canvas.draw()
        (legend,) = fig.legends
        box = legend.get_window_extent()
        assert box.y0 >= 0 and box.x0 >= 0 and box.x1 <= fig.bbox.x1, stem
        assert all(box.y1 <= ax.bbox.y0 for ax in fig.axes), stem
    for stem, labels in (
        ('fig8_olr', ['(a)', '(b)']),
        ('fig9_geodynamics', ['(a)', '(b)', '(c)']),
    ):
        axes = figures[f'chili_{stem}'].axes
        assert [ax.get_title('left')[-5:-2] for ax in axes] == labels
        assert not any('mathbf' in t.get_text() for ax in axes for t in ax.texts)


def _main_arguments(tool, tmp_path, monkeypatch, *extra):
    """Arguments for ``main`` on small data, with the figure functions replaced."""
    seen = []
    for name in [n for n in dir(tool) if n.startswith('plot_')]:
        monkeypatch.setattr(tool, name, lambda *args, _name=name, **kwargs: seen.append(_name))
    monkeypatch.setattr(tool, 'ensure_chili_repo', lambda path: path)
    monkeypatch.setattr(tool.sys, 'argv', ['plot', '--output', str(tmp_path / 'out'), *extra])
    return seen


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
    closes it; a function without a figure leaves no file."""
    _main_arguments(tool, tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'plot_psurf', lambda *args: tool.plt.figure())

    tool.main()

    written = sorted(path.name for path in (tmp_path / 'out').iterdir())
    assert written == ['chili_psurf_vs_time.pdf', 'chili_psurf_vs_time.png']
    assert tool.plt.get_fignums() == []


def test_make_figure_writes_the_png_and_labels_it_with_the_commits_of_its_runs(tool, tmp_path):
    """``make_figure`` writes one figure as PNG to the given path, in a folder
    that it makes, under the active style, with the commits of ``run_dirs`` in
    the label, and reads the run folders that the keywords name."""
    repo = tmp_path / 'repo'
    _comparison_data(repo / 'intercomparison' / 'outputs')
    earth = _run_folder(tmp_path / 'earth')
    other = _run_with_log(tmp_path / 'other', '81e1514466a590a9fd97acea99150ea5d978a751')
    out = tmp_path / 'new_folder' / 'figure.png'
    matplotlib.rcParams.update(DARK_BY_HAND)

    fig = tool.make_figure(
        [earth, None, str(other)],
        out,
        fig_id='chili_psurf_vs_time',
        chili_repo=str(repo),
        earth=str(earth),
    )

    assert out.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    assert labels == ['MOAI', 'PROTEUS (d0c7f080/81e15144)']
    assert fig.axes[0].lines[1].get_ydata() == pytest.approx([25.0, 140.0, 310.0])
    assert _hex(fig.get_facecolor()) == '#05070b'
    assert len(fig.legends) == 0 and fig.axes[0].get_legend() is not None


def test_make_figure_refuses_an_unknown_figure_and_a_figure_without_data(
    tool, tmp_path, monkeypatch
):
    """An unknown figure name and a figure with no model and no run data raise
    ``ValueError`` and write no file; the grid figure without ``grid_dir`` says
    so; a wrong name is refused before the comparison data is fetched; a
    figure that cannot be written is closed."""

    def no_clone(*args, **kwargs):
        raise AssertionError('a refused figure must not fetch the comparison data')

    monkeypatch.setattr(tool.subprocess, 'run', no_clone)
    repo = tmp_path / 'repo'
    (repo / 'intercomparison' / 'outputs').mkdir(parents=True)
    out = tmp_path / 'figure.png'

    missing = tmp_path / 'no_checkout'
    with pytest.raises(ValueError, match="unknown figure 'chili_fig10'; known: chili_fig1_"):
        tool.make_figure([], out, fig_id='chili_fig10', chili_repo=missing)
    with pytest.raises(ValueError, match="no data for figure 'chili_fig5_venus_atm'"):
        tool.make_figure([], out, fig_id='chili_fig5_venus_atm', chili_repo=repo)
    with pytest.raises(ValueError, match="figure 'chili_grid_solidification' needs grid_dir"):
        tool.make_figure([], out, fig_id='chili_grid_solidification', chili_repo=missing)
    assert not out.exists() and not missing.exists()
    (tmp_path / 'a_file').write_text('')
    with pytest.raises(OSError):
        tool.make_figure(
            [],
            tmp_path / 'a_file' / 'figure.png',
            fig_id='chili_psurf_vs_time',
            chili_repo=repo,
        )
    assert tool.plt.get_fignums() == []
    fig = tool.make_figure(
        [], out, fig_id='chili_grid_solidification', chili_repo=repo, grid_dir=str(tmp_path)
    )
    assert out.exists() and len(fig.axes) == 1
