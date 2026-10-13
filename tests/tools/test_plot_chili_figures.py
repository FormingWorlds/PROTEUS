"""
Unit tests for the figures of ``tools/plot_chili_comparison.py``, drawn from
small tables in a temporary folder, under a dark style set by hand and under
both proteus-mpl styles.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from ._plot_chili_helpers import (
    GAS_COLOURS,
    LITERAL_NEUTRALS,
    MODEL_COLOURS,
    MODEL_KEYS,
    _all_figures,
    _colours,
    _comparison_data,
    _grid_case,
    _hex,
    _run_folder,
    _run_table,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('close_figures'),
]


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
    zero = [ln for ln in ax_rel.lines if list(ln.get_ydata()) == [0, 0]][0]
    assert (_hex(zero.get_color()), zero.get_linestyle()) == ('#000000', '--')
    assert tool._iw_fischer11(2000.0) == pytest.approx(-7.1498, abs=1e-4)
    assert tool._iw_oneill02(2000.0) == pytest.approx(-7.4078, abs=1e-4)


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


def test_each_file_stem_draws_its_own_figure_from_its_own_run(tool, tmp_path):
    """Each file stem gives the figure of its name: the panel count and an axis
    label identify it; Figures 6 and 7 draw the Venus run, the surface pressure
    figure and Figures 1 to 4, 8 and 9 the Earth run, Figure 5 the Venus run,
    and Figure 2 the grid runs, each grid line at the opacity of its carbon level."""
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
    assert run_line('fig9_geodynamics').get_ydata()[[0, -1]] == pytest.approx([3300.0, 1800.0])
    melt = figures['chili_fig1_melt_fraction'].axes[0].lines[-2:]
    assert [ln.get_label()[-5:] for ln in melt] == ['Earth', 'Venus']
    assert [ln.get_xdata()[-1] for ln in melt] == pytest.approx([11.3, 22.6])
    water = {'fig3_atm_composition': (True, False), 'fig5_venus_atm': (False, True)}
    for stem, of_earth_and_venus in water.items():
        heights = [p.get_height() for ax in figures[f'chili_{stem}'].axes for p in ax.patches]
        assert (300.0 in heights, 450.0 in heights) == of_earth_and_venus, stem
    budget = [p.get_height() for p in figures['chili_fig4_mass_budgets'].axes[0].patches]
    assert budget.count(3.0) == 8 and 6.0 not in budget
    cross = [
        ln for ln in figures['chili_fig2_milestones'].axes[2].lines if ln.get_markersize() == 11
    ]
    assert [list(ln.get_xdata()) for ln in cross] == [[11.3]]
    panel = figures['chili_fig2_milestones'].axes[2]
    grid_run = [ln for ln in panel.lines if ln.get_markersize() == 9]
    assert [list(ln.get_xdata()) for ln in grid_run] == [[11.3]]
    alphas = {
        _hex(ln.get_color()): ln.get_alpha() for ln in panel.lines if ln.get_markersize() == 6
    }
    assert alphas == {'#bbbbbb': 0.45, '#eb7d0f': 0.45}


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
    axis height has no letter, a bar just above it has one."""
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
    path = tmp_path / 'more' / 'moai' / 'evolution-moai-earth-data.csv'
    small = pd.read_csv(path, index_col=0)
    cases = (
        ('massH_atm(kg)', 4e19, ['C']),
        ('massC_atm(kg)', 3e19, ['H']),
        ('massC_atm(kg)', 5e19, ['H', 'C']),
    )
    for column, height, shown in cases:
        small[column], kept = height, small[column].copy()
        small.to_csv(path)
        budget = tool.plot_fig4(tmp_path / 'more', None, tool._new_style('x'))
        assert [t.get_text() for t in budget.axes[0].texts if len(t.get_text()) == 1] == shown
        small[column] = kept


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
        assert [ax.get_title('left') for ax in axes] == labels
        assert not any('mathbf' in t.get_text() for ax in axes for t in ax.texts)


def test_legend_of_figure_2_names_each_model_that_is_drawn_with_its_cross(tool, tmp_path):
    """Figure 2 has an entry for each model with a point in the figure and for
    the current run, each with its grid line and the cross of its nominal
    case; the cross of the current run is larger than that of the submitted
    run; a model without data has no entry."""
    data, grid = _grid_case(tmp_path)
    run = tool._load_proteus_helpfile(_run_folder(tmp_path / 'earth'))

    fig = tool.plot_fig2(data, run, tool._new_style('x'), grid_dir=grid)

    (legend,) = fig.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [*MODEL_COLOURS, 'PROTEUS (x)']
    crosses = [ln for ln in legend.findobj(matplotlib.lines.Line2D) if ln.get_marker() == 'x']
    assert [(ln.get_markersize(), ln.get_markeredgewidth()) for ln in crosses] == [
        (7, 1.5)
    ] * 7 + [(11, 3.0)]
    assert [_hex(ln.get_color()) for ln in crosses] == [*MODEL_COLOURS.values(), '#eb7d0f']
    few = tool.plot_fig2(_comparison_data(tmp_path / 'few'), None, tool._new_style('x'))
    assert [t.get_text() for t in few.legends[0].get_texts()] == ['MOAI']


def test_legend_of_figure_7_shows_the_cross_of_each_proteus_result(tool, tmp_path):
    """The legend entries of the submitted and of the current run in Figure 7
    carry their cross in the PROTEUS colour, at the size drawn in the figure;
    the marker key stays grey."""
    data = _comparison_data(tmp_path / 'data', models=('proteus', 'moai'))

    fig = tool.plot_fig7(data, _run_table(), tool._new_style('x'))

    legend = fig.axes[1].get_legend()
    crosses = [ln for ln in legend.findobj(matplotlib.lines.Line2D) if ln.get_marker() == 'x']
    seen = [
        (_hex(ln.get_color()), ln.get_markersize(), ln.get_markeredgewidth()) for ln in crosses
    ]
    assert seen == [('#eb7d0f', 8, 1.5), ('#eb7d0f', 10, 2.5), ('#444444', 8, 1.5)]
    drawn = [
        ln for ln in fig.axes[0].lines if ln.get_marker() == 'x' and ln.get_color() == '#EB7D0F'
    ]
    assert sorted(ln.get_markersize() for ln in drawn) == [8, 10]


def test_gas_legend_lists_a_gas_only_where_a_bar_of_it_can_be_seen(tool, tmp_path):
    """The legend of a composition figure lists a gas when its largest bar in
    any panel is at least a thousandth of the axis range: O2 at 2e-5 bar is
    not in the legend of Figure 5, and SO2, present only at 5% melt, is in the
    legend of Figure 3."""
    data = _comparison_data(tmp_path / 'data')
    style = tool._new_style('x')

    venus = tool.plot_fig5(data, _run_table(), style)
    earth = tool.plot_fig3(data, _run_table(), style)

    def legend(fig):
        return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    assert legend(venus) == ['H$_2$O', 'SO$_2$']
    assert legend(earth) == ['H$_2$O', 'O$_2$', 'SO$_2$']
    assert min(p.get_height() for p in venus.axes[0].patches[-2:]) == pytest.approx(1e-8)
    assert max(p.get_height() for p in venus.axes[0].patches[-2:]) == pytest.approx(2e-5)


def test_text_is_in_the_font_of_the_style_and_the_run_name_is_bold(tool, tmp_path, style):
    """No text of a figure uses bold mathtext; panel labels and titles are bold
    by their font weight; the tick label of the current run is its name in
    bold in the text colour, the other names keep the tick colour."""
    figures = _all_figures(tool, tmp_path)

    for name, fig in figures.items():
        texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert not [t for t in texts if 'mathbf' in t or '\\bf' in t], name
    for stem in ('fig2_milestones', 'fig3_atm_composition', 'fig8_olr', 'fig9_geodynamics'):
        panel = figures[f'chili_{stem}'].axes[0]
        title = panel.title if panel.get_title() else panel._left_title
        assert title.get_text().startswith('(a)') and title.get_fontweight() == 'bold', stem
    for stem in ('fig4_mass_budgets', 'fig6_fO2_vs_T', 'fig7_volatiles'):
        label = [t for t in figures[f'chili_{stem}'].axes[0].texts if t.get_text()[:3] == '(a)']
        assert [t.get_fontweight() for t in label] == ['bold'], stem
    text_colour = _hex(matplotlib.rcParams['text.color'])
    for stem, panel in (
        ('fig3_atm_composition', 0),
        ('fig4_mass_budgets', 2),
        ('fig5_venus_atm', 0),
    ):
        ticks = figures[f'chili_{stem}'].axes[panel].get_xticklabels()
        run = [t for t in ticks if t.get_text() == 'PROTEUS (d0c7f080)']
        assert [(t.get_fontweight(), _hex(t.get_color())) for t in run] == [
            ('bold', text_colour)
        ]
        others = {(t.get_fontweight(), _hex(t.get_color())) for t in ticks if t not in run}
        assert others == {('normal', _hex(matplotlib.rcParams['xtick.color']))}, stem


def test_reference_labels_of_figure_9_have_no_box_and_the_water_label_is_below_its_line(
    tool, tmp_path
):
    """The 3 reference labels of Figure 9 have no filled box that can cover a
    curve, and the water label hangs below the water line, where no model
    curve runs."""
    fig = tool.plot_fig9(
        _comparison_data(tmp_path / 'data'), _run_table(), tool._new_style('x')
    )

    labels = {t.get_text(): t for ax in fig.axes for t in ax.texts}
    assert set(labels) == {
        'Core-mantle boundary (PROTEUS)',
        'Solid mantle viscosity',
        'Water STP viscosity',
    }
    assert [t.get_bbox_patch() for t in labels.values()] == [None, None, None]
    water = labels['Water STP viscosity']
    assert water.get_va() == 'top' and water.get_position()[1] < 1e-3
    assert labels['Solid mantle viscosity'].get_va() == 'bottom'


def test_carbon_levels_of_figure_2_rise_in_opacity(tool, tmp_path):
    """The grid lines of the low, middle and high carbon level have the
    opacities 0.45, 0.7 and 1.0, for a model and for the current grid."""
    data, grid = _grid_case(tmp_path)
    table = pd.read_csv(data / 'moai' / 'evolution-moai-earth-data.csv', index_col=0)
    for level in ('Clow', 'Cmid', 'Chigh'):
        table.to_csv(data / 'moai' / f'evolution-moai-earth-grid-Hlow-{level}-data.csv')
        if level != 'Clow':
            _run_folder(grid / f'chili_grid_earth_Hlow_{level}')

    panel = tool.plot_fig2(data, None, tool._new_style('x'), grid_dir=grid).axes[2]

    for colour, size in (('#aa3377', 6), ('#EB7D0F', 9)):
        lines = [
            ln for ln in panel.lines if ln.get_color() == colour and ln.get_markersize() == size
        ]
        assert [ln.get_alpha() for ln in lines] == [0.45, 0.7, 1.0], colour
    assert len([ln for ln in panel.lines if ln.get_marker() == 'o']) == 8
