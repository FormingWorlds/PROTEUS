"""Unit tests for ``proteus.grid.plot_grid``.

Covers the ECDF/status plotting helpers without a display:
- the ECDF is a percentage of ALL cases in a subset, so cases with no value (never
  solidified) keep the curve below 100 %
- a log-preset output with zeros is drawn on a 0 + log (symlog) axis, zeros at 0
- the output unit scale is applied once to the plotted values
- status bars are horizontal, coloured by category with the cycle colour nearest
  green, blue and red of the active style, and ordered by category then count,
  with a 'Not started' bar for grid cases that have no folder
- each end of the colour map is trimmed to 2:1 contrast against the axes face
  colour of the style, light or dark
- a few swept values take the cycle colours and a legend; more take the colour map
  and a colourbar whose labels are exactly the swept values
- the plots follow the active style (proteus-mpl light or dark via the style key)
  instead of forcing ticks, edges or greys
- plot format resolution, output-variable validation and the figure files

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')  # headless backend for file-only tests

from proteus.grid.plot_grid import (  # noqa: E402
    ECDF_LINEWIDTH,
    MAX_COLOURBAR_LABELS,
    MIN_CURVE_CONTRAST,
    _contrast,
    _hue_distance,
    _plot_cell,
    _set_axis_scale,
    _trim_colourmap,
    _value_colours,
    axis_scale,
    category_colours,
    clean_series,
    count_not_started,
    ecdf,
    ecdf_figsize,
    label_colourbar,
    make_grid_plots,
    plot_ecdf_grid,
    plot_status_summary,
    plot_style,
    resolve_plot_format,
    status_counts,
    validate_output_variables,
)
from proteus.utils.plot import get_colourmap  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# ECDF construction
# ---------------------------------------------------------------------------


def test_ecdf_reaches_hundred_percent_when_every_case_has_a_value():
    """With no missing cases the ECDF rises monotonically from 0 to 100 percent and
    x is sorted (with a leading duplicate for the steps-post start).
    """
    x, y = ecdf(np.array([30.0, 10.0, 20.0]))
    assert np.all(np.diff(y) >= 0)
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(100.0)
    assert x == pytest.approx([10.0, 10.0, 20.0, 30.0])


def test_ecdf_levels_off_at_fraction_of_cases_with_a_value():
    """3 values out of 4 cases plateau at 75 %; normalising by the 3 finite values
    (the old behaviour) would wrongly reach 100 %.
    """
    _, y = ecdf(np.array([10.0, 20.0, 30.0]), n_total=4)
    assert y[-1] == pytest.approx(75.0)
    assert y[1:] == pytest.approx([25.0, 50.0, 75.0])


def test_ecdf_empty_input_returns_empty_arrays():
    """An empty series yields empty x and y rather than raising or dividing by 0."""
    x, y = ecdf(np.array([]), n_total=0)
    assert len(x) == 0
    assert len(y) == 0


def test_clean_series_keeps_zero_and_negative_values():
    """Only NaN and inf are dropped: zeros and negatives are real outcomes (a 0 bar
    surface pressure after escape) and are drawn, on a 0 + log axis if need be.
    """
    kept = clean_series(pd.Series([1.0, 0.0, -2.0, np.inf, np.nan, 10.0]))
    assert sorted(kept.tolist()) == pytest.approx([-2.0, 0.0, 1.0, 10.0])
    assert len(clean_series(pd.Series([np.nan, -np.inf]))) == 0


def test_plot_cell_applies_unit_scale_once_and_normalises_by_subset():
    """Phi_global (fraction) is drawn in percent: 0.2 and 0.4 become 20 and 40, and
    with a third case lacking a value the curve stops at 2/3. A double scale would
    give 2000; no scale would give 0.2; normalising by values would give 100 %.
    """
    df = pd.DataFrame({'p': [1.0, 1.0, 1.0, 2.0], 'Phi_global': [0.2, 0.4, np.nan, 0.9]})
    fig, ax = plt.subplots()
    try:
        assert _plot_cell(ax, df, 'p', 1.0, 'Phi_global', 'k')
        line = ax.lines[0]
        assert line.get_xdata() == pytest.approx([20.0, 20.0, 40.0])
        assert line.get_ydata()[-1] == pytest.approx(200.0 / 3.0)
        assert not _plot_cell(ax, df.assign(Phi_global=np.nan), 'p', 1.0, 'Phi_global', 'k')
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    'values, preset_log, expected',
    [
        ([1.0, 100.0, np.nan], True, ('log', None)),
        ([0.0, 3.0e-3, 2.0e2], True, ('symlog', 1.0e-3)),
        ([60395.0, 60430.0], True, ('linear', None)),
        ([0.0, -5.0], True, ('linear', None)),
        ([1.0, 100.0], False, ('linear', None)),
    ],
    ids=['positive_decades', 'zeros_present', 'narrow_span', 'none_positive', 'linear_preset'],
)
def test_axis_scale_picks_log_zero_plus_log_or_linear(values, preset_log, expected):
    """A log preset is drawn on a log axis over >= 1 decade, on a 0 + log (symlog)
    axis when zeros occur, with the linear part ending at the decade below the
    smallest positive value (3e-3 -> 1e-3), and linear otherwise.
    """
    name, linthresh = axis_scale(pd.Series(values), preset_log)
    assert name == expected[0]
    assert linthresh == pytest.approx(expected[1]) if expected[1] else linthresh is None


def test_zero_plus_log_axis_draws_zero_cases_at_zero():
    """On a 0 + log axis the escaped-atmosphere case (P_surf = 0) is a step at
    x = 0 and the axis starts at 0, so 1 of 3 cases shows as 33 % at 0 bar.
    """
    values = pd.Series([0.0, 50.0, 300.0])
    fig, ax = plt.subplots()
    try:
        assert _plot_cell(ax, pd.DataFrame({'p': [1.0] * 3, 'P_surf': values}), 'p', 1.0,
                          'P_surf', 'k')  # fmt: skip
        _set_axis_scale(ax, *axis_scale(values, preset_log=True), values)
        assert ax.get_xscale() == 'symlog'
        assert -10.0 < ax.get_xlim()[0] < 0.0  # just below 0, so the step at 0 is visible
        x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        assert x[1] == pytest.approx(0.0)
        assert y[1] == pytest.approx(100.0 / 3.0)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


# Swept values of the tutorial grid (input/tutorials/tutorial_grid.toml).
SEMIMAJORAXIS = [0.03, 0.1224744871391589, 0.5]
H_BUDGET = [1000.0, 5000.0, 9000.0]
WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)


def test_value_colours_few_values_take_the_cycle_and_a_legend():
    """Three swept values (fewer than the 10 colours of the default cycle) take the
    first three cycle colours in rising order and no norm (a legend); a value read
    back from the CSV with a changed last digit still gets its colour.
    """
    cycle = [
        matplotlib.colors.to_rgb(c)
        for c in matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    ]
    colour_of, norm = _value_colours(SEMIMAJORAXIS, log_norm=True, cmap=get_colourmap())
    assert norm is None
    assert [colour_of(v) for v in SEMIMAJORAXIS] == cycle[:3]
    assert colour_of(0.12247448713915894) == cycle[1]


def test_value_colours_many_numeric_values_take_a_log_or_linear_norm():
    """Twelve values (more than the cycle) take the colour map through a norm; with
    log_norm, 0.1 sits halfway between 0.01 and 1 (a linear norm puts it at 0.09).
    A string axis with as many values still gets a legend, one colour per value.
    """
    cmap = get_colourmap('viridis')
    many_log = list(np.logspace(-2, 0, 12))
    _, norm = _value_colours(many_log, log_norm=True, cmap=cmap)
    assert isinstance(norm, matplotlib.colors.LogNorm)
    assert norm(0.1) == pytest.approx(0.5)
    _, lin = _value_colours(many_log, log_norm=False, cmap=cmap)
    assert lin(0.1) == pytest.approx((0.1 - 0.01) / 0.99)
    names = [f'module_{i:02d}' for i in range(12)]
    colour_of, str_norm = _value_colours(names, log_norm=False, cmap=cmap)
    assert str_norm is None
    assert len({colour_of(n) for n in names}) == 12


def test_value_colours_single_and_string_values():
    """A single tested value still maps to a finite colour, and two string values
    (janus, agni) each take their own cycle colour, unknown strings a grey.
    """
    colour_of, _ = _value_colours([0.7], log_norm=False, cmap=get_colourmap())
    assert np.all(np.isfinite(colour_of(0.7)))
    colour_of, norm = _value_colours(['janus', 'agni'], log_norm=False, cmap=get_colourmap())
    assert norm is None
    assert colour_of('agni') != colour_of('janus')
    assert colour_of('vulcan') not in (colour_of('agni'), colour_of('janus'))


@pytest.mark.parametrize(
    'name, face, trimmed_end',
    [('viridis', WHITE, 1.0), ('batlowK_r', WHITE, 0.0), ('batlowK_r', BLACK, 1.0)],
    ids=['light_top_on_white', 'light_bottom_on_white', 'dark_top_on_black'],
)
def test_trim_colourmap_cuts_the_end_that_vanishes_on_the_face(name, face, trimmed_end):
    """The end too close to the face colour is cut until it has 2:1 contrast with
    it, and not much further; the other end is untouched. On a dark face it is the
    dark end of batlowK_r that goes, which a trim against white would keep.
    """
    cmap = get_colourmap(name)
    trimmed = _trim_colourmap(cmap, face)
    kept_end = 1.0 - trimmed_end
    assert _contrast(cmap(trimmed_end), face) < MIN_CURVE_CONTRAST  # untrimmed end vanishes
    assert _contrast(trimmed(trimmed_end), face) >= MIN_CURVE_CONTRAST - 0.02
    assert _contrast(trimmed(trimmed_end), face) < MIN_CURVE_CONTRAST + 0.3  # not over-trimmed
    assert trimmed(kept_end)[:3] == pytest.approx(cmap(kept_end)[:3], abs=0.01)


def test_trim_colourmap_leaves_a_map_with_contrasting_ends_unchanged():
    """A diverging map (vik) is dark at both ends, so nothing is cut on white."""
    cmap = get_colourmap('vik')
    trimmed = _trim_colourmap(cmap, WHITE)
    assert trimmed(0.0)[:3] == pytest.approx(cmap(0.0)[:3], abs=0.01)
    assert trimmed(1.0)[:3] == pytest.approx(cmap(1.0)[:3], abs=0.01)


def _colourbar(values, norm):
    """Return a figure and a colourbar over ``values`` labelled by label_colourbar."""
    fig, ax = plt.subplots()
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=get_colourmap())
    cbar = fig.colorbar(mappable, ax=ax)
    label_colourbar(cbar, values)
    fig.canvas.draw()
    return fig, cbar


@pytest.mark.parametrize(
    'values, labels',
    [(SEMIMAJORAXIS, ['0.03', '0.1225', '0.5']), (H_BUDGET, ['1000', '5000', '9000'])],
    ids=['semimajoraxis_logspace', 'h_budget_linspace'],
)
def test_colourbar_labels_every_swept_value_on_a_log_norm(values, labels):
    """On a log colour scale the default ticks label only some values (0.1225 alone
    for orbit.semimajoraxis, 10^3 for H_budget) and add minor ticks at values not in
    the grid; the labelled colourbar shows exactly the swept values and no minor tick.
    """
    norm = matplotlib.colors.LogNorm(vmin=min(values), vmax=max(values))
    fig, cbar = _colourbar(values, norm)
    try:
        assert [t.get_text() for t in cbar.ax.get_yticklabels()] == labels
        assert cbar.ax.get_yticks() == pytest.approx(values)
        assert len(cbar.ax.yaxis.get_minorticklocs()) == 0
    finally:
        plt.close(fig)


def test_colourbar_with_many_values_labels_a_subset_of_them():
    """Twenty swept values keep a tick each, but only every 3rd is labelled (at most
    8 labels), each label is one of the swept values, and the top value 20 takes the
    label of 19 so both ends of the range are labelled.
    """
    values = list(np.linspace(1.0, 20.0, 20))
    fig, cbar = _colourbar(values, matplotlib.colors.Normalize(1.0, 20.0))
    try:
        texts = [t.get_text() for t in cbar.ax.get_yticklabels()]
        assert len(texts) == 20
        shown = [t for t in texts if t]
        assert shown == ['1', '4', '7', '10', '13', '16', '20']
    finally:
        plt.close(fig)


def test_colourbar_labels_the_top_of_a_twelve_value_log_axis():
    """On 12 log-spaced semi-major axes from 0.03 to 0.5 every 2nd tick is labelled
    from the bottom, which used to leave the 0.5 end bare (last label 0.3872); the
    top tick is now labelled and the one just below it is not.
    """
    values = list(np.geomspace(0.03, 0.5, 12))
    norm = matplotlib.colors.LogNorm(vmin=0.03, vmax=0.5)
    fig, cbar = _colourbar(values, norm)
    try:
        texts = [t.get_text() for t in cbar.ax.get_yticklabels()]
        assert [k for k, t in enumerate(texts) if t] == [0, 2, 4, 6, 8, 11]
        assert texts[0] == '0.03'
        assert texts[-1] == '0.5'
    finally:
        plt.close(fig)


class _TickRecorder:
    """Stand-in colourbar that records the ticks and labels label_colourbar sets."""

    def set_ticks(self, ticks, labels):
        self.ticks, self.labels = list(ticks), list(labels)

    def minorticks_off(self):
        pass


@pytest.mark.parametrize('n_values', range(1, 61))
def test_colourbar_labels_both_ends_and_stay_spaced(n_values):
    """For 1 to 60 swept values: every value is a tick, both ends are labelled, at
    most MAX_COLOURBAR_LABELS are labelled, and labels are at least
    ceil(n / MAX_COLOURBAR_LABELS) ticks apart.
    """
    cbar = _TickRecorder()
    label_colourbar(cbar, list(range(n_values)))
    shown = [k for k, label in enumerate(cbar.labels) if label]
    step = math.ceil(n_values / MAX_COLOURBAR_LABELS)
    assert cbar.ticks == [float(v) for v in range(n_values)]
    assert shown[0] == 0 and shown[-1] == n_values - 1
    assert len(shown) <= MAX_COLOURBAR_LABELS
    assert all(b - a >= step for a, b in zip(shown, shown[1:]))


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------


def _status_frame():
    """Five started cases: 2 solidified, 1 target time, 2 atmosphere errors."""
    return pd.DataFrame(
        {
            'status': ['Completed (solidified)', 'Error (Atmosphere model)',
                       'Completed (target time)', 'Completed (solidified)',
                       'Error (Atmosphere model)'],
            'status_category': ['Completed', 'Error', 'Completed', 'Completed', 'Error'],
        }
    )  # fmt: skip


def test_status_counts_orders_by_category_then_count_with_not_started():
    """Completed bars come first (largest first), then Error, then Not started; the
    Not started count is the number passed in.
    """
    counts = status_counts(_status_frame(), n_not_started=3)
    assert list(counts['status']) == [
        'Completed (solidified)',
        'Completed (target time)',
        'Error (Atmosphere model)',
        'Not started',
    ]
    assert list(counts['count']) == [2, 1, 2, 3]
    assert list(counts['category']) == ['Completed', 'Completed', 'Error', 'Not started']


def test_status_counts_without_category_column_is_unknown():
    """A CSV written before the status_category column existed counts as Unknown."""
    counts = status_counts(pd.DataFrame({'status': ['Running', 'Running']}))
    assert list(counts['category']) == ['Unknown']
    assert list(counts['count']) == [2]
    with pytest.raises(ValueError, match='status'):
        status_counts(pd.DataFrame({'x': [1]}))


def test_plot_status_summary_draws_horizontal_bars_in_category_colours(tmp_path, monkeypatch):
    """Bars are horizontal, so long statuses sit one under the other as y labels in
    category order (top to bottom), and take their category colour; the file is
    written with the requested extension.
    """
    seen = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy_savefig(fig, *args, **kwargs):
        ax = fig.axes[0]
        seen['labels'] = [t.get_text() for t in ax.get_yticklabels()]
        seen['colours'] = [p.get_facecolor()[:3] for p in ax.patches]
        seen['inverted'] = ax.yaxis_inverted()
        return real_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', spy_savefig)
    out = plot_status_summary(_status_frame(), tmp_path, 'demo', 'pdf', n_not_started=1)
    assert out.name == 'summary_grid_statuses.pdf'
    assert out.exists()
    assert seen['labels'] == ['Completed (solidified)', 'Completed (target time)',
                              'Error (Atmosphere model)', 'Not started']  # fmt: skip
    assert seen['inverted']  # first category at the top
    palette = category_colours()
    expected = [palette[c] for c in ('Completed', 'Completed', 'Error', 'Not started')]
    assert seen['colours'] == [pytest.approx(c) for c in expected]


def test_category_colours_pick_green_blue_red_from_the_cycle():
    """In the default matplotlib cycle Completed is its green (C2), Running its blue
    (C0) and Error its red (C3), not simply the first three cycle colours (blue,
    orange, green); the greys sit between the text and face colours.
    """
    with matplotlib.rc_context(
        {'axes.prop_cycle': matplotlib.rcParamsDefault['axes.prop_cycle']}
    ):
        cycle = [
            matplotlib.colors.to_rgb(c)
            for c in matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        ]
        colours = category_colours()
    assert colours['Completed'] == cycle[2]
    assert colours['Running'] == cycle[0]
    assert colours['Error'] == cycle[3]
    assert colours['Unknown'] == pytest.approx((0.5, 0.5, 0.5), abs=0.01)
    assert colours['Not started'] == pytest.approx((0.75, 0.75, 0.75), abs=0.01)
    assert _hue_distance(colours['Error'], 0.0) < 10.0


@pytest.mark.parametrize('variant', ['light', 'dark'])
def test_proteus_mpl_style_colours_stand_out_on_its_face(variant):
    """In both proteus-mpl variants the status colours are distinct, green, blue and
    red hues, and every curve colour (cycle and trimmed map) has 2:1 contrast with
    the face, which black curves and edges on the dark face would not have.
    """
    pytest.importorskip('proteus_mpl')
    with plot_style(variant):
        face = matplotlib.colors.to_rgb(matplotlib.rcParams['axes.facecolor'])
        colours = category_colours()
        cmap = _trim_colourmap(get_colourmap('batlowK_r'))
    assert len({colours[c] for c in ('Completed', 'Running', 'Error')}) == 3
    assert _hue_distance(colours['Completed'], 120.0) < 40.0
    assert _hue_distance(colours['Running'], 210.0) < 40.0
    assert _hue_distance(colours['Error'], 0.0) < 40.0
    assert (
        min(_contrast(cmap(t), face) for t in np.linspace(0, 1, 11))
        >= MIN_CURVE_CONTRAST - 0.02
    )
    if variant == 'dark':
        assert _contrast(BLACK, face) < MIN_CURVE_CONTRAST  # why black is not used


def test_plot_style_restores_the_settings_and_needs_proteus_mpl(monkeypatch, caplog):
    """The style applies only inside the block. Without proteus-mpl an explicit
    'dark' raises an ImportError naming the package, while the unset key falls back
    to the active style with an info line instead of failing.
    """
    before = matplotlib.rcParams['axes.facecolor']
    with plot_style('matplotlib'):
        matplotlib.rcParams['axes.facecolor'] = '#123456'
    assert matplotlib.rcParams['axes.facecolor'] == before
    monkeypatch.setitem(__import__('sys').modules, 'proteus_mpl', None)
    with pytest.raises(ImportError, match='pip install proteus-mpl'):
        with plot_style('dark'):
            pass
    with caplog.at_level('INFO'), plot_style(None):
        assert matplotlib.rcParams['axes.facecolor'] == before
    assert any('proteus-mpl is not installed' in r.message for r in caplog.records)
    assert matplotlib.rcParams['axes.facecolor'] == before


def test_unset_style_is_proteus_mpl_light_and_matplotlib_keeps_the_active_style():
    """With proteus-mpl installed the unset key draws in its light variant (the
    Paper face #F2F5F7 and its colour cycle), the same as style = "light"; the
    'matplotlib' value keeps the active style's white face and default cycle.
    """
    pytest.importorskip('proteus_mpl')
    with plot_style(None):
        unset = (matplotlib.rcParams['axes.facecolor'], _cycle())
    with plot_style('light'):
        light = (matplotlib.rcParams['axes.facecolor'], _cycle())
    with matplotlib.rc_context(matplotlib.rcParamsDefault), plot_style('matplotlib'):
        plain = (matplotlib.rcParams['axes.facecolor'], _cycle())
    assert unset == light
    assert matplotlib.colors.to_hex(unset[0]).upper() == '#F2F5F7'
    assert plain[0] == 'white'
    assert plain[1] != light[1]


def _cycle():
    """Return the colour cycle of the active style as hex strings."""
    return [
        matplotlib.colors.to_hex(c)
        for c in matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    ]


def test_count_not_started_uses_product_of_axis_lengths(caplog):
    """A 2 x 3 grid has 6 cases; with folders for cases 0, 1 and 5 (and a stray
    index 7 outside the grid) 3 have not started.
    """
    tested = {'a': [1.0, 2.0], 'b': ['x', 'y', 'z']}
    df = pd.DataFrame({'case_number': [0, 1, 5, 7]})
    with caplog.at_level('WARNING'):
        assert count_not_started(df, tested) == 3
    assert any('not started' in rec.message for rec in caplog.records)
    assert count_not_started(pd.DataFrame({'case_number': range(6)}), tested) == 0


# ---------------------------------------------------------------------------
# Plot options and ECDF figure
# ---------------------------------------------------------------------------


def test_resolve_plot_format_precedence_and_validation():
    """An explicit request beats the base format; case is normalised; the base
    format is next, then 'png'; an unsupported format raises naming it.
    """
    assert resolve_plot_format('png', base_fmt='pdf') == 'png'
    assert resolve_plot_format('PDF') == 'pdf'
    assert resolve_plot_format(None, base_fmt='pdf') == 'pdf'
    assert resolve_plot_format(None) == 'png'
    with pytest.raises(ValueError, match='svg'):
        resolve_plot_format('svg', base_fmt='pdf')


def test_validate_output_variables_drops_missing_without_substitute(caplog):
    """Absent outputs are dropped with a warning (never replaced by Time), and
    present ones are returned in the requested order.
    """
    df = pd.DataFrame({'Time': [1.0], 'P_surf': [2.0], 'T_surf': [3.0]})
    with caplog.at_level('WARNING'):
        present = validate_output_variables(df, ['T_surf', 'not_a_column', 'P_surf'])
    assert present == ['T_surf', 'P_surf']
    assert any('not_a_column' in rec.message for rec in caplog.records)


def test_plot_ecdf_grid_writes_numeric_and_string_rows(tmp_path):
    """One figure covers a numeric (colourbar) and a string (legend) parameter row;
    a call with no tested parameter present in the data raises.
    """
    df = pd.DataFrame(
        {
            'planet.mass_tot': [0.5, 0.5, 1.0, 1.0],
            'atmos_clim.module': ['agni', 'janus', 'agni', 'janus'],
            'P_surf': [1.0, 0.0, 3.0, 4.0],  # a 0 on the log P_surf axis
            'T_surf': [300.0, 400.0, 500.0, 600.0],
        }
    )
    tested = {'planet.mass_tot': [0.5, 1.0], 'atmos_clim.module': ['agni', 'janus']}
    out = plot_ecdf_grid(df, tested, ['P_surf', 'T_surf'], tmp_path, 'demo', 'batlowK_r')
    assert out.name == 'ecdf_grid_plot.png'
    assert out.exists()
    with pytest.raises(ValueError, match='at least one tested parameter'):
        plot_ecdf_grid(df, {'orbit.eccentricity': [0.1]}, ['P_surf'], tmp_path, 'demo')


def test_ecdf_grid_follows_the_style_and_labels_the_swept_values(tmp_path, monkeypatch):
    """With the tutorial orbital distances, the legend reads 0.03, 0.1225 and 0.5
    (not only 0.1225), the curves take the cycle colours, and the panels keep the
    style's ticks: outward and without top ticks under a style that sets them so,
    where the old fixed inward ticks on all four sides would show.
    """
    df = pd.DataFrame({'orbit.semimajoraxis': SEMIMAJORAXIS * 2,
                       'T_surf': [2900.0, 1800.0, 1500.0, 3000.0, 1900.0, 1450.0]})  # fmt: skip
    seen = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy_savefig(fig, *args, **kwargs):
        ax = fig.axes[0]
        seen['legend'] = [t.get_text() for t in ax.get_legend().get_texts()]
        seen['colours'] = [matplotlib.colors.to_rgb(line.get_color()) for line in ax.lines]
        seen['top_ticks'] = ax.xaxis.get_major_ticks()[0].tick2line.get_visible()
        seen['direction'] = ax.xaxis.get_major_ticks()[0]._tickdir
        return real_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', spy_savefig)
    with matplotlib.rc_context({'xtick.direction': 'out', 'xtick.top': False}):
        cycle = [
            matplotlib.colors.to_rgb(c)
            for c in matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        ]
        plot_ecdf_grid(df, {'orbit.semimajoraxis': SEMIMAJORAXIS}, ['T_surf'], tmp_path, 'tut')
    assert seen['legend'] == ['0.03', '0.1225', '0.5']
    assert seen['colours'] == cycle[:3]
    assert seen['top_ticks'] is False
    assert seen['direction'] == 'out'


@pytest.mark.parametrize(
    'font, rows, cols, expected',
    [(10.0, 4, 5, (18.0, 10.4)), (10.0, 1, 1, (3.6, 4.0)), (14.5, 1, 2, (10.44, 5.8))],
    ids=['default_font_four_rows', 'default_font_one_row', 'proteus_mpl_font_one_row'],
)
def test_ecdf_figsize_scales_panels_with_the_font(font, rows, cols, expected):
    """Panels are 3.6 x 2.6 inch at 10 pt and grow with the font (14.5 pt in
    proteus-mpl), and one row is at least 0.4 inch per pt tall so the 48-character
    y label fits; a fixed 2.6 inch row would cut it in the proteus-mpl style.
    """
    with matplotlib.rc_context({'font.size': font}):
        size = ecdf_figsize(rows, cols)
    assert size == pytest.approx(expected)
    assert size[0] / cols == pytest.approx(3.6 * font / 10.0)  # panel width follows the font


def _summary_csvs(tmp_path, completed_rows):
    """Write 'all' and 'completed' summary CSVs; 'all' adds one Error case."""
    completed = pd.DataFrame(completed_rows)
    error = pd.DataFrame(
        [{'case_number': 99, 'status': 'Error (Atmosphere model)', 'status_category': 'Error'}]
    )
    paths = {'all': tmp_path / 'all.csv', 'completed': tmp_path / 'completed.csv'}
    pd.concat([completed, error], ignore_index=True).to_csv(paths['all'], sep='\t', index=False)
    completed.to_csv(paths['completed'], sep='\t', index=False)
    return paths


def _opts(**overrides):
    """Return validated-style [post_processing] options with every plot on."""
    opts = {
        'plot_status': True,
        'plot_ecdf': True,
        'output_variables': ['T_surf', 'not_a_column'],
        'colormap': None,
        'plot_format': None,
    }
    opts.update(overrides)
    return opts


def test_make_grid_plots_writes_both_plots_in_the_base_format(tmp_path, monkeypatch):
    """Both plots land in post_processing/grid_plots in the base run's format when
    plot_format is unset; a missing output is skipped, not fatal. The status plot
    reads the all CSV (with the Error case) and the ECDF the completed one.
    """
    from proteus.grid import plot_grid

    seen = {}
    real_status, real_ecdf = plot_grid.plot_status_summary, plot_grid.plot_ecdf_grid

    def spy_status(df, *args, **kwargs):
        seen['status'] = sorted(df['status_category'])
        return real_status(df, *args, **kwargs)

    def spy_ecdf(df, tested, outputs, *args, **kwargs):
        seen['ecdf'] = (len(df), list(outputs))
        return real_ecdf(df, tested, outputs, *args, **kwargs)

    monkeypatch.setattr(plot_grid, 'plot_status_summary', spy_status)
    monkeypatch.setattr(plot_grid, 'plot_ecdf_grid', spy_ecdf)
    rows = [
        {'case_number': i, 'status': 'Completed (solidified)', 'status_category': 'Completed',
         'planet.mass_tot': m, 'T_surf': t}
        for i, (m, t) in enumerate([(0.5, 300.0), (1.0, 900.0)])
    ]  # fmt: skip
    paths = _summary_csvs(tmp_path, rows)
    make_grid_plots(_opts(), paths, {'planet.mass_tot': [0.5, 1.0]}, tmp_path, 'pdf')
    plots = tmp_path / 'post_processing' / 'grid_plots'
    assert sorted(p.name for p in plots.iterdir()) == [
        'ecdf_grid_plot.pdf',
        'summary_grid_statuses.pdf',
    ]
    assert seen['status'] == ['Completed', 'Completed', 'Error']
    assert seen['ecdf'] == (2, ['T_surf'])


def test_make_grid_plots_honours_switches_and_rejects_no_valid_output(tmp_path):
    """plot_status = false and plot_ecdf = false write nothing; an ECDF with no
    requested output present raises instead of drawing an empty figure.
    """
    rows = [
        {
            'case_number': 0,
            'status': 'Completed (solidified)',
            'status_category': 'Completed',
            'planet.mass_tot': 1.0,
            'T_surf': 300.0,
        }
    ]
    paths = _summary_csvs(tmp_path, rows)
    tested = {'planet.mass_tot': [1.0]}
    make_grid_plots(_opts(plot_status=False, plot_ecdf=False), paths, tested, tmp_path)
    assert not (tmp_path / 'post_processing').exists()
    with pytest.raises(ValueError, match='No valid output_variables'):
        make_grid_plots(
            _opts(plot_status=False, output_variables=['not_a_column']), paths, tested, tmp_path
        )


def test_validate_output_variables_drops_non_numeric_columns(caplog):
    """A text column (status) or a string parameter cannot be drawn as an ECDF; it
    is dropped with a warning instead of raising a TypeError while plotting.
    """
    df = pd.DataFrame({'status': ['Completed'], 'atmos_clim.module': ['agni'], 'T_surf': [3.0]})
    with caplog.at_level('WARNING'):
        present = validate_output_variables(df, ['status', 'T_surf', 'atmos_clim.module'])
    assert present == ['T_surf']
    assert any('not numeric' in rec.message for rec in caplog.records)


def test_plot_ecdf_grid_closes_its_figure_on_error_and_draws_thick_lines(tmp_path, monkeypatch):
    """A failure while drawing closes the figure (no figure left open), and the
    curves and legend entries use ECDF_LINEWIDTH, thicker than the default 1.5 pt.
    """
    from proteus.grid import plot_grid

    df = pd.DataFrame({'atmos_clim.module': ['agni', 'janus'], 'T_surf': [300.0, 400.0]})
    tested = {'atmos_clim.module': ['agni', 'janus']}
    drawn = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy_savefig(fig, *args, **kwargs):
        ax = fig.axes[0]
        drawn['lines'] = [line.get_linewidth() for line in ax.lines]
        drawn['legend'] = [h.get_linewidth() for h in ax.get_legend().get_lines()]
        return real_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', spy_savefig)
    plt.close('all')
    plot_ecdf_grid(df, tested, ['T_surf'], tmp_path, 'demo')
    assert ECDF_LINEWIDTH > 2.0
    assert drawn['lines'] == [ECDF_LINEWIDTH, ECDF_LINEWIDTH]
    assert drawn['legend'] == [ECDF_LINEWIDTH, ECDF_LINEWIDTH]
    assert plt.get_fignums() == []

    def boom(*args, **kwargs):
        raise RuntimeError('drawing failed')

    monkeypatch.setattr(plot_grid, '_draw_ecdf_rows', boom)
    with pytest.raises(RuntimeError, match='drawing failed'):
        plot_ecdf_grid(df, tested, ['T_surf'], tmp_path, 'demo')
    assert plt.get_fignums() == []


def test_log_axis_scale_sets_a_log_x_axis():
    """A 'log' result from axis_scale gives a log x axis without a forced limit."""
    values = pd.Series([1.0, 100.0])
    fig, ax = plt.subplots()
    try:
        ax.plot(values, [50.0, 100.0])
        _set_axis_scale(ax, 'log', None, values)
        assert ax.get_xscale() == 'log'
        assert ax.get_xlim()[0] > 0.0
    finally:
        plt.close(fig)


def test_zero_plus_log_axis_keeps_negative_values_in_view():
    """With a negative value the 0 + log axis is not cut just below 0 (which would
    hide the case at -50); the linear part still ends at the decade below 300.
    """
    values = pd.Series([-50.0, 0.0, 300.0])
    scale, linthresh = axis_scale(values, preset_log=True)
    assert (scale, linthresh) == ('symlog', pytest.approx(100.0))
    fig, ax = plt.subplots()
    try:
        ax.plot(values, [1.0, 2.0, 3.0])
        _set_axis_scale(ax, scale, linthresh, values)
        assert ax.get_xscale() == 'symlog'
        assert ax.get_xlim()[0] <= -50.0
    finally:
        plt.close(fig)


def test_make_grid_plots_skips_ecdf_without_completed_cases(tmp_path, caplog):
    """A grid with no completed case still gets its status plot; the ECDF is skipped
    with a warning instead of drawing an empty figure or raising.
    """
    paths = _summary_csvs(tmp_path, [])
    # The completed CSV keeps its header with no rows, as write_summary_csvs writes it.
    columns = ['case_number', 'status', 'status_category', 'planet.mass_tot', 'T_surf']
    pd.DataFrame(columns=columns).to_csv(paths['completed'], sep='\t', index=False)
    with caplog.at_level('WARNING'):
        make_grid_plots(_opts(), paths, {'planet.mass_tot': [1.0]}, tmp_path, 'png')
    plots = tmp_path / 'post_processing' / 'grid_plots'
    assert sorted(p.name for p in plots.iterdir()) == ['summary_grid_statuses.png']
    assert any('No completed cases' in r.message for r in caplog.records)
