"""Unit tests for ``proteus.grid.plot_grid``.

Covers the ECDF/status plotting helpers without a display:
- the ECDF is a percentage of ALL cases in a subset, so cases with no value (never
  solidified) keep the curve below 100 %
- a log-preset output with zeros is drawn on a 0 + log (symlog) axis, zeros at 0
- the output unit scale is applied once to the plotted values
- status bars are coloured by category and ordered by category then count, with a
  'Not started' bar for grid cases that have no folder
- the colour map's light end is trimmed to 2:1 contrast on white, whichever end it is
- plot format resolution, output-variable validation and the figure files

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')  # headless backend for file-only tests

from proteus.grid.plot_grid import (  # noqa: E402
    CATEGORY_COLOURS,
    MAX_CURVE_LUMINANCE,
    _luminance,
    _plot_cell,
    _set_axis_scale,
    _trim_colourmap,
    _value_colours,
    axis_scale,
    clean_series,
    count_not_started,
    ecdf,
    plot_ecdf_grid,
    plot_status_summary,
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


def test_value_colours_numeric_linear_and_log_norms():
    """Numeric values get a continuous norm; with log_norm, 10 sits halfway between
    1 and 100 (a linear norm would put it at 0.0909).
    """
    cmap = get_colourmap('viridis')
    _, is_numeric, norm = _value_colours([0.5, 1.0, 2.0], log_norm=False, cmap=cmap)
    assert is_numeric
    assert norm(1.0) == pytest.approx(1.0 / 3.0)
    _, _, log_norm = _value_colours([1.0, 10.0, 100.0], log_norm=True, cmap=cmap)
    assert log_norm(10.0) == pytest.approx(0.5)


def test_value_colours_degenerate_and_non_positive_log_ranges():
    """A single tested value still maps to a finite colour, and a log norm over a
    range including 0 falls back to a linear norm rather than failing.
    """
    cmap = get_colourmap('viridis')
    colour_of, _, _ = _value_colours([0.7], log_norm=False, cmap=cmap)
    assert np.all(np.isfinite(colour_of(0.7)))
    _, _, norm = _value_colours([0.0, 1.0], log_norm=True, cmap=cmap)
    assert not isinstance(norm, matplotlib.colors.LogNorm)
    assert norm(0.5) == pytest.approx(0.5)


def test_value_colours_string_uses_discrete_legend_mapping():
    """String values are non-numeric with no norm, and each maps to its own colour."""
    colour_of, is_numeric, norm = _value_colours(
        ['janus', 'agni'], log_norm=False, cmap=get_colourmap('viridis')
    )
    assert not is_numeric
    assert norm is None
    assert colour_of('agni') != colour_of('janus')


@pytest.mark.parametrize(
    'name, light_end', [('viridis', 1.0), ('batlowK_r', 0.0)], ids=['light_top', 'light_bottom']
)
def test_trim_colourmap_makes_the_light_end_visible(name, light_end):
    """The light end is cut until it has 2:1 contrast on white (luminance <= 0.475),
    whichever side it is on; the dark end is untouched. The old fixed 15 % trim left
    batlowK_r at luminance ~0.6, a faint pink.
    """
    cmap = get_colourmap(name)
    trimmed = _trim_colourmap(cmap)
    dark_end = 1.0 - light_end
    assert _luminance(cmap(light_end)) > 0.7  # untrimmed light end is too pale
    assert _luminance(trimmed(light_end)) <= MAX_CURVE_LUMINANCE + 0.01
    assert _luminance(trimmed(light_end)) > MAX_CURVE_LUMINANCE - 0.05  # not over-trimmed
    assert trimmed(dark_end)[:3] == pytest.approx(cmap(dark_end)[:3], abs=0.01)


def test_trim_colourmap_leaves_a_map_with_dark_ends_unchanged():
    """A diverging map (vik) is dark at both ends, so nothing is trimmed."""
    cmap = get_colourmap('vik')
    trimmed = _trim_colourmap(cmap)
    assert trimmed(0.0)[:3] == pytest.approx(cmap(0.0)[:3], abs=0.01)
    assert trimmed(1.0)[:3] == pytest.approx(cmap(1.0)[:3], abs=0.01)


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


def test_plot_status_summary_colours_bars_by_category(tmp_path, monkeypatch):
    """Each bar takes its category colour, so the same status keeps its colour
    across grids; the file is written with the requested extension.
    """
    seen = {}
    real_bar = matplotlib.axes.Axes.bar

    def spy_bar(self, x, height, *args, **kwargs):
        seen['colours'] = kwargs.get('color')
        return real_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'bar', spy_bar)
    out = plot_status_summary(_status_frame(), tmp_path, 'demo', 'pdf', n_not_started=1)
    assert out.name == 'summary_grid_statuses.pdf'
    assert out.exists()
    assert seen['colours'] == [CATEGORY_COLOURS[c] for c in
                               ('Completed', 'Completed', 'Error', 'Not started')]  # fmt: skip


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
