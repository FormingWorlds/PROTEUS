"""Unit tests for tools/_figure_geometry.py, the text geometry check of the docs figures.

Every figure is a bare `Figure()`, which has no renderer until `measure` attaches one,
so the tests also cover a figure that a plot function closed before it saved it.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_script = Path(__file__).resolve().parents[2] / 'tools' / '_figure_geometry.py'
_spec = importlib.util.spec_from_file_location('_figure_geometry', _script)
geom = sys.modules.setdefault('_figure_geometry', importlib.util.module_from_spec(_spec))
if not hasattr(geom, 'measure'):
    _spec.loader.exec_module(geom)


@pytest.fixture(autouse=True)
def default_style():
    """Draw every figure in the default style with the plain (not tight) save box."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        yield


def _line_figure():
    """One axes with a diagonal line, with room for texts around it."""
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.plot([0, 1], [0, 1])
    return fig, ax


def test_clean_figure_gives_an_empty_table():
    """A figure with all texts inside and a legend off the line has no finding."""
    fig, ax = _line_figure()
    ax.set_xlabel('x')
    ax.plot([], [], label='entry')
    ax.legend(loc='upper left')
    table = geom.measure(fig)
    assert table['texts'] > 5
    assert table['cut_off'] == [] and table['outside_figure'] == []
    assert table['tick_overlaps'] == [] and table['tick_crowding'] == []
    assert table['covered'] == []


@pytest.mark.parametrize(
    'x, y',
    [(1.3, 0.5), (-0.6, 0.5), (0.5, 1.3), (0.5, -0.3)],
    ids=['right', 'left', 'top', 'bottom'],
)
def test_text_past_each_edge_of_the_figure_is_found(x, y):
    """A text past the right, left, top or bottom edge is cut off; a text at the centre is not."""
    fig, _ = _line_figure()
    fig.text(x, y, 'past the edge')
    fig.text(0.5, 0.5, 'centre')
    table = geom.measure(fig)
    assert table['cut_off'] == [(-1, 'text', 'past the edge')]
    assert table['outside_figure'] == table['cut_off']


@pytest.mark.parametrize('by_argument', [True, False])
def test_tight_box_keeps_a_text_outside_the_figure(by_argument):
    """A tight box grows to the text: it is outside the figure box but not cut off."""
    fig, _ = _line_figure()
    fig.text(1.3, 0.5, 'far right')
    if by_argument:
        table = geom.measure(fig, bbox_inches='tight')
    else:
        matplotlib.rcParams['savefig.bbox'] = 'tight'
        table = geom.measure(fig)
    assert table['cut_off'] == []
    assert table['outside_figure'] == [(-1, 'text', 'far right')]


def test_tight_box_cuts_a_text_that_is_out_of_the_layout():
    """A text that the layout ignores lies outside the tight box, with or without a pad."""
    fig, _ = _line_figure()
    fig.text(1.05, 0.5, 'not in layout').set_in_layout(False)
    assert geom.measure(fig, bbox_inches='tight')['cut_off'] == [(-1, 'text', 'not in layout')]
    # The text ends near 5 inches; a pad of 3 inches puts the edge of the file past 6
    assert geom.measure(fig, bbox_inches='tight', pad_inches=3.0)['cut_off'] == []


def test_only_drawn_tick_labels_count():
    """Hidden labels and labels of ticks outside the view limits are not texts."""
    fig, ax = _line_figure()
    with_labels = geom.measure(fig)['texts']
    ax.set_xticks([0.0, 0.5, 1.0, 5.0])
    ax.set_xlim(0, 1)
    ax.tick_params(labelleft=False)
    table = geom.measure(fig)
    shown = [t.get_text() for _, kind, t in geom._drawn_texts(fig) if kind == 'tick']
    assert shown == ['0.0', '0.5', '1.0']
    assert table['texts'] == 3 < with_labels
    ax.tick_params(labelbottom=False)
    assert geom.measure(fig)['texts'] == 0


def test_tick_labels_of_two_axes_that_intersect_are_reported():
    """Two axes that share an edge put their end labels on each other."""
    fig = Figure(figsize=(4, 3), dpi=100)
    left = fig.add_axes([0.1, 0.2, 0.4, 0.6])
    right = fig.add_axes([0.5, 0.2, 0.4, 0.6])
    for ax in (left, right):
        ax.set_xticks([0, 1])
        ax.set_xlim(0, 1)
        ax.set_yticks([])
    table = geom.measure(fig)
    assert table['tick_overlaps'] == [(0, '1', 1, '0')]
    assert table['tick_crowding'] == []
    right.set_position([0.6, 0.2, 0.3, 0.6])
    assert geom.measure(fig)['tick_overlaps'] == []


def test_tick_labels_of_one_axes_count_as_crowded_above_the_share():
    """Labels that collide are crowded; labels whose boxes only touch are not."""
    fig, ax = _line_figure()
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([0.50, 0.52], ['1234567', '7654321'])
    crowded = geom.measure(fig)['tick_crowding']
    assert len(crowded) == 1
    axes_index, first, second, share = crowded[0]
    assert (axes_index, first, second) == (0, '1234567', '7654321')
    assert geom.CROWDED < share <= 1.0
    # Shift the second label by nearly a label width: the boxes overlap by a sliver
    width = ax.xaxis.get_major_ticks()[0].label1.get_window_extent().width
    step = 0.95 * width / ax.bbox.width
    ax.set_xticks([0.2, 0.2 + step], ['1234567', '7654321'])
    assert geom.measure(fig)['tick_crowding'] == []


def test_legend_and_label_on_data_are_covered():
    """A legend on the line and a text on a scatter point are reported with their axes."""
    fig, ax = _line_figure()
    ax.plot([], [], label='entry')
    ax.legend(loc='center', title='Series')
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'Series', 1)]
    ax.get_legend().remove()
    ax.scatter([0.8], [0.2])
    ax.text(0.8, 0.2, 'label on the point', ha='center', va='center')
    assert geom.measure(fig)['covered'] == [(0, 'text', 'label on the point', 1)]


def test_reference_line_and_hidden_data_are_not_data():
    """A text on an `axhline` or on an invisible line covers no data."""
    fig, ax = _line_figure()
    ax.lines[0].set_visible(False)
    ax.axhline(0.5)
    ax.text(0.5, 0.5, 'threshold', ha='center', va='center')
    assert geom.measure(fig)['covered'] == []
    ax.lines[0].set_visible(True)
    assert geom.measure(fig)['covered'] == [(0, 'text', 'threshold', 1)]
    # A hidden scatter point under the text adds nothing; a visible one adds 1 artist
    points = ax.scatter([0.5], [0.5], visible=False)
    assert geom.measure(fig)['covered'] == [(0, 'text', 'threshold', 1)]
    points.set_visible(True)
    assert geom.measure(fig)['covered'] == [(0, 'text', 'threshold', 2)]


def test_watch_savefig_measures_each_save_and_restores_the_method():
    """Each save inside the block gives one table; the method is restored on an error too."""
    original = Figure.savefig
    fig, _ = _line_figure()
    fig.text(1.3, 0.5, 'far right')
    with geom.watch_savefig() as tables:
        fig.savefig(io.BytesIO(), format='png')
        fig.savefig(io.BytesIO(), format='png', bbox_inches='tight')
    assert Figure.savefig is original
    assert len(tables) == 2
    assert [len(t['cut_off']) for _, t in tables] == [1, 0]
    with pytest.raises(RuntimeError, match='maker failed'):
        with geom.watch_savefig() as tables:
            raise RuntimeError('maker failed')
    assert Figure.savefig is original and tables == []


def test_data_that_its_axes_clips_covers_nothing():
    """A line that runs past its view limits is not under a legend outside the axes."""
    fig, ax = _line_figure()
    ax.plot([0, 3], [0.5, 0.5])
    ax.set_xlim(0, 1)
    ax.plot([], [], label='entry')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Outside')
    assert geom.measure(fig)['covered'] == []
    ax.get_legend().set_bbox_to_anchor((0.3, 0.5))
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'Outside', 2)]


def test_unclipped_line_under_a_legend_outside_the_axes_is_covered():
    """A line with the clip switched off is drawn under the legend, so it counts."""
    fig, ax = _line_figure()
    (stray,) = ax.plot([0, 3], [0.5, 0.5], clip_on=False)
    ax.set_xlim(0, 1)
    ax.plot([], [], label='entry')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Outside')
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'Outside', 1)]
    stray.set_clip_on(True)
    assert geom.measure(fig)['covered'] == []


def test_legend_that_touches_the_axes_edge_covers_nothing():
    """A legend that starts on the axes edge shares no area with the axes."""
    fig, ax = _line_figure()
    ax.plot([0, 3], [0.5, 0.5])
    ax.set_xlim(0, 1)
    ax.plot([], [], label='entry')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=0, title='Edge')
    assert geom.measure(fig)['covered'] == []
    ax.get_legend().set_bbox_to_anchor((0.9, 0.5))
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'Edge', 1)]


def test_legend_of_one_axes_on_the_data_of_another_is_covered():
    """A legend that reaches into a second axes is reported for the data it covers there."""
    fig = Figure(figsize=(4, 3), dpi=100)
    left = fig.add_axes([0.1, 0.2, 0.3, 0.6])
    right = fig.add_axes([0.6, 0.2, 0.3, 0.6])
    right.plot([0, 1], [0.5, 0.5])
    left.plot([], [], label='entry')
    left.legend(loc='center', bbox_to_anchor=(2.2, 0.5), title='Stray')
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'Stray', 1)]
    left.get_legend().set_visible(False)
    assert geom.measure(fig)['covered'] == []
    left.get_legend().set_visible(True)
    right.set_visible(False)
    assert geom.measure(fig)['covered'] == []
    right.set_visible(True)
    left.set_visible(False)
    assert geom.measure(fig)['covered'] == []


def test_figure_level_legend_and_text_on_data_are_covered():
    """A legend and a text of the figure itself are measured, with the index -1."""
    fig, ax = _line_figure()
    (line,) = ax.plot([0, 1], [1, 0], label='entry')
    fig.legend(handles=[line], loc='center', title='Figure legend')
    fig.text(0.35, 0.35, 'note', ha='center', va='center')
    rows = geom.measure(fig)['covered']
    assert sorted(rows) == [(-1, 'legend', 'Figure legend', 2), (-1, 'text', 'note', 1)]
    fig.legends[0].set_loc('upper right')
    fig.texts[0].set_position((0.05, 0.05))
    assert geom.measure(fig)['covered'] == []


def test_pad_of_the_save_call_reaches_the_measurement():
    """The pad of `savefig` sets the saved area; for the 'layout' pad the style pad is used."""
    fig, _ = _line_figure()
    fig.text(1.05, 0.5, 'not in layout').set_in_layout(False)
    with geom.watch_savefig() as tables:
        fig.savefig(io.BytesIO(), format='png', bbox_inches='tight', pad_inches=3.0)
        fig.savefig(io.BytesIO(), format='png', bbox_inches='tight', pad_inches=0.0)
    assert [len(table['cut_off']) for _, table in tables] == [0, 1]
    # 'layout' is a valid pad of savefig; the measurement must not fail on it
    table = geom.measure(fig, bbox_inches='tight', pad_inches='layout')
    assert table['cut_off'] == [(-1, 'text', 'not in layout')]
    matplotlib.rcParams['savefig.pad_inches'] = 3.0
    assert geom.measure(fig, bbox_inches='tight', pad_inches='layout')['cut_off'] == []
    # A numpy number and a whole number are pads too; True is 1 inch, as `savefig` takes it
    for pad in (np.float32(3.0), np.int64(3), 3):
        assert geom.measure(fig, bbox_inches='tight', pad_inches=pad)['cut_off'] == []
    near, _ = _line_figure()
    near.text(0.9, 0.5, 'x').set_in_layout(False)
    matplotlib.rcParams['savefig.pad_inches'] = 0.0
    assert geom.measure(near, bbox_inches='tight', pad_inches=True)['cut_off'] == []
    assert geom.measure(near, bbox_inches='tight', pad_inches=False)['cut_off'] == [
        (-1, 'text', 'x')
    ]
    matplotlib.rcParams['savefig.pad_inches'] = 1.0
    assert len(geom.measure(near, bbox_inches='tight', pad_inches=False)['cut_off']) == 1


def test_layout_pad_of_a_constrained_layout_sets_the_saved_area():
    """With a constrained layout, the 'layout' pad is the pad of the layout, not of the style."""
    fig = Figure(figsize=(4, 3), dpi=100, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.5, h_pad=0.5)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    # The text ends about 0.4 inch right of the axes, between the style pad and the layout pad
    fig.text(0.9, 0.5, 'edge').set_in_layout(False)
    matplotlib.rcParams['savefig.pad_inches'] = 0.01
    assert geom.measure(fig, bbox_inches='tight', pad_inches='layout')['cut_off'] == []
    assert geom.measure(fig, bbox_inches='tight')['cut_off'] == [(-1, 'text', 'edge')]
    fig.set_layout_engine('none')
    assert len(geom.measure(fig, bbox_inches='tight', pad_inches='layout')['cut_off']) == 1


def test_scatter_point_that_its_axes_clips_covers_nothing():
    """A scatter point beyond the view limits counts under a legend only with the clip off."""
    fig, ax = _line_figure()
    ax.lines[0].remove()
    points = ax.scatter([1.25], [0.5])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([], [], label='entry')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Outside')
    assert geom.measure(fig)['covered'] == []
    points.set_clip_on(False)
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'Outside', 1)]
    # The points of a hidden axes are not drawn, so they are no data
    ax.set_visible(False)
    other = fig.add_axes([0.05, 0.05, 0.1, 0.1])
    other.plot([], [], label='entry')
    other.legend(loc='center', bbox_to_anchor=(9.5, 4.5), title='Over the hidden axes')
    assert geom.measure(fig)['covered'] == []
    ax.set_visible(True)
    assert [row[:3] for row in geom.measure(fig)['covered']] == [
        (0, 'legend', 'Outside'),
        (1, 'legend', 'Over the hidden axes'),
    ]


def test_tick_label_at_the_end_of_the_axis_is_measured():
    """With limits (0, 0.3) the last tick lies at 0.30000000000000004; its label is drawn."""
    fig, ax = _line_figure()
    ax.set_xlim(0, 0.3)
    ax.set_yticks([])
    count = geom.measure(fig)['texts']
    labels = [t.get_text() for _, kind, t in geom._drawn_texts(fig) if kind == 'tick']
    assert max(tick.get_loc() for tick in ax.xaxis.get_major_ticks()) > 0.3
    assert float(labels[-1]) == pytest.approx(0.3) and float(labels[0]) == 0.0
    assert count == len(labels) >= 4
    # A tick clearly past the limit is still no drawn text
    ax.set_xticks([0.0, 0.3 + 1e-6], ['start', 'past the limit'])
    ax.set_xlim(0, 0.3)
    assert geom.measure(fig)['texts'] == 1
    assert [t.get_text() for _, kind, t in geom._drawn_texts(fig) if kind == 'tick'] == [
        'start'
    ]


def test_tick_at_the_end_of_a_log_axis_is_measured():
    """The slack is taken on the scale of the axis: a decade tick at the limit of a log axis counts."""
    fig, ax = _line_figure()
    ax.set_yticks([])
    ax.set_xscale('log')
    ax.set_xticks([1.0, 10.0, 100.0, 1000.0000000002], ['a', 'b', 'c', 'd'])
    ax.minorticks_off()
    ax.set_xlim(1.0, 1000.0000000001)
    assert geom.measure(fig)['texts'] == 4
    ax.set_xlim(1.0, 900.0)
    assert geom.measure(fig)['texts'] == 3
    # 5e-11 is 0.3 decades below the limit 1e-10, though it is close to it as a number
    ax.set_xticks([5e-11, 1e-5, 1.0], ['below', 'b', 'c'])
    ax.set_xlim(1e-10, 1.0)
    assert geom.measure(fig)['texts'] == 2
    assert [t.get_text() for _, kind, t in geom._drawn_texts(fig) if kind == 'tick'] == [
        'b',
        'c',
    ]


@pytest.mark.parametrize(
    'kind, artists',
    [('bar', 1), ('hist', 3), ('fill_between', 1), ('imshow', 1), ('pcolormesh', 1)],
)
def test_legend_on_bars_filled_areas_and_images_is_covered(kind, artists):
    """A legend over a bar, the 3 bars of a histogram, a filled area or an image lies on data."""
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    if kind == 'bar':
        ax.bar([0.5], [1.0], width=0.4)
    elif kind == 'hist':
        ax.hist([0.4, 0.5, 0.5, 0.6], bins=3, range=(0.3, 0.7))
    elif kind == 'fill_between':
        ax.fill_between([0.3, 0.7], [0.0, 0.0], [1.0, 1.0])
    elif kind == 'imshow':
        ax.imshow([[0.0, 1.0], [1.0, 0.0]], extent=(0.3, 0.7, 0.0, 1.0), aspect='auto')
    else:
        ax.pcolormesh([0.3, 0.5, 0.7], [0.0, 0.5, 1.0], [[0.0, 1.0], [1.0, 0.0]])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([], [], label='entry')
    ax.legend(loc='center', title='On the data')
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'On the data', artists)]
    # The same legend beside the data covers nothing
    ax.get_legend().set_bbox_to_anchor((0.05, 0.5))
    assert geom.measure(fig)['covered'] == []


def test_span_across_the_axes_is_no_data():
    """A band of `axvspan` is a reference area in axes coordinates, like a reference line."""
    fig, ax = _line_figure()
    ax.lines[0].remove()
    ax.axvspan(0.3, 0.7)
    ax.plot([], [], label='entry')
    ax.legend(loc='center', title='On the band')
    assert geom.measure(fig)['covered'] == []
    ax.bar([0.5], [1.0], width=0.2)
    assert geom.measure(fig)['covered'] == [(0, 'legend', 'On the band', 1)]


def test_tick_labels_count_as_crowded_above_one_fifth_of_the_smaller_box():
    """2 labels of 1 axes that share 0.30 of the box are crowded; a share of 0.15 is not."""
    box = Bbox.from_extents(0, 0, 10, 10)
    near = [(0, 'a', box), (0, 'b', Bbox.from_extents(7, 0, 17, 10))]
    apart = [(0, 'a', box), (0, 'b', Bbox.from_extents(8.5, 0, 18.5, 10))]
    assert geom._tick_pairs(near) == ([], [(0, 'a', 'b', 0.3)])
    assert geom._tick_pairs(apart) == ([], [])
    assert 0.15 < geom.CROWDED < 0.30
    # On 2 axes any shared area is an overlap, whatever its share
    across = [(0, 'a', box), (1, 'b', Bbox.from_extents(9.5, 0, 19.5, 10))]
    assert geom._tick_pairs(across) == ([(0, 'a', 1, 'b')], [])


def test_drawn_texts_are_the_texts_that_the_renderer_draws(monkeypatch):
    """On a figure with a twin axes, a log axis and a tick at the axis end, the check lists
    every text that the Agg renderer draws and no other."""
    from matplotlib.backends.backend_agg import RendererAgg

    fig = Figure(figsize=(6, 4), dpi=100)
    left = fig.add_axes([0.1, 0.15, 0.35, 0.7])
    right = fig.add_axes([0.6, 0.15, 0.35, 0.7])
    left.plot([0.0, 0.3], [1.0, 2.0], label='series')
    left.set_xlim(0, 0.3)
    left.legend(title='Legend')
    twin = left.twinx()
    twin.plot([0.0, 0.3], [10.0, 20.0])
    twin.xaxis.get_ticklabels()  # fills the label texts of the hidden x axis of the twin
    right.loglog([1.0, 1000.0], [1.0, 1000.0])
    right.set_title('Title')
    fig.text(0.5, 0.95, 'Figure text', ha='center')
    table = geom.measure(fig)

    drawn = []
    draw_text = RendererAgg.draw_text
    monkeypatch.setattr(
        RendererAgg,
        'draw_text',
        lambda self, gc, x, y, s, *a, **k: (
            drawn.append(s),
            draw_text(self, gc, x, y, s, *a, **k),
        )[1],
    )
    fig.canvas.draw()
    listed = sorted(t.get_text() for _, _, t in geom._drawn_texts(fig))
    assert listed == sorted(drawn) and table['texts'] == len(drawn) > 20
    assert not twin.xaxis.get_visible() and any(
        t.get_text() for t in twin.xaxis.get_ticklabels()
    )
