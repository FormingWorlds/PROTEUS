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

import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

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


def test_text_outside_a_plain_save_is_cut_off():
    """Without a tight box the file is the figure box, so a text beyond it is lost."""
    fig, _ = _line_figure()
    fig.text(1.3, 0.5, 'far right')
    fig.text(0.5, 0.5, 'inside')
    table = geom.measure(fig)
    assert table['cut_off'] == [(-1, 'text', 'far right')]
    assert table['outside_figure'] == [(-1, 'text', 'far right')]


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
