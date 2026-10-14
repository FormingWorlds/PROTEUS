"""Measure the text geometry of a matplotlib figure at the moment it is saved.

`watch_savefig` wraps `Figure.savefig` for the duration of a block, so the figure of any
plot function is measured without a change to that function. `measure` returns, for one
figure: the texts that the saved file cuts off, the texts outside the figure box (a tight
bounding box keeps them), the tick labels that intersect across axes or crowd on one
axes, and the legends and texts that lie on data (a line, a scatter point, a bar, a
filled area or an image).
"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Real

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# Half a pixel of slack: a text box that ends on the edge is not outside.
TOL_PX = 0.5
LABEL_CHARS = 40
# Share of the smaller label box above which 2 tick labels of one axes count as crowded.
# Stacked decade labels reach 0.18 without a visible collision; colliding labels start at 0.22.
CROWDED = 0.2


def _in_view(axis, loc: float) -> bool:
    """True when matplotlib draws the tick at `loc`.

    As `Axis._update_ticks` does: inside the view limits on the scale of the axis, with a
    slack of 1e-10 of their span, so that a tick at 0.30000000000000004 on limits (0, 0.3)
    counts.
    """
    scale = axis.get_transform()
    low, high = sorted(scale.transform(axis.get_view_interval()))
    slack = 1e-10 * (high - low)
    return low - slack <= scale.transform(loc) <= high + slack


def _drawn_texts(fig: Figure) -> list[tuple[int, str, Text]]:
    """List (axes index or -1, kind, text artist) for every text that the figure draws.

    A tick label counts only when it is switched on and matplotlib draws its tick: the
    ticks outside the view limits and the ticks of a hidden axis (the x axis of a twin
    axes) have artists that are not drawn.
    """
    rows, tick_ids = [], set()
    for i, ax in enumerate(fig.axes):
        for axis in (ax.xaxis, ax.yaxis):
            drawn = ax.get_visible() and axis.get_visible()
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for label in (tick.label1, tick.label2):
                    tick_ids.add(id(label))
                    shown = drawn and label.get_visible() and label.get_text().strip()
                    if shown and _in_view(axis, tick.get_loc()):
                        rows.append((i, 'tick', label))
    for text in fig.findobj(Text):
        if id(text) in tick_ids or not text.get_visible() or not text.get_text().strip():
            continue
        index = fig.axes.index(text.axes) if text.axes in fig.axes else -1
        rows.append((index, 'text', text))
    return rows


def _under(artist, ax, area: Bbox, renderer) -> bool:
    """True when a data artist of `ax` (a line, scatter, bar, filled area or image) lies in `area`."""
    data = ax.transData
    if isinstance(artist, Line2D):
        paths, filled = [artist.get_path()] if artist.get_transform() == data else [], False
    elif isinstance(artist, Patch):
        own = artist.get_data_transform() == data
        return own and artist.get_transform().transform_path(artist.get_path()).intersects_bbox(
            area
        )
    elif isinstance(artist, AxesImage):
        shared = Bbox.intersection(area, artist.get_window_extent(renderer))
        return shared is not None and min(shared.width, shared.height) > TOL_PX
    elif artist.get_offset_transform() == data and len(artist.get_offsets()):
        return bool(area.count_contains(np.asarray(data.transform(artist.get_offsets()))))
    else:
        paths = artist.get_paths() if artist.get_transform() == data else []
        filled = not isinstance(artist, LineCollection)
    shown = [data.transform_path(path) for path in paths if len(path.vertices)]
    return any(path.intersects_bbox(area, filled=filled) for path in shown)


def _data_under(ax, box: Bbox, renderer) -> int:
    """Count the data artists of `ax` that pass under `box`.

    Data artists are the lines, scatters, bars and other patches, filled areas and images
    in data coordinates. For an artist that its axes clips, only the part of `box` inside
    the axes counts, and a box that only touches the axes edge covers nothing. A hidden
    axes has no data.
    """
    inside = Bbox.intersection(box, ax.bbox)
    if inside is not None and min(inside.width, inside.height) <= TOL_PX:
        inside = None
    hits = 0
    artists = [*ax.lines, *ax.patches, *ax.images, *ax.collections] if ax.get_visible() else []
    for artist in artists:
        area = inside if artist.get_clip_on() else box
        if area is not None and artist.get_visible() and _under(artist, ax, area, renderer):
            hits += 1
    return hits


def _saved_area(fig: Figure, renderer, bbox_inches, pad_inches) -> Bbox:
    """Area of the saved file in the pixel coordinates of the figure.

    The pad follows `savefig`: the pads of a constrained layout for `'layout'`, a number
    as given (True is 1 inch), and the pad of the active style otherwise.
    """
    if bbox_inches is None:
        bbox_inches = matplotlib.rcParams['savefig.bbox']
    if bbox_inches != 'tight':
        return fig.bbox
    engine = fig.get_layout_engine()
    if pad_inches == 'layout' and isinstance(engine, ConstrainedLayoutEngine):
        w_pad, h_pad = engine.get()['w_pad'], engine.get()['h_pad']
    elif isinstance(pad_inches, Real):
        w_pad = h_pad = pad_inches
    else:
        w_pad = h_pad = matplotlib.rcParams['savefig.pad_inches']
    tight = fig.get_tightbbox(renderer)
    corners = (tight.x0 - w_pad, tight.y0 - h_pad, tight.x1 + w_pad, tight.y1 + h_pad)
    return Bbox.from_extents(*(c * fig.dpi for c in corners))


def _leaves(box: Bbox, ref: Bbox) -> bool:
    """True when `box` extends past `ref` by more than the tolerance."""
    return (
        box.x0 < ref.x0 - TOL_PX
        or box.y0 < ref.y0 - TOL_PX
        or box.x1 > ref.x1 + TOL_PX
        or box.y1 > ref.y1 + TOL_PX
    )


def _tick_pairs(ticks: list) -> tuple[list, list]:
    """Split intersecting tick labels into pairs across axes and crowded pairs on one axes."""
    across, crowded = [], []
    for n, (i, label, box) in enumerate(ticks):
        for j, other, obox in ticks[n + 1 :]:
            if not box.overlaps(obox):
                continue
            if i != j:
                across.append((i, label, j, other))
                continue
            shared = Bbox.intersection(box, obox)
            smaller = min(box.width * box.height, obox.width * obox.height)
            share = shared.width * shared.height / smaller if smaller else 0.0
            if share > CROWDED:
                crowded.append((i, label, other, round(share, 2)))
    return across, crowded


def _covered(fig: Figure, renderer) -> list[tuple[int, str, str, int]]:
    """List the legends and texts of the axes and of the figure that lie on data of an axes."""
    rows = []
    for i, owner in [(-1, fig), *enumerate(fig.axes)]:
        legends = owner.legends if owner is fig else [owner.get_legend()]
        boxes = [('text', t.get_text(), t) for t in owner.texts if t.get_text().strip()]
        boxes += [('legend', g.get_title().get_text() or 'legend', g) for g in legends if g]
        for kind, label, artist in boxes:
            if not (owner.get_visible() and artist.get_visible()):
                continue
            box = artist.get_window_extent(renderer)
            count = sum(_data_under(ax, box, renderer) for ax in fig.axes)
            if count:
                rows.append((i, kind, label[:LABEL_CHARS], count))
    return rows


def measure(fig: Figure, bbox_inches=None, pad_inches=None) -> dict:
    """Measure the text geometry of a figure as `savefig` writes it.

    Parameters
    ----------
    fig : Figure
        Figure to measure. It is drawn once on an Agg canvas.
    bbox_inches, pad_inches
        The arguments of the same name that the `savefig` call got; None means the
        value of the active style.

    Returns
    -------
    dict
        `texts`: number of drawn texts. `cut_off`: texts outside the saved area, as
        (axes index, kind, text). `outside_figure`: texts outside the figure box, same
        form. `tick_overlaps`: (axes, label, axes, label) for tick labels of different
        axes that intersect. `tick_crowding`: (axes, label, label, share) for 2 tick
        labels of one axes that share more than `CROWDED` of the smaller label box.
        `covered`: (axes index or -1 for the figure, kind, label, number of data artists)
        for a legend or a text that lies on data.
    """
    if not hasattr(fig.canvas, 'get_renderer'):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    saved = _saved_area(fig, renderer, bbox_inches, pad_inches)
    rows = [
        (i, kind, t.get_text()[:LABEL_CHARS], t.get_window_extent(renderer))
        for i, kind, t in _drawn_texts(fig)
    ]
    ticks = [(i, label, box) for i, kind, label, box in rows if kind == 'tick']
    across, crowded = _tick_pairs(ticks)
    return {
        'texts': len(rows),
        'cut_off': [(i, k, s) for i, k, s, b in rows if _leaves(b, saved)],
        'outside_figure': [(i, k, s) for i, k, s, b in rows if _leaves(b, fig.bbox)],
        'tick_overlaps': across,
        'tick_crowding': crowded,
        'covered': _covered(fig, renderer),
    }


@contextmanager
def watch_savefig():
    """Measure every figure that is saved inside the block.

    Yields
    ------
    list
        One (file name, table of `measure`) pair per `savefig` call, in call order.
    """
    tables = []
    original = Figure.savefig

    def measured_savefig(self, fname, *args, **kwargs):
        tables.append(
            (str(fname), measure(self, kwargs.get('bbox_inches'), kwargs.get('pad_inches')))
        )
        return original(self, fname, *args, **kwargs)

    Figure.savefig = measured_savefig
    try:
        yield tables
    finally:
        Figure.savefig = original
