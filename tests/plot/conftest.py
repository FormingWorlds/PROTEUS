"""Shared fixtures of the plot tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

# Text and axes face colour of a dark style [0-255], with 3 different channels each.
_TEXT, _FACE = (224, 208, 192), (16, 32, 48)


def _between(fade):
    return tuple((t + (f - t) * fade) / 255 for t, f in zip(_TEXT, _FACE))


# Per style: the rc entries and the neutral colours a plot must show in it.
STYLES = {
    'default_style': dict(
        rc={}, text='black', face='white', dark_grey='#444444', mid_grey='grey'
    ),
    'dark_style': dict(
        rc={'text.color': '#e0d0c0', 'axes.facecolor': '#102030'},
        text='#e0d0c0',
        face='#102030',
        dark_grey=_between(0x44 / 0xFF),
        mid_grey=_between(0x80 / 0xFF),
    ),
}


@pytest.fixture(params=list(STYLES))
def style(request):
    """One entry of ``STYLES`` as a namespace."""
    return SimpleNamespace(**STYLES[request.param])


@pytest.fixture
def draw_real():
    """Call a plot function on real Agg figures and return the figures it made.

    ``draw(func, *args, rc=None, **kwargs)`` runs ``func`` in the default
    matplotlib style with the ``rc`` entries on top, and keeps the figures
    open until the test ends.
    """
    backend = mpl.get_backend()
    plt.switch_backend('agg')
    made = []

    def draw(func, *args, rc=None, **kwargs):
        before = set(plt.get_fignums())
        try:
            with mpl.style.context('default'), mpl.rc_context(rc), patch.object(plt, 'close'):
                func(*args, **kwargs)
        finally:
            new = [plt.figure(n) for n in plt.get_fignums() if n not in before]
            made.extend(new)
        return new

    yield draw
    for fig in made:
        plt.close(fig)
    plt.switch_backend(backend)
