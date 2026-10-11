"""Shared fixtures of the plot tests."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def draw_real(monkeypatch):
    """Call a plot function on real Agg figures and return the figures it made.

    ``draw(func, *args, rc=None, **kwargs)`` runs ``func`` in the default
    matplotlib style with the ``rc`` entries on top, and keeps the figures
    open until the test ends.
    """
    backend, close, subplots = mpl.get_backend(), plt.close, plt.subplots
    plt.switch_backend('agg')
    made = []

    def _subplots(*args, **kwargs):
        out = subplots(*args, **kwargs)
        made.append(out[0])
        return out

    def draw(func, *args, rc=None, **kwargs):
        first = len(made)
        with mpl.style.context('default'), mpl.rc_context(rc or {}):
            monkeypatch.setattr(plt, 'subplots', _subplots)
            monkeypatch.setattr(plt, 'close', lambda *a, **k: None)
            func(*args, **kwargs)
        return made[first:]

    yield draw
    for fig in made:
        close(fig)
    plt.switch_backend(backend)
