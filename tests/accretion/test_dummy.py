"""Tests for the timeline-replay accretion module.

This file targets accretion/dummy.py (get_timeline). The dummy module
replays a timeline written earlier, which is how impact consequences are
driven from a known event sequence, so what it must guarantee is that the
configured path is honoured, that path expansion happens, and that the
configured time offset reaches the loaded events.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from proteus.accretion.common import TIMELINE_COLUMNS
from proteus.accretion.dummy import get_timeline

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_ROWS = (
    (
        1.0e5,
        6.0e24,
        6.4e23,
        6.64e24,
        1.3e4,
        1.15e4,
        0.7,
        6.371e6,
        3.39e6,
        5510.0,
        3930.0,
        1.496e11,
        1.4e11,
        0.05,
        1,
        4,
    ),
    (
        5.0e5,
        6.64e24,
        1.0e23,
        6.74e24,
        1.2e4,
        1.1e4,
        0.3,
        6.4e6,
        2.0e6,
        5510.0,
        3930.0,
        1.4e11,
        1.35e11,
        0.02,
        1,
        7,
    ),
)


def _timeline_file(path):
    """Write a two-impact timeline and return its path."""
    lines = [','.join(TIMELINE_COLUMNS)]
    lines += [','.join(repr(v) for v in row) for row in _ROWS]
    path.write_text('\n'.join(lines) + '\n')
    return path


def _config(timeline_path, time_offset=0.0):
    """Build the minimal config shape get_timeline reads."""
    return SimpleNamespace(
        accretion=SimpleNamespace(
            module='dummy',
            time_offset=time_offset,
            dummy=SimpleNamespace(timeline_path=str(timeline_path)),
        )
    )


@pytest.mark.unit
def test_get_timeline_reads_the_configured_file(tmp_path):
    """The replayed impacts come from the configured path, in time order.

    This is the path the whole impact-consequence chain is tested through,
    so it has to deliver every row, ordered, with the physical content
    intact rather than a truncated or reordered subset.
    """
    config = _config(_timeline_file(tmp_path / 'impacts.csv'))

    events = get_timeline(config)

    assert len(events) == 2
    assert [e.time for e in events] == [1.0e5, 5.0e5]

    # Content survives the round trip: the second impact is the smaller
    # one, so a reader that silently reused the first row would fail here.
    assert events[0].M_impactor == pytest.approx(6.4e23)
    assert events[1].M_impactor == pytest.approx(1.0e23)
    assert events[1].M_target_before == pytest.approx(events[0].M_merged_after)


@pytest.mark.unit
def test_get_timeline_applies_the_configured_offset(tmp_path):
    """The accretion time offset reaches the loaded events.

    The offset maps a dynamical model's zero point onto the PROTEUS clock.
    Reading it from the wrong config level, which is the plausible wiring
    mistake, would silently place every impact at the wrong epoch. A
    non-zero offset here shifts both impacts by exactly that amount while
    leaving their spacing untouched.
    """
    path = _timeline_file(tmp_path / 'impacts.csv')

    baseline = get_timeline(_config(path))
    shifted = get_timeline(_config(path, time_offset=3.0e6))

    assert shifted[0].time == pytest.approx(1.0e5 + 3.0e6)
    assert shifted[1].time == pytest.approx(5.0e5 + 3.0e6)

    # Spacing is preserved, so the offset is a shift and not a rescale.
    assert (shifted[1].time - shifted[0].time) == pytest.approx(
        baseline[1].time - baseline[0].time
    )

    # A negative offset is legal and moves impacts earlier, which is how a
    # run starting after disk dispersal is expressed.
    earlier = get_timeline(_config(path, time_offset=-5.0e4))
    assert earlier[0].time == pytest.approx(5.0e4)


@pytest.mark.unit
def test_get_timeline_expands_and_validates_the_path(tmp_path, monkeypatch):
    """User-supplied paths are expanded, and a bad one fails immediately.

    Config paths routinely carry ``~`` or ``$FWL_DATA``; an unexpanded
    path would fail with a confusing not-found error naming a literal
    tilde. A genuinely missing file must still raise, so the expansion
    does not mask a typo.
    """
    _timeline_file(tmp_path / 'impacts.csv')
    monkeypatch.setenv('TEST_TIMELINE_DIR', str(tmp_path))

    events = get_timeline(_config('$TEST_TIMELINE_DIR/impacts.csv'))
    assert len(events) == 2
    assert events[0].id_impactor == 4

    with pytest.raises(FileNotFoundError, match='does not exist'):
        get_timeline(_config(tmp_path / 'absent.csv'))
