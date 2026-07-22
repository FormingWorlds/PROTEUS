"""Tests for the accretion wrapper and its initialisation contract.

This file targets accretion/wrapper.py (init_accretion). The wrapper is
what the main loop calls once at start-up, so what it must guarantee is
that a run with accretion disabled is untouched, that the configured
backend is the one consulted, and that impacts falling outside the
simulated interval are reported rather than dropped in silence.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from proteus.accretion.common import TIMELINE_COLUMNS
from proteus.accretion.wrapper import init_accretion

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
    """Write a two-impact timeline at 1e5 and 5e5 yr."""
    lines = [','.join(TIMELINE_COLUMNS)]
    lines += [','.join(repr(v) for v in row) for row in _ROWS]
    path.write_text('\n'.join(lines) + '\n')
    return path


def _handler(module=None, timeline_path=None, time_offset=0.0, time_start=0.0):
    """Build the minimal Proteus handler shape init_accretion reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(
                module=module,
                time_offset=time_offset,
                dummy=SimpleNamespace(
                    timeline_path=None if timeline_path is None else str(timeline_path)
                ),
            )
        ),
        hf_row={'Time': time_start},
    )


@pytest.mark.unit
def test_disabled_accretion_returns_no_impacts(tmp_path):
    """A run without accretion gets an empty schedule and reads no files.

    Every existing configuration has accretion off, so this path must stay
    a pure no-op: an empty list, and no attempt to touch a timeline. The
    file check matters because a stray read would make the disabled path
    fail on configs that never mention a timeline at all.
    """
    handler = _handler(module=None, timeline_path=tmp_path / 'never_written.csv')

    events = init_accretion(handler)

    assert events == []
    assert not (tmp_path / 'never_written.csv').exists()

    # The handler is not mutated on the disabled path.
    assert handler.hf_row == {'Time': 0.0}


@pytest.mark.unit
def test_enabled_backend_returns_the_scheduled_impacts(tmp_path):
    """The configured backend supplies the schedule the main loop consults.

    The returned list is what the timestep clamp and the impact handler
    read on every step, so it has to arrive complete and in time order,
    with the physical content of each record preserved.
    """
    handler = _handler(module='dummy', timeline_path=_timeline_file(tmp_path / 't.csv'))

    events = init_accretion(handler)

    assert len(events) == 2
    assert [e.time for e in events] == [1.0e5, 5.0e5]
    assert events[0].mass_delta == pytest.approx(6.4e23)

    # Chain continuity survives the wrapper, so the schedule describes one
    # growing body rather than a set of unrelated impacts.
    assert events[1].M_target_before == pytest.approx(events[0].M_merged_after)


@pytest.mark.unit
def test_impacts_before_the_run_starts_are_reported_and_excluded(tmp_path, caplog):
    """Impacts outside the simulated interval are announced, not swallowed.

    The configuration owns the planet's initial mass and orbit, so an
    impact landing before the run begins cannot be applied without
    contradicting it. Dropping it silently would understate the planet's
    accretion history with no trace in the log, so the count and the
    missing mass are reported and the offset is named as the fix.
    """
    path = _timeline_file(tmp_path / 't.csv')

    # Start the run after the first impact but before the second.
    handler = _handler(module='dummy', timeline_path=path, time_start=2.0e5)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        events = init_accretion(handler)

    assert [e.time for e in events] == [5.0e5]

    warning = '\n'.join(r.getMessage() for r in caplog.records)
    assert '1 impact' in warning
    assert 'time_offset' in warning
    # The mass that will not be accreted is quantified, so the size of the
    # omission is visible rather than merely its existence.
    assert '0.107' in warning  # 6.4e23 kg expressed in Earth masses

    # An impact landing exactly on the start time is already accounted for
    # by the initial condition and is excluded too.
    boundary = _handler(module='dummy', timeline_path=path, time_start=1.0e5)
    assert [e.time for e in init_accretion(boundary)] == [5.0e5]

    # Shifting the timeline forward brings both impacts back into range,
    # which is the documented remedy.
    shifted = _handler(module='dummy', timeline_path=path, time_offset=3.0e5, time_start=2.0e5)
    assert len(init_accretion(shifted)) == 2
