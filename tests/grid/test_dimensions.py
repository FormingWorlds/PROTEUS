"""Unit tests for ``proteus.grid.dimensions``.

Covers the axis builders shared by ``proteus grid`` and ``proteus grid-analyse``:
- each method builds the documented values (log spacing, inclusive arange endpoint)
- an axis that would be empty, or a bound a method cannot use, raises ValueError
  instead of producing a silent one-point or zero-point axis
- an unknown method raises

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.grid.dimensions import (
    DIMENSION_METHODS,
    dimension_arange,
    dimension_direct,
    dimension_linspace,
    dimension_logspace,
    dimension_values,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_linspace_spans_both_endpoints_and_rejects_zero_count():
    """linspace includes start and stop; a zero count raises instead of giving no axis."""
    assert dimension_linspace(0.35, 0.95, 3) == pytest.approx([0.35, 0.65, 0.95])
    with pytest.raises(ValueError, match='at least 1'):
        dimension_linspace(0.0, 1.0, 0)


def test_logspace_is_geometric_not_linear():
    """logspace over [1e-3, 1e-1] has middle value 1e-2; linear spacing would give 0.0505."""
    values = dimension_logspace(1e-3, 1e-1, 3)
    assert values == pytest.approx([1e-3, 1e-2, 1e-1])
    assert abs(values[1] - 0.0505) > 0.04  # linear-spacing bug lands at 0.0505


def test_logspace_rejects_non_positive_bounds():
    """A non-positive logspace bound has no logarithm, so it raises (not NaN values)."""
    with pytest.raises(ValueError, match='positive'):
        dimension_logspace(0.0, 1.0, 3)
    with pytest.raises(ValueError, match='positive'):
        dimension_logspace(-1.0, 1.0, 3)


def test_arange_appends_stop_when_step_does_not_divide_range():
    """A step of 1.5 over [1, 5] gives 1, 2.5, 4, then the appended endpoint 5."""
    assert dimension_arange(1.0, 5.0, 1.5) == pytest.approx([1.0, 2.5, 4.0, 5.0])
    # np.arange(0.1, 0.4, 0.1) already ends at ~0.4 by rounding: no duplicate endpoint.
    values = dimension_arange(0.1, 0.4, 0.1)
    assert len(values) == 4
    assert values == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_arange_single_point_and_negative_step():
    """start == stop gives that one value; a negative step counts down to stop."""
    assert dimension_arange(5.0, 5.0, 1.0) == pytest.approx([5.0])
    assert dimension_arange(5.0, 1.0, -2.0) == pytest.approx([5.0, 3.0, 1.0])


@pytest.mark.parametrize(
    'start, stop, step, expected',
    [
        (1e-9, 5e-9, 1e-9, [1e-9, 2e-9, 3e-9, 4e-9, 5e-9]),
        (1e-10, 4e-10, 1e-10, [1e-10, 2e-10, 3e-10, 4e-10]),
        (1e5, 100002.0, 1.0, [1e5, 100001.0, 100002.0]),
        (1e20, 3e20, 1e20, [1e20, 2e20, 3e20]),
    ],
    ids=['nano_scale', 'sub_nano_scale', 'large_offset', 'huge_scale'],
)
def test_arange_keeps_every_point_at_any_magnitude(start, stop, step, expected):
    """Equality is judged against the step, not an absolute 1e-8: np.isclose's
    default collapsed the 1e-9 axis to [5e-9] and dropped 100002 from the 1e5 axis.
    """
    values = dimension_arange(start, stop, step)
    assert len(values) == len(expected)
    assert values == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert values[0] == start
    assert values[-1] == stop


@pytest.mark.parametrize(
    'start, stop, step',
    [(5.0, 1.0, 1.0), (1.0, 5.0, -1.0), (1.0, 5.0, 0.0)],
    ids=['swapped_bounds', 'step_sign_away_from_stop', 'zero_step'],
)
def test_arange_rejects_step_that_cannot_reach_stop(start, stop, step):
    """A step that cannot reach stop raises, instead of the old silent one-point axis."""
    with pytest.raises(ValueError, match='cannot reach'):
        dimension_arange(start, stop, step)
    # The same table through dimension_values raises the same way.
    with pytest.raises(ValueError):
        dimension_values({'method': 'arange', 'start': start, 'stop': stop, 'step': step})


def test_direct_dedups_sorts_numbers_and_keeps_strings():
    """direct removes duplicates, sorts numbers, and keeps strings in first-seen
    order (a set would order them by the per-process string hash).
    """
    assert dimension_direct([3.0, 1.0, 2.0, 1.0]) == pytest.approx([1.0, 2.0, 3.0])
    assert dimension_direct(['janus', 'agni', 'janus']) == ['janus', 'agni']
    assert dimension_direct(['c', 'a', 'b', 'a']) == ['c', 'a', 'b']
    assert dimension_direct([3.0, 1.0, 3.0], sort=False) == pytest.approx([3.0, 1.0])
    with pytest.raises(ValueError, match='has no values'):
        dimension_direct([])


def test_dimension_values_dispatches_by_method_and_rejects_unknown():
    """dimension_values reads each method's own keys; an unknown or missing method raises."""
    assert set(DIMENSION_METHODS) == {'direct', 'linspace', 'logspace', 'arange'}
    assert dimension_values(
        {'method': 'linspace', 'start': 0.0, 'stop': 1.0, 'count': 3}
    ) == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match='Unknown grid dimension method'):
        dimension_values({'method': 'bogus', 'values': [1.0]})
    with pytest.raises(ValueError):
        dimension_values({'values': [1.0]})  # missing method


@pytest.mark.parametrize(
    'start, stop, step',
    [(1.0, 1.0 + 1e-9, 0.5), (2.0, 2.0 - 1e-9, -0.5)],
    ids=['rising', 'falling'],
)
def test_arange_stop_within_tolerance_of_start_is_one_point(start, stop, step):
    """A stop that differs from start by less than ARANGE_STEP_RTOL * |step| (rounding
    in a written value) gives the single stop value; np.arange plus the appended
    endpoint would give two near-duplicate points.
    """
    values = dimension_arange(start, stop, step)
    assert len(values) == 1
    assert values[0] == pytest.approx(stop, rel=1e-15, abs=0.0)
