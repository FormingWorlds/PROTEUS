"""Tests for the impact-timeline data structures and their validation.

This file targets accretion/common.py (ImpactEvent, validate_timeline,
read_timeline, next_event, due_events). The timeline is the interface
between a dynamical model and every consequence PROTEUS applies at an
impact, so the invariants exercised here are mass closure of a perfect
merger, the escape-velocity floor on the collision velocity, boundedness
of the impact geometry, and the one-way handover of mass from one impact
to the next.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from proteus.accretion.common import (
    MASS_CLOSURE_RTOL,
    MAX_INTERIMPACT_MASS_LOSS_FRAC,
    TIMELINE_COLUMNS,
    ImpactEvent,
    due_events,
    landing_time,
    next_event,
    read_timeline,
    snap_to_impact,
    validate_timeline,
    write_timeline,
)
from proteus.utils.constants import const_G
from proteus.utils.helper import SUBYEAR_TIME_RESOLUTION, format_subyear_time

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _event(**overrides) -> ImpactEvent:
    """Build a physically self-consistent impact record.

    Roughly a Mars-mass impactor onto a proto-Earth at 1 au: the masses
    close, the collision velocity sits above the mutual escape velocity,
    and the geometry is in range. Individual fields are overridden by
    tests that want one quantity broken at a time.
    """
    base = dict(
        time=1.0e5,
        M_target_before=6.0e24,
        M_impactor=6.4e23,
        M_merged_after=6.64e24,
        v_impact=1.30e4,
        v_esc=1.15e4,
        impact_parameter=0.7,
        R_target_before=6.371e6,
        R_impactor=3.390e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.496e11,
        a_after=1.400e11,
        e_before=0.02,
        e_after=0.05,
        id_target=1,
        id_impactor=4,
    )
    base.update(overrides)
    return ImpactEvent(**base)


def _write_timeline(path, rows, sep=',', header_extra=''):
    """Write rows to a timeline file using the documented column order."""
    lines = [header_extra] if header_extra else []
    lines.append(sep.join(TIMELINE_COLUMNS))
    for row in rows:
        lines.append(sep.join(repr(row[c]) for c in TIMELINE_COLUMNS))
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_event_deltas_report_the_added_mass_and_orbit_change():
    """The derived deltas are what the impact handler applies to the planet.

    Mass is applied additively and the orbit multiplicatively, because the
    configuration owns the planet's initial state and a borrowed history
    moves it rather than replacing it. The mass delta must therefore equal
    the impactor mass exactly, not the merged mass, which is the plausible
    wrong reading and differs here by a factor of ten.
    """
    event = _event()

    assert event.mass_delta == pytest.approx(6.4e23, rel=1e-12)
    # Discrimination: the merged mass is an order of magnitude larger, so
    # a handler that added it instead could not pass this tolerance.
    assert abs(event.M_merged_after - event.mass_delta) > 0.5 * event.M_merged_after

    assert event.semimajoraxis_ratio == pytest.approx(1.400e11 / 1.496e11, rel=1e-12)
    # An inward scattering must shrink the orbit, never grow it.
    assert event.semimajoraxis_ratio < 1.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_merged_mass_must_close_against_the_colliding_pair():
    """Perfect merging conserves mass, so the timeline must too.

    A merged mass that does not equal the sum of the two bodies would let
    the planet gain or lose mass that no process accounts for, breaking
    the run's mass budget. The tolerance is checked from both sides: a
    round-off perturbation is accepted, a per-mille error is not.
    """
    validate_timeline([_event()])

    # Round-off scale perturbation stays inside the closure tolerance.
    validate_timeline([_event(M_merged_after=6.64e24 * (1.0 + 1.0e-9))])

    # A per-mille discrepancy is a real error and must be rejected.
    for factor in (1.0 + 1.0e-3, 1.0 - 1.0e-3):
        with pytest.raises(ValueError, match='does not close'):
            validate_timeline([_event(M_merged_after=6.64e24 * factor)])

    # Dropping the impactor entirely is the classic wrong formula.
    with pytest.raises(ValueError, match='does not close'):
        validate_timeline([_event(M_merged_after=6.0e24)])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_collision_velocity_cannot_fall_below_mutual_escape_velocity():
    """A collision velocity below the escape velocity is kinematically impossible.

    Two bodies falling together from rest already arrive at the mutual
    escape velocity, since v_impact = sqrt(v_inf^2 + v_esc^2). Anything
    slower means the velocities were mismatched or swapped, which would
    feed a nonsensical impact energy to the loss law downstream.
    """
    # Parabolic limit, v_inf = 0: the two velocities coincide and are legal.
    validate_timeline([_event(v_impact=1.15e4, v_esc=1.15e4)])

    # Hyperbolic approach: strictly faster, also legal.
    validate_timeline([_event(v_impact=2.00e4, v_esc=1.15e4)])

    # Swapped fields, the realistic mistake, must be caught.
    with pytest.raises(ValueError, match='below the mutual escape velocity') as excinfo:
        validate_timeline([_event(v_impact=1.15e4, v_esc=1.30e4)])

    # Both velocities appear, so the reader can see which pair was swapped
    # without reopening the timeline file. Matched on the values rather than
    # on their formatting, so reformatting the message does not fail this.
    quoted = [float(n) for n in re.findall(r'[0-9.]+e[+-][0-9]+', str(excinfo.value))]
    assert any(v == pytest.approx(1.15e4, rel=1e-6) for v in quoted)
    assert any(v == pytest.approx(1.30e4, rel=1e-6) for v in quoted)

    # The escape velocity floor absorbs round-trip formatting deviations
    # while rejecting velocities clearly below the mutual escape speed.
    validate_timeline([_event(v_impact=1.15e4 * (1.0 - 1.0e-7), v_esc=1.15e4)])
    with pytest.raises(ValueError, match='below the mutual escape velocity'):
        validate_timeline([_event(v_impact=1.15e4 * (1.0 - 1.0e-3), v_esc=1.15e4)])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_geometry_and_eccentricity_stay_in_range():
    """The impact parameter and post-impact eccentricity are bounded.

    The impact parameter is the sine of the impact angle, so it lives in
    [0, 1]; head-on and grazing are both legal endpoints. Eccentricity
    must stay below unity, because a body on an unbound orbit has left
    the system and cannot be the planet PROTEUS is following.
    """
    for b in (0.0, 0.5, 1.0):
        validate_timeline([_event(impact_parameter=b)])
    for bad_b in (-0.01, 1.01):
        with pytest.raises(ValueError, match='impact parameter'):
            validate_timeline([_event(impact_parameter=bad_b)])

    validate_timeline([_event(e_after=0.0)])
    validate_timeline([_event(e_after=0.999)])
    # e = 1 is the parabolic escape boundary and is already unbound.
    for bad_e in (1.0, 1.5, -0.01):
        with pytest.raises(ValueError, match='eccentricity'):
            validate_timeline([_event(e_after=bad_e)])

    # The pre-impact eccentricity carries the same bound and is checked by name,
    # since the applied orbit change is the difference of the two and an unbound
    # value on either side makes that difference meaningless.
    validate_timeline([_event(e_before=0.0)])
    validate_timeline([_event(e_before=0.999)])
    for bad_e in (1.0, 1.5, -0.01):
        with pytest.raises(ValueError, match='e_before'):
            validate_timeline([_event(e_before=bad_e)])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_masses_radii_and_orbits_must_be_finite_and_positive():
    """Every extensive quantity in a record must be a positive real number.

    A zero radius makes the density and escape velocity diverge, a
    negative semi-major axis is an unbound orbit, and a NaN propagates
    silently into the impact energy. All three must fail at load time
    rather than mid-run.
    """
    for field in ('M_target_before', 'M_impactor', 'R_target_before', 'a_before'):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match='finite and > 0'):
                validate_timeline([_event(**{field: bad})])

    # Densities are used for the impactor-to-target ratio in the loss law.
    with pytest.raises(ValueError, match='finite and > 0'):
        validate_timeline([_event(rho_impactor=0.0)])


@pytest.mark.unit
def test_semimajoraxis_ratio_must_be_finite_and_positive():
    """Semi-major axis ratio must be positive and finite."""
    with pytest.raises(ValueError, match='semimajoraxis_ratio must be finite and > 0'):
        validate_timeline([_event(a_before=1e200, a_after=1e-200)])


@pytest.mark.unit
def test_event_time_must_be_finite():
    """Event time must be finite; negative finite times are permitted."""
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='time must be finite'):
            validate_timeline([_event(time=bad)])

    validate_timeline([_event(time=-10.0)])


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_timeline_must_advance_in_time_and_carry_mass_forward():
    """Consecutive impacts describe one body growing, in order.

    Each impact's target is the body the previous impact produced, so the
    masses must chain. Times must increase strictly, otherwise two impacts
    could land in the same timestep window ambiguously. A timeline whose
    masses do not chain is describing two different planets, which is the
    likely outcome of selecting the wrong survivor.
    """
    first = _event(
        time=1.0e5, M_target_before=6.0e24, M_impactor=6.4e23, M_merged_after=6.64e24
    )
    second = _event(
        time=5.0e5, M_target_before=6.64e24, M_impactor=1.0e23, M_merged_after=6.74e24
    )
    validate_timeline([first, second])
    # Impacts 1e-4 yr apart are a valid history; they land in one step.
    close = _event(
        time=1.0e5 + 1.0e-4, M_target_before=6.64e24, M_impactor=1.0e23, M_merged_after=6.74e24
    )
    validate_timeline([first, close])

    # Time running backwards, and two impacts at the same instant.
    for bad_time in (1.0e5, 5.0e4):
        with pytest.raises(ValueError, match='increase strictly'):
            validate_timeline(
                [
                    first,
                    _event(
                        time=bad_time,
                        M_target_before=6.64e24,
                        M_impactor=1.0e23,
                        M_merged_after=6.74e24,
                    ),
                ]
            )

    # Second impact starts from a mass the first one did not produce.
    with pytest.raises(ValueError, match='different bodies'):
        validate_timeline(
            [
                first,
                _event(
                    time=5.0e5,
                    M_target_before=3.0e24,
                    M_impactor=1.0e23,
                    M_merged_after=3.10e24,
                ),
            ]
        )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_mass_may_only_drop_between_impacts_and_only_by_an_atmosphere():
    """Between two impacts a body can lose atmosphere but cannot accrete.

    The timeline reports the perfect-merger mass, whereas the dynamical
    model hands the next collision a body that has already shed whatever
    atmosphere the impact stripped. That gap is legitimate, so the chain
    check is one-way: a drop within the envelope-mass ceiling passes, a
    gain of any size fails, and a drop too large to be atmosphere fails
    because it means a different planet's rows were spliced in.
    """
    first = _event(
        time=1.0e5, M_target_before=6.0e24, M_impactor=6.4e23, M_merged_after=6.64e24
    )

    def _second(M_target_before, **kwargs):
        """Second impact starting from a stated target mass."""
        return _event(
            time=5.0e5,
            M_target_before=M_target_before,
            M_impactor=1.0e23,
            M_merged_after=M_target_before + 1.0e23,
            **kwargs,
        )

    # A one percent atmosphere, the Morrigan default, stripped entirely.
    validate_timeline([first, _second(6.64e24 * 0.99)])

    # Discrimination: that same one percent is four orders of magnitude
    # above the closure tolerance, so the previous line would fail under a
    # strict-equality chain check.
    assert 0.01 > 1.0e3 * MASS_CLOSURE_RTOL

    # Just inside and just outside the ceiling, from both sides.
    validate_timeline(
        [first, _second(6.64e24 * (1.0 - 0.999 * MAX_INTERIMPACT_MASS_LOSS_FRAC))]
    )
    with pytest.raises(ValueError, match='different bodies'):
        validate_timeline(
            [first, _second(6.64e24 * (1.0 - 1.001 * MAX_INTERIMPACT_MASS_LOSS_FRAC))]
        )

    # Raising the ceiling admits an envelope-rich body.
    validate_timeline([first, _second(6.64e24 * 0.80)], max_mass_loss_frac=0.25)

    # Nothing feeds the body between impacts, so even a per-mille gain is
    # a bookkeeping error, well short of the loss the same size is given.
    with pytest.raises(ValueError, match='cannot gain mass'):
        validate_timeline([first, _second(6.64e24 * 1.001)])

    # Round-off on the handover is still accepted from the upper side.
    validate_timeline([first, _second(6.64e24 * (1.0 + 1.0e-9))])


@pytest.mark.unit
def test_read_timeline_parses_both_delimiters_and_applies_the_offset(tmp_path):
    """Timeline files are read tolerantly and shifted onto the PROTEUS clock.

    A dynamical model measures time from disk dispersal while PROTEUS
    measures it from the start of its own evolution, so the offset is
    applied on load rather than at every use. Rows arriving out of order
    are sorted, and comment lines are ignored, so a hand-written file
    behaves like a generated one.
    """
    rows = [
        dict(
            zip(
                TIMELINE_COLUMNS,
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
                    0.03,
                    0.02,
                    1,
                    7,
                ),
            )
        ),
        dict(
            zip(
                TIMELINE_COLUMNS,
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
                    0.02,
                    0.05,
                    1,
                    4,
                ),
            )
        ),
    ]

    # Written newest-first and with a comment header, both of which the
    # reader must cope with.
    comma = _write_timeline(tmp_path / 'c.csv', rows, sep=',', header_extra='# impacts')
    events = read_timeline(str(comma))

    assert len(events) == 2
    assert events[0].time < events[1].time
    assert events[0].time == pytest.approx(1.0e5)
    assert events[0].id_impactor == 4

    # Whitespace separation gives the identical parse.
    space = _write_timeline(tmp_path / 's.txt', rows, sep=' ')
    assert [e.time for e in read_timeline(str(space))] == [e.time for e in events]

    # The offset shifts every row by the same amount, preserving spacing.
    shifted = read_timeline(str(comma), time_offset=2.0e6)
    assert shifted[0].time == pytest.approx(1.0e5 + 2.0e6)
    assert shifted[1].time - shifted[0].time == pytest.approx(events[1].time - events[0].time)


@pytest.mark.unit
@pytest.mark.parametrize(
    'sep, width, tail',
    [
        (',', 0, ''),
        (', ', 0, ''),
        (' ,', 0, ''),
        (' , ', 0, ''),
        (' ', 0, ''),
        ('   ', 0, ''),
        (' ', 22, ''),
        (' ', 0, '   '),
    ],
)
def test_read_timeline_reads_the_time_back_exactly(tmp_path, sep, width, tail):
    """A time the default python-engine parser reads 1 ulp off is read back exactly,
    with time as the last column, for comma, padded comma, single and multiple space
    separators, right-aligned columns and trailing spaces."""
    t = 14405738.971969359
    row = (t, 6.0e24, 6.4e23, 6.64e24, 1.3e4, 1.15e4, 0.7, 6.371e6, 3.39e6, 5510.0, 3930.0)
    values = dict(zip(TIMELINE_COLUMNS, row + (1.496e11, 1.4e11, 0.02, 0.05, 1, 4)))
    cols = [c for c in TIMELINE_COLUMNS if c != 'time'] + ['time']
    path = tmp_path / 'padded.csv'
    lines = [
        sep.join(f.rjust(width) for f in fields) + tail
        for fields in (cols, [repr(values[c]) for c in cols])
    ]
    path.write_text('\n'.join(lines) + '\n')
    events = read_timeline(str(path))
    assert len(events) == 1
    assert events[0].time == t


@pytest.mark.unit
def test_read_timeline_rejects_unusable_files(tmp_path):
    """A malformed timeline fails at load, not part-way through a run.

    A missing column would silently disable one impact consequence, an
    empty file would make an enabled accretion module a no-op, and a
    missing file usually means an unexpanded path. All three are reported
    with the offending detail so the config can be fixed.
    """
    with pytest.raises(FileNotFoundError, match='does not exist'):
        read_timeline(str(tmp_path / 'absent.csv'))

    full = dict(
        zip(
            TIMELINE_COLUMNS,
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
                0.02,
                0.05,
                1,
                4,
            ),
        )
    )

    # Empty: header present, no impacts returns empty list.
    empty = _write_timeline(tmp_path / 'empty.csv', [])
    assert read_timeline(str(empty)) == []

    # Missing a column the impact handler needs.
    trimmed = tmp_path / 'partial.csv'
    keep = [c for c in TIMELINE_COLUMNS if c != 'v_esc']
    trimmed.write_text(','.join(keep) + '\n' + ','.join(repr(full[c]) for c in keep) + '\n')
    with pytest.raises(ValueError, match='missing required columns'):
        read_timeline(str(trimmed))

    # Physically invalid rows are rejected on load as well as in memory.
    broken = _write_timeline(tmp_path / 'broken.csv', [{**full, 'M_merged_after': 9.9e24}])
    with pytest.raises(ValueError, match='does not close'):
        read_timeline(str(broken))


@pytest.mark.unit
def test_empty_timeline_round_trip(tmp_path):
    """Writing an empty timeline and reading it back returns an empty list."""
    path = tmp_path / 'empty_timeline.tsv'
    write_timeline([], str(path))
    assert path.exists()
    assert read_timeline(str(path)) == []


@pytest.mark.unit
def test_scheduling_helpers_apply_each_impact_exactly_once():
    """The step window is half-open, so no impact is skipped or repeated.

    next_event drives the timestep clamp and must look strictly ahead, or
    the loop would clamp to the impact it has just applied and stall.
    due_events excludes the window's start and includes its end, so an
    impact landing exactly on a step boundary is applied by that step and
    not again by the next one.
    """
    first = _event(
        time=1.0e5, M_target_before=6.0e24, M_impactor=6.4e23, M_merged_after=6.64e24
    )
    second = _event(
        time=5.0e5, M_target_before=6.64e24, M_impactor=1.0e23, M_merged_after=6.74e24
    )
    events = [first, second]

    assert next_event(events, 0.0) is first
    # Strictly ahead: standing exactly on an impact returns the following one.
    assert next_event(events, 1.0e5) is second
    assert next_event(events, 5.0e5) is None

    # A step landing exactly on the impact time applies it.
    assert due_events(events, 0.0, 1.0e5) == [first]
    # The next step must not apply it again.
    assert due_events(events, 1.0e5, 3.0e5) == []
    # A long step sweeps up everything it spans, in order.
    assert due_events(events, 0.0, 1.0e6) == [first, second]
    assert due_events(events, 6.0e5, 1.0e6) == []


@pytest.mark.unit
def test_impacts_closer_than_the_name_resolution_share_one_landing_time():
    """An impact 1e-4 yr after the next one moves the landing to its time; one 2e-3 yr
    or exactly 1e-3 yr later lands on its own step; none left gives an infinite
    landing time."""
    times = (1.0e5, 1.0e5 + 1.0e-4, 1.0e5 + 2.1e-3)
    events = [
        _event(time=t, M_target_before=m, M_impactor=1.0e22, M_merged_after=m + 1.0e22)
        for t, m in zip(times, (6.0e24, 6.01e24, 6.02e24))
    ]
    assert [landing_time(events, t) for t in (0.0, times[1])] == [times[1], times[2]]
    assert landing_time(events, times[2]) == float('inf')
    # A gap of exactly 1e-3 yr is not merged; decimal half-points 1e-3 yr apart
    # get distinct names however the sum rounds.
    pair = [_event(time=t) for t in (0.0, SUBYEAR_TIME_RESOLUTION)]
    assert landing_time(pair, -1.0) == pair[0].time
    for a, b in ((1.0015, 1.0025), (984.7875, 984.7885), (136364572.4025, 136364572.4035)):
        pair = [_event(time=t) for t in (a, b)]
        land = landing_time(pair, 0.0)
        later = landing_time(pair, land)
        assert later == float('inf') or format_subyear_time(later) != format_subyear_time(land)


@pytest.mark.unit
@pytest.mark.parametrize(
    'gaps, landings',
    [((6.0e-4, 6.0e-4), (1.2e-3,)), ((9.0e-4,) * 3, (2.7e-3,)), ((9.0e-4, 1.1e-4), (1.01e-3,))],
    ids=['chain-of-three', 'chain-of-four', 'chain-ending-close'],
)
def test_a_chain_of_close_impacts_lands_in_one_step(gaps, landings):
    """Impacts each less than 1e-3 yr after the one before land together at the last,
    so walking the timeline as the main loop does gives landing rows at least 1e-3 yr
    apart with distinct snapshot names and applies every impact once."""
    times = np.cumsum((1.0e5, *gaps, 5.0e-3))
    events = [
        _event(time=t, M_target_before=6.0e24, M_impactor=1.0e22, M_merged_after=6.01e24)
        for t in times
    ]
    rows, t = [], 0.0
    while (land := landing_time(events, t)) < float('inf'):
        rows.append((land, len(due_events(events, t, land))))
        t = land
    expected = [1.0e5 + d for d in landings] + [times[-1]]
    np.testing.assert_allclose([r[0] for r in rows], expected, rtol=0.0, atol=1e-9)
    assert [r[1] for r in rows] == [len(gaps) + 1, 1]
    names = [format_subyear_time(r[0]) for r in rows]
    assert len(set(names)) == len(names)


@pytest.mark.unit
def test_a_landing_step_a_few_ulp_short_ends_on_the_impact():
    """Rounding can end the step aimed at an impact just below its time; the step
    end is moved onto the impact, so the impact is applied by that step, once,
    and the row time equals the impact time."""
    t = 1.0e8 / 3.0
    event = _event(time=t, M_target_before=6.0e24, M_impactor=6.4e23, M_merged_after=6.64e24)
    short = float(np.nextafter(np.nextafter(t, 0.0), 0.0))
    landed = snap_to_impact(short, t)
    assert landed == t
    assert due_events([event], t - 3.0e3, landed) == [event]
    assert due_events([event], landed, landed + 3.0e3) == []
    # A step that ends well short, past the impact, or with none pending is kept.
    for time, t_impact in ((t - 1.0, t), (t + 1.0e-6, t), (t, float('inf'))):
        assert snap_to_impact(time, t_impact) == time
    # The window is a relative 1e-12 of the step end, and 1e-12 yr below 1 yr.
    assert snap_to_impact(t * (1.0 - 0.9e-12), t) == t
    assert snap_to_impact(0.5, 0.5 + 0.9e-12) == 0.5 + 0.9e-12
    assert snap_to_impact(t * (1.0 - 1.1e-12), t) == t * (1.0 - 1.1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_validator_accepts_the_analytic_two_body_collision():
    """A record built from the two-body relations passes validation.

    Cross-checks the validator against the analytical limit rather than
    against itself: the mutual escape velocity is computed here from
    v_esc = sqrt(2 G (M1 + M2) / (R1 + R2)) and the collision velocity
    from v = sqrt(v_inf^2 + v_esc^2), the same closed forms the dynamical
    model uses. A validator with the velocity comparison inverted, or with
    the escape velocity built from one body instead of the pair, would
    reject this physically legal record.
    """
    M_t, M_i = 6.0e24, 6.4e23
    R_t, R_i = 6.371e6, 3.390e6

    v_esc = np.sqrt(2.0 * const_G * (M_t + M_i) / (R_t + R_i))
    v_inf = 5.0e3
    v_impact = np.sqrt(v_inf**2 + v_esc**2)

    # Pin analytic mutual escape velocity for the two-body collision pair
    # (root of 2 G (M_t + M_i) / (R_t + R_i) = 9.5292e3 m/s).
    assert v_esc == pytest.approx(9.5292e3, rel=1e-4)
    # The single-body escape velocity is 1.1212e4 m/s, 18% higher and far
    # outside the tolerance, so the pair convention is discriminated
    # rather than merely assumed.
    v_esc_single = np.sqrt(2.0 * const_G * M_t / R_t)
    assert v_esc_single == pytest.approx(1.1212e4, rel=1e-3)
    assert abs(v_esc_single - v_esc) > 0.1 * v_esc

    event = _event(
        M_target_before=M_t,
        M_impactor=M_i,
        M_merged_after=M_t + M_i,
        R_target_before=R_t,
        R_impactor=R_i,
        v_esc=v_esc,
        v_impact=v_impact,
    )
    validate_timeline([event])

    assert event.v_impact > event.v_esc
    assert event.mass_delta == pytest.approx(M_i, rel=1e-12)
