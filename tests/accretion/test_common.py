"""Tests for the impact-timeline data structures and their validation.

This file targets accretion/common.py (ImpactEvent, validate_timeline,
read_timeline, next_event, due_events). The timeline is the interface
between a dynamical model and every consequence PROTEUS applies at an
impact, so the invariants exercised here are mass closure of a perfect
merger, the escape-velocity floor on the collision velocity, boundedness
of the impact geometry, and continuity of the target mass along the
chain.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.accretion.common import (
    TIMELINE_COLUMNS,
    ImpactEvent,
    due_events,
    next_event,
    read_timeline,
    validate_timeline,
)
from proteus.utils.constants import const_G

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
    with pytest.raises(ValueError, match='below the mutual escape velocity'):
        validate_timeline([_event(v_impact=1.15e4, v_esc=1.30e4)])


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
                0.05,
                1,
                4,
            ),
        )
    )

    # Empty: header present, no impacts.
    empty = _write_timeline(tmp_path / 'empty.csv', [])
    with pytest.raises(ValueError, match='contains no impacts'):
        read_timeline(str(empty))

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

    # Pin the analytic escape velocity itself, so a change in the
    # constants or the pair convention shows up here. Hand value:
    # 2 G (M_t + M_i) = 8.8635e14, over R_t + R_i = 9.761e6 m, gives
    # 9.0805e7 m2/s2 and a root of 9.5292e3 m/s.
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
