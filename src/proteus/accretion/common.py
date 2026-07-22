# Shared data structures for giant-impact accretion
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import define, field

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger('fwl.' + __name__)

# Columns an impact timeline must carry, in the order they are documented.
# Every consequence PROTEUS applies at an impact is derived from these, so a
# timeline missing any of them is rejected rather than partially applied.
TIMELINE_COLUMNS = (
    'time',
    'M_target_before',
    'M_impactor',
    'M_merged_after',
    'v_impact',
    'v_esc',
    'impact_parameter',
    'R_target_before',
    'R_impactor',
    'rho_target',
    'rho_impactor',
    'a_before',
    'a_after',
    'e_after',
    'id_target',
    'id_impactor',
)

# Mass closure of a perfect merger. Tight, because the merged mass is a plain
# sum in the dynamical model, so anything looser would hide a real error.
MASS_CLOSURE_RTOL = 1e-6

# Collision velocity cannot fall below the mutual escape velocity, since it is
# sqrt(v_inf^2 + v_esc^2). The tolerance absorbs round-trip formatting only.
VELOCITY_FLOOR_RTOL = 1e-6


@define(frozen=True)
class ImpactEvent:
    """One giant impact on the planet PROTEUS is following.

    Times are on the PROTEUS time axis, so any offset between the
    dynamical model's zero point and the start of the PROTEUS run has
    already been applied. Everything else is SI.

    Attributes
    ----------
    time: float
        Time of the impact [yr].
    M_target_before: float
        Mass of the target immediately before the impact [kg].
    M_impactor: float
        Mass of the impactor [kg].
    M_merged_after: float
        Mass of the merged body [kg], before any atmospheric loss.
    v_impact: float
        Collision velocity [m s-1].
    v_esc: float
        Mutual escape velocity of the pair [m s-1].
    impact_parameter: float
        Impact parameter, the sine of the impact angle [1]. Zero is a
        head-on collision, one is a grazing collision.
    R_target_before: float
        Radius of the target immediately before the impact [m].
    R_impactor: float
        Radius of the impactor [m].
    rho_target: float
        Bulk density of the target [kg m-3].
    rho_impactor: float
        Bulk density of the impactor [kg m-3].
    a_before: float
        Semi-major axis of the target before the impact [m].
    a_after: float
        Semi-major axis of the merged body [m].
    e_after: float
        Eccentricity of the merged body [1].
    id_target: int
        Identifier of the target body.
    id_impactor: int
        Identifier of the impactor.
    """

    time: float = field()
    M_target_before: float = field()
    M_impactor: float = field()
    M_merged_after: float = field()
    v_impact: float = field()
    v_esc: float = field()
    impact_parameter: float = field()
    R_target_before: float = field()
    R_impactor: float = field()
    rho_target: float = field()
    rho_impactor: float = field()
    a_before: float = field()
    a_after: float = field()
    e_after: float = field()
    id_target: int = field(default=-1)
    id_impactor: int = field(default=-1)

    @property
    def mass_delta(self) -> float:
        """Mass added to the planet by this impact [kg]."""
        return self.M_merged_after - self.M_target_before

    @property
    def semimajoraxis_ratio(self) -> float:
        """Factor by which this impact scales the semi-major axis [1].

        The orbit is applied as a ratio rather than an absolute value
        because the PROTEUS configuration owns the planet's orbit; a
        borrowed impact history moves it proportionally instead of
        replacing it.
        """
        return self.a_after / self.a_before


def _check_event_physics(event: ImpactEvent, index: int) -> None:
    """Raise if an impact record is not physically self-consistent.

    Parameters
    ----------
    event : ImpactEvent
        Record to check.
    index : int
        Position in the timeline, used in error messages.

    Raises
    ------
    ValueError
        If any mass, radius, density, or velocity is non-positive, if the
        merged mass does not close, if the collision velocity is below the
        mutual escape velocity, or if the impact parameter or eccentricity
        falls outside its range.
    """
    where = f'impact {index} at t = {event.time:.4e} yr'

    for name in (
        'M_target_before',
        'M_impactor',
        'M_merged_after',
        'R_target_before',
        'R_impactor',
        'rho_target',
        'rho_impactor',
        'a_before',
        'a_after',
        'v_impact',
        'v_esc',
    ):
        value = getattr(event, name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'{where}: {name} must be finite and > 0, got {value!r}')

    # Perfect merging: the merged body carries the mass of both bodies.
    expected = event.M_target_before + event.M_impactor
    if abs(event.M_merged_after - expected) > MASS_CLOSURE_RTOL * expected:
        raise ValueError(
            f'{where}: merged mass {event.M_merged_after:.6e} kg does not close '
            f'against {event.M_target_before:.6e} + {event.M_impactor:.6e} = '
            f'{expected:.6e} kg'
        )

    # A collision velocity below the mutual escape velocity is unreachable:
    # v_impact = sqrt(v_inf^2 + v_esc^2) >= v_esc for any approach velocity.
    if event.v_impact < event.v_esc * (1.0 - VELOCITY_FLOOR_RTOL):
        raise ValueError(
            f'{where}: collision velocity {event.v_impact:.6e} m/s is below the '
            f'mutual escape velocity {event.v_esc:.6e} m/s'
        )

    if not 0.0 <= event.impact_parameter <= 1.0:
        raise ValueError(
            f'{where}: impact parameter must be in [0, 1], got {event.impact_parameter!r}'
        )

    if not 0.0 <= event.e_after < 1.0:
        raise ValueError(
            f'{where}: post-impact eccentricity must be in [0, 1), got {event.e_after!r}'
        )


def validate_timeline(events: Sequence[ImpactEvent]) -> None:
    """Check a whole timeline for self-consistency.

    Every record must be physically valid on its own, times must increase
    strictly so each impact can be scheduled unambiguously, and the mass
    handed from one impact to the next must be continuous.

    Parameters
    ----------
    events : sequence of ImpactEvent
        Timeline to check, in time order.

    Raises
    ------
    ValueError
        If any record is invalid, if two impacts share a time or run
        backwards, or if the target mass jumps between consecutive impacts.
    """
    previous: ImpactEvent | None = None

    for index, event in enumerate(events):
        _check_event_physics(event, index)

        if previous is not None:
            if event.time <= previous.time:
                raise ValueError(
                    f'impact {index} at t = {event.time:.4e} yr does not follow '
                    f'impact {index - 1} at t = {previous.time:.4e} yr; times must '
                    'increase strictly'
                )

            # The body that emerges from one impact is the target of the
            # next, so a mass discontinuity means the rows describe
            # different planets.
            if abs(event.M_target_before - previous.M_merged_after) > (
                MASS_CLOSURE_RTOL * previous.M_merged_after
            ):
                raise ValueError(
                    f'impact {index}: target mass {event.M_target_before:.6e} kg does '
                    f'not continue from the previous merged mass '
                    f'{previous.M_merged_after:.6e} kg; the rows describe different bodies'
                )

        previous = event


def read_timeline(path: str, time_offset: float = 0.0) -> list[ImpactEvent]:
    """Read an impact timeline from file.

    Accepts comma- or whitespace-separated columns with a header row;
    lines beginning with ``#`` are ignored. Environment variables and
    ``~`` in the path are expanded.

    Parameters
    ----------
    path : str
        Path to the timeline file.
    time_offset : float
        Added to every time in the file [yr], mapping the dynamical
        model's zero point onto the PROTEUS time axis.

    Returns
    -------
    events : list of ImpactEvent
        Timeline in time order.

    Raises
    ------
    FileNotFoundError
        If the timeline file does not exist.
    ValueError
        If required columns are missing, if the file holds no impacts, or
        if the timeline fails validation.
    """
    resolved = os.path.expandvars(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f'Impact timeline file does not exist: {resolved}')

    table = pd.read_csv(resolved, sep=None, engine='python', comment='#')
    table.columns = [str(c).strip() for c in table.columns]

    missing = [c for c in TIMELINE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f'Impact timeline {resolved} is missing required columns: {missing}. '
            f'Expected all of: {list(TIMELINE_COLUMNS)}'
        )

    if len(table) == 0:
        raise ValueError(
            f'Impact timeline {resolved} contains no impacts. Disable the accretion '
            'module instead of supplying an empty timeline.'
        )

    table = table.sort_values('time', kind='stable')

    events = [
        ImpactEvent(
            time=float(row['time']) + time_offset,
            M_target_before=float(row['M_target_before']),
            M_impactor=float(row['M_impactor']),
            M_merged_after=float(row['M_merged_after']),
            v_impact=float(row['v_impact']),
            v_esc=float(row['v_esc']),
            impact_parameter=float(row['impact_parameter']),
            R_target_before=float(row['R_target_before']),
            R_impactor=float(row['R_impactor']),
            rho_target=float(row['rho_target']),
            rho_impactor=float(row['rho_impactor']),
            a_before=float(row['a_before']),
            a_after=float(row['a_after']),
            e_after=float(row['e_after']),
            id_target=int(row['id_target']),
            id_impactor=int(row['id_impactor']),
        )
        for _, row in table.iterrows()
    ]

    validate_timeline(events)

    log.info('Read %d impacts from %s', len(events), resolved)
    return events


def next_event(events: Sequence[ImpactEvent], time: float) -> ImpactEvent | None:
    """Return the first impact strictly after the given time.

    Parameters
    ----------
    events : sequence of ImpactEvent
        Timeline in time order.
    time : float
        Current simulation time [yr].

    Returns
    -------
    event : ImpactEvent or None
        The next scheduled impact, or None once the timeline is exhausted.
    """
    for event in events:
        if event.time > time:
            return event
    return None


def due_events(
    events: Sequence[ImpactEvent], time_previous: float, time_now: float
) -> list[ImpactEvent]:
    """Return the impacts falling in a time interval.

    The interval is half-open, excluding ``time_previous`` and including
    ``time_now``, so an impact is applied exactly once no matter how the
    timestep lands on it.

    Parameters
    ----------
    events : sequence of ImpactEvent
        Timeline in time order.
    time_previous : float
        Simulation time at the start of the step [yr].
    time_now : float
        Simulation time at the end of the step [yr].

    Returns
    -------
    due : list of ImpactEvent
        Impacts to apply for this step, in time order.
    """
    return [e for e in events if time_previous < e.time <= time_now]
