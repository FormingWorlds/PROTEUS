# Generic accretion wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.constants import M_earth

if TYPE_CHECKING:
    from proteus.accretion.common import ImpactEvent
    from proteus.proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def init_accretion(handler: Proteus) -> list[ImpactEvent]:
    """Prepare the impact timeline for a run.

    Builds the list of giant impacts the planet will experience, either by
    running a dynamical model or by replaying a timeline written earlier.
    The list is fixed at initialisation and consulted on every step, in
    the same way the stellar evolution track is.

    Parameters
    ----------
    handler : Proteus
        Proteus object instance.

    Returns
    -------
    events : list of ImpactEvent
        Impacts to apply during the run, in time order. Empty when no
        accretion module is selected.
    """
    config = handler.config
    module = config.accretion.module

    if module is None:
        return []

    log.info('Preparing accretion model')
    log.info('')

    match module:
        case 'dummy':
            from proteus.accretion.dummy import get_timeline
        case 'morrigan':
            from proteus.accretion.morrigan import get_timeline
        case _:
            raise ValueError(f"Invalid accretion module: '{module}'")

    events = get_timeline(config)

    return _drop_events_before_start(events, handler.hf_row.get('Time', 0.0))


def apply_impact(handler: Proteus, event: ImpactEvent) -> None:
    """Apply one giant impact's consequences to the running planet.

    Called once for each impact, at the end of the timestep that lands on
    its time, so the orbit and structure of that step already use the grown
    planet and the next interior solve evolves it from there.

    The impactor mass is added to the planet's total mass and the interior
    structure is re-solved, so the radius, gravity and the core/mantle split
    follow the new mass at the configured core fraction. The orbit change is
    applied as a discrete jump to both the configuration, which pins the
    orbit when tides are off, and the running row, which the tidal evolution
    carries forward when tides are on, so the jump persists under either.

    Parameters
    ----------
    handler : Proteus
        Proteus object instance, mutated in place.
    event : ImpactEvent
        The impact to apply.
    """
    from proteus.interior_energetics.wrapper import solve_structure

    config = handler.config
    hf_row = handler.hf_row

    log.info(
        'Giant impact at t = %.4e yr: target %d struck by %d, adding %.4f M_earth',
        event.time,
        event.id_target,
        event.id_impactor,
        event.mass_delta / M_earth,
    )

    # Grow the planet by the impactor mass and re-solve the structure. mass_tot
    # is in Earth masses; the event's mass delta is the impactor mass in kg.
    config.planet.mass_tot += event.mass_delta / M_earth
    solve_structure(
        handler.directories, config, handler.hf_all, hf_row, handler.directories['output']
    )

    # Move the orbit by the impact's proportional change in semi-major axis and
    # its post-impact eccentricity, writing both the configuration and the row.
    ratio = event.semimajoraxis_ratio
    config.orbit.semimajoraxis *= ratio
    config.orbit.eccentricity = event.e_after
    hf_row['semimajorax'] *= ratio
    hf_row['eccentricity'] = event.e_after

    log.info(
        '    planet is now %.4f M_earth at %.5f AU, e = %.4f',
        config.planet.mass_tot,
        config.orbit.semimajoraxis,
        config.orbit.eccentricity,
    )


def _drop_events_before_start(
    events: list[ImpactEvent], time_start: float
) -> list[ImpactEvent]:
    """Remove impacts that precede the start of the simulation.

    The configuration owns the planet's initial mass and orbit, so an
    impact that lands before the run begins cannot be applied without
    contradicting it. Such impacts are reported rather than dropped in
    silence, since they usually mean the time offset needs adjusting.

    Parameters
    ----------
    events : list of ImpactEvent
        Timeline, in time order.
    time_start : float
        Simulation time at the start of the run [yr].

    Returns
    -------
    kept : list of ImpactEvent
        Impacts at or after the start of the run.
    """
    kept = [e for e in events if e.time > time_start]
    dropped = len(events) - len(kept)

    if dropped:
        missed_mass = sum(e.mass_delta for e in events if e.time <= time_start)
        log.warning(
            '%d impact(s) fall at or before the start of the run (t = %.4e yr) and '
            'will not be applied, because the configured planet mass and orbit define '
            'the initial state. They would have added %.4f M_earth. Adjust '
            'accretion.time_offset to bring them into the simulated interval.',
            dropped,
            time_start,
            missed_mass / M_earth,
        )

    log.info('Scheduled %d impact(s)', len(kept))
    if kept:
        log.info('    first at %.4e yr, last at %.4e yr', kept[0].time, kept[-1].time)

    return kept
