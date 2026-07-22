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
