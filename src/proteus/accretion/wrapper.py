# Generic accretion wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.constants import M_earth, element_list, noble_gases

if TYPE_CHECKING:
    from proteus.accretion.common import ImpactEvent
    from proteus.proteus import Proteus

log = logging.getLogger('fwl.' + __name__)

# Volatile and noble-gas elements whose whole-planet budgets are conserved
# across an impact's mass growth. The rock-forming elements (Si, Mg, Fe, Na)
# are not carried in the volatile budget; the planet's rock mass grows through
# the structure solve (mass_tot and the equation of state), not here.
_VOLATILE_ELEMENTS = ('H', 'O', 'C', 'N', 'S', *noble_gases)


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

    # Advise when the Aragog re-melt initial condition is not guaranteed molten.
    # The re-melt re-applies the run's temperature-mode initial condition, and
    # only some modes guarantee it is fully molten; the others are only as molten
    # as the user's temperature or entropy value. Emitted here, after the file
    # logger exists, rather than in the config validator, which runs before it.
    _MOLTEN_MODES = ('liquidus_super', 'accretion', 'adiabatic_from_cmb')
    if (
        config.interior_energetics.module == 'aragog'
        and config.planet.temperature_mode not in _MOLTEN_MODES
    ):
        log.warning(
            "Accretion on Aragog with temperature_mode='%s': each impact re-melts "
            'the mantle by re-applying this initial condition, which is not guaranteed '
            "fully molten. Use temperature_mode='liquidus_super' for a molten re-melt, "
            'or confirm the initial melt fraction is what you intend.',
            config.planet.temperature_mode,
        )
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
    from proteus.interior_energetics.wrapper import remelt_mantle, solve_structure

    config = handler.config
    hf_row = handler.hf_row

    log.info(
        'Giant impact at t = %.4e yr: target %d struck by %d, adding %.4f M_earth',
        event.time,
        event.id_target,
        event.id_impactor,
        event.mass_delta / M_earth,
    )

    # Snapshot the whole-planet volatile budgets before the structure re-solve.
    # solve_structure recomputes the ppmw-mode budgets against the grown mass,
    # which would let even a dry impactor inflate the volatile inventory as if
    # the added rock carried the planet's volatile content. Volatiles are
    # conserved across the mass growth and grow only through the explicit
    # delivery below; the rock mass grows through the structure solve itself.
    volatile_budgets = _snapshot_volatile_budgets(hf_row)

    # Grow the planet by the impactor mass and re-solve the structure. mass_tot
    # is in Earth masses; the event's mass delta is the impactor mass in kg.
    config.planet.mass_tot += event.mass_delta / M_earth
    solve_structure(
        handler.directories, config, handler.hf_all, hf_row, handler.directories['output']
    )

    # Restore the conserved volatile budgets over the mass-scaled values the
    # structure solve wrote, so the growth adds rock, not volatiles.
    _restore_volatile_budgets(hf_row, volatile_budgets)

    # Re-melt the mantle to its molten initial condition, so the interior
    # evolves from a fully molten state after the impact.
    remelt_mantle(handler.directories, config, hf_row, handler.interior_o, event)

    # A mantle that had crystallised is now a magma ocean again, so lift the
    # one-way solidification latch; otherwise outgassing would stay frozen and
    # the volatiles would be treated as locked in a solid mantle for good.
    if getattr(handler, 'crystallized', False):
        handler.crystallized = False
        log.info('    solidification latch cleared: the mantle is molten again')

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

    # Strip part of the pre-impact atmosphere, before the impactor's own
    # volatiles are delivered: the shock blows off what the planet already
    # has, while the delivery goes into the molten post-impact planet. The
    # outgassing step later this iteration re-equilibrates the atmosphere
    # against the debited totals.
    _strip_impact_atmosphere(config, hf_row, event)

    # Deliver the impactor's volatiles into the whole-planet element budgets,
    # so the outgassing step later this iteration re-equilibrates with them. The
    # amount per element is the impactor mass times the configured content in
    # ppmw; every element defaults to zero (a dry impactor), so this is a no-op
    # unless a run opts in. Only the keys with a non-zero content are touched, so
    # an element deferred to the chemistry step (e.g. oxygen under ic_chemistry)
    # is left alone when nothing is delivered for it.
    _deliver_impactor_volatiles(config, hf_row, event.M_impactor)

    # Refresh the tracked-element total from the conserved budgets plus any
    # delivered volatiles. solve_structure set M_ele from the mass-scaled
    # values it computed, which the restore above has overridden.
    _refresh_tracked_element_total(hf_row)


def _deliver_impactor_volatiles(config, hf_row: dict, m_impactor: float) -> None:
    """Add the impactor's volatile content to the whole-planet element budgets.

    Parameters
    ----------
    config : Config
        Model configuration; reads ``accretion.impactor_<e>_ppmw``.
    hf_row : dict
        Current helpfile row, whose ``<e>_kg_total`` budgets are grown in place.
    m_impactor : float
        Impactor mass [kg].
    """
    delivered = {}
    for element in ('H', 'C', 'N', 'S', 'O'):
        ppmw = getattr(config.accretion, f'impactor_{element}_ppmw')
        if ppmw <= 0.0:
            continue
        added = m_impactor * ppmw / 1.0e6
        key = f'{element}_kg_total'
        hf_row[key] = hf_row.get(key, 0.0) + added
        delivered[element] = added

    if delivered:
        log.info(
            '    delivered impactor volatiles [kg]: %s',
            ', '.join(f'{e}={v:.3e}' for e, v in delivered.items()),
        )


def _impact_loss_fraction(config, hf_row: dict, event: ImpactEvent) -> float:
    """Fraction of the atmosphere removed by this impact [0-1].

    Dispatches on ``accretion.atmloss_module``. The constant module returns
    the configured fixed fraction and stands in for the coming ZEPHYRUS
    collision-loss law, which will compute the fraction from the impact
    parameters this function already receives; PROTEUS itself deliberately
    ships no impact loss physics.

    Parameters
    ----------
    config : Config
        Model configuration; reads ``accretion.atmloss_module`` and
        ``accretion.atmloss_frac``.
    hf_row : dict
        Current helpfile row (the planet state a loss law reads).
    event : ImpactEvent
        The impact being applied (the collision parameters a loss law reads).

    Returns
    -------
    float
        Loss fraction in [0, 1]. Zero when the loss is disabled.

    Raises
    ------
    ValueError
        If a loss module returns a fraction outside [0, 1]. The debit
        partitioning is only meaningful on that interval, so a provider
        violating it is a contract error, not a value to clamp silently.
    """
    module = config.accretion.atmloss_module
    if module is None:
        return 0.0

    match module:
        case 'constant':
            f_loss = float(config.accretion.atmloss_frac)
        case _:
            raise ValueError(f"Invalid accretion.atmloss_module: '{module}'")

    if not 0.0 <= f_loss <= 1.0:
        raise ValueError(
            f'Impact atmosphere loss fraction must be in [0, 1], got {f_loss!r} '
            f"from atmloss_module '{module}'"
        )
    return f_loss


def _strip_impact_atmosphere(config, hf_row: dict, event: ImpactEvent) -> None:
    """Remove part of the atmosphere at an impact and debit the element budgets.

    The lost mass is the loss fraction times the atmospheric reservoir, and is
    partitioned over the elements in proportion to their atmospheric masses
    through the same path continuous escape uses, so the per-element loss can
    never exceed what the atmosphere holds and the dissolved interior inventory
    is untouched. The debited mass is added to the cumulative escaped-mass
    ledger, which the desiccation gate audits: without that booking, a planet
    dried out partly by impacts would be refused desiccation later.

    Parameters
    ----------
    config : Config
        Model configuration.
    hf_row : dict
        Current helpfile row, mutated in place.
    event : ImpactEvent
        The impact being applied.
    """
    from proteus.escape.wrapper import calc_new_elements

    f_loss = _impact_loss_fraction(config, hf_row, event)
    if f_loss <= 0.0:
        return

    # Atmospheric reservoir mass, summed the same way the debit partitions it.
    m_atm = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
    if m_atm <= 0.0:
        log.info('    impact atmosphere loss: no atmosphere to strip')
        return

    before = {e: float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list}
    tgt = calc_new_elements(
        hf_row,
        dt=0.0,
        reservoir='outgas',
        min_thresh=config.outgas.mass_thresh,
        esc_mass=f_loss * m_atm,
    )
    for e, mass in tgt.items():
        hf_row[f'{e}_kg_total'] = mass

    # Book the actually-debited mass (including any desiccation-floor
    # truncation) into the escaped-mass ledger the desiccation gate audits.
    lost = sum(before[e] - float(tgt.get(e, before[e])) for e in before)
    hf_row['esc_kg_cumulative'] = float(hf_row.get('esc_kg_cumulative', 0.0)) + lost

    log.info(
        '    impact stripped %.1f%% of the atmosphere: %.3e kg removed',
        100.0 * f_loss,
        lost,
    )


def _snapshot_volatile_budgets(hf_row: dict) -> dict:
    """Capture the whole-planet volatile element budgets [kg].

    Parameters
    ----------
    hf_row : dict
        Current helpfile row.

    Returns
    -------
    budgets : dict
        Mapping of volatile element symbol to its ``<e>_kg_total`` value [kg],
        for the elements conserved across an impact's mass growth. Only the
        elements that already carry a budget in the row are captured, so the
        restore conserves what existed rather than fabricating zero-valued keys
        for volatiles the run does not track.
    """
    return {
        e: float(hf_row[f'{e}_kg_total'])
        for e in _VOLATILE_ELEMENTS
        if f'{e}_kg_total' in hf_row
    }


def _restore_volatile_budgets(hf_row: dict, budgets: dict) -> None:
    """Write conserved volatile element budgets back into the helpfile row.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, mutated in place.
    budgets : dict
        Snapshot returned by :func:`_snapshot_volatile_budgets`.
    """
    for element, kg in budgets.items():
        hf_row[f'{element}_kg_total'] = kg


def _refresh_tracked_element_total(hf_row: dict) -> None:
    """Recompute the total tracked-element mass ``M_ele`` [kg].

    Mirrors the aggregation in
    :func:`proteus.outgas.wrapper.calc_target_elemental_inventories`, summing
    every tracked element's ``<e>_kg_total``. Called after the volatile budgets
    are restored and the impactor delivery is added, so ``M_ele`` reflects the
    conserved-plus-delivered inventory rather than the mass-scaled values the
    structure solve produced.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, whose ``M_ele`` is updated in place.
    """
    hf_row['M_ele'] = sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list)


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
