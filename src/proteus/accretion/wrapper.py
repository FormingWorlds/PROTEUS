# Generic accretion wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.constants import M_earth, element_list

if TYPE_CHECKING:
    from proteus.accretion.common import ImpactEvent
    from proteus.proteus import Proteus

log = logging.getLogger('fwl.' + __name__)

# Rock-forming elements, whose mass grows through the structure solve
# (mass_tot and the equation of state) rather than the volatile budgets.
_ROCK_ELEMENTS = ('Si', 'Mg', 'Fe', 'Na')

# Every other tracked element's whole-planet budget is conserved across an
# impact's mass growth. Derived from the tracked-element registry so an
# element added there is conserved by default unless declared rock-forming.
# The set includes the noble gases, so a planet-matching impactor carries
# them in proportion like every other volatile.
_VOLATILE_ELEMENTS = tuple(e for e in element_list if e not in _ROCK_ELEMENTS)

# Elements configurable through the per-element ppmw fields. The ppmw mode
# can only deliver these; the planet-matching mode covers the full volatile
# set above, noble gases included.
_PPMW_ELEMENTS = ('H', 'C', 'N', 'S', 'O')


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

    # Size every volatile consequence from the pre-impact state, before any of
    # it is applied: what the impact strips from the target's atmosphere, and
    # what the impactor carries, split into the part delivered into the planet
    # and the part its own atmosphere loses with the collision.
    log.info(
        '    impactor volatiles: %s; atmosphere loss: %s',
        config.accretion.impactor_volatiles,
        config.accretion.atmloss_module or 'off',
    )
    strip = _target_strip_amounts(config, hf_row, event)
    content = _impactor_volatile_content(config, handler.hf_all, event)
    delivered, impactor_lost = _partition_impactor_content(config, hf_row, content)

    # Snapshot the whole-planet volatile budgets before the structure re-solve.
    # solve_structure recomputes the ppmw-mode budgets against the grown mass,
    # which would let even a dry impactor inflate the volatile inventory as if
    # the added rock carried the planet's volatile content. Volatiles are
    # conserved across the mass growth and change only through the strip and
    # delivery below; the rock mass grows through the structure solve itself.
    volatile_budgets = _snapshot_volatile_budgets(hf_row)

    # Grow the interior anchor by the impactor's rock alone: the merger mass
    # minus the impactor's full volatile content. The anchor and the volatile
    # budgets are the two halves of the whole-planet mass, so each impact
    # channel must land in exactly one of them; the delivered volatiles and
    # the target strip move the budgets below, and the impactor's lost
    # atmosphere never enters the planet at all. The whole-planet mass then
    # closes to before + rock + delivered - stripped, which can be a net
    # shrink when a small impactor blows off a heavier atmosphere.
    # mass_tot is in Earth masses; the amounts are in kg.
    impactor_rock = event.mass_delta - sum(content.values())
    config.planet.mass_tot += impactor_rock / M_earth
    solve_structure(
        handler.directories, config, handler.hf_all, hf_row, handler.directories['output']
    )

    # Restore the conserved volatile budgets over the mass-scaled values the
    # structure solve wrote, so the growth adds rock, not volatiles.
    _restore_volatile_budgets(hf_row, volatile_budgets)

    # Apply the sized consequences to the whole-planet budgets and refresh
    # the tracked-element total the budgets aggregate into.
    _apply_volatile_consequences(config, hf_row, strip, delivered, impactor_lost)

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


def _apply_volatile_consequences(
    config, hf_row: dict, strip: dict, delivered: dict, impactor_lost: dict
) -> None:
    """Apply an impact's sized volatile changes to the whole-planet budgets.

    Debits the stripped target atmosphere, books it into the escaped-mass
    ledger the desiccation gate audits, credits the delivered impactor
    volatiles, and refreshes the tracked-element total. The outgassing step
    later this iteration re-equilibrates the atmosphere against the updated
    totals; an element deferred to the chemistry step (e.g. oxygen under
    ic_chemistry) is re-derived there either way.

    Parameters
    ----------
    config : Config
        Model configuration, read for the strip-percentage log line.
    hf_row : dict
        Current helpfile row, mutated in place.
    strip, delivered, impactor_lost : dict
        Per-element masses [kg] sized from the pre-impact state.
    """
    for e, removed in strip.items():
        key = f'{e}_kg_total'
        hf_row[key] = max(0.0, float(hf_row.get(key, 0.0)) - removed)
    if strip:
        stripped_total = sum(strip.values())
        hf_row['esc_kg_cumulative'] = (
            float(hf_row.get('esc_kg_cumulative', 0.0)) + stripped_total
        )
        log.info(
            '    impact stripped %.1f%% of the atmosphere: %.3e kg removed',
            100.0 * float(config.accretion.atmloss_frac),
            stripped_total,
        )
    for e, added in delivered.items():
        key = f'{e}_kg_total'
        hf_row[key] = float(hf_row.get(key, 0.0)) + added
    if delivered:
        log.info(
            '    delivered impactor volatiles [kg]: %s',
            ', '.join(f'{e}={v:.3e}' for e, v in delivered.items()),
        )
    if impactor_lost:
        log.info(
            '    impactor atmosphere lost with the collision [kg]: %s (%.3e total)',
            ', '.join(f'{e}={v:.3e}' for e, v in impactor_lost.items()),
            sum(impactor_lost.values()),
        )

    # Refresh the tracked-element total from the conserved budgets plus the
    # strip and delivery. solve_structure set M_ele from the mass-scaled
    # values it computed, which the updates above have overridden.
    _refresh_tracked_element_total(hf_row)


def _primordial_mass_fractions(hf_all) -> dict:
    """Volatile mass fractions of the planet at formation [kg/kg].

    Reads the settled initial state from the run's own history: the last row
    of the init epoch (``Time < 1`` yr, the same discriminator the outgassing
    warm start uses), or the first row when no init row exists. The helpfile
    is persisted, so a resumed run recovers the same formation composition
    without any extra state.

    Parameters
    ----------
    hf_all : pd.DataFrame
        Full helpfile history of the run.

    Returns
    -------
    dict
        Mapping of volatile element to ``<e>_kg_total / M_planet`` at the
        formation state.

    Raises
    ------
    RuntimeError
        If no history is available or the formation row carries no positive
        planet mass; the impactor composition would be undefined.
    """
    if hf_all is None or len(hf_all) == 0:
        raise RuntimeError(
            'Cannot scale impactor volatiles to the planet: no helpfile history '
            'is available to read the formation composition from.'
        )

    init_rows = hf_all[hf_all['Time'] < 1.0]
    t0 = init_rows.iloc[-1] if len(init_rows) else hf_all.iloc[0]

    m_planet = float(t0.get('M_planet', 0.0))
    if m_planet <= 0.0:
        raise RuntimeError(
            'Cannot scale impactor volatiles to the planet: the formation row '
            f'carries M_planet = {m_planet!r}.'
        )

    fractions = {e: float(t0.get(f'{e}_kg_total', 0.0)) / m_planet for e in _VOLATILE_ELEMENTS}
    log.info(
        '    formation composition (M_planet=%.3e kg at t=%.2e yr): %s',
        m_planet,
        float(t0.get('Time', 0.0)),
        ', '.join(f'{e}={x:.2e}' for e, x in fractions.items() if x > 0.0),
    )
    return fractions


def _impactor_volatile_content(config, hf_all, event: ImpactEvent) -> dict:
    """Total volatile mass the impactor carries, per element [kg].

    Dispatches on ``accretion.impactor_volatiles``: a dry impactor carries
    nothing; ``match_planet`` scales the planet's formation mass fractions to
    the impactor mass, on the assumption that every embryo in the dynamical
    model co-formed from the same disk material; ``ppmw`` uses the configured
    per-element budgets. Only positive contributions are returned.

    Under ``O_mode = 'ic_chemistry'`` oxygen is excluded from the content:
    the volatile O budget is chemistry-derived (the next outgassing call
    re-equilibrates it against the fO2 buffer for the grown planet), so a
    delivered O mass would be overwritten while its subtraction from the
    interior anchor persisted. The impactor's oxygen then arrives as part of
    its rock, which is where oxide-bound oxygen belongs.
    """
    mode = config.accretion.impactor_volatiles
    content: dict[str, float] = {}

    if mode == 'match_planet':
        fractions = _primordial_mass_fractions(hf_all)
        for e, x0 in fractions.items():
            if x0 > 0.0:
                content[e] = x0 * event.M_impactor
    elif mode == 'ppmw':
        for e in _PPMW_ELEMENTS:
            ppmw = getattr(config.accretion, f'impactor_{e}_ppmw')
            if ppmw > 0.0:
                content[e] = event.M_impactor * ppmw / 1.0e6

    o_mode = getattr(getattr(config.planet, 'elements', None), 'O_mode', None)
    if o_mode == 'ic_chemistry':
        content.pop('O', None)

    return content


def _partition_impactor_content(config, hf_row: dict, content: dict) -> tuple[dict, dict]:
    """Split the impactor's volatiles into a delivered and a lost part [kg].

    The impactor's internal partitioning is unknowable, so the planet's own
    atmosphere-versus-interior split per element at impact time is mirrored
    onto it. With impact atmosphere loss active the impactor's atmospheric
    part is lost with the collision (the impactor is disrupted and its
    gravity is lower than the target's) and only the dissolved part is
    delivered; with loss disabled the whole content is delivered. The mirror
    understates a smaller body's atmospheric fraction (it equilibrates at
    lower surface pressure), so delivery is somewhat overestimated; the
    coming ZEPHYRUS collision law replaces this convention.

    For an element the planet no longer holds, the per-element mirror is
    undefined and the planet's bulk atmospheric fraction is used instead.

    Returns
    -------
    (delivered, lost) : tuple of dict
        Per-element masses delivered into the planet and lost to space [kg].
    """
    if config.accretion.atmloss_module is None:
        return dict(content), {}

    # Bulk atmospheric fraction as the fallback mirror for elements the
    # planet no longer tracks a budget for.
    tot_all = sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in _VOLATILE_ELEMENTS)
    atm_all = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in _VOLATILE_ELEMENTS)
    f_atm_bulk = atm_all / tot_all if tot_all > 0.0 else 0.0

    delivered: dict[str, float] = {}
    lost: dict[str, float] = {}
    mirror: dict[str, float] = {}
    for e, mass in content.items():
        total_e = float(hf_row.get(f'{e}_kg_total', 0.0))
        if total_e > 0.0:
            f_atm = float(hf_row.get(f'{e}_kg_atm', 0.0)) / total_e
        else:
            f_atm = f_atm_bulk
        f_atm = min(max(f_atm, 0.0), 1.0)
        mirror[e] = f_atm
        lost_e = mass * f_atm
        if lost_e > 0.0:
            lost[e] = lost_e
        if mass - lost_e > 0.0:
            delivered[e] = mass - lost_e

    if mirror:
        log.info(
            '    impactor atmospheric fraction per element (planet mirror): %s',
            ', '.join(f'{e}={f:.2f}' for e, f in mirror.items()),
        )
    return delivered, lost


def _target_strip_amounts(config, hf_row: dict, event: ImpactEvent) -> dict:
    """Mass the impact strips from the target's atmosphere, per element [kg].

    Sizes the debit from the pre-impact state without mutating it: the loss
    fraction times the atmospheric reservoir, partitioned over the elements in
    proportion to their atmospheric masses through the same path continuous
    escape uses, so the per-element loss can never exceed what the atmosphere
    holds and the dissolved interior inventory is untouched. An atmosphere
    below the outgassing mass threshold is treated as nothing to strip, the
    same convention continuous escape applies to it.
    """
    from proteus.escape.wrapper import calc_new_elements

    f_loss = _impact_loss_fraction(config, hf_row, event)
    if f_loss <= 0.0:
        return {}

    m_atm = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
    if m_atm < config.outgas.mass_thresh:
        log.info(
            '    impact atmosphere loss: atmosphere below the mass threshold, not stripped'
        )
        return {}

    tgt = calc_new_elements(
        hf_row,
        dt=0.0,
        reservoir='outgas',
        min_thresh=config.outgas.mass_thresh,
        esc_mass=f_loss * m_atm,
    )
    strip = {}
    for e, new_total in tgt.items():
        removed = float(hf_row.get(f'{e}_kg_total', 0.0)) - float(new_total)
        if removed > 0.0:
            strip[e] = removed
    return strip


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
