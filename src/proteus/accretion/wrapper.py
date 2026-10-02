# Generic accretion wrapper
from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

from proteus.utils.constants import AU, M_earth, element_list, noble_gases, vol_element_list
from proteus.utils.coupler import helpfile_path

if TYPE_CHECKING:
    from proteus.accretion.common import ImpactEvent
    from proteus.config import Config
    from proteus.proteus import Proteus

log = logging.getLogger('fwl.' + __name__)

# Volatile elements whose whole-planet budgets are conserved across impact mass
# growth. This set matches what update_planet_mass sums into M_ele.
_VOLATILE_ELEMENTS = tuple(e for e in element_list if e in vol_element_list or e in noble_gases)

# Elements configurable through the per-element ppmw fields. The ppmw mode
# can only deliver these; the planet-matching mode covers the full volatile
# set above, noble gases included.
_PPMW_ELEMENTS = ('H', 'C', 'N', 'S', 'O')

# Where the run records the impact timeline it resolved at initialisation, in
# its own output directory. A resumed run replays this file instead of asking
# the module for a timeline again.
_RESOLVED_TIMELINE_FILE = 'impact_timeline.csv'

# Longest step of the init stage [yr]: the time-stepper's static 1 yr step, which
# only shrinks on retry; the solver-derived dt (interior_energetics/wrapper.py) cannot exceed it.
_INIT_STAGE_HORIZON_YR = 1.0

# Ceiling on the planet's eccentricity after an impact applies its change. An
# impact excites a bound orbit; it cannot unbind one, and the rest of the model
# assumes a closed orbit throughout.
_ECC_MAX = 0.99


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

    # Advise when the Aragog re-melt temperature mode does not guarantee a molten
    # state. Only 'liquidus_super' guarantees a fully molten mantle.
    _GUARANTEED_MOLTEN_MODES = ('liquidus_super',)
    if (
        config.interior_energetics.module == 'aragog'
        and config.planet.temperature_mode not in _GUARANTEED_MOLTEN_MODES
    ):
        log.warning(
            "Accretion on Aragog with temperature_mode='%s': each impact re-melts "
            'the mantle by raising it to this initial condition, which is not guaranteed '
            "fully molten. Use temperature_mode='liquidus_super' for a molten re-melt, "
            'or confirm the initial melt fraction is what you intend.',
            config.planet.temperature_mode,
        )
    log.info('')

    from proteus.accretion.common import read_timeline, write_timeline

    resolved_path = os.path.join(handler.directories['output'], _RESOLVED_TIMELINE_FILE)

    # Resumed runs replay the previously resolved timeline to ensure consistent
    # impact history across restarts.
    if config.params.resume and os.path.exists(resolved_path):
        # Written on the PROTEUS axis with the offset already applied, so it
        # must not be offset a second time.
        events = read_timeline(resolved_path, time_offset=0.0)
        log.info('Replaying the impact timeline resolved at the start of this run')
    else:
        match module:
            case 'dummy':
                from proteus.accretion.dummy import get_timeline
            case 'timeline':
                from proteus.accretion.timeline import get_timeline
            case 'morrigan':
                from proteus.accretion.morrigan import get_timeline
            case _:
                raise ValueError(f"Invalid accretion module: '{module}'")

        events = get_timeline(config)
        write_timeline(events, resolved_path)

    if not events and module in ('timeline', 'morrigan'):
        log.warning("Accretion module '%s' resolved to 0 impacts", module)

    return _drop_events_before_start(
        events, handler.hf_row.get('Time', 0.0), resumed=bool(config.params.resume)
    )


def _as_float(val: object) -> float:
    """Convert value to float, returning NaN on ValueError or TypeError."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return float('nan')


def _current_orbit(hf_row: dict, config: Config) -> tuple[float, float]:
    """Return the current planetary orbit (semi-major axis [m] and eccentricity [1]).

    Reads from ``hf_row`` when available, falling back to ``config.orbit``
    when row entries are missing, non-finite, or unphysical (non-positive
    semi-major axis or negative eccentricity).

    Parameters
    ----------
    hf_row : dict
        Current step helpfile row.
    config : Config
        Model configuration.

    Returns
    -------
    tuple of float
        Current semi-major axis in metres [m] and eccentricity [1].
    """
    a = _as_float(hf_row.get('semimajorax'))
    e = _as_float(hf_row.get('eccentricity'))
    e_cfg = float(config.orbit.eccentricity)
    if not (math.isfinite(a) and a > 0.0):
        return float(config.orbit.semimajoraxis) * AU, e_cfg
    return a, (e if math.isfinite(e) and e >= 0.0 else e_cfg)


def restore_accretion_state(handler: Proteus) -> None:
    """Rebuild the accretion state a resumed run cannot read from its TOML.

    Each impact grows ``config.planet.mass_tot`` and moves
    ``config.orbit.semimajoraxis`` and ``config.orbit.eccentricity``. The
    configuration is the run's specification, rebuilt from file on every start,
    so none of that survives a restart: without this, a resumed run would solve
    the structure against the planet's original mass, discarding the growth of
    every impact before the resume point, and would snap the orbit back to its
    configured value on the first step whenever tides are off, because that
    path re-pins the row from the configuration each iteration.

    The mass is rebuilt from ``M_accreted_rock``, the cumulative rock the
    impacts added, on top of the configured mass rather than from ``M_planet``:
    the anchor carries rock alone, while ``M_planet`` also carries the volatile
    budgets, so anchoring on it would fold the volatiles into the rock and
    drift further on every subsequent resume. The volatile change the Zalmoxis
    target adds is read from ``M_accreted_net`` in the row; a helpfile written
    before that column existed reads it as zero, and the rock is stored there.

    Call after :func:`init_accretion`, so the timeline is still resolved
    against the configured mass and orbit and a re-run dynamical model selects
    the same body it selected originally.

    On resume the helpfile row is checked in this order:

    1. ``Time``, ``M_accreted_rock`` and ``n_impacts_applied`` must be finite
       and non-negative, and the counter must be an integer.
    2. Rock with a zero or absent counter is refused when an accretion module
       is selected, since that run predates the counter. With accretion off it
       is accepted with a warning and the mass is rebuilt from the ledger.
    3. With a module selected, the counter is checked against the resolved
       timeline: it may not be below the number of impacts at or before the
       resume time, and any surplus must be the next impacts after the resume
       time, each within the init stage (``t <= 1 yr``): only there can an
       impact land after the row's Time, since the init steps restart from
       zero. Those surplus impacts are dropped from the schedule.

    Parameters
    ----------
    handler : Proteus
        Proteus object instance, whose configuration is updated in place.

    Raises
    ------
    RuntimeError
        If ``Time``, ``M_accreted_rock`` or ``n_impacts_applied`` is not a
        finite non-negative number, ``M_accreted_net`` is not finite, the
        counter is not an integer, rock is recorded with no counter while a
        module is selected, or the counter disagrees with the resolved
        timeline.
    """
    config = handler.config

    # Restore accretion state from the helpfile ledger rather than config settings,
    # ensuring mass history persists even if accretion is later disabled.
    if not config.params.resume:
        return

    hf_row = handler.hf_row
    out_dir = getattr(handler, 'directories', {}).get('output', '.')
    hf_name = helpfile_path(out_dir)

    t_raw = hf_row.get('Time')
    t_num = _as_float(0.0 if t_raw is None else t_raw)
    if not (0.0 <= t_num < math.inf):
        raise RuntimeError(
            f'Resume refused: {hf_name} contains invalid Time = {t_raw!r}. '
            'Restart the simulation.'
        )
    resume_time = t_num

    m_raw = hf_row.get('M_accreted_rock')
    accreted = _as_float(0.0 if m_raw is None else m_raw)
    if not (0.0 <= accreted < math.inf):
        raise RuntimeError(
            f'Resume refused: {hf_name} contains invalid M_accreted_rock = {m_raw!r}. '
            'Restart the simulation.'
        )

    net_raw = hf_row.get('M_accreted_net')
    net = _as_float(0.0 if net_raw is None else net_raw)
    if not math.isfinite(net):
        raise RuntimeError(
            f'Resume refused: {hf_name} contains invalid M_accreted_net = {net_raw!r}. '
            'Restart the simulation.'
        )
    legacy = net == 0.0 and accreted > 0.0
    if legacy:
        log.info('Helpfile %s has no M_accreted_net: it holds the rock alone', hf_name)
        net = accreted

    n_raw = hf_row.get('n_impacts_applied')
    n_num = _as_float(0.0 if n_raw is None else n_raw)
    if not (n_num >= 0.0 and n_num.is_integer()):
        raise RuntimeError(
            f'Resume refused: {hf_name} contains corrupt n_impacts_applied = {n_raw!r}. '
            'Restart the simulation.'
        )

    module_on = config.accretion.module is not None

    if accreted > 0.0 and n_num == 0.0:
        if module_on:
            raise RuntimeError(
                f'Resume refused: {hf_name} records M_accreted_rock = {accreted:.6e} kg, '
                f'but n_impacts_applied is {n_raw!r}. This run predates the impact counter '
                'and cannot be resumed safely. Restart the simulation.'
            )
        log.warning(
            'Helpfile %s records M_accreted_rock = %.6e kg but n_impacts_applied is %r. '
            'Accretion is disabled for this resume; restoring mass and orbit from the ledger.',
            hf_name,
            accreted,
            n_raw,
        )

    n_applied = int(n_num)
    resolved_path = os.path.join(out_dir, _RESOLVED_TIMELINE_FILE)
    n_drop = 0
    if module_on and os.path.exists(resolved_path):
        from proteus.accretion.common import read_timeline

        all_events = read_timeline(resolved_path, time_offset=0.0)
        events_before = sum(1 for ev in all_events if 0.0 < ev.time <= resume_time)
        if n_applied < events_before:
            raise RuntimeError(
                f'Resume refused: {hf_name} records n_impacts_applied = {n_applied}, '
                f'but {events_before} impact(s) precede the resume time {resume_time} yr. '
                'Restart the simulation.'
            )
        # A counted impact after the resume time can only have landed during the
        # init stage, whose steps never reach beyond _INIT_STAGE_HORIZON_YR.
        later = [ev for ev in all_events if ev.time > resume_time]
        n_drop = n_applied - events_before
        if n_drop > len(later):
            raise RuntimeError(
                f'Resume refused: {hf_name} records n_impacts_applied = {n_applied}, '
                f'but the resolved timeline holds only {events_before + len(later)} '
                'impact(s). Restart the simulation.'
            )
        if any(ev.time > _INIT_STAGE_HORIZON_YR for ev in later[:n_drop]):
            raise RuntimeError(
                f'Resume refused: {hf_name} records n_impacts_applied = {n_applied}, '
                f'but only {events_before} impact(s) precede the resume time {resume_time} yr '
                f'and the next {n_drop} cannot have landed during the init stage '
                f'(t <= {_INIT_STAGE_HORIZON_YR} yr). Restart the simulation.'
            )
        if n_drop > 0:
            log.info(
                'Resume: %d impact(s) after the resume time landed during the init stage '
                'and are not applied again: %s',
                n_drop,
                ', '.join(f'{ev.time:g} yr' for ev in later[:n_drop]),
            )

    hf_row['n_impacts_applied'] = n_applied
    if getattr(handler, 'hf_all', None) is not None and len(handler.hf_all) > 0:
        handler.hf_all.loc[handler.hf_all.index[-1], 'n_impacts_applied'] = float(n_applied)

    pending = getattr(handler, 'impact_events', None)
    if module_on and pending:
        handler.impact_events = [ev for ev in pending if ev.time > resume_time][n_drop:]

    if legacy:
        # Store it, or later steps add to zero and the Zalmoxis target loses the rock.
        hf_row['M_accreted_net'] = net
        if getattr(handler, 'hf_all', None) is not None and len(handler.hf_all) > 0:
            handler.hf_all.loc[handler.hf_all.index[-1], 'M_accreted_net'] = net
    if accreted <= 0.0:
        # Inform user when continuing from configured mass, which occurs either
        # prior to any impacts or when resuming from an older helpfile format.
        if config.accretion.module is not None and net == 0.0:
            log.info(
                'No accreted rock recorded before this resume: continuing from the '
                'configured mass of %.4f M_earth. If this run had already applied an '
                'impact, its helpfile predates the ledger and that growth is lost.',
                config.planet.mass_tot,
            )
        return

    config.planet.mass_tot += accreted / M_earth

    base_a, base_e = _current_orbit(hf_row, config)
    config.orbit.semimajoraxis = base_a / AU
    config.orbit.eccentricity = base_e

    log.info(
        'Restored accretion state: %.4f M_earth at %.5f AU, e = %.4f '
        '(%.3e kg of rock accreted before the resume)',
        config.planet.mass_tot,
        config.orbit.semimajoraxis,
        config.orbit.eccentricity,
        accreted,
    )


def discard_preimpact_snapshot(handler: Proteus) -> None:
    """Drop the interior snapshot a step wrote before an impact re-melted it.

    The interior writes its snapshot while the step is solved, which is before
    the impacts falling in that step are applied at the end of it. When a step
    both writes a snapshot and lands an impact, the snapshot therefore holds
    the mantle from before the re-melt while the helpfile row it shares a time
    with already carries the impact's mass, orbit and volatile budgets.
    Resuming from that pair would restore a mantle the impact had melted while
    treating the impact as already applied, so the re-melt would be lost with
    nothing to signal it.

    Removing the snapshot leaves the row without a complete pair, so
    :func:`proteus.utils.coupler.select_resumable_snapshot` walks back to the
    last step that has one and truncates the helpfile to it. The impact then
    falls after the resume point and is applied again in full. The cost is the
    steps between the two snapshots, which are recomputed.

    The snapshot is only removed when an older one survives it. Removing the
    last one would leave a run with no interior state on disk at all: a resume
    would find no complete pair and refuse, and the run's own interior history
    would end at the impact. That case is reported instead, since the snapshot
    it keeps describes the mantle from before the re-melt and a resume from it
    would carry that inconsistency.

    Only the interior modules that write a snapshot need this. The dummy and
    boundary interiors carry their state in the helpfile row itself, which is
    already post-impact, and SPIDER has no re-melt path and is refused for
    accretion runs before the first impact.

    Parameters
    ----------
    handler : Proteus
        Proteus object instance, read for the output directory, the interior
        module and the current time.
    """
    if handler.config.interior_energetics.module != 'aragog':
        return

    from proteus.interior_energetics.aragog import (
        discard_snapshot,
        earlier_snapshot_exists,
    )

    output = handler.directories['output']
    time = float(handler.hf_row['Time'])

    if not earlier_snapshot_exists(output, time):
        log.warning(
            '    the interior snapshot at %.4e yr predates this step re-melt and '
            'is the only one on disk, so it is kept: resuming from it would start '
            'from a mantle this impact had already melted',
            time,
        )
        return

    if discard_snapshot(output, time):
        log.info(
            '    discarded the interior snapshot at %.4e yr: it predates this '
            "step's re-melt, so a resume continues from the previous one",
            time,
        )


def apply_impact(handler: Proteus, event: ImpactEvent) -> None:
    """Apply one giant impact's consequences to the running planet.

    Called once for each impact, at the end of the timestep that lands on
    its time, so the orbit and structure of that step already use the grown
    planet and the next interior solve evolves it from there.

    The impactor's rock is added to the planet's total mass and the interior
    structure is re-solved, so the radius, gravity and the core/mantle split
    follow the new mass at the configured core fraction. ``M_accreted_net``
    records the rock and, with the Zalmoxis structure, the delivered volatiles
    minus the stripped atmosphere, which the Zalmoxis target adds to
    ``mass_tot``. The orbit change updates the running row base (which tides
    evolve) and the configuration reflects the current post-impact orbit.

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
    net_before = _net_ledger(hf_row)  # refuse a corrupt ledger before anything moves

    ratio = event.semimajoraxis_ratio
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            f'Impact event at t = {event.time:.4e} yr has non-positive or non-finite '
            f'semimajoraxis_ratio = {ratio!r}'
        )

    log.info(
        'Giant impact at t = %.4e yr: target %d struck by %d, adding %.4f M_earth',
        event.time,
        event.id_target,
        event.id_impactor,
        event.mass_delta / M_earth,
    )

    # Calculate volatile stripping and delivery from the pre-impact state
    # before applying any mass updates.
    log.info(
        '    impactor volatiles: %s; atmosphere loss: %s',
        config.accretion.impactor_volatiles,
        config.accretion.atmloss_module or 'off',
    )
    f_loss = _impact_loss_fraction(config, hf_row, event)
    strip = _target_strip_amounts(config, hf_row, f_loss)
    content = _impactor_volatile_content(config, handler.hf_all, event, hf_row=hf_row)
    delivered, impactor_lost = _partition_impactor_content(config, hf_row, content, f_loss)

    # Snapshot volatile budgets before the structure solve to prevent ppmw
    # recomputation from artificially inflating volatile inventories.
    volatile_budgets = _snapshot_volatile_budgets(hf_row)

    from proteus.accretion.common import MASS_CLOSURE_RTOL

    impactor_rock = event.mass_delta - sum(content.values())
    # Validate that volatile mass does not exceed impactor mass beyond numerical
    # closure tolerance. Small negative remainders within tolerance clamp to zero.
    rock_tol = MASS_CLOSURE_RTOL * (event.M_target_before + event.M_impactor)
    if impactor_rock < -rock_tol:
        raise ValueError(
            f'Impactor volatile content {sum(content.values()):.6e} kg exceeds the '
            f'{event.mass_delta:.6e} kg it adds to the planet, so the impact would '
            f'remove {-impactor_rock:.4e} kg of rock from the interior. With '
            f'accretion.impactor_volatiles = {config.accretion.impactor_volatiles!r}, '
            'the content is set by '
            + (
                'the per-element accretion.impactor_<e>_ppmw budgets, which must '
                'total below 1e6 ppmw.'
                if config.accretion.impactor_volatiles == 'ppmw'
                else "the planet's own composition at the time of impact."
            )
        )
    impactor_rock = max(impactor_rock, 0.0)
    config.planet.mass_tot += impactor_rock / M_earth

    # Record accreted rock and impact counts in the helpfile so resumed runs
    # can restore the accumulated mass and event state.
    hf_row['M_accreted_rock'] = float(hf_row.get('M_accreted_rock') or 0.0) + impactor_rock
    hf_row['M_accreted_net'] = net_before + impactor_rock  # keeps V unchanged for this solve
    hf_row['n_impacts_applied'] = int(float(hf_row.get('n_impacts_applied') or 0.0)) + 1
    solve_structure(
        handler.directories,
        config,
        handler.hf_all,
        hf_row,
        handler.directories['output'],
        thermal_solve=False,
    )

    # Restore the conserved volatile budgets over the mass-scaled values the
    # structure solve wrote, so the growth adds rock, not volatiles.
    _restore_volatile_budgets(hf_row, volatile_budgets)

    # Apply the sized consequences to the whole-planet budgets and refresh
    # the tracked-element total the budgets aggregate into.
    _apply_volatile_consequences(hf_row, strip, delivered, impactor_lost, f_loss)

    # The ledger records the volatile change for the Zalmoxis target; not in the
    # init stage, which rebuilds the budgets from config this iteration.
    net_volatiles = 0.0
    if _tracks_volatile_mass(config) and not getattr(handler, 'init_stage', False):
        net_volatiles = sum(delivered.values()) - sum(strip.values())
    hf_row['M_accreted_net'] += net_volatiles

    # Raise the mantle to its initial condition; hotter parts keep their state.
    remelt_mantle(handler.directories, config, hf_row, handler.interior_o, event)

    # A mantle that had crystallised is now a magma ocean again, so lift the
    # one-way solidification latch; otherwise outgassing would stay frozen and
    # the volatiles would be treated as locked in a solid mantle for good.
    if getattr(handler, 'crystallized', False):
        handler.crystallized = False
        log.info('    solidification latch cleared: the mantle is molten again')

    # Clear the desiccation latch if volatiles were delivered to allow
    # outgassing to resume. The latch will re-evaluate on the next step.
    if delivered and getattr(handler, 'desiccated', False):
        handler.desiccated = False
        log.info('    desiccation latch cleared: the impact delivered volatiles')

    # Apply the impact's relative orbit change to the running row base,
    # updating config to reflect the new orbit and clamping eccentricity.
    base_a, base_e = _current_orbit(hf_row, config)

    requested = base_e + event.eccentricity_change
    eccentricity = min(max(requested, 0.0), _ECC_MAX)

    # A saturated clamp means the impact asked for an orbit the rest of the model
    # cannot represent, so report it rather than absorbing it. Clamping in silence
    # is how a compounding drift in the applied change hides for a whole run.
    if abs(requested - eccentricity) > 1e-12:
        log.warning(
            '    impact asked for eccentricity %.4f, clamped to %.4f: the change it '
            'applies (%+.4f) takes the orbit outside the representable range',
            requested,
            eccentricity,
            event.eccentricity_change,
        )

    new_a = base_a * ratio
    config.orbit.semimajoraxis = new_a / AU
    config.orbit.eccentricity = eccentricity
    hf_row['semimajorax'] = new_a
    hf_row['eccentricity'] = eccentricity

    log.info(
        '    planet is now %.4f M_earth at %.5f AU, e = %.4f',
        config.planet.mass_tot,
        config.orbit.semimajoraxis,
        config.orbit.eccentricity,
    )


def _tracks_volatile_mass(config: Config) -> bool:
    """Whether the ledger records volatile changes: only Zalmoxis solves a whole-planet target."""
    return config.interior_struct.module == 'zalmoxis'


def accreted_volatile_mass(hf_row: dict) -> float:
    """Volatile mass impacts and escape moved [kg], for the Zalmoxis whole-planet target.

    ``M_accreted_net`` less ``M_accreted_rock`` (delivered less stripped and
    escaped). Only accretion writes the ledger, so it is zero for a run that
    never had an accretion module and keeps applying after accretion is turned
    off on resume. ``mass_tot`` carries the rock.
    """
    return _net_ledger(hf_row) - float(hf_row.get('M_accreted_rock') or 0.0)


def _net_ledger(hf_row: dict) -> float:
    """Return the ``M_accreted_net`` ledger [kg], absent read as zero.

    Raises
    ------
    RuntimeError
        If the ledger is not finite, which would give the Zalmoxis target a
        non-finite volatile mass.
    """
    net = float(hf_row.get('M_accreted_net') or 0.0)
    if not math.isfinite(net):
        raise RuntimeError(
            f'M_accreted_net is not finite ({net!r}); the mass ledger is corrupt.'
        )
    return net


def debit_escaped_mass(config: Config, hf_row: dict, escaped: float) -> None:
    """Record the volatile mass escape removed in the ``M_accreted_net`` ledger.

    The Zalmoxis target is ``mass_tot`` plus the ledger's volatile part, less
    the volatiles its mantle EOS does not hold, so escaped volatiles left out
    of the ledger would come back as rock at the next structure solve. The
    debit includes any element the escape step set to zero below the
    outgassing threshold, which ``esc_kg_cumulative`` does not count.

    Parameters
    ----------
    config : Config
        Model configuration; read for the accretion and structure modules.
    hf_row : dict
        Current helpfile row; ``M_accreted_net`` is lowered in place.
    escaped : float
        Volatile mass the escape step removed from the element budgets [kg].
    """
    if (
        config.accretion.module is None
        or not _tracks_volatile_mass(config)
        or not 0.0 < escaped < math.inf
    ):
        return
    hf_row['M_accreted_net'] = _net_ledger(hf_row) - escaped


def _apply_volatile_consequences(
    hf_row: dict, strip: dict, delivered: dict, impactor_lost: dict, f_loss: float
) -> None:
    """Apply an impact's sized volatile changes to the whole-planet budgets.

    Debits the stripped target atmosphere from the whole-planet and the
    atmospheric budgets, books it into the escaped-mass ledger the
    desiccation gate audits, credits the delivered impactor volatiles to the
    budgets and to the gate's baseline ``M_vol_initial`` (once escape has
    set one), and refreshes the tracked-element total. The outgassing step
    later this iteration re-equilibrates the atmosphere against the updated
    totals; an element deferred to the chemistry step (e.g. oxygen under
    ic_chemistry) is re-derived there either way.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, mutated in place.
    strip, delivered, impactor_lost : dict
        Per-element masses [kg] sized from the pre-impact state.
    f_loss : float
        Collision loss fraction in [0, 1], reported in the strip log line.
    """
    # Debit the atmosphere too: escape on this step sizes its loss from it.
    for e in element_list:
        if e in strip:
            removed = strip[e]
            hf_row[f'{e}_kg_total'] = max(
                0.0, float(hf_row.get(f'{e}_kg_total', 0.0)) - removed
            )
            hf_row[f'{e}_kg_atm'] = max(0.0, float(hf_row.get(f'{e}_kg_atm', 0.0)) - removed)
    if strip:
        stripped_total = sum(strip.values())
        hf_row['esc_kg_cumulative'] = (
            float(hf_row.get('esc_kg_cumulative', 0.0)) + stripped_total
        )
        log.info(
            '    impact stripped %.1f%% of the atmosphere: %.3e kg removed',
            100.0 * f_loss,
            stripped_total,
        )
    for e, added in delivered.items():
        hf_row[f'{e}_kg_total'] = float(hf_row.get(f'{e}_kg_total', 0.0)) + added
    # Credit the escape-balance baseline, or the desiccation gate reads the
    # delivered mass as loss it may accept without escape.
    if (hf_row.get('M_vol_initial') or 0.0) > 0.0:
        hf_row['M_vol_initial'] += sum(delivered.values())
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

    # Refresh tracked-element total and whole-planet mass from updated budgets
    # so M_planet remains consistent with M_int + M_ele for subsequent modules.
    from proteus.interior_energetics.wrapper import update_planet_mass

    update_planet_mass(hf_row)


def _primordial_mass_fractions(hf_all, hf_row=None) -> dict:
    """Volatile mass fractions of the planet at formation [kg/kg].

    Reads the settled initial state from the run's own history: the last row
    of the init epoch (``Time < 1`` yr, the same discriminator the outgassing
    warm start uses), or the first row when no init row exists. When no history
    is available yet (step 0), falls back to the current step row.

    Parameters
    ----------
    hf_all : pd.DataFrame or None
        Full helpfile history of the run.
    hf_row : dict or None, optional
        Current step row, consulted when ``hf_all`` is None or empty.

    Returns
    -------
    dict
        Mapping of volatile element to ``<e>_kg_total / M_planet`` at the
        formation state.

    Raises
    ------
    RuntimeError
        If neither history nor step row is available, or the formation row
        carries no positive planet mass.
    """
    if hf_all is not None and len(hf_all) > 0:
        init_rows = hf_all[hf_all['Time'] < 1.0]
        t0 = init_rows.iloc[-1] if len(init_rows) else hf_all.iloc[0]
    elif hf_row is not None:
        t0 = hf_row
    else:
        raise RuntimeError(
            'Cannot scale impactor volatiles to the planet: no helpfile history '
            'is available to read the formation composition from.'
        )

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


def _impactor_volatile_content(config, hf_all, event: ImpactEvent, hf_row=None) -> dict:
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
        fractions = _primordial_mass_fractions(hf_all, hf_row=hf_row)
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


def _partition_impactor_content(
    config, hf_row: dict, content: dict, f_loss: float
) -> tuple[dict, dict]:
    """Split the impactor's volatiles into a delivered and a lost part [kg].

    The impactor's internal partitioning is unknowable, so the planet's own
    atmosphere-versus-interior split per element at impact time is mirrored
    onto it. The impactor's atmospheric part is then lost with the same
    collision loss fraction that strips the target's atmosphere, and the
    remainder of its content is delivered: a fast head-on impact loses
    nearly all of it, a slow grazing one delivers most of it, and with loss
    disabled the whole content arrives. The mirror understates a smaller
    body's atmospheric fraction (it equilibrates at lower surface
    pressure), so delivery is somewhat overestimated.

    For an element the planet no longer holds, the per-element mirror is
    undefined and the planet's bulk atmospheric fraction is used instead.

    Parameters
    ----------
    config : Config
        Model configuration; read for the loss-module switch.
    hf_row : dict
        Current helpfile row, supplying the partitioning mirror.
    content : dict
        Per-element volatile mass the impactor carries [kg].
    f_loss : float
        Collision loss fraction in [0, 1] applied to the atmospheric part.

    Returns
    -------
    (delivered, lost) : tuple of dict
        Per-element masses delivered into the planet and lost to space [kg].
    """
    if config.accretion.atmloss_module is None or f_loss <= 0.0:
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
        lost_e = mass * f_atm * f_loss
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


def _target_strip_amounts(config, hf_row: dict, f_loss: float) -> dict:
    """Mass the impact strips from the target's atmosphere, per element [kg].

    Sizes the debit from the pre-impact state without mutating it: each element
    loses the loss fraction of its own atmospheric mass, which is what
    partitioning the total stripped mass in proportion to the atmospheric
    abundances amounts to. The collision reaches only the atmosphere, so the
    per-element loss is capped at the whole-planet total as well, and the
    dissolved interior inventory is left intact.

    The debit is deliberately NOT routed through the continuous-escape path.
    That path applies a desiccation floor which zeroes an element's
    whole-planet total once it falls below the outgassing mass threshold, a
    reasonable convention for an element being ground down over many steps but
    wrong for a single collision: it would delete dissolved mantle inventory
    the impact never touched and book it as mass lost to space.

    An atmosphere below the outgassing mass threshold is treated as nothing to
    strip, the same convention continuous escape applies to it.

    Parameters
    ----------
    config : Config
        Model configuration; read for the outgassing mass threshold.
    hf_row : dict
        Current helpfile row, read only.
    f_loss : float
        Collision loss fraction in [0, 1] from :func:`_impact_loss_fraction`.
    """
    if f_loss <= 0.0:
        return {}

    m_atm = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
    if m_atm < config.outgas.mass_thresh:
        log.info(
            '    impact atmosphere loss: atmosphere below the mass threshold, not stripped'
        )
        return {}

    strip = {}
    for e in element_list:
        atm_e = float(hf_row.get(f'{e}_kg_atm', 0.0))
        removed = min(f_loss * atm_e, float(hf_row.get(f'{e}_kg_total', 0.0)))
        if removed > 0.0:
            strip[e] = removed
    return strip


# Atmosphere mass fraction above which the Kegerreis et al. (2020) erosion
# law leaves its fitted thin-atmosphere regime (of order 1 percent of the
# planet mass) far enough to warrant a warning.
_ATMLOSS_THIN_ATM_WARN = 0.03


def _impact_loss_fraction(config, hf_row: dict, event: ImpactEvent) -> float:
    """Fraction of the atmosphere removed by this impact [0-1].

    Dispatches on ``accretion.atmloss_module``. The constant module returns
    the configured fixed fraction; the zephyrus module evaluates the
    giant-impact erosion scaling law of Kegerreis et al. (2020) through
    ``zephyrus.collision.mass_loss``, fed entirely from the impact record so
    the speed, masses, radii, densities, and angle stay in the one frame the
    dynamical model produced them in (Morrigan bodies carry no modelled
    atmosphere, matching the law's atmosphere-excluded mass and radius
    convention, and its ``v_impact`` is the speed at first contact). The
    returned fraction applies to the target's atmosphere and to a
    volatile-bearing impactor's atmospheric part alike. PROTEUS itself ships
    no impact loss physics.

    When the zephyrus law is selected and the planet's atmosphere exceeds a
    few percent of its mass, the fitted thin-atmosphere regime no longer
    covers the impact and a warning is logged; the fraction is still
    returned, since staying inside the fitted domain is the run
    configuration's responsibility.

    Parameters
    ----------
    config : Config
        Model configuration; reads ``accretion.atmloss_module`` and
        ``accretion.atmloss_frac``.
    hf_row : dict
        Current helpfile row (the planet state the domain check reads).
    event : ImpactEvent
        The impact being applied (the collision parameters the law reads).

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
    ImportError
        If the zephyrus module is selected but the installed fwl-zephyrus
        does not provide the collision law.
    """
    atmloss_module = config.accretion.atmloss_module
    if atmloss_module is None:
        return 0.0

    match atmloss_module:
        case 'constant':
            f_loss = float(config.accretion.atmloss_frac)
        case 'zephyrus':
            try:
                from zephyrus.collision import mass_loss
            except ImportError as exc:
                raise ImportError(
                    "accretion.atmloss_module = 'zephyrus' needs a fwl-zephyrus "
                    'installation that provides zephyrus.collision; upgrade the '
                    'fwl-zephyrus package.'
                ) from exc

            m_atm = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
            m_planet = float(hf_row.get('M_planet', 0.0))
            if m_planet > 0.0 and m_atm / m_planet > _ATMLOSS_THIN_ATM_WARN:
                log.warning(
                    '    the atmosphere is %.1f%% of the planet mass, beyond the '
                    'thin-atmosphere regime (about 1%%) the impact erosion law is '
                    'fitted for; the eroded fraction is extrapolated',
                    100.0 * m_atm / m_planet,
                )

            f_loss = float(
                mass_loss(
                    v_c=event.v_impact,
                    M_i=event.M_impactor,
                    M_t=event.M_target_before,
                    rho_i=event.rho_impactor,
                    rho_t=event.rho_target,
                    R_i=event.R_impactor,
                    R_t=event.R_target_before,
                    b=event.impact_parameter,
                )
            )
            log.info('    impact erosion law: loss fraction %.3f', f_loss)
        case _:
            raise ValueError(f"Invalid accretion.atmloss_module: '{atmloss_module}'")

    if not 0.0 <= f_loss <= 1.0:
        raise ValueError(
            f'Impact atmosphere loss fraction must be in [0, 1], got {f_loss!r} '
            f"from atmloss_module '{atmloss_module}'"
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


def _drop_events_before_start(
    events: list[ImpactEvent], time_start: float, resumed: bool = False
) -> list[ImpactEvent]:
    """Remove impacts that precede the current point on the time axis.

    On a fresh run the configuration owns the planet's initial mass and orbit,
    so an impact landing before the run begins cannot be applied without
    contradicting it. Such impacts are reported rather than dropped in silence,
    since they usually mean the time offset needs adjusting.

    On a resume the same filter serves the opposite purpose: it removes impacts
    the earlier session already applied, whose mass the planet is carrying and
    whose rock is restored from the helpfile. Those are not missing from the
    run, so they are reported as already applied and the offset advice is
    withheld, because acting on it would apply them a second time.

    Parameters
    ----------
    events : list of ImpactEvent
        Timeline, in time order.
    time_start : float
        Simulation time at the start of the run [yr].
    resumed : bool
        Whether this run is resuming an earlier session.

    Returns
    -------
    kept : list of ImpactEvent
        Impacts after the current point on the time axis.
    """
    kept = [e for e in events if e.time > time_start]
    dropped = len(events) - len(kept)

    if dropped:
        missed_mass = sum(e.mass_delta for e in events if e.time <= time_start)
        if resumed:
            log.info(
                '%d impact(s) fall at or before the resume point (t = %.4e yr) and were '
                'applied by an earlier session, adding %.4f M_earth that the planet is '
                'already carrying.',
                dropped,
                time_start,
                missed_mass / M_earth,
            )
        else:
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
