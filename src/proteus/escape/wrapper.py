# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.escape.common import calc_unfract_fluxes
from proteus.utils.constants import M_sun, element_list, noble_gases, secs_per_year
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Largest share of the escapable reservoir a single step may remove. Measured
# across 835606 escape steps in 470 grid cases: the median step loses 1.9e-05 of
# the reservoir and the p99 6.4e-03, so 0.25 binds on 0.044 % of steps.
ESCAPE_STEP_MAX_FRAC = 0.25


def reservoir_key(reservoir: str) -> str:
    """Return the helpfile suffix naming the reservoir escape draws from.

    Parameters
    ----------
        reservoir : str
            Reservoir name from ``config.escape.reservoir``.

    Returns
    -------
        key : str
            Helpfile column suffix, appended to an element symbol.

    Raises
    ------
        ValueError
            If the reservoir is unsupported or unrecognised.
    """
    match reservoir:
        case 'bulk':
            return '_kg_total'
        case 'outgas':
            return '_kg_atm'
        case 'pxuv':
            raise ValueError('Fractionation at p_xuv is not yet supported')
        case _:
            raise ValueError(f"Invalid escape reservoir '{reservoir}'")


def escapable_mass(hf_row: dict, reservoir: str) -> float:
    """Return the mass escape can draw on this step [kg].

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        reservoir : str
            Reservoir name from ``config.escape.reservoir``.

    Returns
    -------
        mass : float
            Summed elemental mass held in that reservoir [kg].
    """
    key = reservoir_key(reservoir)
    return float(sum(float(hf_row.get(f'{e}{key}', 0.0)) for e in element_list))


def limit_escape_step(
    hf_row: dict,
    dt: float,
    reservoir: str,
    max_frac: float = ESCAPE_STEP_MAX_FRAC,
) -> float:
    """Cap the mass a single escape step may remove, and record what it asked for.

    The bulk rate is set without reference to how much mass is actually left, so
    over a long step it can ask for many times the escapable reservoir. Nothing
    downstream catches that: with ``reservoir = "outgas"`` the per-element ratios
    come from ``*_kg_atm`` while the debit lands on ``*_kg_total``, so the
    non-negativity clamp in :func:`calc_new_elements` never fires and interior
    inventory is drained without a trace.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only. Gains
            ``esc_clamp_frac``, the requested loss as a fraction of the
            reservoir, which stays above ``max_frac`` on a capped step.
        dt : float
            Time interval over which escape is occuring [yr]
        reservoir : str
            Reservoir the per-element loss is drawn from.
        max_frac : float
            Largest share of the reservoir one step may remove.

    Returns
    -------
        esc_mass : float
            Mass to remove over this step [kg], capped at ``max_frac`` of the
            reservoir.
    """
    requested = float(hf_row.get('esc_rate_total', 0.0)) * secs_per_year * float(dt)
    if not np.isfinite(requested) or requested <= 0.0:
        hf_row['esc_clamp_frac'] = 0.0
        return 0.0

    escapable = escapable_mass(hf_row, reservoir)
    if not np.isfinite(escapable):
        # An upstream failure has left the reservoir unreadable. Sizing a loss
        # against it would spread the non-finite value through every inventory.
        hf_row['esc_clamp_frac'] = 0.0
        log.warning(
            'Escapable reservoir is not finite, so this step removes nothing; '
            'the inventories it is drawn from need checking.'
        )
        return 0.0
    if escapable <= 0.0:
        # Nothing left to draw on, so nothing can leave.
        hf_row['esc_clamp_frac'] = 0.0
        log.debug('Escapable reservoir is empty, so this step removes nothing')
        return 0.0

    frac = requested / escapable
    hf_row['esc_clamp_frac'] = frac
    if frac <= max_frac:
        return requested

    allowed = max_frac * escapable
    log.warning(
        'Escape asked for %.3e kg this step, %.3gx the escapable reservoir of '
        '%.3e kg; the loss is capped at %.3e kg, a fraction %.3g of it.',
        requested,
        frac,
        escapable,
        allowed,
        max_frac,
    )
    return allowed


def escape_dt_limit(
    clamp_frac: float,
    dt: float,
    max_frac: float = ESCAPE_STEP_MAX_FRAC,
) -> float:
    """Return the largest next step [yr] that would not repeat this overshoot.

    The cap bounds the loss on the step that overshot, but leaves the step
    length that produced it unchanged, so the same rate asks for the same
    excess again. Shortening the next step by the overshoot ratio brings the
    request back to the cap, measured against the reservoir the step started
    from.

    That step then removes ``max_frac`` of that reservoir, so measured against
    what remains an unchanged rate settles at ``max_frac / (1 - max_frac)``,
    still above the cap. The cap therefore keeps binding and the step keeps
    shortening while the rate holds, which runs a reservoir down over many
    steps rather than emptying it in one. The step stops shortening at the
    floor set by ``escape.step_dt_floor_frac``.

    Parameters
    ----------
        clamp_frac : float
            Requested loss as a fraction of the escapable reservoir, as
            recorded by :func:`limit_escape_step`.
        dt : float
            Length of the step that produced the request [yr].
        max_frac : float
            Largest share of the reservoir one step may remove.

    Returns
    -------
        dt_limit : float
            Step length that places the same rate at ``max_frac`` of the
            reservoir this step drew on [yr]. Infinite when the request was
            within the cap, so no limit applies.
    """
    if not np.isfinite(clamp_frac) or clamp_frac <= max_frac:
        return float('inf')
    if not np.isfinite(dt) or dt <= 0.0:
        return float('inf')
    return dt * max_frac / clamp_frac


def run_escape(
    config: Config,
    hf_row: dict,
    dirs: dict | None = None,
    dt: float = 0.0,
    stellar_track=None,
    atmosphere_only: bool = False,
    interior_o=None,
) -> None:
    """Run Escape submodule.

    Generic function to run escape calculation using ZEPHYRUS or dummy.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dirs: dict
            Dictionary of directories
        dt : float
            Time interval over which escape is occuring [yr]
        atmosphere_only : bool
            If True, size the per-element loss from the atmospheric reservoir
            regardless of ``config.escape.reservoir``. Set once the mantle has
            solidified: dissolved volatiles are then frozen into the solid and
            the atmosphere is the only reservoir that can supply escape.
        interior_o : Interior_t | None
            Interior state. When given, its ``escape_dt_limit`` is set so a
            capped step shortens the next one; see :func:`escape_dt_limit`.
    """
    dirs = dirs or {}

    if not config.escape.module:
        # Keep a minimal, dependency-free disabled path for unit tests. The
        # per-step records are cleared alongside the rates so a step that ran
        # before escape was switched off cannot read as the current one.
        hf_row['esc_rate_total'] = 0.0
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0
        hf_row['esc_clamp_frac'] = 0.0
        hf_row['esc_step_kg'] = 0.0
        if interior_o is not None:
            interior_o.escape_dt_limit = float('inf')
        log.info(f'Escape is disabled, bulk rate = {hf_row["esc_rate_total"]:.2e} kg s-1')
        return

    # Snapshot the initial bulk volatile inventory on the first escape call.
    # This baseline is used by `outgas.wrapper.check_desiccation` to verify
    # that any subsequent collapse of `*_kg_total` is actually accounted for
    # by cumulative escape, rather than by an upstream AGNI/outgas failure
    # zeroing the atmosphere as a side effect (CHILI sweep R7/R21 cascade).
    m_init_prev = hf_row.get('M_vol_initial', None)
    try:
        m_init_prev_f = float(m_init_prev) if m_init_prev is not None else 0.0
    except (TypeError, ValueError):
        m_init_prev_f = 0.0
    if not np.isfinite(m_init_prev_f) or m_init_prev_f <= 0.0:
        # Issue #677 fix: include O in the baseline. The desiccation gate
        # compares (M_vol_initial - cur_m_ele) against 1.5 * esc_kg_cumulative;
        # for the comparison to be consistent, the baseline and cur_m_ele
        # must both account for O loss (calc_new_elements now debits
        # O_kg_total, escape/common.py now produces esc_rate_O, so esc_kg_cum
        # implicitly carries the O contribution).
        m_vol_baseline = sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list)
        hf_row['M_vol_initial'] = m_vol_baseline
        # Reset the cumulative escape counter alongside the baseline so the
        # ratio (lost vs escaped) starts from a consistent zero.
        hf_row['esc_kg_cumulative'] = 0.0

    if config.escape.module == 'dummy':
        run_dummy(config, hf_row, atmosphere_only=atmosphere_only)

    elif config.escape.module == 'zephyrus':
        run_zephyrus(config, hf_row, stellar_track, atmosphere_only=atmosphere_only)

    elif config.escape.module == 'boreas':
        from proteus.escape.boreas import run_boreas

        run_boreas(config, hf_row, dirs)

    else:
        if dirs.get('output'):
            UpdateStatusfile(dirs, 20)
        raise ValueError(f'Invalid escape model: {config.escape.module}')

    log.info(f'Bulk escape rate = {hf_row["esc_rate_total"]:.2e} kg s-1')

    log.info('Elemental escape fluxes:')
    for e in element_list:
        esc_e = float(hf_row.get(f'esc_rate_{e}', 0.0))
        if esc_e > 0:
            log.info('    %2s = %.2e kg s-1' % (e, esc_e))

    # Reservoir the per-element loss is drawn from. With a solidified mantle
    # the atmosphere is the only escapable reservoir, so the loss is sized from
    # `*_kg_atm` (atmospheric abundance). This keeps the per-element `*_kg_total`
    # debit proportional to the atmosphere, matching the uniform atmospheric
    # scaling that `outgas.wrapper.run_crystallized` applies in the same step.
    reservoir = 'outgas' if atmosphere_only else config.escape.reservoir

    max_frac = float(config.escape.step_max_frac)

    # Cap what one step may remove before anything acts on it.
    esc_step_kg = limit_escape_step(hf_row, dt, reservoir, max_frac=max_frac)

    # Ask the interior for a shorter next step when the cap bound this one, so
    # the same rate stops re-requesting the same excess. Set on every call, so
    # a step that stays within the cap clears any limit left by an earlier one.
    if interior_o is not None:
        interior_o.escape_dt_limit = escape_dt_limit(
            float(hf_row.get('esc_clamp_frac', 0.0)), dt, max_frac=max_frac
        )

    before_kg = sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list)

    # calculate new elemental inventories from loss over duration `dt`
    solvevol_target = calc_new_elements(
        hf_row,
        dt,
        reservoir,
        min_thresh=config.outgas.mass_thresh,
        esc_mass=esc_step_kg,
    )

    # store new elemental inventories
    for e, mass in solvevol_target.items():
        hf_row[f'{e}_kg_total'] = mass

    # The mass that actually left. Measured from the inventories, not the
    # request, because the threshold gate can decline to debit; bounded by the
    # applied loss, because that same gate also zeroes an element under the
    # threshold and escape must not be credited with the truncation.
    drop_kg = before_kg - sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list)
    # Test both operands, not the result: `min` returns whichever argument comes
    # first when the other is not a number, so a non-finite one would survive.
    if np.isfinite(drop_kg) and np.isfinite(esc_step_kg):
        removed_kg = max(0.0, min(drop_kg, esc_step_kg))
    else:
        removed_kg = 0.0

    # Publish it so every consumer of this step works from the same mass.
    # `outgas.wrapper.run_crystallized` rescales the atmosphere by this value,
    # so it has to be the loss the inventories took rather than the one the
    # rate asked for: a declined debit that still drained the atmosphere would
    # part the two records of the same step.
    hf_row['esc_step_kg'] = removed_kg
    if removed_kg > 0.0:
        hf_row['esc_kg_cumulative'] = float(hf_row.get('esc_kg_cumulative', 0.0)) + removed_kg


def run_dummy(config: Config, hf_row: dict, atmosphere_only: bool = False):
    """Run dummy escape model.

    Uses a fixed mass loss rate and does not fractionate.

    Parameters
    ----------
        config : Config
            Configuration options for the escape module
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        atmosphere_only : bool
            If True, size the per-element fluxes from the atmospheric reservoir
            (used once the mantle has solidified).
    """

    # Set sound speed to zero
    hf_row['cs_xuv'] = 0.0

    # Set Pxuv to Psurf (if available)
    if 'P_surf' in hf_row:
        hf_row['p_xuv'] = hf_row['P_surf']
    if 'R_int' in hf_row:
        hf_row['R_xuv'] = hf_row['R_int']

    # Set bulk escape rate based on value from user
    hf_row['esc_rate_total'] = float(
        getattr(getattr(config.escape, 'dummy', None), 'rate', 0.0)
    )

    # Always unfractionating (best-effort: unit tests may not populate all keys)
    try:
        reservoir = getattr(config.escape, 'reservoir', None)
        if atmosphere_only and isinstance(reservoir, str):
            reservoir = 'outgas'
        if isinstance(reservoir, str):
            calc_unfract_fluxes(
                hf_row,
                reservoir=reservoir,
                min_thresh=config.outgas.mass_thresh,
            )
    except (KeyError, ValueError, TypeError) as exc:
        # calc_unfract_fluxes needs a fully-populated hf_row; a partial row
        # (e.g. in a unit test) can raise. Zero the per-element rates to keep
        # the row well-formed, but warn: esc_rate_total is left unchanged, so
        # sum(esc_rate_e) no longer matches it.
        log.warning(
            'calc_unfract_fluxes failed (%s); zeroing per-element escape rates. '
            'esc_rate_total=%.3e is unchanged, so the per-element sum no longer '
            'matches it.',
            exc,
            float(hf_row.get('esc_rate_total', 0.0)),
        )
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0


def run_zephyrus(
    config: Config, hf_row: dict, stellar_track=None, atmosphere_only: bool = False
) -> float:
    """Run ZEPHYRUS escape model.

    Calculates the bulk mass loss rate of all elements.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        atmosphere_only : bool
            If True, size the per-element fluxes from the atmospheric reservoir
            (used once the mantle has solidified).
    """

    from zephyrus.escape import EL_escape

    # Compute energy-limited escape
    mlr = EL_escape(
        config.escape.zephyrus.tidal,  # tidal contribution (True/False)
        hf_row['semimajorax'],  # planetary semi-major axis [m]
        hf_row['eccentricity'],  # eccentricity
        hf_row['M_planet'],  # planetary mass [kg]
        config.star.mass * M_sun,  # stellar mass [kg] (config is in M_sun)
        config.escape.zephyrus.efficiency,  # efficiency factor
        hf_row['R_int'],  # planetary radius [m]
        hf_row['R_xuv'],  # XUV optically thick planetary radius [m]
        hf_row['F_xuv'],  # [W m-2]
        scaling=3,
    )

    hf_row['esc_rate_total'] = mlr
    hf_row['p_xuv'] = config.escape.zephyrus.Pxuv
    hf_row['R_xuv'] = 0.0  # to be calc'd by atmosphere module

    # Always unfractionating - escaping in bulk
    try:
        reservoir = getattr(config.escape, 'reservoir', None)
        if atmosphere_only and isinstance(reservoir, str):
            reservoir = 'outgas'
        if isinstance(reservoir, str):
            calc_unfract_fluxes(
                hf_row,
                reservoir=reservoir,
                min_thresh=config.outgas.mass_thresh,
            )
    except (KeyError, ValueError, TypeError) as exc:
        # calc_unfract_fluxes needs a fully-populated hf_row; a partial row
        # (e.g. in a unit test) can raise. Zero the per-element rates to keep
        # the row well-formed, but warn: esc_rate_total is left unchanged, so
        # sum(esc_rate_e) no longer matches it.
        log.warning(
            'calc_unfract_fluxes failed (%s); zeroing per-element escape rates. '
            'esc_rate_total=%.3e is unchanged, so the per-element sum no longer '
            'matches it.',
            exc,
            float(hf_row.get('esc_rate_total', 0.0)),
        )
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0

    log.debug(f'escape rate = {mlr}')
    return float(mlr)


def calc_new_elements(
    hf_row: dict,
    dt: float,
    reservoir: str,
    min_thresh: float = 1e10,
    esc_mass: float | None = None,
):
    """Calculate new elemental inventory based on escape rate.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time-step length [years]
        min_thresh: float
            Minimum threshold for element mass [kg]. Inventories below this are set to zero.
        esc_mass : float | None
            Mass to remove over this step [kg]. Defaults to the unrestricted
            ``esc_rate_total * dt``; pass the value from
            :func:`limit_escape_step` to apply the per-step cap.

    Returns
    -------
        tgt : dict
            Volatile element whole-planet inventories [kg]
    """
    # which reservoir?

    log.info(f'Calculating new elemental inventories from escape, reservoir = {reservoir}')

    key = reservoir_key(reservoir)

    # Calculate mass of elements in the reservoir. Issue #677 fix:
    # include O so the per-element subtraction sums to esc_mass and
    # the planetary O budget responds to escape (CALLIOPE's next call
    # may overwrite O_kg_total, but the bulk MLR is now attributed to
    # all elements proportionally rather than concentrated on H+C+N+S).
    res: dict[str, float] = {}
    for e in element_list:
        res[e] = float(hf_row.get(f'{e}{key}', 0.0))

    M_vols = float(sum(res.values()))
    # Nothing to share out, either because the reservoir is spent or because an
    # upstream failure left it unreadable. Return the totals unchanged: `res`
    # holds the reservoir the loss is sized from, which is `*_kg_atm` for
    # `outgas`, so returning it would overwrite the whole-planet inventory with
    # the atmospheric one and erase volatiles still held in the mantle.
    if not np.isfinite(M_vols):
        log.warning(
            'Volatile reservoir is not finite in the escape calculation, so no '
            'inventory is debited this step; the reservoir needs checking.'
        )
        return {e: float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list}
    if M_vols < min_thresh:
        log.debug('Total mass of volatiles below threshold in escape calculation')
        return {e: float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list}

    # compute mass ratios in escaping reservoir.
    # With `outgas.vapourise=True` the rock-forming elements can take a share of
    # the outflow, diluting what is available to H/C/N/O/S. This depends on the
    # selected reservoir (outgas vs bulk).
    emr = {e: (res[e] / M_vols if M_vols > 0 else 0.0) for e in res}

    # total escaped mass over dt [kg]
    if esc_mass is None:
        esc_mass = float(hf_row.get('esc_rate_total', 0.0)) * secs_per_year * float(dt)
    esc_mass = float(esc_mass)

    # compute new TOTAL inventories
    tgt: dict[str, float] = {}
    for e in res:
        lost = esc_mass * emr[e]
        old_total = float(hf_row.get(f'{e}_kg_total', 0.0))
        if not np.isfinite(old_total):
            # Leave it as it stands. Clamping below would turn it into a zero
            # that reads as an element escape has depleted, hiding whatever
            # upstream failure produced it.
            log.warning(
                'Whole-planet inventory of %s is not finite, so escape leaves '
                'it untouched this step; it needs checking upstream.',
                e,
            )
            tgt[e] = old_total
            continue
        new_total = old_total - lost
        # The desiccation floor treats a major volatile that drops below
        # min_thresh as fully depleted. Noble gases are intrinsically trace
        # (Earth-like whole-planet inventories sit orders of magnitude below
        # min_thresh), so applying the same absolute floor would zero a
        # realistic noble inventory on the first escape step. Exempt them and
        # only clamp to non-negative.
        if e not in noble_gases and new_total < min_thresh:
            new_total = 0.0
        tgt[e] = max(0.0, new_total)

    return tgt
