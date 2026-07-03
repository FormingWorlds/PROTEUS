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


def run_escape(
    config: Config,
    hf_row: dict,
    dirs: dict | None = None,
    dt: float = 0.0,
    stellar_track=None,
    atmosphere_only: bool = False,
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
    """
    dirs = dirs or {}

    if not config.escape.module:
        # Keep a minimal, dependency-free disabled path for unit tests.
        hf_row['esc_rate_total'] = 0.0
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0
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

    # Accumulate cumulative escaped mass [kg] for the desiccation gate. This
    # is the integral of `esc_rate_total * dt` over all escape calls and is
    # persisted to the helpfile so it survives resume.
    esc_step_kg = float(hf_row.get('esc_rate_total', 0.0)) * secs_per_year * float(dt)
    if np.isfinite(esc_step_kg) and esc_step_kg > 0.0:
        hf_row['esc_kg_cumulative'] = float(hf_row.get('esc_kg_cumulative', 0.0)) + esc_step_kg

    # Reservoir the per-element loss is drawn from. With a solidified mantle
    # the atmosphere is the only escapable reservoir, so the loss is sized from
    # `*_kg_atm` (atmospheric abundance). This keeps the per-element `*_kg_total`
    # debit proportional to the atmosphere, matching the uniform atmospheric
    # scaling that `outgas.wrapper.run_crystallized` applies in the same step.
    reservoir = 'outgas' if atmosphere_only else config.escape.reservoir

    # calculate new elemental inventories from loss over duration `dt`
    solvevol_target = calc_new_elements(
        hf_row,
        dt,
        reservoir,
        min_thresh=config.outgas.mass_thresh,
    )

    # store new elemental inventories
    for e, mass in solvevol_target.items():
        hf_row[f'{e}_kg_total'] = mass


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

    Returns
    -------
        tgt : dict
            Volatile element whole-planet inventories [kg]
    """
    # which reservoir?

    log.info(f'Calculating new elemental inventories from escape, reservoir = {reservoir}')

    match reservoir:
        case 'bulk':
            key = '_kg_total'
        case 'outgas':
            key = '_kg_atm'
        case 'pxuv':
            raise ValueError('Fractionation at p_xuv is not yet supported')
        case _:
            raise ValueError(f"Invalid escape reservoir '{reservoir}'")

    # Calculate mass of elements in the reservoir. Issue #677 fix:
    # include O so the per-element subtraction sums to esc_mass and
    # the planetary O budget responds to escape (CALLIOPE's next call
    # may overwrite O_kg_total, but the bulk MLR is now attributed to
    # all elements proportionally rather than concentrated on H+C+N+S).
    res: dict[str, float] = {}
    for e in element_list:
        res[e] = float(hf_row.get(f'{e}{key}', 0.0))

    log.debug(
        'Total mass of hydrogen in atmosphere before escape: %e kg'
        % hf_row.get('H_kg_atm', 0.0)
    )
    log.debug(
        'Total mass of hydrogen in interior before escape: %e kg'
        % hf_row.get('H_kg_liquid', 0.0)
    )

    M_vols = float(sum(res.values()))
    for e in element_list:
        if e == 'O':
            continue
        else:
            log.debug('mass ratio of element %s %.4f', e, (res[e] / M_vols))

    # check if we just desiccated the planet...
    if M_vols < min_thresh:
        log.debug('    Total mass of volatiles below threshold in escape calculation')
        return res

    # compute mass ratios in escaping reservoir
    emr = {e: (res[e] / M_vols if M_vols > 0 else 0.0) for e in res}

    # total escaped mass over dt [kg]
    esc_mass = float(hf_row.get('esc_rate_total', 0.0)) * secs_per_year * float(dt)

    # compute new TOTAL inventories
    tgt: dict[str, float] = {}
    for e in res:
        lost = esc_mass * emr[e]
        old_total = float(hf_row.get(f'{e}_kg_total', 0.0))
        log.info(e)
        log.info('element total before escape: %s' % old_total)
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
