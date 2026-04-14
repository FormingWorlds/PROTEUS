# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.escape.common import calc_unfract_fluxes
from proteus.utils.constants import element_list, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def run_escape(
    config: Config,
    hf_row: dict,
    dirs: dict | None = None,
    dt: float = 0.0,
    stellar_track=None,
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
    """
    dirs = dirs or {}

    if not config.escape.module:
        # Keep a minimal, dependency-free disabled path for unit tests.
        hf_row['esc_rate_total'] = 0.0
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0
        log.info(f'Escape is disabled, bulk rate = {hf_row["esc_rate_total"]:.2e} kg s-1')
        return

    elif config.escape.module == 'dummy':
        run_dummy(config, hf_row)

    elif config.escape.module == 'zephyrus':
        run_zephyrus(config, hf_row, stellar_track)

    elif config.escape.module == 'boreas':
        from proteus.escape.boreas import run_boreas

        run_boreas(config, hf_row, dirs)

    else:
        raise ValueError(f'Invalid escape model: {config.escape.module}')

    log.info(f'Bulk escape rate = {hf_row["esc_rate_total"]:.2e} kg s-1')

    log.info('Elemental escape fluxes:')
    for e in element_list:
        esc_e = float(hf_row.get(f'esc_rate_{e}', 0.0))
        if esc_e > 0:
            log.info('    %2s = %.2e kg s-1' % (e, esc_e))

    # calculate new elemental inventories from loss over duration `dt`
    solvevol_target = calc_new_elements(
        hf_row,
        dt,
        config.escape.reservoir,
        min_thresh=config.outgas.mass_thresh,
    )

    # store new elemental inventories
    for e, mass in solvevol_target.items():
        hf_row[f'{e}_kg_total'] = mass


def run_dummy(config: Config, hf_row: dict):
    """Run dummy escape model.

    Uses a fixed mass loss rate and does not fractionate.

    Parameters
    ----------
        config : Config
            Configuration options for the escape module
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
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
        if isinstance(reservoir, str):
            calc_unfract_fluxes(
                hf_row,
                reservoir=reservoir,
                min_thresh=config.outgas.mass_thresh,
            )
    except (KeyError, ValueError, TypeError):
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0


def run_zephyrus(config: Config, hf_row: dict, stellar_track=None) -> float:
    """Run ZEPHYRUS escape model.

    Calculates the bulk mass loss rate of all elements.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    from zephyrus.escape import EL_escape

    log.info('Running EL escape (ZEPHYRUS) ...')

    # Compute energy-limited escape
    mlr = EL_escape(
        config.escape.zephyrus.tidal,  # tidal contribution (True/False)
        hf_row['semimajorax'],  # planetary semi-major axis [m]
        hf_row['eccentricity'],  # eccentricity
        hf_row['M_planet'],  # planetary mass [kg]
        config.star.mass,  # stellar mass [kg]
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
        if isinstance(reservoir, str):
            calc_unfract_fluxes(
                hf_row,
                reservoir=reservoir,
                min_thresh=config.outgas.mass_thresh,
            )
    except (KeyError, ValueError, TypeError):
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0

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
    match reservoir:
        case 'bulk':
            key = '_kg_total'
        case 'outgas':
            key = '_kg_atm'
        case 'pxuv':
            raise ValueError('Fractionation at p_xuv is not yet supported')
        case _:
            raise ValueError(f"Invalid escape reservoir '{reservoir}'")

    # calculate mass of elements in the reservoir
    res: dict[str, float] = {}
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        res[e] = float(hf_row.get(f'{e}{key}', 0.0))
    M_vols = float(sum(res.values()))

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
        new_total = old_total - lost
        if new_total < min_thresh:
            new_total = 0.0
        tgt[e] = max(0.0, new_total)

    return tgt
