# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import element_list, ergcm2stoWm2, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def RunEscape(config:Config, hf_row:dict, dt:float, stellar_track):
    """Run Escape submodule.

    Generic function to run escape calculation using ZEPHYRUS or dummy.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time interval over which escape is occuring [yr]
        stellar_track : mors star object
            Mors star object storing spada track data.
    """

    if not config.escape.module:
        # solvevol_target is undefined?
        pass
    elif config.escape.module == 'zephyrus':
        hf_row["esc_rate_total"] = RunZEPHYRUS(config, hf_row, stellar_track)
    elif config.escape.module == 'dummy':
        hf_row["esc_rate_total"] = config.escape.dummy.rate
    else:
        raise ValueError(f"Invalid escape model: {config.escape.module}")

    log.info(
        "Bulk escape rate: %.2e kg yr-1 = %.2e kg s-1"
        % (hf_row["esc_rate_total"] * secs_per_year, hf_row["esc_rate_total"])
        )

    # calculate new elemental inventories
    solvevol_target = calc_new_elements(hf_row, dt, config.escape.reservoir)

    # store new elemental inventories
    for e in element_list:
        if e == "O":
            continue
        hf_row[e + "_kg_total"] = solvevol_target[e]

def RunZEPHYRUS(config, hf_row, stellar_track):
    """Run energy-limited escape (for now) model.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        stellar_track : mors star object
            Mors star object storing spada track data.
    Returns
    -------
        mlr : float
            Bulk escape rate [kg s-1]
    """

    from zephyrus.escape import EL_escape

    log.info("Running EL escape (ZEPHYRUS) ...")

    # Get the age of the star at time t to compute XUV flux at that time
    age_star = hf_row["age_star"] / 1e6 # [Myrs]

    # Interpolating the XUV flux at the age of the star
    Fxuv_star_SI = ((stellar_track.Value(age_star, 'Lx') + stellar_track.Value(age_star, 'Leuv'))
                             / (4 * np.pi * (hf_row["semimajorax"] * 1e2)**2)) * ergcm2stoWm2

    log.info("    age_star     = %.1e Myr"%age_star)
    log.info("    Fxuv_star_SI = %.1e W m-2"%Fxuv_star_SI)

    # Compute energy-limited escape
    mlr = EL_escape(config.escape.zephyrus.tidal, #tidal contribution (True/False)
                    hf_row["semimajorax"], #planetary semi-major axis [m]
                    hf_row["eccentricity"], #eccentricity
                    hf_row["M_planet"], #planetary mass [kg]
                    config.star.mass, #stellar mass [kg]
                    config.escape.zephyrus.efficiency, #efficiency factor
                    hf_row["R_int"], #planetary radius [m]
                    hf_row["R_xuv"], #XUV optically thick planetary radius [m]
                    Fxuv_star_SI,   # [kg s-1]
                    scaling = 3)

    return mlr

def calc_new_elements(hf_row:dict, dt:float, reservoir:str):
    """Calculate new elemental inventory based on escape rate.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time-step length [years]
        reservoir: str
            Element reservoir representing the escaping composition (bulk, outgas, pxuv)

    Returns
    -------
        tgt : dict
            Volatile element whole-planet inventories [kg]
    """

    # which reservoir?
    match reservoir:
        case "bulk":
            key = "_kg_total"
        case "outgas":
            key = "_kg_atm"
        case "pxuv":
            raise ValueError("Fractionation at p_xuv is not yet supported")
        case _:
            raise ValueError(f"Invalid escape reservoir '{reservoir}'")

    # calculate total mass of volatile elements (except oxygen, which is set by fO2)
    tgt = {}
    for e in element_list:
        if e=='O':
            continue
        tgt[e] = hf_row[e+key]
    M_vols = sum(list(tgt.values()))

    # calculate the current mass mixing ratio for each element
    #     if escape is unfractionating, this should be conserved
    emr = {}
    for e in element_list:
        if e=='O':
            continue
        emr[e] = tgt[e]/M_vols
        log.debug("    %s (%s) mass ratio = %.2e "%(e,reservoir,emr[e]))

    # for each element, calculate new whole-planet mass inventory
    for e in element_list:
        if e=='O':
            continue

        # subtract lost mass from total mass of element e
        tgt[e] = tgt[e] - hf_row["esc_rate_total"] * emr[e] * dt * secs_per_year

        # do not allow negative masses
        tgt[e] = max(0.0, tgt[e])

    return tgt
