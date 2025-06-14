# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.constants import element_list, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_escape(config:Config, hf_row:dict, dt:float, stellar_track):
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
        hf_row["esc_rate_total"] = run_zephyrus(config, hf_row, stellar_track)

    elif config.escape.module == 'dummy':
        hf_row["esc_rate_total"] = config.escape.dummy.rate

    else:
        raise ValueError(f"Invalid escape model: {config.escape.module}")

    log.info(
        "Bulk escape rate: %.2e kg yr-1 = %.2e kg s-1"
        % (hf_row["esc_rate_total"] * secs_per_year, hf_row["esc_rate_total"])
        )

    # calculate new elemental inventories
    solvevol_target = calc_new_elements(hf_row, dt,
                                            config.escape.reservoir,
                                            min_thresh=config.outgas.mass_thresh)

    # store new elemental inventories
    for e in element_list:
        if e == "O":
            continue
        hf_row[e + "_kg_total"] = solvevol_target[e]

def run_zephyrus(config, hf_row, stellar_track):
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

    # Compute energy-limited escape
    mlr = EL_escape(config.escape.zephyrus.tidal, #tidal contribution (True/False)
                    hf_row["semimajorax"], #planetary semi-major axis [m]
                    hf_row["eccentricity"], #eccentricity
                    hf_row["M_planet"], #planetary mass [kg]
                    config.star.mass, #stellar mass [kg]
                    config.escape.zephyrus.efficiency, #efficiency factor
                    hf_row["R_int"],    # planetary radius [m]
                    hf_row["R_xuv"],    # XUV optically thick planetary radius [m]
                    hf_row["F_xuv"],    # [W m-2]
                    scaling = 3)

    return mlr

def calc_new_elements(hf_row:dict, dt:float, reservoir:str, min_thresh:float=1e10):
    """Calculate new elemental inventory based on escape rate.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time-step length [years]
        reservoir: str
            Element reservoir representing the escaping composition (bulk, outgas, pxuv)
        min_thresh: float
            Minimum threshold for element mass [kg]. Inventories below this are set to zero.

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

    # calculate mass of volatile elements in reservoir (except oxygen, which is set by fO2)
    res = {}
    for e in element_list:
        if e=='O':
            continue
        res[e] = hf_row[e+key]
    M_vols = sum(list(res.values()))

    # check if we just desiccated the planet...
    if M_vols < min_thresh:
        log.debug("    Total mass of volatiles below threshold in escape calculation")
        return res

    # calculate the current mass mixing ratio for each element
    #     if escape is unfractionating, this should be conserved
    emr = {}
    for e in res.keys():
        emr[e] = res[e]/M_vols
        log.debug("    %2s (%s) mass ratio = %.2e "%(e,reservoir,emr[e]))

    # for each element, calculate new TOTAL mass inventory
    tgt = {}
    log.info("Elemental escape fluxes:")
    for e in res.keys():

        # elemental escape flux [kg/year]
        esc_rate_elem = hf_row["esc_rate_total"] * emr[e] * secs_per_year
        if esc_rate_elem > 1:
            log.info("    %2s = %.2e kg yr-1"%(e,esc_rate_elem))

        # subtract lost mass from TOTAL mass of element e
        tgt[e] = hf_row[e+"_kg_total"] - esc_rate_elem * dt

        # below threshold, set to zero
        if tgt[e] < min_thresh:
            log.debug("    %2s total inventory below thresh, set to zero"%(e))
            tgt[e] = 0.0

        # do not allow negative masses
        tgt[e] = max(0.0, tgt[e])

    return tgt
