# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.constants import element_list, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_escape(config:Config, hf_row:dict, dt:float):
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
    """

    if not config.escape.module:
        log.info(f"Escape is disabled, bulk rate = {hf_row["esc_rate_total"]:.2e} kg s-1")
        run_dummy(0.0, hf_row)
        return # return now, since elemental inventories will be unchanged

    elif config.escape.module == 'dummy':
        run_dummy(config.escape.dummy.rate, hf_row)
        calc_unfract_fluxes(hf_row, reservoir=config.escape.reservoir,
                                    min_thresh=config.outgas.mass_thresh)

    elif config.escape.module == 'zephyrus':
        run_zephyrus(config, hf_row)
        calc_unfract_fluxes(hf_row, reservoir=config.escape.reservoir,
                                    min_thresh=config.outgas.mass_thresh)

    elif config.escape.module == 'boreas':
        from proteus.escape.boreas import run_boreas
        run_boreas(config, hf_row)

    else:
        raise ValueError(f"Invalid escape model: {config.escape.module}")

    log.info(f"Bulk escape rate = {hf_row["esc_rate_total"]:.2e} kg s-1")

    # calculate new elemental inventories from loss over duration `dt`
    solvevol_target = calc_new_elements(hf_row, dt, config.outgas.mass_thresh)

    # store new elemental inventories
    for e in element_list:
        hf_row[e + "_kg_total"] = solvevol_target[e]

def run_dummy(Mdot: float, hf_row:dict):
    """Run dummy escape model.

    Uses a fixed mass loss rate and does not fractionate.

    Parameters
    ----------
        Mdot : float
            Bulk escape rate [kg s-1]
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    # Set bulk escape rate based on value from user
    hf_row["esc_rate_total"] = Mdot

    # Set sound speed to zero
    hf_row["cs_xuv"] = 0.0

    # Set Pxuv to Psurf
    hf_row["P_xuv"] = hf_row["P_surf"]

def run_zephyrus(config:Config, hf_row:dict)->float:
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

    hf_row["esc_rate_total"] = mlr
    hf_row["P_xuv"] = config.escape.zephyrus.Pxuv

def calc_unfract_fluxes(hf_row:dict, reservoir:str, min_thresh:float):
    """Calculate elemental escape rates, without fractionating.

    Updates the elemental escape fluxes in hf_row.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        reservoir: str
            Element reservoir representing the escaping composition (bulk, outgas, pxuv)
        min_thresh: float
            Minimum threshold for element mass [kg]. Inventories below this are set to zero.
    """

     # which reservoir?
    match reservoir:
        case "bulk":
            key = "_kg_total"
        case "outgas":
            key = "_kg_atm"
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
        return

    # calculate the current mass mixing ratio for each element
    #     if escape is unfractionating, this should be conserved
    for e in res.keys():
        hf_row[e+"_mmr_xuv"] = res[e]/M_vols
        log.debug("    %2s (%s) mass ratio = %.2e "%(e,reservoir,hf_row[e+"_mmr_xuv"]))

    # for each element, calculate new TOTAL mass inventory
    log.info("Elemental escape fluxes:")
    for e in res.keys():

        # elemental escape flux [kg/s]
        hf_row["esc_rate_"+e] = hf_row["esc_rate_total"] * hf_row[e+"_mmr_xuv"]

        # Print info
        if hf_row["esc_rate_"+e] > 1:
            log.info("    %2s = %.2e kg yr-1"%(e,hf_row["esc_rate_"+e]*secs_per_year))


def calc_new_elements(hf_row:dict, dt:float, min_thresh:float):
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

    # New target elemental inventory
    tgt = {}

    for e in element_list:
        # subtract lost mass from TOTAL mass of element e
        tgt[e] = hf_row[e+"_kg_total"] - hf_row["esc_rate_"+e] * dt * secs_per_year

        # below threshold, set to zero
        if tgt[e] < min_thresh:
            log.debug("    %2s total inventory below threshold, setting to 0 kg"%(e))
            tgt[e] = 0.0

        # do not allow negative masses
        tgt[e] = max(0.0, tgt[e])

    return tgt
