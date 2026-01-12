# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.escape.common import calc_unfract_fluxes
from proteus.utils.constants import element_list, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_escape(config:Config, hf_row:dict, dirs:dict, dt:float):
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

    if not config.escape.module:
        log.info(f"Escape is disabled, bulk rate = {hf_row["esc_rate_total"]:.2e} kg s-1")
        run_dummy(config, hf_row)
        return # return now, since elemental inventories will be unchanged

    elif config.escape.module == 'dummy':
        run_dummy(config, hf_row)

    elif config.escape.module == 'zephyrus':
        run_zephyrus(config, hf_row)

    elif config.escape.module == 'boreas':
        from proteus.escape.boreas import run_boreas
        run_boreas(config, hf_row, dirs)

    else:
        raise ValueError(f"Invalid escape model: {config.escape.module}")

    log.info(f"Bulk escape rate = {hf_row["esc_rate_total"]:.2e} kg s-1")

    log.info("Elemental escape fluxes:")
    for e in element_list:
        if hf_row["esc_rate_"+e] > 0:
            log.info("    %2s = %.2e kg s-1"%(e,hf_row["esc_rate_"+e]))

    # calculate new elemental inventories from loss over duration `dt`
    solvevol_target = calc_new_elements(hf_row, dt, config.outgas.mass_thresh)

    # store new elemental inventories
    for e in element_list:
        hf_row[e + "_kg_total"] = solvevol_target[e]

def run_dummy(config:Config, hf_row:dict):
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
    hf_row["cs_xuv"] = 0.0

    # Set Pxuv to Psurf
    hf_row["p_xuv"] = hf_row["P_surf"]
    hf_row["R_xuv"] = hf_row["R_int"]

    # Set bulk escape rate based on value from user
    if not config.escape.module:
        hf_row["esc_rate_total"] = 0.0
        for e in element_list:
            if e != 'O':
                hf_row[f"esc_rate_{e}"] = 0.0

    else:
        hf_row["esc_rate_total"] = config.escape.dummy.rate

        # Always unfractionating
        calc_unfract_fluxes(hf_row, reservoir=config.escape.reservoir,
                                    min_thresh=config.outgas.mass_thresh)

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
    hf_row["p_xuv"] = config.escape.zephyrus.Pxuv
    hf_row["R_xuv"] = 0.0  # to be calc'd by atmosphere module

    # Always unfractionating - escaping in bulk
    calc_unfract_fluxes(hf_row, reservoir=config.escape.reservoir,
                                    min_thresh=config.outgas.mass_thresh)

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
