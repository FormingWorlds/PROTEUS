# Functions used to handle escape

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

log = logging.getLogger("PROTEUS")


def RunDummyEsc(hf_row:dict, dt:float):
    """Run dummy escape model.

    Parameters
    ----------
        hf_row : dict 
            Dictionary of helpfile variables, at this iteration only
        dt : float 
            Time interval over which escape is occuring [yr]

    Returns
    ----------
        esc_result : dict 
            Dictionary of elemental mass deltas [kg]

    """
    log.info("Running dummy escape...")

    # Hardcoded dummy value of bulk volatile escape rate [kg/s]
    phi = 2e7

    # store value
    out = {}
    out["rate_bulk"] = phi

    # calculate total mass of volatiles
    M_vols = 0.0
    for e in element_list:
        if e=='O': continue 
        M_vols += hf_row[e+"_kg_total"]

    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio
    for e in element_list:
        if e=='O': continue

        # current elemental mass ratio in atmosphere 
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new atmosphere mass of element e, keeping a constant mixing ratio of that element 
        e_atm = emr * (M_vols - phi * dt * secs_per_year)

        # calculate change in total mass of element e
        out[e+"_dm"] = e_atm - hf_row[e+"_kg_total"]

    return out


def RunZEPHYRUS():
    """Zephyrus wrapper 

    Not yet implemented.

    """
    log.info("Running ZEPHYRUS...")
    raise Exception("Not yet implemented")


