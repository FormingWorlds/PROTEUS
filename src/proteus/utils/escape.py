# Functions used to handle escape

from proteus.utils.modules_ext import *
from proteus.utils.constants import *
from proteus.utils.helper import *

log = logging.getLogger("PROTEUS")


def RunDummyEsc(hf_row:dict, dt:float, phi_bulk:float):
    """Run dummy escape model.

    Parameters
    ----------
        hf_row : dict 
            Dictionary of helpfile variables, at this iteration only
        dt : float 
            Time interval over which escape is occuring [yr]
        phi_bulk : float 
            Bulk escape rate [kg s-1]

    Returns
    ----------
        esc_result : dict 
            Dictionary of updated total elemental mass inventories [kg]

    """
    log.info("Running dummy escape...")

    # store value
    out = {}
    out["rate_bulk"] = phi_bulk

    # calculate total mass of volatiles (except oxygen, which is set by fO2)
    M_vols = 0.0
    for e in element_list:
        if e=='O': continue 
        M_vols += hf_row[e+"_kg_total"]


    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio
    for e in element_list:
        if e=='O': continue

        # current elemental mass ratio in total 
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new total mass of element e, keeping a constant mixing ratio of that element 
        out[e+"_kg_total"] = emr * (M_vols - phi_bulk * dt * secs_per_year)

    return out


def RunZEPHYRUS():
    """Zephyrus wrapper 

    Not yet implemented.

    """
    log.info("Running ZEPHYRUS...")
    raise Exception("Not yet implemented")


