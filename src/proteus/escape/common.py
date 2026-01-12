
# Shared functions used by escape calculation
from __future__ import annotations

import logging

from proteus.utils.constants import element_list

log = logging.getLogger("fwl."+__name__)

def calc_unfract_fluxes(hf_row:dict, reservoir:str, min_thresh:float):
    """Calculate elemental escape rates, without fractionating.

    Updates the elemental escape fluxes in hf_row.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        reservoir: str
            Element reservoir representing the escaping composition (bulk, outgas)
        min_thresh: float
            Minimum threshold for element mass [kg]. Inventories below this are set to zero.
    """

    # which composition sets bulk escape?
    match reservoir:
        case "bulk": # bulk planet
            key = "_kg_total"
        case "outgas": # atmosphere
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
        emr = res[e]/M_vols
        log.debug("    %2s (%s) mass ratio = %.2e "%(e,reservoir,emr))
        hf_row["esc_rate_"+e] = hf_row["esc_rate_total"] * emr
