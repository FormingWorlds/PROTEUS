# Generic outgassing wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.calliope import calc_surface_pressures, calc_target_masses
from proteus.outgas.common import expected_keys
from proteus.utils.constants import element_list, gas_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calc_target_elemental_inventories(dirs:dict, config:Config, hf_row:dict):
    """
    Calculate total amount of volatile elements in the planet
    """

    if config.outgas.module == 'calliope':
        calc_target_masses(dirs, config, hf_row)

def check_desiccation(config:Config, hf_row:dict) -> bool:
    """
    Check if the planet has desiccated. This is done by checking if all volatile masses
    are below a threshold.

    Parameters
    ----------
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only

    Returns
    -------
        bool
            True if desiccation occurred, False otherwise
    """

    # check if desiccation has occurred
    for e in element_list:
        if e == 'O':
            continue
        if hf_row[e + "_kg_total"] > config.outgas.mass_thresh:
            log.info("Not desiccated, %s = %.2e kg" % (e, hf_row[e + "_kg_total"]))
            return False # return, and allow run_outgassing to proceed

    return True


def run_outgassing(dirs:dict, config:Config, hf_row:dict):
    '''
    Run outgassing model to get new volatile surface pressures

    Parameters
    ----------
        dirs : dict
            Dictionary of directory paths
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    '''

    log.info("Solving outgassing...")

    # Run outgassing calculation
    if config.outgas.module == 'calliope':
        calc_surface_pressures(dirs, config, hf_row)

    # calculate total atmosphere mass (from sum of volatile masses)
    hf_row["M_atm"] = 0.0
    for s in gas_list:
        hf_row["M_atm"] += hf_row[s + "_kg_atm"]

    # print outgassed partial pressures (in order of descending abundance)
    mask = [hf_row[s+"_vmr"] for s in gas_list]
    for i in np.argsort(mask)[::-1]:
        s = gas_list[i]
        _p = hf_row[s+"_bar"]
        _x = hf_row[s+"_vmr"]
        _s = "    %-6s     = %-9.2f bar (%.2e VMR)" % (s,_p,_x)
        if _p > 0.01:
            log.info(_s)
        else:
            # don't spam log with species of negligible abundance
            log.debug(_s)

    # print total pressure and mmw
    log.info("    total      = %-9.2f bar"%hf_row["P_surf"])
    log.info("    mmw        = %-9.5f g mol-1"%(hf_row["atm_kg_per_mol"]*1e3))

def run_desiccated(config:Config, hf_row:dict):
    '''
    Handle desiccation of the planet. This substitutes for run_outgassing when the planet
    has lost its entire volatile inventory.

    Parameters
    ----------
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    '''

    # if desiccated, set all gas masses to zero
    log.info("Desiccation has occurred - no volatiles remaining")

    # Do not set these to zero - avoid divide by zero elsewhere in the code
    excepted_keys = ["atm_kg_per_mol"]
    for g in gas_list:
        excepted_keys.append(f"{g}_vmr")

    # Set most values to zero
    for k in expected_keys():
        if k not in excepted_keys:
            hf_row[k] = 0.0
