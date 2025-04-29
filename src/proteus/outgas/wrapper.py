# Generic outgassing wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.outgas.calliope import calc_surface_pressures, calc_target_masses
from proteus.outgas.common import expected_keys
from proteus.utils.constants import element_list, gas_list
from proteus.utils.helper import UpdateStatusfile

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
        if hf_row[e + "_kg_total"] > config.outgas.mass_thresh:
            log.debug("Not desiccated, %s = %.2e kg" % (e, hf_row[e + "_kg_total"]))
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

    # Run outgassing calculation
    if config.outgas.module == 'calliope':
        try:
            calc_surface_pressures(dirs, config, hf_row)
        except RuntimeError as e:
            log.error("Outgassing calculation failed")
            UpdateStatusfile(dirs, 27)
            raise e

    # calculate total atmosphere mass (from sum of volatile masses)
    # this will need to be changed when rock vapours are included
    hf_row["M_atm"] = 0.0
    for s in gas_list:
        hf_row["M_atm"] += hf_row[s + "_kg_atm"]

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

    excepted_keys = ["atm_kg_per_mol"]

    # Set most values to zero
    for k in expected_keys():
        if k not in excepted_keys:
            hf_row[k] = 0.0
