# Generic outgassing wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.outgas.calliope import calc_surface_pressures, calc_target_masses
from proteus.utils.constants import gas_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calc_target_elemental_inventories(dirs:dict, config:Config, hf_row:dict):
    """
    Calculate total amount of volatile elements in the planet
    """

    if config.outgas.module == 'calliope':
        calc_target_masses(dirs, config, hf_row)
    else:
        raise ValueError("Unsupported outgassing module selected!")

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
        calc_surface_pressures(dirs, config, hf_row)
    else:
        raise ValueError("Unsupported outgassing module selected!")

    # calculate total atmosphere mass (from sum of volatile masses)
    # this will need to be changed when rock vapours are included
    hf_row["M_atm"] = 0.0
    for s in gas_list:
        hf_row["M_atm"] += hf_row[s + "_kg_atm"]

def run_desiccated(config:Config, hf_row:dict) -> bool:
    '''
    Handle desiccation of the planet. This substitutes for run_outgassing when the planet
    has lost its entire volatile inventory.

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
    '''

    # check if desiccation has occurred
    desiccated = False
    for g in gas_list:
        if hf_row[g + "_kg_total"] > config.outgas.mass_thresh:
            desiccated = False
            break
        else:
            desiccated = True

    # if desiccated, set all gas masses to zero
