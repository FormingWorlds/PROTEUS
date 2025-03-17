# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.outgas.calliope import calc_surface_pressures, calc_target_masses
from proteus.utils.constants import element_list, gas_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

TRUNC_MASS = 1e5

def calc_target_elemental_inventories(dirs:dict, config:Config, hf_row:dict):
    """
    Calculate total amount of volatile elements in the planet
    """

    if config.outgas.module == 'calliope':
        calc_target_masses(dirs, config, hf_row)
    else:
        raise Exception("Unsupported outgassing module selected!")


def run_outgassing(dirs:dict, config:Config, hf_row:dict):
    '''
    Run outgassing model to get new volatile surface pressures
    '''

    # Floating point errors can be problematic here.
    #    Ensure that zero mass values stay at zero by setting all element mass inventories
    #    which are less than TRUNC_MASS to be equal to zero.
    for e in element_list:
        if hf_row[e + "_kg_total"] < TRUNC_MASS:
            hf_row[e + "_kg_total"] = 0.0

    # Run outgassing calculation
    if config.outgas.module == 'calliope':
        calc_surface_pressures(dirs, config, hf_row)
    else:
        raise Exception("Unsupported outgassing module selected!")

    # calculate total atmosphere mass (from sum of volatile masses)
    # this will need to be changed when rock vapours are included
    hf_row["M_atm"] = 0.0
    for s in gas_list:
        hf_row["M_atm"] += hf_row[s + "_kg_atm"]
