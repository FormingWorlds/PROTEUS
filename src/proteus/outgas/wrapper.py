# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.helper import PrintHalfSeparator
from proteus.utils.constants import volatile_species, element_list
from proteus.outgas.calliope import (
    calc_surface_pressures,
    calc_target_masses
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calc_target_elemental_inventories(config:Config, hf_row:dict):
    """
    Calculate total amount of volatile elements in the planet
    """

    if config.outgas.module == 'calliope':
        calc_target_masses(config, hf_row)
    else:
        raise Exception("Unsupported outgassing module selected!")


def run_outgassing(config:Config, hf_row:dict):
    '''
    Run outgassing model to get new volatile surface pressures
    '''

    if config.outgas.module == 'calliope':
        calc_surface_pressures(config, hf_row)
    else:
        raise Exception("Unsupported outgassing module selected!")

    # calculate total atmosphere mass (from sum of volatile masses)
    # this will need to be changed when rock vapours are included
    hf_row["M_atm"] = 0.0
    for s in volatile_species:
        hf_row["M_atm"] += hf_row[s + "_kg_atm"]
