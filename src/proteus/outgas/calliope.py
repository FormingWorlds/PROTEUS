# Function and classes used to run CALLIOPE
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.constants import volatile_species, element_list

log = logging.getLogger("fwl."+__name__)

def construct_options(config:Config, hf_row:dict):
    """
    Construct CALLIOPE options dictionary
    """

    solvevol_inp = {}

    # Planet properties
    solvevol_inp["M_mantle"]    =   hf_row["M_mantle"]
    solvevol_inp["Phi_global"]  =   hf_row["Phi_global"]
    solvevol_inp["gravity"]     =   hf_row["gravity"]
    solvevol_inp["mass"]        =   hf_row["M_planet"]
    solvevol_inp["radius"]      =   hf_row["R_planet"]

    # Surface properties
    solvevol_inp["T_magma"]     =   hf_row["T_magma"]
    solvevol_inp['fO2_shift_IW'] =  config.outgas.fO2_shift_IW

    # Elemental inventory
    solvevol_inp['hydrogen_earth_oceans'] = config.delivery.elements.H_oceans
    solvevol_inp['CH_ratio']    =           config.delivery.elements.CH_ratio
    solvevol_inp['nitrogen_ppmw'] =         config.delivery.elements.N_ppmw
    solvevol_inp['sulfur_ppmw'] =           config.delivery.elements.S_ppmw

    # Volatile inventory
    for s in volatile_species:
        solvevol_inp[f'{s}_initial_bar'] =  config[f'{s}_initial_bar']
        solvevol_inp[f'{s}_included'] =     config[f'{s}_included']

    return solvevol_inp
