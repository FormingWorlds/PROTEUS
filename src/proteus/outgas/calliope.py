# Function and classes used to run CALLIOPE
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.constants import volatile_species, element_list
from proteus.utils.helper import UpdateStatusfile

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
        solvevol_inp[f'{s}_initial_bar'] =  getattr(config.delivery.volatiles, s)

        included = getattr(config.outgas.calliope, f'include_{s}')
        solvevol_inp[f'{s}_included'] = 1 if included else 0

        if (s in ["H2O","CO2","N2","S2"]) and not included:
            UpdateStatusfile(dirs, 20)
            raise RuntimeError(f"Missing required volatile {s}")

    return solvevol_inp
