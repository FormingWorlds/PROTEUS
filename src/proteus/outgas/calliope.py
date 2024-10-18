# Function and classes used to run CALLIOPE
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.constants import volatile_species, element_list
from proteus.utils.helper import UpdateStatusfile

from calliope.solve import (
    get_target_from_params,
    get_target_from_pressures,
    equilibrium_atmosphere
)

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


def calc_target_masses(config:Config, hf_row:dict):
    # make solvevol options
    solvevol_inp = construct_options(config, hf_row)

    # calculate target mass of atoms (except O, which is derived from fO2)
    if config.delivery.initial == 'elements':
        solvevol_target = get_target_from_params(solvevol_inp)
    else:
        solvevol_target = get_target_from_pressures(solvevol_inp)

    # prevent numerical issues
    for key in solvevol_target.keys():
        if solvevol_target[key] < 1.0e4:
            solvevol_target[key] = 0.0

    # store in hf_row as elements
    for e in element_list:
        if e == "O":
            continue
        hf_row[e + "_kg_total"] = solvevol_target[e]


def calc_surface_pressures(config:Config, hf_row:dict):
    # make solvevol options
    solvevol_inp = construct_options(config, hf_row)

    # convert masses to dict for calliope
    solvevol_target = {}
    for e in element_list:
        if e == "O":
            continue
        solvevol_target[e] = hf_row[e + "_kg_total"]

    # get atmospheric compositison
    solvevol_result = equilibrium_atmosphere(solvevol_target, solvevol_inp)
    for k in solvevol_result.keys():
        if k in hf_row.keys():
            hf_row[k] = solvevol_result[k]
