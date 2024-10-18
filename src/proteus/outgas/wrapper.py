# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.helper import PrintHalfSeparator
from proteus.utils.constants import volatile_species, element_list

from calliope.solve import (
    equilibrium_atmosphere,
    get_target_from_params,
    get_target_from_pressures,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calc_target_elemental_inventories(config:Config, solvevol_inp:dict, hf_row:dict):
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


def run_outgas(solvevol_inp, hf_row):
    '''
    Run volatile outgassing model
    '''

    PrintHalfSeparator()

    solvevol_target = {}
    for e in element_list:
        if e == "O":
            continue
        solvevol_target[e] = hf_row[e + "_kg_total"]

    solvevol_result = equilibrium_atmosphere(solvevol_target, solvevol_inp)

    #    store results
    for k in solvevol_result.keys():
        if k in hf_row.keys():
            hf_row[k] = solvevol_result[k]

    #    calculate total atmosphere mass
    hf_row["M_atm"] = 0.0
    for s in volatile_species:
        hf_row["M_atm"] += hf_row[s + "_kg_atm"]
