# Function and classes used to run CALLIOPE
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

from calliope.constants import molar_mass, ocean_moles
from calliope.solve import (
    equilibrium_atmosphere,
    get_target_from_params,
    get_target_from_pressures,
)
from proteus.utils.constants import element_list, vol_list
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger("fwl."+__name__)

# Constants
mass_ocean = ocean_moles * molar_mass['H2']

def construct_options(dirs:dict, config:Config, hf_row:dict):
    """
    Construct CALLIOPE options dictionary
    """

    invalid = False  # Invalid options set by config

    solvevol_inp = {}

    # Planet properties
    solvevol_inp["M_mantle"]    = hf_row["M_mantle"]
    solvevol_inp["Phi_global"]  = hf_row["Phi_global"]
    solvevol_inp["gravity"]     = hf_row["gravity"]
    solvevol_inp["radius"]      = hf_row["R_int"]

    # Surface properties
    solvevol_inp["T_magma"]     =  hf_row["T_magma"]
    solvevol_inp['fO2_shift_IW'] = config.outgas.fO2_shift_IW

    # Volatile inventory
    for s in vol_list:
        solvevol_inp[f'{s}_initial_bar'] = config.delivery.volatiles.get_pressure(s)

        included = config.outgas.calliope.is_included(s)
        solvevol_inp[f'{s}_included'] = int(included)

        if (s in ("H2O","CO2","N2","S2")) and not included:
            UpdateStatusfile(dirs, 20)
            log.error(f"Missing required volatile {s}")
            exit(1)

    # Set by volatiles?
    if config.delivery.initial == 'volatiles':
        return solvevol_inp

    # Calculate hydrogen inventory...

    #    absolute part (H_kg = H_oceans * number_ocean_moles * molar_mass['H2'])
    H_abs = float(config.delivery.elements.H_oceans) * mass_ocean

    #    relative part (H_kg = H_rel * 1e-6 * M_mantle)
    H_rel = config.delivery.elements.H_ppmw * 1e-6 * hf_row["M_mantle"]

    #    use whichever was set (one of these will be zero)
    if H_abs < 1.0:
        if H_rel < 1.0:
            log.error("Hydrogen inventory is unspecified")
            invalid = True
        else:
            H_kg = H_rel
    elif H_rel < 1.0:
        H_kg = H_abs
    else:
        log.error("Hydrogen inventory must be specified by H_oceans or H_ppmw, not both")
        invalid = True
        H_kg = -1 # dummy value

    # Calculate carbon inventory (we need CH_ratio for calliope)
    CH_ratio = float(config.delivery.elements.CH_ratio)
    C_ppmw   = float(config.delivery.elements.C_ppmw)
    if CH_ratio > 1e-10:
        # check that C_ppmw isn't also set
        if C_ppmw > 1e-10:
            log.error("Carbon inventory must be specified by CH_ratio or C_ppmw, not both")
            invalid = True
    else:
        # calculate C/H ratio for calliope from C_kg and H_kg
        CH_ratio = config.delivery.elements.C_ppmw * 1e-6 * hf_row["M_mantle"] / H_kg

    # Calculate nitrogen inventory (we need N_ppmw for calliope)
    NH_ratio = float(config.delivery.elements.NH_ratio)
    N_ppmw   = float(config.delivery.elements.N_ppmw)
    if NH_ratio > 1e-10:
        # check that N_ppmw isn't also set
        if N_ppmw > 1e-10:
            log.error("Nitrogen inventory must be specified by NH_ratio or N_ppmw, not both")
            invalid = True
        # calculate N_ppmw
        N_ppmw = 1e6 * NH_ratio * H_kg / hf_row["M_mantle"]

    # Calculate sulfur inventory (we need S_ppmw for calliope)
    SH_ratio = float(config.delivery.elements.SH_ratio)
    S_ppmw   = float(config.delivery.elements.S_ppmw)
    if SH_ratio > 1e-10:
        # check that S_ppmw isn't also set
        if S_ppmw> 1e-10:
            log.error("Sulfur inventory must be specified by SH_ratio or S_ppmw, not both")
            invalid = True
        # calculate S_ppmw
        S_ppmw = 1e6 * SH_ratio * H_kg / hf_row["M_mantle"]

    # Volatile abundances are over-specified in the config file.
    # The code exits here, rather than above, in case there are multiple
    #   instances of volatiles being over-specified in the file.
    if invalid:
        log.error("  a) set X by metallicity, e.g. XH_ratio=1.2 and X_ppmw=0")
        log.error("  b) set X by concentration, e.g. XH_ratio=0 and X_ppmw=2.01")
        UpdateStatusfile(dirs, 20)
        exit(1)

    # Pass elemental inventory
    solvevol_inp['hydrogen_earth_oceans'] = H_kg / mass_ocean
    solvevol_inp['CH_ratio']    =           CH_ratio
    solvevol_inp['nitrogen_ppmw'] =         N_ppmw
    solvevol_inp['sulfur_ppmw'] =           S_ppmw

    return solvevol_inp


def calc_target_masses(dirs:dict, config:Config, hf_row:dict):
    # make solvevol options
    solvevol_inp = construct_options(dirs, config, hf_row)

    # calculate target mass of atoms (except O, which is derived from fO2)
    if config.delivery.initial == 'elements':
        solvevol_target = get_target_from_params(solvevol_inp)
    else:
        solvevol_target = get_target_from_pressures(solvevol_inp)

    # store in hf_row as elements
    for e in solvevol_target.keys():
        if e == "O":
            continue
        hf_row[e + "_kg_total"] = solvevol_target[e]

def calc_surface_pressures(dirs:dict, config:Config, hf_row:dict):
    # make solvevol options
    solvevol_inp = construct_options(dirs, config, hf_row)

    # convert masses to dict for calliope
    solvevol_target = {}
    for e in element_list:
        if e == "O":
            continue

        # save to dict
        solvevol_target[e] = hf_row[e + "_kg_total"]

    # Do not allow low temperatures
    if solvevol_inp["T_magma"] < config.outgas.calliope.T_floor:
        solvevol_inp["T_magma"] = config.outgas.calliope.T_floor
        log.warning("Outgassing temperature clipped to %.1f K"%solvevol_inp["T_magma"])

    # get atmospheric compositison
    solvevol_result = equilibrium_atmosphere(solvevol_target, solvevol_inp, rtol=1e-7)
    for k in solvevol_result.keys():
        if k in hf_row.keys():
            hf_row[k] = solvevol_result[k]

    # print info
    log.info("    total  : %-8.2f bar"%hf_row["P_surf"])
    log.info("    mmw    : %-8.4f g mol-1"%(hf_row["atm_kg_per_mol"]*1e3))
