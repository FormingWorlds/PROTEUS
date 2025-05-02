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

from proteus.outgas.common import expected_keys
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
        if s != "O2":
            pressure = config.delivery.volatiles.get_pressure(s)
            included = config.outgas.calliope.is_included(s)
        else:
            pressure = 0.0
            included = True

        solvevol_inp[f'{s}_initial_bar'] = float(pressure)
        solvevol_inp[f'{s}_included'] = int(included)

        if (s in ("H2O","CO2","N2","S2")) and not included:
            UpdateStatusfile(dirs, 20)
            raise ValueError(f"Missing required volatile {s}")

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
        raise ValueError("Invalid volatile inventory configuration")

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
        hf_row[e + "_kg_total"] = solvevol_target[e]


def construct_guess(hf_row:dict, target:dict, mass_thresh:float) -> dict | None:
    """
    Construct initial guess for CALLIOPE.

    Returns None for time=0, otherwise returns a dictionary of partial pressures.

    Parameters
    ----------
    hf_row : dict
        Dictionary containing the current state of the planet
    target : dict
        Dictionary containing the target elemental inventories [kg]
    mass_thresh : float
        Minimum threshold for element mass [kg]. Inventories below this are set to zero.

    Returns
    -------
    p_guess : dict | None
        Dictionary containing the guess for the surface pressures [bar]
    """

    log.debug("Initial guess for CALLIOPE")

    # During initial phase, allow CALLIOPE to make its own guess
    if hf_row["Time"] < 1:
        log.debug("    providing None, allowing CALLIOPE to guess")
        return None

    # Dictionary of partial pressures [bar] for H2O, CO2, N2, S2
    p_guess = {}

    # Use previous value from hf_row
    log.debug("    using previous partial pressures from hf_row")
    for s in vol_list:
        p_guess[s] = hf_row[f"{s}_bar"]

    # Check if elemental inventory is zero => guess zero pressure
    for s in vol_list:

        # check if any of the elements are zero in the planet
        is_zero = False
        for e in element_list:
            if e == "O":
                continue
            if (e in s) and (target[e] < mass_thresh): # kg
                is_zero = True
                break

        # if any of the elements are zero, set guess to zero
        if is_zero:
            p_guess[s] = 0.0
            log.debug("    %s: guess set to zero"%s)

    return p_guess

def flag_included_volatiles(guess:dict, config:Config) -> dict:
    """
    Determine which volatiles are included in the outgassing calculation

    Parameters
    ----------
    guess : dict
        Dictionary containing the guess for the surface pressures [bar]
    config : Config
        Configuration object

    Returns
    -------
    p_included : dict
        Dictionary containing the inclusion status of each volatile (true/false)
    """

    # Included based on config
    p_included = {}
    for s in vol_list:
        if s == "O2":
            p_included[s] = True
        else:
            p_included[s] = bool(getattr(config.outgas.calliope, f"include_{s}"))

    # If guess is none, just do what config suggests
    if guess is None:
        return p_included

    # Check if partial pressure is zero => do not include volatile
    for s in vol_list:
        if s != "O2":
            p_included[s] = p_included[s] and (guess[s] > 0.0)

    return p_included

def calc_surface_pressures(dirs:dict, config:Config, hf_row:dict):

    # Inform
    log.debug("Running CALLIOPE...")

    # make solvevol options
    opts = construct_options(dirs, config, hf_row)

    # convert masses to dict for calliope
    target = {}
    for e in element_list:
        if e != 'O':
            target[e] = hf_row[e + "_kg_total"]

    # construct guess for CALLIOPE
    p_guess = construct_guess(hf_row, target, config.outgas.mass_thresh)

    # check if gas is included or not
    p_incl = flag_included_volatiles(p_guess, config)

    # Set included
    for s in vol_list:
        opts[f'{s}_included'] = int(p_incl[s])

    # Do not allow low temperatures
    if opts["T_magma"] < config.outgas.calliope.T_floor:
        opts["T_magma"] = config.outgas.calliope.T_floor
        log.warning("Outgassing temperature clipped to %.1f K"%opts["T_magma"])

    # get atmospheric compositison
    try:
        solvevol_result = equilibrium_atmosphere(target, opts,
                                                    xtol=config.outgas.calliope.xtol,
                                                    rtol=config.outgas.calliope.rtol,
                                                    atol=config.outgas.mass_thresh,
                                                    nguess=int(1e3), nsolve=int(3e3),
                                                    p_guess=p_guess,
                                                    print_result=False)
    except RuntimeError as e:
        log.error("Outgassing calculation with CALLIOPE failed")
        UpdateStatusfile(dirs, 27)
        raise e

    # Get result
    for k in expected_keys():
        if k in solvevol_result:
            hf_row[k] = solvevol_result[k]
