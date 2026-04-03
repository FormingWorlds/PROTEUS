# Function and classes used to run CALLIOPE
from __future__ import annotations  # noqa: I001

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
from proteus.utils.constants import C_solar, N_solar, S_solar, element_list, vol_list
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger('fwl.' + __name__)

# Constants
mass_ocean = ocean_moles * molar_mass['H2']


def construct_options(dirs: dict, config: Config, hf_row: dict):
    """
    Construct CALLIOPE options dictionary
    """


    solvevol_inp = {}

    # Planet properties
    solvevol_inp['M_mantle'] = hf_row['M_mantle']
    solvevol_inp['gravity'] = hf_row['gravity']
    solvevol_inp['radius'] = hf_row['R_int']

    # Mantle melt fraction
    if config.outgas.calliope.solubility:
        solvevol_inp['Phi_global'] = hf_row['Phi_global']
    else:
        solvevol_inp['Phi_global'] = 0.0

    # Surface properties
    solvevol_inp['T_magma'] = hf_row['T_magma']
    solvevol_inp['fO2_shift_IW'] = config.outgas.fO2_shift_IW

    # Volatile inventory
    for s in vol_list:
        if s != 'O2':
            pressure = config.planet.gas_prs.get_pressure(s)
            included = config.outgas.calliope.is_included(s)
        else:
            pressure = 0.0
            included = True

        solvevol_inp[f'{s}_initial_bar'] = float(pressure)
        solvevol_inp[f'{s}_included'] = int(included)

        if (s in ('H2O', 'CO2', 'N2', 'S2')) and not included:
            UpdateStatusfile(dirs, 20)
            raise ValueError(f'Missing required volatile {s}')

    # Set by partial pressures?
    if config.planet.volatile_mode == 'gas_prs':
        return solvevol_inp

    # --- Element abundance mode ---
    elem = config.planet.elements

    # Reservoir mass for ppmw calculations
    M_mantle = hf_row['M_mantle']
    if config.planet.volatile_reservoir == 'mantle+core':
        M_reservoir = hf_row['M_int']  # M_mantle + M_core
    else:
        M_reservoir = M_mantle

    # Hydrogen inventory [kg]
    match elem.H_mode:
        case 'oceans':
            H_kg = float(elem.H_budget) * mass_ocean
        case 'ppmw':
            H_kg = float(elem.H_budget) * 1e-6 * M_reservoir
        case 'kg':
            H_kg = float(elem.H_budget)

    if H_kg < 1.0:
        log.error('Hydrogen inventory is zero or unspecified (H_mode=%s, H_budget=%g)',
                  elem.H_mode, elem.H_budget)
        UpdateStatusfile(dirs, 20)
        raise ValueError('Hydrogen inventory must be > 0')

    # C/N/S inventories
    if elem.use_metallicity:
        # Scale from solar metallicity relative to H
        CH_ratio = elem.metallicity * C_solar
        N_kg = elem.metallicity * N_solar * H_kg
        S_kg = elem.metallicity * S_solar * H_kg
    else:
        # Carbon
        C_kg = _resolve_element(elem.C_mode, elem.C_budget, H_kg, M_reservoir, 'C')
        CH_ratio = C_kg / H_kg if H_kg > 0 else 0.0

        # Nitrogen
        N_kg = _resolve_element(elem.N_mode, elem.N_budget, H_kg, M_reservoir, 'N')

        # Sulfur
        S_kg = _resolve_element(elem.S_mode, elem.S_budget, H_kg, M_reservoir, 'S')

    # Convert to CALLIOPE's internal units (always relative to M_mantle)
    N_ppmw = 1e6 * N_kg / M_mantle if M_mantle > 0 else 0.0
    S_ppmw = 1e6 * S_kg / M_mantle if M_mantle > 0 else 0.0

    solvevol_inp['hydrogen_earth_oceans'] = H_kg / mass_ocean
    solvevol_inp['CH_ratio'] = CH_ratio
    solvevol_inp['nitrogen_ppmw'] = N_ppmw
    solvevol_inp['sulfur_ppmw'] = S_ppmw

    return solvevol_inp


def _resolve_element(mode: str, budget: float, H_kg: float, M_reservoir: float, name: str) -> float:
    """Convert element mode+budget to absolute mass [kg].

    Parameters
    ----------
    mode : str
        'X/H' (mass ratio to H), 'ppmw' (relative to M_reservoir), or 'kg'.
    budget : float
        The value in the units defined by mode.
    H_kg : float
        Hydrogen mass [kg] (for X/H mode).
    M_reservoir : float
        Reservoir mass [kg] for ppmw (M_mantle or M_int).
    name : str
        Element name (for error messages).

    Returns
    -------
    float
        Element mass [kg].
    """
    ratio_key = f'{name}/H'
    match mode:
        case _ if mode == ratio_key:
            return float(budget) * H_kg
        case 'ppmw':
            return float(budget) * 1e-6 * M_reservoir
        case 'kg':
            return float(budget)
        case _:
            raise ValueError(f"Unknown {name}_mode: '{mode}'. Expected '{ratio_key}', 'ppmw', or 'kg'")


def calc_target_masses(dirs: dict, config: Config, hf_row: dict):
    # make solvevol options
    solvevol_inp = construct_options(dirs, config, hf_row)

    # calculate target mass of atoms (except O, which is derived from fO2)
    if config.planet.volatile_mode == 'elements':
        solvevol_target = get_target_from_params(solvevol_inp)
    else:
        solvevol_target = get_target_from_pressures(solvevol_inp)

    # store in hf_row as elements
    for e in solvevol_target.keys():
        hf_row[e + '_kg_total'] = solvevol_target[e]


def construct_guess(hf_row: dict, target: dict, mass_thresh: float) -> dict | None:
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

    log.debug('Initial guess for CALLIOPE')

    # During initial phase, allow CALLIOPE to make its own guess
    if hf_row['Time'] < 1:
        log.debug('    providing None, allowing CALLIOPE to guess')
        return None

    # Dictionary of partial pressures [bar] for H2O, CO2, N2, S2
    p_guess = {}

    # Use previous value from hf_row
    log.debug('    using previous partial pressures from hf_row')
    for s in vol_list:
        p_guess[s] = hf_row[f'{s}_bar']

    # Check if elemental inventory is zero => guess zero pressure
    for s in vol_list:
        # check if any of the elements are zero in the planet
        is_zero = False
        for e in element_list:
            if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
                continue
            if (e in s) and (target[e] < mass_thresh):  # kg
                is_zero = True
                break

        # if any of the elements are zero, set guess to zero
        if is_zero:
            p_guess[s] = 0.0
            log.debug('    %s: guess set to zero' % s)

    return p_guess


def flag_included_volatiles(guess: dict, config: Config) -> dict:
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
        if s == 'O2':
            p_included[s] = True
        else:
            p_included[s] = bool(getattr(config.outgas.calliope, f'include_{s}'))

    # If guess is none, just do what config suggests
    if guess is None:
        return p_included

    # Check if partial pressure is zero => do not include volatile
    for s in vol_list:
        if s != 'O2':
            p_included[s] = p_included[s] and (guess[s] > 0.0)

    return p_included


def calc_surface_pressures(dirs: dict, config: Config, hf_row: dict):
    # Inform
    log.debug('Running CALLIOPE...')

    # make solvevol options
    opts = construct_options(dirs, config, hf_row)

    # convert masses to dict for calliope
    target = {}
    for e in element_list:
        if e != 'O':
            target[e] = hf_row[e + '_kg_total']

    # construct guess for CALLIOPE
    p_guess = construct_guess(hf_row, target, config.outgas.mass_thresh)

    # check if gas is included or not
    p_incl = flag_included_volatiles(p_guess, config)

    # Set included
    for s in vol_list:
        opts[f'{s}_included'] = int(p_incl[s])

    # Do not allow low temperatures
    if opts['T_magma'] < config.outgas.T_floor:
        opts['T_magma'] = config.outgas.T_floor
        log.warning('Outgassing temperature clipped to %.1f K' % opts['T_magma'])

    # get atmospheric compositison
    try:
        solvevol_result = equilibrium_atmosphere(
            target,
            opts,
            xtol=config.outgas.solver_atol,
            rtol=config.outgas.solver_rtol,
            atol=config.outgas.mass_thresh,
            nguess=int(1e3),
            nsolve=int(3e3),
            p_guess=p_guess,
            print_result=False,
            opt_solver=False,
        )
    except RuntimeError as e:
        log.error('Outgassing calculation with CALLIOPE failed')
        UpdateStatusfile(dirs, 27)
        raise e

    # Get result
    for k in expected_keys():
        if k in solvevol_result:
            hf_row[k] = solvevol_result[k]
