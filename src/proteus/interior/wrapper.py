# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize as optimise

from proteus.interior.aragog import AragogRunner
from proteus.interior.common import Interior_t
from proteus.utils.constants import M_earth, R_earth, const_G, element_list
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def update_gravity(hf_row:dict):
    '''
    Update surface gravity.
    '''
    hf_row["gravity"] = const_G * hf_row["M_int"] / (hf_row["R_int"] * hf_row["R_int"])

def calculate_core_mass(hf_row:dict, config:Config):
    '''
    Calculate the core mass of the planet.
    '''
    hf_row["M_core"] = config.struct.core_density * 4.0/3.0 * np.pi * (hf_row["R_int"] * config.struct.corefrac)**3.0

def get_nlevb(config:Config):
    '''
    Get number of interior basic-nodes (level edges) from config.
    '''
    match config.interior.module :
        case "spider":
            return int(config.interior.spider.num_levels)
        case "aragog":
            return int(config.interior.aragog.num_levels)
        case "dummy":
            return 2
    raise ValueError(f"Invalid interior module selected '{config.interior.module}'")

def determine_interior_radius(dirs:dict, config:Config, hf_all:pd.DataFrame, hf_row:dict):
    '''
    Determine the interior radius (R_int) of the planet.

    This uses the interior model's hydrostatic integration to estimate the planet's
    interior mass from a given radius. The radius is then adjusted until the interior mass
    achieves the target mass provided by the user in the config file.
    '''

    log.info("Using %s interior module to solve structure"%config.interior.module)

    # Initial guess for interior radius and gravity
    int_o = Interior_t(get_nlevb(config))
    int_o.ic = 1
    hf_row["R_int"] = R_earth
    calculate_core_mass(hf_row, config)
    hf_row["gravity"] = 9.81

    # Target mass
    M_target = config.struct.mass_tot * M_earth

    # We need to solve for the state hf_row[M_tot] = config.struct.mass_tot
    # This function takes R_int as the input value, and returns the mass residual
    def _resid(x):
        hf_row["R_int"] = x

        log.debug("Try R = %.2e m = %.3f R_earth"%(x,x/R_earth))

        # Use interior model to get dry mass from radius
        calculate_core_mass(hf_row, config)
        run_interior(dirs, config, hf_all, hf_row, int_o, verbose=False)
        update_gravity(hf_row)

        # Calculate residual
        res = hf_row["M_tot"] - M_target
        log.debug("    yields M = %.5e kg , resid = %.3e kg"%(hf_row["M_tot"], res))

        return res

    # Set tolerance
    match config.interior.module:
        case 'aragog':
            rtol = config.interior.aragog.tolerance
        case 'spider':
            rtol = config.interior.spider.tolerance
        case _:
            rtol = 1e-7

    # Find the radius
    r = optimise.root_scalar(_resid, method='secant', xtol=1e3, rtol=rtol, maxiter=10,
                                    x0=hf_row["R_int"], x1=hf_row["R_int"]*1.02)
    hf_row["R_int"] = float(r.root)
    calculate_core_mass(hf_row, config)
    run_interior(dirs, config, hf_all, hf_row, int_o)
    update_gravity(hf_row)

    # Result
    log.info("Found solution for interior structure")
    log.info("M_tot: %.1e kg = %.3f M_earth"%(hf_row["M_tot"], hf_row["M_tot"]/M_earth))
    log.info("R_int: %.1e m  = %.3f R_earth"%(hf_row["R_int"], hf_row["R_int"]/R_earth))
    log.info(" ")

def solve_structure(dirs:dict, config:Config, hf_all:pd.DataFrame, hf_row:dict):
    '''
    Solve for the planet structure based on the method set in the configuration file.

    If the structure is set by the radius, then this is trivial because the radius is used
    as an input to the interior modules anyway. If the structure is set by mass, then it is
    solved as an inverse problem for now.
    '''

    # Set total mass by radius
    # We might need here to setup a determine_interior_mass function as mass calculation depends on gravity
    if config.struct.set_by == 'radius_int':
        # radius defines interior structure
        hf_row["R_int"] = config.struct.radius_int * R_earth
        calculate_core_mass(hf_row, config)
        # initial guess for mass, which will be updated by the interior model
        hf_row["M_int"] = 1.2 * M_earth
        update_gravity(hf_row)

    # Set by total mass (mantle + core + volatiles)
    elif config.struct.set_by == 'mass_tot':
        determine_interior_radius(dirs, config, hf_all, hf_row)

    # Otherwise, error
    else:
        log.error("Invalid constraint on interior structure: %s"%config.struct.set_by)

def run_interior(dirs:dict, config:Config,
                    hf_all:pd.DataFrame, hf_row:dict,
                    interior_o:Interior_t, verbose:bool=True):
    """Run interior mantle evolution model.

    Parameters
    ----------
        dirs : dict
            Dictionary of directories.
        config : Config
            Model configuration
        hf_all : pd.DataFrame
            Dataframe of historical runtime variables
        hf_row : dict
            Dictionary of current runtime variables
        interior_o : Interior_t
            Interior struct.
        verbose : bool
            Verbose printing enabled.
    """

    # Use the appropriate interior model
    if verbose:
        log.info("Evolve interior...")
    log.debug("Using %s module to evolve interior"%config.interior.module)

    # Write tidal heating file
    if config.interior.tidal_heat:
        interior_o.write_tides(dirs["output"])

    if config.interior.module == 'spider':
        # Import
        from proteus.interior.spider import ReadSPIDER, RunSPIDER

        # Run SPIDER
        RunSPIDER(dirs, config, hf_all, hf_row, interior_o)
        sim_time, output = ReadSPIDER(dirs, config, hf_row["R_int"], interior_o)

    elif config.interior.module == 'aragog':
        AragogRunnerInstance = AragogRunner(config, dirs, hf_row, hf_all,
                                            interior_o)
        # Run Aragog
        sim_time, output = AragogRunnerInstance.run_solver(hf_row, interior_o,
                                                           dirs)

    elif config.interior.module == 'dummy':
        # Import
        from proteus.interior.dummy import run_dummy_int

        # Run dummy interior
        sim_time, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    # Read output
    for k in output.keys():
        if k in hf_row.keys():
            hf_row[k] = output[k]

    # Update rheological parameters
    interior_o.update_rheology()

    # Ensure values are >= 0
    for k in ("M_mantle","M_mantle_liquid","M_mantle_solid","M_core","Phi_global"):
        hf_row[k] = max(hf_row[k], 0.0)

    # Check that the new temperature is remotely reasonable
    if not (0 < hf_row["T_magma"] < 1e6):
        UpdateStatusfile(dirs, 21)
        raise ValueError("T_magma is out of range: %g K"%hf_row["T_magma"])


    # Update dry interior mass
    hf_row["M_int"] = hf_row["M_mantle"]  + hf_row["M_core"]

    # Update total planet mass
    M_volatiles = 0.0
    for e in element_list:
        if (e == 'O'):
            # do not include oxygen, because it varies over time in order to set fO2.
            continue
        M_volatiles += hf_row[e+"_kg_total"]
    hf_row["M_tot"] = hf_row["M_int"] + M_volatiles

    # Apply step limiters
    if hf_row["Time"] > 0:

        # Prevent increasing melt fraction, if enabled
        T_magma_prev    = float( hf_all.iloc[-1]["T_magma"] )
        Phi_global_prev = float( hf_all.iloc[-1]["Phi_global"] )
        if config.atmos_clim.prevent_warming and (interior_o.ic==2):
            hf_row["Phi_global"] = min(hf_row["Phi_global"],Phi_global_prev)
            hf_row["T_magma"] = min(hf_row["T_magma"],T_magma_prev)

        # Do not allow massive increases to T_surf, always
        dT_delta  = config.interior.spider.tsurf_atol
        dT_delta += config.interior.spider.tsurf_rtol * T_magma_prev
        if hf_row["T_magma"] > T_magma_prev + dT_delta:
            log.warning("Prevented large increase to T_magma!")
            log.warning("   Clipped from %.2f K"%hf_row["T_magma"])
            hf_row["T_magma"]    = T_magma_prev + dT_delta
            hf_row["Phi_global"] = Phi_global_prev

    # Print result of interior module
    if verbose:
        log.info("    T_magma    = %.3f K"%hf_row["T_magma"])
        log.info("    Phi_global = %.3f  "%hf_row["Phi_global"])
        log.info("    RF_depth   = %.3f  " %hf_row["RF_depth"])
        log.info("    F_int      = %.2e W m-2" %hf_row["F_int"])
        log.info("    F_tidal    = %.2e W m-2" %hf_row["F_tidal"])
        log.info("    F_radio    = %.2e W m-2" %hf_row["F_radio"])

    # Actual time step size
    interior_o.dt = float(sim_time) - hf_row["Time"]


def get_all_output_times(output_dir:str, model:str):

    if model == "spider":
        from proteus.interior.spider import get_all_output_times as _get_output_times
    elif model == "aragog":
        from proteus.interior.aragog import get_all_output_times as _get_output_times
    else:
        return []

    return _get_output_times(output_dir)

def read_interior_data(output_dir:str, model:str, times:list):

    if len(times) == 0:
        return []

    if model == "spider":
        from proteus.interior.spider import read_jsons
        return read_jsons(output_dir, times)

    elif model == "aragog":
        from proteus.interior.aragog import read_ncdfs
        return read_ncdfs(output_dir, times)

    else:
        return []
