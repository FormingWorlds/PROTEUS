# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import scipy.optimize as optimise

from proteus.utils.constants import M_earth, R_earth, const_G
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def update_gravity(hf_row:dict):
    '''
    Update surface gravity.
    '''
    hf_row["gravity"] = const_G * hf_row["M_int"] / (hf_row["R_int"] * hf_row["R_int"])

def determine_interior_radius(dirs:dict, config:Config, hf_all:pd.DataFrame, hf_row:dict):
    '''
    Determine the interior radius (R_int) of the planet.

    This uses the interior model's hydrostatic integration to estimate the planet's
    interior mass from a given radius. The radius is then adjusted until the interior mass
    achieves the target mass provided by the user in the config file.

    For the dummy interior, the radius is simply specified by the user in the config file.
    '''

    log.info("Using %s interior module to solve strcture"%config.interior.module)

    solve_g = True
    if config.interior.module == 'aragog':
        log.warning("Cannot solve for interior structure with aragog")
        solve_g = False

    # Initial guess for interior radius and gravity
    IC_INTERIOR = 1
    hf_row["R_int"]   = R_earth
    hf_row["gravity"] = 9.81

    # Target mass
    M_target = config.struct.mass * M_earth

    # We need to solve for the state M_int = config.struct.mass
    # This function takes R_int as the input value, and returns the mass residual
    def _resid(x):
        hf_row["R_int"] = x

        log.debug("Try R = %.2e m = %.3f R_earth"%(x,x/R_earth))

        run_interior(dirs, config, IC_INTERIOR, hf_all, hf_row, verbose=False)
        if solve_g:
            update_gravity(hf_row)

        res = hf_row["M_int"] - M_target
        log.debug("    yields M = %.5e kg , resid = %.3e kg"%(hf_row["M_int"], res))

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
    r = optimise.root_scalar(_resid, method='secant', xtol=1e3, rtol=rtol, maxiter=12,
                                    x0=hf_row["R_int"], x1=hf_row["R_int"]*1.02)
    hf_row["R_int"] = float(r.root)
    run_interior(dirs, config, IC_INTERIOR, hf_all, hf_row)
    # update_gravity(hf_row)

    # Result
    log.info("Found solution for interior structure")
    log.info("M_int: %.1e kg = %.3f M_earth"%(hf_row["M_int"], hf_row["M_int"]/M_earth))
    log.info("R_int: %.1e m  = %.3f R_earth"%(hf_row["R_int"], hf_row["R_int"]/R_earth))
    log.info(" ")


def run_interior(dirs:dict, config:Config, IC_INTERIOR:int,
                 hf_all:pd.DataFrame, hf_row:dict, verbose:bool=True):
    '''
    Run interior model
    '''

    # Use the appropriate interior model
    if verbose:
        log.info("Evolve interior...")
    log.debug("Using %s module to evolve interior"%config.interior.module)

    if config.interior.module == 'spider':
        # Import
        from proteus.interior.spider import ReadSPIDER, RunSPIDER

        # Run SPIDER
        RunSPIDER(dirs, config, IC_INTERIOR, hf_all, hf_row)
        sim_time, output = ReadSPIDER(dirs, config, hf_row["R_int"])

    elif config.interior.module == 'aragog':
        # Import
        from proteus.interior.aragog import RunAragog

        # Run Aragog
        sim_time, output = RunAragog(config, dirs, IC_INTERIOR, hf_row, hf_all)

    elif config.interior.module == 'dummy':
        # Import
        from proteus.interior.dummy import RunDummyInt

        # Run dummy interior
        sim_time, output = RunDummyInt(config, dirs, IC_INTERIOR, hf_row, hf_all)

    # Read output
    for k in output.keys():
        if k in hf_row.keys():
            hf_row[k] = output[k]

    # Check that the new temperature is remotely reasonable
    if not (0 < hf_row["T_magma"] < 1e6):
        log.error("T_magma is out of range: %g K"%hf_row["T_magma"])
        UpdateStatusfile(dirs, 21)
        exit(1)

    # Update dry interior mass
    hf_row["M_int"] = hf_row["M_mantle"]  + hf_row["M_core"]

    # Prevent increasing melt fraction
    if config.atmos_clim.prevent_warming and (IC_INTERIOR==2):
        hf_row["Phi_global"] = min(hf_row["Phi_global"], hf_all.iloc[-1]["Phi_global"])
        hf_row["T_magma"] = min(hf_row["T_magma"], hf_all.iloc[-1]["T_magma"])

    # Print result of interior module
    if verbose:
        log.info("    T_magma    = %.1f K"%hf_row["T_magma"])
        log.info("    Phi_global = %.3f  "%hf_row["Phi_global"])
        log.info("    F_int      = %.2e" %hf_row["F_int"])
        log.info("    RF_depth   = %.2e" %hf_row["RF_depth"])

    # Time step size
    dt = float(sim_time) - hf_row["Time"]

    # Return timestep
    return dt

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
