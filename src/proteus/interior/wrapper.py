# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from proteus.utils.constants import const_G, R_earth, M_earth
from proteus.utils.helper import PrintHalfSeparator

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def update_gravity(hf_config:dict):
    '''
    Update surface gravity.
    '''
    hf_row["gravity"] = const_G * hf_row["M_int"] / (hf_row["R_int"] * hf_row["R_int"])

def determine_interior_radius(dirs:dict, config:Config, loop_counter:dict, hf_all:pd.DataFrame, hf_row:dict):
    '''
    Determine the interior radius (R_int) of the planet.

    This uses the interior model's hydrostatic integration to estimate the planet's
    interior mass from a given radius. The radius is then adjusted until the interior mass
    achieves the target mass provided by the user in the config file.

    For the dummy interior, the radius is simply specified by the user in the config file.
    '''

    # Trivial dummy case
    if config.interior.module == 'dummy':
        hf_row["R_int"] = config.interior.dummy.radius * R_earth
        return

    # Other cases...

    # Initial guess for interior radius
    hf_row["R_int"] = R_earth * 0.9

    # Run the interior module to see what mass we get.
    IC_INTERIOR = 1
    _dt = run_interior(dirs, config, loop_counter, IC_INTERIOR, hf_all, hf_row)

    # Timestep dt should be zero
    print("Timestep:",_dt)


    print(hf_row["M_int"]/M_earth)



def run_interior(dirs:dict, config:Config, IC_INTERIOR:int, hf_all:pd.DataFrame, hf_row:dict):
    '''
    Run interior model
    '''

    # Use the appropriate interior model

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

    # Prevent increasing melt fraction
    if config.atmos_clim.prevent_warming and (IC_INTERIOR==2):
        hf_row["Phi_global"] = min(hf_row["Phi_global"], hf_all.iloc[-1]["Phi_global"])
        hf_row["T_magma"] = min(hf_row["T_magma"], hf_all.iloc[-1]["T_magma"])

    log.debug("    T_magma    = %.1f K"%hf_row["T_magma"])
    log.debug("    Phi_global = %.3f  "%hf_row["Phi_global"])
    log.debug("    F_int      = %.2e" %hf_row["F_int"])
    log.debug("    RF_depth   = %.2e" %hf_row["RF_depth"])

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
