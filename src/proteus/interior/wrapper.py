# Generic interior wrapper
from __future__ import annotations

import logging

import pandas as pd

from proteus.interior.dummy import RunDummyInt
from proteus.interior.spider import ReadSPIDER, RunSPIDER
from proteus.utils.helper import PrintHalfSeparator

log = logging.getLogger("fwl."+__name__)

def RunInterior(dirs:dict, OPTIONS:dict, loop_counter:dict, IC_INTERIOR:int, hf_all:pd.DataFrame, hf_row:dict):
    '''
    Run interior model
    '''

    PrintHalfSeparator()

    # Use the appropriate interior model
    if OPTIONS["interior_model"] == 0:
        # Run SPIDER
        RunSPIDER(dirs, OPTIONS, IC_INTERIOR, loop_counter, hf_all, hf_row)
        sim_time, output = ReadSPIDER(dirs, OPTIONS, hf_row["R_planet"])

    elif OPTIONS["interior_model"] == 1:
        # Not supported
        raise Exception("Aragog interface not yet implemented")

    elif OPTIONS["interior_model"] == 2:
        # Run dummy interior
        sim_time, output = RunDummyInt(OPTIONS, dirs, IC_INTERIOR, hf_row, hf_all)

    # Read output
    for k in output.keys():
        if k in hf_row.keys():
            hf_row[k] = output[k]

    # Prevent increasing melt fraction
    if (OPTIONS["prevent_warming"] == 1) and (loop_counter["total"] >= loop_counter["init_loops"]):
        hf_row["Phi_global"] = min(hf_row["Phi_global"], hf_all.iloc[-1]["Phi_global"])
        hf_row["T_magma"] = min(hf_row["T_magma"], hf_all.iloc[-1]["T_magma"])

    log.debug("    T_magma    = %.1f K"%hf_row["T_magma"])
    log.debug("    Phi_global = %.3f  "%hf_row["Phi_global"])

    # Time step size
    dt = float(sim_time) - hf_row["Time"]

    # Return timestep
    return dt
