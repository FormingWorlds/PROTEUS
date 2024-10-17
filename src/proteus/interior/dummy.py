# Dummy interior module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.interior.timestep import next_step
from proteus.utils.constants import secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Run the dummy interior module
def RunDummyInt(config:Config, dirs:dict, IC_INTERIOR:int, hf_row:dict, hf_all:pd.DataFrame):
    log.info("Running dummy interior...")

    # Output dictionary
    output = {}
    output["M_mantle"]  = hf_row["M_mantle"]
    output["M_core"]    = hf_row["M_core"]
    output["F_int"]     = hf_row["F_atm"]

    # Parameters
    tmp_init = 3500.0    # Initial magma temperature
    tmp_liq  = 2700.0    # Liquidus
    tmp_sol  = 1700.0    # Solidus
    cp_m     = 1792.0    # Mantle heat capacity, J kg-1 K-1
    cp_c     = 880.0     # Core heat capacity, J kg-1 K-1
    area     = 4 * np.pi * hf_row["R_planet"]**2

    # Get mantle melt fraction as a function of temperature
    def _calc_phi(tmp:float):
        # Too hot
        if tmp >= tmp_liq:
            return 1.0
        # Too cold
        elif tmp <= tmp_sol:
            return 0.0
        # Just right
        return  (tmp-tmp_sol)/(tmp_liq-tmp_sol )

    # Rate of surface temperature change
    dTdt = output["F_int"] * area / (cp_m*output["M_mantle"] + cp_c*output["M_core"])

    # Timestepping
    if IC_INTERIOR==1:
        output["T_magma"] = tmp_init
        dt = 0.0
    else:
        dt = next_step(config, dirs, hf_row, hf_all, 1.0)
        output["T_magma"] = hf_row["T_magma"] - dTdt * dt * secs_per_year

    # Determine the new melt fraction
    output["Phi_global"]        = _calc_phi(output["T_magma"])
    output["M_mantle_liquid"]   = output["M_mantle"] * output["Phi_global"]
    output["M_mantle_solid"]    = output["M_mantle"] - output["M_mantle_liquid"]
    output["RF_depth"]          = output["Phi_global"] * (1- config.struct.corefrac)

    sim_time = hf_row["Time"] + dt
    return sim_time, output
