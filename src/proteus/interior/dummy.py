# Dummy interior module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.interior.timestep import next_step
from proteus.utils.constants import M_earth, R_earth, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calculate_simple_core_mass(radius:float, corefrac:float):
    '''
    Calculate the mass of a planet's core based on the density of Earth's.
    '''

    earth_fr = 0.55     # earth core radius fraction
    earth_fm = 0.325    # earth core mass fraction  (https://arxiv.org/pdf/1708.08718.pdf)

    # Estimate core density
    core_rho = (3.0 * earth_fm * M_earth) / (4.0 * np.pi * ( earth_fr * R_earth )**3.0 )  # core density [kg m-3]

    # Get core mass
    core_mass = core_rho * 4.0/3.0 * np.pi * (radius * corefrac )**3.0

    return core_mass


def calculate_simple_mantle_mass(radius:float, corefrac:float)->float:
    '''
    A very simple interior structure model.

    This calculates mantle mass given planetary mass, radius, and core fraction. This
    assumes a core density equal to that of Earth's, and that the planet mass is simply
    the sum of mantle and core.
    '''

    # Volume of mantle shell
    mantle_volume = (4 * np.pi / 3) * (radius**3 - (radius*corefrac)**3)

    # Fixed density [g cm-3]
    # This is roughly representative of the average density across Earth's
    #    upper and lower mantle layers.
    mantle_rho = 4.55

    # Get mass [in SI units]
    mantle_mass = mantle_volume * mantle_rho * 1e3

    log.debug("Total mantle mass = %.2e kg" % mantle_mass)
    if mantle_mass <= 0.0:
        raise Exception("Something has gone wrong (mantle mass is negative)")

    return mantle_mass


# Run the dummy interior module
def RunDummyInt(config:Config, dirs:dict, IC_INTERIOR:int, hf_row:dict, hf_all:pd.DataFrame):

    # Output dictionary
    output = {}
    output["F_int"] = hf_row["F_atm"]

    # Interior structure
    output["M_core"]   = calculate_simple_core_mass(hf_row["R_int"], config.struct.corefrac)
    output["M_mantle"] = calculate_simple_mantle_mass(hf_row["R_int"], config.struct.corefrac)

    # Parameters
    tmp_init = config.interior.dummy.ini_tmagma # Initial magma temperature
    tmp_liq  = 2700.0    # Liquidus
    tmp_sol  = 1700.0    # Solidus
    cp_m     = 1792.0    # Mantle specific heat capacity, J kg-1 K-1
    cp_c     = 880.0     # Core specific heat capacity, J kg-1 K-1
    area     = 4 * np.pi * hf_row["R_int"]**2

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

    # Interior heat capacity [J K-1]
    cp_int = cp_m*output["M_mantle"] + cp_c*output["M_core"]

    # Subtract tidal contribution to the total heat flux.
    #    This energy is generated only in the mantle, not in the core.
    if config.orbit.module == 'dummy':
        output["F_tide"] = config.orbit.dummy.H_tide * output["M_mantle"] / area
    else:
        output["F_tide"] = 0.0
    output["F_int"] -= output["F_tide"]

    # Rate of surface temperature change (this will be negative)
    dTdt = -output["F_int"] * area / cp_int

    # Timestepping
    if IC_INTERIOR==1:
        output["T_magma"] = tmp_init
        dt = 0.0
    else:
        dt = next_step(config, dirs, hf_row, hf_all, 1.0)
        output["T_magma"] = hf_row["T_magma"] + dTdt * dt * secs_per_year

    # Determine the new melt fraction
    output["Phi_global"]        = _calc_phi(output["T_magma"])
    output["M_mantle_liquid"]   = output["M_mantle"] * output["Phi_global"]
    output["M_mantle_solid"]    = output["M_mantle"] - output["M_mantle_liquid"]
    output["RF_depth"]          = output["Phi_global"] * (1- config.struct.corefrac)

    sim_time = hf_row["Time"] + dt
    return sim_time, output
