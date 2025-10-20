# Dummy interior module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.interior.common import Interior_t
from proteus.interior.timestep import next_step
from proteus.utils.constants import secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calculate_simple_mantle_mass(radius:float, corefrac:float, density:float)->float:
    '''
    A very simple interior structure model.

    This calculates mantle mass given planetary mass, radius, and core fraction. This
    assumes a core density equal to that of Earth's, and that the planet mass is simply
    the sum of mantle and core.
    '''

    # Volume of mantle shell
    mantle_volume = (4 * np.pi / 3) * (radius**3 - (radius*corefrac)**3)

    # Get mass [in SI units]
    mantle_mass = mantle_volume * density

    log.debug("Total mantle mass = %.2e kg" % mantle_mass)
    if mantle_mass <= 0.0:
        raise Exception("Something has gone wrong (mantle mass is negative)")

    return mantle_mass


# Run the dummy interior module
def run_dummy_int(config:Config, dirs:dict,
                    hf_row:dict, hf_all:pd.DataFrame, interior_o:Interior_t):

    # Output dictionary
    output = {}
    output["F_int"] = hf_row["F_atm"]

    # Interior structure
    output["M_mantle"] = calculate_simple_mantle_mass(hf_row["R_int"],
                                                      config.struct.corefrac,
                                                      config.interior.dummy.mantle_rho)

    # Physical parameters
    tmp_liq  = config.interior.dummy.mantle_tliq    # Liquidus
    tmp_sol  = config.interior.dummy.mantle_tsol    # Solidus
    tmp_init = config.interior.dummy.ini_tmagma     # Initial magma temperature
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
    cp_int = config.interior.dummy.mantle_cp*output["M_mantle"] \
                + config.struct.core_heatcap*hf_row["M_core"]

    # Subtract tidal contribution to the total heat flux.
    #    This heat energy is generated only in the mantle, not in the core.
    tidal_flux = 0.0
    if config.interior.tidal_heat:
        tidal_flux = interior_o.tides[0] * output["M_mantle"] / area
    output["F_tidal"] = tidal_flux

    # Radiogenic heating constant with time
    output["F_radio"] = config.interior.dummy.H_radio * output["M_mantle"] / area

    # Total flux loss
    F_loss = output["F_int"] - output["F_tidal"] - output["F_radio"]

    # Rate of surface temperature change (this will be negative) [K/s]
    dTdt = -F_loss * area / cp_int

    # Timestepping
    if interior_o.ic==1:
        output["T_magma"] = tmp_init
        dt = 0.0
    else:
        # calculate new time-step [years]
        dt = next_step(config, dirs, hf_row, hf_all, 1.0)

        # limit time-step based on max change to T_magma
        dtmp_max = hf_row["T_magma"] * config.interior.dummy.tmagma_rtol + config.interior.dummy.tmagma_atol
        dt = min(dt, abs(dtmp_max / dTdt) / secs_per_year) # years

        # update T_magma
        output["T_magma"] = hf_row["T_magma"] +  dTdt * dt * secs_per_year

    # Store scalars
    output["T_pot"]             = float(output["T_magma"])
    output["Phi_global"]        = _calc_phi(output["T_magma"])
    output["Phi_global_vol"]    = output["Phi_global"]
    output["M_mantle_liquid"]   = output["M_mantle"] * output["Phi_global"]
    output["M_mantle_solid"]    = output["M_mantle"] - output["M_mantle_liquid"]
    output["RF_depth"]          = output["Phi_global"] * (1- config.struct.corefrac)
    R_core = config.struct.corefrac * hf_row["R_int"]

    # Store arrays
    interior_o.phi     = np.array([output["Phi_global"]])
    interior_o.mass    = np.array([output["M_mantle"]])
    interior_o.visc    = np.array([1.0])    # placeholder to be updated elsewhere
    interior_o.density = np.array([config.interior.dummy.mantle_rho])
    interior_o.temp    = np.array([output["T_magma"]])
    interior_o.pres    = np.array([hf_row["P_surf"]])
    interior_o.radius  = np.array([R_core, hf_row["R_int"]])

    sim_time = hf_row["Time"] + dt
    return sim_time, output
