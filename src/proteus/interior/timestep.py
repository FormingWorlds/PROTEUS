# Contains routines for setting the model timestep
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Constants
ILOOK = 2       # Ideal number of steps to look back
SFINC = 1.10    # Scale factor for step size increase
SFDEC = 0.75    # Scale factor for step size decrease
SMALL = 1e-10   # Small number

def _estimate_solid(hf_all:pd.DataFrame, i1:int, i2:int) -> float:
    '''
    Estimate the time remaining until the planet solidifies.
    '''

    # HF at times
    h1 = hf_all.iloc[i1]
    h2 = hf_all.iloc[i2]

    # Check if planet has already solidified
    if h2["Phi_global"] < SMALL:
        dt_solid = np.inf

    else:
        # Change in time and global melt frac
        dt = h2["Time"]       - h1["Time"]
        dp = h2["Phi_global"] - h1["Phi_global"]

        # Estimate how long until p=0
        if abs(dp) < SMALL:
            dt_solid = np.inf
        else:
            #  dp/dt * dt + p2 = 0    ->   dt = -p2/(dp/dt)
            dt_solid = abs(-1.0 * h2["Phi_global"] / (dp/dt))

    log.debug("Solidification expected in %.3e yrs"%dt_solid)

    return dt_solid

def _estimate_radeq(hf_all:pd.DataFrame, i1:int, i2:int) -> float:
    '''
    Estimate the time remaining until the energy balance is achieved.
    '''

    # HF at times
    h1 = hf_all.iloc[i1]
    h2 = hf_all.iloc[i2]

    # Flux residuals
    f2 = h2["F_atm"] - h2["F_tidal"] - h2["F_radio"]
    f1 = h1["F_atm"] - h1["F_tidal"] - h1["F_radio"]

    # Check if planet is already at radeq
    if abs(f2) < SMALL:
        dt_radeq = np.inf

    else:
        # Change in time and global melt frac
        dt = h2["Time"] - h1["Time"]
        df = f2 - f1

        # Estimate how long until f=0
        if abs(df) < SMALL:
            dt_radeq = np.inf
        else:
            dt_radeq = abs(-1.0 * f2 / (df/dt))

    log.debug("Energy balance expected in %.3e yrs"%dt_radeq)

    return dt_radeq


def next_step(config:Config, dirs:dict, hf_row:dict, hf_all:pd.DataFrame, step_sf:float) -> float:
    '''
    Determine the size of the next interior time-step.

    Parameters
    -----------
        config : dict
            Dictionary of configuration options
        dirs : dict
            Dictionary of directories
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        hf_all : pd.DataFrame
            Dataframe containing simulation variables (now and historic)
        step_sf : float
            Scale factor to apply to step size


    Returns
    -----------
        dtswitch : float
            Optimal step size [years].
    '''

    # First years, use small step
    if hf_row["Time"] < 2.0:
        dtswitch = 1.0
        log.info("Time-stepping intent: static")

    else:
        # How far to look back?
        if ILOOK >= len(hf_all["Time"]):
            i1 = -2
        else:
            i1 = int(-1 * ILOOK)
        i2 = -1

        # Estimate time until solidification
        dt_solid = _estimate_solid(hf_all, i1, i2)
        dt_radeq = _estimate_radeq(hf_all, i1, i2)

        # Proportional time-step calculation
        if config.params.dt.method == 'proportional':
            log.info("Time-stepping intent: proportional")
            dtswitch = hf_row["Time"] / config.params.dt.proportional.propconst

        # Dynamic time-step calculation
        elif config.params.dt.method == 'adaptive':

            # Try to maintain a minimum step size of dt_initial at first
            if hf_row["Time"] > config.params.dt.initial:
                dtprev = float(hf_all.iloc[-1]["Time"] - hf_all.iloc[-2]["Time"])
            else:
                dtprev = config.params.dt.initial
            log.debug("Previous step size: %.2e yr"%dtprev)

            # Change in F_int
            F_int_2  = hf_all.iloc[i2]["F_int"]
            F_int_1  = hf_all.iloc[i1]["F_int"]
            F_int_12 = abs(F_int_2 - F_int_1)

            # Change in F_atm
            F_atm_2  = hf_all.iloc[i2]["F_atm"]
            F_atm_1  = hf_all.iloc[i1]["F_atm"]
            F_atm_12 = abs(F_atm_2 - F_atm_1)

            # Change in global melt fraction
            phi_2  = hf_all.iloc[i2]["Phi_global"]
            phi_1  = hf_all.iloc[i1]["Phi_global"]
            phi_12 = abs(phi_2 - phi_1)

            # Determine new time-step given the tolerances
            dt_rtol = config.params.dt.adaptive.rtol
            dt_atol = config.params.dt.adaptive.atol
            speed_up = True
            speed_up = speed_up and ( F_int_12 < dt_rtol*abs(F_int_2) + dt_atol )
            speed_up = speed_up and ( F_atm_12 < dt_rtol*abs(F_atm_2) + dt_atol )
            speed_up = speed_up and ( phi_12   < dt_rtol*abs(phi_2  ) + dt_atol )

            if speed_up:
                dtswitch = dtprev * SFINC
                log.info("Time-stepping intent: speed up")
            else:
                dtswitch = dtprev * SFDEC
                log.info("Time-stepping intent: slow down")

            # Do not allow step size to exceed predicted point of termination
            dtswitch = min(dtswitch, dt_solid)
            dtswitch = min(dtswitch, dt_radeq)

        # Always use the maximum time-step, which can be adjusted in the cfg file
        elif config.params.dt.method == 'maximum':
            log.info("Time-stepping intent: maximum")
            dtswitch = config.params.dt.maximum

        # Handle all other inputs
        else:
            UpdateStatusfile(dirs, 20)
            raise Exception(f"Invalid time-stepping method: {config.params.dt.method}")

        # Step scale factor (is always <= 1.0)
        dtswitch *= step_sf

        # Step-size limits
        dtswitch = min(dtswitch, config.params.dt.maximum )    # Ceiling
        dtswitch = max(dtswitch, config.params.dt.minimum )    # Floor

    log.info("New time-step is %1.2e years" % dtswitch)
    return dtswitch
