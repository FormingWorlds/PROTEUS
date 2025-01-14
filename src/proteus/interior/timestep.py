# Contains routines for setting the model timestep
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

ILOOK = 4 # Number of steps to use for extrapolation fitting

def _estimate_solidify(hf_all:pd.DataFrame) -> float:
    '''
    Estimate the time remaining until the planet solidifies.
    '''




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
        # Estimate solidification time


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

            if ILOOK >= len(hf_all["Time"]):
                i2 = -2
            else:
                i2 = int(-1 * ILOOK)
            i1 = -1

            # Change in F_int
            F_int_2  = hf_all.iloc[i2]["F_int"]
            F_int_1  = hf_all.iloc[i1]["F_int"]
            F_int_12 = abs(F_int_1 - F_int_2)

            # Change in F_atm
            F_atm_2  = hf_all.iloc[i2]["F_atm"]
            F_atm_1  = hf_all.iloc[i1]["F_atm"]
            F_atm_12 = abs(F_atm_1 - F_atm_2)

            # Change in global melt fraction
            phi_2  = hf_all.iloc[i2]["Phi_global"]
            phi_1  = hf_all.iloc[i1]["Phi_global"]
            phi_12 = abs(phi_1 - phi_2)

            # Determine new time-step given the tolerances
            dt_rtol = config.params.dt.adaptive.rtol
            dt_atol = config.params.dt.adaptive.atol
            speed_up = True
            speed_up = speed_up and ( F_int_12 < dt_rtol*abs(F_int_2) + dt_atol )
            speed_up = speed_up and ( F_atm_12 < dt_rtol*abs(F_atm_2) + dt_atol )
            speed_up = speed_up and ( phi_12   < dt_rtol*abs(phi_2  ) + dt_atol )

            if speed_up:
                dtswitch = dtprev * 1.1
                log.info("Time-stepping intent: speed up")
            else:
                dtswitch = dtprev * 0.75
                log.info("Time-stepping intent: slow down")


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

        # Step-size ceiling
        dtswitch = min(dtswitch, config.params.dt.maximum )    # Absolute

        # Step-size floor
        dtswitch = max(dtswitch, hf_row["Time"]*0.0001)     # Relative
        dtswitch = max(dtswitch, config.params.dt.minimum )    # Absolute

    log.info("New time-step is %1.2e years" % dtswitch)
    return dtswitch
