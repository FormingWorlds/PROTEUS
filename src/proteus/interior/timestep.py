# Contains routines for setting the model timestep
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def next_step(config:Config, dirs:dict, hf_row:dict, hf_all:pd.DataFrame, step_sf:float) -> float:

    # Time stepping adjustment
    if hf_row["Time"] < 2.0:
        # First year, use small step
        dtswitch = 1.0
        log.info("Time-stepping intent: static")

    else:
        if config.params.dt.method == 'proportional':
            # Proportional time-step calculation
            log.info("Time-stepping intent: proportional")
            dtswitch = hf_row["Time"] / config.params.dt.proportional.propconst

        elif config.params.dt.method == 'adaptive':
            # Dynamic time-step calculation

            # Try to maintain a minimum step size of dt_initial at first
            if hf_row["Time"] > config.params.dt.initial:
                dtprev = float(hf_all.iloc[-1]["Time"] - hf_all.iloc[-2]["Time"])
            else:
                dtprev = config.params.dt.initial
            log.debug("Previous step size: %.2e yr"%dtprev)

            # Change in F_int
            F_int_2  = hf_all.iloc[-2]["F_int"]
            F_int_1  = hf_all.iloc[-1]["F_int"]
            F_int_12 = abs(F_int_1 - F_int_2)

            # Change in F_atm
            F_atm_2  = hf_all.iloc[-2]["F_atm"]
            F_atm_1  = hf_all.iloc[-1]["F_atm"]
            F_atm_12 = abs(F_atm_1 - F_atm_2)

            # Change in global melt fraction
            phi_2  = hf_all.iloc[-2]["Phi_global"]
            phi_1  = hf_all.iloc[-1]["Phi_global"]
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


        elif config.params.dt.method == 'maximum':
            # Always use the maximum time-step, which can be adjusted in the cfg file
            log.info("Time-stepping intent: maximum")
            dtswitch = config.params.dt.maximum

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
