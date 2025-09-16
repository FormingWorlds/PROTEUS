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
LBAVG = 3      # Number of steps to average over
SFINC = 1.5    # Scale factor for step size increase
SFDEC = 0.8    # Scale factor for step size decrease
SMALL = 1e-8   # Small number

def _hf_from_iters(hf_all:pd.DataFrame, i1:int, i2:int):
    # Get helpfile rows for two different iterations

    # i2 must be larger than i1
    if i1 >= i2:
        log.error("Cannot compare helpfile rows (i1=%d  >=  i2=%d)"%(i1, i2))

    # Return HF rows at the requested iterations
    return dict(hf_all.iloc[i1]), dict(hf_all.iloc[i2])


def _estimate_solid(hf_all:pd.DataFrame, i1:int, i2:int) -> float:
    '''
    Estimate the time remaining until the planet solidifies.
    '''

    # HF at times
    h1, h2 = _hf_from_iters(hf_all, i1, i2)

    # Melt fractions
    p1 = h1["Phi_global"]
    p2 = h2["Phi_global"]

    # Check if planet has already solidified
    if p2 < SMALL:
        dt_solid = np.inf

    else:
        # Change in time and global melt frac
        dt = h2["Time"] - h1["Time"]
        dp = p2 - p1

        # Estimate how long Δt until p=0
        if abs(dp/p2) < SMALL:
            dt_solid = np.inf
        else:
            #  dp/dt * Δt + p2 = 0    ->   Δt = -p2/(dp/dt)
            dt_solid = abs(-1.0 * p2 / (dp/dt))

    log.debug("Solidification expected in %.3e yrs"%dt_solid)

    return dt_solid

def _estimate_radeq(hf_all:pd.DataFrame, i1:int, i2:int) -> float:
    '''
    Estimate the time remaining until the energy balance is achieved.
    '''

    # HF at times
    h1, h2 = _hf_from_iters(hf_all, i1, i2)

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
        if abs(df/f2) < SMALL:
            dt_radeq = np.inf
        else:
            dt_radeq = abs(-1.0 * f2 / (df/dt))

    log.debug("Energy balance expected in %.3e yrs"%dt_radeq)

    return dt_radeq


def _estimate_escape(hf_all:pd.DataFrame, i1:int, i2:int) -> float:
    '''
    Estimate the time remaining until the surface pressure is zero.
    '''

    # HF at times
    h1, h2 = _hf_from_iters(hf_all, i1, i2)

    # Surface pressures
    p1 = h1["P_surf"]
    p2 = h2["P_surf"]

    # Change in time and global melt frac
    dt = h2["Time"] - h1["Time"]
    dp = p2 - p1

    # Estimate how long Δt until p=0
    if abs(dp/p2) < SMALL:
        # already escaped
        dt_escape = np.inf
    else:
        #  dp/dt * Δt + p2 = 0    ->   Δt = -p2/(dp/dt)
        dt_escape = abs(-1.0 * p2 / (dp/dt))

    log.debug("Escape expected in %.3e yrs"%dt_escape)

    return dt_escape


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

    elif LBAVG+5 >= len(hf_all["Time"]):
        dtswitch = config.params.dt.initial
        log.info("Time-stepping intent: initial")

    else:
        i2 = -1
        i1 = -2
        i0 = i1 - LBAVG

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

            # Change in F_atm
            F_atm_2  = hf_all["F_atm"].iloc[i2]
            F_atm_1  = np.median(hf_all["F_atm"].iloc[i0:i1])
            F_atm_12 = abs(F_atm_2 - F_atm_1)

            # Change in global melt fraction
            phi_2  = hf_all["Phi_global"].iloc[i2]
            phi_1  = np.median(hf_all["Phi_global"].iloc[i0:i1])
            phi_12 = abs(phi_2 - phi_1)

            # Determine new time-step given the tolerances
            dt_rtol = config.params.dt.adaptive.rtol
            dt_atol = config.params.dt.adaptive.atol
            speed_up = True
            speed_up = speed_up and ( F_atm_12 < dt_rtol*abs(F_atm_2) + dt_atol )
            speed_up = speed_up and ( phi_12   < dt_rtol*abs(phi_2  ) + dt_atol )

            if speed_up:
                dtswitch = dtprev * SFINC
                log.info("Time-stepping intent: speed up")
            else:
                dtswitch = dtprev * SFDEC
                log.info("Time-stepping intent: slow down")

            # Do not allow step size to exceed predicted point of termination
            if config.params.stop.solid.enabled:
                dtswitch = min(dtswitch, _estimate_solid(hf_all, i1, i2))
            if config.params.stop.radeqm.enabled:
                dtswitch = min(dtswitch, _estimate_radeq(hf_all, i1, i2))
            if config.params.stop.escape.enabled:
                dtswitch = min(dtswitch, _estimate_escape(hf_all, i1, i2)*1.1)

        # Always use the maximum time-step, which can be adjusted in the cfg file
        elif config.params.dt.method == 'maximum':
            log.info("Time-stepping intent: maximum")
            dtswitch = config.params.dt.maximum

        # Handle all other inputs
        else:
            UpdateStatusfile(dirs, 20)
            raise ValueError(f"Invalid time-stepping method: {config.params.dt.method}")

        # Step scale factor (is always <= 1.0)
        dtswitch *= step_sf

        # Max step size
        dtswitch = min(dtswitch, config.params.dt.maximum )

        # Min step size
        dtminimum = config.params.dt.minimum
        if step_sf > 0.99:
            dtminimum += config.params.dt.minimum_rel * hf_row["Time"]
        dtswitch = max(dtswitch, dtminimum)

    log.info("New time-step target is %.2e years" % dtswitch)
    return dtswitch
