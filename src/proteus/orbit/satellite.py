# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from proteus.utils.constants import const_G, secs_per_hour

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def Ltot(ω, a, params):
    """
    Total angular momentum of Earth-Moon system.
    """
    I, _, G, Mpl, Msa, _ = params
    return I*ω + Mpl*(G*(Mpl+Msa)*a)**0.5

def dω_dt(a, ω, params):
    """
    ODE describing evolution of Earth rotation based on Eq. 58 from Korenaga (2023).
    """
    I, L, G, Mpl, Msa, dE_tidal = params
    return - dE_tidal / (I * ω + (G * Mpl * Msa * I)/(a*(L - I*ω)))

def da_dt(a, ω, params):
    """
    ODE describing evolution of semimajor axis based on Eq. 59 from Korenaga (2023).
    """
    I, L, *_ = params
    return - 2*I*a / (L-I*ω) * dω_dt(a, ω, params)

def orbitals(t, z, params):
    """
    Helper function for solving coupled ODEs.
    """
    a, ω = z
    return [da_dt(a, ω, params), dω_dt(a, ω, params)]

def update_satellite(hf_row:dict, config:Config, dt:float):
    """Evolve the Satellite's orbital parameters module.

    Updates the semi-major axis and primary rotation
    frequency based on angular momentum conservation.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : dict
            Dictionary of configuration options
        dt : float
            Time interval over which escape is occuring [yr]
    """
    Rpl = hf_row["R_int"]
    Mpl = hf_row["M_int"]

    # Calculate bulk tidal power
    dE_tidal = hf_row["F_tidal"] * 4 * np.pi * Rpl**2 # Js-1

    # Calculate moment of inertia of planet (assuming solid sphere)
    I = 2/5 * Mpl * Rpl**2 # kg.m-1

    # Time step
    current_time = float(hf_row["Time"])

    # Use config parameters as initial guess
    if current_time <= 1:
        # Set satellite semimajor axis, satellite mass, and planet rotation frequency from config.
        hf_row["semimajorax_sat"] = float(config.orbit.semimajoraxis_sat)  # m
        hf_row["M_sat"]           = float(config.orbit.mass_sat)           # kg

        Msa = hf_row["M_sat"]

        if config.orbit.axial_period is None:
            # set by user to 'none', use 1:1 SOR
            hf_row["axial_period"] = float(hf_row["orbital_period"])
        else:
            hf_row["axial_period"] = float(config.orbit.axial_period) * secs_per_hour

        # Calculate system angular momentum
        sma = float(hf_row["semimajorax_sat"])
        omega = 2 * np.pi / float(hf_row["axial_period"])

        hf_row["plan_sat_am"] = I*omega + Mpl*(const_G*(Mpl+Msa)*sma)**0.5
        log.info("    sys.am = %.5f kg.m2.s-1"%(hf_row["plan_sat_am"]))

        return

        # hf_row["plan_sat_am"] = config.orbit.plan_sat_am                    # kg.m2.s-1
        # L = hf_row["plan_sat_am"]

        # hf_row["axial_period"] = 2 * np.pi * I / (L - Mpl*(const_G*(Mpl+Msa)*sma)**0.5)
        # log.info("    axial. = %.5f h "%(hf_row["axial_period"]/secs_per_hour))

        # hf_row["semimajorax_sat"] = ((L - I*omega)/Mpl)**2 / (const_G*(Mpl+Msa))
        # log.info("    smaxis = %.5f km"%(hf_row["semimajorax_sat"]/1000))
    else:
        # Find previous_time from which to evolve orbit to current_time
        previous_time = current_time - dt

    # Set semimajor axis, rotation frequency, satellite mass, and system AM from config.
    sma = float(hf_row["semimajorax_sat"])
    omega = 2 * np.pi / float(hf_row["axial_period"])
    Msa = hf_row["M_sat"]

    # Could be allowed to vary to mimic resonance effects
    L = hf_row["plan_sat_am"]

    # Collect system parameters at previous_time
    params = (I, L, const_G, Mpl, Msa, dE_tidal)

    # Find new satellite semimajor axis and axial frequency using RK5(4) integration method
    log.debug("Integrate satellite's sma and planet's omega with solve_ivp")
    sol = solve_ivp(orbitals, [previous_time, current_time], [sma, omega], args=(params,))

    # Update semimajor axis and axial period
    hf_row["semimajorax_sat"] = sol.y[0][-1]
    hf_row["axial_period"] = 2 * np.pi / sol.y[1][-1]
