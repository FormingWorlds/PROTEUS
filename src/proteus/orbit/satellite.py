# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from proteus.utils.constants import AU, const_G

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def Ltot(ω, a, params):
    """
    Total angular momentum of Earth-Moon system.
    """
    I, G, Mpl, Msa, *_ = params
    return I*ω + Mpl*(G*(Mpl+Msa)*a)

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
    Msa = hf_row["M_sat"]

    sma = float(hf_row["semimajorax_sat"])
    omega = float(hf_row["planet_rotation_freq"])

    # Time step
    current_time = float(hf_row["Time"])

    # Use config parameters as initial guess
    if current_time <= 1:
        # Set semimajor axis and rotation frequency from config.
        hf_row["semimajorax_sat"]    = config.orbit.semimajoraxis_sat * AU
        hf_row["planet_rotation_freq"] = config.orbit.rotation_freq
        return
    else:
        # Find previous_time from which to evolve orbit to current_time
        previous_time = current_time - dt

    # Calculate bulk tidal power
    dE_tidal = hf_row["F_tidal"] * 4 * np.pi * Rpl**2 # Js-1

    # Define moment of inertia of Earth
    I = 8.04e37 # kg.m-1
    # Define combined angular momentum of Earth-Moon system
    L = 3.61e34 # kg.m2.s-1

    # Collect system parameters at previous_time
    params = (I, L, const_G, Mpl, Msa, dE_tidal)

    # Find new semimajor axis and eccentricity using RK5(4) integration method
    log.debug("Integrate sma and omega with solve_ivp")
    sol = solve_ivp(orbitals, [previous_time, current_time], [sma, omega], args=(params,))

    # Update semimajor axis and eccentricity
    hf_row["semimajorax_sat"]    = sol.y[0][-1]
    hf_row["planet_rotation_freq"] = 2 * np.pi / sol.y[1][-1]
