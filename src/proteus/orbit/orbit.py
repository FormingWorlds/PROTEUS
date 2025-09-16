# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scipy.integrate import solve_ivp

from proteus.utils.constants import AU, const_G

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def de_dt(a, e, params):
    """
    ODE describing evolution of orbital eccentricity based on Eq. 16 from Driscoll & Barnes (2015).
    """
    Imk2, Mst, G, Rpl, Mpl = params
    return (21/2) * Imk2 * Mst**1.5 * G**0.5 * Rpl**5 / (Mpl * a**6.5) * e

def da_dt(a, e, params):
    """
    ODE describing evolution of semimajor axis based on Eq. 15 from Driscoll & Barnes (2015).
    """
    return 2 * a * e * de_dt(a, e, params)

def orbitals(t, z, params):
    """
    Helper function for solving coupled ODEs.
    """
    a, e = z
    return [da_dt(a, e, params), de_dt(a, e, params)]


def evolve_orbital(hf_row:dict, config:Config, dt:float):
    """Evolve the planet's orbital parameters module.

    Updates the semi-major axis and eccentricity.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : dict
            Dictionary of configuration options
        dt : float
            Time interval over which escape is occuring [yr]
    """
    Imk2 = hf_row["Imk2"]

    Rpl = hf_row["R_int"]
    Mpl = hf_row["M_int"]
    Mst = hf_row["M_star"]

    sma = float(hf_row["semimajorax"])
    ecc = float(hf_row["eccentricity"])

    # Time step
    current_time = float(hf_row["Time"])

    # Use config parameters as initial guess
    if current_time <= 1:
        # Set semimajor axis and eccentricity from config.
        hf_row["semimajorax"]  = config.orbit.semimajoraxis * AU
        hf_row["eccentricity"] = config.orbit.eccentricity
        return
    else:
        # Find previous_time from which to evolve orbit to current_time
        previous_time = current_time - dt

    # Collect system parameters at previous_time
    params = (Imk2, Mst, const_G, Rpl, Mpl)

    # Find new semimajor axis and eccentricity using RK5(4) integration method
    log.debug("Integrate sma and ecc with solve_ivp")
    sol = solve_ivp(orbitals, [previous_time, current_time], [sma, ecc], args=(params,))

    # Update semimajor axis and eccentricity
    hf_row["semimajorax"]  = sol.y[0][-1]
    hf_row["eccentricity"] = sol.y[1][-1]
