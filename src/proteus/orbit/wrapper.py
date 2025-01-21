# Generic orbital dynamics stuff
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import AU, const_G, secs_per_day

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def update_separation(hf_row:dict, sma:float, ecc:float):
    '''
    Calculate time-averaged orbital separation on an elliptical path.

    Converts from AU to metres.
    https://physics.stackexchange.com/a/715749

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
        sma: float
            Semimajor axis [AU]
        ecc: float
            Eccentricity
    '''

    hf_row["separation"] = sma * AU *  (1 + 0.5*ecc*ecc)

def update_period(hf_row:dict, sma:float):
    '''
    Calculate orbital period on an elliptical path.

    Assuming that M_volatiles << M_star + M_mantle + M_core.
    https://en.wikipedia.org/wiki/Elliptic_orbit#Orbital_period

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
        sma: float
            Semimajor axis [AU]
    '''

    # Total mass of system, kg
    M_total = hf_row["M_star"] + hf_row["M_tot"]

    # Sanity check
    if M_total < 1e3:
       log.error("Unreasonable star+planet mass: %.5e kg"%M_total)

    # Standard gravitational parameter (planet mass + star mass)
    mu = const_G * M_total

    # Semimajor axis in SI units
    a = sma * AU

    # Orbital period [seconds]
    hf_row["period"] = 2 * np.pi * (a*a*a/mu)**0.5


def run_orbit(hf_row:dict, config:Config, interior_o):
    """Update parameters relating to orbital evolution and tides.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration
        interior_o: Interior_t
            Struct containing interior arrays at current time.
    """


    log.info("Evolve orbit...")

    # Update orbital separation
    update_separation(hf_row, config.orbit.semimajoraxis, config.orbit.eccentricity)

    # Update orbital period
    update_period(hf_row, config.orbit.semimajoraxis)
    log.info("    period: %.1f days"%(hf_row["period"]/secs_per_day))

    # Initialise, set tidal heating to zero
    interior_o.tides = np.zeros(len(interior_o.phi))

    # Call tides module
    if config.orbit.module is None:
        pass

    elif config.orbit.module == 'dummy':
        from proteus.orbit.dummy import run_dummy_tides
        interior_o.tides = run_dummy_tides(config, interior_o.phi)

    else:
        log.error("Unsupported tides module")

    log.info("    median tidal power density: %.1e W kg-1"%np.median(interior_o.tides))
