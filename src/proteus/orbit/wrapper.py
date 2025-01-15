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


def update_tides(hf_row:dict, config:Config):
    '''
    Update power density associated with tides.
    '''

    # Number of interior levels (0 if using dummy interior)
    interior_nlev = 0
    if config.interior.module == "spider":
        interior_nlev = config.interior.spider.num_levels - 1
    elif config.interior.module == "aragog":
        interior_nlev = config.interior.aragog.num_levels - 1

    # Melt fraction (must be array, can have length=1)
    if interior_nlev == 0:
        phi_array = np.array([hf_row["Phi_global"]])
        tides_array = np.array([0.0]) # default value
    else:
        phi_array = np.ones(interior_nlev) * hf_row["Phi_global"] # PLACEHOLDER - FIX ME!!
        log.warning("fix me ^^")
        tides_array = np.zeros(interior_nlev) # default value

    # Call tides module
    if config.orbit.module is None:
        pass

    elif config.orbit.module == 'dummy':
        from proteus.orbit.dummy import run_dummy_tides
        tides_array = run_dummy_tides(config, phi_array)

    else:
        log.error("Unsupported tides module")

    log.info("    median tidal power density: %.1e W kg-1"%np.median(tides_array))
    return tides_array


def run_orbit(hf_row:dict, config:Config):
    '''
    Update parameters relating to orbital evolution and tides.
    '''

    log.info("Evolve orbit...")

    # Update orbital separation
    update_separation(hf_row, config.orbit.semimajoraxis, config.orbit.eccentricity)

    # Update orbital period
    update_period(hf_row, config.orbit.semimajoraxis)
    log.info("    period: %.1f days"%(hf_row["period"]/secs_per_day))

    # Update tidal heating
    tides = update_tides(hf_row, config)

    return tides
