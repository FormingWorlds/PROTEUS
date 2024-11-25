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

    # Standard gravitational parameter (planet mass + star mass)
    mu = const_G * (hf_row["M_star"] + hf_row["M_core"] + hf_row["M_mantle"])

    # Semimajor axis in SI units
    a = sma * AU

    # Orbital period [seconds]
    hf_row["period"] = 2 * np.pi * (a*a*a/mu)**0.5


def update_tides(hf_row:dict, config:Config):
    '''
    Update power density associated with tides.
    '''

    if config.orbit.module == 'dummy':
        log.info("    tidal power density: %.1e W kg-1"%(config.orbit.dummy.H_tide))

    # [call to some other tidal heating module goes here]

    else:
        # no interior module selected
        if config.interior.tidal_heat:
            log.warning("Tidal heating is enabled but no orbit module was selected")


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
    update_tides(hf_row, config)
