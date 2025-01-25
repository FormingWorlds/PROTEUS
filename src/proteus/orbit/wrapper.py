# Generic orbital dynamics stuff
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior.common import Interior_t
from proteus.utils.constants import AU, const_G, secs_per_day

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def init_orbit(handler:Proteus):
    '''
    Initialise orbit and tides stuff.
    '''

    log.info("Preparing orbit/tides model")

    if handler.config.orbit.module == "lovepy":
        import os

        from proteus.orbit.lovepy import import_lovepy

        lib = os.path.join(handler.directories["proteus"], "src/proteus/orbit/lovepy.jl")
        import_lovepy(lib)


def update_separation(hf_row:dict):
    '''
    Calculate time-averaged orbital separation on an elliptical path.
    https://physics.stackexchange.com/a/715749

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    '''

    sma = hf_row["semimajorax"] # already in SI units
    ecc = hf_row["eccentricity"]

    hf_row["separation"] = sma *  (1 + 0.5*ecc*ecc)

def update_period(hf_row:dict):
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

    # Semimajor axis is already in SI units
    sma = hf_row["semimajorax"]

    # Orbital period [seconds]
    hf_row["period"] = 2 * np.pi * (sma*sma*sma/mu)**0.5


def run_orbit(hf_row:dict, config:Config, dirs:dict, interior_o:Interior_t):
    """Update parameters relating to orbital evolution and tides.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration.
        dirs: dict
            Dictionary of directories.
        interior_o: Interior_t
            Struct containing interior arrays at current time.
    """

    log.info("Evolve orbit and tides...")

    # Set semimajor axis and eccentricity.
    #    In the future, these could be allowed to evolve in time.
    hf_row["semimajorax"]  = config.orbit.semimajoraxis * AU
    hf_row["eccentricity"] = config.orbit.eccentricity

    # Update orbital separation and period
    update_separation(hf_row)
    update_period(hf_row)
    log.info("    period = %.3f days"%(hf_row["period"]/secs_per_day))

    # Exit here if not modelling tides
    if config.orbit.module is None:
        return

    # Initialise, set tidal heating to zero
    interior_o.tides = np.zeros(len(interior_o.phi))

    # Call tides module
    if config.orbit.module == 'dummy':
        from proteus.orbit.dummy import run_dummy_orbit
        hf_row["Imk2"] = run_dummy_orbit(config, interior_o)

    elif config.orbit.module == 'lovepy':
        from proteus.orbit.lovepy import run_lovepy
        hf_row["Imk2"] = run_lovepy(hf_row, dirs, interior_o)

    # Print info
    log.info("    H_tide = %.1e W kg-1 (mean) "%np.mean(interior_o.tides))
    log.info("    Im(k2) = %.1e "%hf_row["Imk2"])
