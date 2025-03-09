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
    module = str(handler.config.orbit.module)
    if module == "None":
        return

    log.info(f"Preparing orbit/tides model '{module}'")
    if not handler.config.interior.tidal_heat:
        log.warning("Tidal heating is disabled within interior configuration!")

    if module == "lovepy":
        from proteus.orbit.lovepy import import_lovepy
        import_lovepy()

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
    Calculate orbital and axial periods, on an elliptical path.

    Assuming that M_volatiles << M_star + M_mantle + M_core.
    https://en.wikipedia.org/wiki/Elliptic_orbit#Orbital_period

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
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
    hf_row["orbital_period"] = 2 * np.pi * (sma*sma*sma/mu)**0.5

    # Axial period [seconds]
    #   Assuming that the planet is tidally locked
    hf_row["axial_period"] = hf_row["orbital_period"]

def update_hillradius(hf_row:dict):
    '''
    Calculate Hill radius.

    Using equation from: http://astro.vaporia.com/start/hillradius.html

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    '''

    sma = hf_row["semimajorax"]
    ecc = hf_row["eccentricity"]
    Mpl = hf_row["M_int"]
    Mst = hf_row["M_star"]

    hf_row["hill_radius"] = sma * (1-ecc) * (Mpl/(3*Mst))**(1.0/3)

def update_rochelimit(hf_row:dict):
    '''
    Calculate Roche limit.

    Using equation from: http://astro.vaporia.com/start/rochelimit.html

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    '''

    Rpl = hf_row["R_int"]
    Mpl = hf_row["M_int"]
    Mst = hf_row["M_star"]

    hf_row["roche_limit"] = Rpl * (2 * Mst/Mpl)**(1.0/3)


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

    # Update orbital separation, orbital period, axial period
    update_separation(hf_row)
    update_period(hf_row)
    log.info("    period = %.3f days"%(hf_row["orbital_period"]/secs_per_day))

    # Update Roche limit
    update_rochelimit(hf_row)
    if hf_row["separation"] < hf_row["roche_limit"]:
        log.warning("Planet is orbiting within the Roche limit of its star")

    # Update Hill radius
    update_hillradius(hf_row)
    if max(hf_row["R_obs"], hf_row["R_xuv"]) > hf_row["hill_radius"]:
        log.warning("Atmosphere extends beyond the Hill radius")

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
        hf_row["Imk2"] = run_lovepy(hf_row, dirs, interior_o,
                                        config.orbit.lovepy.visc_thresh)

    # Print info
    log.info("    H_tide = %.1e W kg-1 (mean) "%np.mean(interior_o.tides))
    log.info("    Im(k2) = %.1e "%hf_row["Imk2"])
