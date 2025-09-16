# Generic orbital dynamics stuff
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior.common import Interior_t
from proteus.utils.constants import AU, L_sun, R_sun, const_G, secs_per_day, secs_per_hour

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

    log.info(f"Preparing tides model '{module}'")
    if not handler.config.interior.tidal_heat:
        log.warning("Tidal heating is disabled within interior configuration!")

    if module == "lovepy":
        from proteus.orbit.lovepy import import_lovepy
        import_lovepy()

def update_separation(hf_row:dict):
    '''
    Calculate time-averaged orbital separation on an elliptical path.
    https://physics.stackexchange.com/a/715749

    Calculate periapsis distance on an elliptical path.
    https://mathworld.wolfram.com/Periapsis.html

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    '''

    sma = hf_row["semimajorax"] # already in SI units
    ecc = hf_row["eccentricity"]

    sma_sat = hf_row["semimajorax_sat"] # already in SI units

    # Time-averaged separation
    hf_row["separation"] = sma * (1 + 0.5*ecc*ecc)

    # Periapsis distance around star
    hf_row["perihelion"] = sma * (1 - ecc)

    # Periapsis distance around planet (assuming circular orbiting satellite)
    hf_row["perigee"]    = sma_sat

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

def update_breakup_period(hf_row:dict):
    '''
    Calculate Breakup period.

    Using equation from: https://arxiv.org/abs/2508.09273
    (Note, the equation contains a typo, it should
    read: 2pi/T = Ω = sqrt( G Mp / Rp^3 ). )

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    '''

    Rpl = hf_row["R_int"]
    Mpl = hf_row["M_int"]

    hf_row["breakup_period"] = 2*np.pi/np.sqrt(const_G*Mpl/(Rpl**3))


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

    # Set semimajor axis and eccentricity, through the desired method...
    if config.orbit.evolve:
        # set by orbital evolution, based on tidal love number
        from proteus.orbit.orbit import evolve_orbital
        evolve_orbital(hf_row, config, interior_o.dt)

    else:
        # orbital parameters are held constant over time
        hf_row["eccentricity"] = config.orbit.eccentricity
        hf_row["semimajorax"]  = config.orbit.semimajoraxis * AU

        # set semi-major axis to obtain a particular bolometric instellation flux
        if config.orbit.instellation_method == 'inst' and config.star.module == 'dummy':
            from proteus.star.dummy import calc_star_luminosity, get_star_radius

            Lbol = calc_star_luminosity(config.star.dummy.Teff, get_star_radius(config)*R_sun)
            S_earth = L_sun / (4 * np.pi * AU * AU)
            S_0 = config.orbit.instellationflux * S_earth

            hf_row["semimajorax"] = np.sqrt( Lbol / (4 * np.pi * S_0))

    # Inform user
    log.info("    Orb SMaxis = %.5f AU"%(hf_row["semimajorax"]/AU))
    log.info("    Orb eccent = %.5f   "%(hf_row["eccentricity"]))

    # Update orbital separation and period, from other variables above
    update_separation(hf_row)
    update_period(hf_row)

    log.info("    Orb period = %.5f days"%(hf_row["orbital_period"]/secs_per_day))

    if config.orbit.satellite:
        # set by orbital evolution, based on tidal love number
        from proteus.orbit.satellite import update_satellite
        update_satellite(hf_row, config, interior_o.dt)

    else:
        # Satellite SMA
        hf_row["semimajorax_sat"] = float(config.orbit.semimajoraxis_sat)

        # Axial period [seconds]
        if config.orbit.axial_period is None:
            # set by user to 'none', use 1:1 SOR
            hf_row["axial_period"] = hf_row["orbital_period"]
        else:
            # set by user with float, use that
            hf_row["axial_period"] = float(config.orbit.axial_period) * secs_per_hour

    # Update Breakup period
    update_breakup_period(hf_row)
    if hf_row["axial_period"] <= hf_row["breakup_period"] + float(config.params.stop.disint.offset_spin):
        log.warning("Planet is spinning faster than the Breakup rate")

    # Update Roche limit
    update_rochelimit(hf_row)
    if hf_row["separation"] <= hf_row["roche_limit"] + float(config.params.stop.disint.offset_roche):
        log.warning("Planet is orbiting within the Roche limit of its star")
    elif hf_row["perihelion"] <= hf_row["roche_limit"] + float(config.params.stop.disint.offset_roche):
        log.warning("Planet is (partially) orbiting within the Roche limit of its star")

    # Update Hill radius
    update_hillradius(hf_row)
    if max(hf_row["R_obs"], hf_row["R_xuv"]) > hf_row["hill_radius"]:
        log.warning("Atmosphere extends beyond the Hill radius")

    # Initialise, set tidal heating to zero
    interior_o.tides = np.zeros(len(interior_o.phi))

    # Call tides module, calculates heating rates and new love number
    if config.orbit.module == 'dummy':
        from proteus.orbit.dummy import run_dummy_orbit
        hf_row["Imk2"] = run_dummy_orbit(config, interior_o)

    elif config.orbit.module == 'lovepy':
        from proteus.orbit.lovepy import run_lovepy
        hf_row["Imk2"] = run_lovepy(hf_row, dirs, interior_o, config)

    else:
        hf_row["Imk2"] = 0.0

    # Print info
    if config.orbit.module is not None:
        log.info("    Pla H_tide = %.1e W kg-1 (mean) "%np.mean(interior_o.tides))
        log.info("    Pla Im(k2) = %.1e "%hf_row["Imk2"])

    # Call tides module for satellite, calculates heating rates and new love number
    # To Do
