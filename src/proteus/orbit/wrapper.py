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

def update_energy_budget(hf_row:dict, dirs:dict):
    '''
    Calculate total dissipated energy.

    First calculate the total mass of all shells that make up the mantle, then
    retrieve the density and power density per shell and multiply the results to
    find the total power. The total energy is found by multiplying the power by
    the time between data points.

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
        dirs: dict
            Dictionary of directories.
    '''

    from proteus.interior.spider import get_all_output_times, get_dict_surface_values_for_specific_time

    sim_time = get_all_output_times(dirs["output"])[-1]  # yr, as an integer value
    if sim_time == 0:
        log.info("Skipping energy budget in first iteration")
        return 0
    else:
        last_sim_time = get_all_output_times(dirs["output"])[-2]  # yr, as an integer value

    keys_t = (('data', 'rho_s'), ('data', 'radius_s'), ('data', 'Htidal_s'))

    data_a = get_dict_surface_values_for_specific_time(keys_t, sim_time, indir=dirs["output"])
    data_b = get_dict_surface_values_for_specific_time(keys_t, last_sim_time, indir=dirs["output"])

    if data_a is not None and len(data_a) > 0:
        sep_a = np.abs(np.diff(data_a[1]))
        vol_a = 4/3 * np.pi * ((data_a[1][:-1] + sep_a)**3 - (data_a[1][:-1])**3)

        avg_rho_a = np.convolve(data_a[0], [0.5, 0.5], "valid")
        avg_Htide_a = np.convolve(data_a[2], [0.5, 0.5], "valid")

        sep_b = np.abs(np.diff(data_b[1]))
        vol_b = 4/3 * np.pi * ((data_b[1][:-1] + sep_b)**3 - (data_b[1][:-1])**3)

        avg_rho_b = np.convolve(data_b[0], [0.5, 0.5], "valid")
        avg_Htide_b = np.convolve(data_b[2], [0.5, 0.5], "valid")

        new_tidal = np.sum(vol_a * avg_rho_a * avg_Htide_a)
        last_tidal = np.sum(vol_b * avg_rho_b * avg_Htide_b)
        hf_row["tot_tidal"] = new_tidal

        total_power = last_tidal + new_tidal

        log.info(f"Power: {last_tidal:.2e} --> {new_tidal:.2e} W")

        total_E = total_power * (sim_time - last_sim_time) * 31556926
        hf_row["tot_tid_E"] = total_E

        log.info("Total Power  = %.1e W)"%hf_row["tot_tidal"])
        log.info("Total Energy = %.1e J)"%hf_row["tot_tid_E"])

        return total_E

    else:
        #log.warning("Skipping energy budget")
        raise ValueError('Could not access interior data.')

def run_orbit(hf_row:dict, hf_all:dict, config:Config, dirs:dict, interior_o:Interior_t):
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

        if config.orbit.dummy.E_max:
            try:
                total_E_last = hf_all["tot_tid_E"].iloc[-1]
            except:
                total_E_last = 0

            if total_E_last > config.orbit.dummy.E_max:
                hf_row["tot_tid_E"] = total_E_last
                return

            else:
                total_E = update_energy_budget(hf_row, dirs)

                if total_E > config.orbit.dummy.E_max:
                    log.info("Reached tidal heating budget! (= %.1e J)"%hf_row["tot_tid_E"])
                    return

        from proteus.orbit.dummy import run_dummy_orbit
        hf_row["Imk2"] = run_dummy_orbit(config, interior_o)

    elif config.orbit.module == 'lovepy':
        from proteus.orbit.lovepy import run_lovepy
        hf_row["Imk2"] = run_lovepy(hf_row, dirs, interior_o,
                                        config.orbit.lovepy.visc_thresh)

    # Print info
    log.info("    H_tide = %.1e W kg-1 (mean) "%np.mean(interior_o.tides))
    log.info("    Im(k2) = %.1e "%hf_row["Imk2"])
