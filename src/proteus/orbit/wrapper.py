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

def update_energy_budget(hf_row:dict, dirs:dict, dt:float, total_E_last:float):
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
        dt : float
            Time interval [yr]
        total_E_last: float
            Energy disipated by tides up till previous timestep.
    '''

    # Time step
    current_time = float(hf_row["Time"])

    if current_time <= 1:
        log.info("Skipping energy budget in first iterations")
        return 0
    else:
        # Find previous_time from which to evolve orbit to current_time
        previous_time = current_time - dt

    # Define keys for data extraction
    keys_t = (('data', 'rho_s'), ('data', 'radius_s'), ('data', 'Htidal_s'))

    # Extract interior data
    from proteus.interior.spider import get_dict_surface_values_for_specific_time
    data_previous = get_dict_surface_values_for_specific_time(keys_t, previous_time, indir=dirs["output"])
    area_current  = get_dict_surface_values_for_specific_time((('data', 'area_b'),), previous_time, indir=dirs["output"])[0]

    # Calculate power input by tides at current time
    power_current = hf_row["F_tidal"] * area_current[0] # current power

    # Find the increase in energy dissipated by tides
    if data_previous is not None and len(data_previous) > 0:

        # Calculate the volume of the interior shells
        sep_b = np.abs(np.diff(data_previous[1]))
        vol_b = 4/3 * np.pi * ((data_previous[1][1:] + sep_b)**3 - (data_previous[1][1:])**3)

        # Find the density and tidal power input in each shell
        avg_rho_b = np.convolve(data_previous[0], [0.5, 0.5], "valid")
        avg_Htide_b = np.convolve(data_previous[2], [0.5, 0.5], "valid")

        # Calculate the power input by tides at previous time
        power_previous = np.sum(vol_b * avg_rho_b * avg_Htide_b)

        # Calculate the mean power and energy input between current and previous time
        hf_row["tot_tidal"] = power_current
        mean_power = 0.5 * (power_current + power_previous)
        mean_E = mean_power * dt * 31556926

        # Update log file
        log.info(f"    Time : {previous_time:.2e} --> {current_time:.2e} yr")
        log.info(f"    Power: {power_previous:.2e} --> {power_current:.2e} W")
        log.info("Mean  Power  = %.1e W)"%mean_power)
        log.info("Mean  Energy = %.1e J)"%mean_E)

        return total_E_last + mean_E

    else:
        #log.warning("Skipping energy budget")
        raise ValueError('Could not access interior data.')

def log_interp(zz, xx, yy):
    '''
    Interpolate based on log scale value zz between xx and yy.
    '''
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

def shutdown_val(trigger:str, hf_row:dict, config:Config):
    '''
    Calculate the shutdown factor.

    Parameters
    -------------
        trigger: str
            parameter that triggered shutdown
        hf_row: dict
            Current helpfile row
        config : dict
            Dictionary of configuration options
    '''
    # Define critical, maximum, and current value of trigger parameter
    if trigger == "t":
        icrit = config.orbit.dummy.t_crit
        imax  = config.orbit.dummy.t_max
        current = hf_row["Time"]

    elif trigger == "E":
        icrit = config.orbit.dummy.E_crit
        imax  = config.orbit.dummy.E_max
        current = hf_row["tot_tid_E"]

    # Define profile used for shutdown
    profile = config.orbit.dummy.shutdown

    # Calculate shutdown factor
    if profile == "direct" or current > imax:
        shutdown = 0

    elif profile == "linear":
        shutdown = np.interp(current, [icrit, imax], [1, 0])

    elif profile == "log":
        shutdown = log_interp(current, [icrit, imax], [1e0, 1e-7])

    elif profile == "quadratic":
        shutdown = np.interp(current**2, [icrit**2, imax**2], [1, 0])

    # Update log file
    log.info(f"Shutdown active: H_tide {config.orbit.dummy.H_tide:.2e} --> {shutdown*config.orbit.dummy.H_tide:.2e} W kg-1")

    return shutdown

def run_orbit(hf_row:dict, hf_all:dict, config:Config, dirs:dict, interior_o:Interior_t):
    """Update parameters relating to orbital evolution and tides.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        hf_all : dict
            Dictionary of all runtime variables
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

        # Set shutdown factor
        shutdown = 1 # Default case without shutdown

        if config.orbit.dummy.shutdown:
            # Past tidal energy values
            try:
                energy_values = np.array(hf_all["tot_tid_E"])
                total_E_last = energy_values[-1]
            except (KeyError, IndexError):
                energy_values = np.array([0])
                total_E_last = 0


            # Current tidal energy value
            total_E = update_energy_budget(hf_row, dirs, interior_o.dt, total_E_last)
            hf_row["tot_tid_E"] = total_E

            # Time shutdown condition
            t_crit = config.orbit.dummy.t_crit
            if t_crit == 0:
                # shutdown dissabled
                t_shutdown = 1

            else:
                # Shutdown enabled
                if hf_row["Time"] > t_crit:
                    trigger = "t"
                    log.info("Tidal shutdown active!")
                    # Calculate shutdown factor based on scaling
                    t_shutdown = shutdown_val(trigger, hf_row, config)
                else:
                    t_shutdown = 1

            # Energy shutdown condition
            E_crit = config.orbit.dummy.E_crit
            if E_crit == 0:
                # shutdown dissabled
                E_shutdown = 1

                # Critical energy is irrelevant
                crit = None

            elif E_crit == 1:
                # Find largest increase in dissipated energy
                largest_step = np.max(np.diff(np.append(energy_values, total_E)))

                # Update critical energy value
                crit = config.orbit.dummy.E_max - 1.1 * largest_step

            else:
                # Critical energy from config
                crit = E_crit

            if crit:
                # Shutdown enabled
                if hf_row["tot_tid_E"] > crit:
                    trigger = "E"
                    log.info("Tidal shutdown active!")
                    # Calculate shutdown factor based on scaling
                    E_shutdown = shutdown_val(trigger, hf_row, config)
                else:
                    E_shutdown = 1

            # Find most restrictive shutdown factor
            shutdown = min(t_shutdown, E_shutdown)

            # Update log file
            log.info("Shutdown fraction = %.3e J)"%shutdown)
            log.info("Total Energy = %.1e J)"%hf_row["tot_tid_E"])

        from proteus.orbit.dummy import run_dummy_orbit
        hf_row["Imk2"] = run_dummy_orbit(config, interior_o, shutdown)

    elif config.orbit.module == 'lovepy':
        from proteus.orbit.lovepy import run_lovepy
        hf_row["Imk2"] = run_lovepy(hf_row, dirs, interior_o,
                                        config.orbit.lovepy.visc_thresh)

    # Print info
    log.info("    H_tide = %.1e W kg-1 (mean) "%np.mean(interior_o.tides))
    log.info("    Im(k2) = %.1e "%hf_row["Imk2"])
