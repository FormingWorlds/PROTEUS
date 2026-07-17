# Generic orbital dynamics stuff
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t
from proteus.utils.constants import (
    AU,
    L_sun,
    M_earth,
    R_earth,
    R_sun,
    const_G,
    secs_per_day,
    secs_per_hour,
)

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def init_orbit(handler: Proteus):
    """
    Initialise orbit and tides stuff.
    """
    module = str(handler.config.orbit.module)
    if module == 'None':
        return

    log.info(f"Preparing tides model '{module}'")
    if not handler.config.interior_energetics.heat_tidal:
        log.warning('Tidal heating is disabled within interior configuration!')

    if module == 'lovepy':
        from proteus.orbit.lovepy import import_lovepy

        import_lovepy()
    elif module == "obliqua":
        from proteus.orbit.obliqua import import_obliqua
        import_obliqua()


def update_separation(hf_row: dict):
    """
    Calculate time-averaged orbital separation on an elliptical path.
    https://physics.stackexchange.com/a/715749

    Calculate periapsis distance on an elliptical path.
    https://mathworld.wolfram.com/Periapsis.html

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    """

    sma = hf_row['semimajorax']  # already in SI units
    ecc = hf_row['eccentricity']

    sma_sat = hf_row['semimajorax_sat']  # already in SI units

    # Time-averaged separation
    hf_row['separation'] = sma * (1 + 0.5 * ecc * ecc)

    # Periapsis distance around star
    hf_row['perihelion'] = sma * (1 - ecc)

    # Periapsis distance around planet (assuming circular orbiting satellite)
    hf_row['perigee'] = sma_sat


def update_period(hf_row: dict):
    """
    Calculate orbital and axial periods, on an elliptical path.

    Assuming that M_volatiles << M_star + M_mantle + M_core.
    https://en.wikipedia.org/wiki/Elliptic_orbit#Orbital_period

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    """

    # Total mass of system, kg
    M_total = hf_row['M_star'] + hf_row['M_planet']

    # Sanity check
    if M_total < 1e3:
        log.error('Unreasonable star+planet mass: %.5e kg' % M_total)

    # Standard gravitational parameter (planet mass + star mass)
    mu = const_G * M_total

    # Semimajor axis is already in SI units
    sma = hf_row['semimajorax']

    # Orbital period [seconds]
    hf_row['orbital_period'] = 2 * np.pi * (sma * sma * sma / mu) ** 0.5


def update_period_sat(hf_row: dict):
    """
    Calculate orbital and axial periods, on an elliptical path for satellite.

    Assuming that M_volatiles << M_satellite + M_mantle + M_core.
    https://en.wikipedia.org/wiki/Elliptic_orbit#Orbital_period

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    """

    # Total mass of system, kg
    M_total = hf_row['M_planet'] + hf_row['M_sat']

    # Sanity check
    if M_total < 1e3:
        log.error('Unreasonable planet+satellite mass: %.5e kg' % M_total)

    # Standard gravitational parameter (planet mass + satellite mass)
    mu = const_G * M_total

    # Semimajor axis is already in SI units
    sma = hf_row['semimajorax_sat']

    # Orbital period [seconds]
    hf_row['orbital_period_sat'] = 2 * np.pi * (sma * sma * sma / mu) ** 0.5


def update_hillradius(hf_row: dict):
    """
    Calculate Hill radius.

    Using equation from: http://astro.vaporia.com/start/hillradius.html

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    """

    sma = hf_row['semimajorax']
    ecc = hf_row['eccentricity']
    Mpl = hf_row['M_int']
    Mst = hf_row['M_star']

    hf_row['hill_radius'] = sma * (1 - ecc) * (Mpl / (3 * Mst)) ** (1.0 / 3)


def update_rochelimit(hf_row: dict):
    """
    Calculate Roche limit.

    Using equation from: http://astro.vaporia.com/start/rochelimit.html

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    """

    Rpl = hf_row['R_int']
    Mpl = hf_row['M_int']
    Mst = hf_row['M_star']

    hf_row['roche_limit'] = Rpl * (2 * Mst / Mpl) ** (1.0 / 3)


def update_breakup_period(hf_row: dict):
    """
    Calculate Breakup period.

    Using equation from: https://arxiv.org/abs/2508.09273
    (Note, the equation contains a typo, it should
    read: 2pi/T = Ω = sqrt( G Mp / Rp^3 ). )

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
    """

    Rpl = hf_row['R_int']
    Mpl = hf_row['M_int']

    hf_row['breakup_period'] = 2 * np.pi / np.sqrt(const_G * Mpl / (Rpl**3))


def run_orbit(hf_row: dict, config: Config, dirs: dict, tides_o: Tides_t, interior_o: Interior_t):
    """Update parameters relating to orbital evolution and tides.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration.
        dirs: dict
            Dictionary of directories.
        tides_o: Tides_t
            Tides data containing Imk2 spectra at current time.
        interior_o: Interior_t
            Struct containing interior arrays at current time.
    """

    log.info('Evolve orbit and tides...')

    # Time step
    current_time = float(hf_row['Time'])

    # Use config parameters as initial guess
    if current_time <= 1:
        hf_row['M_sat'] = config.orbit.satellite.mass_sat * M_earth  # [kg]
        hf_row['R_sat'] = config.orbit.satellite.radius_sat * R_earth  # [m]
        hf_row['C_sat'] = config.orbit.satellite.c_factor_sat * hf_row['M_sat'] * hf_row['R_sat']**2

        # Set independent orbital parameters from config.
        hf_row['semimajorax'] = config.orbit.semimajoraxis * AU
        hf_row['eccentricity'] = config.orbit.eccentricity

        # set semi-major axis to obtain a particular bolometric instellation flux
        if config.orbit.instellation_method == 'inst' and config.star.module == 'dummy':
            from proteus.star.dummy import calc_star_luminosity, get_star_radius

            Lbol = calc_star_luminosity(config.star.dummy.Teff, get_star_radius(config) * R_sun)
            S_earth = L_sun / (4 * np.pi * AU * AU)
            S_0 = config.orbit.instellationflux * S_earth

            hf_row['semimajorax'] = np.sqrt(Lbol / (4 * np.pi * S_0))

        hf_row['semimajorax_sat'] = config.orbit.satellite.semimajoraxis_sat * AU
        hf_row['eccentricity_sat'] = config.orbit.satellite.eccentricity_sat

        hf_row['aps_prec_angle'] = config.orbit.satellite.aps_prec_angle

        # Update orbital period (dependent)
        update_period(hf_row)

        # Axial period [seconds]
        if config.orbit.axial_period is None:
            # set by user to 'none', use 1:1 SOR
            hf_row['axial_period'] = hf_row['orbital_period']
        else:
            # set by user with float, use that
            hf_row['axial_period'] = float(config.orbit.axial_period) * secs_per_hour

        # Update satellite orbital period (dependent)
        update_period_sat(hf_row)

        # Axial period [seconds]
        if config.orbit.satellite.axial_period_sat is None:
            # set by user to 'none', use 1:1 SOR
            hf_row['axial_period_sat'] = hf_row['orbital_period_sat']
        else:
            # set by user with float, use that
            hf_row['axial_period_sat'] = float(config.orbit.satellite.axial_period_sat) * secs_per_hour

    else:
        # Set independent orbital parameters, through the desired method... (Star-Planet)
        if config.orbit.star_planet_model is not None:
            # set by orbital evolution, based on tidal love number
            from proteus.orbit.orbit import evolve_orbit_star

            evolve_orbit_star(hf_row, config, tides_o, interior_o.dt)

        else:
            # set semi-major axis to obtain a particular bolometric instellation flux
            if config.orbit.instellation_method == 'inst' and config.star.module == 'dummy':
                from proteus.star.dummy import calc_star_luminosity, get_star_radius

                Lbol = calc_star_luminosity(config.star.dummy.Teff, get_star_radius(config) * R_sun)
                S_earth = L_sun / (4 * np.pi * AU * AU)
                S_0 = config.orbit.instellationflux * S_earth

                hf_row['semimajorax'] = np.sqrt(Lbol / (4 * np.pi * S_0))

        # Set independent orbital parameters, through the desired method... (Planet-Satellite)
        if config.orbit.planet_satellite_model is not None:
            # set by orbital evolution, based on tidal love number
            from proteus.orbit.satellite import evolve_orbit_satellite

            evolve_orbit_satellite(hf_row, config, tides_o, interior_o)

        # Update orbital period, from independent variables above
        update_period(hf_row)

        # Update satellite orbital period, from independent variables above
        update_period_sat(hf_row)

    # Inform user
    log.info('    Orb SMaxis = %.5f AU    (Planet)' % (hf_row['semimajorax'] / AU))
    log.info('    Orb eccent = %.5f       (Planet)' % (hf_row['eccentricity']))
    log.info('    Orb period = %.5f days  (Planet)' % (hf_row['orbital_period'] / secs_per_day))
    log.info('    Orb spin   = %.5f days  (Planet)' % (hf_row['axial_period'] / secs_per_day))

    if config.orbit.satellite:
        log.info('    Orb SMaxis = %.5f AU    (Satellite)' % (hf_row['semimajorax_sat'] / AU))
        log.info('    Orb eccent = %.5f       (Satellite)' % (hf_row['eccentricity_sat']))
        log.info('    Orb period = %.5f days  (Satellite)' % (hf_row['orbital_period_sat'] / secs_per_day))
        log.info('    Orb spin   = %.5f days  (Satellite)' % (hf_row['axial_period_sat'] / secs_per_day))

    # Update dependent orbital parameters, from independent variables above
    # Update separation
    update_separation(hf_row)

    # Update Breakup period
    update_breakup_period(hf_row)
    if hf_row['axial_period'] <= hf_row['breakup_period'] + float(
        config.params.stop.disint.offset_spin
    ):
        log.warning('Planet is spinning faster than the Breakup rate')

    # Update Roche limit
    update_rochelimit(hf_row)
    if hf_row['separation'] <= hf_row['roche_limit'] + float(
        config.params.stop.disint.offset_roche
    ):
        log.warning('Planet is orbiting within the Roche limit of its star')
    elif hf_row['perihelion'] <= hf_row['roche_limit'] + float(
        config.params.stop.disint.offset_roche
    ):
        log.warning('Planet is (partially) orbiting within the Roche limit of its star')

    # Update Hill radius
    update_hillradius(hf_row)
    if max(hf_row['R_obs'], hf_row['R_xuv']) > hf_row['hill_radius']:
        log.warning('Atmosphere extends beyond the Hill radius')

    # Call tidal heating module, if enabled
    # Initialise, set tidal heating to zero
    interior_o.tides = np.zeros(len(interior_o.phi))

    # Call tides module, calculates heating rates and new love number
    if config.orbit.module == 'dummy':
        from proteus.orbit.dummy import run_dummy_tides

        hf_row['Imk2'] = run_dummy_tides(config, interior_o)

    elif config.orbit.module == 'lovepy':
        from proteus.orbit.lovepy import run_lovepy

        hf_row['Imk2'] = run_lovepy(hf_row, dirs, interior_o, tides_o, config)

    elif config.orbit.module == 'obliqua':
        from proteus.orbit.obliqua import run_obliqua

        Imk = run_obliqua(hf_row, dirs, interior_o, tides_o, config)

        if config.orbit.obliqua.n == [2]:
            hf_row['Imk2'] = Imk
        else:
            hf_row["Imk2"] = 0.0
        # Since Obliqua returns the frequency dependent Love number for arbitrary
        # degree (n), we set Imk2 to either the mean value (if n=2) or 0.0 to
        # avoid confusion with other degrees (Imk3, Imk4, etc.). Note, the user
        # can access the Love number(s) from the tides_o object along with the
        # corresponding forcing frequencies and modes.

    else:
        hf_row['Imk2'] = 0.0

    # Print info
    if config.orbit.module == 'obliqua':
        log.info('    Pla H_tide = %.1e W kg-1 (mean) ' % np.mean(interior_o.tides))
        log.info('    Pla Im(k)  = %.1e ' % Imk)

    elif config.orbit.module is not None:
        log.info('    Pla H_tide = %.1e W kg-1 (mean) ' % np.mean(interior_o.tides))
        log.info('    Pla Im(k2) = %.1e ' % hf_row['Imk2'])


    # If satellite orbital evolution is enabled, then extract the satellite love number from
    # the provided lookup file
    if config.orbit.planet_satellite_model == "ps1d_evec":
        from proteus.orbit.obliqua import LN_from_lookup

        log.info('    Extracting Love number from satellite lookup table')

        LN_from_lookup(hf_row, tides_o, config)


def read_tides_data(output_dir: str, model: str, times: list):
    if len(times) == 0:
        return []

    if model == 'obliqua':
        from proteus.orbit.obliqua import read_ncdfs

        return read_ncdfs(output_dir, times)

    else:
        return []
