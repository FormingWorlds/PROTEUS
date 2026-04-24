from __future__ import annotations

import builtins
import importlib
import io
import logging
import os
from contextlib import redirect_stdout

log = logging.getLogger('fwl.' + __name__)

def _exoatlas_dir(fwl_dir: str) -> str:
    """Get the path to the exoatlas data directory."""
    return os.path.join(fwl_dir, 'planet_reference', 'exoatlas')

def get_exoatlas_data(fwl_dir: str) -> tuple:
    """Import exoatlas while suppressing third-party print() spam.

    Data are accessed from the solarsys and exoplanets classes. These have methods for
    getting planet and stellar parameters, and their uncertainties.

    For example, to get the mass and uncertainty of all planets:
    - `masses = exoplanets.mass()`
    - `mass_uncertainties = exoplanets.mass_uncertainty()`

    For example, to get the stellar age of the host star of TRAPPIST-1b:
    - `age = exoplanets['TRAPPIST-1b'].stellar_age()`

    Add .value to get the raw values without units, e.g. `age.value`.


    Available parameters include: airmass, altaz, angular_separation, argument_of_periastron, dec, density, depth_snr, depth_uncertainty, detected_in_astrometry, detected_in_disk_kinematics, detected_in_eclipse_timing_variations, detected_in_imaging, detected_in_microlensing, detected_in_orbital_brightness_modulations, detected_in_pulsar, detected_in_pulsation_timing, detected_in_rv, detected_in_transit, discovery_facility, discovery_method, discovery_publication, discovery_year, distance, distance_modulus, eccentricity, emission_signal, emission_snr, escape_parameter, escape_velocity, hostname, imaging_contrast, impact_velocity, inclination, insolation, is_controversial, kludge_mass, kludge_radius, kludge_stellar_age, letter, log_relative_insolation, magnitude_B, magnitude_H, magnitude_IC, magnitude_J, magnitude_K, magnitude_T, magnitude_V, magnitude_W1, magnitude_W2, magnitude_W3, magnitude_W4, magnitude_g, magnitude_gaia, magnitude_i, magnitude_kep, magnitude_r, magnitude_u, magnitude_z, mass, mass_estimated_from_radius_assuming_chen_and_kipping, mass_estimated_from_radius_assuming_rockyish, msini, name, number_of_emission_measurements, number_of_planets, number_of_stars, number_of_transmission_measurements, obliquity, orbital_velocity, period, planet_luminosity, pmdec, pmra, projected_obliquity, ra, radius, radius_estimated_from_mass_assuming_chen_and_kipping, reflection_signal, reflection_snr, relative_cumulative_xuv_insolation, relative_escape_velocity, relative_insolation, rv_semiamplitude, scale_height, scaled_radius, scaled_radius_from_radii, scaled_semimajoraxis, scaled_semimajoraxis_from_semimajoraxis, semimajoraxis, semimajoraxis_from_period, semimajoraxis_from_transit_scaled_semimajoraxis, shows_ttv, stellar_age, stellar_brightness, stellar_brightness_in_telescope_units, stellar_density, stellar_logg, stellar_luminosity, stellar_luminosity_from_radius_and_teff, stellar_mass, stellar_metallicity, stellar_radius, stellar_rotation_period, stellar_spectral_type, stellar_teff, stellar_vsini, surface_gravity, systemic_rv, table, targeted_fractional_uncertainty_precision, teq, tidyhostname, tidyname, transit_depth, transit_depth_from_radii, transit_duration, transit_duration_from_orbit, transit_impact_parameter, transit_impact_parameter_from_inclination, transit_midpoint, transit_scaled_radius, transit_scaled_semimajoraxis, transmission_signal, transmission_snr'

    Arguments
    ------------
    - fwl_dir: str, The path to the FWL data directory

    Returns
    ------------
    - ea: module, The exoatlas module.
    - solarsys: SolarSystem, The exoatlas SolarSystem object.
    - exoplanets: Exoplanets, The exoatlas Exoplanets object.
    """

    os.environ["EXOATLAS_DATA"] = _exoatlas_dir(fwl_dir)

    original_print = builtins.print
    stdout_buffer = io.StringIO()
    try:
        builtins.print = lambda *args, **kwargs: None
        with redirect_stdout(stdout_buffer):
            ea = importlib.import_module("exoatlas")
            solarsys_factory = getattr(ea, "SolarSystem", None)
            solarsys = solarsys_factory() if callable(solarsys_factory) else None

            exoplanets_factory = getattr(ea, "Exoplanets", None)
            exoplanets = exoplanets_factory() if callable(exoplanets_factory) else None
            if exoplanets is not None:
                exoplanets = exoplanets[exoplanets.radius() > 0] # with measured radius
                exoplanets = exoplanets[exoplanets.mass() > 0]   # with measured mass
    finally:
        builtins.print = original_print

    return ea, solarsys, exoplanets


def list_planets(fwl_dir: str, flat: bool = False) -> list[str]:
    """List the names of all planets in the databases.

    Grouped by planetary system, with solar system planets first.

    Arguments
    ------------
    - fwl_dir: str, The path to the FWL data directory.
    - flat: bool, Return a flat list of planet names. Otherwise, group by system.

    Returns
    ------------
    - names: list of str, The names of all planets in the databases.
    """

    # load data
    _, solarsys, exoplanets = get_exoatlas_data(fwl_dir)

    names = []

    # Loop through solar system planets
    sysnames = [str(pl.name()[0]) for pl in solarsys]
    log.info("Solar System planets:")
    log.info("    " + ", ".join(sysnames))

    # Store solar system planets
    if not flat:
        names.append(sysnames)
    else:
        names.extend(sysnames)

    # Loop through exoplanets
    log.info("\nExoplanets:")
    sys = "UNSET"
    sysnames = []
    for pl in exoplanets:
        name = str(pl.name()[0])

        # don't group by system if requested
        if flat:
            names.append(name)
            log.info(name)

        # group by system
        else:
            # same system
            if name[:-2] == sys:
                sysnames.append(name)
            # new system
            else:
                if sysnames:
                    names.extend(sysnames)
                    log.info("    " + ", ".join(sysnames))
                sysnames = [name]
                sys = name[:-2]

    return names


def get_sys(fwl_dir: str, name: str, quiet: bool = False) -> dict:
    """Get the observed parameters of all planets in a named system.

    Arguments
    ------------
    - fwl_dir: str, The path to the FWL data directory.
    - name: str, The name of the system (e.g. "TRAPPIST-1").
    - quiet: bool, If True, suppress printing of observation details.

    Returns
    ------------
    - obs_sys: dict, The observed parameters of all planets in the system, with uncertainties.
    """

    # load data
    _, solarsys, exoplanets = get_exoatlas_data(fwl_dir)

    # initialize system dict
    name = name.replace("_", " ")
    obs_sys = {"_name": name.replace(" ", "_")}

    # solar system
    if name.lower() in ("sun", "solar", "solarsystem"):
        for pl in solarsys:
            quiet or log.info(" ")
            obs_sys[str(pl.name()[0])] = get_obs(str(pl.name()[0]), quiet=quiet)
        return obs_sys

    # check for planets alphabetically
    for pl in "bcdefghijklmnopqrstuvwxyz":
        quiet or log.info(" ")
        plname = name + " " + pl
        obs_pl = get_obs(plname, quiet=quiet)
        if len(obs_pl) > 1:
            obs_sys[pl] = obs_pl
        else:
            break

    return obs_sys


def get_obs(fwl_dir: str, name: str, quiet: bool = False) -> dict:
    """Get the measured parameters of a planet, by name

    Arguments
    ------------
    - fwl_dir: str, The path to the FWL data directory.
    - name: str, The name of the planet.
    - quiet: bool, If True, suppress printing of observation details.

    Returns
    ------------
    - obs_pl: dict, The observed parameters of the planet, with uncertainties.
    """

    # get data
    _, solarsys, exoplanets = get_exoatlas_data(fwl_dir)

    name = name.replace("_", " ")
    obs_pl = {"_name": name.replace(" ", "_")}

    # check databases
    if name in exoplanets.name():
        pl = exoplanets[name]
        quiet or log.info(f"Measurements of exoplanet {name}")

    elif name in solarsys.name():
        pl = solarsys[name]
        quiet or log.info(f"Measurements of Solar System planet {name}")

    else:
        log.error(f"Planet '{name}' not found in databases")
        return obs_pl

    # get parameters
    for key, lk in (
        ("r_phot", "radius"),
        ("mass_tot", "mass"),
        ("Teff", "stellar_teff"),
        ("instellation", "relative_insolation"),
    ):
        val = [getattr(pl, lk)(), getattr(pl, lk + "_uncertainty")()]
        obs_pl[key] = [float(v.value[0]) for v in val]

    # print info
    for k, v in obs_pl.items():
        if k[0] == "_":
            continue
        quiet or log.info(f"    {k:16s}: {v[0]:10g} ± {v[1]:<10g}")

    return obs_pl
