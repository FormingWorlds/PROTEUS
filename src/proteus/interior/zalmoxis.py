# Zalmoxis interior module
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import platformdirs
from scipy.interpolate import interp1d

from proteus.config import Config
from proteus.utils.constants import (
    M_earth,
    R_earth,
    element_list,
)
from proteus.utils.data import get_zalmoxis_EOS, get_zalmoxis_melting_curves
from zalmoxis.zalmoxis import main

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA',
                                   platformdirs.user_data_dir('fwl_data')))

# Set up logging
logger = logging.getLogger("fwl."+__name__)

def get_zalmoxis_output_filepath(outdir: str):
    """Returns the output file path for Zalmoxis data.
    Args:
        outdir (str): Output directory.
    Returns:
        str: Path to the output file.
    """
    return os.path.join(outdir, "data", "zalmoxis_output.dat")

def load_zalmoxis_configuration(config:Config, hf_row:dict):
    """Loads the model configuration for Zalmoxis and calculates the dry mass of the planet based on the total mass and the mass of volatiles.
    Args:
        config (Config): The configuration object containing the Zalmoxis parameters.
        hf_row (dict): A dictionary containing the mass of volatiles and other parameters.
    Returns:
        dict: A dictionary containing the Zalmoxis configuration parameters.
    """

    # Setup target planet mass (input parameter) as the total mass of the planet (dry mass + volatiles) [kg]
    total_planet_mass = config.struct.mass_tot * M_earth

    logger.info(f"Total target planet mass (dry mass + volatiles): {total_planet_mass} kg with EOS choice: {config.struct.zalmoxis.EOSchoice}")

    # Calculate the total mass of volatiles in the planet
    M_volatiles = 0.0
    for e in element_list:
        if (e == 'O'):
            # do not include oxygen, because it varies over time in order to set fO2.
            continue
        M_volatiles += hf_row[e+"_kg_total"]

    logger.info(f"Volatile mass: {M_volatiles} kg")

    # Calculate the target planet mass (dry mass) by subtracting the mass of volatiles from the total planet mass
    planet_mass = total_planet_mass - M_volatiles

    logger.info(f"Target planet mass (dry mass): {planet_mass} kg ")

    return {
        "planet_mass": planet_mass,
        "core_mass_fraction": config.struct.zalmoxis.coremassfrac,
        "mantle_mass_fraction": config.struct.zalmoxis.mantle_mass_fraction,
        "weight_iron_fraction": config.struct.zalmoxis.weight_iron_frac,
        "temperature_mode": config.struct.zalmoxis.temperature_mode,
        "surface_temperature": config.struct.zalmoxis.surface_temperature,
        "center_temperature": config.struct.zalmoxis.center_temperature,
        "temp_profile_file": config.struct.zalmoxis.temperature_profile_file,
        "EOS_CHOICE": config.struct.zalmoxis.EOSchoice,
        "num_layers": config.struct.zalmoxis.num_levels,
        "max_iterations_outer": config.struct.zalmoxis.max_iterations_outer,
        "tolerance_outer": config.struct.zalmoxis.tolerance_outer,
        "max_iterations_inner": config.struct.zalmoxis.max_iterations_inner,
        "tolerance_inner": config.struct.zalmoxis.tolerance_inner,
        "relative_tolerance": config.struct.zalmoxis.relative_tolerance,
        "absolute_tolerance": config.struct.zalmoxis.absolute_tolerance,
        "maximum_step": config.struct.zalmoxis.maximum_step,
        "adaptive_radial_fraction": config.struct.zalmoxis.adaptive_radial_fraction,
        "max_center_pressure_guess": config.struct.zalmoxis.max_center_pressure_guess,
        "target_surface_pressure": config.struct.zalmoxis.target_surface_pressure,
        "pressure_tolerance": config.struct.zalmoxis.pressure_tolerance,
        "max_iterations_pressure": config.struct.zalmoxis.max_iterations_pressure,
        "pressure_adjustment_factor": config.struct.zalmoxis.pressure_adjustment_factor,
        "verbose": config.struct.zalmoxis.verbose,
        "iteration_profiles_enabled": config.struct.zalmoxis.iteration_profiles_enabled
    }

def load_zalmoxis_material_dictionaries():
    """
    Loads the material dictionaries for Zalmoxis.
    Returns:
        tuple: A tuple containing three dictionaries for iron/silicate planets, iron/Tdep_silicate planets, and water planets.
    """
    return get_zalmoxis_EOS()

def load_zalmoxis_solidus_liquidus_functions(EOS_CHOICE):
    """Loads the solidus and liquidus functions for Zalmoxis based on the EOS choice.
    Args:
        EOS_CHOICE (str): The EOS choice for Zalmoxis.
    Returns:
        tuple: A tuple containing the solidus and liquidus functions.
    """
    if EOS_CHOICE == "Tabulated:iron/Tdep_silicate":
        return get_zalmoxis_melting_curves()

def scale_temperature_profile_for_aragog(config:Config, mantle_radii: np.ndarray, mantle_temperature_profile: np.ndarray):
    """Scales the temperature profile obtained from Zalmoxis to match the number of levels required by Aragog.
    Args:
        config (Config): The configuration object containing the Aragog parameters.
        mantle_radii (np.ndarray): The radial positions of the mantle layers from Zalmoxis.
        mantle_temperature_profile (np.ndarray): The temperature profile of the mantle layers from Zalmoxis.
    Returns:
        np.ndarray: The scaled temperature profile matching the number of levels in Aragog.
    """

    # Number of levels in Aragog mesh
    mesh_grid_size = config.interior.aragog.num_levels - 1

    # Create new evenly spaced radial positions for Aragog
    radii_to_interpolate = np.linspace(mantle_radii[0], mantle_radii[-1], mesh_grid_size)

    # Interpolate the temperature profile onto the new radial positions
    scaled_temperature_profile = np.interp(radii_to_interpolate, mantle_radii, mantle_temperature_profile)

    # Create a cubic spline interpolation function
    cubic_interp_func = interp1d(mantle_radii, mantle_temperature_profile, kind='cubic')

    # Interpolate the temperature profile onto the new radial positions
    scaled_temperature_profile = cubic_interp_func(radii_to_interpolate)

    return scaled_temperature_profile

def zalmoxis_solver(config:Config, outdir: str, hf_row:dict):
    """Runs the Zalmoxis solver to compute the interior structure of a planet.
    Args:
        config (Config): The configuration object containing the Zalmoxis parameters.
        outdir (str): The output directory where results will be saved.
        hf_row (dict): A dictionary containing the mass of volatiles and other parameters.
    Returns:
        float: The core-mantle boundary radius.
    """

    # Load the Zalmoxis configuration parameters
    config_params = load_zalmoxis_configuration(config, hf_row)

    # Get the output location for Zalmoxis output and create the file if it does not exist
    output_zalmoxis = get_zalmoxis_output_filepath(outdir)
    open(output_zalmoxis, 'a').close()

    # Run the Zalmoxis main function to compute the interior structure
    model_results = main(config_params, material_dictionaries=load_zalmoxis_material_dictionaries(), melting_curves_functions=load_zalmoxis_solidus_liquidus_functions(config_params["EOS_CHOICE"]), input_dir=os.path.join(outdir, "data"))

    # Extract results from the model
    radii = model_results['radii']
    density = model_results['density']
    gravity = model_results['gravity']
    pressure = model_results['pressure']
    temperature = model_results['temperature']
    mass_enclosed = model_results["mass_enclosed"]
    cmb_mass = model_results["cmb_mass"]
    core_mantle_mass = model_results["core_mantle_mass"]
    converged = model_results["converged"]
    converged_pressure = model_results["converged_pressure"]
    converged_density = model_results["converged_density"]
    converged_mass = model_results["converged_mass"]

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Extract the planet radius and core-mantle boundary radius
    planet_radius = radii[-1]
    cmb_radius = radii[cmb_index]

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = mass_enclosed[-1] / (4/3 * np.pi * radii[-1]**3)

    # Final results of the Zalmoxis interior model
    logger.info("Found solution for interior structure with Zalmoxis")
    logger.info(f"Interior (dry calculated mass) mass: {mass_enclosed[-1]} kg or approximately {mass_enclosed[-1] / M_earth:.2f} M_earth")
    logger.info(f"Interior radius: {planet_radius:.2e} m or {planet_radius / R_earth:.2f} R_earth")
    logger.info(f"Core radius: {cmb_radius:.2e} or {cmb_radius / R_earth:.2f} R_earth")
    logger.info(f"Core-mantle boundary mass: {mass_enclosed[cmb_index]:.2e} kg")
    logger.info(f"Mantle density at the core-mantle boundary: {density[cmb_index]:.2e} kg/m^3")
    logger.info(f"Core density at the core-mantle boundary: {density[cmb_index - 1]:.2e} kg/m^3")
    logger.info(f"Pressure at the core-mantle boundary: {pressure[cmb_index]:.2e} Pa")
    logger.info(f"Pressure at the center: {pressure[0]:.2e} Pa")
    logger.info(f"Average density: {average_density:.2e} kg/m^3")
    logger.info(f"Core-mantle boundary mass fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}")
    logger.info(f"Core radius fraction: {cmb_radius / planet_radius:.4f}")
    logger.info(f"Inner mantle radius fraction: {radii[np.argmax(mass_enclosed >= core_mantle_mass)] / planet_radius:.4f}")
    logger.info(f"Overall Convergence Status: {converged} with Pressure: {converged_pressure}, Density: {converged_density}, Mass: {converged_mass}")

    # Update the surface radius, interior radius, and mass in the hf_row
    hf_row["R_int"] = planet_radius
    hf_row["M_int"] = mass_enclosed[-1]
    hf_row["M_core"] = mass_enclosed[cmb_index]
    hf_row["gravity"] = gravity[-1]

    logger.info(f"Saving Zalmoxis output to {output_zalmoxis}")

    # Select mantle arrays (to match the mesh needed for Aragog)
    mantle_radii = radii[cmb_index:]
    mantle_pressure = pressure[cmb_index:]
    mantle_density = density[cmb_index:]
    mantle_gravity = gravity[cmb_index:]
    mantle_temperature = temperature[cmb_index:]

    # Scale mantle temperature to match Aragog temperature profile format
    mantle_temperature_scaled = scale_temperature_profile_for_aragog(config, mantle_radii, mantle_temperature)

    # Write temperature profile to a separate file for Aragog to read
    np.savetxt(os.path.join(outdir, "data", "zalmoxis_output_temp.txt"), mantle_temperature_scaled)

    # Save final grids to the output file for the mantle for Aragog
    with open(output_zalmoxis, 'w') as f:
        for i in range(len(mantle_radii)):
            f.write(f"{mantle_radii[i]:.17e} {mantle_pressure[i]:.17e} {mantle_density[i]:.17e} {mantle_gravity[i]:.17e} {mantle_temperature[i]:.17e}\n")

    return cmb_radius
