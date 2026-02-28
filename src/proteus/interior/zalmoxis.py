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

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

# Set up logging
logger = logging.getLogger('fwl.' + __name__)


def get_zalmoxis_output_filepath(outdir: str):
    """Returns the output file path for Zalmoxis data.
    Args:
        outdir (str): Output directory.
    Returns:
        str: Path to the output file.
    """
    return os.path.join(outdir, 'data', 'zalmoxis_output.dat')


def load_zalmoxis_configuration(config: Config, hf_row: dict):
    """Loads the model configuration for Zalmoxis and calculates the dry mass of the planet based on the total mass and the mass of volatiles.
    Args:
        config (Config): The configuration object containing the Zalmoxis parameters.
        hf_row (dict): A dictionary containing the mass of volatiles and other parameters.
    Returns:
        dict: A dictionary containing the Zalmoxis configuration parameters.
    """

    # Setup target planet mass (input parameter) as the total mass of the planet (dry mass + volatiles) [kg]
    total_planet_mass = config.struct.mass_tot * M_earth

    logger.info(
        'Total target planet mass (dry mass + volatiles): %s kg '
        'with EOS: core=%s, mantle=%s, ice=%s',
        total_planet_mass,
        config.struct.zalmoxis.core_eos,
        config.struct.zalmoxis.mantle_eos,
        config.struct.zalmoxis.ice_layer_eos or '(none)',
    )

    # Calculate the total mass of 'wet' elements in the planet
    M_volatiles = 0.0
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        M_volatiles += hf_row[e + '_kg_total']

    logger.info(f'Volatile mass: {M_volatiles} kg')

    # Calculate the target planet mass (dry mass) by subtracting the mass of volatiles from the total planet mass
    planet_mass = total_planet_mass - M_volatiles

    logger.info(f'Target planet mass (dry mass): {planet_mass} kg ')

    # Build per-layer EOS config dict from PROTEUS config fields
    layer_eos_config = {
        'core': config.struct.zalmoxis.core_eos,
        'mantle': config.struct.zalmoxis.mantle_eos,
    }
    if config.struct.zalmoxis.ice_layer_eos:
        layer_eos_config['ice_layer'] = config.struct.zalmoxis.ice_layer_eos

    return {
        'planet_mass': planet_mass,
        'core_mass_fraction': config.struct.zalmoxis.coremassfrac,
        'mantle_mass_fraction': config.struct.zalmoxis.mantle_mass_fraction,
        'temperature_mode': config.struct.zalmoxis.temperature_mode,
        'surface_temperature': config.struct.zalmoxis.surface_temperature,
        'center_temperature': config.struct.zalmoxis.center_temperature,
        'temp_profile_file': config.struct.zalmoxis.temperature_profile_file,
        'layer_eos_config': layer_eos_config,
        'num_layers': config.struct.zalmoxis.num_levels,
        'max_iterations_outer': config.struct.zalmoxis.max_iterations_outer,
        'tolerance_outer': config.struct.zalmoxis.tolerance_outer,
        'max_iterations_inner': config.struct.zalmoxis.max_iterations_inner,
        'tolerance_inner': config.struct.zalmoxis.tolerance_inner,
        'relative_tolerance': config.struct.zalmoxis.relative_tolerance,
        'absolute_tolerance': config.struct.zalmoxis.absolute_tolerance,
        'maximum_step': config.struct.zalmoxis.maximum_step,
        'adaptive_radial_fraction': config.struct.zalmoxis.adaptive_radial_fraction,
        'max_center_pressure_guess': config.struct.zalmoxis.max_center_pressure_guess,
        'target_surface_pressure': config.struct.zalmoxis.target_surface_pressure,
        'pressure_tolerance': config.struct.zalmoxis.pressure_tolerance,
        'max_iterations_pressure': config.struct.zalmoxis.max_iterations_pressure,
        'verbose': config.struct.zalmoxis.verbose,
        'iteration_profiles_enabled': config.struct.zalmoxis.iteration_profiles_enabled,
    }


def load_zalmoxis_material_dictionaries():
    """Load Zalmoxis material property dictionaries.

    EOS file paths are derived from the Zalmoxis source names
    (e.g. ``Seager2007`` → ``EOS/static/Seager2007/``,
    ``WolfBower2018_MgSiO3`` → ``EOS/dynamic/WolfBower2018_MgSiO3/P-T/``),
    not from ``interior.eos_dir``.

    Returns
    -------
    tuple
        Four dictionaries: iron/silicate, iron/T-dep silicate (WolfBower2018),
        water planets, iron/RTPress100TPa silicate.
    """
    return get_zalmoxis_EOS()


def load_zalmoxis_solidus_liquidus_functions(mantle_eos: str, config: Config):
    """Loads the solidus and liquidus functions for Zalmoxis based on the mantle EOS.

    Parameters
    ----------
    mantle_eos : str
        Mantle EOS string (e.g. "WolfBower2018_MgSiO3").
    config : Config
        PROTEUS configuration object.

    Returns
    -------
    tuple or None
        (solidus_func, liquidus_func) if T-dependent EOS, else None.
    """
    if mantle_eos == 'WolfBower2018:MgSiO3':
        return get_zalmoxis_melting_curves(config)
    return None


def scale_temperature_profile_for_aragog(
    config: Config, mantle_radii: np.ndarray, mantle_temperature_profile: np.ndarray
):
    """Scales the temperature profile obtained from Zalmoxis to match the number of levels required by Aragog.
    Args:
        config (Config): The configuration object containing the configuration parameters.
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
    scaled_temperature_profile = np.interp(
        radii_to_interpolate, mantle_radii, mantle_temperature_profile
    )

    # Create a cubic spline interpolation function
    cubic_interp_func = interp1d(mantle_radii, mantle_temperature_profile, kind='cubic')

    # Interpolate the temperature profile onto the new radial positions
    scaled_temperature_profile = cubic_interp_func(radii_to_interpolate)

    return scaled_temperature_profile


def write_spider_mesh_file(
    outdir: str,
    mantle_radii: np.ndarray,
    mantle_pressure: np.ndarray,
    mantle_density: np.ndarray,
    mantle_gravity: np.ndarray,
    num_basic: int,
) -> str:
    """Write an external mesh file for SPIDER from Zalmoxis mantle profiles.

    Interpolates the Zalmoxis mantle arrays onto uniformly-spaced SPIDER
    basic and staggered nodes, then writes the mesh file in the format
    expected by SPIDER's ``SetMeshFromExternalFile()``.

    Parameters
    ----------
    outdir : str
        PROTEUS output directory (file is written to ``outdir/data/``).
    mantle_radii : np.ndarray
        Radial positions from CMB to surface, ascending [m].
    mantle_pressure : np.ndarray
        Pressure at each radius [Pa].
    mantle_density : np.ndarray
        Density at each radius [kg/m^3].
    mantle_gravity : np.ndarray
        Gravity magnitude at each radius [m/s^2] (positive).
    num_basic : int
        Number of SPIDER basic nodes (shell boundaries).

    Returns
    -------
    str
        Path to the written mesh file.
    """
    num_staggered = num_basic - 1
    R_surf = float(mantle_radii[-1])
    R_cmb = float(mantle_radii[0])

    # Basic nodes: uniform spacing from surface to CMB (descending r)
    r_b = np.linspace(R_surf, R_cmb, num_basic)
    # Staggered nodes: midpoints between consecutive basic nodes
    r_s = 0.5 * (r_b[:-1] + r_b[1:])

    # Interpolate Zalmoxis profiles onto node positions
    # mantle_radii is ascending, np.interp requires ascending xp
    P_b = np.interp(r_b, mantle_radii, mantle_pressure)
    rho_b = np.interp(r_b, mantle_radii, mantle_density)
    g_b = np.interp(r_b, mantle_radii, mantle_gravity)

    P_s = np.interp(r_s, mantle_radii, mantle_pressure)
    rho_s = np.interp(r_s, mantle_radii, mantle_density)
    g_s = np.interp(r_s, mantle_radii, mantle_gravity)

    # Negate gravity for SPIDER convention (inward-pointing, negative)
    g_b = -np.abs(g_b)
    g_s = -np.abs(g_s)

    # Write mesh file
    mesh_path = os.path.join(outdir, 'data', 'spider_mesh.dat')
    with open(mesh_path, 'w') as f:
        f.write(f'# {num_basic} {num_staggered}\n')
        for i in range(num_basic):
            f.write(f'{r_b[i]:.15e} {P_b[i]:.15e} {rho_b[i]:.15e} {g_b[i]:.15e}\n')
        for i in range(num_staggered):
            f.write(f'{r_s[i]:.15e} {P_s[i]:.15e} {rho_s[i]:.15e} {g_s[i]:.15e}\n')

    logger.info(
        'Wrote SPIDER mesh file: %s (%d basic + %d staggered nodes)',
        mesh_path,
        num_basic,
        num_staggered,
    )

    # Warn if CMB pressure is high enough that SPIDER's solid-phase EOS
    # tables may be evaluated outside their valid entropy range
    P_cmb = float(P_b[-1])  # last basic node = CMB
    if P_cmb > 400e9:
        logger.warning(
            'CMB pressure from Zalmoxis mesh is %.1f GPa (> 400 GPa). '
            'SPIDER solid-phase EOS tables (WolfBower2018) have limited entropy '
            'coverage (~2400 J/kg/K). If the initial adiabat crosses the liquidus '
            'at these pressures, the solid-phase lookup will be clamped to the table '
            'edge, which can produce unphysical material properties (negative thermal '
            'expansion) and cause CVode convergence failure.',
            P_cmb / 1e9,
        )

    return mesh_path


def zalmoxis_solver(config: Config, outdir: str, hf_row: dict, num_spider_nodes: int = 0):
    """Run the Zalmoxis solver to compute the interior structure of a planet.

    Parameters
    ----------
    config : Config
        Configuration object.
    outdir : str
        Output directory where results will be saved.
    hf_row : dict
        Dictionary containing volatile masses and other parameters.
    num_spider_nodes : int
        Number of SPIDER basic nodes. If > 0, writes a SPIDER mesh file
        and returns its path as the second element of the return tuple.

    Returns
    -------
    cmb_radius : float
        Core-mantle boundary radius [m].
    spider_mesh_file : str or None
        Path to the SPIDER mesh file, or None if ``num_spider_nodes == 0``.
    """

    # Load the Zalmoxis configuration parameters
    config_params = load_zalmoxis_configuration(config, hf_row)

    # Get the output location for Zalmoxis output and create the file if it does not exist
    output_zalmoxis = get_zalmoxis_output_filepath(outdir)
    open(output_zalmoxis, 'a').close()

    # Run the Zalmoxis main function to compute the interior structure
    model_results = main(
        config_params,
        material_dictionaries=load_zalmoxis_material_dictionaries(),
        melting_curves_functions=load_zalmoxis_solidus_liquidus_functions(
            config.struct.zalmoxis.mantle_eos, config
        ),
        input_dir=os.path.join(outdir, 'data'),
    )

    # Extract results from the model
    radii = model_results['radii']
    density = model_results['density']
    gravity = model_results['gravity']
    pressure = model_results['pressure']
    temperature = model_results['temperature']
    mass_enclosed = model_results['mass_enclosed']
    cmb_mass = model_results['cmb_mass']
    core_mantle_mass = model_results['core_mantle_mass']
    converged = model_results['converged']
    converged_pressure = model_results['converged_pressure']
    converged_density = model_results['converged_density']
    converged_mass = model_results['converged_mass']

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Extract the planet radius and core-mantle boundary radius
    planet_radius = radii[-1]
    cmb_radius = radii[cmb_index]

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = mass_enclosed[-1] / (4 / 3 * np.pi * radii[-1] ** 3)

    # Final results of the Zalmoxis interior model
    logger.info('Found solution for interior structure with Zalmoxis')
    logger.info(
        f'Interior (dry calculated mass) mass: {mass_enclosed[-1]} kg or approximately {mass_enclosed[-1] / M_earth:.2f} M_earth'
    )
    logger.info(
        f'Interior radius: {planet_radius:.2e} m or {planet_radius / R_earth:.2f} R_earth'
    )
    logger.info(f'Core radius: {cmb_radius:.2e} or {cmb_radius / R_earth:.2f} R_earth')
    logger.info(f'Core-mantle boundary mass: {mass_enclosed[cmb_index]:.2e} kg')
    logger.info(f'Mantle density at the core-mantle boundary: {density[cmb_index]:.2e} kg/m^3')
    logger.info(
        f'Core density at the core-mantle boundary: {density[cmb_index - 1]:.2e} kg/m^3'
    )
    logger.info(f'Pressure at the core-mantle boundary: {pressure[cmb_index]:.2e} Pa')
    logger.info(f'Pressure at the center: {pressure[0]:.2e} Pa')
    logger.info(f'Average density: {average_density:.2e} kg/m^3')
    logger.info(
        f'Core-mantle boundary mass fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}'
    )
    logger.info(f'Core radius fraction: {cmb_radius / planet_radius:.4f}')
    logger.info(
        f'Inner mantle radius fraction: {radii[np.argmax(mass_enclosed >= core_mantle_mass)] / planet_radius:.4f}'
    )
    logger.info(
        f'Overall Convergence Status: {converged} with Pressure: {converged_pressure}, Density: {converged_density}, Mass: {converged_mass}'
    )

    # Update the surface radius, interior radius, and mass in the hf_row
    hf_row['R_int'] = planet_radius
    hf_row['M_int'] = mass_enclosed[-1]
    hf_row['M_core'] = mass_enclosed[cmb_index]
    hf_row['gravity'] = gravity[-1]

    logger.info(f'Saving Zalmoxis output to {output_zalmoxis}')

    # Select mantle arrays (to match the mesh needed for Aragog)
    mantle_radii = radii[cmb_index:]
    mantle_pressure = pressure[cmb_index:]
    mantle_density = density[cmb_index:]
    mantle_gravity = gravity[cmb_index:]
    mantle_temperature = temperature[cmb_index:]

    # Scale mantle temperature to match Aragog temperature profile format
    mantle_temperature_scaled = scale_temperature_profile_for_aragog(
        config, mantle_radii, mantle_temperature
    )

    # Write temperature profile to a separate file for Aragog to read
    np.savetxt(
        os.path.join(outdir, 'data', 'zalmoxis_output_temp.txt'), mantle_temperature_scaled
    )

    # Save final grids to the output file for the mantle for Aragog
    with open(output_zalmoxis, 'w') as f:
        for i in range(len(mantle_radii)):
            f.write(
                f'{mantle_radii[i]:.17e} {mantle_pressure[i]:.17e} {mantle_density[i]:.17e} {mantle_gravity[i]:.17e} {mantle_temperature[i]:.17e}\n'
            )

    # Write SPIDER mesh file if requested
    spider_mesh_file = None
    if num_spider_nodes > 0:
        spider_mesh_file = write_spider_mesh_file(
            outdir,
            mantle_radii,
            mantle_pressure,
            mantle_density,
            mantle_gravity,
            num_spider_nodes,
        )

    return cmb_radius, spider_mesh_file
