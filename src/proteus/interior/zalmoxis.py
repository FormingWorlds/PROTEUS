# Zalmoxis interior module
from __future__ import annotations

import logging
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from proteus.config import Config
from proteus.utils.constants import (
    M_earth,
    R_earth,
    const_G,
    earth_center_pressure,
    element_list,
)
from proteus.utils.data import get_Seager_EOS

log = logging.getLogger("fwl."+__name__)

# Path to location at which to save the output data from Zalmoxis
def get_zalmoxis_output_filepath(outdir: str):
    """Returns the output file path for Zalmoxis data.
    Args:
        outdir (str): Output directory.
    Returns:
        str: Path to the output file.
    """
    return os.path.join(outdir, "data", "zalmoxis_output.dat")

def get_tabulated_eos(pressure, material_dictionary, material, interpolation_functions={}):
    """
    Retrieves density from tabulated EOS data for a given material and choice of EOS."""
    props = material_dictionary[material]
    try:
        eos_file = props["eos_file"]
        # Caching: Store interpolation functions for reuse
        if eos_file not in interpolation_functions:
            data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
            pressure_data = data[:, 1] * 1e9
            density_data = data[:, 0] * 1e3
            interpolation_functions[eos_file] = interp1d(pressure_data, density_data, bounds_error=False, fill_value="extrapolate")

        interpolation_function = interpolation_functions[eos_file]  # Retrieve from cache
        density = interpolation_function(pressure)  # Call to cached function

        if density is None or np.isnan(density):
            raise ValueError(f"Density calculation failed for {material} at {pressure:.2e} Pa.")

        return density

    except (ValueError, OSError) as e: # Catch file errors
        print(f"Error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
        return None
    except Exception as e: # Other errors
        print(f"Unexpected error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
        return None

def get_density_from_eos(pressure, material, eos_choice, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS.
    Args:
        pressure (float): Pressure [Pa].
        material (str): Material type (e.g., "core", "mantle").
        eos_choice (str): Equation of state choice.
        interpolation_functions (dict): Cache for interpolation functions.
    Returns:
        density (float): Calculated density [kg/m^3].
    """
    # Get material properties for Zalmoxis
    material_properties_iron_silicate_planets, material_properties_water_planets = get_Seager_EOS()

    # Select the equation of state and calculate density
    if eos_choice == "Tabulated:iron/silicate":
        return get_tabulated_eos(pressure, material_properties_iron_silicate_planets, material, interpolation_functions)
    elif eos_choice == "Tabulated:water":
        return get_tabulated_eos(pressure, material_properties_water_planets, material, interpolation_functions)
    else:
        log.error(f"Invalid EOS choice: {eos_choice}.")

def interior_structure_odes(radius, y, cmb_mass, inner_mantle_mass, eos_choice, interpolation_cache):
    """
    Calculates the derivatives of mass, gravity, and pressure with respect to radius for a planetary model.

    Parameters:
    radius (float): The current radius at which the ODEs are evaluated [m].
    y (list or array): The state vector containing mass [kg], gravity [m/s^2], and pressure [Pa] at the current radius.
    cmb_mass (float): The core-mantle boundary mass [kg].
    inner_mantle_mass (float): The mass of the inner mantle [kg].
    eos_choice (str): The equation of state choice for material properties.
    interpolation_cache (dict): A cache for interpolation to speed up calculations.

    Returns the list of derivatives of mass, gravity, and pressure with respect to radius.
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Define material based on enclosed mass within a certain mass fraction
    if eos_choice == "Tabulated:iron/silicate":
        # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
        if mass < cmb_mass:
            # Core
            material = "core"
        else:
            # Mantle
            material = "mantle"
    elif eos_choice == "Tabulated:water":
        # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
        if mass < cmb_mass:
            # Core
            material = "core"
        elif mass < inner_mantle_mass:
            # Inner mantle
            material = "bridgmanite_shell"
        else:
            # Outer layer
            material = "water_ice_layer"

    # Calculate density at the current radius, using pressure from y
    current_density = get_density_from_eos(pressure, material, eos_choice, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None:
        log.warning(f"Density calculation failed at radius {radius}.")

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * const_G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]

def zalmoxis_solver(config:Config, outdir:str, hf_row:dict):

    """
    Solves the interior structure of a planet using the Zalmoxis model.

        This function calculates the interior structure of a planet, including its
        core-mantle boundary radius, total mass, density, pressure, and gravity
        profiles. It uses an iterative solver to converge on the planet's interior
        properties based on input parameters and material properties.

        Parameters:
            config (Config): Configuration object containing structural and solver parameters.
            outdir (str): Directory path where the output files will be saved.
            hf_row (dict): Dictionary to store the calculated surface radius, mass, and gravity.

        Returns:
            float: The calculated core-mantle boundary radius of the planet.

        Key Steps:
            1. Retrieve material properties and setup initial assumptions.
            2. Initialize guesses for the planet radius and CMB radius.
            3. Iteratively solve for the planet's interior structure:
                - Outer loop: Converge on the total mass and radius.
                - Inner loop: Adjust density based on pressure and mass profiles.
                - Innermost loop: Adjust pressure to match the target surface pressure.
            4. Calculate final properties such as average density, core radius fraction,
               and CMB mass fraction.
            5. Save the mantle's radius, pressure, and density profiles to an output file.

        Notes:
            - The function uses the Noack et al. 2020 scaling law for initial radius guesses.
            - Convergence criteria are based on tolerances for mass, radius, and pressure.

        Output:
            - Saves the mantle's radius, pressure, and density profiles to a file in the
              specified output directory.
            - Updates the `hf_row` dictionary with the calculated surface radius, mass, and gravity.
    """

    # Setup target planet mass (input parameter) as the total mass of the planet (dry mass + volatiles) [kg]
    total_planet_mass = config.struct.mass_tot * M_earth

    log.info(f"Total target planet mass (dry mass + volatiles): {total_planet_mass} kg")

    # Calculate the total mass of volatiles in the planet
    M_volatiles = 0.0
    for e in element_list:
        if (e == 'O'):
            # do not include oxygen, because it varies over time in order to set fO2.
            continue
        M_volatiles += hf_row[e+"_kg_total"]

    log.info(f"Volatile mass: {M_volatiles} kg")

    # Calculate the target planet mass (dry mass) by subtracting the mass of volatiles from the total planet mass
    planet_mass = total_planet_mass - M_volatiles

    log.info(f"Target planet mass (dry mass): {planet_mass} kg ")

    # Setup assumptions
    core_mass_fraction = config.struct.zalmoxis.coremassfrac
    inner_mantle_mass_fraction = config.struct.zalmoxis.inner_mantle_mass_fraction
    weight_iron_fraction = config.struct.zalmoxis.weight_iron_frac
    eos_choice = config.struct.zalmoxis.EOSchoice

    # Setup calculation parameters for the iterative solver
    num_layers = config.struct.zalmoxis.num_levels
    max_iterations_outer = config.struct.zalmoxis.max_iterations_outer
    tolerance_outer = config.struct.zalmoxis.tolerance_outer
    max_iterations_inner = config.struct.zalmoxis.max_iterations_inner
    tolerance_inner = config.struct.zalmoxis.tolerance_inner
    relative_tolerance = config.struct.zalmoxis.relative_tolerance
    absolute_tolerance = config.struct.zalmoxis.absolute_tolerance
    target_surface_pressure = config.struct.zalmoxis.target_surface_pressure
    pressure_tolerance = config.struct.zalmoxis.pressure_tolerance
    max_iterations_pressure = config.struct.zalmoxis.max_iterations_pressure
    pressure_adjustment_factor = config.struct.zalmoxis.pressure_adjustment_factor

    # Setup initial guesses for the planet radius and core-mantle boundary mass
    radius_guess = 1000*(7030-1840*weight_iron_fraction)*(planet_mass/M_earth)**0.282 # Initial guess for the interior planet radius [m] based on the scaling law in Noack et al. 2020
    cmb_mass = 0 # Initial guess for the core-mantle boundary mass [kg]
    inner_mantle_mass = 0 # Initial guess for the inner mantle mass [kg]

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer): # Outer loop for radius and mass convergence
        # Setup initial guess for the radial grid based on the radius guess
        radii = np.linspace(0, radius_guess, num_layers)

        # Initialize arrays for mass, gravity, density, and pressure grids
        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        # Setup initial guess for the core-mantle boundary mass
        cmb_mass = core_mass_fraction * planet_mass

        # Setup initial guess for the inner mantle boundary mass
        inner_mantle_mass = (core_mass_fraction + inner_mantle_mass_fraction) * planet_mass

        # Setup initial guess for the pressure at the center of the planet (needed for solving the ODEs)
        pressure[0] = earth_center_pressure

        for inner_iter in range(max_iterations_inner): # Inner loop for density adjustment
            old_density = density.copy() # Store the old density for convergence check

            # Initialize empty cache for interpolation functions for density calculations
            interpolation_cache = {}

            # Setup initial pressure guess at the center of the planet based on empirical scaling law derived from the hydrostatic equilibrium equation
            pressure_guess = earth_center_pressure * (planet_mass/M_earth)**2 * (radius_guess/R_earth)**(-4)

            for pressure_iter in range(max_iterations_pressure): # Innermost loop for pressure adjustment

                # Setup initial conditions for the mass, gravity, and pressure at the center of the planet
                y0 = [0, 0, pressure_guess]

                # Solve the ODEs using solve_ivp
                sol = solve_ivp(lambda r, y: interior_structure_odes(r, y, cmb_mass, inner_mantle_mass, eos_choice, interpolation_cache),
                    (radii[0], radii[-1]), y0, t_eval=radii, rtol=relative_tolerance, atol=absolute_tolerance, method='RK45', dense_output=True)

                # Extract mass, gravity, and pressure grids from the solution
                mass_enclosed = sol.y[0]
                gravity = sol.y[1]
                pressure = sol.y[2]

                # Extract the calculated surface pressure from the last element of the pressure array
                surface_pressure = pressure[-1]

                # Calculate the pressure difference between the calculated surface pressure and the target surface pressure
                pressure_diff = surface_pressure - target_surface_pressure

                # Check for convergence of the surface pressure and overall pressure positivity
                if abs(pressure_diff) < pressure_tolerance and np.all(pressure > 0):
                    #log.info(f"Surface pressure converged after {pressure_iter + 1} iterations and all pressures are positive.")
                    break  # Exit the pressure adjustment loop

                # Update the pressure guess at the center of the planet based on the pressure difference at the surface using an adjustment factor
                pressure_guess -= pressure_diff * pressure_adjustment_factor

            # Update density grid based on the mass enclosed within a certain mass fraction
            for i in range(num_layers):
                if eos_choice == "Tabulated:iron/silicate":
                    # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
                    if mass_enclosed[i] < cmb_mass:
                        # Core
                        material = "core"
                    else:
                        # Mantle
                        material = "mantle"
                elif eos_choice == "Tabulated:water":
                    # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
                    if mass_enclosed[i] < cmb_mass:
                        # Core
                        material = "core"
                    elif mass_enclosed[i] < inner_mantle_mass:
                        # Inner mantle
                        material = "bridgmanite_shell"
                    else:
                        # Outer layer
                        material = "water_ice_layer"

                # Calculate the new density using the equation of state
                new_density = get_density_from_eos(pressure[i], material, eos_choice)

                # Handle potential errors in density calculation
                if new_density is None:
                    log.warning(f"Density calculation failed at radius {radii[i]}. Using previous density.")
                    new_density = old_density[i]

                # Update the density grid with a weighted average of the new and old density
                density[i] = 0.5 * (new_density + old_density[i])

            # Check for convergence of density using relative difference between old and new density
            relative_diff_inner = np.max(np.abs((density - old_density) / (old_density + 1e-20)))
            if relative_diff_inner < tolerance_inner:
                #log.info(f"Inner loop converged after {inner_iter + 1} iterations.")
                break # Exit the inner loop

        # Extract the calculated total interior mass of the planet from the last element of the mass array
        calculated_mass = mass_enclosed[-1]

        # Update the total interior radius by scaling the initial guess based on the calculated mass
        radius_guess = radius_guess * (planet_mass / calculated_mass)**(1/3)

        # Update the core-mantle boundary radius based on the calculated mass grid
        cmb_radius = radii[np.argmax(mass_enclosed >= cmb_mass)]

        # Update the core-mantle boundary mass based on the core mass fraction and calculated total interior mass of the planet
        cmb_mass = core_mass_fraction * calculated_mass

        # Update the inner mantle mass based on the inner mantle mass fraction and calculated total interior mass of the planet
        inner_mantle_mass = (core_mass_fraction + inner_mantle_mass_fraction) * calculated_mass

        # Calculate relative differences of the calculated total interior mass
        relative_diff_outer_mass = abs((calculated_mass - planet_mass) / planet_mass)

        # Check for convergence of the calculated total interior mass and core-mantle boundary radius of the planet
        if relative_diff_outer_mass < tolerance_outer:
            log.info(f"Outer loop (total mass) converged after {outer_iter + 1} iterations.")
            break  # Exit the outer loop

        #log.info(f"Outer iteration {outer_iter+1} took {end_time - start_time:.2f} seconds")

        # Check if maximum iterations for outer loop are reached
        if outer_iter == max_iterations_outer - 1:
            log.warning(f"Maximum outer iterations ({max_iterations_outer}) reached. Total mass may not be fully converged.")

    # Extract the final calculated total interior radius of the planet
    planet_radius = radii[-1]

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Extract the core-mantle boundary radius from the radii array
    cmb_radius = radii[cmb_index]

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = mass_enclosed[-1] / (4/3 * np.pi * planet_radius**3)

    # Final results of the Zalmoxis interior model
    log.info("Found solution for interior structure with Zalmoxis")
    log.info(f"Interior (dry calculated mass) mass: {mass_enclosed[-1]} kg or approximately {mass_enclosed[-1] / M_earth:.2f} M_earth")
    log.info(f"Interior radius: {planet_radius:.2e} m or {planet_radius / R_earth:.2f} R_earth")
    log.info(f"Core radius: {cmb_radius:.2e} or {cmb_radius / R_earth:.2f} m")
    log.info(f"Core-mantle boundary mass: {mass_enclosed[cmb_index]:.2e} kg")
    log.info(f"Mantle density at the core-mantle boundary: {density[cmb_index]:.2e} kg/m^3")
    log.info(f"Core density at the core-mantle boundary: {density[cmb_index - 1]:.2e} kg/m^3")
    log.info(f"Pressure at the core-mantle boundary: {pressure[cmb_index]:.2e} Pa")
    log.info(f"Pressure at the center: {pressure[0]:.2e} Pa")
    log.info(f"Average density: {average_density:.2e} kg/m^3")
    log.info(f"Core-mantle boundary mass fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}")
    log.info(f"Core radius fraction: {cmb_radius / planet_radius:.3f}")

    # Update the surface radius, interior radius, and mass in the hf_row
    hf_row["R_int"] = planet_radius
    hf_row["M_int"] = mass_enclosed[-1]
    hf_row["M_core"] = mass_enclosed[cmb_index]
    hf_row["gravity"] = gravity[-1]

    # Get the output location for Zalmoxis output
    output_zalmoxis = get_zalmoxis_output_filepath(outdir)
    log.info(f"Saving Zalmoxis output to {output_zalmoxis}")

    # Select values from the arrays for the mantle (to match the mesh needed for Aragog)
    mantle_radii = radii[cmb_index:]
    mantle_pressure = pressure[cmb_index:]
    mantle_density = density[cmb_index:]

    # Save final grids to the output file for the mantle
    with open(output_zalmoxis, 'w') as f:
        for i in range(len(mantle_radii)):
            f.write(f"{mantle_radii[i]:.17e} {mantle_pressure[i]:.17e} {mantle_density[i]:.17e}\n")

    return cmb_radius
