# Zalmoxis interior module

import logging
import time, os

from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import solve_ivp

from proteus.utils.constants import const_G, earth_center_pressure, R_earth, M_earth
from proteus.utils.data import get_Seager_EOS
from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Path to location at which to save the output data from Zalmoxis
def get_zalmoxis_output_filepath(outdir: str):
    return os.path.join(outdir, "data", "zalmoxis_output.dat")

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
    material_properties = get_Seager_EOS()

    # Shorthand for material properties
    props = material_properties[material]

    # Select the equation of state and calculate density
    if eos_choice == "Tabulated":
        try:
            eos_file = props["eos_file"]
            # Caching: Store interpolation functions for reuse
            if eos_file not in interpolation_functions:
                data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
                pressure_data = data[:, 1] * 1e9 # Convert from GPa to Pa
                density_data = data[:, 0] * 1e3 # Convert from g/cm^3 to kg/m^3
                interpolation_functions[eos_file] = interp1d(pressure_data, density_data, bounds_error=False, fill_value="extrapolate")

            interpolation_function = interpolation_functions[eos_file]  # Retrieve from cache
            density = interpolation_function(pressure)  # Call to cached function

            if density is None or np.isnan(density):
                log.error(f"Density calculation failed for {material} at {pressure:.2e} Pa.")

            return density

        except (ValueError, OSError) as e: # Catch file errors
            log.error(f"Error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
            return None
        except Exception as e: # Other errors
            log.error(f"Unexpected error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
            return None

    else:
        log.error(f"Invalid EOS choice: {eos_choice}.")

def interior_structure_odes(radius, y, cmb_mass, eos_choice, interpolation_cache):
    """
    Calculates the derivatives of mass, gravity, and pressure with respect to radius for a planetary model.

    Parameters:
    radius (float): The current radius at which the ODEs are evaluated [m].
    y (list or array): The state vector containing mass [kg], gravity [m/s^2], and pressure [Pa] at the current radius.
    cmb_mass (float): The core-mantle boundary mass [kg].
    eos_choice (str): The equation of state choice for material properties.
    interpolation_cache (dict): A cache for interpolation to speed up calculations.
    num_layers (int): The number of layers in the planetary model.

    Returns the list of derivatives of mass, gravity, and pressure with respect to radius.
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Define the material type based on the enclosed mass up to the core-mantle boundary
    if mass < cmb_mass:
        # Core
        material = "core"
    else:
        # Mantle
        material = "mantle"

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

def zalmoxis_solver(config:Config, outdir:str):

    """
    Zalmoxis interior model solver.

    """
    # Get material properties for Zalmoxis
    material_properties = get_Seager_EOS()

    # Setup target planet mass (input parameter)
    planet_mass = config.struct.mass_tot * M_earth

    # Setup assumptions
    core_radius_fraction = config.struct.corefrac
    core_mass_fraction = config.struct.coremassfrac
    weight_iron_fraction = config.struct.weight_iron_frac
    eos_choice = config.interior.zalmoxis.EOSchoice

    # Setup calculation parameters for the iterative solver
    num_layers = config.interior.zalmoxis.num_levels
    max_iterations_outer = config.interior.zalmoxis.max_iterations_outer
    tolerance_outer = config.interior.zalmoxis.tolerance_outer
    tolerance_radius = config.interior.zalmoxis.tolerance_radius
    max_iterations_inner = config.interior.zalmoxis.max_iterations_inner
    tolerance_inner = config.interior.zalmoxis.tolerance_inner
    relative_tolerance = config.interior.zalmoxis.relative_tolerance
    absolute_tolerance = config.interior.zalmoxis.absolute_tolerance
    target_surface_pressure = config.interior.zalmoxis.target_surface_pressure
    pressure_tolerance = config.interior.zalmoxis.pressure_tolerance
    max_iterations_pressure = config.interior.zalmoxis.max_iterations_pressure
    pressure_adjustment_factor = config.interior.zalmoxis.pressure_adjustment_factor

    # Setup initial guesses for the planet radius and core-mantle boundary radius
    radius_guess = 1000*(7030-1840*weight_iron_fraction)*(planet_mass/M_earth)**0.282 # Initial guess for the interior planet radius [m] based on the scaling law in Noack et al. 2020
    cmb_radius = core_radius_fraction * radius_guess # Initial guess for the core-mantle boundary radius [m]
    cmb_radius_previous = cmb_radius # Initial guess for the previous core-mantle boundary radius [m]

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer): # Outer loop for radius and mass convergence
        # Start timing the outer loop
        start_time = time.time()
        # Setup initial guess for the radial grid based on the radius guess
        radii = np.linspace(0, radius_guess, num_layers)

        # Initialize arrays for mass, gravity, density, and pressure grids
        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        # Setup initial guess for the core-mantle boundary mass
        cmb_mass = core_mass_fraction * planet_mass

        # Setup initial guess for the pressure at the center of the planet (needed for solving the ODEs)
        pressure[0] = earth_center_pressure

        # Setup initial guess for the density grid
        for i in range(num_layers):
            if radii[i] < cmb_radius:
                log.info(f"Density at outer iteration {outer_iter+1} and radius {radii[i]} belongs to the core.")
                density[i] = material_properties["core"]["rho0"]
            else:
                log.info(f"Density at outer iteration {outer_iter+1} and radius {radii[i]} belongs to the mantle.")
                density[i] = material_properties["mantle"]["rho0"]

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
                sol = solve_ivp(lambda r, y: interior_structure_odes(r, y, cmb_mass, eos_choice, interpolation_cache),
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
                    log.info(f"Surface pressure converged after {pressure_iter + 1} iterations and all pressures are positive.")
                    break  # Exit the pressure adjustment loop

                # Update the pressure guess at the center of the planet based on the pressure difference at the surface using an adjustment factor
                pressure_guess -= pressure_diff * pressure_adjustment_factor

            # Update density grid based on the calculated pressure and mass enclosed
            for i in range(num_layers):
                # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
                if mass_enclosed[i] < cmb_mass:
                    # Core
                    material = "core"
                else:
                    # Mantle
                    material = "mantle"

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
                log.info(f"Inner loop converged after {inner_iter + 1} iterations.")
                break # Exit the inner loop

        # Extract the calculated total interior mass of the planet from the last element of the mass array
        calculated_mass = mass_enclosed[-1]

        # Update the total interior radius by scaling the initial guess based on the calculated mass
        radius_guess = radius_guess * (planet_mass / calculated_mass)**(1/3)

        # Update the core-mantle boundary radius based on the calculated mass grid
        cmb_radius = radii[np.argmax(mass_enclosed >= cmb_mass)]

        # Update the core-mantle boundary mass based on the core mass fraction and calculated total interior mass of the planet
        cmb_mass = core_mass_fraction * calculated_mass

        # Calculate relative differences of the calculated total interior mass and core-mantle boundary radius
        relative_diff_outer = abs((calculated_mass - planet_mass) / planet_mass)
        relative_diff_radius = abs((cmb_radius - cmb_radius_previous) / cmb_radius)

        # Check for convergence of the calculated total interior mass and core-mantle boundary radius of the planet
        if relative_diff_outer < tolerance_outer and relative_diff_radius < tolerance_radius:
            log.info(f"Outer loop (cmb radius and total mass) converged after {outer_iter + 1} iterations.")
            break  # Exit the outer loop

        # Update previous core-mantle boundary radius for the next iteration
        cmb_radius_previous = cmb_radius

        # End timing the outer loop
        end_time = time.time()
        log.info(f"Outer iteration {outer_iter+1} took {end_time - start_time:.2f} seconds")

        # Check if maximum iterations for outer loop are reached
        if outer_iter == max_iterations_outer - 1:
            log.warning(f"Maximum outer iterations ({max_iterations_outer}) reached. Radius and mass may not be fully converged.")

    # Extract the final calculated total interior radius of the planet
    planet_radius = radius_guess

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = calculated_mass / (4/3 * np.pi * planet_radius**3)

    # Final results of the Zalmoxis interior model
    log.info("Zalmoxis interior structure model results:")
    log.info(f"Interior mass: {calculated_mass:.2e} kg or {calculated_mass / M_earth:.2f} M_earth")
    log.info(f"Interior radius: {planet_radius:.2e} m or {planet_radius / R_earth:.2f} R_earth")
    log.info(f"Core radius: {cmb_radius:.2e} m")
    log.info(f"Core-mantle boundary mass: {cmb_mass:.2e} kg")
    log.info(f"Mantle density at the core-mantle boundary: {density[cmb_index]:.2f} kg/m^3")
    log.info(f"Core density at the core-mantle boundary: {density[cmb_index - 1]:.2f} kg/m^3")
    log.info(f"Pressure at the core-mantle boundary: {pressure[cmb_index]:.2e} Pa")
    log.info(f"Pressure at the center: {pressure[0]:.2e} Pa")
    log.info(f"Average density: {average_density:.2f} kg/m^3")
    log.info(f"Core-mantle boundary mass fraction: {cmb_mass / calculated_mass:.2f}")
    log.info(f"Core radius fraction: {cmb_radius / planet_radius:.2f}")

    # Get the output location for Zalmoxis output
    output_zalmoxis = get_zalmoxis_output_filepath(outdir)
    log.info(f"Saving Zalmoxis output to {output_zalmoxis}")

    # Save final grids for radius, density, gravity, pressure, and mass enclosed to the output file
    with open(output_zalmoxis, 'w') as f:
        for i in range(len(radii)):
            f.write(f"{radii[i]} {density[i]} {gravity[i]} {pressure[i]} {mass_enclosed[i]}\n")

    return radii, density, gravity, pressure, mass_enclosed
