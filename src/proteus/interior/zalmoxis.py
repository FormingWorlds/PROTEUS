# Zalmoxis interior module

import logging
from proteus.utils.data import get_Zalmoxis
from scipy.interpolate import interp1d
import numpy as np
from proteus.utils.constants import const_G

log = logging.getLogger("fwl."+__name__)

def get_density_from_eos(pressure, material, eos_choice, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS.
    Args:
        pressure (float): Pressure in Pascals.
        material (str): Material type (e.g., "core", "mantle").
        eos_choice (str): Equation of state choice.
        interpolation_functions (dict): Cache for interpolation functions.
    Returns:
        float: Calculated density in kg/m^3.
    """
    # Get material properties for Zalmoxis
    material_properties = get_Zalmoxis()

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
    Calculate the derivatives of mass, gravity, and pressure with respect to radius for a planetary model.

    Parameters:
    radius (float): The current radius at which the ODEs are evaluated.
    y (list or array): The state vector containing mass, gravity, and pressure at the current radius.
    cmb_mass (float): The core-mantle boundary mass.
    eos_choice (str): The equation of state choice for material properties.
    interpolation_cache (dict): A cache for interpolation to speed up calculations.
    num_layers (int): The number of layers in the planetary model.

    Returns:
    list: The derivatives of mass, gravity, and pressure with respect to radius.
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Define material based on enclosed mass up to the core-mantle boundary
    if mass < cmb_mass:
        material = "core"
    else:
        material = "mantle"

    # Calculate density at the current radius, using pressure from y
    current_density = get_density_from_eos(pressure, material, eos_choice, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None:
        log.warning(f"Density calculation failed at radius {radius}. Using previous density.")

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * const_G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]





