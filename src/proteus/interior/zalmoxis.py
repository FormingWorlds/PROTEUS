# Zalmoxis interior module

from proteus.utils.data import get_Zalmoxis
from scipy.interpolate import interp1d
import numpy as np

def calculate_density(pressure, material, eos_choice, interpolation_functions={}):
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
                pressure_data = data[:, 1] * 1e9 # Convert to Pa
                density_data = data[:, 0] * 1e3 # Convert to kg/m^3
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

    else:
        raise ValueError("Invalid EOS choice.")




