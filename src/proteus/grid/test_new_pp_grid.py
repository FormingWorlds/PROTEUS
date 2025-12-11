from __future__ import annotations

from pathlib import Path

import post_processing_updated as pp

# Point this to your actual grid folder
grid_path = Path("//projects/p315557/Paper_1/DATA/Grids/escape_grid_1Msun")

# Test function load_grid_cases
data = pp.load_grid_cases(grid_path)
print("\nFirst case keys:")
print(data[0].keys())
print("\nFirst case status:")
print(data[0]["status"])
print("\nFirst case init parameters (top level keys):")
print(data[0]["init_parameters"].keys())
print("\nFirst case output dataframe head:")
print(data[0]["output_values"].head())

# Test function get_grid_parameters_from_toml
param_grid = pp.get_grid_parameters_from_toml('/home2/p315557/PROTEUS/input/ensembles/escape_grid_1Msun.toml')
print("\nExtracted grid parameters:\n")
for k, v in param_grid.items():
    print(f"{k} -> {v}")
print("\nTotal parameters found:", len(param_grid))

# Test function extract_grid_output
test_parameter = "P_surf"   # <-- change this to a real column
values = pp.extract_grid_output(data, test_parameter)
print("\nExtracted Values:")
print(values)
print("\nTotal extracted values:", len(values))

# Test function extract_solidification_time

try:
    solid_times = pp.extract_solidification_time(data)
    print("\nExtracted Solidification Times:")
    print(solid_times)
except ValueError as e:
    print(f"\nError caught: {e}")
