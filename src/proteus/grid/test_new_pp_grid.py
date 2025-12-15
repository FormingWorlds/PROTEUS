from __future__ import annotations

from pathlib import Path

import post_processing_updated as pp

# Point this to your actual grid folder
grid_name = "escape_grid_1Msun"
grid_path = Path(f"/projects/p315557/Paper_1/DATA/Grids/{grid_name}")

# Test function load_grid_cases
data = pp.load_grid_cases(grid_path)

# Test function get_grid_parameters_from_toml
param_grid = pp.get_grid_parameters_from_toml(grid_path)
print("\nExtracted grid parameters:", param_grid)

# Test function extract_grid_output
output_get = "P_surf", "T_surf"
for output in output_get:
    values = pp.extract_grid_output(data, output)
    #print("\nExtracted output values for parameter", output)
    #print(values)

# Test function to extract Phi_crit
phi_crit_values = pp.load_phi_crit(grid_path)
#print("\nExtracted Phi_crit Values:")
#print(phi_crit_values)

# Test function extract_solidification_time
solid_times = pp.extract_solidification_time(data, grid_path)
#print("\nExtracted Solidification Times:")
#print(solid_times)

# Test function extract_grid_output_at_solidificationcreate csv
pp.export_simulation_summary(data, grid_path, param_grid, "nogit_csv_output/simulation_summary.csv")
