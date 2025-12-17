from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import pandas as pd
import post_processing_updated as pp

grid_name = "escape_grid_1Msun"
grid_path = Path(f"/projects/p315557/Paper_1/DATA/Grids/{grid_name}")
# grid_name= "toi561b_grid"
# grid_path = Path(f"/projects/p315557/TOI-561b/{grid_name}")
# grid_name= "escape_on_off_janus_agni_1Msun"
# grid_path = Path(f"/projects/p315557/Paper_1/DATA/Comparison_escape_on_off/{grid_name}")

# Load grid data
data = pp.load_grid_cases(grid_path)
# Store the input parameters for each case based on the tested parameters in the grid and store the tested parameters in the grid
input_param_grid_per_case, tested_params_grid = pp.get_tested_grid_parameters(data,grid_path)

# Generate CSV file for all grid, only completed simulations, and running error simulations
#pp.generate_summary_csv(data,input_param_grid_per_case,grid_path,grid_name)
#pp.generate_completed_summary_csv(data,input_param_grid_per_case,grid_path,grid_name)
#pp.generate_running_error_summary_csv(data,input_param_grid_per_case,grid_path,grid_name)

# Load the final extracted data CSV for all simulations
path_csv_all_simulations = grid_path / "post_processing/extracted_data" / f"{grid_name}_final_extracted_data_all.csv"
all_simulations_data_csv = pd.read_csv(path_csv_all_simulations, sep="\t")
# Plot grid status
pp.plot_grid_status(all_simulations_data_csv,grid_path,grid_name)

# Load the final extracted data CSV for completed simulations
path_csv_completed_simulations = grid_path / "post_processing/extracted_data" / f"{grid_name}_final_extracted_data_completed.csv"
completed_simulations_data_csv = pd.read_csv(path_csv_completed_simulations, sep="\t")

# Group output data by tested parameters in the grid
grouped_data = {}
columns_output = ['solidification_time', 'Phi_global', 'T_surf', 'P_surf', 'atm_kg_per_mol', 'esc_rate_total']
for col in columns_output:
    group = pp.group_output_by_parameter(
        completed_simulations_data_csv,
        tested_params_grid,
        [col]
    )
    grouped_data.update(group)

# Define parameter and output settings for the ECDF grid plots
param_settings_grid = {
    "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.viridis,    "log_scale": False},
    "orbit.semimajoraxis":        {"label": "a [AU]",                                 "colormap": cm.viridis,   "log_scale": False},
    "escape.zephyrus.efficiency": {"label": r"$\rm \epsilon$",                            "colormap": cm.viridis,   "log_scale": False},
    "escape.zephyrus.Pxuv":       {"label": r"P$_{\rm XUV}$ [bar]",                       "colormap": cm.viridis,  "log_scale": True},
    "outgas.fO2_shift_IW":        {"label": r"$\rm \log_{10} fO_2 [IW]$",                "colormap": cm.viridis, "log_scale": False},
    "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.viridis,   "log_scale": False},
    "delivery.elements.H_oceans": {"label": "[H] [oceans]",                           "colormap": cm.viridis,   "log_scale": False}}
output_settings_grid = {
    'solidification_time': {"label": "Solidification [yr]",             "log_scale": True,  "scale": 1.0},
    'Phi_global':          {"label": "Melt fraction [%]",               "log_scale": False, "scale": 100.0},
    'T_surf':              {"label": r"T$_{\rm surf}$ [$10^3$ K]",                 "log_scale": False,  "scale": 1.0/1000.0},
    'P_surf':              {"label": r"P$_{\rm surf}$ [bar]",          "log_scale": True,  "scale": 1.0},
    'atm_kg_per_mol':      {"label": "MMW [g/mol]",                     "log_scale": False,  "scale": 1000.0},
    'esc_rate_total':      {"label": "Escape rate [kg/s]",              "log_scale": True,  "scale": 1.0}}

pp.ecdf_grid_plot(grouped_data, param_settings_grid, output_settings_grid, grid_path, grid_name)
