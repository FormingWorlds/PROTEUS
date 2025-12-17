from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import post_processing_updated as pp

# Load configuration from TOML file
toml_file = Path("/home2/p315557/PROTEUS/input/ensembles/example.grid_analyse.toml")  # Update with the correct path
with open(toml_file, "rb") as f:
    cfg = tomllib.load(f)

# Get grid path and name
grid_path = Path(cfg["grid_path"])
grid_name = Path(pp.get_grid_name(grid_path))
print(grid_path)
print(f"Analyzing grid: {grid_name}")


# Load grid data
data = pp.load_grid_cases(grid_path)
input_param_grid_per_case, tested_params_grid = pp.get_tested_grid_parameters(data, grid_path)

# Generate summary CSV files
if cfg.get("update_csv", True):
    pp.generate_summary_csv(data, input_param_grid_per_case, grid_path, grid_name)
    pp.generate_completed_summary_csv(data, input_param_grid_per_case, grid_path, grid_name)
    pp.generate_running_error_summary_csv(data, input_param_grid_per_case, grid_path, grid_name)

# Plot grid status
if cfg.get("plot_status", True):
    path_csv_all_simulations = grid_path / "post_processing/extracted_data" / f"{grid_name}_final_extracted_data_all.csv"
    all_simulations_data_csv = pd.read_csv(path_csv_all_simulations, sep="\t")
    pp.plot_grid_status(all_simulations_data_csv, grid_path, grid_name)
    print("Plot grid status summary is available.")

# Formatting for ECDF plot: group output data by parameters and prepare settings for plotting only for Completed simulations
path_csv_completed_simulations = grid_path / "post_processing/extracted_data" / f"{grid_name}_final_extracted_data_completed.csv"
completed_simulations_data_csv = pd.read_csv(path_csv_completed_simulations, sep="\t")
columns_output = list(cfg["output_variables"].keys())
grouped_data = {}
for col in columns_output:
    group = pp.group_output_by_parameter(
        completed_simulations_data_csv,
        tested_params_grid,
        [col]
    )
    grouped_data.update(group)
param_settings_grid, output_settings_grid = pp.load_ecdf_plot_settings(cfg)

# Generate ECDF plot
if cfg.get("plot_ecdf", True):
    pp.ecdf_grid_plot(
        grouped_data,
        param_settings_grid,
        output_settings_grid,
        grid_path,
        grid_name
    )
    print("ECDF grid plot is available.")
