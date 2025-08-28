# This script is used to post-process a grid of PROTEUS simulations.
# It extracts the output values from the simulation cases, saves them to a CSV file,
# and generates plots for the grid statuses based on the extracted data. It also generates
# ECDF single plots and a big grid plot for the input parameters vs extracted outputs.

# The users need to specify the path to the grid directory and the grid name. (see the example below)
# He also needs to specify the output columns to extract from the 'runtime_helpfile.csv' of each case and
# update the related plotting variables accordingly. This can be done in the `run_grid_analyze` function (see below).
from __future__ import annotations

import os

import matplotlib.cm as cm

from proteus.grid.post_processing_grid import (
    ecdf_grid_plot,
    ecdf_single_plots,
    extract_grid_output,
    extract_solidification_time,
    get_grid_parameters,
    group_output_by_parameter,
    load_extracted_data,
    load_grid_cases,
    plot_dir_exists,
    plot_grid_status,
    save_error_running_cases,
    save_grid_data_to_csv,
)


def run_grid_analyze(path_to_grid: str, grid_name: str, update_csv: bool = True):
    """
    Run the post-processing of a PROTEUS grid, extracting simulation data in a CSV file and generating plots.

    Parameters
    ----------
    path_to_grid : str
        Path to the directory containing the grid folder.

    grid_name : str
        Name of the grid folder to process.

    update_csv : bool, optional
        If True, the CSV file will be updated or created. If False, it will skip the CSV extraction step if the file already exists.
    """

    # ------------------------------------------------------------
    #  1) Build all the folder/filename strings
    # ------------------------------------------------------------
    grid_path = os.path.join(path_to_grid, grid_name) + os.sep
    postprocess_path = os.path.join(grid_path, "post_processing_grid") + os.sep
    data_dir = os.path.join(postprocess_path, "extracted_data") + os.sep
    os.makedirs(data_dir, exist_ok=True)


    plots_path = os.path.join(postprocess_path, "plots_grid") + os.sep
    plot_dir_exists(plots_path)

    csv_file = os.path.join(data_dir, f"{grid_name}_extracted_data.csv")

    # ------------------------------------------------------------
    #  2) Define which outputs to pull from each case's runtime_helpfile.csv
    # ------------------------------------------------------------
    # User choose the output to extract from 'runtime_helpfile.csv' of each case (always the [-1] element of the column).
    # For the units, check the file src/proteus/utils/coupler.py, lines 348-400 (keys)

    output_to_extract = ['Time','esc_rate_total','Phi_global','P_surf','T_surf','M_planet','R_obs','p_xuv', 'R_xuv', 'atm_kg_per_mol',
                        'H_kg_atm','O_kg_atm','C_kg_atm','N_kg_atm','S_kg_atm', 'Si_kg_atm', 'Mg_kg_atm', 'Fe_kg_atm', 'Na_kg_atm',
                        'H2O_kg_atm','CO2_kg_atm', 'O2_kg_atm', 'H2_kg_atm', 'CH4_kg_atm', 'CO_kg_atm', 'N2_kg_atm', 'NH3_kg_atm',
                        'S2_kg_atm', 'SO2_kg_atm', 'H2S_kg_atm', 'SiO_kg_atm','SiO2_kg_atm', 'MgO_kg_atm', 'FeO2_kg_atm', 'runtime']

    # ------------------------------------------------------------
    #  STEP 1: CSV extraction (only if update_csv=True or CSV missing)
    # ------------------------------------------------------------

    if update_csv or not os.path.isfile(csv_file):

        print('-----------------------------------------------------------')
        print(f'Step 1 : Post-processing the grid {grid_name} ...')
        print('-----------------------------------------------------------')

        extracted_value = {}                                                                # Initialize the dictionary to store extracted values

        cases_data = load_grid_cases(grid_path)                                             # Load all simulation cases
        grid_parameters, case_init_param = get_grid_parameters(grid_path)                   # Extract grid parameters

        for param in output_to_extract:
            extracted_value[param] = extract_grid_output(cases_data, param)                 # Extract output values

        solidification_times = extract_solidification_time(cases_data)                      # Extract the solidification time
        extracted_value['solidification_time'] = solidification_times                       # Add solidification time to the extracted_values

        save_grid_data_to_csv(grid_name, cases_data, grid_parameters, case_init_param,
                        extracted_value, solidification_times, data_dir)                  # Save all the extracted data to a CSV file
        print(f'--> CSV file written to: {csv_file}')

        save_error_running_cases(
            grid_name=grid_name,
            cases_data=cases_data,
            grid_parameters=grid_parameters,
            case_params=case_init_param,
            extracted_value=extracted_value,
            output_to_extract=output_to_extract,
            output_dir=data_dir)
    else:
        print('-----------------------------------------------------------')
        print(f'Step 1 : Skipped (CSV already exists at {csv_file})')
        print('-----------------------------------------------------------')


    # ---------------------------------------------
    #  STEP 2: Load data from CSV and make plots
    # ---------------------------------------------

    print('-----------------------------------------------------------')
    print(f'Step 2 : Loading data and plotting for grid {grid_name} ...')
    print('-----------------------------------------------------------')


    df, grid_params, extracted_outputs = load_extracted_data(data_dir, grid_name)       # Load the data
    grouped_data = group_output_by_parameter(df, grid_params, extracted_outputs)        # Group extracted outputs by grid parameters

    # Histogram of grid statuses
    plot_grid_status(df, plots_path, grid_name)                                         # Plot the grid statuses in an histogram

    # Single ECDF Plots
    # The user needs to comment the parameters he didn't used in the grid/ add the ones non-listed here. Same for the outputs.
    param_settings_single = {
        # "orbit.semimajoraxis":        {"label": "Semi-major axis [AU]",                   "colormap": cm.plasma,   "log_scale": False},
        # "escape.zephyrus.Pxuv":       {"label": r"$P_{XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
        "escape.zephyrus.efficiency": {"label": r"Escape efficiency factor $\epsilon$",   "colormap": cm.spring,   "log_scale": False},
        # "outgas.fO2_shift_IW":        {"label": r"$\log_{10}(fO_2)$ [IW]",                "colormap": cm.coolwarm, "log_scale": False},
        # "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.rainbow,  "log_scale": False},
        "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
        "delivery.elements.H_oceans": {"label": "[H] [Earth's oceans]",                   "colormap": cm.winter,   "log_scale": False},
        "delivery.elements.SH_ratio": {"label": "S/H ratio",                              "colormap": cm.autumn,   "log_scale": False},
        # "escape.reservoir":           {"label": "Reservoir",                              "colormap": cm.viridis,  "log_scale": False}
        # "escape.module":           {"label": "Escape module",                              "colormap": cm.RdYlGn,  "log_scale": False}
        }
    output_settings_single = {
        'esc_rate_total':      {"label": "Total escape rate [kg/s]",                  "log_scale": True,  "scale": 1.0},
        'Phi_global':          {"label": "Melt fraction [%]",                         "log_scale": False, "scale": 100.0},
        'P_surf':              {"label": "Surface pressure [bar]",                    "log_scale": True,  "scale": 1.0},
        'atm_kg_per_mol':      {"label": "Mean molecular weight (MMW) [g/mol]",       "log_scale": False,  "scale": 1000.0},
        'solidification_time': {"label": "Solidification time [yr]",                  "log_scale": True,  "scale": 1.0},
        'T_surf':              {"label": r"T$_{surf}$ [K]",                 "log_scale": False,  "scale": 1.0},
        'M_planet':            {"label": r"M$_p$ [M$_\oplus$]",             "log_scale": False,  "scale": 1.0/5.9722e24},
        'H_kg_atm':            {"label": r"[H$_{atm}$] [kg]",               "log_scale": True,  "scale": 1.0}
        }
    ecdf_single_plots(grid_params=grid_params, grouped_data=grouped_data, param_settings=param_settings_single, output_settings=output_settings_single, plots_path=plots_path)

    # ECDF Grid Plot
    # The user needs to comment the parameters he didn't used in the grid/ add the ones non-listed here. Same for the outputs.
    param_settings_grid = {
        # "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.rainbow,    "log_scale": False},
        # "orbit.semimajoraxis":        {"label": "a [AU]",                                 "colormap": cm.plasma,   "log_scale": False},
        "escape.zephyrus.efficiency": {"label": r"$\epsilon$",                            "colormap": cm.spring,   "log_scale": False},
        # "escape.zephyrus.Pxuv":       {"label": r"$P_{XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
        # "outgas.fO2_shift_IW":        {"label": r"$\log_{10}(fO_2 / IW)$",                "colormap": cm.coolwarm, "log_scale": False},
        "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
        "delivery.elements.H_oceans": {"label": "[H] [oceans]",                           "colormap": cm.winter,   "log_scale": True},
        "delivery.elements.SH_ratio": {"label": "S/H ratio",                              "colormap": cm.autumn,   "log_scale": True},
        # "escape.reservoir":           {"label": "Reservoir",                              "colormap": cm.viridis,  "log_scale": False}
        # "escape.module":           {"label": "Escape module",                              "colormap": cm.RdYlGn,  "log_scale": False}
        }
    output_settings_grid = {
        'solidification_time': {"label": "Solidification [yr]",             "log_scale": True,  "scale": 1.0},
        'Phi_global':          {"label": "Melt fraction [%]",               "log_scale": False, "scale": 100.0},
        'P_surf':              {"label": r"P$_{surf}$ [bar]",          "log_scale": True,  "scale": 1.0},
        'esc_rate_total':      {"label": "Escape rate [kg/s]",              "log_scale": True,  "scale": 1.0},
        'T_surf':              {"label": r"T$_{surf}$ [$10^3$ K]",                 "log_scale": False,  "scale": 1.0/1000.0},
        'atm_kg_per_mol':      {"label": "MMW [g/mol]",                     "log_scale": False,  "scale": 1000.0},
        # 'M_planet':            {"label": r"M$_p$ [M$_\oplus$]",             "log_scale": False,  "scale": 1.0/5.9722e24},
        'H_kg_atm':            {"label": r"[H$_{atm}$] [kg]",               "log_scale": True,  "scale": 1.0},
        #'O_kg_atm':            {"label": r"[O$_{atm}$] [kg]",               "log_scale": True,  "scale": 1.0},
        #'C_kg_atm':            {"label": r"[C$_{atm}$] [kg]",               "log_scale": True,  "scale": 1.0},
        #'N_kg_atm':            {"label": r"[N$_{atm}$] [kg]",               "log_scale": True,  "scale": 1.0},
        #'S_kg_atm':            {"label": r"[S$_{atm}$] [kg]",               "log_scale": True,  "scale": 1.0}
        }
    ecdf_grid_plot(grid_params=grid_params, grouped_data=grouped_data, param_settings=param_settings_grid, output_settings=output_settings_grid, plots_path=plots_path)

    print('-----------------------------------------------------------')
    print(f'Plots saved in {plots_path}')
    print(f'Post-processing of grid {grid_name} completed successfully!')
    print('-----------------------------------------------------------')
    print('If you want to change the parameters to post-process the grid, please edit the code in PROTEUS/src/proteus/grid/post_processing_grid.py')
    print('-----------------------------------------------------------')
