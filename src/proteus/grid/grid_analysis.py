# This script is used to post-process a grid of PROTEUS simulations.
# It extracts the output values from the simulation cases, saves them to a CSV file,
# and generates plots for the grid statuses based on the extracted data. It also generates
# ECDF single plots and a big grid plot for the input parameters vs extracted outputs.

# The users need to specify the path to the grid directory and the grid name. He also needs 
# to specify the output columns to extract from the 'runtime_helpfile.csv' of each case. And 
# update the related plotting variables accordingly.

import os
from postprocess_grid import load_grid_cases, get_grid_parameters, extract_grid_output, extract_solidification_time, \
                            save_grid_data_to_csv, load_extracted_data, plot_dir_exists, group_output_by_parameter, \
                            plot_grid_status, ecdf_single_plots, ecdf_grid_plot 
import matplotlib.cm as cm


if __name__ == '__main__':

    # ---------------------------------------------
    #  Initialization
    # ---------------------------------------------

    # User needs to specify paths
    path_to_grid = '/home2/p315557/PROTEUS/output/scratch/'
    grid_name   = 'escape_grid_habrok_7_params_1Msun'

    update_csv = True  # Set to True if you want to update the CSV file / False if you want to skip the CSV update step

    grid_path   = f'{path_to_grid}{grid_name}/'             # Path to the grid directory
    postprocess_path = f'{grid_path}post_processing_grid/'  # Path to the postprocess directory
    data_dir    = f'{postprocess_path}extracted_data/'     # Path to the directory where the data will be saved
    os.makedirs(data_dir, exist_ok=True)                    # Create the directory if it doesn't exist
    plots_path  = f'{postprocess_path}plots_grid/'          # Path to the directory where the plots will be saved
    plot_dir_exists(plots_path)                             # Check if the plot directory exists. If not, create it.
    csv_file = os.path.join(data_dir, f'{grid_name}_extracted_data.csv')   # Path to the CSV file where the extracted data will be saved

    # User choose the output to extract from 'runtime_helpfile.csv' of each case (always the [-1] element of the column). For the units, check the file src/proteus/utils/coupler.py, lines 348-400 (keys)
    output_to_extract = ['esc_rate_total','Phi_global','P_surf','T_surf','M_planet','atm_kg_per_mol',
                        'H_kg_atm','O_kg_atm','C_kg_atm','N_kg_atm','S_kg_atm', 'Si_kg_atm', 'Mg_kg_atm', 'Fe_kg_atm', 'Na_kg_atm',
                        'H2O_kg_atm','CO2_kg_atm', 'O2_kg_atm', 'H2_kg_atm', 'CH4_kg_atm', 'CO_kg_atm', 'N2_kg_atm', 'NH3_kg_atm',     
                        'S2_kg_atm', 'SO2_kg_atm', 'H2S_kg_atm', 'SiO_kg_atm','SiO2_kg_atm', 'MgO_kg_atm', 'FeO2_kg_atm']      
    
    # ---------------------------------------------
    #  STEP 1: Post-processing the grid → producing CSV
    # ---------------------------------------------
    # Only run Step 1 if update_csv=True OR if the CSV does not exist yet
    
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
        "orbit.semimajoraxis":        {"label": "Semi-major axis [AU]",                   "colormap": cm.plasma,   "log_scale": False},
        "escape.zephyrus.Pxuv":       {"label": r"$P_{XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
        "escape.zephyrus.efficiency": {"label": r"Escape efficiency factor $\epsilon$",   "colormap": cm.spring,   "log_scale": False},
        "outgas.fO2_shift_IW":        {"label": r"$\log_{10}(fO_2)$ [IW]",                "colormap": cm.coolwarm, "log_scale": False},
        "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.rainbow,  "log_scale": False},
        "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
        "delivery.elements.H_oceans": {"label": "[H] [Earth's oceans]",                   "colormap": cm.winter,   "log_scale": False}, 
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
        "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.rainbow,    "log_scale": False},
        "orbit.semimajoraxis":        {"label": "a [AU]",                                 "colormap": cm.plasma,   "log_scale": False},
        "escape.zephyrus.efficiency": {"label": r"$\epsilon$",                            "colormap": cm.spring,   "log_scale": False},
        "escape.zephyrus.Pxuv":       {"label": r"$P_{XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
        "outgas.fO2_shift_IW":        {"label": r"$\log_{10}(fO_2 / IW)$",                "colormap": cm.coolwarm, "log_scale": False},
        "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
        "delivery.elements.H_oceans": {"label": "[H] [oceans]",                           "colormap": cm.winter,   "log_scale": False}}
    output_settings_grid = {
        'solidification_time': {"label": "Solidification [yr]",             "log_scale": True,  "scale": 1.0},
        'Phi_global':          {"label": "Melt fraction [%]",               "log_scale": False, "scale": 100.0},
        'P_surf':              {"label": "Surface pressure [bar]",          "log_scale": True,  "scale": 1.0},
        'esc_rate_total':      {"label": "Escape rate [kg/s]",              "log_scale": True,  "scale": 1.0},
        'atm_kg_per_mol':      {"label": "MMW [g/mol]",                     "log_scale": False,  "scale": 1000.0},
        'T_surf':              {"label": r"T$_{surf}$ [K]",                 "log_scale": False,  "scale": 1.0},
        'M_planet':            {"label": r"M$_p$ [M$_\oplus$]",             "log_scale": False,  "scale": 1.0/5.9722e24}
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
    print('If you want to change the parameters to post-process the grid, please edit the code in PROTEUS/src/proteus/grid/grid_analysis.py')
    print('-----------------------------------------------------------')
