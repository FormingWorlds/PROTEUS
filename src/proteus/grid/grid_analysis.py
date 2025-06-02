import os
from functions_grid_analyze_and_plot import load_grid_cases, get_grid_parameters, extract_grid_output, extract_solidification_time, save_grid_data_to_csv, load_extracted_data, plot_dir_exists, group_output_by_parameter, plot_grid_status

if __name__ == '__main__':

    # User needs to specify paths
    path_to_grid = '/home2/p315557/PROTEUS/output/scratch/'
    grid_name   = 'escape_grid_habrok_7_params_1Msun'

    grid_path   = f'{path_to_grid}{grid_name}/'     # Path to the grid directory
    data_dir    = f'{grid_path}/extracted_data/'    # Path to the directory where the data will be saved
    os.makedirs(data_dir, exist_ok=True)            # Create the directory if it doesn't exist
    plots_path  = f'{grid_path}plots_grid/'        # Path to the directory where the plots will be saved
    plot_dir_exists(plots_path)                     # Check if the plot directory exists. If not, create it.

    # User choose the parameters to post-process the grid
    output_to_extract = ['esc_rate_total','Phi_global','P_surf','T_surf','M_planet','atm_kg_per_mol']      # Output columns to extract from 'runtime_helpfile.csv' of each case. For the units, check the file src/proteus/utils/coupler.py, lines 348-400 (keys)

    ### Step 1: Post-processing the grid 

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

    ### Step 2: Load data and plot

    print('-----------------------------------------------------------')
    print(f'Step 2 : Loading data and plotting for grid {grid_name} ...') 
    print('-----------------------------------------------------------')
    
    df, grid_params, extracted_outputs = load_extracted_data(data_dir, grid_name)       # Load the data
    grouped_data = group_output_by_parameter(df, grid_params, extracted_outputs)        # Group extracted outputs by grid parameters
    
    # Plots
    plot_grid_status(df, plots_path, grid_name)                                         # Plot the grid statuses in an histogram

    print('-----------------------------------------------------------')
    print(f'Plots saved in {plots_path}')
    print(f'Post-processing of grid {grid_name} completed successfully!')
    print('-----------------------------------------------------------')
    print('If you want to change the parameters to post-process the grid, please edit the code in PROTEUS/tools/post_processing_grid/grid_analysis.py')
    print('-----------------------------------------------------------')
