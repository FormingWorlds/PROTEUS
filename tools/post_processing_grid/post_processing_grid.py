import os
from pathlib import Path
import pandas as pd
import toml
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import csv


def load_grid_cases(grid_dir: Path):
    """
    Load information for each simulation of a PROTEUS grid.

    Read runtime_helpfile.csv, init_parameters (from toml file), and status 
    files for each simulation of the grid.

    Parameters
    ----------
        grid_dir : Path
            Path to the grid directory containing the case_* folders

    Returns
    ----------
        combined_data : list
            List of dictionaries, each containing:
                - 'init_parameters' (dict):
                    Parameters loaded from init_coupler.toml
                - 'output_values' (pandas.DataFrame):
                    Data from runtime_helpfile.csv
                - 'status' (str or None):
                    Status string from status file (if available)
    """
    combined_data = []
    grid_dir = Path(grid_dir)

    # Load all cases from the grid
    for case in grid_dir.glob('case_*'):
        runtime_file = case / 'runtime_helpfile.csv'
        init_file = case / 'init_coupler.toml'
        status_file = case / 'status'

        if runtime_file.exists() and init_file.exists():
            try:
                df = pd.read_csv(runtime_file, sep='\t')
                init_params = toml.load(open(init_file))

                # Read status file if it exists, otherwise set to None
                status = None
                if status_file.exists():
                    with open(status_file, 'r') as sf:
                        status = sf.read().replace('\n', ' ').strip()

                combined_data.append({
                    'init_parameters': init_params,
                    'output_values': df,
                    'status': status
                })

            except Exception as e:
                print(f"Error processing {case.name}: {e}")

    statuses = [str(case.get('status', 'unknown')).strip() or 'unknown' for case in combined_data]
    status_counts = pd.Series(statuses).value_counts().sort_values(ascending=False)
    print('-----------------------------------------------------------')
    print(f"Total number of simulations: {len(statuses)}")
    print('-----------------------------------------------------------')
    print("Number of simulations per status:")
    for status, count in status_counts.items():
        print(f"  - '{status}': {count}")
    print('-----------------------------------------------------------')
    return combined_data

def plot_grid_status(cases_data, plot_dir: Path, status_colors: dict = None):
    """
    Plot the status of simulation from the PROTEUS grid.

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing the status of all simulation from the grid.

    plot_dir : Path
        Path to the plots directory

    status_colors : dict, optional
        A dictionary mapping statuses to specific colors. If None, a default palette is used.

    Returns
    -------
        Plot saved to the specified directory.
    """

    # Extract and clean statuses
    statuses = [case.get('status', 'unknown') or 'unknown' for case in cases_data]
    status_counts = pd.Series(statuses).value_counts().sort_values(ascending=False)
    # print("Unique statuses found:", pd.Series(statuses).unique())
    
    # Set colors for the bars
    if status_colors:
        palette = {str(status): status_colors.get(str(status), 'gray') for status in status_counts.index}
    else:
        palette = sns.color_palette("Accent", len(status_counts))
        palette = dict(zip(status_counts.index, palette))

    # Prepare dataframe for plotting
    plot_df = pd.DataFrame({
        'Status': status_counts.index,
        'Count': status_counts.values
    })

    #sns.set(style="white")
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        data=plot_df,
        x='Status',
        y='Count',
        hue='Status',  # required to apply the palette
        palette=palette,
        dodge=False,
        edgecolor='black'  # edge color added here
    )
    
    # Remove legend if it was created
    if ax.legend_:
        ax.legend_.remove()

    # Add text on top of bars
    total_simulations = len(cases_data)
    for i, count in enumerate(status_counts.values):
        percentage = (count / total_simulations) * 100
        ax.text(
            i, count + 1,
            f"{count} ({percentage:.1f}%)",
            ha='center', va='bottom', fontsize=10
        )


    plt.title(f"Total number of simulations: {total_simulations}", fontsize=16)
    plt.xlabel("Simulation status", fontsize=16)
    plt.ylabel("Number of simulations", fontsize=16)
    plt.yticks(fontsize=12)  
    plt.xticks(fontsize=12)
    plt.tight_layout()
    output_path = plot_dir+'grid_status_summary.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot grid_status_summary.png saved to {output_path}")

def get_grid_parameters(grid_dir: str):
    """
    Extracts grid parameters names and values from the manager.log file
    
    Parameters
    ----------
    grid_dir : str
        Path to the grid directory
    
    Returns
    -------
    dict
        A dictionary where each key is the parameter name and the value is a list of parameter values.
    """
    # Dictionary to store the parameters and their values
    parameters_dict = {}

    # Open and read the log file
    log_file = os.path.join(grid_dir, 'manager.log')
    with open(log_file, 'r') as file:
        lines = file.readlines()

    # Regular expression patterns to find the relevant lines
    param_pattern = re.compile(r'-- dimension: (.+)')
    value_pattern = re.compile(r'values\s+:\s+\[([^\]]+)\]')

    current_dimension = None

    for line in lines:
        # Look for dimension lines to start processing the parameters
        dimension_match = param_pattern.search(line)
        if dimension_match:
            current_dimension = dimension_match.group(1).strip()

        # If we have found a dimension, look for the values line
        if current_dimension and 'values' in line:
            value_match = value_pattern.search(line)
            if value_match:
                # Extract the values from the line and convert them into a list of floats or strings
                values_str = value_match.group(1).strip()
                values = [eval(value.strip()) for value in values_str.split(',')]

                # Store the values under the current dimension
                parameters_dict[current_dimension] = values
                current_dimension = None  # Reset current dimension after processing

    # Print the extracted parameters
    print('-----------------------------------------------------------')
    print("Extracted Parameters:")
    print("-----------------------------------------------------------")
    for param, values in parameters_dict.items():
        print(f"{param}: {values}")
    print("-----------------------------------------------------------")
    return parameters_dict

def extract_grid_output(cases_data, parameter_name):
    """
    Extract a specific parameter from the 'output_values' of each simulation case.

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing simulation data.

    parameter_name : str
        The name of the parameter to extract from 'output_values'.

    Returns
    -------
    parameter_values : list
        A list containing the extracted values of the specified parameter for all cases.
    """
    parameter_values = []
    columns_printed = False  # Flag to print columns only once

    for case in cases_data:
        df = case['output_values']
        
        # Check if the parameter exists in the output dataframe
        if parameter_name in df.columns:
            # Extract the last value of the parameter from the last row
            parameter_value = df[parameter_name].iloc[-1]
            parameter_values.append(parameter_value)
        else:
            if not columns_printed:
                print(f"Warning: Parameter '{parameter_name}' does not exist in case '{case['init_parameters'].get('name', 'Unknown')}'")
                print(f"Available columns in this case: {', '.join(df.columns)}")
                columns_printed = True
    
    # Print the extracted values
    print(f"Extracted output (at last time step) : {parameter_name} ")

    return parameter_values

def extract_solidification_time(cases_data, phi_crit):
    """
    Extract the solidification time for planets that reach Phi_global < phi_crit.

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing simulation data.

    phi_crit : float
        The critical value of melt fraction at which a planet is considered solidified.

    Returns
    -------
    solidification_times : list
        A list containing the solidification times for all solidified planets of the grid. 
        If a planet never solidifies, it will have NaN in the list.
    """
    solidification_times = []
    columns_printed = False
    
    for i, case in enumerate(cases_data):
        df = case['output_values']

        if 'Phi_global' in df.columns and 'Time' in df.columns:
            condition = df['Phi_global'] < phi_crit
            if condition.any():
                first_index = condition.idxmax()  # gives the first True index
                solid_time = df.loc[first_index, 'Time']
                solidification_times.append(solid_time)
            else:
                solidification_times.append(np.nan)  # Append NaN if condition is not met
        else:
            if not columns_printed:
                print("Warning: 'Phi_global' and/or 'Time' columns not found in some cases.")
                print(f"Available columns: {', '.join(df.columns)}")
                columns_printed = True
            solidification_times.append(np.nan)  # Append NaN if columns are missing

    # Count the number of cases labeled as 10 Completed (solidified)
    status_10_cases = [case for case in cases_data if (case.get('status') or '').strip() == '10 Completed (solidified)']
    completed_count = len(status_10_cases)

    # Count only valid solidification times (non-NaN)
    valid_solidification_times = [time for time in solidification_times if not np.isnan(time)]
    valid_solidified_count = len(valid_solidification_times)

    print('-----------------------------------------------------------')
    print(f"Extracted solidification times (Phi_global < {phi_crit})")
    print(f"→ Found {valid_solidified_count} valid solidified cases based on Phi_global")
    print(f"→ Found {completed_count} cases with status '10 Completed (solidified)' ")
    
    if valid_solidified_count != completed_count:
        print("WARNING: The number of valid solidified planets does not match the number of planets with status: '10 Completed (solidified)'")
        print("\nChecking final Phi_global values for all status '10 Completed (solidified)' cases:")
        for i, case in enumerate(status_10_cases):
            df = case['output_values']
            if 'Phi_global' in df.columns:
                final_phi = df['Phi_global'].iloc[-1]
                #print(f"[Status Case {i}] Final Phi_global = {final_phi}")
            else:
                print(f"[Status Case {i}] Phi_global column missing.")
    else:
        print("Solidified planets count matches the number of planets with status: '10 Completed (solidified)'.")
    print('-----------------------------------------------------------')

    return solidification_times

def save_grid_data_to_csv(grid_name, cases_data, grid_parameters, extracted_value, output_to_extract, phi_crit, output_dir: Path):
    """
    Save all simulation information (status, grid parameters, output values, solidification times)
    into CSV files for later analysis (doing plots).

    Parameters
    ----------
    grid_name : str
        Name of the grid.

    cases_data : list
        List of dictionaries containing the status of all the simulations cases in the grid.

    grid_parameters : dict
        Dictionary containing the grid parameters.

     extracted_value : dict
        Dictionary containing the extracted output values for each parameter.

    output_to_extract : list
        List of output values extracted from each simulation in the grid.
    
    phi_crit : float
        Critical melt fraction value to determine if the planet solidifies.

    output_dir : Path
        Directory where the CSV file will be saved.
    """
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the header
    header = [
        "#############################################################################################################",
        f"Grid name:                {grid_name}",
        f"Total number of cases:    {len(cases_data)}",
        f"phi_crit:                 {phi_crit}",
        "----------------------------------------------------------",
        " Grid Parameters",
        "----------------------------------------------------------"
    ]
    
    max_label_length = max(len(param) for param in grid_parameters.keys())  
    for param, values in grid_parameters.items():
        aligned_param = f"{param: <{max_label_length}}"  
        values_str = f"[{', '.join(map(str, values))}]"  
        header.append(f"{aligned_param}: {values_str}")
    
    header.extend([
        "----------------------------------------------------------",
        "This file contains the following columns:",
        f"| Case number | {' | '.join(extracted_value.keys())} |",
        "#############################################################################################################"
    ])

    # Prepare the CSV file path
    csv_file = output_dir / f"{grid_name}_simulation_data.csv"

    # Define the column headers for the CSV file
    csv_headers = ["Case number"] + list(extracted_value.keys())

    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header as comments at the top of the file
        for line in header:
            writer.writerow([f"#{line}"])  # Add a '#' to make it a comment in the CSV file
        
        # Write the header for the data section
        writer.writerow(csv_headers)
        
        # Write the data for each case
        for case_index, case_data in enumerate(cases_data):
            row = [case_index + 1]  # Start with the case number
            for param in extracted_value.keys():
                row.append(extracted_value[param][case_index])  # Extract the value for the current case
            writer.writerow(row)

    print(f"Data has been successfully saved to {csv_file}.")

if __name__ == '__main__':

    # Paths to the grid and the plot folder
    grid_name = 'escape_grid_4_params_Pxuv_a_epsilon_fO2'
    grid_path = '/home2/p315557/outputs_Norma2/good_grids/' + grid_name + '/'
    plots_path = '/home2/p315557/PROTEUS/tools/post_processing_grid/plots/' + grid_name + '/'
    data_dir = '/home2/p315557/PROTEUS/tools/post_processing_grid/processed_data/' + grid_name + '/'

    # User choose the parameters to post-process the grid
    plot=True                                                                   # True or False
    output_to_extract = ['esc_rate_total',                                      # List of output values to extract from the runtime_helpfile
                        'Phi_global', 
                        'P_surf',   
                        'atm_kg_per_mol']
    phi_crit = 0.005                                                            # Critical melt fraction for solidification 
    extracted_value = {}

    # Post-processing the grid  
    cases_data = load_grid_cases(grid_path)                                     # Load all simulation cases
    if plot:
        plot_grid_status(cases_data, plots_path)                                # Plot the summary histogram of the grid status
    grid_parameters = get_grid_parameters(grid_path)                            # Extract grid parameters
    for param in output_to_extract:
        extracted_value[param] = extract_grid_output(cases_data, param)        # Extract output values
    solidification_times = extract_solidification_time(cases_data, phi_crit)    # Extract the solidification time
    extracted_value['solidification_time'] = solidification_times
    
    # Save all the extracted data to a CSV file 
    save_grid_data_to_csv(grid_name, cases_data, grid_parameters, extracted_value, solidification_times, phi_crit, data_dir)
