import os
from pathlib import Path
import pandas as pd
import toml
import seaborn as sns
import matplotlib.pyplot as plt
import re

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
    output_path = plot_dir+'simulations_status_summary.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

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
    print('-----------------------------------------------------------')
    print(f"Extracted output (at last time step) : {parameter_name} ")
    print('-----------------------------------------------------------')

    return parameter_values

def extract_solidification_time(cases_data, phi_crit):
    """
    Extract the solidification time for planet that reach Phi_global < phi_crit.

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
        If a planet never solidifies, it will not be included in the list.
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
                print(f"[Case {i}] Solidification at index {first_index} → Time = {solid_time}, Phi = {df.loc[first_index, 'Phi_global']}")

        else:
            if not columns_printed:
                print("Warning: 'Phi_global' and/or 'Time' columns not found in some cases.")
                print(f"Available columns: {', '.join(df.columns)}")
                columns_printed = True

    print('-----------------------------------------------------------')
    print(f"Extracted solidification times ")
    print('at timesteps when Phi_global < {phi_crit}')
    print('-----------------------------------------------------------')
    print(len(solidification_times))
    
    return solidification_times



if __name__ == '__main__':

    # Paths
    grid_path = '/home2/p315557/outputs_Norma2/good_grids/escape_grid_4_params_Pxuv_a_epsilon_fO2/'
    plots_path = '/home2/p315557/PROTEUS/post_processing_grids/plots/escape_grid_4_params_Pxuv_a_epsilon_fO2/'

    # Plots : True or False
    plot=True

    # Load simulation cases
    cases_data = load_grid_cases(grid_path)

    # Plot = True or False, if True, it will plot the status of all grid simulations
    if plot:
        plot_grid_status(cases_data, plots_path)

    # Extract grid parameters
    grid_parameters = get_grid_parameters(grid_path)

    # List of output to extract
    output_to_extract = ['esc_rate_total', 'Phi_global', 'P_surf', 'atm_kg_per_mol']
    for param in output_to_extract:
        extracted_values = extract_grid_output(cases_data, param)
    # Extract the solidification time
    phi_crit = 0.005 # Critical melt fraction for solidification
    solidification_times = extract_solidification_time(cases_data, phi_crit)