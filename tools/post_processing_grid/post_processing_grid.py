import os
from pathlib import Path
import pandas as pd
import toml
import re
import numpy as np
import csv
import ast
from typing import Tuple, Dict, List, Any


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
    log_file = os.path.join(grid_dir, 'manager.log')

    if not os.path.exists(log_file):
        print(f"Error: manager.log not found at {log_file}")
        return {}, {}

    with open(log_file, 'r') as file:
        lines = file.readlines()

    param_grid = {}
    case_params = {}

    dimension_pattern = re.compile(r"parameter:\s*(\S+)")
    values_pattern = re.compile(r"values\s*:\s*\[(.*?)\]")
    case_line_pattern = re.compile(r"\]\s+(\d+):\s+(\{.*\})")  # Fixed pattern

    current_param = None

    for line in lines:
        line = line.strip()

        dim_match = dimension_pattern.search(line)
        if dim_match:
            current_param = dim_match.group(1)
            continue

        val_match = values_pattern.search(line)
        if val_match and current_param:
            try:
                val_list = ast.literal_eval(f"[{val_match.group(1)}]")
                param_grid[current_param] = val_list
                current_param = None
            except Exception as e:
                print(f"Error parsing values for {current_param}: {e}")

        case_match = case_line_pattern.search(line)
        if case_match:
            case_num = int(case_match.group(1))
            case_dict_str = case_match.group(2)
            try:
                case_params[case_num] = ast.literal_eval(case_dict_str)
            except Exception as e:
                print(f"Error parsing case {case_num}: {e}")

    return param_grid, case_params

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

def save_grid_data_to_csv(grid_name, cases_data, grid_parameters, case_params, extracted_value, output_to_extract, phi_crit, output_dir: Path):
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

    case_params :  Dict[int, Dict[str, Any]]
        Dictionary containing each case number with the name and values of the tested parameters in this grid.

     extracted_value : dict
        Dictionary containing the extracted output values for each parameter.

    output_to_extract : list
        List of output values extracted from each simulation in the grid.
    
    phi_crit : float
        Critical melt fraction value to determine if the planet solidifies.

    output_dir : Path
        Directory where the CSV file will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV file path
    csv_file = output_dir / f"{grid_name}_extracted_data.csv"

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header block
        writer.writerow(["#############################################################################################################"])
        writer.writerow([f"Grid name:                {grid_name}"])
        writer.writerow([f"Total number of cases:    {len(cases_data)}"])
        writer.writerow([f"Dimension of the grid:    {len(grid_parameters)}"])
        writer.writerow([f"phi_crit:                 {phi_crit}"])
        writer.writerow(["----------------------------------------------------------"])
        writer.writerow([" Grid Parameters"])
        writer.writerow(["----------------------------------------------------------"])
        max_label_length = max(len(param) for param in grid_parameters.keys())
        for param, values in grid_parameters.items():
            aligned_param = f"{param: <{max_label_length}}"
            values_str = f"[{', '.join(map(str, values))}]"
            writer.writerow([f"{aligned_param}: {values_str}"])
        writer.writerow(["----------------------------------------------------------"])
        writer.writerow(["Extracted output values:" f"[{', '.join(extracted_value.keys())}]"])    
        writer.writerow(["----------------------------------------------------------"])
        writer.writerow([f"| Case number | Status | {' | '.join(grid_parameters.keys())} | {' | '.join(extracted_value.keys())} |"])
        writer.writerow(["#############################################################################################################"])
        writer.writerow([])

        # CSV table header
        writer.writerow(["Case number", "Status"] + list(grid_parameters.keys()) + list(extracted_value.keys()))

        # Data rows
        for case_index, case_data in enumerate(cases_data):
            status = case_data.get('status', 'unknown') or 'unknown'
            row = [case_index, f"'{status}'"]

            # Use case_params (from get_grid_parameters) to pull values for each param
            case_param_values = case_params.get(case_index, {})

            for param in grid_parameters.keys():
                row.append(case_param_values.get(param, 'NA'))

            # Add extracted output values
            for param in extracted_value.keys():
                value_list = extracted_value.get(param, [])
                row.append(value_list[case_index] if case_index < len(value_list) else 'NA')

            writer.writerow(row)

    print(f"Extracted data has been successfully saved to {csv_file}.")
    print('-----------------------------------------------------------')

if __name__ == '__main__':

    # Paths to the grid folder
    grid_name = 'escape_grid_4_params_Pxuv_a_epsilon_fO2'
    grid_path = f'/home2/p315557/outputs_Norma2/good_grids/{grid_name}/'
    data_dir = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_processed_data/{grid_name}/'
    # User choose the parameters to post-process the grid
    output_to_extract = ['esc_rate_total',                                      # List of output values to extract from the runtime_helpfile
                        'Phi_global', 
                        'P_surf',   
                        'atm_kg_per_mol']
    phi_crit = 0.005                                                            # Critical melt fraction for solidification 
    extracted_value = {}

    # Post-processing the grid  
    cases_data = load_grid_cases(grid_path)                                     # Load all simulation cases
    grid_parameters, case_init_param = get_grid_parameters(grid_path)           # Extract grid parameters
    for param in output_to_extract:
        extracted_value[param] = extract_grid_output(cases_data, param)         # Extract output values
    solidification_times = extract_solidification_time(cases_data, phi_crit)    # Extract the solidification time
    extracted_value['solidification_time'] = solidification_times
    
    # Save all the extracted data to a CSV file 
    save_grid_data_to_csv(grid_name, cases_data, grid_parameters, case_init_param, extracted_value, solidification_times, phi_crit, data_dir)

    print('-----------------------------------------------------------')
    print("Post-processing completed. Let's do some plots !")
    print('(Please check for any warning messages above before going further.)')
    print('-----------------------------------------------------------')    