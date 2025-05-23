from pathlib import Path
import pandas as pd
import toml
import os
import re
import ast
import numpy as np
import csv
from typing import Tuple, Dict, List, Any


def load_grid_cases(grid_dir: Path):
    """
    Load information for each simulation of a PROTEUS grid.
    Read 'runtime_helpfile.csv', 'init_coupler.toml' and status
    files for each simulation of the grid.

    Parameters
    ----------
        grid_dir : Path or str
            Path to the grid directory containing the 'case_*' folders

    Returns
    ----------
        combined_data : list
            List of dictionaries, each containing:
                - 'init_parameters' (dict): Parameters loaded from `init_coupler.toml`.
                - 'output_values' (pandas.DataFrame): Data from `runtime_helpfile.csv`.
                - 'status' (str): Status string from the `status` file, or 'Unknown' if unavailable.
    """

    combined_data = []
    grid_dir = Path(grid_dir)

    # Collect and sort the case directories
    case_dirs = list(grid_dir.glob('case_*'))
    case_dirs.sort(key=lambda p: int(p.name.split('_')[1]))

    for case in case_dirs:
        runtime_file = case / 'runtime_helpfile.csv'
        init_file = case / 'init_coupler.toml'
        status_file = case / 'status'

        # Load init parameters
        init_params = {}
        if init_file.exists():
            try:
                init_params = toml.load(open(init_file))
            except Exception as e:
                print(f"Error reading init file in {case.name}: {e}")

        # Read runtime_helpfile.csv if available
        df = None
        if runtime_file.exists():
            try:
                df = pd.read_csv(runtime_file, sep='\t')
            except Exception as e:
                print(f"WARNING : Error reading runtime_helpfile.csv for {case.name}: {e}")

        # Read status file if available
        status = 'Unknown'
        if status_file.exists():
            try:
                raw_lines = [ln.strip() for ln in status_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
                if len(raw_lines) >= 2:
                    status = raw_lines[1]
                elif raw_lines:
                    status = raw_lines[0]
                else:
                    status = 'Empty'
            except Exception as e:
                print(f"WARNING : Error reading status file in {case.name}: {e}")
        else:
            print(f"WARNING : Missing status file in {case.name}")

        # # THIS IS ONLY FOR MY CURRENT GRID ON HABROK 
        # if status in ('Unknown', 'Empty'):
        #     status = 'Disk quota exceeded'

        combined_data.append({
            'init_parameters': init_params,
            'output_values'  : df,
            'status'         : status
        })

    # --- summary printout ---
    statuses = [c['status'] for c in combined_data]
    status_counts = pd.Series(statuses).value_counts().sort_values(ascending=False)

    print('-----------------------------------------------------------')
    print(f"Total number of simulations: {len(statuses)}")
    print('-----------------------------------------------------------')
    print("Number of simulations per status:")
    for st, count in status_counts.items():
        print(f"  - {st:<45} : {count}")
    print('-----------------------------------------------------------')

    return combined_data

    return combined_data

def get_grid_parameters(grid_dir: str):
    """
    Extract grid parameter names and values from the 'manager.log' file.

    Parameters
    ----------
    grid_dir : str
        Path to the directory of the PROTEUS grid

    Returns
    -------
    param_grid : dict
        A dictionary where each key is a parameter name, and its corresponding values used for the entire grid is a list.

    case_params : dict
        Dictionary containing each case number with the name and values of the tested parameters in this grid.
    """

    log_file = os.path.join(grid_dir, 'manager.log')

    # Check if the 'manager.log' file exists
    if not os.path.exists(log_file):
        print(f"Error: manager.log not found at {log_file}")
        return {}, {}

    # Read all lines from the 'manager.log' file
    with open(log_file, 'r') as file:
        lines = file.readlines()

    param_grid = {}
    case_params = {}

    # Expressions to match the relevant lines
    dimension_pattern = re.compile(r"parameter:\s*(\S+)")
    values_pattern = re.compile(r"values\s*:\s*\[(.*?)\]")
    case_line_pattern = re.compile(r"\]\s+(\d+):\s+(\{.*\})")

    current_param = None
    for line in lines:
        line = line.strip()
        # Check if the line defines a new parameter
        dim_match = dimension_pattern.search(line)
        if dim_match:
            current_param = dim_match.group(1)
            continue
        # Check if the line defines values for the current parameter
        val_match = values_pattern.search(line)
        if val_match and current_param:
            try:
                val_list = ast.literal_eval(f"[{val_match.group(1)}]")
                param_grid[current_param] = val_list
                current_param = None
            except Exception as e:
                print(f"Error parsing values for {current_param}: {e}")
        # Check if the line contains case-specific data
        case_match = case_line_pattern.search(line)
        if case_match:
            case_num = int(case_match.group(1))
            case_dict_str = case_match.group(2)
            try:
                case_params[case_num] = ast.literal_eval(case_dict_str)
            except Exception as e:
                print(f"Error parsing case {case_num}: {e}")

    return param_grid, case_params

def extract_grid_output(cases_data: list, parameter_name: str):
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
        A list containing the extracted values of the specified parameter for all cases of the grid.
    """

    parameter_values = []
    columns_printed = False  # Flag to print columns only once

    for case in cases_data:
        df = case['output_values']
        if df is None:
            continue  # Skip cases with no output
        if parameter_name in df.columns:
            parameter_value = df[parameter_name].iloc[-1]
            parameter_values.append(parameter_value)
        else:
            if not columns_printed:
                print(f"Warning: Parameter '{parameter_name}' does not exist in case '{case['init_parameters'].get('name', 'Unknown')}'")
                print(f"Available columns in this case: {', '.join(df.columns)}")
                columns_printed = True

    # Print the extracted output values for the specified parameter
    print(f"Extracted output (at last time step) : {parameter_name} ")

    return parameter_values

def extract_solidification_time(cases_data: list, phi_crit: float):
    """
    Extract the solidification time at the time step where the condition
    'Phi_global' < phi_crit is first satisfied for each planet.

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing simulation data.

    phi_crit : float
        The critical melt fraction value below which a planet is considered solidified.
        A typical value is 0.005.

    Returns
    -------
    solidification_times : list
        A list containing the solidification times for all solidified planets of the grid.
        If a planet never solidifies, it will have a NaN in the list.
    """

    solidification_times = []
    columns_printed = False

    for i, case in enumerate(cases_data):
        df = case['output_values']
        # Check if the required columns exist in the dataframe
        if df is None:
            solidification_times.append(0.0)
            continue

        if 'Phi_global' in df.columns and 'Time' in df.columns:
            condition = df['Phi_global'] < phi_crit
            if condition.any():
                first_index = condition.idxmax()
                solid_time = df.loc[first_index, 'Time'] # Get the index of the time at which the condition is first satisfied
                solidification_times.append(solid_time)
            else:
                solidification_times.append(0.0)  # Append NaN if condition is not satisfied
        else:
            if not columns_printed:
                print("Warning: 'Phi_global' and/or 'Time' columns not found in some cases.")
                print(f"Available columns: {', '.join(df.columns)}")
                columns_printed = True
            solidification_times.append(np.nan)  # Append NaN if columns are missing

    # Count the number of cases with a status = '10 Completed (solidified)'
    status_10_cases = [case for case in cases_data if (case.get('status') or '').strip() == 'Completed (solidified)']
    completed_count = len(status_10_cases)
    # Count only valid solidification times (non-NaN)
    valid_solidification_times = [time for time in solidification_times if not np.isnan(time)]
    valid_solidified_count = len(valid_solidification_times)

    print('-----------------------------------------------------------')
    print(f"Extracted solidification times (Phi_global < {phi_crit})")
    print(f"→ Found {valid_solidified_count} valid solidified cases based on Phi_global")
    print(f"→ Found {completed_count} cases with status 'Completed (solidified)' ")
    # Check if the number of valid solidified cases matches the number of cases with status '10 Completed (solidified)' in the grid, to be sure the extraction is correct
    if valid_solidified_count != completed_count:
        print("WARNING: The number of valid solidified planets does not match the number of planets with status: '10 Completed (solidified)'")
        # To debug, the user can uncomment the following lines to print the solidification times for all plaent with '10 Completed (solidified)'
        # print("\nChecking final Phi_global values for all status '10 Completed (solidified)' cases:")
        # for i, case in enumerate(status_10_cases):
        #     df = case['output_values']
        #     if 'Phi_global' in df.columns:
        #         final_phi = df['Phi_global'].iloc[-1]
        #         print(f"[Status Case {i}] Final Phi_global = {final_phi}")
        #     else:
        #         print(f"[Status Case {i}] Phi_global column missing.")
    else:
        print("Solidified planets count matches the number of planets with status: 'Completed (solidified)'.")
    print('-----------------------------------------------------------')

    return solidification_times

def save_grid_data_to_csv(grid_name: str, cases_data: list, grid_parameters: dict, case_params: Dict[int, Dict[str, Any]],
                          extracted_value: dict, output_to_extract: list, phi_crit: float, output_dir: Path):
    """
    Save all simulation information (status, grid parameters, output values) into a CSV file
    for later analysis (using plot_grid.py to make plots for instance).

    Parameters
    ----------
    grid_name : str
        Name of the grid.

    cases_data : list
        List of dictionaries containing simulation data.

    grid_parameters : dict
        A dictionary where each key is a parameter name, and its corresponding values used for the entire grid is a list.

    case_params : dict
        Dictionary containing each case number with the name and values of the tested parameters in this grid.

     extracted_value : dict
        A list containing the extracted values of the specified parameter for all cases of the grid.

    output_to_extract : list
        List of output values extracted from each simulation in the grid.

    phi_crit : float
        The critical melt fraction value used to determine if a planet is considered solidified.
        A typical value is 0.005.

    output_dir : Path
        The directory where the generated CSV file will be saved. If the directory does not exist,
        it will be created.
    """
    # Check if the output directory exist, if not create it
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

        # Write data rows
        for case_index, case_data in enumerate(cases_data):
            status = case_data.get('status', 'Unknown') or 'Unknown'
            row = [case_index, f"'{status}'"]
            # Add grid parameters values for each case
            case_param_values = case_params.get(case_index, {})
            for param in grid_parameters.keys():
                row.append(case_param_values.get(param, 'NA'))
            # Add extracted output values for each case
            for param in extracted_value.keys():
                value_list = extracted_value.get(param, [])
                row.append(value_list[case_index] if case_index < len(value_list) else 'NA')
            # Write the row to the CSV file
            writer.writerow(row)

    print(f"Extracted data has been successfully saved to {csv_file}.")
    print('-----------------------------------------------------------')

if __name__ == '__main__':

    # User needs to specify paths
    grid_name   = 'escape_grid_habrok_7_params_1Msun'
    grid_path   = f'/home2/p315557/PROTEUS/output/scratch/{grid_name}/'
    data_dir    = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_processed_data/{grid_name}/'

    # User choose the parameters to post-process the grid
    output_to_extract = ['esc_rate_total','Phi_global', 'P_surf','atm_kg_per_mol']      # Output columns to extract from 'runtime_helpfile.csv' of each case
    phi_crit = 0.005                                                                    # Critical melt fraction for the solidification condition

    # Post-processing the grid
    extracted_value = {}                                                                # Initialize the dictionary to store extracted values
    cases_data = load_grid_cases(grid_path)                                             # Load all simulation cases
    grid_parameters, case_init_param = get_grid_parameters(grid_path)                   # Extract grid parameters
    for param in output_to_extract:
        extracted_value[param] = extract_grid_output(cases_data, param)                 # Extract output values
    solidification_times = extract_solidification_time(cases_data, phi_crit)            # Extract the solidification time
    extracted_value['solidification_time'] = solidification_times                       # Add solidification time to the extracted_values
    save_grid_data_to_csv(grid_name, cases_data, grid_parameters, case_init_param,
                      extracted_value, solidification_times, phi_crit, data_dir)        # Save all the extracted data to a CSV file

    # Done with the post-processing step :)
    print("Post-processing completed. Let's do some plots !")
    print('(Please check for any WARNING messages above before going further.)')
    print('-----------------------------------------------------------')
