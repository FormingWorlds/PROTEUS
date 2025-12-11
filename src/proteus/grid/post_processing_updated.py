from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import toml


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

        # Combine all info about simulations into a list of dictionaries
        combined_data.append({
            'init_parameters': init_params,
            'output_values'  : df,
            'status'         : status
        })

    #
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

def get_grid_parameters_from_toml(toml_path: str):
    """
    Extract grid parameter names and values from a PROTEUS ensemble TOML file.

    Parameters
    ----------
    toml_path : str
        Path to the ensemble TOML file (e.g. input/ensembles/blabla.toml)

    Returns
    -------
    param_grid : dict
        Dictionary where each key is a parameter name and the value is a list
        of all grid values for that parameter.
    """

    if not os.path.exists(toml_path):
        print(f"Error: TOML file not found at {toml_path}")
        return {}

    with open(toml_path, "r") as f:
        lines = f.readlines()

    param_grid = {}

    # Regex patterns
    section_pattern = re.compile(r'^\s*\["(.+?)"\]\s*$')
    method_pattern = re.compile(r'^\s*method\s*=\s*"(.+?)"\s*$')
    values_pattern = re.compile(r'^\s*values\s*=\s*(\[.*\])\s*$')
    start_pattern  = re.compile(r'^\s*start\s*=\s*([0-9.eE+-]+)\s*$')
    stop_pattern   = re.compile(r'^\s*stop\s*=\s*([0-9.eE+-]+)\s*$')
    count_pattern  = re.compile(r'^\s*count\s*=\s*(\d+)\s*$')

    current_param = None
    current_method = None
    start = stop = count = None

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Detect new parameter block
        section_match = section_pattern.match(line)
        if section_match:
            current_param = section_match.group(1)
            current_method = None
            start = stop = count = None
            continue

        if current_param is None:
            continue

        # Detect method
        method_match = method_pattern.match(line)
        if method_match:
            current_method = method_match.group(1)
            continue

        # Direct method values
        values_match = values_pattern.match(line)
        if values_match and current_method == "direct":
            try:
                values = ast.literal_eval(values_match.group(1))
                param_grid[current_param] = values
            except Exception as e:
                print(f"Error parsing values for {current_param}: {e}")
            continue

        # Logspace parameters
        start_match = start_pattern.match(line)
        if start_match:
            start = float(start_match.group(1))
            continue

        stop_match = stop_pattern.match(line)
        if stop_match:
            stop = float(stop_match.group(1))
            continue

        count_match = count_pattern.match(line)
        if count_match:
            count = int(count_match.group(1))

            # Once all three exist, generate logspace
            if current_method == "logspace" and start is not None and stop is not None:
                values = np.logspace(np.log10(start), np.log10(stop), count).tolist()
                param_grid[current_param] = values

    return param_grid

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

    for case_index, case in enumerate(cases_data):
        df = case['output_values']
        if df is None:
            print(f"Warning: No output values found for case number '{case_index}'")
            parameter_values.append(np.nan)  # Append NaN if no output values
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

def extract_solidification_time(cases_data: list):
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
            solidification_times.append(np.nan)  # Append NaN if no output values
            continue

        if 'Phi_global' in df.columns and 'Time' in df.columns:
            solidification_params = case['init_parameters'].get('params.stop.solid.phi_crit')
            if solidification_params is None or 'phi_crit' not in solidification_params:
                raise ValueError(f"Error: 'phi_crit' not found in init_parameters of case {i}. ")
            phi_crit = solidification_params['phi_crit']

            condition = df['Phi_global'] < phi_crit
            if condition.any():
                first_index = condition.idxmax()
                solid_time = df.loc[first_index, 'Time'] # Get the index of the time at which the condition is first satisfied
                solidification_times.append(solid_time)
            else:
                solidification_times.append(np.nan)  # Append NaN if condition is not satisfied
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
    valid_solidification_times = [time for time in solidification_times if not np.isnan(time) and time > 0.0]
    valid_solidified_count = len(valid_solidification_times)

    print('-----------------------------------------------------------')
    print(f"Extracted solidification times (Phi_global < {phi_crit})")
    print(f"→ Found {valid_solidified_count} valid solidified cases based on Phi_global")
    print(f"→ Found {completed_count} cases with status 'Completed (solidified)' ")
    # Check if the number of valid solidified cases matches the number of cases with status '10 Completed (solidified)' in the grid, to be sure the extraction is correct
    if valid_solidified_count != completed_count:
        print("WARNING: The number of valid solidified planets does not match the number of planets with status: 'Completed (solidified)'")
    else:
        print("Solidified planets count matches the number of planets with status: 'Completed (solidified)'.")
    print('-----------------------------------------------------------')

    return solidification_times
