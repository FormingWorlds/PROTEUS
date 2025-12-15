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

def get_grid_parameters_from_toml(grid_dir: str):
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

    grid_dir = Path(grid_dir)
    toml_path = grid_dir / "copy.grid.toml"

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

def load_phi_crit(grid_dir: str | Path):
    grid_dir = Path(grid_dir)
    ref_file = grid_dir / "ref_config.toml"

    if not ref_file.exists():
        raise FileNotFoundError(f"ref_config.toml not found in {grid_dir}")

    ref = toml.load(open(ref_file))

    # Navigate structure safely
    try:
        phi_crit = ref["params"]["stop"]["solid"]["phi_crit"]
    except KeyError:
        raise KeyError("phi_crit not found in ref_config.toml")

    return phi_crit

def extract_solidification_time(cases_data: list, grid_dir: str | Path):
    # Load phi_crit once
    phi_crit = load_phi_crit(grid_dir)

    solidification_times = []
    columns_printed = False

    for i, case in enumerate(cases_data):
        df = case['output_values']

        if df is None:
            solidification_times.append(np.nan)
            continue

        if 'Phi_global' in df.columns and 'Time' in df.columns:
            condition = df['Phi_global'] < phi_crit

            if condition.any():
                idx = condition.idxmax()
                solidification_times.append(df.loc[idx, 'Time'])
            else:
                solidification_times.append(np.nan)

        else:
            if not columns_printed:
                print("Warning: Missing Phi_global or Time column.")
                print("Columns available:", df.columns.tolist())
                columns_printed = True
            solidification_times.append(np.nan)

    return solidification_times


def export_simulation_summary(cases_data: list, grid_dir: str | Path, param_grid: dict, output_file: str | Path):
    """
    Export a summary of simulation cases to a CSV file with tested grid parameters only.

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing simulation data.
    grid_dir : str or Path
        Path to the grid folder containing `cfgs/case_XXXXXX.toml` files.
    param_grid : dict
        Dictionary of tested grid parameters (from get_grid_parameters_from_toml).
        Only these parameters will be included from each case.
    output_file : str or Path
        Path to the output CSV file.
    """
    grid_dir = Path(grid_dir)
    output_file = Path(output_file)

    summary_rows = []
    tested_param_names = list(param_grid.keys())

    for case_index, case in enumerate(cases_data):
        row = {}

        # Case number and status
        row['case_number'] = case_index
        row['status'] = case['status']

        # Load tested parameters from the case TOML
        case_toml = grid_dir / f"cfgs/case_{case_index:06d}.toml"
        if case_toml.exists():
            try:
                case_params = toml.load(open(case_toml))
                for param in tested_param_names:
                    row[param] = case_params.get(param, None)
            except Exception as e:
                print(f"Warning: Could not read {case_toml}: {e}")
        else:
            print(f"Warning: TOML file not found for case {case_index}: {case_toml}")

        # Output values (last time step)
        df = case['output_values']
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        summary_rows.append(row)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Save as tab-separated CSV
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"Simulation summary exported to {output_file}")
