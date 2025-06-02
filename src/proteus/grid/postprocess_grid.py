import toml
import re
import ast
import csv
from typing import Tuple, Dict, List, Any
from pathlib import Path
import os
import pandas as pd
import numpy as np
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

##### Functions for extracting grid data #####

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

def extract_solidification_time(cases_data: list, phi_crit: float = 0.005):
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
    valid_solidification_times = [time for time in solidification_times if not np.isnan(time) and time > 0.0]
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

def extract_CHNOS_totals(cases_data: List[dict]) -> List[pd.Series]:
    """
    For each case in `cases_data`, compute the row-wise total of H, C, N, O, S.

    Parameters
    ----------
    cases_data : List[dict]
        Each dict must have an 'output_values' key holding a pandas DataFrame.

    Returns
    -------
    totals_list : List[pd.Series]
        One Series per case, indexed as in the original DataFrame, containing
        the sum H+C+N+O+S at each row. Cases missing any required column are skipped.
    """
    # Columns to sum
    _cols = ['H_kg_total', 'C_kg_total', 'N_kg_total', 'O_kg_total', 'S_kg_total']

    totals_list: List[pd.Series] = []
    warned = False

    for case in cases_data:
        df = case.get('output_values')
        name = case.get('init_parameters', {}).get('name', '<unnamed>')
        if df is None:
            continue

        missing = [c for c in _cols if c not in df.columns]
        if missing:
            if not warned:
                print(f"Warning: missing columns {missing!r} in case '{name}'. "
                      f"Available: {df.columns.tolist()!r}")
                warned = True
            continue

        # Compute row-wise sum of the five elements
        total = df[_cols].sum(axis=1)
        total.name = f"{name}_CHNOS_total"
        totals_list.append(total)
    
    print(f"Extracted CHNOS totals for {len(totals_list)} cases.")
    print('-----------------------------------------------------------')
    return totals_list

def save_grid_data_to_csv(grid_name: str, cases_data: list, grid_parameters: dict, case_params: Dict[int, Dict[str, Any]],
                          extracted_value: dict, output_to_extract: list, output_dir: Path, phi_crit: float = 0.005):
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

    output_dir : Path
        The directory where the generated CSV file will be saved. If the directory does not exist,
        it will be created.

    phi_crit : float
        The critical melt fraction value used to determine if a planet is considered solidified.
        A typical value is 0.005.
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


##### Functions for plotting grid data results #####

def load_extracted_data(data_path : str | Path, grid_name :str):

    """
    Load extracted data from the CSV file generated with post_processing.py, returning a DataFrame for plotting.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing the CSV file with the extracted data.
    grid_name : str
        Name of the grid

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the extracted simulation data.

    grid_params : dict
        Dictionary of grid parameter names and their value lists.

    extracted_outputs : list of str
        List of extracted output variable names.
    """

    csv_file = os.path.join(data_path, f"{grid_name}_extracted_data.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at: {csv_file}")

    with open(csv_file, 'r') as f:
        lines = f.readlines()

    data_start_idx = None
    grid_params = {}
    extracted_outputs = []
    grid_dimension = None

    # Get grid dimension
    for i, line in enumerate(lines):
        if line.strip().startswith("Dimension of the grid"):
            grid_dimension = int(line.split(":")[-1].strip())
            break

    if grid_dimension is None:
        raise ValueError("Could not find 'Dimension of the grid' in the CSV file.")

    # Extract grid parameters
    for i, line in enumerate(lines):
        if "Grid Parameters" in line:
            grid_param_start_idx = i + 2  # Skip divider after header
            break
    else:
        raise ValueError("Could not find 'Grid Parameters' section in the CSV file.")

    for line in lines[grid_param_start_idx:]:
        line = line.strip().strip('"')  # Remove quotes and whitespace

        # Stop at the next divider or output section
        if line.startswith("----------------------------------------------------------") or "Extracted output values" in line:
            break

        if ':' in line:
            param_name, param_values = line.split(":", 1)
            param_name = param_name.strip()
            param_values = param_values.strip().strip("[]").split(",")
            param_values = [val.strip() for val in param_values]
            param_values = [float(val) if is_float(val) else val for val in param_values]
            grid_params[param_name] = param_values

    # Check dimensions
    if len(grid_params) != grid_dimension:
        raise ValueError(f"Mismatch: Expected {grid_dimension} grid parameters, found {len(grid_params)}.")

    # Extract output names
    for line in lines:
        line = line.strip().strip('"')  # Remove quotes and whitespace

        if line.startswith("Extracted output values"):
            if ":" in line:
                _, outputs_part = line.split(":", 1)
                outputs_part = outputs_part.strip().strip("[]")
                extracted_outputs = [s.strip() for s in outputs_part.split(",")]
            break

    # Find start of actual data
    for i, line in enumerate(lines):
        if line.strip().startswith("Case number"):
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError("Could not find CSV header line starting with 'Case number'.")

    # Extract the actual data section and read it
    data_section = ''.join(lines[data_start_idx:])
    df = pd.read_csv(StringIO(data_section))

    return df, grid_params, extracted_outputs

def is_float(value):
    """Helper function to check if a string can be converted to a float."""
    try:
        float(value)   # Try converting string to float
        return True
    except ValueError:
        return False   # Fail not a valid float

def plot_dir_exists(plot_dir: Path):

    """
    Check if the plot directory exists. If not, create it.

    Parameters
    ----------
    plot_dir : Path
        Path object pointing to the desired plot directory.
    """
    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created plot directory: {plot_dir}")

def group_output_by_parameter(df,grid_parameters,outputs):
    """
    Groups output values (like solidification times) by a specific grid parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results including value of the grid parameter and the corresponding extracted output.

    grid_parameters : str
        Column name of the grid parameter to group by (like 'escape.zephyrus.Pxuv').

    outputs : str
        Column name of the output to extract (like 'solidification_time').

    Returns
    -------
    dict
        Dictionary where each key is of the form '[output]_per_[parameter]', and each value is a dict {param_value: [output_values]}.
    """
    grouped = {}

    for param in grid_parameters:
        for output in outputs:
            key_name = f"{output}_per_{param}"
            value_dict = {}
            for param_value in df[param].dropna().unique():
                subset = df[df[param] == param_value]
                output_values = subset[output].replace([np.inf, -np.inf], np.nan)
                output_values = output_values.dropna()
                output_values = output_values[output_values > 0]  # Remove zeros and negatives

                value_dict[param_value] = output_values

            grouped[key_name] = value_dict

    return grouped

def plot_grid_status(cases_data, plot_dir: Path, grid_name: str, status_colors: dict = None):
    """
    Plot the status of simulations from the PROTEUS grid with improved x-axis readability.

    Parameters
    ----------
    cases_data : list or DataFrame
        Contains the status of all simulations from the grid.

    plot_dir : Path
        Path to the plots directory.

    grid_name : str
        Name of the grid, used for the plot title.

    status_colors : dict, optional
        A dictionary mapping statuses to specific colors. If None, a default palette is used.
    """

    # Extract and clean statuses
    statuses = cases_data['Status'].fillna('unknown').astype(str)
    status_counts = statuses.value_counts().sort_values(ascending=False)

    # Set colors for the bars
    if status_colors:
        formatted_status_keys = [s.replace(" ", "\n") for s in status_counts.index]
        palette = {formatted: status_colors.get(original, 'gray')
                for formatted, original in zip(formatted_status_keys, status_counts.index)}
    else:
        palette = sns.color_palette("Accent", len(status_counts))
        formatted_status_keys = [s.replace("d (", "d \n (") for s in status_counts.index]
        palette = dict(zip(formatted_status_keys, palette))

    # Prepare dataframe for plotting
    plot_df = pd.DataFrame({
        'Status': formatted_status_keys,
        'Count': status_counts.values
    })

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        data=plot_df,
        x='Status',
        y='Count',
        hue='Status',
        palette=palette,
        dodge=False,
        edgecolor='black'
    )

    # Remove legend if it was created
    if ax.legend_:
        ax.legend_.remove()

    # Add value labels above bars
    total_simulations = len(cases_data)
    for i, count in enumerate(status_counts.values):
        percentage = (count / total_simulations) * 100
        ax.text(
            i, count + 1,
            f"{count} ({percentage:.1f}%)",
            ha='center', va='bottom', fontsize=10
        )

    # Boxed total in upper right
    plt.gca().text(
        0.97, 0.94,
        f"Total number of simulations : {total_simulations}",
        transform=plt.gca().transAxes,
        ha='right', va='top',
        fontsize=14,
        #bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black")
    )

    plt.grid(alpha=0.2, axis='y')
    plt.title(f"Grid statuses summary : {grid_name}", fontsize=16)
    plt.xlabel("Simulation statuses", fontsize=16)
    plt.ylabel("Number of simulations", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    output_path = plot_dir + 'grid_statuses_summary.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Summary plot of grid statuses is available")

def ecdf_single_plots(grid_params: dict, grouped_data: dict, param_settings: dict, output_settings: dict, plots_path: str):
    """
    Generates and saves one ECDF plot per combination of output and input parameter.

    Parameters
    ----------

    grid_params : dict
        A mapping from parameter names (e.g. "orbit.semimajoraxis") to arrays/lists of tested values.

    grouped_data : dict
        Dictionary where each key is of the form '[output]_per_[parameter]', and each value is a dict {param_value: [output_values]}.

    param_settings : dict
        For each input-parameter key, a dict containing:
            - "label": label of the colormap for the corresponding input parameter
            - "colormap": a matplotlib colormap (e.g. mpl.cm.plasma)
            - "log_scale": bool, whether to color-normalize on a log scale

    output_settings : dict
        For each output key, a dict containing:
            - "label": label of the x-axis for the corresponding output quantity
            - "log_scale": bool, whether to plot the x-axis on log scale
            - "scale":    float, a factor to multiply raw values by before plotting

    plots_path : str
        Path to the grid where to create "single_plots_ecdf" and save all .png plots
    """
    # Create output directory if not already there
    output_dir = os.path.join(plots_path, "single_plots_ecdf")
    os.makedirs(output_dir, exist_ok=True)

    for output_name, out_settings in output_settings.items():
        for param_name, settings in param_settings.items():
            tested_param = grid_params.get(param_name, [])
            if len(tested_param) <= 1:
                # Skip if only a single value was tested
                continue

            # Plot settings for this input parameter
            param_label = settings["label"]
            cmap      = settings["colormap"]
            color_log = settings.get("log_scale", False)

            # Plot settings for this output
            x_label  = out_settings["label"]
            x_log    = out_settings.get("log_scale", False)
            scale    = out_settings.get("scale", 1.0)

            # Determine if the parameter array is numeric
            is_numeric = np.issubdtype(np.array(tested_param).dtype, np.number)

            if is_numeric:
                # Continuous colormap: Normalize either linearly or in log-space
                if color_log:
                    norm = mpl.colors.LogNorm(vmin=min(tested_param), vmax=max(tested_param))
                else:
                    norm = mpl.colors.Normalize(vmin=min(tested_param), vmax=max(tested_param))
                color_func = lambda v: cmap(norm(v))
                colorbar_needed = True
            else:
                # Categorical colormap: map each unique value to one color
                unique_vals = sorted(set(tested_param))
                cats_cmap   = mpl.colormaps.get_cmap(cmap.name).resampled(len(unique_vals))
                color_map   = {val: cats_cmap(i) for i, val in enumerate(unique_vals)}
                color_func  = lambda val: color_map[val]
                colorbar_needed = False

            # Create a new figure & axes
            fig, ax = plt.subplots(figsize=(10, 6))
            data_key = f"{output_name}_per_{param_name}"

            for val in tested_param:
                # Skip if no data for this value
                if val not in grouped_data.get(data_key, {}):
                    continue
                raw = np.array(grouped_data[data_key][val]) * scale
                sns.ecdfplot(
                    data=raw,
                    log_scale=x_log,
                    stat="proportion",
                    color=color_func(val),
                    linewidth=3,
                    ax=ax
                )

            # Axis formatting
            ax.set_xlabel(x_label, fontsize=14)
            ax.set_ylabel("Normalized cumulative fraction of simulations", fontsize=14)
            ax.grid(alpha=0.1)

            # Colorbar or legend
            if colorbar_needed:
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
                cbar.set_label(param_label, fontsize=14)
                # Set ticks at each tested parameter value
                ticks = sorted(set(tested_param))
                cbar.set_ticks(ticks)
            else:
                # Build a legend for categorical values
                unique_vals = sorted(set(tested_param))
                handles = [
                    mpl.lines.Line2D([0], [0], color=color_map[val], lw=3, label=str(val))
                    for val in unique_vals
                ]
                ax.legend(handles=handles, loc="lower right", title=param_label)

            # Save and close
            fname = f"ecdf_{output_name}_per_{param_name.replace('.', '_')}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()

    print(f"All single ECDF plots are available at {output_dir}")

def ecdf_grid_plot(grid_params: dict, grouped_data: dict, param_settings: dict, output_settings: dict, plots_path: str):
    """
    Creates a grid of ECDF plots where each row corresponds to one input parameter
    and each column corresponds to one output. Saves the resulting figure as a PNG.

    Parameters
    ----------

    grid_params : dict
        A mapping from parameter names (e.g. "orbit.semimajoraxis") to arrays/lists of tested values.

    grouped_data : dict
        Dictionary where each key is of the form '[output]_per_[parameter]', and each value is a dict {param_value: [output_values]}.

    param_settings : dict
        For each input-parameter key, a dict containing:
            - "label": label of the colormap for the corresponding input parameter
            - "colormap": a matplotlib colormap (e.g. mpl.cm.plasma)
            - "log_scale": bool, whether to color-normalize on a log scale

    output_settings : dict
        For each output key, a dict containing:
            - "label": label of the x-axis for the corresponding output quantity
            - "log_scale": bool, whether to plot the x-axis on log scale
            - "scale":    float, a factor to multiply raw values by before plotting

    plots_path : str
        Path to the grid where to create "single_plots_ecdf" and save all .png plots
    """
    # Ensure output directory exists
    os.makedirs(plots_path, exist_ok=True)

    # List of parameter names (rows) and output names (columns)
    param_names = list(param_settings.keys())
    out_names   = list(output_settings.keys())

    # Create subplot grid: rows = parameters, columns = outputs
    n_rows = len(param_names)
    n_cols = len(out_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)

    # Loop through parameters (rows) and outputs (columns)
    for i, param_name in enumerate(param_names):
        tested_param = grid_params.get(param_name, [])
        settings     = param_settings[param_name]
        # Determine coloring
        is_numeric = np.issubdtype(np.array(tested_param).dtype, np.number)
        if is_numeric:
            norm = mpl.colors.LogNorm(vmin=min(tested_param), vmax=max(tested_param)) if settings.get("log_scale", False) else mpl.colors.Normalize(vmin=min(tested_param), vmax=max(tested_param))
            color_func = lambda v: settings["colormap"](norm(v))
            colorbar_needed = True
        else:
            unique_vals = sorted(set(tested_param))
            cmap = mpl.colormaps.get_cmap(settings["colormap"]).resampled(len(unique_vals))
            color_map = {val: cmap(j) for j, val in enumerate(unique_vals)}
            color_func = lambda v: color_map[v]
            colorbar_needed = False

        for j, output_name in enumerate(out_names):
            ax = axes[i][j]
            out_settings = output_settings[output_name]

            # Plot one ECDF per tested parameter value
            for val in tested_param:
                data_key = f"{output_name}_per_{param_name}"
                if val not in grouped_data.get(data_key, {}):
                    continue
                raw = np.array(grouped_data[data_key][val]) * out_settings.get("scale", 1.0)
                sns.ecdfplot(
                    data=raw,
                    log_scale=out_settings.get("log_scale", False),
                    color=color_func(val),
                    linewidth=4,
                    ax=ax
                )

            # Configure x-axis labels, ticks, grids
            if i == n_rows - 1:
                ax.set_xlabel(out_settings["label"], fontsize=22)
                ax.xaxis.set_label_coords(0.5, -0.3)
                ax.tick_params(axis='x', labelsize=22)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            # Configure y-axis (shared label added later)
            if j == 0:
                ticks = [0.0, 0.5, 1.0]
                ax.set_yticks(ticks)
                ax.tick_params(axis='y', labelsize=22)
            else:
                ax.set_ylabel("")
                ax.set_yticks(ticks)
                ax.tick_params(axis='y', labelleft=False)
            
            ax.grid(alpha=0.4)

        # After plotting all outputs for this parameter (row), add colorbar or legend
        if colorbar_needed:
            sm = mpl.cm.ScalarMappable(cmap=settings["colormap"], norm=norm)
            cbar = fig.colorbar(sm, ax=axes[i, :], pad=0.08, aspect=10)
            cbar.set_label(settings["label"], fontsize=24)
            cbar.ax.yaxis.set_label_coords(5.5, 0.5)
            ticks = sorted(set(tested_param))
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=22)
        else:
            handles = [mpl.lines.Line2D([0], [0], color=color_map[val], lw=4, label=str(val)) for val in unique_vals]
            ax.legend(handles=handles, fontsize=24,bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add a single, shared y-axis label
    fig.text(0.04, 0.5, 'Normalized cumulative fraction of simulations', va='center', rotation='vertical', fontsize=40)

    # Tweak layout and save
    plt.tight_layout(rect=[0.08, 0.02, 1, 0.97])
    filename = "ecdf_grid_plot.png"
    out_path = os.path.join(plots_path, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Grid ECDF plot saved at {out_path}")