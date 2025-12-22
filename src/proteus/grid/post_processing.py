from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
from matplotlib import cm

# ---------------------------------------------------------
# Data loading, extraction, and CSV generation functions
# ---------------------------------------------------------

def get_grid_name(grid_path: str | Path) -> str:
    """
    Returns the grid name (last part of the path) from the given grid path.

    Parameters
    ----------
    grid_path : str or Path
        Full path to the grid directory.

    Returns
    -------
    grid_name : str
        Name of the grid directory.
    """
    grid_path = Path(grid_path)
    if not grid_path.is_dir():
        raise ValueError(f"{grid_path} is not a valid directory")
    return grid_path.name

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
                - 'init_parameters' (dict): All input parameters loaded from `init_coupler.toml`.
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
                init_params = toml.load(init_file)
            except Exception as e:
                print(f"Error reading init file in {case.name}: {e}")

        # Read runtime_helpfile.csv
        df = None
        if runtime_file.exists():
            try:
                df = pd.read_csv(runtime_file, sep='\t')
            except Exception as e:
                print(f"WARNING : Error reading runtime_helpfile.csv for {case.name}: {e}")

        # Read status file
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

    # Print summary of statuses
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

def get_tested_grid_parameters(cases_data: list, grid_dir: str | Path):
    """
    Extract tested grid parameters per case using:
      - copy.grid.toml to determine which parameters were varied in the grid
      - init_parameters already loaded by load_grid_cases for each simulation of the grid

    Parameters
    ----------
    cases_data : list
        Output of load_grid_cases.
    grid_dir : str or Path
        Path to the grid directory containing copy.grid.toml.

    Returns
    -------
    case_params : dict
        Dictionary mapping case index -> {parameter_name: value}
    tested_params : dict
        Dictionary of tested grid parameters and their grid values (directly from copy.grid.toml)
    """

    grid_dir = Path(grid_dir)

    # 1. Load tested input parameters in the grid
    raw_params = toml.load(grid_dir / "copy.grid.toml")

    # Keep only the parameters and their values (ignore 'method' keys)
    tested_params = {}
    for key, value in raw_params.items():
        if isinstance(value, dict) and "values" in value:
            # Only store the 'values' list
            tested_params[key] = value["values"]

    grid_param_paths = list(tested_params.keys())

    # 2.Extract those parameters from loaded cases for each case of the grid
    case_params = {}

    for idx, case in enumerate(cases_data):
        params_for_case = {}
        init_params = case["init_parameters"]

        for path in grid_param_paths:
            keys = path.split(".")
            val = init_params

            try:
                for k in keys:
                    val = val[k]
                params_for_case[path] = val
            except (KeyError, TypeError):
                params_for_case[path] = None

        case_params[idx] = params_for_case

    return case_params, tested_params

def load_phi_crit(grid_dir: str | Path):
    """"
    Load the critical melt fraction (phi_crit) from the reference configuration file of the grid.

    Parameters
    ----------
    grid_dir : str or Path
        Path to the grid directory containing ref_config.toml.

    Returns
    -------
    phi_crit : float
        The critical melt fraction value loaded from the reference configuration.

    """
    grid_dir = Path(grid_dir)
    ref_file = grid_dir / "ref_config.toml"

    if not ref_file.exists():
        raise FileNotFoundError(f"ref_config.toml not found in {grid_dir}")

    ref = toml.load(open(ref_file))

    # Find phi_crit value
    try:
        phi_crit = ref["params"]["stop"]["solid"]["phi_crit"]
    except KeyError:
        raise KeyError("phi_crit not found in ref_config.toml")

    return phi_crit

def extract_solidification_time(cases_data: list, grid_dir: str | Path):
    """"
    Extract solidification time for each simulation of the grid for
    the condition Phi_global < phi_crit at last time step.

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing simulation data.

    grid_dir : str or Path
        Path to the grid directory containing ref_config.toml.

    Returns
    -------
    solidification_times : list
        A list containing the solidification times for all cases of the grid.
    """

    # Load phi_crit once
    phi_crit = load_phi_crit(grid_dir)

    solidification_times = []
    columns_printed = False

    for i, case in enumerate(cases_data):
        df = case['output_values']

        if df is None:
            solidification_times.append(np.nan)
            continue

        # Condition for complete solidification
        if 'Phi_global' in df.columns and 'Time' in df.columns:
            condition = df['Phi_global'] < phi_crit

            if condition.any():
                idx = condition.idxmax()
                solidification_times.append(df.loc[idx, 'Time'])
            else:
                solidification_times.append(np.nan) # if planet is not solidified, append NaN

        else:
            if not columns_printed:
                print("Warning: Missing Phi_global or Time column.")
                print("Columns available:", df.columns.tolist())
                columns_printed = True
            solidification_times.append(np.nan)

    return solidification_times

def generate_summary_csv(cases_data: list, case_params: dict, grid_dir: str | Path, grid_name: str):
    """
    Generate CSV file summarizing all simulation cases in the grid, including:
        - Case status
        - Values of tested grid parameters
        - All extracted values from runtime_helpfile.csv (at last timestep)
        - Solidification time

    Parameters
    ----------
    cases_data : list
        List of dictionaries containing simulation data.
    case_params : dict
        Dictionary mapping case index -> {parameter_name: value}
    grid_dir : str or Path
        Path to the grid directory containing ref_config.toml.
    grid_name : str
        Name of the grid.
    """

    # Compute solidification times
    solidification_times = extract_solidification_time(cases_data, grid_dir)

    # Extract data for each case
    summary_rows = []
    for case_index, case in enumerate(cases_data):
        row = {}

        # Case status
        row["case_number"] = case_index
        row["status"] = case["status"]

        # Values of tested grid parameters for each case
        params = case_params.get(case_index, {})
        for k, v in params.items():
            row[k] = v

        # Output values (at last timestep)
        df = case["output_values"]
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        # Solidification time
        row["solidification_time"] = solidification_times[case_index]

        summary_rows.append(row)

    # Create DataFrame and save it in the grid directory in post_processing/extracted_data/
    summary_df = pd.DataFrame(summary_rows)
    output_dir = grid_dir / "post_processing" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{grid_name}_final_extracted_data_all.csv"
    summary_df.to_csv(output_file, sep="\t", index=False)

def generate_completed_summary_csv(cases_data: list, case_params: dict, grid_dir: str | Path, grid_name: str):
    """
    Same function as generate_summary_csv, but only include fully 'Completed' cases.
    """
    # Compute solidification times
    solidification_times = extract_solidification_time(cases_data, grid_dir)

    # Extract data for each fully completed case
    summary_rows = []
    for case_index, case in enumerate(cases_data):
        status = case.get("status", "").lower()
        if not status.startswith("completed"):
            continue  # skip non-completed cases

        row = {}

        # Case status
        row["case_number"] = case_index
        row["status"] = case["status"]

        # Values of tested grid parameters for each case
        params = case_params.get(case_index, {})
        for k, v in params.items():
            row[k] = v

        # Output values (at last timestep)
        df = case["output_values"]
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        # Solidification time
        row["solidification_time"] = solidification_times[case_index]

        summary_rows.append(row)

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    output_dir = grid_dir / "post_processing" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Updated CSV name to indicate only completed cases
    output_file = output_dir / f"{grid_name}_final_extracted_data_completed.csv"
    summary_df.to_csv(output_file, sep="\t", index=False)

def generate_running_error_summary_csv(cases_data: list, case_params: dict, grid_dir: str | Path, grid_name: str):
    """
    Same function as generate_summary_csv, but only include 'Running' and 'Error' cases.
    """
    # Compute solidification times
    solidification_times = extract_solidification_time(cases_data, grid_dir)

    # Extract data for each running or error case
    summary_rows = []
    for case_index, case in enumerate(cases_data):
        status = case.get("status", "").lower()
        if not (status.startswith("running") or status.startswith("error")):
            continue  # skip cases that are neither running nor error

        row = {}

        # Case status
        row["case_number"] = case_index
        row["status"] = case["status"]

        # Values of tested grid parameters for each case
        params = case_params.get(case_index, {})
        for k, v in params.items():
            row[k] = v

        # Output values (at last timestep)
        df = case["output_values"]
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        # Solidification time
        row["solidification_time"] = solidification_times[case_index]

        summary_rows.append(row)

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    output_dir = grid_dir / "post_processing" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Updated CSV name to indicate only running/error cases
    output_file = output_dir / f"{grid_name}_final_extracted_data_running_error.csv"
    summary_df.to_csv(output_file, sep="\t", index=False)

# ---------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------

def plot_grid_status(df: pd.DataFrame, grid_dir: str | Path, grid_name: str):
    """
    Plot histogram summary of number of simulation statuses in
    the grid using the generated CSV file for all cases.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from grid_name_final_extracted_data_all.csv.

    grid_dir : Path
        Path to the grid directory.

    grid_name : str
        Name of the grid.
    """

    if "status" not in df.columns:
        raise ValueError("CSV must contain a 'status' column")

    # Clean and count statuses
    statuses = df["status"].astype(str)
    status_counts = statuses.value_counts().sort_values(ascending=False)
    total_simulations = len(df)

    # Format status labels for better readability
    formatted_status_keys = [s.replace(" (", " \n (") for s in status_counts.index]
    palette = sns.color_palette("Accent", len(status_counts))
    palette = dict(zip(formatted_status_keys, palette))

    # Prepare DataFrame for plotting
    plot_df = pd.DataFrame({
        "Status": formatted_status_keys,
        "Count": status_counts.values
    })

    # Plot histogram
    plt.figure(figsize=(11, 7))
    ax = sns.barplot(
        data=plot_df,
        x="Status",
        y="Count",
        hue="Status",
        palette=palette,
        dodge=False,
        edgecolor="black"
    )

    # Remove legend
    if ax.legend_:
        ax.legend_.remove()

    # Add counts and percentages above bars per status
    for i, count in enumerate(status_counts.values):
        percentage = 100 * count / total_simulations
        offset = 0.01 * status_counts.max()
        ax.text(
            i,
            count + offset,
            f"{count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=14
        )

    # Add total number of simulations text
    ax.text(
        0.97,
        0.94,
        f"Total number of simulations : {total_simulations}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16
    )

    # Formatting
    ax.grid(alpha=0.2, axis="y")
    ax.set_title(f"Simulation status summary for grid {grid_name}", fontsize=16)
    ax.set_xlabel("Simulation status", fontsize=16)
    ax.set_ylabel("Number of simulations", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # Save
    output_dir = Path(grid_dir) / "post_processing" / "grid_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"summary_grid_statuses_{grid_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def flatten_input_parameters(d: dict, parent_key: str = "") -> dict:
    """
    Flattens a nested input-parameter dictionary from a TOML configuration
    into a flat mapping of dot-separated parameter paths to their plotting
    configuration.

    Parameters
    ----------
    d : dict
        Nested dictionary describing input parameters (from TOML).
    parent_key : str, optional
        Accumulated parent key for recursive calls.

    Returns
    -------
    flat : dict
        Dictionary mapping parameter paths (e.g. ``"escape.zephyrus.Pxuv"``)
        to their corresponding configuration dictionaries.
    """

    flat = {}

    for k, v in d.items():
        if k == "colormap":
            continue

        new_key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, dict) and "label" in v:
            # Leaf parameter block
            flat[new_key] = v
        elif isinstance(v, dict):
            # Recurse deeper
            flat.update(flatten_input_parameters(v, new_key))

    return flat

def load_ecdf_plot_settings(cfg):
    """
    Load ECDF plotting settings for both input parameters and output variables
    from a configuration dictionary loaded from TOML.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from a TOML file.

    Returns
    -------
    param_settings : dict
        Mapping of input-parameter paths to plotting settings. Each value
        is a dict containing:
            - "label" : str
                Label for the parameter (used in colorbar).
            - "colormap" : matplotlib colormap
                Colormap used to color ECDF curves.
            - "log_scale" : bool
                Whether to normalize colors on a logarithmic scale.

    output_settings : dict
        Mapping of output variable names to plotting settings. Each value
        is a dict containing:
            - "label" : str
                X-axis label for the ECDF plot.
            - "log_scale" : bool
                Whether to use a logarithmic x-axis.
            - "scale" : float
                Factor applied to raw output values before plotting.
    """

    # Load input parameter settings
    raw_params = cfg["input_parameters"]
    default_cmap = getattr(cm, raw_params.get("colormap", "viridis"))

    # Flatten input parameters dictionary
    flat_params = flatten_input_parameters(raw_params)

    param_settings = {}
    for key, val in flat_params.items():
        param_settings[key] = {
            "label": val.get("label", key),
            "colormap": default_cmap,
            "log_scale": val.get("log_scale", False),
        }

    output_settings = {}
    for key, val in cfg.get("output_variables", {}).items():
        output_settings[key] = {
            "label": val.get("label", key),
            "log_scale": val.get("log_scale", False),
            "scale": val.get("scale", 1.0),
        }

    return param_settings, output_settings

def group_output_by_parameter(df, grid_parameters, outputs):
    """
    Groups output values (like P_surf) by a specific grid parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results including value of the grid parameter and the corresponding extracted output.

    grid_parameters : str
        Column name of the grid parameter to group by (like 'escape.zephyrus.efficiency').

    outputs : str
        Column name of the output to extract (like 'P_surf').

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
                output_values = subset[output].replace([np.inf, -np.inf], np.nan) # Replace inf with NaN
                output_values = output_values.dropna() # Remove NaN values
                output_values = output_values[output_values > 0]  # Keep only positive values

                value_dict[param_value] = output_values

            grouped[key_name] = value_dict

    return grouped

def latex(label: str) -> str:
    """
    Wraps a label in dollar signs for LaTeX formatting if it contains 2 backslashes.
    """
    return f"${label}$" if "\\" in label else label

def ecdf_grid_plot(grouped_data: dict, param_settings: dict, output_settings: dict, grid_dir: str | Path, grid_name: str):
    """
    Creates ECDF grid plots where each row corresponds to one input parameter
    and each column corresponds to one output. Saves the resulting figure as a PNG.

    Parameters
    ----------

    grid_params : dict
        Dictionary of tested grid parameters and their grid values (directly from copy.grid.toml)

    grouped_data : dict
        Dictionary where each key is of the form '[output]_per_[parameter]', and each value is a dict {param_value: [output_values]}.

    param_settings : dict
        For each input-parameter key, a dict containing:
            - "label": label of the colormap for the corresponding input parameter
            - "colormap": a matplotlib colormap (e.g. mpl.cm.plasma)
            - "log_scale": bool, whether to color-normalize on a log scale

    output_settings : dict
        For each output key, a dict containing:
            - "label": label of the x-axis for the corresponding output column
            - "log_scale": bool, whether to plot the x-axis on log scale
            - "scale": float, a factor to multiply raw values by before plotting

    plots_path : str
        Path to the grid where to create "single_plots_ecdf" and save all .png plots
    """

    # Load tested grid parameters
    raw_params = toml.load(grid_dir / "copy.grid.toml")
    tested_params = {}
    for key, value in raw_params.items():
        if isinstance(value, dict) and "values" in value:
            # Only store the 'values' list
            tested_params[key] = value["values"]
    grid_params = tested_params

    # List of parameter names (rows) and output names (columns)
    param_names = list(param_settings.keys())
    out_names   = list(output_settings.keys())

    # Create subplot grid: rows = input parameters, columns = outputs variables
    n_rows = len(param_names)
    n_cols = len(out_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.75 * n_rows), squeeze=False, gridspec_kw={'wspace': 0.1,  'hspace': 0.2})

    # Loop through parameters (rows) and outputs (columns)
    for i, param_name in enumerate(param_names):
        tested_param = grid_params.get(param_name, [])
        if not tested_param:
            print(f"⚠️ Skipping {param_name} — no tested values found in grid_params")
            continue
        settings = param_settings[param_name]

        # Determine if parameter is numeric or string for coloring
        is_numeric = np.issubdtype(np.array(tested_param).dtype, np.number)
        if is_numeric:
            vmin, vmax = min(tested_param), max(tested_param)
            if vmin == vmax:
                vmin, vmax = vmin - 1e-9, vmax + 1e-9
            if settings.get("log_scale", False):
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            def color_func(v):
                return settings["colormap"](norm(v))
            colorbar_needed = True
        else:
            unique_vals = sorted(set(tested_param))
            cmap = mpl.colormaps.get_cmap(settings["colormap"]).resampled(len(unique_vals))
            color_map = {val: cmap(j) for j, val in enumerate(unique_vals)}
            def color_func(v):
                return color_map[v]
            colorbar_needed = False

        for j, output_name in enumerate(out_names):
            ax = axes[i][j]
            out_settings = output_settings[output_name]

            # Add panel number in upper-left corner
            panel_number = i * n_cols + j + 1
            ax.text(
                0.03, 0.95,
                str(panel_number),
                transform=ax.transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left',
                color='black',
                bbox=dict(facecolor='white', edgecolor='silver', boxstyle='round,pad=0.2', alpha=0.8)
            )

            # Plot one ECDF per tested parameter value
            for val in tested_param:
                data_key = f"{output_name}_per_{param_name}"
                if val not in grouped_data.get(data_key, {}):
                    continue
                raw = np.array(grouped_data[data_key][val]) * out_settings.get("scale", 1.0)

                # Plot ECDF
                sns.ecdfplot(
                    data=raw,
                    log_scale=out_settings.get("log_scale", False),
                    color=color_func(val),
                    linewidth=4,
                    linestyle='-',
                    ax=ax
                )

            # Configure x-axis labels, ticks, grids
            if i == n_rows - 1:
                ax.set_xlabel(latex(out_settings["label"]), fontsize=22)
                ax.xaxis.set_label_coords(0.5, -0.3)
                ax.tick_params(axis='x', labelsize=22)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            # Configure y-axis (shared label added later)
            if j == 0:
                ax.set_ylabel("")
                ticks = [0.0, 0.5, 1.0]
                ax.set_yticks(ticks)
                ax.tick_params(axis='y', labelsize=22)
            else:
                ax.set_ylabel("")
                ax.set_yticks(ticks)
                ax.tick_params(axis='y', labelleft=False)

            ax.grid(alpha=0.4)

        # After plotting all outputs for this parameter (row), add colorbar or legend
        if colorbar_needed: # colorbar for numeric parameters
            sm = mpl.cm.ScalarMappable(cmap=settings["colormap"], norm=norm)
            rightmost_ax = axes[i, -1] # Get the rightmost axis in the current row
            cbar = fig.colorbar(sm,ax=rightmost_ax,pad=0.03,aspect=10)
            cbar.set_label(latex(settings["label"]), fontsize=24)
            cbar.ax.yaxis.set_label_coords(6, 0.5)
            ticks = sorted(set(tested_param))
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=22)
        else: # legend for string parameters
            handles = [mpl.lines.Line2D([0], [0], color=color_map[val], lw=4, label=str(val)) for val in unique_vals]
            ax.legend(handles=handles, fontsize=24,bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add a single, shared y-axis label
    fig.text(0.07, 0.5, 'Empirical cumulative fraction of grid simulations', va='center', rotation='vertical', fontsize=40)

    # Save figure
    output_dir = grid_dir / "post_processing" / "grid_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ecdf_grid_plot_{grid_name}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main(grid_analyse_toml_file: str | Path = None):

    # Load configuration from grid_analyse.toml
    with open(grid_analyse_toml_file, "rb") as f:
        cfg = tomllib.load(f)

    # Get grid path and name
    grid_path = Path(cfg["grid_path"])
    grid_name = get_grid_name(grid_path)

    print(grid_path)
    print(f"Analyzing grid: {grid_name}")

    # Load grid data
    data = load_grid_cases(grid_path)
    input_param_grid_per_case, tested_params_grid = get_tested_grid_parameters(
        data, grid_path
    )

    # --- Summary CSVs ---
    update_csv = cfg.get("update_csv", True)

    summary_dir = grid_path / "post_processing" / "extracted_data"
    summary_csv_all = summary_dir / f"{grid_name}_final_extracted_data_all.csv"
    summary_csv_completed = summary_dir / f"{grid_name}_final_extracted_data_completed.csv"
    summary_csv_running_error = summary_dir / f"{grid_name}_final_extracted_data_running_error.csv"

    if update_csv:
        generate_summary_csv(
            data, input_param_grid_per_case, grid_path, grid_name
        )
        generate_completed_summary_csv(
            data, input_param_grid_per_case, grid_path, grid_name
        )
        generate_running_error_summary_csv(
            data, input_param_grid_per_case, grid_path, grid_name
        )
    else:
        # Check that CSVs exist
        for f in [summary_csv_all, summary_csv_completed, summary_csv_running_error]:
            if not f.exists():
                raise FileNotFoundError(
                    f"{f.name} not found in {summary_dir}, "
                    "but update_csv is set to False. Please set update_csv to True to generate it."
                )

    # --- Plot grid status ---
    if cfg.get("plot_status", True):
        all_simulations_data_csv = pd.read_csv(summary_csv_all, sep="\t")
        plot_grid_status(all_simulations_data_csv, grid_path, grid_name)
        print("Plot grid status summary is available.")

    # --- ECDF plots ---
    if cfg.get("plot_ecdf", True):
        completed_simulations_data_csv = pd.read_csv(summary_csv_completed, sep="\t")
        columns_output = list(cfg["output_variables"].keys())
        grouped_data = {}
        for col in columns_output:
            group = group_output_by_parameter(
                completed_simulations_data_csv,
                tested_params_grid,
                [col],
            )
            grouped_data.update(group)

        param_settings_grid, output_settings_grid = load_ecdf_plot_settings(cfg)
        ecdf_grid_plot(
            grouped_data,
            param_settings_grid,
            output_settings_grid,
            grid_path,
            grid_name,
        )
        print("ECDF grid plot is available.")


if __name__ == "__main__":
    main()
