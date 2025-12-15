from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml

# ---------------------------------------------------------
# Data loading and processing functions
# ---------------------------------------------------------

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

def get_tested_grid_parameters(cases_data: list, grid_dir: str | Path):
    """
    Extract tested grid parameters per case using:
      - copy.grid.toml to determine which parameters were varied
      - init_parameters already loaded by load_grid_cases

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
        Dictionary of tested grid parameters and their grid values
        (directly from copy.grid.toml)
    """

    grid_dir = Path(grid_dir)

    # ---------------------------------------------------------
    # 1) Load tested grid parameter definitions
    # ---------------------------------------------------------
    raw_params = toml.load(grid_dir / "copy.grid.toml")

    # Keep only the parameters and their values (ignore 'method' keys)
    tested_params = {}
    for key, value in raw_params.items():
        if isinstance(value, dict) and "values" in value:
            # Only store the 'values' list
            tested_params[key] = value["values"]

    grid_param_paths = list(tested_params.keys())

    # ---------------------------------------------------------
    # 2) Extract those parameters from loaded cases
    # ---------------------------------------------------------
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

def generate_summary_csv(
    cases_data: list,
    case_params: dict,
    grid_dir: str | Path,
    grid_name: str
):
    """
    Export a summary of simulation cases to a TSV file using
    already-loaded data only.

    Column order:
      - case metadata
      - tested grid parameters
      - runtime_helpfile.csv (last timestep)
      - solidification_time
    """
    # ---------------------------------------------------------
    # Compute solidification times ONCE
    # ---------------------------------------------------------
    solidification_times = extract_solidification_time(cases_data, grid_dir)

    summary_rows = []

    for case_index, case in enumerate(cases_data):
        row = {}

        # -----------------------------------------------------
        # Case metadata
        # -----------------------------------------------------
        row["case_number"] = case_index
        row["status"] = case["status"]

        # -----------------------------------------------------
        # Tested grid parameters
        # -----------------------------------------------------
        params = case_params.get(case_index, {})
        for k, v in params.items():
            row[k] = v

        # -----------------------------------------------------
        # Output values (last timestep)
        # -----------------------------------------------------
        df = case["output_values"]
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        # -----------------------------------------------------
        # Solidification time (LAST column)
        # -----------------------------------------------------
        row["solidification_time"] = solidification_times[case_index]

        summary_rows.append(row)

    # ---------------------------------------------------------
    # Create DataFrame and save
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    output_dir = grid_dir / "post_processing" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{grid_name}_final_extracted_data_all.csv"
    summary_df.to_csv(output_file, sep="\t", index=False)

def generate_completed_summary_csv(
    cases_data: list,
    case_params: dict,
    grid_dir: str | Path,
    grid_name: str
):
    """
    Export a summary of simulation cases to a TSV file, but only
    include cases whose status starts with 'completed'.

    Column order:
      - case metadata
      - tested grid parameters
      - runtime_helpfile.csv (last timestep)
      - solidification_time
    """
    # ---------------------------------------------------------
    # Compute solidification times ONCE
    # ---------------------------------------------------------
    solidification_times = extract_solidification_time(cases_data, grid_dir)

    summary_rows = []

    for case_index, case in enumerate(cases_data):
        status = case.get("status", "").lower()
        if not status.startswith("completed"):
            continue  # skip non-completed cases

        row = {}

        # -----------------------------------------------------
        # Case metadata
        # -----------------------------------------------------
        row["case_number"] = case_index
        row["status"] = case["status"]

        # -----------------------------------------------------
        # Tested grid parameters
        # -----------------------------------------------------
        params = case_params.get(case_index, {})
        for k, v in params.items():
            row[k] = v

        # -----------------------------------------------------
        # Output values (last timestep)
        # -----------------------------------------------------
        df = case["output_values"]
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        # -----------------------------------------------------
        # Solidification time (LAST column)
        # -----------------------------------------------------
        row["solidification_time"] = solidification_times[case_index]

        summary_rows.append(row)

    # ---------------------------------------------------------
    # Create DataFrame and save
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    output_dir = grid_dir / "post_processing" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Updated CSV name to indicate filtered results
    output_file = output_dir / f"{grid_name}_final_extracted_data_completed.csv"
    summary_df.to_csv(output_file, sep="\t", index=False)

def generate_running_error_summary_csv(
    cases_data: list,
    case_params: dict,
    grid_dir: str | Path,
    grid_name: str
):
    """
    Export a summary of simulation cases to a TSV file, but only
    include cases whose status starts with 'Running' or 'Error'.

    Column order:
      - case metadata
      - tested grid parameters
      - runtime_helpfile.csv (last timestep)
      - solidification_time
    """
    # ---------------------------------------------------------
    # Compute solidification times ONCE
    # ---------------------------------------------------------
    solidification_times = extract_solidification_time(cases_data, grid_dir)

    summary_rows = []

    for case_index, case in enumerate(cases_data):
        status = case.get("status", "").lower()
        if not (status.startswith("running") or status.startswith("error")):
            continue  # skip cases that are neither running nor error

        row = {}

        # -----------------------------------------------------
        # Case metadata
        # -----------------------------------------------------
        row["case_number"] = case_index
        row["status"] = case["status"]

        # -----------------------------------------------------
        # Tested grid parameters
        # -----------------------------------------------------
        params = case_params.get(case_index, {})
        for k, v in params.items():
            row[k] = v

        # -----------------------------------------------------
        # Output values (last timestep)
        # -----------------------------------------------------
        df = case["output_values"]
        if df is not None:
            for col in df.columns:
                row[col] = df[col].iloc[-1]

        # -----------------------------------------------------
        # Solidification time (LAST column)
        # -----------------------------------------------------
        row["solidification_time"] = solidification_times[case_index]

        summary_rows.append(row)

    # ---------------------------------------------------------
    # Create DataFrame and save
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    output_dir = grid_dir / "post_processing" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Updated CSV name to indicate filtered results
    output_file = output_dir / f"{grid_name}_final_extracted_data_running_error.csv"
    summary_df.to_csv(output_file, sep="\t", index=False)

# ---------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------

def plot_grid_status(cases_data: list, grid_dir: str | Path, grid_name: str):
    """
    Plot the status of simulations from the PROTEUS grid with improved x-axis readability.

    Parameters
    ----------
    cases_data : list
        List of dictionaries returned by `load_grid_cases`.

    plot_dir : Path
        Path to the plots directory.

    grid_name : str
        Name of the grid, used for the plot title.

    status_colors : dict, optional
        A dictionary mapping statuses to specific colors. If None, a default palette is used.
    """

    # Extract and clean statuses
    statuses = [case.get('status', 'Unknown') for case in cases_data]
    statuses = pd.Series(statuses, name='Status')
    status_counts = statuses.value_counts().sort_values(ascending=False)

    # Set colors for the bars
    palette = sns.color_palette("Accent", len(status_counts))
    formatted_status_keys = [s.replace(" (", " \n (") for s in status_counts.index]
    palette = dict(zip(formatted_status_keys, palette))

    # Prepare dataframe for plotting
    plot_df = pd.DataFrame({
        'Status': formatted_status_keys,
        'Count': status_counts.values
    })

    plt.figure(figsize=(11, 7))
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
        offset = 0.005 * status_counts.max()  # 1% of the max count
        ax.text(
            i, count + offset,
            f"{count} ({percentage:.1f}%)",
            ha='center', va='bottom', fontsize=14
        )

    # Boxed total in upper right
    plt.gca().text(
        0.97, 0.94,
        f"Total number of simulations : {total_simulations}",
        transform=plt.gca().transAxes,
        ha='right', va='top',
        fontsize=16
    )

    plt.grid(alpha=0.2, axis='y')
    plt.title(f"Simulation status summary for grid {grid_name}", fontsize=16)
    plt.xlabel("Simulation status", fontsize=16)
    plt.ylabel("Number of simulations", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()

    output_dir = grid_dir / "post_processing" / "grid_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"summary_grid_statuses_{grid_name}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

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

def ecdf_grid_plot(grid_params: dict, grouped_data: dict, param_settings: dict, output_settings: dict, grid_dir: str | Path, grid_name: str):
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
        if not tested_param:
            print(f"⚠️ Skipping {param_name} — no tested values found in grid_params")
            continue

        settings = param_settings[param_name]

        # Determine coloring
        is_numeric = np.issubdtype(np.array(tested_param).dtype, np.number)
        if is_numeric:
            vmin, vmax = min(tested_param), max(tested_param)
            if vmin == vmax:
                # avoid log/normalize errors with constant values
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
            panel_number = i * n_cols + j + 1  # number of panels left-to-right, top-to-bottom
            ax.text(
                0.02, 0.98,               # relative position in axes coordinates
                str(panel_number),         # text to display
                transform=ax.transAxes,    # use axis-relative coordinates
                fontsize=18,
                fontweight='bold',
                va='top',                  # vertical alignment
                ha='left',                  # horizontal alignment
                color='black'
            )

            # Plot one ECDF per tested parameter value
            for val in tested_param:
                data_key = f"{output_name}_per_{param_name}"
                if val not in grouped_data.get(data_key, {}):
                    continue
                raw = np.array(grouped_data[data_key][val]) * out_settings.get("scale", 1.0)
                # Plot ECDf if output == df['H_kg_atm'] then plot only values > 1e10 AND psurf > 1 bar
                if output_name.endswith('_kg_atm'):
                    raw = np.clip(raw, 1e15, None)
                elif output_name.endswith('P_surf'):
                    raw = np.clip(raw, 1, None)
                else:
                    raw = raw

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
                ax.set_xlabel(out_settings["label"], fontsize=22)
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
        if colorbar_needed:
            sm = mpl.cm.ScalarMappable(cmap=settings["colormap"], norm=norm)
            # attach the colorbar to the right‐most subplot in row i:
            rightmost_ax = axes[i, -1]
            cbar = fig.colorbar(sm,ax=rightmost_ax,pad=0.03,aspect=10)
            cbar.set_label(settings["label"], fontsize=24)
            # This is for plot 0.194Msun
            # if param_name == "orbit.semimajoraxis":
            #     cbar.ax.yaxis.set_label_coords(9.5, 0.5)
            # else:
            #     cbar.ax.yaxis.set_label_coords(6, 0.5)
            # This is for 1Msun
            cbar.ax.yaxis.set_label_coords(6, 0.5)
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
    output_dir = grid_dir / "post_processing" / "grid_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ecdf_grid_plot_{grid_name}.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
