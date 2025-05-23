from pathlib import Path
import os
import pandas as pd
import numpy as np
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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

def plot_grid_status(cases_data, plot_dir: Path, status_colors: dict = None):
    """
    Plot the status of simulations from the PROTEUS grid with improved x-axis readability.

    Parameters
    ----------
    cases_data : list or DataFrame
        Contains the status of all simulations from the grid.

    plot_dir : Path
        Path to the plots directory.

    status_colors : dict, optional
        A dictionary mapping statuses to specific colors. If None, a default palette is used.
    """

    # Extract and clean statuses
    statuses = df['Status'].fillna('unknown').astype(str)
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
    plt.title(f"Grid status summary : {grid_name}", fontsize=16)
    plt.xlabel("Simulation status", fontsize=16)
    plt.ylabel("Number of simulations", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    output_path = plot_dir + 'grid_status_summary.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Summary plot of grid status is available")

if __name__ == '__main__':

    # User needs to specify paths
    grid_name   = 'escape_grid_habrok_7_params_1Msun'
    data_dir    = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_processed_data/{grid_name}/'
    plots_path  = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_plots/{grid_name}/'

    # Load and organize data before plotting
    df, grid_params, extracted_outputs = load_extracted_data(data_dir, grid_name)       # Load the data
    plot_dir_exists(plots_path)                                                         # Check if the plot directory exists. If not, create it.
    grouped_data = group_output_by_parameter(df, grid_params, extracted_outputs)        # Group extracted outputs by grid parameters

    # Plots
    plot_grid_status(df, plots_path)                                                    # Plot the grid status in an histogram

    # # Single plot of all input parameters and extracted outputs
    # # Define input parameter settings
    # param_settings = {
    #     "orbit.semimajoraxis":        {"label": "Semi-major axis [AU]",                   "colormap": cm.plasma,   "log_scale": False},
    #     "escape.zephyrus.Pxuv":       {"label": r"$P_{XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
    #     "escape.zephyrus.efficiency": {"label": r"Escape efficiency factor $\epsilon$",   "colormap": cm.spring,   "log_scale": False},
    #     "outgas.fO2_shift_IW":        {"label": r"$\log_{10}(fO_2)$ [IW]",                "colormap": cm.coolwarm, "log_scale": False},
    #     "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.Dark2,    "log_scale": False},
    #     "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
    #     "delivery.elements.H_oceans": {"label": "[H] [Earth's oceans]",                   "colormap": cm.winter,   "log_scale": False}}

    # # Define extracted output settings
    # output_settings = {
    #     #'esc_rate_total':      {"label": "Total escape rate [kg/s]",                  "log_scale": True,  "scale": 1.0},
    #     #'Phi_global':          {"label": "Melt fraction [%]",                         "log_scale": False, "scale": 100.0},
    #     #'P_surf':              {"label": "Surface pressure [bar]",                    "log_scale": True,  "scale": 1.0},
    #     #'atm_kg_per_mol':      {"label": "Mean molecular weight (MMW) [g/mol]",       "log_scale": False,  "scale": 1000.0},
    #     'solidification_time': {"label": "Solidification time [yr]",                  "log_scale": True,  "scale": 1.0}}


    # # Loop over extracted outputs and input parameters
    # for output_name, out_settings in output_settings.items():
    #     for param_name, settings in param_settings.items():
    #         tested_param = grid_params[param_name]  # Get the values of the input parameter
    #         if len(tested_param) <= 1:
    #             continue
    #         # Extract plot settings for this input parameter
    #         param_label = settings["label"]
    #         cmap = settings["colormap"]
    #         color_log = settings.get("log_scale", False)
    #         # Extract plot settings for this output
    #         x_label = out_settings["label"]
    #         x_log = out_settings.get("log_scale", False)
    #         scale = out_settings.get("scale", 1.0)

    #         # Determine colormap and color function if the parameter is numeric or string
    #         is_numeric = np.issubdtype(np.array(tested_param).dtype, np.number)
    #         if is_numeric:
    #             norm = mpl.colors.LogNorm(vmin=min(tested_param), vmax=max(tested_param)) if color_log else mpl.colors.Normalize(vmin=min(tested_param), vmax=max(tested_param))
    #             color_func = lambda v: cmap(norm(v))
    #             colorbar_needed = True
    #         else:
    #             unique_vals = sorted(set(tested_param))
    #             cats_cmap = mpl.cm.get_cmap(cmap, len(unique_vals))
    #             color_map = {val: cats_cmap(i) for i, val in enumerate(unique_vals)}
    #             color_func = lambda val: color_map[val]
    #             colorbar_needed = False

    #         # Create figure
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         data_key = f'{output_name}_per_{param_name}'
    #         for val in tested_param:
    #             if val not in grouped_data[data_key]:
    #                 continue
    #             raw = np.array(grouped_data[data_key][val]) * scale
    #             sns.ecdfplot(
    #             data=raw,
    #             log_scale=x_log,
    #             stat="proportion",
    #             color=color_func(val),
    #             linewidth=3,
    #             ax=ax)
    #         # Set axis labels
    #         ax.set_xlabel(x_label, fontsize=14)
    #         ax.set_ylabel("Normalized cumulative fraction of simulations", fontsize=14)
    #         ax.grid(alpha=0.1)

    #         # Add colorbar or legend
    #         if colorbar_needed:
    #             sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    #             cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    #             cbar.set_label(param_label, fontsize=14)
    #             ticks = sorted(set(tested_param))
    #             cbar.set_ticks(ticks)
    #         else:
    #             handles = [mpl.lines.Line2D([0], [0], color=color_map[val], lw=3, label=str(val)) for val in unique_vals]
    #             ax.legend(handles=handles, loc='lower right')

    #         # Save the figure
    #         output_dir = os.path.join(plots_path, "test_solid_ecdf_by_param_and_output")
    #         os.makedirs(output_dir, exist_ok=True)
    #         fname = f"ecdf_{output_name}_per_{param_name.replace('.', '_')}.png"
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(output_dir, fname), dpi=300)
    #         plt.close()
    # print(f"All single plots are available in the ecdf_by_param_and_output/ folder")

    # Grid plot 

    # # Define input parameter settings
    # param_settings = {
    #     "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.Dark2,    "log_scale": False},
    #     "orbit.semimajoraxis":        {"label": "a [AU]",                                 "colormap": cm.plasma,   "log_scale": False},
    #     "escape.zephyrus.efficiency": {"label": r"$\epsilon$",                            "colormap": cm.spring,   "log_scale": False},
    #     "escape.zephyrus.Pxuv":       {"label": r"$P_{XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
    #     "outgas.fO2_shift_IW":        {"label": r"$\log_{10}(fO_2 / IW)$",                "colormap": cm.coolwarm, "log_scale": False},
    #     "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
    #     "delivery.elements.H_oceans": {"label": "[H] [oceans]",                           "colormap": cm.winter,   "log_scale": False}}

    # # Define extracted output settings
    # output_settings = {
    #     'solidification_time': {"label": "Solidification [yr]",             "log_scale": True,  "scale": 1.0},
    #     'Phi_global':          {"label": "Melt fraction [%]",               "log_scale": False, "scale": 100.0},
    #     'P_surf':              {"label": "Surface pressure [bar]",          "log_scale": True,  "scale": 1.0},
    #     'esc_rate_total':      {"label": "Escape rate [kg/s]",              "log_scale": True,  "scale": 1.0},
    #     'atm_kg_per_mol':      {"label": "MMW [g/mol]",                     "log_scale": False,  "scale": 1000.0}}

    # # Prepare parameter and output lists
    # param_names = list(param_settings.keys())
    # out_names = list(output_settings.keys())

    # # Create subplot grid: rows = parameters, columns = outputs
    # n_rows = len(param_names)
    # n_cols = len(out_names)
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    
    # # Loop through parameters (rows) and outputs (columns)
    # for i, param_name in enumerate(param_names):
    #     tested_param = grid_params[param_name]
    #     settings = param_settings[param_name]
    #     # Determine coloring
    #     is_numeric = np.issubdtype(np.array(tested_param).dtype, np.number)
    #     if is_numeric:
    #         norm = mpl.colors.LogNorm(vmin=min(tested_param), vmax=max(tested_param)) if settings.get("log_scale", False) else mpl.colors.Normalize(vmin=min(tested_param), vmax=max(tested_param))
    #         color_func = lambda v: settings["colormap"](norm(v))
    #         colorbar_needed = True
    #     else:
    #         unique_vals = sorted(set(tested_param))
    #         cmap = mpl.cm.get_cmap(settings["colormap"], len(unique_vals))
    #         color_map = {val: cmap(j) for j, val in enumerate(unique_vals)}
    #         color_func = lambda v: color_map[v]
    #         colorbar_needed = False

    #     for j, output_name in enumerate(out_names):
    #         ax = axes[i][j]
    #         out_settings = output_settings[output_name]
    #         # Plot each ECDF
    #         for val in tested_param:
    #             data_key = f"{output_name}_per_{param_name}"
    #             if val not in grouped_data[data_key]:
    #                 continue
    #             raw = np.array(grouped_data[data_key][val]) * out_settings.get("scale", 1.0)
    #             sns.ecdfplot(
    #                 data=raw,
    #                 log_scale=out_settings.get("log_scale", False),
    #                 #stat="proportion",
    #                 color=color_func(val),
    #                 linewidth=4,
    #                 ax=ax
    #             )
    #         # Labels and grid
    #         if i == n_rows - 1:
    #             ax.set_xlabel(out_settings["label"], fontsize=22)
    #             ax.xaxis.set_label_coords(0.5, -0.3)
    #             ax.tick_params(axis='x', labelsize=22)
    #         else :
    #             ax.tick_params(axis='x', labelbottom=False)
    #         if j == 0:
    #             ax.set_ylabel("")
    #             ticks = [0.0, 0.5, 1.0]
    #             ax.set_yticks(ticks)
    #             ax.tick_params(axis='y', labelsize=22)  # show tick labels with size
    #         else:
    #             ax.set_ylabel("")
    #             ax.set_yticks(ticks)
    #             ax.tick_params(axis='y', labelleft=False)
    #         # Grid
    #         ax.grid(alpha=0.4)
    #     # Colorbar or legend
    #     if colorbar_needed:
    #         sm = mpl.cm.ScalarMappable(cmap=settings["colormap"], norm=norm)
    #         cbar = fig.colorbar(sm, ax=ax, pad=0.08, aspect=10)
    #         cbar.set_label(settings["label"], fontsize=24)
    #         cbar.ax.yaxis.set_label_coords(5.5, 0.5)
    #         ticks = sorted(set(tested_param))
    #         cbar.set_ticks(ticks)
    #         cbar.ax.tick_params(labelsize=22) 
    #     else:
    #         handles = [mpl.lines.Line2D([0], [0], color=color_map[val], lw=4, label=str(val)) for val in unique_vals]
    #         ax.legend(handles=handles, fontsize=24,bbox_to_anchor=(1.01, 1), loc='upper left')

    # # Add a single shared y-axis label
    # fig.text(0.04, 0.5, 'Normalized cumulative fraction of simulations', va='center', rotation='vertical', fontsize=40)
    # # Adjust layout and save
    # plt.tight_layout(rect=[0.08, 0.02, 1, 0.97])
    # fig.savefig(os.path.join(plots_path, "ecdf_param_output_grid_test_solid.png"), dpi=300)
    # plt.close(fig)

    # print(f"Grid plot is available in the ecdf_param_output_grid.png file")
