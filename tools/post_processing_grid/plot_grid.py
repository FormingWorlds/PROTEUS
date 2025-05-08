from pathlib import Path
from io import StringIO

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatterMathtext

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

    print(f"Grid status histogram successfully saved to {output_path}")

def plot_hist_kde(values, ax, color, plot_hist=True, plot_kde=True, cumulative=True, log_x=False, bins=100, bw_adjust=0.3, hist_element="step", kde_kwargs={}, hist_kwargs={}):
    """
    Plot a cumulative or non-cumulative histograms and/or KDE curves on the given x-axis.

    Parameters
    ----------
    values : array-like
        Data points on the x-axis used to calculate the histogram/KDE.
    ax : matplotlib.axes.Axes
        Axis to draw the plot on.
    color : str
        Color for both histogram and KDE.
    plot_hist : bool
        Whether to plot a histogram.
    plot_kde : bool
        Whether to plot a KDE curve.
    cumulative : bool
        Whether to plot the cumulative distribution.
    log_x : bool
        Whether to use a logarithmic x-axis.
    bw_adjust : float
        Bandwidth adjustment for KDE.
    bins : int
        Number of bins for the histogram.
    hist_element : str
        'step' or 'bars' for histogram style.
    kde_kwargs : dict
        Additional kwargs for sns.kdeplot.
    hist_kwargs : dict
        Additional kwargs for sns.histplot.
    """

    # Apply log transform if needed
    if log_x:
        values = np.log10(values)
    
    if plot_hist:
        sns.histplot(
            values,
            bins=bins,
            cumulative=cumulative,
            stat="density",
            element=hist_element,
            color=color,
            ax=ax,
            kde=False,
            **hist_kwargs
        )

    if plot_kde:
        sns.kdeplot(
            values,
            cumulative=cumulative,
            bw_adjust=bw_adjust,
            clip=(np.min(values), np.max(values)),
            common_grid=True,
            color=color,
            ax=ax,
            **kde_kwargs
        )

    # Set axis labels and formatting
    if log_x:
        # Keep axis in linear scale (values already log-transformed),
        # but show ticks as powers of 10
        ax.set_xlabel("log10(x)")
        ax.set_xticks(np.log10(ticks := np.geomspace(np.nanmin(10**values), np.nanmax(10**values), num=5)))
        ax.set_xticklabels([f"{int(tick):.0e}" for tick in ticks])
    else:
        ax.set_xscale('linear')

def safe_log_formatter(x, _):
    try:
        if x > 0:
            return r'$10^{{{}}}$'.format(int(np.log10(x)))
        else:
            return ''
    except Exception:
        return ''

def plot_distributions(data_dict, xlabel, ylabel, colormap, vmin=None, vmax=None, key_label="Parameter", ax=None, save_path=None, bw_adjust=0.3, log_x=True, tick_values=None, norm=None, plot_hist=True, plot_kde=True, cumulative=True, bins=100):
    """
    Plot cumulative KDE curves for one of the output parameters of the grid (like esc_rate_total, solidification_time) on the x-axis.
    The different curves correspond to a input parameter from the grid with a color mapped on the right side of the plot.

    Parameters
    ----------
    data_dict : dict
        Dictionary of datasets, where each key represents a unique variable (e.g., semi-major axis),
        and each value is a list or array of values (e.g., solidification times).

    xlabel : str
        Label for the x-axis.

    ylabel : str
        Label for the y-axis.

    cmap : matplotlib.colors.Colormap
        The colormap used to assign colors to each dataset based on their corresponding key value.

    vmin : float
        The minimum value of the key variable to normalize the colormap.

    vmax : float
        The maximum value of the key variable to normalize the colormap.

    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.

    save_path : str, optional
        Path to save the generated plot. If None, the plot will not be saved.
    
    bw_adjust : float, optional, default=0.3
        Bandwidth adjustment factor : controls the smoothness of the KDE.
        Smaller values make the KDE more sensitive to data, while larger values smooth it out.
    
    log_x : bool, optional
        If True, sets the x-axis to logarithmic scale.


    tick_values : list of float, optional
        Values to use as ticks on the colorbar. Useful for discrete parameter steps.


    Returns
    -------
    '"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    keys = sorted(data_dict.keys())
    if vmin is None:
        vmin = min(keys)
    if vmax is None:
        vmax = max(keys)

    # Use provided or default norm
    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)

    # Plot each group
    for key in keys:
        values = np.array(data_dict[key])

        # Filter out invalid values for log scale
        values = values[np.isfinite(values)]  # Remove inf and NaN
        if log_x:
            values = values[values > 0]       # Remove zero and negatives
            values = np.log10(values)
        
        if len(values) < 2:
            print(f"Skipping key {key} due to insufficient valid data points.")
            continue

        color = colormap(norm(key))

        plot_hist_kde(values=values, ax=ax, color=color, plot_hist=plot_hist, plot_kde=plot_kde, cumulative=cumulative, log_x=log_x, bins=bins, bw_adjust=0.3, hist_element="step", kde_kwargs={}, hist_kwargs={})
    
    ax.set_xlabel(f"log10({xlabel})" if log_x else xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)

    # Colorbar setup
    sm.set_array(np.linspace(vmin, vmax, 100))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(key_label, fontsize=12)

    # Use grid param values as colorbar ticks
    if tick_values is not None:
        cbar.set_ticks(tick_values)
        if isinstance(norm, mcolors.LogNorm):
            # Format log ticks with LaTeX-style math text (e.g., 10⁻³)
            cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in tick_values])
            cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        else:
            # Linear scale ticks
            cbar.set_ticklabels([str(t) for t in tick_values])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)



def generate_single_plots(extracted_outputs, grouped_data, grid_params, plots_path, param_label_map, colormaps_by_param, output_label_map, log_scale_grid_params, log_x=True, plot_hist=True, plot_kde=True, cumulative=True, bins=100):
    """
    Generate and save normalized cumulative distribution plots for each output vs. grid parameter.

    Parameters:
    ----------
    extracted_outputs : list of str
        List of extracted output quantities to plot.

    grouped_data : dict
        Data for each output/parameter pair. (like solidification_time_per_semimajoraxis)

    grid_params : dict
        Parameter values used in the grid.

    plots_path : str
        Directory where plots will be saved.

    param_label_map : dict
        Dictionary containing the label of the grid parameter for the plot.

    colormaps_by_param : dict
        Dictionary containing the colormap to use for each grid parameter for the plot.

    output_label_map : dict, optional
        Dictionary containing the label of the extracted output quantity for the plot.

    log_scale_grid_params : list of str, optional
        Parameters to use log scale for colormap normalization. (like escape.zephyrus.Pxuv)
    """

    if param_label_map is None:
        raise ValueError("param_label_map must be provided.")

    if colormaps_by_param is None:
        raise ValueError("colormaps_by_param must be provided.")

    if output_label_map is None:
        raise ValueError("output_label_map must be provided.")

    if log_scale_grid_params is None:
        log_scale_grid_params = []

    for output_name in extracted_outputs:
        for param, cmap in colormaps_by_param.items():
            data_key = f"{output_name}_per_{param}"

            if data_key not in grouped_data:
                print(f"WARNING: Skipping {data_key} — not found in grouped_data")
                continue

            data_dict_raw = grouped_data[data_key]
            data_dict = data_dict_raw.copy()

            if output_name == "Phi_global":
                data_dict = {k: [v_i * 100 for v_i in v] for k, v in data_dict.items()}

            tick_values = grid_params.get(param)
            vmin = min(tick_values)
            vmax = max(tick_values)

            xlabel = output_label_map.get(output_name, output_name.replace("_", " ").title())
            ylabel = "Normalized cumulative fraction of simulations"
            key_label = param_label_map.get(param, param.replace("_", " ").title())

            save_dir = Path(plots_path) / 'single_plot'
            plot_dir_exists(save_dir)

            save_name = f"cumulative_{output_name}_vs_{param.replace('.', '_')}.png"
            save_path = save_dir / save_name

            fig, ax = plt.subplots(figsize=(10, 6))

            norm = (
                mcolors.LogNorm(vmin=vmin, vmax=vmax)
                if param in log_scale_grid_params
                else mcolors.Normalize(vmin=vmin, vmax=vmax)
            )

            df = pd.DataFrame(data_dict)
            keys = list(df.columns)

            plot_distributions(
                data_dict=data_dict,
                xlabel=xlabel,
                ylabel=ylabel,
                colormap=cmap,
                vmin=vmin,
                vmax=vmax,
                key_label=key_label,
                log_x=log_x,
                tick_values=tick_values,
                save_path=save_path,
                norm=norm,
                ax=ax,
                bw_adjust=0.3,
                plot_hist=plot_hist,
                plot_kde=plot_kde,
                cumulative=cumulative,         
                bins=bins   
            )
            plt.close(fig)

    print("All single plots saved.")

# def generate_grid_plot(extracted_outputs, grouped_data, grid_params, plots_path, param_label_map, colormaps_by_param, output_label_map, log_scale_grid_params, log_x=True, bins=100):
#     """
#     Generate and save normalized cumulative distribution plots for each output vs. grid parameters.
#     This creates a subplot where each column corresponds to one extracted output, and each row corresponds to one grid parameter.

#     Parameters:
#     ----------
#     extracted_outputs : list of str
#         List of extracted output quantities to plot.

#     grouped_data : dict
#         Data for each output/parameter pair. (like solidification_time_per_semimajoraxis)

#     grid_params : dict
#         Parameter values used in the grid.

#     plots_path : str
#         Directory where plots will be saved.

#     param_label_map : dict
#         Dictionary containing the label of the grid parameter for the plot.

#     colormaps_by_param : dict
#         Dictionary containing the colormap to use for each grid parameter for the plot.

#     output_label_map : dict, optional
#         Dictionary containing the label of the extracted output quantity for the plot.

#     log_scale_grid_params : list of str, optional
#         Parameters to use log scale for colormap normalization. (like escape.zephyrus.Pxuv)
#     """

#     if param_label_map is None:
#         raise ValueError("param_label_map must be provided.")
#     if colormaps_by_param is None:
#         raise ValueError("colormaps_by_param must be provided.")
#     if output_label_map is None:
#         raise ValueError("output_label_map must be provided.")
#     if log_scale_grid_params is None:
#         log_scale_grid_params = []

#     num_cols = len(extracted_outputs)
#     num_rows = len(grid_params)

#     # Create subplots with the appropriate layout
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 16, num_rows * 12))
#     plt.subplots_adjust(hspace=0.15, wspace=0.15)

#     if num_rows == 1:
#         axes = np.expand_dims(axes, axis=0)  # Make sure it's 2D if only one row.

#     for i, output_name in enumerate(extracted_outputs):
#         for j, (param, cmap) in enumerate(colormaps_by_param.items()):
#             data_key = f"{output_name}_per_{param}"

#             if data_key not in grouped_data:
#                 print(f"WARNING: Skipping {data_key} — not found in grouped_data")
#                 continue

#             data_dict_raw = grouped_data[data_key]
#             data_dict = data_dict_raw.copy()

#             if output_name == "Phi_global":
#                 data_dict = {k: [v_i * 100 for v_i in v] for k, v in data_dict.items()}

#             tick_values = grid_params.get(param)
#             vmin = min(tick_values)
#             vmax = max(tick_values)

#             xlabel = output_label_map.get(output_name, output_name.replace("_", " ").title())
#             ylabel = "Normalized cumulative fraction of simulations"
#             key_label = param_label_map.get(param, param.replace("_", " ").title())

#             # Select the axis for this subplot
#             ax = axes[j, i]

#             norm = (
#                 mcolors.LogNorm(vmin=vmin, vmax=vmax)
#                 if param in log_scale_grid_params
#                 else mcolors.Normalize(vmin=vmin, vmax=vmax)
#             )

#             plot_kde_cumulative(
#                 data_dict=data_dict,
#                 xlabel=xlabel,
#                 ylabel=ylabel,
#                 cmap=cmap,
#                 vmin=vmin,
#                 vmax=vmax,
#                 key_label=key_label,
#                 log_x=log_x,
#                 tick_values=tick_values,
#                 save_path=None,  # Don't save the plot yet
#                 norm=norm,
#                 ax=ax
#             )

#             # Set titles for each subplot
#             if i == 0:
#                 ax.set_ylabel(ylabel, fontsize=12)
#             if j == num_rows - 1:
#                 ax.set_xlabel(xlabel, fontsize=12)

#             # Customize plot with grid and ticks
#             ax.grid(alpha=0.2)
#             ax.set_ylim(0, 1.02)

#     # Save the complete subplot figure
#     save_dir = Path(plots_path) / 'grid_plot'
#     plot_dir_exists(save_dir)

#     save_name = "cumulative_grid_plot.png"
#     save_path = save_dir / save_name
#     plt.savefig(save_path, dpi=300)
#     plt.close(fig)

#     print(f"All subplot plots saved to {save_path}")

if __name__ == '__main__':

    # User needs to specify paths
    grid_name   = 'escape_grid_habrok_5_params_a_0.1_1Msun_agni'
    data_dir    = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_processed_data/{grid_name}/'
    plots_path  = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_plots/{grid_name}/'

    # Load and organize data before plotting
    df, grid_params, extracted_outputs = load_extracted_data(data_dir, grid_name)       # Load the data
    plot_dir_exists(plots_path)                                                         # Check if the plot directory exists. If not, create it.
    grouped_data = group_output_by_parameter(df, grid_params, extracted_outputs)        # Group extracted outputs by grid parameters

    ## Plots
    #plot_grid_status(df, plots_path)                                                    # Plot the grid status in an histogram

    ############## TEST PLOT ##############
    # # Extract the grouped dict
    # grouped_values = grouped_data['P_surf_per_escape.zephyrus.Pxuv']

    # # Set up colormap and normalization
    # colormap = cm.cividis
    # param_vals = sorted(grouped_values.keys())
    # norm = plt.Normalize(vmin=min(param_vals), vmax=max(param_vals))

    # # Create a single plot with all curves
    # fig, ax = plt.subplots(figsize=(7, 5))

    # for pxuv_val in param_vals:
    #     values = grouped_values[pxuv_val].dropna().values  # Clean NaNs
    #     if len(values) == 0:
    #         continue  # Skip empty

    #     plot_hist_kde(
    #         values=values,
    #         ax=ax,
    #         color=colormap(norm(pxuv_val)),
    #         plot_hist=True, 
    #         plot_kde=True,
    #         cumulative=True,
    #         log_x=False,       # Log-scale for P_surf
    #         bins=10
    #     )

    # # Add colorbar for Pxuv mapping
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label(r"$P_{XUV}$ [bar]")

    # ax.set_ylabel("Cumulative density")
    # ax.set_xlabel("Surface pressure [bar]")
    # ax.set_title("Cumulative P_surf distributions for different $P_{XUV}$ values")

    # plt.tight_layout()
    # plt.savefig(plots_path + "test_plot_P_surf_per_Pxuv.png", dpi=300)
    # plt.close()

    plt.figure(figsize=(10, 6))
    for pxuv in grid_params['escape.zephyrus.Pxuv'] : 
        sns.histplot(data=np.array(grouped_data['P_surf_per_escape.zephyrus.Pxuv'][pxuv]),
                    bins=5,
                    kde=True,
                    kde_kws={'bw_adjust': 0.5},
                    stat="density",
                    element="step",
                    cumulative=False,
                    log_scale=False,
                    fill=False,
                    linewidth=1.5, 
                    color=cm.cividis(pxuv)
                    )
    plt.xlabel("Surface pressure [bar]")
    plt.ylabel("Empirical cumulative density")    
    plt.savefig(plots_path + "test_plot_P_surf_per_Pxuv.png", dpi=300)

    # grouped_data = {
    #     'P_surf_per_escape.zephyrus.Pxuv': {
    #         1e-05: np.random.lognormal(mean=0, sigma=1, size=500),
    #         0.01:  np.random.lognormal(mean=1, sigma=1, size=500),
    #         10.0:  np.random.lognormal(mean=2, sigma=1, size=500)
    #     }
    # }
    # pxuv_vals = [1e-05, 0.01, 10.0]

    # # --- figure + axis ---
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # --- set up normalization + ScalarMappable for the colorbar ---
    # norm = mcolors.LogNorm(vmin=min(pxuv_vals), vmax=max(pxuv_vals))
    # # if you want linear scaling, use Normalize instead:
    # # norm = mcolors.Normalize(vmin=min(pxuv_vals), vmax=max(pxuv_vals))

    # sm = cm.ScalarMappable(norm=norm, cmap='cividis')
    # sm.set_array([])  # dummy array for the mappable

    # # --- loop over pxuv values, plotting each line ---
    # for pxuv in pxuv_vals:
    #     color = cm.cividis(norm(pxuv))
    #     sns.histplot(
    #         data=grouped_data['P_surf_per_escape.zephyrus.Pxuv'][pxuv],
    #         bins=30,
    #         kde=True,
    #         kde_kws={'bw_adjust': 0.1},
    #         stat="density",
    #         element="step",
    #         cumulative=True,
    #         fill=False,
    #         linewidth=1.5,
    #         color=color,
    #         ax=ax
    #     )

    # # --- labels ---
    # ax.set_xlabel("Surface pressure [bar]")
    # ax.set_ylabel("Cumulative density")

    # # --- move y-axis ticks & label to the right ---
    # # ax.yaxis.tick_right()
    # # ax.yaxis.set_label_position("right")
    # # ax.tick_params(axis='y', labelright=True, labelleft=False)

    # # --- add the colorbar ---
    # cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    # cbar.set_label("Pxuv value")

    # # --- save or show ---
    # plt.tight_layout()
    # plt.savefig(plots_path + "test_plot_P_surf_per_Pxuv.png", dpi=300)

    # # Single plots
    # param_label_map = {
    # "orbit.semimajoraxis": "Semi-major axis [AU]",
    # "escape.zephyrus.Pxuv": r"$P_{XUV}$ [bar]",
    # "escape.zephyrus.efficiency": r"Escape efficiency factor, $\epsilon$",
    # "outgas.fO2_shift_IW":r"$log_{10}$($fO_2$) [$\Delta$ IW]",
    # #"atmos_clim.module": "Atmospheric module",
    # "delivery.elements.CH_ratio": "C/H ratio",
    # "delivery.elements.H_oceans": "[H] [oceans]",
    # "star.mass": r"Stellar mass [M$_\odot$]"}
    # output_label_map = {
    # "solidification_time": "Solidification time [yr]",
    # "esc_rate_total": "Total escape rate [kg/s]",
    # "Phi_global": "Melt fraction [%]",
    # "P_surf": "Surface pressure [bar]",
    # "atm_kg_per_mol": "Atmospheric mass [kg/mol]"}
    # colormaps_by_param = {
    #     "orbit.semimajoraxis": cm.plasma,
    #     "escape.zephyrus.Pxuv": cm.cividis,
    #     "escape.zephyrus.efficiency": cm.spring,
    #     "outgas.fO2_shift_IW": cm.coolwarm,
    #     "delivery.elements.CH_ratio":cm.copper,
    #     #"atmos_clim.module": cm.Dark2,
    #     "delivery.elements.H_oceans": cm.winter,
    #     #"star.mass": cm.RdYlBu
    #     }

    # log_scale_grid_params = ["escape.zephyrus.Pxuv"]

    # generate_single_plots(
    # extracted_outputs=extracted_outputs,
    # grouped_data=grouped_data,
    # grid_params=grid_params,
    # plots_path=plots_path,
    # param_label_map=param_label_map,
    # colormaps_by_param=colormaps_by_param,
    # output_label_map=output_label_map,
    # log_scale_grid_params=log_scale_grid_params,
    # log_x=True,
    # plot_hist=True,
    # plot_kde=True,
    # cumulative=True,
    # bins=100)

    # generate_grid_plot(
    # extracted_outputs=extracted_outputs,
    # grouped_data=grouped_data,
    # grid_params=grid_params,
    # plots_path=plots_path,
    # param_label_map=param_label_map,
    # colormaps_by_param=colormaps_by_param,
    # output_label_map=output_label_map,
    # log_scale_grid_params=log_scale_grid_params)

    # print('-----------------------------------------------------')
    # print("All plots completed. Let's do some analyse now :) !")
    # print('-----------------------------------------------------')
