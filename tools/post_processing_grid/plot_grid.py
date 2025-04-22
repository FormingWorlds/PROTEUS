import pandas as pd
from io import StringIO
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatterMathtext

def load_extracted_data(csv_path):

    """
    Load extracted data from the CSV file generated with post_processing.py, returning a DataFrame for plotting.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing extracted data.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the extracted simulation data.

    grid_params : dict
        Dictionary of grid parameter names and their value lists.

    extracted_outputs : list of str
        List of extracted output variable names.
    """
    with open(csv_path, 'r') as f:
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
                output_values = subset[output].dropna().tolist()
                value_dict[param_value] = output_values

            grouped[key_name] = value_dict

    return grouped

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
    statuses = df['Status'].fillna('unknown').astype(str)
    status_counts = statuses.value_counts().sort_values(ascending=False)
    
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
    output_path = plot_dir+'grid_status_summary.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot grid_status_summary.png saved to {output_path}")

def kde_cumulative_linear(values, color, ax, bw_adjust=0.3, **kwargs):
    """
    Plot a cumulative KDE curve on the given x-axis.

    Parameters
    ----------
    values : array-like
        Data points on the x-axis used to calculate the KDE.

    color : str or matplotlib color
        The color of the KDE curve.

    ax : matplotlib.axes.Axes
        The axis on which to plot the KDE.

    bw_adjust : float, optional, default=0.3
        Bandwidth adjustment factor : controls the smoothness of the KDE. 
        Smaller values make the KDE more sensitive to data, while larger values smooth it out.

    **kwargs : keyword arguments, optional
        Additional arguments passed to `sns.kdeplot`, such as `label`, `linewidth`, etc.

    """
    sns.kdeplot(
        values,
        cumulative=True,
        bw_adjust=bw_adjust,
        clip=(np.min(values), np.max(values)),
        common_grid=True,
        color=color,
        ax=ax,
        **kwargs
    )

def kde_cumulative_log(values, color, ax, bw_adjust=0.3, **kwargs):
    """
    Plot a cumulative KDE curve on the given x-axis with a log scale for the x-axis.

    Parameters
    ----------
    values : array-like
        Data points on the x-axis used to calculate the KDE.

    color : str or matplotlib color
        The color of the KDE curve.

    ax : matplotlib.axes.Axes
        The axis on which to plot the KDE.

    bw_adjust : float, optional, default=0.3
        Bandwidth adjustment factor : controls the smoothness of the KDE. 
        Smaller values make the KDE more sensitive to data, while larger values smooth it out.

    **kwargs : keyword arguments, optional
        Additional arguments passed to `sns.kdeplot`, such as `label`, `linewidth`, etc.
    """
    # Plot the cumulative KDE
    sns.kdeplot(
        values,
        cumulative=True,
        bw_adjust=bw_adjust,
        clip=(np.min(values), np.max(values)),
        common_grid=True,
        color=color,
        ax=ax,
        **kwargs
    )
    
    # Set x-axis to log scale
    ax.set_xscale('log')

    # Format the x-axis to show powers of 10
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: r'$10^{{{}}}$'.format(int(np.log10(x)))))

def plot_kde_cumulative_linear(data_dict, xlabel, ylabel, cmap=plt.cm.plasma, vmin=None, vmax=None, key_label="Parameter", ax=None, save_path=None, bw_adjust=0.3, tick_values=None, norm=None):
    """
    Plot cumulative KDE curves for one of the output parameters of the grid (like esc_rate_total, solidification_time) on the x-axis.
    The different curves correspond to a input parameter from the grid with a color mapped on the right side of the plot.

    Parameters
    ----------
    data_dict : dict
        Dictionary of datasets, where each key represents a unique variable (e.g., semi-major axis, temperature),
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot each group
    for key in keys:
        values = np.array(data_dict[key])
        if len(values) < 2:
            continue
        color = cmap(norm(key))
        kde_cumulative_linear(values, color=color, ax=ax, bw_adjust=bw_adjust, label=str(key))

    ax.set_xlabel(xlabel, fontsize=12)
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

def plot_kde_cumulative_linear_clean(
    data_dict,
    xlabel,
    ylabel,
    cmap=plt.cm.plasma,
    vmin=None,
    vmax=None,
    key_label="Parameter",
    ax=None,
    save_path=None,
    bw_adjust=0.3,
    tick_values=None,
    norm=None,
    return_cbar=False  # New parameter
):
    """
    Plot cumulative KDE curves for one of the output parameters of the grid.
    Optionally return a colorbar scalar mappable to use for shared colorbar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    keys = sorted(data_dict.keys())
    if vmin is None:
        vmin = min(keys)
    if vmax is None:
        vmax = max(keys)

    # Use provided or default normalization
    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot each dataset
    for key in keys:
        values = np.array(data_dict[key])
        if len(values) < 2:
            continue
        color = cmap(norm(key))
        kde_cumulative_linear(values, color=color, ax=ax, bw_adjust=bw_adjust, label=str(key))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)

    # Only return colorbar smappable, don't draw it (caller will handle that)
    if return_cbar:
        sm.set_array(np.linspace(vmin, vmax, 100))  # Needed for colorbar
        sm.key_label = key_label  # Store label for later
        sm.tick_values = tick_values  # Custom attribute to help outside
        return sm

    # Otherwise, draw internal colorbar normally
    sm.set_array(np.linspace(vmin, vmax, 100))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(key_label, fontsize=12)

    if tick_values is not None:
        cbar.set_ticks(tick_values)
        if isinstance(norm, mcolors.LogNorm):
            cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in tick_values])
            cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        else:
            cbar.set_ticklabels([str(t) for t in tick_values])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    return None

def plot_kde_cumulative_log(data_dict, xlabel, ylabel, cmap=plt.cm.plasma, vmin=None, vmax=None, key_label="Parameter", ax=None, save_path=None, bw_adjust=0.3, tick_values=None, norm=None):
    """
    Plot cumulative KDE curves for one of the output parameters of the grid (like esc_rate_total, solidification_time) on the x-axis.
    The different curves correspond to a input parameter from the grid with a color mapped on the right side of the plot.

    Parameters
    ----------
    data_dict : dict
        Dictionary of datasets, where each key represents a unique variable (e.g., semi-major axis, temperature),
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot each group
    for key in keys:
        values = np.array(data_dict[key])
        if len(values) < 2:
            continue
        color = cmap(norm(key))
        kde_cumulative_log(values, color=color, ax=ax, bw_adjust=bw_adjust, label=str(key))

    ax.set_xlabel(xlabel, fontsize=12)
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

def generate_single_plots_linear(extracted_outputs, grouped_data, grid_params, plots_path, param_label_map, colormaps_by_param, output_label_map, log_scale_grid_params):
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

            save_dir = Path(plots_path) / 'single_plot_linear'
            plot_dir_exists(save_dir)

            save_name = f"cumulative_{output_name}_vs_{param.replace('.', '_')}.png"
            save_path = save_dir / save_name

            fig, ax = plt.subplots(figsize=(10, 6))

            norm = (
                mcolors.LogNorm(vmin=vmin, vmax=vmax)
                if param in log_scale_grid_params
                else mcolors.Normalize(vmin=vmin, vmax=vmax)
            )

            plot_kde_cumulative_linear(
                data_dict=data_dict,
                xlabel=xlabel,
                ylabel=ylabel,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                key_label=key_label,
                tick_values=tick_values,
                save_path=save_path,
                norm=norm,
                ax=ax
            )
            plt.close(fig)

    print("All linear-scale single plots saved.")

def generate_single_plots_log(extracted_outputs, grouped_data, grid_params, plots_path, param_label_map, colormaps_by_param, output_label_map, log_scale_grid_params):
    """
    Generate and save normalized cumulative distribution plots for each output vs. grid parameter. The x-axis is in log scale.

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

            save_dir = Path(plots_path) / 'single_plot_log'
            plot_dir_exists(save_dir)

            save_name = f"cumulative_{output_name}_vs_{param.replace('.', '_')}_log.png"
            save_path = save_dir / save_name

            fig, ax = plt.subplots(figsize=(10, 6))

            norm = (
                mcolors.LogNorm(vmin=vmin, vmax=vmax)
                if param in log_scale_grid_params
                else mcolors.Normalize(vmin=vmin, vmax=vmax)
            )

            plot_kde_cumulative_log(
                data_dict=data_dict,
                xlabel=xlabel,
                ylabel=ylabel,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                key_label=key_label,
                tick_values=tick_values,
                save_path=save_path,
                norm=norm,
                ax=ax
            )

            plt.close(fig)

    print("All log-scale single plots saved.")

def generate_grid_plot(extracted_outputs, grouped_data, grid_params, plots_path, param_label_map, colormaps_by_param, output_label_map, log_scale_grid_params):
    """
    Generate and save normalized cumulative distribution plots for each output vs. grid parameters.
    This creates a subplot where each column corresponds to one extracted output, and each row corresponds to one grid parameter.

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
    
    num_cols = len(extracted_outputs)
    num_rows = len(grid_params)
    
    # Create subplots with the appropriate layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 16, num_rows * 12))
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # Make sure it's 2D if only one row.
    
    for i, output_name in enumerate(extracted_outputs):
        for j, (param, cmap) in enumerate(colormaps_by_param.items()):
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

            # Select the axis for this subplot
            ax = axes[j, i]

            norm = (
                mcolors.LogNorm(vmin=vmin, vmax=vmax)
                if param in log_scale_grid_params
                else mcolors.Normalize(vmin=vmin, vmax=vmax)
            )
            
            plot_kde_cumulative_linear(
                data_dict=data_dict,
                xlabel=xlabel,
                ylabel=ylabel,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                key_label=key_label,
                tick_values=tick_values,
                save_path=None,  # Don't save the plot yet
                norm=norm,
                ax=ax
            )
            
            # Set titles for each subplot
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if j == num_rows - 1:
                ax.set_xlabel(xlabel, fontsize=12)
            
            # Customize plot with grid and ticks
            ax.grid(alpha=0.2)
            ax.set_ylim(0, 1.02)
    
    # Save the complete subplot figure
    save_dir = Path(plots_path) / 'grid_plot'
    plot_dir_exists(save_dir)
    
    save_name = "cumulative_grid_plot_linear.png"
    save_path = save_dir / save_name
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"All subplot plots saved to {save_path}")

def generate_grid_plot_clean(extracted_outputs, grouped_data, grid_params, plots_path, param_label_map, colormaps_by_param, output_label_map, log_scale_grid_params):
    """
    Generate and save normalized cumulative distribution plots for each output vs. grid parameters.
    This creates a subplot where each column corresponds to one extracted output, and each row corresponds to one grid parameter.

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
    if param_label_map is None or colormaps_by_param is None or output_label_map is None:
        raise ValueError("param_label_map, colormaps_by_param, and output_label_map must be provided.")
    if log_scale_grid_params is None:
        log_scale_grid_params = []

    num_cols = len(extracted_outputs)
    num_rows = len(grid_params)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    param_list = list(grid_params.keys())

    for i, output_name in enumerate(extracted_outputs):
        for j, param in enumerate(param_list):
            data_key = f"{output_name}_per_{param}"
            cmap = colormaps_by_param[param]

            if data_key not in grouped_data:
                print(f"WARNING: Skipping {data_key} — not found in grouped_data")
                continue

            data_dict_raw = grouped_data[data_key]
            data_dict = {k: [v_i * 100 if output_name == "Phi_global" else v_i for v_i in v] for k, v in data_dict_raw.items()}

            tick_values = grid_params.get(param)
            vmin, vmax = min(tick_values), max(tick_values)

            xlabel = output_label_map.get(output_name, output_name.replace("_", " ").title())
            norm = (
                mcolors.LogNorm(vmin=vmin, vmax=vmax)
                if param in log_scale_grid_params
                else mcolors.Normalize(vmin=vmin, vmax=vmax)
            )

            ax = axes[j, i]
            cbar = plot_kde_cumulative_linear_clean(
                data_dict=data_dict,
                xlabel="",
                ylabel="",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                key_label=None,
                tick_values=tick_values,
                save_path=None,
                norm=norm,
                ax=ax,
                return_cbar=True
            )

            if i != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if j != num_rows - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ax.grid(alpha=0.2)
            ax.set_ylim(0, 1.02)

    # Add shared y-axis label
    fig.text(0.04, 0.5, "Normalized cumulative fraction of simulations", va='center', rotation='vertical', fontsize=16)

    # Add x-axis labels under each column
    for i, output_name in enumerate(extracted_outputs):
        xlabel = output_label_map.get(output_name, output_name.replace("_", " ").title())
        axes[-1, i].set_xlabel(xlabel, fontsize=12)

    # Create a single colorbar on the right side
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    fig.colorbar(cbar, cax=cbar_ax, orientation='vertical', label="Parameter value")

    # Save plot
    save_dir = Path(plots_path) / 'grid_plot'
    plot_dir_exists(save_dir)
    save_path = save_dir / "cumulative_grid_plot_linear_clean.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"All subplot plots saved to {save_path}")

if __name__ == '__main__':

    # Paths to the csv file and plot directory
    grid_name = 'escape_grid_4_params_Pxuv_a_epsilon_fO2'
    data_dir = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_processed_data/{grid_name}/{grid_name}_extracted_data.csv'
    plots_path = f'/home2/p315557/PROTEUS/tools/post_processing_grid/nogit_plots/{grid_name}/'

    # Load the data and check if the plot directory exists
    df, grid_params, extracted_outputs = load_extracted_data(data_dir)
    plot_dir_exists(plots_path)
    # Group extracted outputs by grid parameters
    grouped_data = group_output_by_parameter(df, grid_params, extracted_outputs)

    # Plot the grid status
    plot_grid_status(df, plots_path)      

    # Single plots 
    param_label_map = {
    "orbit.semimajoraxis": "Semi-major axis [AU]",
    "escape.zephyrus.Pxuv": r"$P_{XUV}$ [bar]",
    "escape.zephyrus.efficiency": r"Escape efficiency factor, $\epsilon$",
    "outgas.fO2_shift_IW":r"$log_{10}$($fO_2$) ($\Delta$ IW)"}   
    output_label_map = {
    "solidification_time": "Solidification time [yr]",
    "esc_rate_total": "Total escape rate [kg/s]",
    "Phi_global": "Melt fraction [%]",
    "P_surf": "Surface pressure [bar]",
    "atm_kg_per_mol": "Atmospheric mass [kg/mol]"}
    colormaps_by_param = {
        "orbit.semimajoraxis": cm.plasma,
        "escape.zephyrus.Pxuv": cm.cividis,
        "escape.zephyrus.efficiency": cm.spring,
        "outgas.fO2_shift_IW": cm.coolwarm}

    log_scale_grid_params = ["escape.zephyrus.Pxuv"]

    # generate_single_plots_linear(
    # extracted_outputs=extracted_outputs,
    # grouped_data=grouped_data,
    # grid_params=grid_params,
    # plots_path=plots_path,
    # param_label_map=param_label_map,
    # colormaps_by_param=colormaps_by_param,
    # output_label_map=output_label_map,
    # log_scale_grid_params=log_scale_grid_params)

    # generate_single_plots_log(
    # extracted_outputs=extracted_outputs,
    # grouped_data=grouped_data,
    # grid_params=grid_params,
    # plots_path=plots_path,
    # param_label_map=param_label_map,
    # colormaps_by_param=colormaps_by_param,
    # output_label_map=output_label_map,
    # log_scale_grid_params=log_scale_grid_params)

    generate_grid_plot(
    extracted_outputs=extracted_outputs,
    grouped_data=grouped_data,
    grid_params=grid_params,
    plots_path=plots_path,
    param_label_map=param_label_map,
    colormaps_by_param=colormaps_by_param,
    output_label_map=output_label_map,
    log_scale_grid_params=log_scale_grid_params)