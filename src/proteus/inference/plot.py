"""Visualization utilities for asynchronous Bayesian optimization.

This module provides functions to visualize worker execution timelines,
timing distributions, and optimization performance metrics.

Functions:
    plot_times: Plot per-worker task timelines and histograms of timing metrics.
    plot_res: Plot regret and best observed value vs. time and iteration.
    plot_result_observables: Plot the best-fit final observables against targets.
    plot_result_parameters: Plot the best-fit parameters against known true values.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import torch
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from proteus import Proteus
from proteus.inference.failures import read_failure_records
from proteus.inference.objective import EPS_CLIP, eval_obj
from proteus.inference.transforms import unnormalize_parameters
from proteus.inference.utils import get_obs
from proteus.plot import plot_dispatch
from proteus.utils.coupler import (
    HelpfileFormatError,
    read_helpfile_table,
    variable_is_logarithmic,
)
from proteus.utils.helper import recursive_get

log = logging.getLogger('fwl.' + __name__)

dtype = torch.double
fmt = 'png'
dpi = 300

# Colourblind-friendly palette (Wong).
WONG = {
    'orange': '#E69F00',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
}

# Plot standards for the observable comparison
OBS_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.axisbelow': True,
}

# Range of the simulated/target ratio axis.
RATIO_CLIP = (1e-2, 1e2)


def plots_perf_timeline(logs, directory, n_init, min_text_width=0.88):
    """Generate timeline and histograms of process durations.

    This function makes multiple plots

    Parameters
    ----------
    - logs (list[dict]): Log entries containing timing and evaluation data.
    - directory (str): Base directory where plots are saved ("plots/" appended).
    - n_init (int): Number of initial evaluations to skip when plotting.
    - min_text_width (float): Minimum bar width threshold for white text.

    Returns
    ----------
    - None
    """
    # Build DataFrame skipping initial entries
    df = pd.DataFrame(logs[n_init:])
    if df.empty:
        log.debug('No logs to display.')
        return

    # Shift timestamps so earliest start_time is zero
    global_t0 = df['start_time'].min()
    df['start'] = df['start_time'] - global_t0
    df['end'] = df['end_time'] - global_t0

    # Identify unique workers and assign colors
    workers = sorted(df['worker'].unique())
    color_map = {}
    for i, w in enumerate(workers):
        if i < 10:
            color_map[w] = plt.cm.tab10(i)
        else:
            color_map[w] = np.clip(np.random.random_sample(3), a_min=0.05, a_max=0.95)

    # Find bar widths and the rightmost endpoint
    bar_widths = df['end'] - df['start']
    min_bar_width = bar_widths.min()
    max_bar_end = df['end'].max()

    # If narrowest bar is too small, stretch the x-axis but never move the bars
    stretch_needed = max(0, min_text_width - min_bar_width)
    # The x-axis must go at least as far as the rightmost bar end
    xlim_max = max_bar_end + stretch_needed

    # Create timeline plot
    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(workers)))
    for _, row in df.iterrows():
        bar_start = row['start']
        bar_end = row['end']
        bar_width = bar_end - bar_start
        bar_center = (bar_start + bar_end) / 2
        worker_y = row['worker']
        # Format annotation text for the bar
        if len(row['x_value']) == 1:
            txt = f'(x, y)\n= ({row["x_value"][0]:.2f},{row["y_value"]:.2f})\n{row["duration"]:.2f}s'
        else:
            txt = f'y = {row["y_value"]:.2f}\n{row["duration"]:.2f}s'
        # Draw the bar
        ax.broken_barh(
            [(bar_start, bar_width)],
            (worker_y - 0.4, 0.8),
            facecolors=color_map[row['worker']],
            edgecolor='k',
        )
        # Mark end of bar
        ax.vlines(
            bar_end, worker_y - 0.5, worker_y + 0.5, color='gray', lw=1, linestyles='dashed'
        )
        # Place text inside bar
        ax.text(
            bar_center,
            worker_y,
            txt,
            va='center',
            ha='center',
            fontsize=10,
            color='white' if bar_width > 0.25 else 'black',
            fontweight='bold',
        )
    # Label axes and grid
    ax.set_yticks(workers)
    ax.set_yticklabels([f'Worker {w}' for w in workers])
    ax.set_xlabel('Wall clock time (s)')
    ax.set_title('Parallel workers')
    ax.grid(True, axis='x', alpha=0.3)

    # Set limits and save figure
    padding = 0.05 * xlim_max
    xlim_max_padded = xlim_max + padding

    ax.set_xlim(left=0, right=xlim_max_padded)
    plt.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'perf_parallel.{fmt}'), dpi=dpi, bbox_inches='tight'
    )
    plt.close(fig)

    # Histogram of total durations
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(df['duration'], bins='auto', color='cornflowerblue', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Process Times', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(directory, 'plots', f'perf_timehist.{fmt}'), dpi=dpi, bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram of BO times
    ax.hist(df['BO_time'], bins='auto', color='cornflowerblue', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of BO Times', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(directory, 'plots', f'perf_BO_timehist.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)

    # Histogram of evaluation times
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(df['t_eval'], bins='auto', color='cornflowerblue', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Evaluation Times', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(directory, 'plots', f'perf_eval_timehist.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)

    # Colored histogram for fit times
    values = df['t_fit'].values
    positions = np.arange(len(values))  # 0, 1, 2, …, len(df)-1

    num_bins = 30
    counts, bin_edges = np.histogram(values, bins=num_bins)
    bin_indices = np.digitize(values, bin_edges[:-1], right=False)

    avg_positions = []
    for b in range(1, num_bins + 1):
        pos_in_bin = positions[bin_indices == b]
        if pos_in_bin.size:
            avg_positions.append(pos_in_bin.mean())
        else:
            avg_positions.append(0)  # or np.nan if you prefer

    norm = mcolors.Normalize(vmin=min(avg_positions), vmax=max(avg_positions))
    cmap = cm.viridis
    colors = [cmap(norm(p)) for p in avg_positions]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_bins):
        ax.bar(
            bin_edges[i],
            counts[i],
            width=bin_edges[i + 1] - bin_edges[i],
            color=colors[i],
            align='edge',
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy mappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Row Index in Bin')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Fit Times', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(directory, 'plots', f'perf_fit_timehist.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)

    # Colored histogram for acquisition times
    values = df['t_ac'].values
    positions = np.arange(len(values))  # 0, 1, 2, …, len(df)-1

    num_bins = 30
    counts, bin_edges = np.histogram(values, bins=num_bins)
    bin_indices = np.digitize(values, bin_edges[:-1], right=False)

    avg_positions = []
    for b in range(1, num_bins + 1):
        pos_in_bin = positions[bin_indices == b]
        if pos_in_bin.size:
            avg_positions.append(pos_in_bin.mean())
        else:
            avg_positions.append(0)

    norm = mcolors.Normalize(vmin=min(avg_positions), vmax=max(avg_positions))
    cmap = cm.viridis
    colors = [cmap(norm(p)) for p in avg_positions]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_bins):
        ax.bar(
            bin_edges[i],
            counts[i],
            width=bin_edges[i + 1] - bin_edges[i],
            color=colors[i],
            align='edge',
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy mappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Row Index in Bin')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Acquisition Times', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(directory, 'plots', f'perf_acquisition_timehist.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)

    # Scatter plot of distance to busy locations
    fig, ax = plt.subplots(figsize=(7, 4))
    df_f = df[df['dist'].notnull()]
    ax.scatter(
        df_f.index,
        df_f['dist'],
        marker='o',
        linestyle='-',
        color='cornflowerblue',
        alpha=0.8,
        label='Distance',
    )

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Query Distance to Busy Location', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(
        os.path.join(directory, 'plots', f'perf_distance_iters.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def plots_perf_converge(D, T, n_init, directory):
    """Plot regret and best observed value over time and iterations.

    Parameters
    ----------
    - D (dict): Contains 'Y' list of objective values.
    - T (list): Elapsed times corresponding to each evaluation.
    - n_init (int): Number of initial evaluations to skip.
    - directory (str): Base dir where "plots/" subfolder will be created.

    Returns
    ----------
    - None
    """

    Y = np.array(D['Y'], copy=None, dtype=float).flatten()  # Flatten in case it's (N,1)
    Y = Y[n_init:]

    y_best = Y[0]
    Y_best = [y_best]

    for i in range(1, len(Y)):
        if Y[i] > y_best:
            y_best = Y[i]

        Y_best.append(y_best)

    Y_best = np.array(Y_best)
    T = np.array(T)  # Assume T aligns with Y

    oracle = 10.0  # Change as appropriate
    regret = np.abs(Y_best - oracle)
    log_regret = np.log10(regret + 1e-12)  # add small number to avoid log(0)
    n = np.arange(len(Y_best))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # Top: log regret vs t
    axes[0].plot(T, log_regret, marker='o')
    axes[0].set_ylabel('log10(Regret)')
    axes[0].set_xlabel('Time [seconds]')
    axes[0].set_title('Log Regret vs Time')
    axes[0].grid(True)
    # axes[0].legend()

    # Bottom: log regret vs n
    axes[1].plot(n, log_regret, marker='o', color='tab:orange')
    axes[1].set_xlabel('Step number')
    axes[1].set_ylabel('log10(Regret)')
    axes[1].set_title('Log Regret vs Step')
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # axes[1].legend()

    plt.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'perf_regret.{fmt}'), dpi=dpi, bbox_inches='tight'
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    ymax = max(1.0, np.amax(Y_best)) + 0.1
    ymin = min(0.0, np.amin(Y_best)) - 0.1

    # Top: log regret vs t
    axes[0].plot(T, Y_best, marker='o')
    axes[0].set_ylabel('Best value of objective')
    axes[0].set_xlabel('Time [seconds]')
    axes[0].set_title('Best Value vs Time')
    axes[0].set_ylim(ymin, ymax)
    axes[0].grid(True)

    # Bottom: log regret vs n
    axes[1].plot(n, Y_best, marker='o', color='tab:orange')
    axes[1].set_xlabel('Step number')
    axes[1].set_ylabel('Best value of objective')
    axes[1].set_title('Best Value vs Step')
    axes[1].set_ylim(ymin, ymax)
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'perf_bestval.{fmt}'), dpi=dpi, bbox_inches='tight'
    )
    plt.close(fig)


def plot_result_objective(D, parameters, n_init, directory, yclip=-12):
    """Plot objective function at each sample that was created.

    Parameters
    ----------
    - D (dict): Contains 'X' and 'Y' lists.
    - parameters (dict): Parameter names and bounds.
    - n_init (int): Number of initial evaluations (to be highlighted).
    - directory (str): Base dir where "plots/" subfolder will be created.
    - yclip (float): Minimum limit on y-axis objective values.

    Returns
    ----------
    - None
    """

    # Get objective function values
    Y = np.array(D['Y'], copy=None, dtype=float).flatten()

    # Best point
    i_best = np.argmax(Y)

    # Clamp values
    mask = Y < yclip
    Y = np.clip(Y, a_min=yclip, a_max=None)

    # Y label
    if np.any(mask):
        ylbl = f'Value of objective\nclipped to J>{yclip}'
    else:
        ylbl = 'Value of objective'

    # Get bounds
    keys = list(parameters.keys())
    d = len(keys)  # number of parameters
    bounds = torch.tensor(
        [[list(parameters.values())[i][j] for i in range(d)] for j in range(2)]
    )

    # Un-normalise X data
    X = unnormalize_parameters(torch.tensor(D['X']), bounds, keys)
    X = np.array(X, copy=None, dtype=float)

    # Colors
    C = np.full_like(Y, 'k', dtype=str)
    C[:n_init] = 'c'
    C[i_best] = 'm'

    # Limits
    ymax = max(1.0, np.amax(Y)) + 0.1
    ymin = min(0.0, np.amin(Y)) - 0.1

    # Plot
    fig, axs = plt.subplots(2, d, figsize=(2.7 * d, 3.2))
    axs[0, 0].set_ylabel('Histogram')
    axs[1, 0].set_ylabel(ylbl)

    # plot scatter points
    for i in range(d):
        # clipped points
        axs[1, i].scatter(
            X[mask, i],
            Y[mask],
            c=list(C[mask]),
            s=11,
            alpha=0.8,
            zorder=4,
            marker='v',
            edgecolors='none',
        )

        # unclipped points
        axs[1, i].scatter(
            X[~mask, i],
            Y[~mask],
            c=list(C[~mask]),
            s=10,
            alpha=0.6,
            zorder=5,
            marker='o',
            edgecolors='none',
        )

        # configure axes
        axs[1, i].set_xlabel(keys[i], fontsize=10)
        if variable_is_logarithmic(keys[i]):
            axs[1, i].set_xscale('log')
        axs[1, i].grid(alpha=0.2, zorder=0)
        axs[1, i].set_ylim(ymin, ymax)
        if i >= 1:
            axs[1, i].set_yticklabels([])
        axs[0, i].set_yticklabels([])
        axs[0, i].set_xticklabels([])

    # plot histograms
    for i in range(d):
        x1 = X[:n_init, i]  # only initial values
        x2 = X[n_init:, i]  # after initial values
        axs[0, i].hist(
            [x2, x1], bins=11, stacked=True, histtype='barstacked', color=['k', 'c'], zorder=2
        )

        # median and stddev
        x_med = np.median(x2)
        x_err = np.std(x2) / len(x2) ** 0.5

        # overplot median in both panels
        for j in (0, 1):
            axs[j, i].axvline(x=x_med, zorder=4, color='r', alpha=0.8)
            axs[j, i].axvline(
                x=x_med + x_err, zorder=4, color='r', alpha=0.5, linestyle='dashed'
            )
            axs[j, i].axvline(
                x=x_med - x_err, zorder=4, color='r', alpha=0.5, linestyle='dashed'
            )

        # overplot best in both panels
        x_best = X[i_best, i]
        for j in (0, 1):
            axs[j, i].axvline(x=x_best, zorder=5, color='m', alpha=0.8)

        title = f'{x_best:g}'
        if i == 0:
            title = f'Best: {x_best:g}'
        axs[0, i].set_title(title, fontsize=8, color='m', weight='bold')

        # grid
        axs[j, i].grid(alpha=0.2, zorder=0, axis='x')

    # save plot
    fig.subplots_adjust(wspace=0.012, hspace=0.022)
    fig.savefig(
        os.path.join(directory, 'plots', f'result_objective.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_result_correlation(pars: dict, obs: dict, directory):
    """Plot correlation between observables and parameters.

    This requires reading output-data files from the disk.

    Parameters
    ----------
    - pars (dict): Parameter names and bounds.
    - obs (dict): Observable names and target values.
    - directory (str): Base dir where the inference was performed.

    Returns
    ----------
    - None
    """

    # Convert to lists
    par_keys = list(pars.keys())
    obs_keys = list(obs.keys())

    # Get directories for all cases of interest. Filtered to directories only:
    # a worker's `_console.log` capture file (or any other stray sibling) also
    # matches the `i_*` glob but is not a case directory.
    cases = sorted(p for p in (Path(directory) / 'workers').glob('w_*/i_*') if p.is_dir())

    # Extract parameters and observables
    X, Y = [], []
    for c in cases:
        # A case that died during start-up has a folder but did not get as far
        # as writing its resolved config, and so has nothing to plot.
        conf_path = c / 'init_coupler.toml'
        if not conf_path.is_file():
            log.warning(f'Missing init_coupler.toml for {c}')
            continue
        conf = toml.load(conf_path)

        # Check success
        hf_path = c / 'runtime_helpfile.csv'
        if not hf_path.is_file():
            log.warning(f'Missing helpfile for {c}')
            continue

        # Read helpfile for observables
        try:
            help = read_helpfile_table(hf_path, min_rows=1)
        except (OSError, HelpfileFormatError) as err:
            log.warning(f'Unreadable helpfile for {c}: {err}')
            continue

        # Get parameters and observables
        xx = [recursive_get(conf, k.split('.')) for k in par_keys]
        yy = list(help.iloc[-1][obs_keys].T)

        # Store these
        X.append(xx)
        Y.append(yy)
    # Axes
    n_par = len(par_keys)
    n_obs = len(obs_keys)

    # Shaped explicitly, so that a study in which no case produced output still
    # has a column per parameter and observable to index.
    X = np.array(X, dtype=float).reshape(-1, n_par)
    Y = np.array(Y, dtype=float).reshape(-1, n_obs)

    # Make plot
    fig, axs = plt.subplots(n_obs, n_par, figsize=(2.7 * n_par, 2.7 * n_obs))
    for i in range(n_par):
        for j in range(n_obs):
            # handle axis
            if n_par == 1 and n_obs == 1:
                ax = axs
            elif n_par == 1:
                ax = axs[j]
            elif n_obs == 1:
                ax = axs[i]
            else:
                ax = axs[j, i]

            # plot data
            xx = X[:, i]
            yy = Y[:, j]
            ax.scatter(xx, yy, color='k', alpha=0.8, s=8, zorder=4)

            # axis grid
            ax.grid(alpha=0.2, zorder=0)

            # observables
            ax.axhline(y=obs[obs_keys[j]], color='g', alpha=0.5, label='Observed')

            # these variables are more natural on a log-scale
            if variable_is_logarithmic(obs_keys[j]):
                ax.set_yscale('log')
            if variable_is_logarithmic(par_keys[i]):
                ax.set_xscale('log')

            # hide tick labels
            if i >= 1:
                ax.set_yticklabels([])
            if j < n_obs - 1:
                ax.set_xticklabels([])

    # Legend
    if n_obs == 1 or n_par == 1:
        axs[0].legend()
    else:
        axs[0, 0].legend()

    # Axis labels
    if n_par == 1 or n_obs == 1:
        axs[0].set_xlabel(par_keys[0], fontsize=10)
    else:
        for i in range(n_par):
            axs[-1, i].set_xlabel(par_keys[i], fontsize=10)

    if n_obs == 1 or n_par == 1:
        axs[0].set_ylabel(obs_keys[0], fontsize=10)
    else:
        for j in range(n_obs):
            axs[j, 0].set_ylabel(obs_keys[j], fontsize=10)

    # Decorate
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(
        os.path.join(directory, 'plots', f'result_correlation.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def _collect_case_observables(directory, obs: dict, sigma: dict | None = None) -> pd.DataFrame:
    """Read the final observables and fit quality of every case on disk.

    The objective is recomputed from the stored helpfile rather than read back
    from the optimiser.

    Returns one row per readable case, with `worker`, `iter`, `case`,
    `excluded`, one column per observable, and `J`.
    """
    obs_keys = list(obs.keys())

    excluded = {
        (int(rec['worker']), int(rec['iter'])) for rec in read_failure_records(directory)
    }

    # Filtered to directories only
    cases = sorted(p for p in (Path(directory) / 'workers').glob('w_*/i_*') if p.is_dir())

    rows = []
    for case in cases:
        hf_path = case / 'runtime_helpfile.csv'
        if not hf_path.is_file():
            log.warning(f'Missing helpfile for {case}')
            continue
        sim = get_obs(hf_path, obs_keys)
        sim = {k: float(sim[k]) for k in obs_keys}
        w, i = int(case.parent.name.split('_')[1]), int(case.name.split('_')[1])
        rows.append(
            {
                'worker': w,
                'iter': i,
                'case': f'w{w}_i{i}',
                'excluded': (w, i) in excluded,
                **sim,
                'J': float(eval_obj(sim, obs, sigma)),
            }
        )

    return pd.DataFrame(rows)


def _best_case(ok: pd.DataFrame, best_config=None) -> pd.Series:
    """Pick the row of the case the results summary reported as the best fit."""
    if best_config is not None:
        case = Path(best_config).parent
        try:
            w, i = int(case.parent.name.split('_')[1]), int(case.name.split('_')[1])
        except (IndexError, ValueError):
            log.warning(f'Cannot read case indices from {best_config}; using the best J')
        else:
            match = ok[(ok['worker'] == w) & (ok['iter'] == i)]
            if len(match):
                return match.iloc[0]
            log.warning(f'Case w_{w}/i_{i} is not among the scored cases; using the best J')
    return ok.loc[ok['J'].idxmax()]


def _panel_ratio(ax, ok: pd.DataFrame, obs: dict, fit: np.ndarray, best_case: str) -> None:
    """Draw every scored case as its ratio to the target observable.

    A target of zero or below has no meaningful ratio so such a row is left empty.
    """
    obs_keys = list(obs.keys())
    y = np.arange(len(obs_keys))[::-1]
    lo, hi = RATIO_CLIP
    # One generator for the whole panel
    rng = np.random.default_rng(42)
    drawn: set = set()

    for j, k in enumerate(obs_keys):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = ok[k].to_numpy(dtype=float) / float(obs[k])
        yy = np.full(r.shape, float(y[j])) + rng.normal(0, 0.07, r.size)
        finite = np.isfinite(r)
        # Three groups per row.
        # Each group takes its legend entry from the first row to populate it.
        for sel, edge, marker, size, shade, text in (
            (finite & (r >= lo) & (r <= hi), None, 'o', 10, '0.72', 'completed cases'),
            (finite & (r < lo), lo * 1.12, '<', 26, '0.45', f'below {lo:g}x target'),
            (finite & (r > hi), hi / 1.12, '>', 26, '0.45', f'above {hi:g}x target'),
        ):
            if not sel.any():
                continue
            ax.scatter(
                r[sel] if edge is None else np.full(int(sel.sum()), edge),
                yy[sel],
                s=size,
                marker=marker,
                color=shade,
                edgecolors='none',
                zorder=2,
                label=None if text in drawn else text,
            )
            drawn.add(text)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_best = fit / np.array([obs[k] for k in obs_keys], dtype=float)

    ax.axvline(1.0, color='k', lw=1.3, zorder=3, label='target')
    ax.scatter(
        ratio_best,
        y,
        s=100,
        marker='D',
        color=WONG['vermillion'],
        edgecolors='k',
        linewidths=0.6,
        zorder=4,
        label=f'best fit ({best_case})',
    )
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, obs_keys)
    ax.set_xlabel('simulated / target  (dimensionless)')
    ax.set_title('Final observables normalised by target', fontsize=11)
    ax.grid(axis='x', color='0.9', lw=0.6)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.92)


def _panel_residual(ax, obs_keys: list[str], pct: np.ndarray, best_J: float) -> None:
    """Draw the signed percent residual of the best-fit case, one bar per row."""
    y = np.arange(len(obs_keys))[::-1]
    ax.barh(
        y,
        pct,
        color=[WONG['blue'] if p < 0 else WONG['orange'] for p in pct],
        edgecolor='k',
        linewidth=0.5,
        height=0.6,
        zorder=3,
    )
    ax.axvline(0.0, color='k', lw=1.3, zorder=4)

    # A single observable leaves no spread to scale the label offsets and the axis limits by.
    span = float(pct.max() - pct.min())
    if span <= 0.0:
        span = max(float(np.abs(pct).max()), 1.0)

    for j, p in enumerate(pct):
        ax.text(
            p + np.sign(p) * 0.025 * span,
            y[j],
            f'{p:+.1f}%',
            va='center',
            ha='left' if p >= 0 else 'right',
            fontsize=9,
            zorder=5,
        )

    ax.set_xlim(pct.min() - 0.30 * span, pct.max() + 0.22 * span)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, obs_keys)
    ax.set_xlabel('(best fit - target) / target  [%]')
    ax.set_title(f'Best-fit residuals, J = {best_J:+.4f}', fontsize=11)
    ax.grid(axis='x', color='0.9', lw=0.6)


def plot_result_observables(obs: dict, directory, best_config=None, sigma=None):
    """Plot the best-fit final observables against the target observables.

    Two panels share the observable rows: every scored case as its ratio to
    the target, and the signed percent residual of the best-fit case.

    Parameters
    ----------
    - obs (dict): Observable names and target values.
    - directory (str): Base dir where the inference was performed.
    - best_config (str | Path | None): Path to the best fitting case's config
      TOML.
    - sigma (dict | None): Uncertainty of each observable

    Returns
    ----------
    - None
    """
    obs_keys = list(obs.keys())

    df = _collect_case_observables(directory, obs, sigma)
    if df.empty:
        log.warning('No case produced observables; skipping the observable comparison')
        return

    ok = df[~df['excluded']].copy()
    if ok.empty:
        log.warning('Every case carries the failure score; skipping the observable comparison')
        return

    best = _best_case(ok, best_config)

    tgt = np.array([obs[k] for k in obs_keys], dtype=float)
    fit = np.array([best[k] for k in obs_keys], dtype=float)

    # Same denominator convention as the printed results table
    denom = np.where(tgt >= 0, tgt + EPS_CLIP, tgt - EPS_CLIP)
    pct = 100.0 * (fit - tgt) / denom

    height = max(3.2, 0.7 * len(obs_keys) + 1.4)
    with plt.rc_context(OBS_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(12.0, height), width_ratios=[1.15, 1.0])

        _panel_ratio(axs[0], ok, obs, fit, str(best['case']))
        _panel_residual(axs[1], obs_keys, pct, float(best['J']))

        fig.suptitle(f'{Path(directory).name}: best-fit vs target observables', fontsize=12.5)
        fig.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'result_observables.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def _position_in_range(x, key: str, bounds) -> np.ndarray:
    """Place values of parameter `key` within `bounds`, 0 at the lower and 1 at the upper.

    Log-scaled parameters are placed in log10, as the optimiser normalises them.
    """
    f = np.log10 if variable_is_logarithmic(key) else np.asarray
    lo, hi = f(float(bounds[0])), f(float(bounds[1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        return (f(np.asarray(x, dtype=float)) - lo) / (hi - lo)


def _panel_position(ax, pos, truth_pos, best_pos, labels, best_case: str) -> None:
    """Draw every scored case (one column of `pos` per row), the truth and the best fit."""
    y = np.arange(len(labels))[::-1]
    jitter = np.random.default_rng(42).normal(0, 0.07, pos.shape)
    ax.scatter(
        pos,
        y + jitter,
        s=10,
        color='0.72',
        edgecolors='none',
        zorder=2,
        label='completed cases',
    )
    ax.scatter(
        truth_pos, y, s=300, marker='|', color='k', linewidths=2.2, zorder=3, label='truth'
    )
    ax.scatter(
        best_pos,
        y,
        s=100,
        marker='D',
        color=WONG['vermillion'],
        edgecolors='k',
        linewidths=0.6,
        zorder=4,
        label=f'best fit ({best_case})',
    )

    # Widen past [0, 1] only as far as an out-of-range truth needs
    ax.set_xlim(min(0.0, truth_pos.min()) - 0.04, max(1.0, truth_pos.max()) + 0.04)
    for edge in (0.0, 1.0):
        ax.axvline(edge, color='0.5', lw=0.8, ls='--', zorder=1)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, labels)
    ax.set_xlabel('position within sampled range  (log10 for log-scaled parameters)')
    ax.set_title('Parameters within their sampled range', fontsize=11)
    ax.grid(axis='x', color='0.9', lw=0.6)
    # Below the axis, since every row spans the full width and would be covered
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)


def plot_result_parameters(
    pars: dict, truth: dict, obs: dict, directory, best_config=None, sigma=None
):
    """Plot the best-fit parameters against the true parameters, if known.

    Parameters
    ----------
    - pars (dict): Parameter names and bounds.
    - truth (dict): True value of each parameter.
    - obs (dict): Observable names and target values, used to score cases.
    - directory (str): Base dir where the inference was performed.
    - best_config (str | Path | None): Path to the best fitting case's config TOML.
    - sigma (dict | None): Uncertainty of each observable, as used by the optimiser.

    Returns
    ----------
    - None
    """
    par_keys = list(pars.keys())

    df = _collect_case_observables(directory, obs, sigma)
    ok = df[~df['excluded']].copy() if not df.empty else df
    if ok.empty:
        log.warning('No scored case to compare with the true parameters; skipping')
        return

    # Parameter values each case actually ran with
    cases = Path(directory) / 'workers'
    confs = [
        toml.load(cases / f'w_{w}' / f'i_{i}' / 'init_coupler.toml')
        for w, i in zip(ok['worker'], ok['iter'])
    ]
    for k in par_keys:
        ok[k] = [float(recursive_get(c, k.split('.'))) for c in confs]
    best = _best_case(ok, best_config)

    pos = np.column_stack([_position_in_range(ok[k], k, pars[k]) for k in par_keys])
    truth_pos = np.array([_position_in_range(truth[k], k, pars[k]) for k in par_keys])
    best_pos = np.array([_position_in_range(best[k], k, pars[k]) for k in par_keys])
    labels = [
        f'{k}\n[{pars[k][0]:g}, {pars[k][1]:g}]'
        + ('  log' if variable_is_logarithmic(k) else '')
        for k in par_keys
    ]

    with plt.rc_context(OBS_STYLE):
        fig, axs = plt.subplots(
            1, 2, figsize=(12.0, max(3.2, 0.8 * len(par_keys) + 1.4)), width_ratios=[1.15, 1.0]
        )
        _panel_position(axs[0], pos, truth_pos, best_pos, labels, str(best['case']))
        _panel_residual(axs[1], labels, 100.0 * (best_pos - truth_pos), float(best['J']))
        axs[1].set_xlabel('(best fit - truth) / sampled range  [%]')
        axs[1].set_title(f'Best-fit parameter error, J = {best["J"]:+.4f}', fontsize=11)
        fig.suptitle(f'{Path(directory).name}: best-fit vs true parameters', fontsize=12.5)
        fig.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'result_parameters.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_proteus(best_config: str):
    """Make PROTEUS plots for the best fitting case.

    This requires reading output-data files from the disk.

    Parameters
    ----------
    - best_config (str): Path to the best fitting case's config TOML file.

    Returns
    ----------
    - None
    """

    handler = Proteus(config_path=best_config)
    handler.extract_archives()

    plot_skip = ('anim_visual', 'visual')

    for key in plot_dispatch.keys():
        if key in plot_skip:
            continue
        plot_dispatch[key](handler)
