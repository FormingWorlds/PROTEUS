"""Visualization utilities for asynchronous Bayesian optimization.

This module provides functions to visualize worker execution timelines,
timing distributions, and optimization performance metrics.

Functions:
    plot_times: Plot per-worker task timelines and histograms of timing metrics.
    plot_res: Plot regret and best observed value vs. time and iteration.
"""
from __future__ import annotations

import os
from glob import glob

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import torch
from botorch.utils.transforms import unnormalize
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from proteus.utils.helper import recursive_get

dtype = torch.double
fmt = "png"
dpi = 300

def plots_perf_timeline(logs, directory, n_init, min_text_width=0.88):
    """Generate timeline and histograms of process durations.

    This function makes multiple plots

    Args:
        logs (list of dict): Log entries containing timing and evaluation data.
        directory (str): Base directory where plots will be saved ("plots/" appended).
        n_init (int): Number of initial evaluations to skip when plotting.
        min_text_width (float): Minimum bar width threshold for white text.
    """
    # Build DataFrame skipping initial entries
    df = pd.DataFrame(logs[n_init:])
    if df.empty:
        print("No logs to display.")
        return

    # Shift timestamps so earliest start_time is zero
    global_t0 = df["start_time"].min()
    df["start"] = df["start_time"] - global_t0
    df["end"] = df["end_time"] - global_t0

    # Identify unique workers and assign colors
    workers = sorted(df["worker"].unique())
    color_map = {}
    for i,w in enumerate(workers):
        if i < 10:
            color_map[w] = plt.cm.tab10(i)
        else:
            color_map[w] = np.clip(np.random.random_sample(3), a_min=0.05, a_max=0.95)

    # Find bar widths and the rightmost endpoint
    bar_widths = df["end"] - df["start"]
    min_bar_width = bar_widths.min()
    max_bar_end = df["end"].max()

    # If narrowest bar is too small, stretch the x-axis but never move the bars
    stretch_needed = max(0, min_text_width - min_bar_width)
    # The x-axis must go at least as far as the rightmost bar end
    xlim_max = max_bar_end + stretch_needed

    # Create timeline plot
    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(workers)))
    for _, row in df.iterrows():
        bar_start = row["start"]
        bar_end = row["end"]
        bar_width = bar_end - bar_start
        bar_center = (bar_start + bar_end) / 2
        worker_y = row["worker"]
        # Format annotation text for the bar
        if len(row['x_value']) == 1:
            txt = f"(x, y)\n= ({row['x_value'][0]:.2f},{row['y_value']:.2f})\n{row['duration']:.2f}s"
        else:
            txt = f"y = {row['y_value']:.2f}\n{row['duration']:.2f}s"
        # Draw the bar
        ax.broken_barh(
            [(bar_start, bar_width)],
            (worker_y - 0.4, 0.8),
            facecolors=color_map[row["worker"]],
            edgecolor='k'
        )
        # Mark end of bar
        ax.vlines(bar_end, worker_y - 0.5, worker_y + 0.5,
                  color="gray", lw=1, linestyles='dashed')
        # Place text inside bar
        ax.text(
            bar_center,
            worker_y,
            txt,
            va="center",
            ha="center",
            fontsize=10,
            color="white" if bar_width > 0.25 else "black",
            fontweight="bold"
        )
    # Label axes and grid
    ax.set_yticks(workers)
    ax.set_yticklabels([f"Worker {w}" for w in workers])
    ax.set_xlabel("Wall clock time (s)")
    ax.set_title("Parallel workers")
    ax.grid(True, axis="x", alpha=0.3)

    # Set limits and save figure
    padding = 0.05 * xlim_max
    xlim_max_padded = xlim_max + padding

    ax.set_xlim(left=0, right=xlim_max_padded)
    plt.tight_layout()

    fig.savefig(os.path.join(directory, "plots", f"perf_parallel.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)

    # Histogram of total durations
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
            df["duration"],
            bins="auto",
            color="cornflowerblue",
            edgecolor="black",
            alpha=0.8
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Process Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(directory, "plots", f"perf_timehist.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram of BO times
    ax.hist(
            df["BO_time"],
            bins="auto",
            color="cornflowerblue",
            edgecolor="black",
            alpha=0.8
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of BO Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(directory, "plots", f"perf_BO_timehist.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)

    # Histogram of evaluation times
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
            df["t_eval"],
            bins="auto",
            color="cornflowerblue",
            edgecolor="black",
            alpha=0.8
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Evaluation Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(directory, "plots", f"perf_eval_timehist.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)

    # Colored histogram for fit times
    values = df['t_fit'].values
    positions = np.arange(len(values))   # 0, 1, 2, …, len(df)-1

    num_bins   = 30
    counts, bin_edges = np.histogram(values, bins=num_bins)
    bin_indices = np.digitize(values, bin_edges[:-1], right=False)

    avg_positions = []
    for b in range(1, num_bins+1):
        pos_in_bin = positions[bin_indices == b]
        if pos_in_bin.size:
            avg_positions.append(pos_in_bin.mean())
        else:
            avg_positions.append(0)    # or np.nan if you prefer

    norm   = mcolors.Normalize(vmin=min(avg_positions), vmax=max(avg_positions))
    cmap   = cm.viridis
    colors = [cmap(norm(p)) for p in avg_positions]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_bins):
        ax.bar(
            bin_edges[i],
            counts[i],
            width=bin_edges[i+1] - bin_edges[i],
            color=colors[i],
            align='edge'
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy mappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Average Row Index in Bin")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Fit Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(directory, "plots", f"perf_fit_timehist.{fmt}"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # Colored histogram for acquisition times
    values = df['t_ac'].values
    positions = np.arange(len(values))   # 0, 1, 2, …, len(df)-1

    num_bins   = 30
    counts, bin_edges = np.histogram(values, bins=num_bins)
    bin_indices = np.digitize(values, bin_edges[:-1], right=False)

    avg_positions = []
    for b in range(1, num_bins+1):
        pos_in_bin = positions[bin_indices == b]
        if pos_in_bin.size:
            avg_positions.append(pos_in_bin.mean())
        else:
            avg_positions.append(0)

    norm   = mcolors.Normalize(vmin=min(avg_positions), vmax=max(avg_positions))
    cmap   = cm.viridis
    colors = [cmap(norm(p)) for p in avg_positions]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_bins):
        ax.bar(
            bin_edges[i],
            counts[i],
            width=bin_edges[i+1] - bin_edges[i],
            color=colors[i],
            align='edge'
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy mappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Average Row Index in Bin")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Acquisition Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(directory, "plots", f"perf_acquisition_timehist.{fmt}"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # Scatter plot of distance to busy locations
    fig, ax = plt.subplots(figsize=(7, 4))
    df_f = df[df["dist"].notnull()]
    ax.scatter(
        df_f.index,
        df_f["dist"],
        marker="o",
        linestyle="-",
        color="cornflowerblue",
        alpha=0.8,
        label="Distance"
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_title("Query Distance to Busy Location",
                fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(os.path.join(directory, "plots", f"perf_distance_iters.{fmt}"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plots_perf_converge(D, T, n_init, directory):
    """Plot regret and best observed value over time and iterations.

    Args:
        D (dict): Contains 'Y' list of objective values.
        T (list): Elapsed times corresponding to each evaluation.
        n_init (int): Number of initial evaluations to skip.
        directory (str): Base dir where "plots/" subfolder will be created.
    """

    Y = np.array(D["Y"], copy=None, dtype=float).flatten()  # Flatten in case it's (N,1)
    Y = Y[n_init:]

    y_best = Y[0]
    Y_best = [y_best]

    for i in range(1,len(Y)):

        if Y[i] > y_best:
            y_best = Y[i]

        Y_best.append(y_best)

    Y_best = np.array(Y_best)
    T = np.array(T)            # Assume T aligns with Y

    oracle = 1.  # Change as appropriate
    regret = np.abs(Y_best - oracle)
    log_regret = np.log10(regret + 1e-12)  # add small number to avoid log(0)
    n = np.arange(len(Y_best))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # Top: log regret vs t
    axes[0].plot(T, log_regret, marker='o')
    axes[0].set_ylabel('log10(Regret)')
    axes[0].set_xlabel('t')
    axes[0].set_title('Log Regret vs Time')
    axes[0].grid(True)
    # axes[0].legend()

    # Bottom: log regret vs n
    axes[1].plot(n, log_regret, marker='o', color='tab:orange')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('log10(Regret)')
    axes[1].set_title('Log Regret vs Step')
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # axes[1].legend()

    plt.tight_layout()


    fig.savefig(os.path.join(directory, "plots", f"perf_regret.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # Top: log regret vs t
    axes[0].plot(T, Y_best, marker='o')
    axes[0].set_ylabel('y')
    axes[0].set_xlabel('t')
    axes[0].set_title('Best Value vs Time')
    axes[0].grid(True)

    # Bottom: log regret vs n
    axes[1].plot(n, Y_best, marker='o', color='tab:orange')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('y')
    axes[1].set_title('Best Value vs Step')
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    fig.savefig(os.path.join(directory, "plots", f"perf_bestval.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)


def plot_result_objective(D, parameters, n_init, directory, yclip=-12):
    """Plot objective function at each sample that was created.

    Args:
        D (dict): Contains 'X' and 'Y' lists.
        parameters (dict): Parameter names and bounds
        n_init (int): Number of initial evaluations (to be highlighted).
        directory (str): Base dir where "plots/" subfolder will be created.
        yclip (float): minimum limit y-axis scale
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
        ylbl = f"Value of objective\nclipped to J>{yclip}"
    else:
        ylbl = "Value of objective"

    # Get bounds
    keys = list(parameters.keys())
    d = len(keys)
    bounds = torch.tensor([[list(parameters.values())[i][j] for i in range(d)] for j in range(2)])

    # Un-normalise X data
    X = unnormalize(D['X'], bounds)
    X = np.array(X, copy=None, dtype=float)

    # Colors
    C = np.full_like(Y, 'k', dtype=str)
    C[:n_init] = 'c'
    C[i_best] = 'm'

    # Limits
    ymax = 1.0
    ymin = np.amin(Y)

    # Plot
    fig,axs = plt.subplots(2, d, figsize=(2.7*d, 3.2))
    axs[0,0].set_ylabel("Histogram")
    axs[1,0].set_ylabel(ylbl)

    # plot scatter points
    for i in range(d):
        # clipped points
        axs[1,i].scatter(X[mask,i], Y[mask], c=list(C[mask]), s=11, alpha=0.8, zorder=4, marker='v', edgecolors='none')

        # unclipped points
        axs[1,i].scatter(X[~mask,i], Y[~mask], c=list(C[~mask]), s=10, alpha=0.6, zorder=5, marker='o', edgecolors='none')

        # configure axes
        axs[1,i].set_xlabel(keys[i], fontsize=10)
        axs[1,i].grid(alpha=0.2, zorder=0)
        axs[1,i].set_ylim(ymin, ymax)
        if i>=1:
            axs[1,i].set_yticklabels([])
        axs[0,i].set_yticklabels([])
        axs[0,i].set_xticklabels([])

    # plot histograms
    for i in range(d):
        x1 = X[:n_init, i]   # only initial values
        x2 = X[n_init:, i]   # after initial values
        axs[0,i].hist([x2,x1], bins=11,
                        stacked=True, histtype='barstacked', color=['k','c'],
                        zorder=2)

        # median and stddev
        x_med = np.median(x2)
        x_std = np.std(x2)
        axs[0,i].set_title(f"{x_med:3f}"+r"$\pm$"+f"{x_std:3f}", fontsize=8, color='r')

        # overplot median in both panels
        for j in (0,1):
            axs[j,i].axvline(x=x_med, zorder=4, color='r', alpha=0.8)

        # overplot best in both panels
        axs[j,i].axvline(x=X[i_best,i], zorder=5, color='m', alpha=0.8)

        # grid
        axs[j,i].grid(alpha=0.2, zorder=0, axis='x')

    # save plot
    fig.subplots_adjust(wspace=0.012, hspace=0.022)
    fig.savefig(os.path.join(directory, "plots", f"result_objective.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)


def plot_result_correlation(pars:dict, obs:dict, directory):
    """Plot correlation between observables and parameters.

    This requires reading output-data files from the disk.

    Args:
        par_keys (dict): Parameter names and bounds
        obs_keys (dict): Observable names and target values
        directory (str): Base dir where the inference was performed.
    """

    # Convert to lists
    par_keys = list(pars.keys())
    obs_keys = list(obs.keys())

    # Get directories for all cases of interest
    cases = glob(directory + "/workers/w_*/i_*/")

    # Extract parameters and observables
    X,Y = [], []
    for c in cases:
        # Read data
        conf = toml.load(c+"init_coupler.toml")
        help = pd.read_csv(c+"runtime_helpfile.csv", delimiter=r"\s+")

        # Get parameters and observables
        xx = [recursive_get(conf,k.split(".")) for k in par_keys]
        yy = list(help.iloc[-1][obs_keys].T)

        # Store these
        X.append(xx)
        Y.append(yy)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    # Axes
    n_par = len(par_keys)
    n_obs = len(obs_keys)

    # Make plot
    fig,axs = plt.subplots(n_obs, n_par, figsize=(2.7*n_par, 2.7*n_obs))
    for i in range(n_par):
        for j in range(n_obs):
            # plot data
            xx = X[:, i]
            yy = Y[:, j]
            axs[j,i].scatter(xx, yy, color='k', alpha=0.8, s=8, zorder=4)

            # axis grid
            axs[j,i].grid(alpha=0.2, zorder=0)

            # observables
            axs[j,i].axhline(y=obs[obs_keys[j]], color='g', alpha=0.5, label="Observed")

            # these variables are more natural on a log-scale
            if ("vmr" in obs_keys[j]) or (obs_keys[j] == "P_surf"):
                axs[j,i].set_yscale("log")

            # hide tick labels
            if i>=1:
                axs[j,i].set_yticklabels([])
            if j<n_obs-1:
                axs[j,i].set_xticklabels([])


    # Legend
    axs[0,0].legend()

    # Axis labels
    for i in range(n_par):
        axs[-1,i].set_xlabel(par_keys[i], fontsize=10)
    for j in range(n_obs):
        axs[j, 0].set_ylabel(obs_keys[j], fontsize=10)

    # Decorate
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(directory, "plots", f"result_correlation.{fmt}"), dpi = dpi, bbox_inches='tight')
    plt.close(fig)
