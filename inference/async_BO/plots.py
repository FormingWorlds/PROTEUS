"""Visualization utilities for asynchronous Bayesian optimization.

This module provides functions to visualize worker execution timelines,
timing distributions, and optimization performance metrics.

Functions:
    plot_times: Plot per-worker task timelines and histograms of timing metrics.
    plot_res: Plot regret and best observed value vs. time and iteration.
"""
from __future__ import annotations

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


def plot_times(logs, directory, n_init, min_text_width=0.88):

    """Generate timeline and histograms of process durations.

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
    color_map = {w: plt.cm.tab10(i) for i, w in enumerate(workers)}

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
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + "parallel.png", dpi = 300)
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
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + "t_hist.png", dpi = 300)
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
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + "BO_t_hist.png", dpi = 300)
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
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + "eval_t_hist.png", dpi = 300)
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
    path = os.path.join(directory, "plots")
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, "fit_t_hist.png"), dpi=300)
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
    path = os.path.join(directory, "plots")
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, "ac_t_hist.png"), dpi=300)
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
    path = directory + "plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + "dist_v_iter.png", dpi=300)
    plt.close(fig)



def plot_res(D, T, n_init, directory, save = True):

    """Plot regret and best observed value over time and iterations.

    Args:
        D (dict): Contains 'Y' list of objective values.
        T (list): Elapsed times corresponding to each evaluation.
        n_init (int): Number of initial evaluations to skip.
        directory (str): Base dir where "plots/" subfolder will be created.
        save (bool): If True, save plots; else display interactively.
    """

    Y = np.array(D["Y"]).flatten()  # Flatten in case it's (N,1)
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

    if save:
        path = directory+"plots/"
        os.makedirs(path, exist_ok=True)
        fig.savefig(path + "reg.png", dpi = 300)
        plt.close(fig)

    else:
         fig.show()


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

    if save:
        path = directory+"plots/"
        os.makedirs(path, exist_ok=True)
        fig.savefig(path + "best_val.png", dpi = 300)
        plt.close(fig)

    else:
        fig.show()
