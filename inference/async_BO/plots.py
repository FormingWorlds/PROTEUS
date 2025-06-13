import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_times(logs, directory, min_text_width=0.88):
    df = pd.DataFrame(logs)
    if df.empty:
        print("No logs to display.")
        return

    # Normalize times so t0 is zero
    global_t0 = df["start_time"].min()
    df["start"] = df["start_time"] - global_t0
    df["end"] = df["end_time"] - global_t0

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

    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(workers)))
    for _, row in df.iterrows():
        bar_start = row["start"]
        bar_end = row["end"]
        bar_width = bar_end - bar_start
        bar_center = (bar_start + bar_end) / 2
        worker_y = row["worker"]
        if len(row['x_value']) == 1:
            txt = f"(x, y)\n= ({row['x_value'][0]:.2f},{row['y_value']:.2f})\n{row['duration']:.2f}s"
        else:
            txt = f"y = {row['y_value']:.2f}\n{row['duration']:.2f}s"
        ax.broken_barh(
            [(bar_start, bar_width)],
            (worker_y - 0.4, 0.8),
            facecolors=color_map[row["worker"]],
            edgecolor='k'
        )
        ax.vlines(bar_end, worker_y - 0.5, worker_y + 0.5, color="gray", lw=1, linestyles='dashed')
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

    ax.set_yticks(workers)
    ax.set_yticklabels([f"Worker {w}" for w in workers])
    ax.set_xlabel("Wall clock time (s)")
    ax.set_title("Parallel workers")
    ax.grid(True, axis="x", alpha=0.3)

    padding = 0.05 * xlim_max  # 5% of the x range as padding
    xlim_max_padded = xlim_max + padding

    ax.set_xlim(left=0, right=xlim_max_padded)
    plt.tight_layout()
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + f"parallel.png", dpi = 300)
    plt.close(fig)

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
    fig.savefig(path + f"t_hist.png", dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

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
    fig.savefig(path + f"BO_t_hist.png", dpi = 300)
    plt.close(fig)


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
    fig.savefig(path + f"eval_t_hist.png", dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
            df["t_lock"],
            bins="auto",
            color="cornflowerblue",
            edgecolor="black",
            alpha=0.8
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Lock Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + f"lock_t_hist.png", dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
            df["t_fit"],
            bins="auto",
            color="cornflowerblue",
            edgecolor="black",
            alpha=0.8
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Fit Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + f"fit_t_hist.png", dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
            df["t_ac"],
            bins="auto",
            color="cornflowerblue",
            edgecolor="black",
            alpha=0.8
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Acquisition Times", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = directory+"plots/"
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + f"ac_t_hist.png", dpi = 300)
    plt.close(fig)

def plot_res(D, T, n_init, directory, save = True):
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

    oracle = 1  # Change as appropriate
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
        fig.savefig(path + f"reg.png", dpi = 300)
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
    #axes[0].legend()

    # Bottom: log regret vs n
    axes[1].plot(n, Y_best, marker='o', color='tab:orange')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('y')
    axes[1].set_title('Best Value vs Step')
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    #axes[1].legend()

    plt.tight_layout()

    if save:
        path = directory+"plots/"
        os.makedirs(path, exist_ok=True)
        fig.savefig(path + f"best_val.png", dpi = 300)
        plt.close(fig)

    else:
        fig.show()
