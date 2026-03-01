#!/usr/bin/env python
"""Memory profiling wrapper for PROTEUS coupled simulations.

Monitors peak RSS (Resident Set Size) of the PROTEUS process and its
children during a simulation run. Writes memory timeseries to CSV and
optionally generates a plot.

Usage:
    python profile_memory.py -c <config.toml> -o <output_dir> [--plot] [--interval 5]

The wrapper samples memory every `--interval` seconds in a background
thread while PROTEUS runs. Output CSV columns:
    wall_time_s, rss_mb, vms_mb

Parameters
----------
-c : str
    Path to PROTEUS TOML config file.
-o : str
    PROTEUS output directory.
--plot : flag
    Generate RSS(t) plot after completion.
--interval : float
    Sampling interval in seconds (default: 5).
--mem-csv : str
    Output CSV path (default: <output_dir>/memory_profile.csv).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import resource


def get_self_rss_mb():
    """Get current process RSS in MB via resource module (no psutil needed)."""
    # ru_maxrss is in bytes on Linux, kilobytes on macOS
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)  # bytes → MB on macOS
    return usage.ru_maxrss / 1024  # KB → MB on Linux


def get_process_tree_rss_mb(pid):
    """Get total RSS of process tree (parent + children) via psutil."""
    if not HAS_PSUTIL:
        return get_self_rss_mb()
    try:
        parent = psutil.Process(pid)
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 * 1024)  # bytes → MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def monitor_memory(pid, csv_path, interval, stop_event):
    """Background thread: sample memory at regular intervals.

    Parameters
    ----------
    pid : int
        PID of the PROTEUS subprocess to monitor.
    csv_path : str
        Path to write memory CSV.
    interval : float
        Sampling interval in seconds.
    stop_event : threading.Event
        Signal to stop monitoring.
    """
    t0 = time.monotonic()
    peak_rss = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time_s", "rss_mb", "peak_rss_mb"])

        while not stop_event.is_set():
            elapsed = time.monotonic() - t0
            rss = get_process_tree_rss_mb(pid)
            peak_rss = max(peak_rss, rss)
            writer.writerow([f"{elapsed:.1f}", f"{rss:.1f}", f"{peak_rss:.1f}"])
            f.flush()
            stop_event.wait(timeout=interval)

    print(f"Peak RSS: {peak_rss:.1f} MB")
    return peak_rss


def make_plot(csv_path, plot_path):
    """Generate RSS(t) plot from memory CSV.

    Parameters
    ----------
    csv_path : str
        Path to memory CSV file.
    plot_path : str
        Path to save plot image.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping plot")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if len(data) == 0:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    t_min = data["wall_time_s"] / 60.0

    ax.plot(t_min, data["rss_mb"], "b-", linewidth=0.8, label="RSS")
    ax.plot(t_min, data["peak_rss_mb"], "r--", linewidth=0.8, label="Peak RSS")

    ax.set_xlabel("Wall time [min]")
    ax.set_ylabel("Memory [MB]")
    ax.set_title("PROTEUS memory profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add 5 GB reference line
    ax.axhline(y=5120, color="gray", linestyle=":", alpha=0.5, label="5 GB limit")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory profiling wrapper for PROTEUS simulations"
    )
    parser.add_argument("-c", "--config", required=True, help="PROTEUS config TOML")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Sampling interval [s]"
    )
    parser.add_argument("--plot", action="store_true", help="Generate RSS(t) plot")
    parser.add_argument("--mem-csv", default=None, help="Memory CSV output path")
    parser.add_argument(
        "--offline", action="store_true", help="Pass --offline to PROTEUS"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Pass --resume to PROTEUS"
    )
    args = parser.parse_args()

    csv_path = args.mem_csv or os.path.join(args.output, "memory_profile.csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # Build PROTEUS command
    cmd = ["proteus", "start", "-c", args.config, "-o", args.output]
    if args.offline:
        cmd.append("--offline")
    if args.resume:
        cmd.append("--resume")

    print(f"Running: {' '.join(cmd)}")
    print(f"Memory CSV: {csv_path}")
    print(f"Sampling interval: {args.interval}s")
    if not HAS_PSUTIL:
        print("Warning: psutil not installed, using resource.getrusage (less accurate)")
    print()

    # Start PROTEUS subprocess
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Start memory monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(proc.pid, csv_path, args.interval, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    # Stream PROTEUS output
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()
    stop_event.set()
    monitor_thread.join(timeout=10)

    print(f"\nPROTEUS exit code: {proc.returncode}")

    if args.plot:
        plot_path = csv_path.replace(".csv", ".png")
        make_plot(csv_path, plot_path)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
