"""Asynchronous Bayesian optimization pipeline utilities.

This module implements functions to:

    * Save and checkpoint optimization state to disk.
    * Generate initial sample locations via a Halton low-discrepancy sequence.
    * Run a worker process that performs BO steps, logs metrics, and checkpoints.
    * Coordinate multiple worker processes for asynchronous Bayesian optimization.

Functions:
    checkpoint: Persist shared data, logs, and timestamps to files.
    init_locs: Create initial candidate points in the unit hypercube.
    worker: Execute BO iterations in a subprocess, updating shared state.
    parallel_process: Set up shared resources, spawn workers, and collect results.
"""
from __future__ import annotations

import os
import pickle
import time
from functools import partial
from multiprocessing import Manager, Process

import torch
from scipy.stats.qmc import Halton

from proteus.inference.plot import plots_perf_converge, plots_perf_timeline
from proteus.inference.BO import BO_step
from proteus.utils.coupler import get_proteus_directories

# Tensor dtype for all computations
dtype = torch.double


def checkpoint(D: dict, logs: list, Ts: list, output_dir: str) -> None:
    """Save the current state of the optimization to disk.

    Creates the output directory if needed, and writes three pickle files:
    * data.pkl: dictionary of observed inputs 'X' and outputs 'Y'
    * logs.pkl: list of per-evaluation log dictionaries
    * Ts.pkl: list of elapsed timestamps for each evaluation

    Args:
        D (dict): Shared dict containing keys 'X' and 'Y'.
        logs (list): Shared list of log dictionaries.
        Ts (list): Shared list of elapsed times (floats).
        output_dir (str): Directory path where pickle files will be saved.
    """
    # Persist the data dictionary
    with open(os.path.join(output_dir, "data.pkl"), "wb") as f_data:
        pickle.dump(dict(D), f_data)

    # Persist the log list
    with open(os.path.join(output_dir, "logs.pkl"), "wb") as f_logs:
        pickle.dump(list(logs), f_logs)

    # Persist the timestamps
    with open(os.path.join(output_dir, "Ts.pkl"), "wb") as f_ts:
        pickle.dump(list(Ts), f_ts)


def init_locs(n_workers: int, D: dict) -> torch.Tensor:
    """Generate initial sample locations using a Halton sequence.

    Args:
        n_workers (int): Number of initial points to generate.
        D (dict): Shared dict with key 'X' to infer problem dimension.

    Returns:
        torch.Tensor: Tensor of shape (n_workers, d) in [0,1]^d for initial sampling.
    """
    # Determine the input dimension from existing data
    d = D["X"].shape[-1]

    # Create a scrambled Halton sampler
    sampler = Halton(d=d, scramble=True)
    samples = sampler.random(n=n_workers)

    # Convert to Torch tensor
    return torch.tensor(samples, dtype=dtype)


def worker(
    process_fun,
    build_obj,
    D_shared,
    B,
    T,
    T0: float,
    x_init: torch.Tensor,
    lock,
    max_len: int,
    worker_id: int,
    log_list,
    output_dir: str
) -> None:
    """Worker subprocess that performs asynchronous BO steps.

    Each worker repeatedly:
      1. Builds its objective function for the current task.
      2. Calls BO_step to propose and evaluate a new point.
      3. Logs timing and performance metrics.
      4. Updates shared data, busy points, and checkpoints.
    Runs until the total number of observations reaches max_len.

    Args:
        process_fun (callable): Partial of BO_step with fixed hyperparameters.
        build_obj (callable): Partial returning a worker-specific objective f.
        D_shared (Manager.dict): Shared dict with keys 'X', 'Y'.
        B (Manager.dict): Shared dict of current busy points per worker.
        T (Manager.list): Shared list for evaluation end times.
        T0 (float): Reference start time (time.perf_counter()).
        x_init (torch.Tensor): Initial query point for first iteration.
        lock: Multiprocessing lock for synchronizing shared access.
        max_len (int): Maximum total number of evaluations.
        worker_id (int): Unique identifier of this worker.
        log_list (Manager.list): Shared list to store per-eval log dicts.
        output_dir (str): Output directory for the whole inference call (abspath).
    """
    task_id = 0

    # Seed PyTorch RNG for reproducibility per process
    torch.manual_seed(os.getpid())

    while True:
        # Check if we've reached the maximum number of evaluations
        with lock:
            current_X = D_shared["X"]
        if len(current_X) >= max_len:
            break

        # For the first iteration, use provided initial point
        x_in = x_init if task_id == 0 else None

        # Build the objective function for this worker and iteration
        f = build_obj(worker=worker_id, iter=task_id)

        # Run BO step and measure wall-clock time
        t_start = time.perf_counter()
        x_new, y_new, t_bo, t_eval, t_lock, t_fit, t_ac, dist = process_fun(
            f=f,
            D=D_shared,
            B=B,
            x_in=x_in,
            lock=lock,
            worker_id=worker_id
        )
        t_end = time.perf_counter()

        # Acquire lock to update shared structures and persist state
        with lock:
            # Append new data to shared X and Y
            D_shared["X"] = torch.cat((D_shared["X"], x_new), dim=0)
            D_shared["Y"] = torch.cat((D_shared["Y"], y_new), dim=0)

            # Record end timestamp
            T.append(t_end)

            # Log metrics for this task
            log_list.append({
                "worker": worker_id,
                "task_id": task_id,
                "start_time": t_start,
                "end_time": t_end,
                "duration": t_end - t_start,
                "BO_time": t_bo,
                "t_eval": t_eval,
                "t_lock": t_lock,
                "t_fit": t_fit,
                "t_ac": t_ac,
                "dist": dist,
                "x_value": x_new.tolist()[0],
                "y_value": float(y_new),
            })

            # Compute relative timestamps and checkpoint
            Ts = [t - T0 for t in list(T)]
            checkpoint(D_shared, log_list, Ts, output_dir)

        task_id += 1


def parallel_process(
    objective_builder,
    kernel,
    n_restarts: int,
    n_samples: int,
    n_workers: int,
    max_len: int,
    output: str,
    ref_config,
    observables,
    parameters
) -> tuple[dict, list, list]:
    """Orchestrate parallel asynchronous Bayesian optimization.

    Loads initial data, sets up shared memory, spawns worker processes,
    waits for completion, and generates diagnostic plots.

    Args:
        objective_builder (callable): Factory to build per-worker objective f.
        kernel: Covariance kernel for the Gaussian process.
        n_restarts (int): Number of restarts in acquisition optimization.
        n_samples (int): Number of raw samples for acquisition optimization.
        n_workers (int): Number of parallel worker processes.
        max_len (int): Target total number of evaluations.
        output (str): Output directory for checkpoints and plots.
        ref_config: Reference config to pass to objective_builder.
        observables: List or dict of target observable values.
        parameters: Dict of parameter bounds for inference.

    Returns:
        D_final (dict): Final 'X' and 'Y' data after all evaluations.
        logs (list): List of per-evaluation log dicts.
        T (list): List of elapsed times from the start of optimization.
    """
    # Partially apply builder and BO_step with fixed settings
    build_obj = partial(
        objective_builder,
        observables=observables,
        parameters=parameters,
        ref_config=ref_config,
        output=output,
    )
    process_fun = partial(
        BO_step,
        k=kernel,
        n_restarts=n_restarts,
        n_samples=n_samples
    )

    # Absolute path to shared output dir
    output_abspath = get_proteus_directories(output)["output"]

    mgr = Manager()

    # Load initial dataset
    D_init_path = os.path.join(output_abspath, "init.pkl")
    if not os.path.isfile(D_init_path):
        raise FileNotFoundError("Cannot find D_init file: " + D_init_path)
    with open(D_init_path, "rb") as f_init:
        D_init = pickle.load(f_init)

    # Initialize shared data structures
    D_shared = mgr.dict({
        "X": D_init["X"],
        "Y": D_init["Y"]
    })
    n_init = len(D_shared["X"])

    lock = mgr.Lock()
    log_list = mgr.list([None] * n_init)

    # Generate initial candidate locations and busy-map
    X_init = init_locs(n_workers, D_shared)
    B = mgr.dict()

    # Shared list for end times
    T = mgr.list()
    T0 = time.perf_counter()

    # Spawn worker processes
    procs = []
    for wid in range(n_workers):
        x0 = X_init[wid].unsqueeze(0)
        with lock:
            B[wid] = x0
        p = Process(
            target=worker,
            args=(
                process_fun,
                build_obj,
                D_shared,
                B,
                T,
                T0,
                x0,
                lock,
                max_len,
                wid,
                log_list,
                output_abspath
            )
        )
        p.start()
        procs.append(p)

    # Wait for all workers to finish
    for p in procs:
        p.join()

    # Collect final results
    D_final = dict(D_shared)
    logs = list(log_list)
    T_elapsed = [t - T0 for t in list(T)]

    # Generate diagnostic plots
    plots_perf_timeline(logs, output_abspath, n_init)
    plots_perf_converge(D_final, T_elapsed, n_init, output_abspath)

    return D_final, logs, T_elapsed
