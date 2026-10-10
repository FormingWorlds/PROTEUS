"""Asynchronous Bayesian optimisation: worker processes that each run BO steps
against shared data, checkpointing as they go, and the orchestration around them.
"""

from __future__ import annotations

import logging
import math
import os
import time
from functools import partial
from multiprocessing import Manager, Process

import pandas as pd
import torch

from proteus.inference.BO import BO_step, init_locs
from proteus.inference.failures import ProteusRunFailure
from proteus.inference.objective import close_worker_runner
from proteus.inference.utils import get_kernel, load_dataset_csv, save_dataset_csv
from proteus.utils.coupler import get_proteus_directories
from proteus.utils.logs import attach_worker_logfile

# Tensor dtype for all computations
dtype = torch.double
log = logging.getLogger('fwl.' + __name__)


# Smallest rise of the best objective that resets the `patience` count
PATIENCE_MIN_GAIN = 0.01


def evaluations_since_improvement(Y: torch.Tensor, n_init: int) -> int:
    """Evaluations since the best objective last rose by more than PATIENCE_MIN_GAIN.

    The best of the initial samples is the first reference. Small rises add up: the
    reference moves only once the best has risen by more than the threshold in total.

    Parameters
    ----------
    - Y (torch.Tensor): Objective values in evaluation order, initial samples first.
    - n_init (int): Number of initial samples at the start of `Y`.

    Returns
    ----------
    - int: Optimisation steps completed after the last improvement.
    """
    y = Y.reshape(-1).tolist()
    ref, last = max(y[:n_init], default=-math.inf), n_init - 1
    for i in range(n_init, len(y)):
        if y[i] > ref + PATIENCE_MIN_GAIN:
            ref, last = y[i], i
    return len(y) - 1 - last


def checkpoint(D: dict, logs: list, Ts: list, output_dir: str) -> None:
    """Save the current state of the optimization to disk.

    Creates the output directory if needed, and writes:
    * data.csv: observed inputs and objective values
    * logs.csv: list of per-evaluation log dictionaries
    * Ts.csv: elapsed timestamps for each evaluation

    Parameters
    ----------
    - D (dict): Shared dict containing keys 'X' and 'Y'.
    - logs (list): Shared list of log dictionaries.
    - Ts (list): Shared list of elapsed times (floats).
    - output_dir (str): Directory path where checkpoint files will be saved.

    Returns
    ----------
    - None
    """
    # Persist BO data
    save_dataset_csv(D['X'], D['Y'], os.path.join(output_dir, 'data.csv'))

    # Persist the log list
    pd.DataFrame(list(logs)).to_csv(os.path.join(output_dir, 'logs.csv'), index=False)

    # Persist the timestamps
    pd.DataFrame({'elapsed_s': list(Ts)}).to_csv(
        os.path.join(output_dir, 'Ts.csv'), index=False
    )


def _parent_logfile() -> str | None:
    """Path of the logfile the inference run's logger is writing, if it has one.

    Read in the parent, because a spawned worker has no logging configuration
    of its own to read it from. Returning the handler's own path rather than
    rebuilding it keeps the logfile named in one place only.
    """
    for handler in logging.getLogger('fwl').handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None


def worker(
    process_fun,
    build_obj,
    D_shared,
    B,
    T,
    T0: float,
    x_init: torch.Tensor,
    n_init: int,
    lock,
    max_len: int,
    worker_id: int,
    log_list,
    output_dir: str,
    logpath: str | None = None,
    log_level: int = logging.INFO,
    stop=None,
    aborts=None,
    patience: int = 0,
    converged=None,
) -> None:
    """Worker subprocess that performs asynchronous BO steps.

    Each worker repeatedly:
      1. Builds its objective function for the current task.
      2. Calls BO_step to propose and evaluate a new point.
      3. Logs timing and performance metrics.
      4. Updates shared data, busy points, and checkpoints.
    Runs until the total number of observations reaches max_len, until
    `stop` is set because a worker's run failed under `abort_on_failure`, or
    until `converged` is set because the best objective stopped improving.

    Parameters
    ----------
    - process_fun (callable): Partial of BO_step with fixed hyperparameters.
    - build_obj (callable): Partial returning a worker-specific objective f.
    - D_shared (Manager.dict): Shared dict with keys 'X', 'Y'.
    - B (Manager.dict): Shared dict of current busy points per worker.
    - T (Manager.list): Shared list for evaluation end times.
    - T0 (float): Reference start time (time.perf_counter()).
    - x_init (torch.Tensor): Initial query point for first iteration.
    - lock: Multiprocessing lock for synchronizing shared access.
    - max_len (int): Maximum total number of evaluations.
    - worker_id (int): Unique identifier of this worker.
    - log_list (Manager.list): Shared list to store per-eval log dicts.
    - output_dir (str): Output directory for the whole inference call (abspath).
    - logpath (str | None): Inference run logfile to reopen when this process has no
      logging configuration of its own.
    - log_level (int): Numeric level to log at, read from the parent.
    - stop (Manager.Event | None): Set by the worker whose run failed under
      `abort_on_failure`, and checked by every worker before it starts
      another evaluation.
    - aborts (Manager.list | None): Receives the failure that set `stop`, so
      the parent can raise it once every worker has exited.
    - patience (int): Stop the study once this many evaluations pass without the
      best objective rising by more than PATIENCE_MIN_GAIN; 0 never stops.
    - converged (Manager.Event | None): Set by the worker that finds the
      `patience` limit reached, and checked like `stop`.

    Returns
    ----------
    - None
    """
    # A spawned worker inherits no logging configuration on MacOS.
    # Reattach before any work starts, so that a failure in
    # the very first iteration is still recorded in the logfile.
    if logpath:
        attach_worker_logfile(logpath, log_level)

    try:
        _worker_loop(
            process_fun,
            build_obj,
            D_shared,
            B,
            T,
            T0,
            x_init,
            n_init,
            lock,
            max_len,
            worker_id,
            log_list,
            output_dir,
            stop,
            patience,
            converged,
        )
    except ProteusRunFailure as failure:
        # Only raised out of the objective under `abort_on_failure`.
        if aborts is not None:
            aborts.append(failure)
        if stop is not None:
            stop.set()
        log.exception(f'Worker {worker_id} stopped the study after a failed run')
        raise
    except BaseException:
        # A worker that dies takes its traceback with it: multiprocessing
        # prints it to the parent's stderr without consulting the logging
        # configuration, so nothing reaches the logfile. Record it here.
        log.exception(f'Worker {worker_id} stopped early and will run no further evaluations')
        raise
    finally:
        # Release this worker's busy point.
        try:
            with lock:
                B.pop(worker_id, None)
        except Exception:
            log.warning(f'Worker {worker_id} could not release its busy point')

        # Stop the PROTEUS process this worker was reusing, if it had one, so
        # that a study which ends early leaves nothing running behind it.
        close_worker_runner()


def _worker_loop(
    process_fun,
    build_obj,
    D_shared,
    B,
    T,
    T0: float,
    x_init: torch.Tensor,
    n_init: int,
    lock,
    max_len: int,
    worker_id: int,
    log_list,
    output_dir: str,
    stop=None,
    patience: int = 0,
    converged=None,
) -> None:
    """Run BO iterations until the evaluation budget or the patience is reached.

    The body of `worker`, separated so that failure reporting and busy-point
    release wrap every exit path.

    Parameters
    ----------
    - See `worker`; arguments are forwarded unchanged.

    Returns
    ----------
    - None
    """
    task_id = 0

    while True:
        # Check if we've reached the maximum number of evaluations

        current_X = D_shared['X']
        if len(current_X) >= max_len:
            log.info(f'Worker {worker_id} exiting')
            break

        # Another worker's run failed under `abort_on_failure`. Checked
        # between evaluations only.
        if stop is not None and stop.is_set():
            log.info(f'Worker {worker_id} exiting: the study is stopping on a failed run')
            break
        if converged is not None and converged.is_set():
            log.info(f'Worker {worker_id} exiting: the best objective stopped improving')
            break

        # For the first iteration, use provided initial point
        x_in = x_init if task_id == 0 else None

        # Build the objective function for this worker and iteration
        f = build_obj(worker=worker_id, iter=task_id)

        # Run BO step and measure wall-clock time
        t_start = time.perf_counter()
        x_new, y_new, t_bo, t_eval, t_lock, t_fit, t_ac, dist = process_fun(
            f=f, D=D_shared, B=B, x_in=x_in, lock=lock, worker_id=worker_id
        )

        t_end = time.perf_counter()

        # Acquire lock to update shared structures and persist state
        with lock:
            # Append new data to shared X and Y
            X = torch.cat((D_shared['X'], x_new), dim=0)
            D_shared['X'] = X
            Y = torch.cat((D_shared['Y'], y_new), dim=0)
            D_shared['Y'] = Y

            # Record end timestamp
            T.append(t_end)

            # Log metrics for this task
            log_list.append(
                {
                    'worker': worker_id,
                    'task_id': task_id,
                    'start_time': t_start,
                    'end_time': t_end,
                    'duration': t_end - t_start,
                    'BO_time': t_bo,
                    't_eval': t_eval,
                    't_lock': t_lock,
                    't_fit': t_fit,
                    't_ac': t_ac,
                    'dist': dist,
                    'x_value': x_new.tolist()[0],
                    'y_value': float(y_new),
                }
            )

            # Compute relative timestamps and checkpoint
            D_snap = dict(D_shared)
            log_snap = list(log_list)
            Ts_snap = [t - T0 for t in list(T)]

        checkpoint(D_snap, log_snap, Ts_snap, output_dir)

        step = len(X) - n_init
        current_best = Y.max().item()
        log.info(f'Step {step:5d}, best objective = {current_best:+.5f}')

        if (
            patience
            and converged is not None
            and not converged.is_set()
            and evaluations_since_improvement(Y, n_init) >= patience
        ):
            log.info(
                f'Stopping the study: the best objective has not risen by more than '
                f'{PATIENCE_MIN_GAIN} in {patience} evaluations'
            )
            converged.set()

        task_id += 1


def parallel_process(
    objective_builder,
    kernel: str,
    acqf: str,
    n_workers: int,
    max_len: int,
    output: str,
    seed: int,
    ref_config: str,
    observables: dict,
    parameters: dict,
    failure_codes: list[int],
    sigma: dict | None = None,
    correlation: dict | None = None,
    patience: int = 0,
) -> tuple[dict, list, list]:
    """Orchestrate parallel asynchronous Bayesian optimization.

    Loads initial data, sets up shared memory, spawns worker processes,
    waits for completion, and generates diagnostic plots.

    Parameters
    ----------
    - objective_builder (callable): Factory to build per-worker objective f.
    - kernel (str): Covariance kernel for the Gaussian process.
    - acqf (str): Acquisition function for BO step.
    - n_workers (int): Number of parallel worker processes.
    - max_len (int): Target total number of evaluations, including initial data.
    - output (str): Output directory for checkpoints and plots.
    - seed (int): Random seed for some degree of reproducibility
    - ref_config (str): Path to reference config to pass to objective_builder.
    - observables (dict): Target observables (keys) and values.
    - parameters (dict):  Parameters (keys) with bounds (values) for inference.
    - failure_codes (list[int]): PROTEUS status codes that complete normally but
      that this run excludes from the fit.
    - sigma (dict | None): Uncertainty of each observable, or None for the
      relative-difference objective.
    - correlation (dict | None): Correlations between the observable uncertainties.
    - patience (int): Evaluations without improvement after which the study
      stops; 0 runs the full budget.

    Returns
    ----------
    - D_final (dict): Final 'X' and 'Y' data after all evaluations.
    - logs (list): List of per-evaluation log dicts.
    - T (list): List of elapsed times from the start of optimization.
    """

    # Partially apply builder and BO_step with fixed settings
    build_obj = partial(
        objective_builder,
        observables=observables,
        parameters=parameters,
        ref_config=ref_config,
        output=output,
        failure_codes=failure_codes,
        sigma=sigma,
        correlation=correlation,
    )

    # Build kernel
    d = len(parameters)
    kernel = get_kernel(kernel, d)

    process_fun = partial(
        BO_step,
        k=kernel,
        acqf=acqf,
    )

    # Absolute path to shared output dir
    output_abspath = get_proteus_directories(output)['output']

    mgr = Manager()

    # Load initial dataset
    D_init_path = os.path.join(output_abspath, 'init.csv')
    if not os.path.isfile(D_init_path):
        raise FileNotFoundError('Cannot find D_init file: ' + D_init_path)
    D_init = load_dataset_csv(D_init_path)

    # Initialize shared data structures
    D_shared = mgr.dict({'X': D_init['X'], 'Y': D_init['Y']})
    n_init = len(D_shared['X'])

    lock = mgr.Lock()
    # Set by the first worker whose run fails under `abort_on_failure`,
    # which also leaves that failure in `aborts` for the parent to re-raise.
    stop = mgr.Event()
    aborts = mgr.list()
    # Set by the first worker that finds the `patience` limit reached
    converged = mgr.Event()
    log_list = mgr.list([None] * n_init)  # no logs from init data

    # Generate initial candidate locations and busy-map
    X_init = init_locs(n_workers, D_shared, acqf=acqf)  # shape (n_workers, d)
    B = mgr.dict()

    # Shared list for end times
    T = mgr.list()
    T0 = time.perf_counter()

    # Set up step constraint
    max_steps = max_len - (n_workers - 1)

    # Read in the parent: a spawned worker has none of this to read from.
    worker_logpath = _parent_logfile()
    worker_log_level = logging.getLogger('fwl').level

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
                n_init,
                lock,
                max_steps,
                wid,
                log_list,
                output_abspath,
                worker_logpath,
                worker_log_level,
                stop,
                aborts,
                patience,
                converged,
            ),
        )
        p.start()
        procs.append(p)

    # Wait for all workers to finish
    for p in procs:
        p.join()

    # A failed run under `abort_on_failure` ends the study here.
    if len(aborts):
        failure = aborts[0]
        n_done = len(D_shared['X'])
        log.error(
            'Study stopped: a run failed and abort_on_failure is set. '
            f'{n_done} evaluation{"" if n_done == 1 else "s"}, initial samples '
            'included, completed before it stopped.'
        )
        raise failure

    # Collect final results
    D_final = dict(D_shared)
    logs = list(log_list)
    T_elapsed = [t - T0 for t in list(T)]
    if converged.is_set():
        log.info(
            f'Stopped early by patience after {len(D_final["X"]) - n_init} of '
            f'{max_len - n_init} optimisation steps'
        )

    # A worker that dies mid-run leaves the run looking complete. Report.
    died = [wid for wid, p in enumerate(procs) if p.exitcode != 0]
    if died:
        names = ', '.join(str(wid) for wid in died)
        log.error(
            f'{len(died)} of {n_workers} workers stopped before the evaluation budget '
            f'was reached (workers {names}). Their exit codes were '
            f'{[procs[wid].exitcode for wid in died]}.'
            f' Results are based on {len(D_final["X"])} evaluations '
            f'rather than the {max_len} requested.'
        )
    # Nothing was added to the initial sample, so there is no optimisation to
    # report and the best-fit summary would describe the initial design alone.
    if len(D_final['X']) <= n_init:
        if died:
            cause = (
                f'{len(died)} of {n_workers} workers stopped early; see the messages '
                'above for the cause.'
            )
        else:
            # Every worker exited on its first budget check.
            n_steps = max_len - n_init
            cause = (
                f'No worker failed: the config asks for {n_steps} optimisation '
                f'step{"" if n_steps == 1 else "s"} across {n_workers} workers. '
                f'Raise n_steps to at least n_workers ({n_workers}).'
            )
        raise RuntimeError('No optimisation steps completed. ' + cause)

    return D_final, logs, T_elapsed
