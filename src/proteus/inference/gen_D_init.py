"""Initial dataset for Bayesian optimisation, from Halton samples of the parameter
box or from a precomputed grid, saved as `init.csv` in the study output.
"""

from __future__ import annotations

import logging
import os
import time
from multiprocessing import Event, Pool
from pathlib import Path

import numpy as np
import toml
import torch
from scipy.stats.qmc import Halton

from proteus.inference.objective import child_timeout_s, eval_obj, prot_builder
from proteus.inference.transforms import normalize_parameters
from proteus.inference.utils import save_dataset_csv
from proteus.utils.coupler import get_proteus_directories, read_helpfile_table
from proteus.utils.helper import recursive_get

# Use double precision for all tensor computations
dtype = torch.double
log = logging.getLogger('fwl.' + __name__)

# Stop signal shared by the initial-sampling pool, set by the first sample that
# raises. Installed in each pool worker by `_init_pool_worker`.
_stop = None


def create_init(config):
    """Create the initial BO dataset using sampling or a precomputed grid.

    Parameters
    ----------
    - config (dict): Inference config containing `init_grid`, `init_samps`,
      parameter bounds, observables, output location, and seed.

    Returns
    ----------
    - int: Number of initial samples written to `init.csv`.

    Raises:
        ValueError: If an explicit sample count is requested but < 2.
    """

    # validate options
    init_grid = str(config['init_grid'])
    if init_grid.lower().strip() == 'none':
        init_grid = None
        init_samps = int(config['init_samps'])
        if init_samps < 1:
            init_samps = int(config['n_workers'])
    else:
        init_grid = get_proteus_directories(init_grid)['output']
        init_samps = None

    # create new initial guess data by sampling bounds
    if init_samps:
        log.info('Source for initial guess: sampling parameter space')
        log.info(f'    nsamp = {init_samps}')
        n_init = sample_from_bounds(
            config['output'],
            config['ref_config'],
            config['parameters'],
            config['observables'],
            init_samps,
            config['seed'],
            config['n_workers'],
            config['failure_codes'],
        )

    # read from grid
    else:
        log.info('Source for initial guess: pre-computed grid')
        log.info(f'    grid = {init_grid}')
        n_init = sample_from_grid(
            config['output'], config['parameters'], config['observables'], init_grid
        )

    return n_init


def sample_from_grid(output: str, params: dict, observables: dict, grid_dir: str):
    """Build initial BO data from an existing PROTEUS grid.

    Reads `case_*` directories in `grid_dir`, extracts parameter values from
    each `init_coupler.toml`, computes objective values from the final row of
    each `runtime_helpfile.csv`, normalizes inputs to [0, 1], and writes
    `init.csv` to the inference output directory.

    Parameters
    ----------
    - output (str): Inference output directory (relative or absolute).
    - params (dict): Parameter bounds used for normalization.
    - observables (dict): Target observables used for objective evaluation.
    - grid_dir (str): Directory containing `case_*` precomputed runs.

    Returns
    ----------
    - int: Number of initial samples written.
    """

    # We need evaluate the objective function at each grid point to provide initial samples
    #     They are normalised to [0,1] within the bounds of each parameter's axis

    # First, read data and config files from the grid
    grid_path = Path(grid_dir)
    cases = sorted(grid_path.glob('case_*'))
    helps = []
    confs = []
    for c in cases:
        # Data
        helps.append(read_helpfile_table(c / 'runtime_helpfile.csv'))

        # Config
        with open(c / 'init_coupler.toml', 'r') as f:
            confs.append(toml.load(f))

    # List of parameter keys for ordering
    keys = list(params.keys())

    # Determine problem dimension (number of parameters)
    dims = len(keys)

    # Determine number of samples (number of grid points)
    nsamp = len(helps)

    # Parameter bounds
    bounds = torch.tensor(
        [[params[k][0] for k in keys], [params[k][1] for k in keys]], dtype=dtype
    )

    # Generate parameter points at which we will evaluate the objective
    #     Each of the parameters are evaluated in space 0-1, normalised to the bounds
    #     This variable is 2D, with shape [nsamp, dims]
    X = torch.zeros(nsamp, dims, dtype=dtype)
    Y = torch.zeros(nsamp, 1, dtype=dtype)
    for i in range(nsamp):
        # Get input parameters from grid configs
        raw_x = [recursive_get(confs[i], k.split('.')) for k in keys]
        raw_x = torch.tensor(raw_x, dtype=dtype)

        # Generate normalised INPUT parameters
        nrm_x = normalize_parameters(raw_x, bounds, keys).flatten()
        X[i, :] = nrm_x[:]  # store (list of floats)

        # Get values of OUTPUT observables from grid point data (list of floats)
        obs_y = helps[i].iloc[-1][observables.keys()].T

        # Evaluate objective and store (float)
        Y[i] = eval_obj(obs_y, observables)

    log.info(f'Generated initial dataset with {nsamp} points in {dims}-dim space')

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)['output']
    D_init_path = Path(proteus_out) / 'init.csv'
    save_dataset_csv(X, Y, str(D_init_path))

    # Return number of samples
    return len(Y.flatten())


def _init_pool_worker(stop) -> None:
    """Give an initial-sampling pool worker the batch's shared stop signal."""
    global _stop
    _stop = stop


def f_aug(x, iter, builder_args):
    """Evaluate a single initial sample using a temporary objective wrapper.

    Any error raised here fails the whole batch, including a failed run under
    `abort_on_failure`. The first one sets the stop signal, so samples not yet
    started return at once instead of running a simulation whose result would
    be discarded. Runs already in progress finish.

    Parameters
    ----------
    - x (torch.Tensor): Candidate input of shape (1, d) in normalized space.
    - iter (int): Iteration index used to namespace output files.
    - builder_args (dict): Context forwarded to `prot_builder`.

    Returns
    ----------
    - torch.Tensor | None: Objective value tensor with shape (1, 1), or None
      for a sample skipped because the batch is stopping.
    """
    if _stop is not None and _stop.is_set():
        return None
    f = prot_builder(
        parameters=builder_args['parameters'],
        observables=builder_args['observables'],
        worker=-1,
        iter=iter,
        ref_config=builder_args['ref_config'],
        output=builder_args['output'],
        failure_codes=builder_args['failure_codes'],
    )
    try:
        return f(x)
    except Exception:
        if _stop is not None:
            _stop.set()
        raise


def _pool_timeout(n_tasks: int, n_workers: int) -> float | None:
    """Total timeout for the initial-sampling pool, or None when disabled.

    Each worker runs about ``ceil(n_tasks / n_workers)`` child PROTEUS runs in
    sequence, each bounded by the per-child timeout, so the batch is allowed
    that long plus a fixed margin.
    """
    per_child = child_timeout_s()
    if per_child is None:
        return None
    per_worker = int(np.ceil(n_tasks / max(n_workers, 1)))
    return per_worker * per_child + 300.0


def sample_from_bounds(
    output: str,
    ref_config: str,
    params: dict,
    observables: dict,
    nsamp: int,
    seed: int,
    n_workers: int,
    failure_codes: list[int],
) -> int:
    """Generate initial BO data by evaluating Halton samples in parameter space.

    Parameters
    ----------
    - output (str): Inference output directory.
    - ref_config (str): Reference PROTEUS config file.
    - params (dict): Parameter bounds for inference.
    - observables (dict): Target observables for objective evaluation.
    - nsamp (int): Number of initial samples to evaluate.
    - seed (int): RNG seed for Halton sequence generation.
    - n_workers (int): Number of parallel workers to use for evaluation.
    - failure_codes (list[int]): PROTEUS status codes that complete normally but
      that this study excludes from the fit.

    Returns
    ----------
    - int: Number of initial samples written.
    """

    # Check number of workers
    if n_workers < 1:
        raise ValueError('Number of workers must be at least 1')
    elif n_workers >= os.cpu_count():
        n_workers = os.cpu_count() - 1
        log.warning(f'Number of workers reduced to {n_workers}')

    # Determine problem dimension (number of parameters)
    dims = len(params)

    # prepare parallel proteus runs
    builder_args = dict(
        parameters=params,
        observables=observables,
        ref_config=ref_config,
        output=output,
        failure_codes=failure_codes,
    )

    # Halton points in [0, 1]^d, shape [nsamp, dims], each axis normalised to its bounds
    sampler = Halton(d=dims, rng=np.random.default_rng(seed), scramble=True)
    X = torch.tensor(sampler.random(n=nsamp), dtype=dtype)

    # Evaluate the objective at each sample, in parallel
    aug_args = [(x[None, :], i, builder_args) for i, x in enumerate(X)]

    t0 = time.perf_counter()
    # A skipped sample only occurs once another has raised, and that error is
    # what `get` raises, so no None reaches the dataset.
    stop = Event()
    with Pool(processes=n_workers, initializer=_init_pool_worker, initargs=(stop,)) as pool:
        async_result = pool.starmap_async(f_aug, aug_args)
        # Bound the whole batch so a worker that wedges outside the per-child
        # subprocess timeout cannot hang the run indefinitely. Leaving the Pool
        # context terminates any still-running workers if this raises.
        results = async_result.get(timeout=_pool_timeout(len(aug_args), n_workers))
    t1 = time.perf_counter()

    log.info(f'Initial sampling took {t1 - t0:.2f}s')

    Y = torch.vstack(results)

    log.info(f'Generated initial dataset with {nsamp} points in {dims}-dim space')

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)['output']
    D_init_path = Path(proteus_out) / 'init.csv'
    save_dataset_csv(X, Y, str(D_init_path))

    # Return number of samples
    return len(Y.flatten())
