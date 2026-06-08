"""Generate and save initial dataset for Bayesian optimization.

This script sets up parameter bounds and true observables for the PROTEUS simulator,
builds the objective function via `prot_builder`, generates a small random sample
of points in the normalized input space, evaluates the objective to obtain outputs,
and saves the resulting dataset to disk for use as the initial data in the BO pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
from botorch.utils.transforms import normalize
from scipy.stats.qmc import Halton

from proteus.inference.objective import eval_obj, prot_builder
from proteus.inference.utils import save_dataset_csv
from proteus.utils.coupler import get_proteus_directories
from proteus.utils.helper import recursive_get

# Use double precision for all tensor computations
dtype = torch.double
log = logging.getLogger('fwl.' + __name__)


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
        if init_samps < 2:
            raise ValueError('Initial guess dataset must contain >1 sample')
    else:
        init_grid = os.path.join(get_proteus_directories()['proteus'], 'output', init_grid)
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
        helps.append(pd.read_csv(c / 'runtime_helpfile.csv', delimiter=r'\s+'))

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
        nrm_x = normalize(raw_x, bounds).flatten()
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


def f_aug(x, iter, builder_args):
    """Evaluate a single initial sample using a temporary objective wrapper.

    Parameters
    ----------
    - x (torch.Tensor): Candidate input of shape (1, d) in normalized space.
    - iter (int): Iteration index used to namespace output files.
    - builder_args (dict): Context forwarded to `prot_builder`.

    Returns
    ----------
    - torch.Tensor: Objective value tensor with shape (1, 1).
    """
    f = prot_builder(
        parameters=builder_args['parameters'],
        observables=builder_args['observables'],
        worker=-1,
        iter=iter,
        ref_config=builder_args['ref_config'],
        output=builder_args['output'],
    )

    return f(x)


def sample_from_bounds(
    output: str,
    ref_config: str,
    params: dict,
    observables: dict,
    nsamp: int,
    seed: int,
    n_workers: int,
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

    # Build the PROTEUS-based objective function with fixed context
    #    This will be used to evaluate the objective function to provide initial samples

    # Determine problem dimension (number of parameters)
    dims = len(params)

    # prepare parallel proteus runs
    builder_args = dict(
        parameters=params, observables=observables, ref_config=ref_config, output=output
    )

    # Generate n random points in [0,1]^d and evaluate the objective
    #     Each of the parameters are evaluated in space 0-1, normalised to the bounds
    #     This variable is 2D, with shape [nsamp, dims]

    sampler = Halton(d=dims, seed=np.random.default_rng(seed), scramble=True)
    X = sampler.random(n=nsamp)
    X = torch.tensor(X, dtype=dtype)

    # X = torch.rand(nsamp, dims,
    #                generator=torch.manual_seed(seed), dtype=dtype)

    # Evaluate the objective function for each of the samples
    #     This variable is 1D, with shape [nsamp]

    aug_args = [(x[None, :], i, builder_args) for i, x in enumerate(X)]

    t0 = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(f_aug, aug_args)
    t1 = time.perf_counter()

    log.info(f'Initial sampling took {t1 - t0:.2f}s')

    Y = torch.vstack(results)

    # Y = torch.stack([f(x[None, :]) for x in X]).reshape(nsamp, 1)

    log.info(f'Generated initial dataset with {nsamp} points in {dims}-dim space')

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)['output']
    D_init_path = Path(proteus_out) / 'init.csv'
    save_dataset_csv(X, Y, str(D_init_path))

    # Return number of samples
    return len(Y.flatten())
