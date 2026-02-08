"""Generate and save initial dataset for Bayesian optimization.

This script sets up parameter bounds and true observables for the PROTEUS simulator,
builds the objective function via `prot_builder`, generates a small random sample
of points in the normalized input space, evaluates the objective to obtain outputs,
and saves the resulting dataset to disk for use as the initial data in the BO pipeline.
"""

from __future__ import annotations

import os
import pickle
import time
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import toml
import torch
from botorch.utils.transforms import normalize
from scipy.stats.qmc import Halton

from proteus.inference.objective import eval_obj, prot_builder
from proteus.utils.coupler import get_proteus_directories
from proteus.utils.helper import recursive_get

# Use double precision for all tensor computations
dtype = torch.double


def create_init(config):
    """
    Create initial guess data file in output folder, using the specified method.
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
        print('Source for initial guess: sampling parameter space')
        print(f'    nsamp = {init_samps}')
        n_init = sample_from_bounds(
            config['output'],
            config['ref_config'],
            config['parameters'],
            config['observables'],
            init_samps,
            config['seed'],
        )

    # read from grid
    else:
        print('Source for initial guess: pre-computed grid')
        print(f'    grid = {init_grid}')
        n_init = sample_from_grid(
            config['output'], config['parameters'], config['observables'], init_grid
        )

    return n_init


def sample_from_grid(output: str, params: dict, observables: dict, grid_dir: str):
    """
    Create initial guess data, using pre-computed grid of models as input
    """

    # We need evaluate the objective function at each grid point to provide initial samples
    #     They are normalised to [0,1] within the bounds of each parameter's axis

    # First, read data and config files from the grid
    cases = glob(grid_dir + '/case_*/')
    helps = []
    confs = []
    for c in cases:
        # Data
        helps.append(pd.read_csv(c + 'runtime_helpfile.csv', delimiter=r'\s+'))

        # Config
        with open(c + 'init_coupler.toml', 'r') as f:
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

    # Package into dataset dict, to be pickled
    D = {'X': X, 'Y': Y}
    print(f'Generated initial dataset with {nsamp} points in {dims}-dim space')

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)['output']
    D_init_path = f'{proteus_out}/init.pkl'
    with open(D_init_path, 'wb') as f_out:
        pickle.dump(D, f_out)

    # Return number of samples
    return len(Y.flatten())


def f_aug(x, iter, builder_args):
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
    output: str, ref_config: str, params: dict, observables: dict, nsamp: int, seed: int
):
    """
    Create initial guess data, sampling the parameter space at random.
    """

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

    avail_cpus = os.cpu_count() - 1
    t0 = time.perf_counter()
    with Pool(processes=avail_cpus) as pool:
        results = pool.starmap(f_aug, aug_args)
    t1 = time.perf_counter()

    print(f'Initial sampling took {(t1 - t0):.2f}s')

    Y = torch.vstack(results)

    # Y = torch.stack([f(x[None, :]) for x in X]).reshape(nsamp, 1)

    # Package into dataset dict
    D = {'X': X, 'Y': Y}
    print(f'Generated initial dataset with {nsamp} points in {dims}-dim space')

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)['output']
    D_init_path = f'{proteus_out}/init.pkl'
    with open(D_init_path, 'wb') as f_out:
        pickle.dump(D, f_out)

    # Return number of samples
    return len(Y.flatten())
