"""Utilities for configuration access and optimization result reporting.

This module provides helper functions to:

    * Extract values from nested dictionaries using dot-separated keys.
    * Flatten nested dictionaries into a single-level dict with compound keys.
    * Identify the best Bayesian optimization run and display true vs.
      simulated observables and inferred parameter values.

Functions:
    get_nested: Retrieve a nested value by a dot-separated key path.
    flatten: Flatten a nested dict into a single-level dict with dot-separated keys.
    print_results: Select the best run and print its observables and parameters.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from math import log as math_log
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.priors.torch_priors import LogNormalPrior

from proteus.inference.objective import EPS_CLIP, eval_obj
from proteus.inference.transforms import unnormalize_parameters
from proteus.utils.constants import gas_list

# Use double precision for tensor computations
dtype = torch.double
log = logging.getLogger('fwl.' + __name__)


def str_time():
    return datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')


def get_nested(config: dict, key: str, sep: str = '.'):
    """Retrieve a value from a nested dictionary using a dot-separated key path.

    Parameters
    ----------
    - config (dict): A potentially nested dictionary.
    - key (str): Dot-separated path to the target value (e.g., "planet.mass_tot").
    - sep (str): Separator for key segments. Defaults to '.'.

    Returns
    ----------
    - object: Value located at the specified nested key.

    Raises:
        KeyError: If any segment in the path is not found.
    """
    val = config
    for part in key.split(sep):
        val = val[part]  # Drill down into nested dictionary
    return val


def flatten(d, parent_key: str = '', sep: str = '.'):
    """Flatten a nested dictionary to a single-level dict with dot-separated keys.

    Parameters
    ----------
    - d (dict): The nested dictionary to flatten.
    - parent_key (str): Prefix for keys during recursion (used internally).
    - sep (str): Separator to use between key segments.

    Returns
    ----------
    - dict: Flattened dictionary where nested keys are joined by `sep`.
    """
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            # Recurse into sub-dictionary
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            # Leaf node: record the value
            items.append((new_key, v))
    return dict(items)


def save_dataset_csv(
    X: torch.Tensor,
    Y: torch.Tensor,
    fpath: str,
) -> None:
    """Persist BO dataset tensors to a plain-text CSV file.

    The file contains one row per sample with columns `x_0...x_(d-1)` and `y`.

    Parameters
    ----------
    - X (torch.Tensor): Input samples, shape (n, d).
    - Y (torch.Tensor): Objective values, shape (n, 1).
    - fpath (str): Output CSV path.

    Returns
    ----------
    - None
    """
    if X.ndim != 2:
        raise ValueError(f'Expected X to be 2D, got shape {tuple(X.shape)}')
    if Y.ndim != 2 or Y.shape[1] != 1:
        raise ValueError(f'Expected Y shape (n, 1), got {tuple(Y.shape)}')
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'X and Y row counts differ: {X.shape[0]} != {Y.shape[0]}')

    data: dict[str, np.ndarray] = {f'x_{i}': X[:, i].cpu().numpy() for i in range(X.shape[1])}
    data['y'] = Y[:, 0].cpu().numpy()
    pd.DataFrame(data).to_csv(fpath, index=False)


def load_dataset_csv(fpath: str) -> dict[str, torch.Tensor]:
    """Load BO dataset tensors from a CSV created by `save_dataset_csv`.

    Parameters
    ----------
    - fpath (str): Path to a dataset CSV containing `x_*` columns and `y`.

    Returns
    ----------
    - dict[str, torch.Tensor]: Dictionary with `X` (n, d) and `Y` (n, 1).
    """
    df = pd.read_csv(fpath)
    x_cols = sorted(
        [c for c in df.columns if c.startswith('x_')], key=lambda col: int(col.split('_', 1)[1])
    )
    if not x_cols:
        raise ValueError(f"No 'x_<index>' columns found in dataset: {fpath}")
    if 'y' not in df.columns:
        raise ValueError(f"Missing 'y' column in dataset: {fpath}")

    X = torch.tensor(df[x_cols].to_numpy(), dtype=dtype)
    Y = torch.tensor(df[['y']].to_numpy(), dtype=dtype)
    return {'X': X, 'Y': Y}


def print_results(D, logs, config, output, n_init):
    """Identify the best evaluation and log its observables and inferred parameters.

    Finds the index of the maximum objective value in D['Y'], then:
      1. Reads the corresponding runtime_helpfile.csv to get simulated observables.
      2. Loads and flattens the input TOML to recover parameter values.
      3. Logs true observables, simulated outputs, and inferred inputs.

    Parameters
    ----------
    - D (dict): Contains 'X' and 'Y', lists of inputs and objective values.
    - logs (list[dict]): Log entries with keys 'worker', 'task_id', etc.
    - config (dict): Original config with observables and parameter mappings.
    - output (str): Path to output directory for this inference run.
    - n_init (int): Number of initial guess data points.

    Returns
    ----------
    - str: Path to the input TOML of the best fitting case.
    """
    # Convert list of objective values to tensor
    X = D['X']
    Y = D['Y']

    # Find best index, ignoring the initial points
    i_opt: int = Y[n_init:].argmax() + n_init
    log_opt = logs[i_opt]
    J_opt: float = Y[i_opt].item()

    # Extract identifiers of the best run
    w = log_opt['worker']
    id = log_opt['task_id']  # iteration index

    # Read simulator output for this run
    out_path = Path(output) / 'workers' / f'w_{w}' / f'i_{id}' / 'runtime_helpfile.csv'
    df = pd.read_csv(out_path, delimiter=r'\s+')

    # True observables from config
    true_y = pd.Series(config['observables'])
    observables = list(true_y.keys())

    # Simulated observables: last row of the CSV
    sim_opt = list(df.iloc[-1][observables].T)

    # Load and flatten the input TOML for this run
    in_path = Path(output) / 'workers' / f'w_{w}' / f'i_{id}' / 'init_coupler.toml'
    with open(in_path, 'r') as f:
        input = toml.load(f)
    input = flatten(input)  # flatten nested config

    # Extract inferred parameter values in original order
    param_keys = list(config['parameters'].keys())

    # Log summary, account for python index vs step index and n_init
    log.info(f'Best case was step {i_opt + 1 - n_init}, with J={J_opt:+.4f}')
    log.info(f'    config: {in_path}')
    log.info(' ')

    log.info(f'{"Observable":28s} |    True     |  Best fit   | Diff %')
    for i, k in enumerate(observables):
        tru = true_y[k]
        obs = sim_opt[i]
        denom = tru + EPS_CLIP if tru >= 0 else tru - EPS_CLIP
        dif = 100 * (obs - tru) / denom
        log.info(f'{k:28s}   {tru:8.4e}    {obs:8.4e}    {dif:+.3f}')
    log.info(' ')

    log.info(f'{"Parameter":28s} | Best fitting value')
    for i, k in enumerate(param_keys):
        log.info(f'{k:28s}   {input[k]:g}')
    log.info(' ')

    # Log parameter statistics
    d = len(param_keys)
    bounds = torch.tensor(
        [[list(config['parameters'].values())[i][j] for i in range(d)] for j in range(2)]
    )
    # remove intial data
    X_samp = np.array(unnormalize_parameters(X, bounds, param_keys), copy=None, dtype=float)[
        n_init:, :
    ]
    log.info(f'Ensemble statistics (N={len(X_samp)})')
    log.info(f'{"Parameter":28s} |   Median  ±  stderr')
    for i, k in enumerate(param_keys):
        x_med = np.median(X_samp[:, i])
        x_err = np.std(X_samp[:, i]) / len(X_samp) ** 0.5
        log.info(f'{k:28s}   {x_med:g} ± {x_err:g}')
    log.info('-----------------------------------')

    return in_path


def get_acqf(name: str, gp: SingleTaskGP, best: float):
    """Build the acquisition function.

    Supports 'UCB', 'LogEI', and 'LogPI' acquisition functions.
    See docs: https://botorch.readthedocs.io/en/latest/acquisition.html

    Parameters
    ----------
    - name (str): Name of the acquisition function.
    - gp (SingleTaskGP): Fitted Gaussian Process model.
    - best (float): Current best observed value for EI/PI.

    Returns
    ----------
    - AcquisitionFunction: The constructed acquisition function.
    """
    if name == 'UCB':
        from botorch.acquisition.analytic import UpperConfidenceBound

        return UpperConfidenceBound(gp, beta=2.0)
    elif name == 'LogEI':
        from botorch.acquisition.analytic import LogExpectedImprovement

        return LogExpectedImprovement(gp, best_f=best)
    elif name == 'LogPI':
        from botorch.acquisition.analytic import LogProbabilityOfImprovement

        return LogProbabilityOfImprovement(gp, best_f=best)
    else:
        raise ValueError(f'Unsupported acquisition function: {name}')


def get_kernel_w_prior(
    ard_num_dims: int,
    batch_shape: torch.Size | None = None,
    use_rbf_kernel: bool = True,
    active_dims: Sequence[int] | None = None,
    nu: float | None = None,
) -> MaternKernel | RBFKernel:
    """Returns an RBF or Matern kernel with priors
    from  [Hvarfner2024vanilla]_.

    Parameters
    ----------
    - ard_num_dims (int): Number of feature dimensions for ARD.
    - batch_shape (torch.Size | None): Batch shape for the covariance module.
    - use_rbf_kernel (bool): Whether to use an RBF kernel.
    - active_dims (Sequence[int] | None): Input dimensions to compute covariances on.
    - nu (float | None): Smoothness parameter for Matern kernels.

    Returns
    ----------
    - MaternKernel | RBFKernel: Kernel configured with priors and constraints.
    """
    base_class = RBFKernel if use_rbf_kernel else partial(MaternKernel, nu=nu)
    lengthscale_prior = LogNormalPrior(
        loc=sqrt(2) + math_log(ard_num_dims) * 0.5, scale=sqrt(3)
    )
    base_kernel = base_class(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        ),
        # pyre-ignore[6] GPyTorch type is unnecessarily restrictive.
        active_dims=active_dims,
    )
    return base_kernel


def get_kernel(kernel: str, d: int) -> MaternKernel | RBFKernel:
    """Return a kernel with priors based on the specified type.

    Parameters
    ----------
    - kernel (str): Kernel type ('RBF', 'MAT1/2', 'MAT3/2', 'MAT5/2').
    - d (int): Number of input dimensions.

    Returns
    ----------
    - MaternKernel | RBFKernel: Configured kernel instance.
    """
    if kernel == 'RBF':
        kernel = get_kernel_w_prior(ard_num_dims=d, use_rbf_kernel=True)
    elif kernel == 'MAT1/2':
        kernel = get_kernel_w_prior(ard_num_dims=d, use_rbf_kernel=False, nu=0.5)
    elif kernel == 'MAT3/2':
        kernel = get_kernel_w_prior(ard_num_dims=d, use_rbf_kernel=False, nu=1.5)
    elif kernel == 'MAT5/2':
        kernel = get_kernel_w_prior(ard_num_dims=d, use_rbf_kernel=False, nu=2.5)
    else:
        raise ValueError('Unknown kernel, choices are RBF or MAT{1/2, 3/2, 5/2}')


def get_obs(out_csv, observables: list[str]):
    """Read the final simulated observables from a runtime helpfile.

    Applies an atmosphere-escaped fallback (zero VMR/MMW) when surface pressure
    has effectively collapsed.

    Parameters
    ----------
    - out_csv (str | Path): Path to `runtime_helpfile.csv`.
    - observables (list[str]): Observable columns to extract.

    Returns
    ----------
    - pandas.Series: Final-row observable values keyed by observable name.
    """
    df_row = pd.read_csv(out_csv, delimiter=r'\s+').iloc[-1]

    # Handle case where atmosphere has escaped
    #   Set VMRs and MMW to zero
    if df_row['P_surf'] < 1e-30:
        df_row['atm_kg_per_mol'] = 0.0
        for g in gas_list:
            df_row[g + '_vmr'] = 0.0

    return df_row[observables].T


def get_obj(true_obs: dict, n, path):
    """Assemble a 2D objective map from a square grid of precomputed runs.

    Parameters
    ----------
    - true_obs (dict): Target observables used in objective evaluation.
    - n (int): Grid side length; expects `n**2` worker outputs.
    - path (str | Path): Base path containing `workers/w_-1/i_<idx>/` runs.

    Returns
    ----------
    - torch.Tensor: Objective matrix of shape (n, n).
    """
    obs_names = true_obs.keys()
    base_path = Path(path)
    results = []
    for i in range(n**2):
        obs_path = base_path / 'workers' / 'w_-1' / f'i_{i}' / 'runtime_helpfile.csv'
        results.append(eval_obj(get_obs(obs_path, obs_names), true_obs))
    Y = torch.vstack(results).reshape(n, n)

    return Y
