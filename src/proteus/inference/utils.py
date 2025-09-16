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

import numpy as np
import pandas as pd
import toml
import torch
from botorch.utils.transforms import unnormalize

# Use double precision for tensor computations
dtype = torch.double

def get_nested(config: dict, key: str, sep: str = "."):
    """Retrieve a value from a nested dictionary using a dot-separated key path.

    Args:
        config (dict): A potentially nested dictionary.
        key (str): Dot-separated path to the target value (e.g., "struct.mass_tot").
        sep (str): Separator for key segments. Defaults to '.'.

    Returns:
        The value located at the specified nested key.

    Raises:
        KeyError: If any segment in the path is not found.
    """
    val = config
    for part in key.split(sep):
        val = val[part]  # Drill down into nested dictionary
    return val


def flatten(d, parent_key: str = '', sep: str = '.'):
    """Flatten a nested dictionary to a single-level dict with dot-separated keys.

    Args:
        d (dict): The nested dictionary to flatten.
        parent_key (str): Prefix for keys during recursion (used internally).
        sep (str): Separator to use between key segments.

    Returns:
        dict: A flattened dictionary where nested keys are joined by `sep`.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recurse into sub-dictionary
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            # Leaf node: record the value
            items.append((new_key, v))
    return dict(items)


def print_results(D, logs, config, output, n_init):
    """Identify the best evaluation and print its observables and inferred parameters.

    Finds the index of the maximum objective value in D['Y'], then:
      1. Reads the corresponding runtime_helpfile.csv to get simulated observables.
      2. Loads and flattens the input TOML to recover parameter values.
      3. Prints true observables, simulated outputs, and inferred inputs.

    Args:
        D (dict): Contains 'X' and 'Y', lists of inputs and objective values.
        logs (list of dict): Log entries with keys 'worker', 'task_id', etc.
        config (dict): Original config with 'observables' and 'parameters' mappings.
        output (str): Path to output directory for this entire inference call.
        n_init (int): Number of initial guess data points.
    """
    # Convert list of objective values to tensor
    X = D["X"]
    Y = D["Y"]

    # Find best index, ignoring the initial points
    i_opt: int = Y[n_init:].argmax() + n_init
    log_opt = logs[i_opt]
    J_opt: float = Y[i_opt].item()

    # Extract identifiers of the best run
    w = log_opt["worker"]
    id = log_opt["task_id"]  # iteration index

    # Read simulator output for this run
    out_path = f"{output}/workers/w_{w}/i_{id}/runtime_helpfile.csv"
    df = pd.read_csv(out_path, delimiter=r"\s+")

    # True observables from config
    true_y = pd.Series(config["observables"])
    observables = list(true_y.keys())

    # Simulated observables: last row of the CSV
    sim_opt = list(df.iloc[-1][observables].T)

    # Load and flatten the input TOML for this run
    in_path = f"{output}workers/w_{w}/i_{id}/init_coupler.toml"
    with open(in_path, "r") as f:
        input = toml.load(f)
    input = flatten(input)  # flatten nested config

    # Extract inferred parameter values in original order
    params = config["parameters"].keys()

    # Print summary to console, account for python index vs step index and n_init
    print(f"Best case sampled at step {i_opt+1-n_init} has J={J_opt:+.4f}")
    print(f"With config at {in_path}")
    print(f"{'Observables':18s} | True        | Simulated   | Diff %")
    for i,k in enumerate(observables):
        tru = true_y[k]
        obs = sim_opt[i]
        dif = 100*(obs-tru)/tru
        print(f"{k:18s}   {tru:8.4e}    {obs:8.4e}    {dif:+.3f}")
    print("")
    print(f"{'Parameters':28s} | Simulated val @ best J")
    for i,k in enumerate(list(params)):
        print(f"{k:28s}   {str(input[k]):20s}")
    print("-----------------------------------")
    print(" ")

    # Print parameter statistics
    d = len(params)
    bounds = torch.tensor([[list(config["parameters"].values())[i][j] for i in range(d)]
                            for j in range(2)])
    # remove intial data
    X_samp = np.array(unnormalize(X, bounds), copy=None, dtype=float)[n_init:,:]
    print(f"Ensemble statistics (N={len(X_samp)})")
    print(f"{'Parameters':28s} | Median ± stddev")
    for i,k in enumerate(list(params)):
        x_med = np.median(X_samp[:,i])
        x_std = np.std(X_samp[:,i])
        print(f"{k:28s}   {x_med:.4f} ± {x_std:.4f}")
    print("-----------------------------------")
    print(" ")
