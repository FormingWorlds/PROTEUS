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

import pandas as pd
import pickle
import os
import toml
import torch

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


def print_results(D, logs, config, output):
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
    """
    # Convert list of objective values to tensor and find best index
    Y = D["Y"]
    i_opt = Y.argmax()
    log_opt = logs[i_opt]

    # Extract identifiers of the best run
    w = log_opt["worker"]
    id = log_opt["task_id"]  # iteration index

    # Read simulator output for this run
    out_path = f"{output}/workers/w_{w}/i_{id}/runtime_helpfile.csv"
    df = pd.read_csv(out_path, delimiter="\t")

    # True observables from config
    true_y = pd.Series(config["observables"])
    observables = list(true_y.keys())

    # Simulated observables: last row of the CSV
    sim_opt = df.iloc[-1][observables].T

    # Load and flatten the input TOML for this run
    in_path = f"{output}/workers/w_{w}/i_{id}/init_coupler.toml"
    with open(in_path, "r") as f:
        input = toml.load(f)
    input = flatten(input)  # flatten nested config

    # Extract inferred parameter values in original order
    params = config["parameters"].keys()
    par_opt = pd.Series({param: input[param] for param in params})

    # Print summary to console
    print("\nBest objective = ", Y[i_opt].item())
    print("\nTrue observables\n", true_y)
    print("\nSimulated observables\n", sim_opt)
    print("\nInferred inputs\n", par_opt)


def read_D_from_pkl(output):

    fpath = os.path.join(output, "data.pkl")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Cannot find inference result data: {fpath}")

    with open(fpath, 'rb') as hdl:
        data = pickle.load(hdl)

    return data
