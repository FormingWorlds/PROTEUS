"""Main entry point for asynchronous Bayesian optimization pipeline with PROTEUS.

This module loads configuration parameters, sets up the environment for parallel processing,
executes the optimization, and handles result printing and checkpointing.
"""
from __future__ import annotations

# system libraries
import os
import time
from datetime import datetime

import toml
import torch

# proteus libraries
from proteus.utils.coupler import get_proteus_directories
from proteus.utils.helper import safe_rm

# bayesopt source files
from proteus.inference.async_BO import checkpoint, parallel_process
from proteus.inference.gen_D_init import create_init
from proteus.inference.objective import prot_builder
from proteus.inference.utils import print_results
import proteus.inference.plot as plotBO

# Use double precision for all tensor computations
dtype = torch.double

# Entry point for inference scheme, providing infererence-config dict
def run_inference(config):

    # dictionary of directories
    dirs = get_proteus_directories(config["output"])

    # path to reference config
    config["ref_config"] = os.path.join(dirs["proteus"], config["ref_config"])
    if not os.path.isfile(config["ref_config"]):
        raise FileNotFoundError("Cannot find reference config: " + config["ref_config"])

    # Create output directory
    safe_rm(dirs["output"])
    os.makedirs(dirs["output"])

    # Save a timestamped copy of the config
    with open(os.path.join(dirs["output"], "copy.infer.toml"), "w") as file:
        timestamp = datetime.now().astimezone().isoformat()
        file.write(f"# Created: {timestamp}\n\n")
        toml.dump(config, file)

    # Ensure there are enough CPU cores for the specified number of workers
    if config["n_workers"] >= os.cpu_count():
        raise RuntimeError(f"Not enough CPU cores for {config['n_workers']} workers")

    # Create plots directory (will not already exist)
    os.mkdir(os.path.join(dirs["output"], "plots"))

    # Create initial guess data through the requested method
    n_init = create_init(config)

    print(f"Starting optimisation with {config["n_workers"]} workers")
    t_0 = time.perf_counter()

    # Execute the parallel BO process
    D_final, logs, Ts = parallel_process(
        prot_builder,
        config["kernel"],
        config["n_restarts"],
        config["n_samples"],
        config["n_workers"],
        config["max_len"],
        config["output"],
        config["ref_config"],
        config["observables"],
        config["parameters"]
    )

    t_1 = time.perf_counter()
    print(f"This took: {(t_1 - t_0):.2f} seconds\n")

    # Print summary of true vs. simulated observables and inferred parameters
    print_results(D_final, logs, config, dirs["output"])

    # Save final data, logs, and timestamps for later analysis
    checkpoint(D_final, logs, Ts, dirs["output"])

    # Make plots
    plotBO.plots_perf_timeline(logs, dirs["output"], n_init)
    plotBO.plots_perf_converge(D_final, Ts, n_init, dirs["output"])
    plotBO.plot_result_objective(D_final, config["parameters"], dirs["output"])


def infer_from_config(config_fpath:str):
    '''Run inference scheme according to the configuration provided'''

    # Load configuration from TOML file
    with open(config_fpath, "r") as file:
        config = toml.load(file)

    # run inference scheme
    run_inference(config)
