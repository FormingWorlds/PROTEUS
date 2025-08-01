"""Main entry point for asynchronous Bayesian optimization pipeline with PROTEUS.

This module loads configuration parameters, sets up the environment for parallel processing,
executes the optimization, and handles result printing and checkpointing.

Example:
    python main.py --config BO_config.toml
"""
from __future__ import annotations

import os

# Prevent workers from using each other's CPUs to avoid oversubscription and improve performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # for Mac

# system libraries
import time
from datetime import datetime

import toml
import torch
from async_BO import checkpoint, parallel_process
from objective import prot_builder
from gen_D_init import create_init
from utils import print_results

# proteus libraries
from proteus.utils.coupler import get_proteus_directories

# Use double precision for all tensor computations
dtype = torch.double

# Entry point for inference scheme
def run_inference(config):

    # dictionary of directories
    dirs = get_proteus_directories(config["output"])

    # path to reference config
    config["ref_config"] = os.path.join(dirs["proteus"], config["ref_config"])
    if not os.path.isfile(config["ref_config"]):
        raise FileNotFoundError("Cannot find reference config: " + config["ref_config"])

    # Create output directory and save a timestamped copy of the config
    os.makedirs(dirs["output"], exist_ok=True)
    with open(os.path.join(dirs["output"], "config.toml"), "w") as file:
        timestamp = datetime.now().astimezone().isoformat()
        file.write(f"# Created: {timestamp}\n\n")
        toml.dump(config, file)

    # Ensure there are enough CPU cores for the specified number of workers
    assert os.cpu_count() - 1 >= config["n_workers"], \
        f"Not enough CPU cores for {config['n_workers']} workers"


    # Create initial guess data through the requested method
    create_init(config)

    print("Starting optimisation\n")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run asynchronous BO with PROTEUS.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML config file containing BO parameters and settings"
    )
    args = parser.parse_args()

    # Load configuration from TOML file
    with open(args.config, "r") as file:
        config = toml.load(file)

    # run inference scheme
    run_inference(config)
