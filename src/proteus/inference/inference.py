"""Main entry point for asynchronous Bayesian optimization pipeline with PROTEUS.

This module loads configuration parameters, sets up the environment for parallel processing,
executes the optimization, and handles result printing and checkpointing.
"""

from __future__ import annotations

# system libraries
import logging
import os
import shutil
import time

import toml
import torch

import proteus.inference.plot as plotBO

# bayesopt source files
from proteus.inference.async_BO import checkpoint, parallel_process
from proteus.inference.gen_D_init import create_init
from proteus.inference.objective import prot_builder
from proteus.inference.utils import print_results, str_time

# proteus libraries
from proteus.utils.coupler import get_proteus_directories
from proteus.utils.helper import safe_rm
from proteus.utils.logs import setup_logger

# Use double precision for all tensor computations
dtype = torch.double
log = logging.getLogger('fwl.' + __name__)


# Entry point for inference scheme, providing infererence-config dict
def run_inference(config):
    """Run the full asynchronous Bayesian inference workflow.

    This function prepares output directories, validates and snapshots the
    reference configuration, creates initial samples, executes the asynchronous
    BO loop, writes checkpoint outputs, and generates diagnostic/result plots.

    Parameters
    ----------
    - config (dict): Parsed inference configuration dictionary.

    Returns
    ----------
    - None
    """

    # dictionary of directories
    dirs = get_proteus_directories(config['output'])

    # Create output directory
    safe_rm(dirs['output'])
    os.makedirs(dirs['output'])

    # Setup logging
    setup_logger(
        logpath=os.path.join(dirs['output'], 'infer.log'),
        logterm=True,
        level=config['logging'],
    )

    # Starting message
    log.info(f'Inference started at {str_time()}')

    # Save a timestamped copy of the reference config
    with open(os.path.join(dirs['output'], 'copy.infer.toml'), 'w') as file:
        file.write(f'# Created: {str_time()}\n\n')
        toml.dump(config, file)

    # Ensure there are enough CPU cores for the specified number of workers
    if config['n_workers'] >= os.cpu_count():
        raise RuntimeError(f'Not enough CPU cores for {config["n_workers"]} workers')

    # Check path to reference config
    config['ref_config'] = os.path.join(dirs['proteus'], config['ref_config'])
    log.info(f'Reference config: {config["ref_config"]}')
    if not os.path.isfile(config['ref_config']):
        raise FileNotFoundError('Cannot find reference config: ' + config['ref_config'])

    # Update ref_config path to point to a copy, in case user removes the original file
    copy_config = os.path.join(os.path.join(dirs['output'], 'ref_config.toml'))
    shutil.copyfile(config['ref_config'], copy_config)
    config['ref_config'] = copy_config

    # Create plots directory (will not already exist)
    os.mkdir(os.path.join(dirs['output'], 'plots'))
    log.info(' ')

    # Create initial guess data through the requested method
    n_init = create_init(config)
    log.info(' ')

    # Maximum number of evaluations during inference (offset by initial evaluations)
    max_len = max(int(config['n_steps']), 1) + n_init

    log.info('Optimisation config:')
    log.info(f'    workers       = {config["n_workers"]}')
    log.info(f'    init samples  = {n_init}')
    log.info(f'    optim steps   = {config["n_steps"]}')
    log.info(f'    kernel        = {config["kernel"]}')
    log.info(f'    acquisition   = {config["acqf"]}')
    log.info(' ')
    t_0 = time.perf_counter()

    # Execute the parallel BO process
    D_final, logs, Ts = parallel_process(
        prot_builder,
        config['kernel'],
        config['acqf'],
        config['n_workers'],
        max_len,
        config['output'],
        config['seed'],
        config['ref_config'],
        config['observables'],
        config['parameters'],
    )

    t_1 = time.perf_counter()
    log.info(f'This took: {t_1 - t_0:.2f} seconds')
    log.info('-----------------------------------')

    # Print summary of true vs. simulated observables and inferred parameters
    best_config = print_results(D_final, logs, config, dirs['output'], n_init)

    # Save final data, logs, and timestamps for later analysis
    log.info(f'Saving results: {dirs["output"]}')
    checkpoint(D_final, logs, Ts, dirs['output'])

    # Make plots
    log.info('Making plots')
    plotBO.plots_perf_timeline(logs, dirs['output'], n_init)
    plotBO.plots_perf_converge(D_final, Ts, n_init, dirs['output'])
    plotBO.plot_result_objective(D_final, config['parameters'], n_init, dirs['output'])
    plotBO.plot_result_correlation(config['parameters'], config['observables'], dirs['output'])

    # Make PROTEUS plots for best fitting case
    plotBO.plot_proteus(best_config)

    # Done
    log.info(f'Inference completed at {str_time()}')


def infer_from_config(config_fpath: str):
    """Load a TOML config file and run inference.

    Parameters
    ----------
    - config_fpath (str): Path to an inference TOML configuration file.

    Returns
    ----------
    - None
    """

    # Load configuration from TOML file
    print(f'Inference config: {config_fpath}')
    with open(config_fpath, 'r') as file:
        config = toml.load(file)

    # run inference scheme
    run_inference(config)
