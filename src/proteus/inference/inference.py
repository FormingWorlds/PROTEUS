"""Main entry point for asynchronous Bayesian optimization pipeline with PROTEUS.

This module loads configuration parameters, sets up the environment for parallel processing,
executes the optimization, and handles result printing and checkpointing.
"""

from __future__ import annotations

# system libraries
import copy
import logging
import math
import os
import shutil
import time

import toml
import torch

import proteus.inference.plot as plotBO

# proteus libraries
from proteus.config import (
    UnknownConfigKeyError,
    find_key_problems,
    format_orphan_message,
    read_config,
    structure_config,
)

# bayesopt source files
from proteus.inference.async_BO import checkpoint, parallel_process
from proteus.inference.failures import set_abort_on_failure, summarise_failures
from proteus.inference.gen_D_init import create_init
from proteus.inference.objective import (
    WORKER_CONFIG_OVERRIDES,
    apply_nested_updates,
    prot_builder,
    set_child_timeout,
)
from proteus.inference.utils import print_results, str_time
from proteus.utils.coupler import get_proteus_directories
from proteus.utils.helper import safe_rm
from proteus.utils.logs import setup_logger

# Use double precision for all tensor computations
dtype = torch.double
log = logging.getLogger('fwl.' + __name__)


# Stand-in for the per-run output folder when validating. Only the shape of the
# value matters here; the real path carries the worker and iteration indices.
_VALIDATION_OUT_PATH = 'workers/w_0/i_0'


def _reject_bad_config(raw: dict, label: str) -> None:
    """Apply the PROTEUS config checks to a raw dict, naming its source in errors.

    Parameters
    ----------
    - raw (dict): Raw TOML dict to check against the PROTEUS config schema.
    - label (str): Source description quoted back in any error message.

    Returns
    ----------
    - None

    Raises:
        UnknownConfigKeyError: If the dict carries keys outside the schema.
        ValueError: If a value fails schema validation.
    """
    orphans, mistyped = find_key_problems(raw)
    if orphans or mistyped:
        raise UnknownConfigKeyError(format_orphan_message(orphans, label, mistyped))
    structure_config(raw, label)


def parameter_bounds(parameters: dict) -> dict[str, tuple[float, float]]:
    """Return the swept-parameter ranges as ordered float pairs.

    Parameters
    ----------
    - parameters (dict): Mapping of dot-separated config keys to [min, max].

    Returns
    ----------
    - dict[str, tuple[float, float]]: Same keys, bounds as (min, max) floats.

    Raises:
        ValueError: If a range is not a pair of numbers, or does not increase.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for key, value in parameters.items():
        numeric = (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(v, (int, float)) for v in value)
        )
        if not numeric:
            raise ValueError(
                f"Bounds for inference parameter '{key}' must be a pair of numbers "
                f'[min, max], got {value!r}'
            )
        low, high = float(value[0]), float(value[1])
        # TOML admits `inf` and `nan`. An infinite bound passes the schema's
        # own range checks and then makes every unnormalised sample infinite.
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(
                f"Bounds for inference parameter '{key}' must be finite, got {value!r}"
            )
        if low >= high:
            raise ValueError(
                f"Bounds for inference parameter '{key}' must increase, got [{low:g}, {high:g}]"
            )
        bounds[key] = (low, high)
    return bounds


def validate_reference_config(ref_config: str, parameters: dict) -> None:
    """Reject a reference config the workers could not run, before any run starts.

    The file is checked exactly as PROTEUS checks its own input, and then again
    with every swept parameter set to each end of its range. A mistyped
    parameter name shows up as an unrecognised key, and a bound outside what
    the schema accepts shows up as a validation failure, both reported here
    rather than as a worker crash part-way through the study.

    Only the two ends of the range are checked, with every parameter moved
    together, so this is a screen rather than a proof. A range whose interior
    holds an invalid combination still passes. Conversely, a schema rule that
    couples two swept parameters can make one of the two variants invalid even
    though most of the space is fine: sweeping both `params.dt.minimum` and
    `params.dt.maximum` can put the minimum above the maximum at one end, and
    the study is refused. Sweep one side of such a pair, or widen the other.

    Parameters
    ----------
    - ref_config (str): Path to the reference PROTEUS config file.
    - parameters (dict): Mapping of dot-separated config keys to [min, max].

    Returns
    ----------
    - None

    Raises:
        UnknownConfigKeyError: If any variant carries keys outside the schema.
        ValueError: If a bound is malformed, or a variant fails validation.
    """
    bounds = parameter_bounds(parameters)
    raw = read_config(ref_config)

    # The file as the user wrote it.
    _reject_bad_config(raw, str(ref_config))

    # The file as a worker will run it. Bounds are cast to float to match what
    # the optimiser writes back into each worker's config.
    for label, index in (('lower', 0), ('upper', 1)):
        updates = {key: pair[index] for key, pair in bounds.items()}
        updates.update(WORKER_CONFIG_OVERRIDES)
        updates['params.out.path'] = _VALIDATION_OUT_PATH
        candidate = apply_nested_updates(copy.deepcopy(raw), updates)
        _reject_bad_config(candidate, f'{ref_config} (parameters at their {label} bounds)')


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

    # Everything that can be rejected from the config alone is rejected here,
    # because the next step empties the output folder and a study re-run after
    # a typo would otherwise destroy the previous study's results.
    if config['n_workers'] >= os.cpu_count():
        raise RuntimeError(f'Not enough CPU cores for {config["n_workers"]} workers')

    config['ref_config'] = os.path.join(dirs['proteus'], config['ref_config'])
    if not os.path.isfile(config['ref_config']):
        raise FileNotFoundError('Cannot find reference config: ' + config['ref_config'])

    validate_reference_config(config['ref_config'], config['parameters'])

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

    # Save a timestamped copy of the inference config
    infer_config = os.path.abspath(os.path.join(dirs['output'], 'copy.infer.toml'))
    with open(infer_config, 'w') as file:
        file.write(f'# Created: {str_time()}\n\n')
        toml.dump(config, file)
    log.info(f'Inference config: {infer_config}')

    # Bound each child PROTEUS run so one wedged simulation cannot hang the
    # whole batch. Tunable via the optional `child_timeout_s` config field;
    # plumbed to worker processes through the environment.
    set_child_timeout(config.get('child_timeout_s'))

    # Whether a failed simulation stops the study or is scored as a poor
    # sample. Defaults to scoring, because a sweep over a wide parameter box
    # is expected to reach combinations the simulator cannot integrate.
    set_abort_on_failure(bool(config.get('abort_on_failure', False)))

    # Default for configs that pre-date this field
    config.setdefault('failure_codes', [])

    log.info(f'Reference config: {config["ref_config"]}')

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
        config['failure_codes'],
    )

    t_1 = time.perf_counter()
    log.info(f'This took: {t_1 - t_0:.2f} seconds')
    log.info('-----------------------------------')

    # Account for the simulations that did not produce a usable result. Runs
    # before the best-fit summary, so the reader sees how much of the study was
    # real before reading what it concluded, and so the breakdown is still
    # reported when every evaluation failed and the summary refuses to print.
    summarise_failures(dirs['output'], len(D_final['X']))

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
    log.info(f'Inference config: {config_fpath}')
    with open(config_fpath, 'r') as file:
        config = toml.load(file)

    # run inference scheme
    run_inference(config)
