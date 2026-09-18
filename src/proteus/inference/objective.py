from __future__ import annotations

import logging
import os
import subprocess
from functools import partial
from pathlib import Path

import pandas as pd
import toml
import torch
from numpy import log10

from proteus.inference.failures import (
    ABORT_ON_FAILURE_ENV,
    CATEGORY_EXCLUDED,
    CATEGORY_FAILURE,
    CHILD_CONSOLE_SUFFIX,
    STATUS_MISSING,
    ProteusRunFailure,
    find_run_logfile,
    record_failure,
)
from proteus.inference.transforms import unnormalize_parameters
from proteus.utils.constants import element_list, gas_list
from proteus.utils.coupler import get_proteus_directories, variable_is_logarithmic
from proteus.utils.helper import ReadStatus

dtype = torch.double
EPS_CLIP = 1e-10
LOG_CLIP = 1e-20
BAD_OBJ_VALUE = -20.0
log = logging.getLogger('fwl.' + __name__)

# Per-child PROTEUS run timeout for inference workers. A single wedged child
# run would otherwise hang the whole batch with no diagnostic. Tunable per
# study via the inference config field `child_timeout_s`. The value is plumbed
# to the worker processes (which may be spawned, and so do not inherit module
# state) through the environment. A value of 0 or below disables the timeout.
DEFAULT_CHILD_TIMEOUT_S = 6 * 3600.0
_CHILD_TIMEOUT_ENV = 'PROTEUS_INFERENCE_CHILD_TIMEOUT_S'

# Config entries every worker overwrites in the reference config, regardless of
# which parameters are being swept. Shared with the startup validation so the
# configuration that is checked is the configuration that is run.
WORKER_CONFIG_OVERRIDES = {
    'params.out.plot_mod': 'none',
    'params.out.logging': 'WARNING',
}

# Config entries every run sets to the same thing, or to a value derived from
# the run index. Excluded from failure reports, which name the swept values.
_FIXED_PARAMETER_KEYS = set(WORKER_CONFIG_OVERRIDES) | {'params.out.path'}


def run_output_dir(output: str, worker: int, iter: int) -> tuple[Path, Path]:
    """Return the output folder of a single evaluation, relative and absolute.

    Parameters
    ----------
    - output (str): Study output folder, relative to the PROTEUS output root.
    - worker (int): Worker identifier. Initial samples use -1.
    - iter (int): Iteration identifier within that worker.

    Returns
    ----------
    - tuple[Path, Path]: The path as the simulator config records it, and the
      absolute path on disk.
    """
    out_dir = Path(output) / 'workers' / f'w_{worker}' / f'i_{iter}'
    return out_dir, Path(get_proteus_directories(str(out_dir))['output'])


def set_child_timeout(seconds: float | None = None) -> None:
    """Record the per-child PROTEUS timeout for inference worker processes.

    Stored in the environment so it is visible to the main process and to any
    spawned pool workers. ``None`` selects ``DEFAULT_CHILD_TIMEOUT_S``.
    """
    value = DEFAULT_CHILD_TIMEOUT_S if seconds is None else seconds
    os.environ[_CHILD_TIMEOUT_ENV] = str(value)


def child_timeout_s() -> float | None:
    """Return the per-child PROTEUS run timeout in seconds, or None to disable.

    Reads the value recorded by ``set_child_timeout`` from the environment so
    it is consistent across the main process and spawned pool workers. Falls
    back to ``DEFAULT_CHILD_TIMEOUT_S`` when unset; a value of 0 or below
    disables the timeout.
    """
    raw = os.environ.get(_CHILD_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_CHILD_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_CHILD_TIMEOUT_S
    return val if val > 0 else None


def apply_nested_updates(config: dict, updates: dict) -> dict:
    """Set dot-separated keys in a nested config dict, in place.

    Parameters
    ----------
    - config (dict): Nested configuration dictionary, modified in place.
    - updates (dict): Mapping of dot-separated key paths to new values.

    Returns
    ----------
    - dict: The same dictionary that was passed in.

    Raises:
        ValueError: If a key path descends through an entry that holds a value
            rather than a table.
    """
    for key, value in updates.items():
        parts = key.split('.')
        d = config
        for i, part in enumerate(parts[:-1]):
            d = d.setdefault(part, {})
            if not isinstance(d, dict):
                prefix = '.'.join(parts[: i + 1])
                raise ValueError(f"Cannot set '{key}': '{prefix}' holds a value, not a section")
        d[parts[-1]] = value
    return config


def update_toml(config_file: str, updates: dict, output_file: str) -> None:
    """Update values in a TOML configuration file.

    Loads the configuration from `config_file`, applies updates provided in the
    `updates` dict (supporting nested keys via dot notation), and writes the
    modified configuration to `output_file`.

    Parameters
    ----------
    - config_file (str): Path to the existing TOML file.
    - updates (dict): Mapping of keys to new values; nested keys via "section.key".
    - output_file (str): Destination path for the updated TOML file.

    Returns
    ----------
    - None
    """
    config_path = Path(config_file)
    output_path = Path(output_file)

    # Load existing config
    with open(config_path, 'r') as f:
        config = toml.load(f)

    # Apply nested updates
    apply_nested_updates(config, updates)

    # Ensure destination directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write updated config
    with open(output_path, 'w') as f:
        toml.dump(config, f)


def run_proteus(
    parameters: dict,
    worker: int,
    iter: int,
    observables: list[str],
    ref_config: str,
    output: str,
) -> tuple[dict, int]:
    """Run the PROTEUS simulator and return selected observables.

    Builds a per-run TOML file, invokes the `proteus` CLI, and reads the resulting CSV.

    Parameters
    ----------
    - parameters (dict): Parameter-value pairs to set in the simulation config.
    - worker (int): Worker identifier for directory organization.
    - iter (int): Iteration identifier for directory organization.
    - observables (list[str]): Names of output columns to return.
    - ref_config (str): Path to the reference TOML config template.
    - output (str): Path to output relative to PROTEUS output folder.

    Returns
    ----------
    - observables_dict (dict): Mapping of observable names to their simulated values.
    - status (int): Status code indicating the outcome of the simulation.

    Raises:
        ProteusRunFailure: If this particular run did not produce a usable
            result. Carries the status code, the path to the run's logfile and
            the parameters that produced it.
        RuntimeError: If the `proteus` command itself cannot be executed, which
            would affect every run rather than this one.
        KeyError: If a requested observable is absent from the output, which
            likewise applies to every run.
    """

    # Construct run-specific paths
    out_dir, out_abs = run_output_dir(output, worker, iter)
    out_cfg = out_abs / 'input.toml'
    out_csv = out_abs / 'runtime_helpfile.csv'

    # Ensure output directory exists
    out_abs.mkdir(parents=True, exist_ok=True)

    # Inject output path into simulation parameters
    parameters['params.out.path'] = str(out_dir)

    # Don't allow workers to make plots or logs
    parameters.update(WORKER_CONFIG_OVERRIDES)

    # Generate config
    update_toml(ref_config, parameters, str(out_cfg))

    # Generate environment
    env = dict(**os.environ)
    env['OMP_NUM_THREADS'] = '1'

    # Swept parameter values only, for the failure report. The output path and
    # the worker overrides are fixed for every run and add no diagnostic value.
    swept = {k: v for k, v in parameters.items() if k not in _FIXED_PARAMETER_KEYS}

    # A run that dies before its logger is configured leaves no logfile behind, so
    # this stream records it.
    console = out_abs.parent / f'{out_abs.name}{CHILD_CONSOLE_SUFFIX}'

    def _failure(reason: str, exit_code: int | None) -> ProteusRunFailure:
        """Assemble a failure report for this run.

        The status file is read here rather than at the point of the raise so
        that a crashed run is described by what PROTEUS recorded about itself,
        not only by its exit code.
        """
        return ProteusRunFailure(
            reason=reason,
            worker=worker,
            iter=iter,
            out_dir=str(out_abs),
            exit_code=exit_code,
            status=ReadStatus(out_abs),
            log_path=find_run_logfile(out_abs),
            console_path=str(console),
            parameters=swept,
        )

    command = ['proteus', 'start', '-c', str(out_cfg), '--offline']
    console.parent.mkdir(parents=True, exist_ok=True)
    # Opened outside the try so that a failure to create it is not mistaken
    # for the simulator being absent.
    stream = open(console, 'w')
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=child_timeout_s(),
        )
    except FileNotFoundError as err:
        # Applies to every run, not just this one, so it is not a sample that
        # can be scored badly and skipped.
        log.error(f"Cannot execute '{command[0]}': command not found")
        raise RuntimeError("Failed to run PROTEUS: 'proteus' command not found") from err
    except subprocess.TimeoutExpired as err:
        timeout = child_timeout_s()
        raise _failure(f'exceeded the {timeout} s timeout', exit_code=None) from err
    except subprocess.CalledProcessError as err:
        raise _failure('the simulator exited with an error', exit_code=err.returncode) from err
    finally:
        stream.close()

    # Re-write config in case simulator mutates or removes it
    update_toml(ref_config, parameters, str(out_cfg))

    # Read status file
    status = ReadStatus(out_abs)

    # Read simulator output. A run that exits cleanly but writes no usable
    # helpfile (killed mid-write, or stopped before the first row) is a failed
    # sample, not a crash of the study.
    try:
        df_row = dict(pd.read_csv(out_csv, delimiter=r'\s+').iloc[-1])
    except (
        FileNotFoundError,
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        IndexError,
    ) as err:
        # A truncated whitespace-delimited file usually presents as a ragged
        # row (ParserError) rather than an empty one, so both are caught.
        raise _failure(
            f'exited cleanly but produced no readable output ({out_csv.name})',
            exit_code=0,
        ) from err

    # Handle case where atmosphere has escaped
    #   Set VMRs and MMW to zero
    if bool(df_row['P_surf'] < 1e-30):
        df_row['atm_kg_per_mol'] = 0.0
        for g in gas_list:
            df_row[g + '_vmr'] = 0.0
        for e in element_list:
            df_row[f'{e}_kg_atm'] = 0.0

    # Make into dict
    try:
        observables_dict = {obs: df_row[obs] for obs in observables}
    except KeyError as e:
        raise KeyError(f"Requested observable '{e.args[0]}' not found") from e

    return observables_dict, status


def log_warp(sq_dist):
    """Map squared distances to the optimization objective scale.

    Parameters
    ----------
    - sq_dist (torch.Tensor): Non-negative squared distance tensor.

    Returns
    ----------
    - torch.Tensor: Warped score, larger for closer matches.
    """
    warped_dist = -torch.log10(sq_dist + EPS_CLIP)
    return warped_dist


def eval_obj(sim_dict, tru_dict):
    """Evaluate objective value from simulated and target observables.

    The metric compares each observable in either linear or log space,
    accumulates squared normalized differences, and applies `log_warp`.
    A small offset is applied to the denominator to avoid division by zero.

    Parameters
    ----------
    - sim_dict (dict): Simulated observable values.
    - tru_dict (dict): Target (reference) observable values.

    Returns
    ----------
    - torch.Tensor: Objective tensor of shape (1, 1).
    """

    sim_vals = []
    tru_vals = []
    for k in sim_dict.keys():
        # some variables scale logarithmically
        if variable_is_logarithmic(k):
            sim_vals.append(log10(max(sim_dict[k], LOG_CLIP)))
            tru_vals.append(log10(max(tru_dict[k], LOG_CLIP)))
        # others are just linear
        else:
            sim_vals.append(sim_dict[k])
            tru_vals.append(tru_dict[k])

    # Convert to tensor and reshape
    sim = torch.tensor(sim_vals, dtype=dtype).reshape(1, -1)
    true_y = torch.tensor(tru_vals, dtype=dtype).reshape(1, -1)

    # Compute normalized difference and squared distance
    denom = torch.where(true_y >= 0, true_y + EPS_CLIP, true_y - EPS_CLIP)
    diff = torch.ones_like(true_y) - sim / denom
    sq_dist = (diff**2).sum(dim=1, keepdim=True)

    obj = log_warp(sq_dist)
    return obj


def J(
    x: torch.Tensor,
    parameters: list[str],
    true_observables: dict[str, float],
    worker: int,
    iter: int,
    output: str,
    ref_config: str,
    failure_codes: list[int] = [],
) -> torch.Tensor:
    """Run PROTEUS, and then compute the objective value for a given normalized input.

    Transforms normalized `x` to raw parameters, runs the simulator,
    and computes the squared-error based objective:

        J = log_10(sum((1 - sim/true)^2) + eps)

    Parameters
    ----------
    - x (torch.Tensor): Normalized input tensor of shape (1, d).
    - parameters (list[str]): Ordered list of parameter keys corresponding to x.
    - true_observables (dict): Mapping of observable names to target values.
    - worker (int): Worker identifier.
    - iter (int): Iteration number.
    - output (str): Path to output folder relative to PROTEUS output folder.
    - ref_config (str): Reference TOML config path.
    - failure_codes (list[int]): PROTEUS status codes that complete normally but
      that this study excludes from the fit.

    Returns
    ----------
    - torch.Tensor: Objective value tensor of shape (1, 1).
    """

    # Map normalized x to raw parameter dict and run PROTEUS
    raw = {parameters[i]: x[0, i].item() for i in range(len(parameters))}
    try:
        sim_vals, sim_status = run_proteus(
            parameters=raw,
            worker=worker,
            iter=iter,
            observables=list(true_observables.keys()),
            ref_config=ref_config,
            output=output,
        )
    except ProteusRunFailure as failure:
        # A parameter combination the simulator cannot integrate is an
        # expected outcome of sweeping a wide box, so it is scored as a poor
        # sample and the study continues. Every such run is reported once,
        # and the full report goes to the failure record.
        # Recorded before the abort check, so an aborted study still leaves
        # the record of what stopped it.
        record_failure(get_proteus_directories(output)['output'], failure)
        if os.environ.get(ABORT_ON_FAILURE_ENV, '0') == '1':
            raise
        log.warning(failure.summary())
        log.debug(failure.report())
        return BAD_OBJ_VALUE * torch.ones((1, 1), dtype=dtype)

    # Runs that exit cleanly but stop in an error state, such as a run halted
    # through its keepalive file (status 25), or that never reach the main loop
    # (status 0 and 1). An unreadable status counts here too: the run's own
    # account of itself is missing, so its output cannot be trusted.
    failed = (20 <= sim_status <= 28) or (sim_status in (0, 1, STATUS_MISSING))

    # Runs that completed normally on an outcome this study does not fit
    # against, named by the `failure_codes` field of the inference config: a
    # run stopped by its clock limit (status 11) or one whose volatiles all
    # escaped (status 15), for instance. Nothing went wrong in such a run, so
    # it is scored as a poor sample but is not reported as a fault.
    excluded = (not failed) and (sim_status in failure_codes)

    # Either way the evaluation carries the failure score instead of a fit
    # quality, and is recorded so that the end-of-study tally covers it.
    if failed or excluded:
        _, out_abs = run_output_dir(output, worker, iter)
        # Built once, so the entry left on disk and the exception raised under
        # `abort_on_failure` describe the same run.
        failure = ProteusRunFailure(
            reason=(
                'exited cleanly but stopped in a failure state'
                if failed
                else 'completed on a status this study excludes'
            ),
            worker=worker,
            iter=iter,
            out_dir=str(out_abs),
            exit_code=0,
            status=sim_status,
            log_path=find_run_logfile(out_abs),
            parameters=raw,
            category=CATEGORY_FAILURE if failed else CATEGORY_EXCLUDED,
        )
        # Recorded before the abort check, so an aborted study still leaves
        # the record of what stopped it.
        record_failure(get_proteus_directories(output)['output'], failure)
        if failed:
            # A clean exit on an error status is as much a fault as a crash,
            # so it honours `abort_on_failure` the same way. An excluded
            # outcome never does: nothing went wrong in such a run.
            if os.environ.get(ABORT_ON_FAILURE_ENV, '0') == '1':
                raise failure
            log.warning(failure.summary())
        else:
            # Nothing went wrong in such a run, so it is reported at info
            # level and, like a fault, on one line.
            log.info(failure.summary())
        # The rest of the report is kept out of the study log
        log.debug(failure.report())
        return BAD_OBJ_VALUE * torch.ones((1, 1), dtype=dtype)

    # Compute value of objective function given these results
    return eval_obj(sim_vals, true_observables)


def prot_builder(
    parameters: dict[str, list[float]],
    observables: dict[str, float],
    worker: int,
    iter: int,
    output: str,
    ref_config: str,
    failure_codes: list[int] = [],
) -> callable:
    """Factory returning a BO-compatible objective function for PROTEUS inference.

    Precomputes bounds for unnormalization and embeds simulation context.

    Parameters
    ----------
    - parameters (dict): Mapping of parameter keys to [low, high] bounds.
    - observables (dict): Target observable values.
    - worker (int): Worker identifier.
    - iter (int): Iteration number (seed) for reproducibility.
    - output (str): Path to output folder relative to PROTEUS output folder.
    - ref_config (str): Reference TOML config path.
    - failure_codes (list[int]): PROTEUS status codes that complete normally but
      that this study excludes from the fit.

    Returns
    ----------
    - callable: Function f(x_norm) -> y_objective.
    """
    # Build bounds tensor for unnormalization
    param_keys = list(parameters.keys())
    d = len(param_keys)
    bounds = torch.tensor(
        [[list(parameters.values())[i][j] for i in range(d)] for j in range(2)], dtype=dtype
    )

    def f(x_norm: torch.Tensor) -> torch.Tensor:
        """Inference objective function accepting normalized inputs.

        Parameters
        ----------
        - x_norm (torch.Tensor): Input tensor in normalized [0, 1]^d space.

        Returns
        ----------
        - torch.Tensor: Objective value tensor of shape (1, 1).
        """
        # Convert normalized to raw inputs
        x_raw = unnormalize_parameters(x_norm, bounds, param_keys)

        # Partially apply J with fixed context
        J_context = partial(
            J,
            parameters=param_keys,
            true_observables=observables,
            worker=worker,
            iter=iter,
            ref_config=ref_config,
            output=output,
            failure_codes=failure_codes,
        )

        J_eval = J_context(x_raw)

        # Check J is finite
        if not torch.isfinite(J_eval).all():
            x_param = {param_keys[i]: x_raw[0, i].item() for i in range(d)}

            log.warning('Non-finite objective value')
            log.warning(f'    Raw input x: {x_param}')
            log.warning(f'    Reference config: {ref_config}')

        # Compute objective
        return J_eval

    return f
