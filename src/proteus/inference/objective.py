from __future__ import annotations

import logging
import math
import os
import subprocess
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
from proteus.inference.runner import ProteusRunner
from proteus.inference.transforms import unnormalize_parameters
from proteus.utils.constants import element_list, gas_list
from proteus.utils.coupler import get_proteus_directories, variable_is_logarithmic
from proteus.utils.helper import ReadStatus

dtype = torch.double
EPS_CLIP = 1e-10
LOG_CLIP = 1e-20
BAD_OBJ_VALUE = -20.0
# Completion statuses excluded from every fit, whatever `failure_codes` holds.
ALWAYS_EXCLUDED_STATUSES = frozenset({29})
log = logging.getLogger('fwl.' + __name__)

# Per-child run timeout, so one wedged run cannot hang the batch. Set per study
# by `child_timeout_s` (0 or below disables it) and passed to the workers,
# which may be spawned, through the environment.
DEFAULT_CHILD_TIMEOUT_S = 6 * 3600.0
_CHILD_TIMEOUT_ENV = 'PROTEUS_INFERENCE_CHILD_TIMEOUT_S'

# 'subprocess' starts one `proteus start` per evaluation; 'runner' reuses one
# PROTEUS process per worker, skipping the Julia load on all but the first.
DISPATCH_SUBPROCESS = 'subprocess'
DISPATCH_RUNNER = 'runner'
DISPATCH_MODES = (DISPATCH_SUBPROCESS, DISPATCH_RUNNER)
_DISPATCH_ENV = 'PROTEUS_INFERENCE_DISPATCH'
_RUNNER_MAX_JOBS_ENV = 'PROTEUS_INFERENCE_RUNNER_MAX_JOBS'

# This worker's reused process, replaced whenever it is stopped.
_RUNNER: ProteusRunner | None = None

# Config entries every worker overwrites in the reference config, regardless of
# which parameters are being swept. Shared with the startup validation so the
# configuration that is checked is the configuration that is run.
WORKER_CONFIG_OVERRIDES = {
    'params.out.plot_mod': 'none',
    'params.out.logging': 'WARNING',
    'params.out.archive_mod': 0,
}

# Folder inside the study output where workers reuse prepared spectral files,
# and the switch for it, passed to the workers through the environment.
SPECTRAL_CACHE_DIR = 'spectral_cache'
SPECTRAL_CACHE_ENV = 'PROTEUS_INFERENCE_SPECTRAL_CACHE'

# Config entries every run sets to the same thing, or to a value derived from
# the run index. Excluded from failure reports, which name the swept values.
_FIXED_PARAMETER_KEYS = set(WORKER_CONFIG_OVERRIDES) | {
    'params.out.path',
    'atmos_clim.spectral_cache',
}


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


def set_dispatch(mode: str | None = None, max_jobs: int | None = None) -> None:
    """Record how workers run each evaluation, and how long a runner is kept.

    Raises:
        ValueError: If `mode` is not one of DISPATCH_MODES.
    """
    mode = DISPATCH_SUBPROCESS if mode is None else str(mode)
    if mode not in DISPATCH_MODES:
        raise ValueError(
            f"Unknown inference dispatch '{mode}', expected one of {', '.join(DISPATCH_MODES)}"
        )
    os.environ[_DISPATCH_ENV] = mode
    os.environ[_RUNNER_MAX_JOBS_ENV] = str(0 if max_jobs is None else int(max_jobs))


def dispatch_mode() -> str:
    """The dispatch recorded by `set_dispatch`.
    Falls back to one process per evaluation.
    """
    mode = os.environ.get(_DISPATCH_ENV, DISPATCH_SUBPROCESS)
    return mode if mode in DISPATCH_MODES else DISPATCH_SUBPROCESS


def worker_runner(console: Path) -> ProteusRunner:
    """This worker's reused PROTEUS process, started on its first evaluation.

    Per-process state, so each worker holds its own and evaluations never queue
    behind one another. The console file is fixed at that first use.
    """
    global _RUNNER
    if _RUNNER is None:
        try:
            max_jobs = int(os.environ.get(_RUNNER_MAX_JOBS_ENV, '0'))
        except ValueError:
            max_jobs = 0
        _RUNNER = ProteusRunner(console, max_jobs=max_jobs)
    return _RUNNER


def close_worker_runner() -> None:
    """Stop this worker's reused PROTEUS process, if it has started one."""
    global _RUNNER
    if _RUNNER is not None:
        _RUNNER.stop()
        _RUNNER = None


def _run_in_new_process(out_cfg: Path, console: Path, env: dict, failure) -> None:
    """Run one simulation in a process of its own.

    `failure` builds this run's report from a reason and an exit code.

    Raises:
        ProteusRunFailure: If this run produced no usable result. The
            underlying error is chained, so the report names the command.
        RuntimeError: If the `proteus` command cannot be executed at all,
            which affects every run rather than this one.
    """
    command = ['proteus', 'start', '-c', str(out_cfg), '--offline']
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
        raise failure(f'exceeded the {timeout} s timeout', exit_code=None) from err
    except subprocess.CalledProcessError as err:
        raise failure('the simulator exited with an error', exit_code=err.returncode) from err
    finally:
        stream.close()


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


def spectral_cache_enabled() -> bool:
    """Whether workers share a spectral cache, as recorded by the parent process."""
    return os.environ.get(SPECTRAL_CACHE_ENV, '1') == '1'


def worker_spectral_cache(output: str, enabled: bool) -> str:
    """The `atmos_clim.spectral_cache` value every worker runs with.

    Set by the inference config's `spectral_cache` switch, whatever the
    reference config holds: a reference config copied from `all_options.toml`
    sets "none" without meaning to. Every evaluation that holds the star fixed
    builds the same prepared file, so enabled workers share one folder.

    Parameters
    ----------
    - output (str): Study output folder, relative to the PROTEUS output root.
    - enabled (bool): The inference config's `spectral_cache` switch.

    Returns
    ----------
    - str: The study's cache folder, or "none" when the cache is off.
    """
    if not enabled:
        return 'none'
    return str(Path(get_proteus_directories(output)['output']) / SPECTRAL_CACHE_DIR)


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
      Not modified; the run-specific entries are added to a copy.
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

    # Swept values for the failure report. The fixed entries go into a copy, so
    # the caller's dict is left unchanged.
    swept = {k: v for k, v in parameters.items() if k not in _FIXED_PARAMETER_KEYS}
    updates = dict(parameters)

    # Inject output path and spectral cache into simulation parameters
    updates['params.out.path'] = str(out_dir)
    updates['atmos_clim.spectral_cache'] = worker_spectral_cache(
        output, spectral_cache_enabled()
    )

    # Don't allow workers to make plots or logs
    updates.update(WORKER_CONFIG_OVERRIDES)

    # Generate config
    update_toml(ref_config, updates, str(out_cfg))

    # Generate environment
    env = dict(**os.environ)
    env['OMP_NUM_THREADS'] = '1'

    # Records a run that dies before its logger exists.
    if dispatch_mode() == DISPATCH_RUNNER:
        console = out_abs.parent / f'runner{CHILD_CONSOLE_SUFFIX}'
    else:
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

    console.parent.mkdir(parents=True, exist_ok=True)

    # Either one `proteus start` per evaluation, or a PROTEUS process this
    # worker keeps alive between evaluations.
    if dispatch_mode() == DISPATCH_RUNNER:
        outcome = worker_runner(console).run(out_cfg, child_timeout_s())
        if outcome is not None:
            raise _failure(*outcome)
    else:
        _run_in_new_process(out_cfg, console, env, _failure)

    # Re-write config in case simulator mutates or removes it
    update_toml(ref_config, updates, str(out_cfg))

    # Read status file
    status = ReadStatus(out_abs)

    # Read simulator output. A run that exits cleanly but writes no usable
    # helpfile (killed mid-write, stopped before the first row, or corrupted
    # on disk) is a failed sample, not a crash of the study.
    unreadable = (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        IndexError,
    )
    try:
        df_row = dict(pd.read_csv(out_csv, delimiter=r'\s+').iloc[-1])
    except unreadable as err:
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


def eval_obj(sim_dict, tru_dict, sigma=None):
    """Evaluate objective value from simulated and target observables.

    The metric compares each observable in either linear or log space,
    accumulates squared normalized differences, and applies `log_warp`.
    A small offset is applied to the denominator to avoid division by zero.

    Parameters
    ----------
    - sim_dict (dict): Simulated observable values.
    - tru_dict (dict): Target (reference) observable values.
    - sigma (dict, optional): 1-sigma uncertainty of each observable, keyed
      like `sim_dict` and in the same units as the observable. If None, the
      relative-difference objective is used.

    Returns
    ----------
    - torch.Tensor: Objective tensor of shape (1, 1).
    """

    # Looked up by name, so the order of `sigma` does not have to match `sim_dict`
    if sigma is not None:
        missing = [k for k in sim_dict.keys() if k not in sigma]
        if missing:
            raise KeyError(f'No sigma given for observables: {missing}')

    sim_vals = []
    tru_vals = []
    sig_vals = []
    for k in sim_dict.keys():
        # some variables scale logarithmically
        if variable_is_logarithmic(k):
            tru_k = max(tru_dict[k], LOG_CLIP)
            sim_vals.append(log10(max(sim_dict[k], LOG_CLIP)))
            tru_vals.append(log10(tru_k))
            if sigma is not None:
                # First-order conversion to dex, sigma / (x ln 10), only accurate while sigma << x.
                sig_vals.append(sigma[k] / (tru_k * math.log(10.0)))
        # others are just linear
        else:
            sim_vals.append(sim_dict[k])
            tru_vals.append(tru_dict[k])
            if sigma is not None:
                sig_vals.append(sigma[k])

    # Convert to tensor and reshape
    sim = torch.tensor(sim_vals, dtype=dtype).reshape(1, -1)
    true_y = torch.tensor(tru_vals, dtype=dtype).reshape(1, -1)

    # Compute normalized difference and squared distance

    if sigma is None:
        # reproduces the original objective function
        denom = torch.where(true_y >= 0, true_y + EPS_CLIP, true_y - EPS_CLIP)
        diff = torch.ones_like(true_y) - sim / denom
    else:
        # Use the provided sigma values, no epsilon offset
        sigma_tensor = torch.tensor(sig_vals, dtype=dtype).reshape(1, -1)
        diff = (sim - true_y) / sigma_tensor

    sq_dist = (diff**2).sum(dim=1, keepdim=True)

    obj = log_warp(sq_dist)

    return obj


def validate_sigma(observables: dict, sigma: dict | None) -> dict | None:
    """Check the observable uncertainties given in the inference config.

    Parameters
    ----------
    - observables (dict): Target observable values.
    - sigma (dict | None): 1-sigma uncertainty of each observable, in the same
      units as the observable, or None to use the relative-difference objective.

    Returns
    ----------
    - dict | None: The uncertainties as floats, or None.

    Raises:
        ValueError: If an observable has no uncertainty, an uncertainty names
            no observable, an uncertainty is not a positive finite number, or a
            log-scaled observable with an uncertainty is not positive.
    """
    if sigma is None:
        return None

    missing = sorted(set(observables) - set(sigma))
    if missing:
        raise ValueError(f'No sigma given for observables: {missing}')
    unknown = sorted(set(sigma) - set(observables))
    if unknown:
        raise ValueError(f'sigma given for names that are not observables: {unknown}')

    out = {}
    for k, v in sigma.items():
        v = float(v)
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"sigma for '{k}' must be a positive finite number, got {v!r}")
        if variable_is_logarithmic(k):
            x = float(observables[k])
            if x <= 0:
                raise ValueError(
                    f"Observable '{k}' is compared in log space, so its value must be "
                    f'positive when a sigma is given, got {x!r}'
                )
        out[k] = v
    return out


def J(
    x: torch.Tensor,
    parameters: list[str],
    true_observables: dict[str, float],
    worker: int,
    iter: int,
    output: str,
    ref_config: str,
    failure_codes: list[int] = [],
    sigma: dict | None = None,
) -> torch.Tensor:
    """Run PROTEUS, and then compute the objective value for a given normalized input.

    Transforms normalized `x` to raw parameters, runs the simulator,
    and computes the squared-error based objective:

        J = log_10(sum((1 - sim/true)^2) + eps)

    or, when `sigma` is given, the same with sum(((sim - true) / sigma)^2).

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
    - sigma (dict | None): Uncertainty of each observable, passed to `eval_obj`.

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
        # Parameters the simulator cannot integrate are expected in a wide box.
        return _handle_unscored(failure, output)

    # A clean exit on an error status (e.g. 25, stopped via the keepalive file),
    # one that never reached the main loop (0, 1), or with no readable status.
    failed = (20 <= sim_status <= 28) or (sim_status in (0, 1, STATUS_MISSING))

    # Runs that completed normally on an outcome this study does not fit
    # against, in `failure_codes` or in ALWAYS_EXCLUDED_STATUSES.
    excluded = (not failed) and (
        sim_status in failure_codes or sim_status in ALWAYS_EXCLUDED_STATUSES
    )

    if failed or excluded:
        _, out_abs = run_output_dir(output, worker, iter)
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
        return _handle_unscored(failure, output)

    # Compute value of objective function given these results
    return eval_obj(sim_vals, true_observables, sigma)


def _handle_unscored(failure: ProteusRunFailure, output: str) -> torch.Tensor:
    """Record an evaluation that has no fit quality, report it, and score it.

    The record is written before the abort check, so an aborted study still
    leaves the record of what stopped it. Only a failure honours
    `abort_on_failure`; an excluded outcome is reported at info level, since
    nothing went wrong in that run. The full report goes to the debug log only.

    Parameters
    ----------
    - failure (ProteusRunFailure): The failed or excluded evaluation.
    - output (str): Study output folder, relative to the PROTEUS output root.

    Returns
    ----------
    - torch.Tensor: The failure score, shape (1, 1).

    Raises:
        ProteusRunFailure: The failure itself, when `abort_on_failure` is set.
    """
    record_failure(get_proteus_directories(output)['output'], failure)
    if failure.category == CATEGORY_FAILURE:
        if os.environ.get(ABORT_ON_FAILURE_ENV, '0') == '1':
            raise failure
        log.warning(failure.summary())
    else:
        log.info(failure.summary())
    log.debug(failure.report())
    return BAD_OBJ_VALUE * torch.ones((1, 1), dtype=dtype)


def prot_builder(
    parameters: dict[str, list[float]],
    observables: dict[str, float],
    worker: int,
    iter: int,
    output: str,
    ref_config: str,
    failure_codes: list[int] = [],
    sigma: dict | None = None,
) -> callable:
    """Factory returning a BO-compatible objective function for PROTEUS inference.

    Precomputes bounds for unnormalization and embeds simulation context.

    Parameters
    ----------
    - parameters (dict): Mapping of parameter keys to [low, high] bounds.
    - observables (dict): Target observable values.
    - worker (int): Worker identifier.
    - iter (int): Iteration identifier within that worker.
    - output (str): Path to output folder relative to PROTEUS output folder.
    - ref_config (str): Reference TOML config path.
    - failure_codes (list[int]): PROTEUS status codes that complete normally but
      that this study excludes from the fit.
    - sigma (dict | None): Uncertainty of each observable, passed to `eval_obj`.

    Returns
    ----------
    - callable: Function f(x_norm) -> y_objective.
    """
    # Bounds as (2, d): lower bounds in row 0, upper in row 1
    param_keys = list(parameters.keys())
    d = len(param_keys)
    bounds = torch.tensor(list(parameters.values()), dtype=dtype).T

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

        J_eval = J(
            x_raw,
            parameters=param_keys,
            true_observables=observables,
            worker=worker,
            iter=iter,
            ref_config=ref_config,
            output=output,
            failure_codes=failure_codes,
            sigma=sigma,
        )

        # Check J is finite
        if not torch.isfinite(J_eval).all():
            x_param = {param_keys[i]: x_raw[0, i].item() for i in range(d)}

            log.warning('Non-finite objective value')
            log.warning(f'    Raw input x: {x_param}')
            log.warning(f'    Reference config: {ref_config}')

        # Compute objective
        return J_eval

    return f
