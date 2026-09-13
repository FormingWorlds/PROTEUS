from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import pandas as pd
import toml
import torch
from numpy import log10

from proteus.inference.transforms import unnormalize_parameters
from proteus.utils.constants import element_list, gas_list
from proteus.utils.coupler import get_proteus_directories, variable_is_logarithmic
from proteus.utils.helper import CommentFromStatus

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

# Whether a failed child run aborts the study or scores a bad objective value.
# Plumbed through the environment for the same reason as the timeout above.
_ABORT_ON_FAILURE_ENV = 'PROTEUS_INFERENCE_ABORT_ON_FAILURE'

# Lines of child stderr retained in a failure report. PROTEUS writes its own
# diagnostics to the run's logfile, but a run that dies before the logger is
# set up (a rejected config, a missing environment variable) leaves nothing
# behind except this stream, so it is kept rather than discarded.
STDERR_TAIL_LINES = 40

# Suffix for the file holding whatever a child wrote to its console. It is
# kept beside the run folder rather than inside it: PROTEUS empties its own
# output folder once it starts, which would unlink a file held open there and
# lose exactly the record this capture exists to keep.
CHILD_CONSOLE_SUFFIX = '_console.log'

# Status written by PROTEUS before its output folder is cleaned, and never
# rewritten until the main loop starts. A child that dies in between leaves no
# status file at all, so a missing file is reported as such rather than being
# silently reported as a generic error.
STATUS_MISSING = -1

# Folder inside the study output holding one record per failed evaluation.
# Written by the workers as they fail and read back once at the end, so that
# the summary covers initial sampling and optimisation alike without the two
# paths having to share any state while they run.
FAILURE_RECORD_DIR = 'failures'

# Fraction of evaluations that may fail before the summary escalates from a
# report to a warning. Above this, the sampled region is mostly unrunnable and
# the posterior is built on too few real evaluations to mean much.
FAILURE_FRACTION_WARN = 0.5

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


def _tail(text: str | bytes | None, lines: int = STDERR_TAIL_LINES) -> str:
    """Return the last `lines` lines of captured child output."""
    if not text:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    return '\n'.join(text.splitlines()[-lines:])


def _tail_file(path: Path, lines: int = STDERR_TAIL_LINES) -> str:
    """Return the last `lines` lines of a child's console file.

    Returns an empty string when the file is missing or unreadable, so a
    failure report is still produced when the capture itself went wrong.
    """
    try:
        with open(path, 'r', errors='replace') as f:
            return _tail(f.read(), lines)
    except OSError:
        return ''


@dataclass(eq=False)
class ProteusRunFailure(RuntimeError):
    """A single child PROTEUS run that did not produce a usable result.

    Carries everything needed to diagnose the run without opening the study
    by hand: which evaluation it was, where its output landed, how it died,
    what PROTEUS recorded in its status file, and the parameter values that
    produced it. Raised for faults that are specific to one evaluation; faults
    that would affect every evaluation (no `proteus` on PATH, an observable
    that no helpfile column provides) stay as ordinary exceptions so they
    abort the study instead of being scored as a bad sample.
    """

    reason: str
    worker: int
    iter: int
    out_dir: str
    exit_code: int | None = None
    status: int = STATUS_MISSING
    log_path: str | None = None
    stderr_tail: str = ''
    parameters: dict = field(default_factory=dict)

    @property
    def status_desc(self) -> str:
        """Human-readable form of the PROTEUS status code."""
        if self.status == STATUS_MISSING:
            return 'no readable status file (died during start-up)'
        return CommentFromStatus(self.status)

    def report(self) -> str:
        """Multi-line description naming the cause and where to look next."""
        lines = [
            f'PROTEUS run failed for worker={self.worker} iter={self.iter}: {self.reason}',
            f'    status    = {self.status} ({self.status_desc})',
        ]
        if self.exit_code is not None:
            lines.append(f'    exit code = {self.exit_code}')
        lines.append(f'    output    = {self.out_dir}')
        if self.log_path:
            lines.append(f'    logfile   = {self.log_path}')
        if self.parameters:
            pretty = ', '.join(f'{k}={v:g}' for k, v in sorted(self.parameters.items()))
            lines.append(f'    parameters = {pretty}')
        if self.stderr_tail:
            lines.append('    last output from the child process:')
            lines.extend(f'      {line}' for line in self.stderr_tail.splitlines())
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.report()

    def __reduce__(self):
        # A failure raised inside a pool worker is pickled to be re-raised in
        # the parent. BaseException.__reduce__ rebuilds from `self.args`,
        # which a dataclass __init__ leaves empty, so the default would fail
        # to reconstruct this class. Rebuild from the fields instead.
        return (
            self.__class__,
            (
                self.reason,
                self.worker,
                self.iter,
                self.out_dir,
                self.exit_code,
                self.status,
                self.log_path,
                self.stderr_tail,
                self.parameters,
            ),
        )


def set_abort_on_failure(abort: bool = False) -> None:
    """Record whether a failed child run should abort the whole study.

    Stored in the environment so it is visible to the main process and to any
    spawned pool workers, matching how the child timeout is plumbed.
    """
    os.environ[_ABORT_ON_FAILURE_ENV] = '1' if abort else '0'


def abort_on_failure() -> bool:
    """Return whether a failed child run should abort the whole study.

    Defaults to False: an inference sweep is expected to visit parameter
    combinations the simulator cannot integrate, and treating those as fatal
    would make most studies unrunnable. Set the inference config field
    `abort_on_failure` to stop at the first failure instead.
    """
    return os.environ.get(_ABORT_ON_FAILURE_ENV, '0') == '1'


def read_status(out_abs: Path | str) -> int:
    """Read the PROTEUS status code from a finished run's output folder.

    Parameters
    ----------
    - out_abs (Path | str): Absolute path to the run's output folder.

    Returns
    ----------
    - int: The status code, or `STATUS_MISSING` when no readable status file
      exists. A missing file is itself diagnostic: PROTEUS deletes the status
      it writes at start-up when it cleans the output folder, and does not
      write another until the main loop begins.
    """
    try:
        with open(Path(out_abs) / 'status', 'r') as f:
            return int(f.readlines()[0].strip())
    except Exception:
        return STATUS_MISSING


def find_run_logfile(out_abs: Path | str) -> str | None:
    """Return the newest PROTEUS logfile in a run's output folder, if any.

    PROTEUS captures uncaught exceptions into this file, so it usually holds
    the traceback for a crashed run. It does not exist for a run that failed
    before the logger was configured.
    """
    logs = sorted(Path(out_abs).glob('proteus_*.log'))
    return str(logs[-1]) if logs else None


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


def record_failure(study_abs: Path | str, failure: ProteusRunFailure) -> str | None:
    """Write a failure record into the study's `failures` folder.

    One file per failed evaluation, named for the worker and iteration that
    produced it, so that concurrent workers never write to the same file and
    no lock is needed. The end-of-study summary reads them back.

    Parameters
    ----------
    - study_abs (Path | str): Absolute path to the study output folder.
    - failure (ProteusRunFailure): The failure to record.

    Returns
    ----------
    - str | None: Path written, or None if the record could not be written.
      Recording is best-effort: a study must not be brought down by a fault in
      its own bookkeeping, so the failure being reported still reaches the log.
    """
    record = asdict(failure)
    record['status_desc'] = failure.status_desc
    target = Path(study_abs) / FAILURE_RECORD_DIR / f'w{failure.worker}_i{failure.iter}.json'
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w') as f:
            json.dump(record, f, indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as err:
        log.warning(
            f'Could not record the failure of worker={failure.worker} '
            f'iter={failure.iter}: {err}'
        )
        return None
    return str(target)


def read_failure_records(study_abs: Path | str) -> list[dict]:
    """Read back every failure record written during a study.

    Parameters
    ----------
    - study_abs (Path | str): Absolute path to the study output folder.

    Returns
    ----------
    - list[dict]: One entry per failed evaluation, ordered by worker then
      iteration. Unreadable records are skipped with a warning rather than
      aborting the summary, which would hide the failures that did parse.
    """
    records = []
    for path in sorted((Path(study_abs) / FAILURE_RECORD_DIR).glob('w*_i*.json')):
        try:
            with open(path, 'r') as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError) as err:
            log.warning(f'Skipping unreadable failure record {path}: {err}')
    return sorted(records, key=lambda r: (r.get('worker', 0), r.get('iter', 0)))


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

    def _failure(
        reason: str, exit_code: int | None, stderr_tail: str = ''
    ) -> ProteusRunFailure:
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
            status=read_status(out_abs),
            log_path=find_run_logfile(out_abs),
            stderr_tail=stderr_tail,
            parameters=swept,
        )

    # Run PROTEUS. Output is kept rather than discarded: a run that dies
    # before its logger is configured leaves no logfile behind, so this stream
    # is the only record of why it refused to start. It goes to a file rather
    # than a pipe because a long run of a chatty module would otherwise buffer
    # hours of output in memory, in every worker at once, to retain a few
    # lines of it.
    command = ['proteus', 'start', '-c', str(out_cfg), '--offline']
    console = out_abs.parent / f'{out_abs.name}{CHILD_CONSOLE_SUFFIX}'
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
        raise _failure(
            f'exceeded the {timeout} s timeout',
            exit_code=None,
            stderr_tail=_tail_file(console),
        ) from err
    except subprocess.CalledProcessError as err:
        raise _failure(
            'the simulator exited with an error',
            exit_code=err.returncode,
            stderr_tail=_tail_file(console),
        ) from err
    finally:
        stream.close()

    # Re-write config in case simulator mutates or removes it
    update_toml(ref_config, parameters, str(out_cfg))

    # Read status file
    status = read_status(out_abs)

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
    if 'P_surf' not in df_row:
        # The helpfile schema is the same for every run, so a missing column
        # is a fault of the setup rather than of this sample. Scoring it as a
        # bad sample would let the study spend its whole budget returning the
        # same failure value and then report success.
        raise KeyError(
            f"Simulator output has no 'P_surf' column ({out_csv}); "
            'every run will produce the same result'
        )
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
    - failure_codes (list[int]): Additional PROTEUS exit codes to treat as failures.

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
        # sample and the study continues. Every such run is reported in full,
        # once, because the alternative is a silently under-sampled study.
        # Recorded before the abort check, so an aborted study still leaves
        # the record of what stopped it.
        record_failure(get_proteus_directories(output)['output'], failure)
        if abort_on_failure():
            raise
        log.warning(failure.report())
        return BAD_OBJ_VALUE * torch.ones((1, 1), dtype=dtype)

    # If status indicates failure, return very bad objective value.
    # Reached by runs that exit cleanly but stop in an error state, such as a
    # run halted through its keepalive file (status 25). An unreadable status
    # is counted as a failure too: the run's own account of itself is missing,
    # so its output cannot be trusted.
    if (
        (20 <= sim_status <= 28)
        or (sim_status in (0, 1, STATUS_MISSING))
        or (sim_status in failure_codes)
    ):
        desc = (
            'no status file written'
            if sim_status == STATUS_MISSING
            else CommentFromStatus(sim_status)
        )
        # Recorded alongside the runs that crashed, so the end-of-study summary
        # counts both kinds of failure rather than only the noisy kind.
        _, out_abs = run_output_dir(output, worker, iter)
        record_failure(
            get_proteus_directories(output)['output'],
            ProteusRunFailure(
                reason='exited cleanly but stopped in a failure state',
                worker=worker,
                iter=iter,
                out_dir=str(out_abs),
                exit_code=0,
                status=sim_status,
                log_path=find_run_logfile(out_abs),
                parameters=raw,
            ),
        )
        log.warning(
            f'PROTEUS run for worker={worker} iter={iter} finished in a failure state: '
            f'status {sim_status} ({desc})'
        )
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
    - failure_codes (list[int]): Additional PROTEUS exit codes to treat as failures.

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
