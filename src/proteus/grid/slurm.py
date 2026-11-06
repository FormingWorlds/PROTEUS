"""Slurm job array that runs the cases of a grid, several cases per job."""

from __future__ import annotations

import logging
import math
import shlex
import subprocess

log = logging.getLogger('fwl.' + __name__)

# Slurm's own default when slurm.conf sets no MaxArraySize
DEFAULT_MAX_ARRAY_SIZE = 1001


def max_array_size() -> int:
    """Return the MaxArraySize of this Slurm cluster, from ``scontrol show config``.

    Array task indices run from 0 to MaxArraySize - 1, so this is the largest number of
    tasks one array can hold. Without ``scontrol``, or when it reports no value, return
    the Slurm default ``DEFAULT_MAX_ARRAY_SIZE``.
    """
    try:
        out = subprocess.run(
            ['scontrol', 'show', 'config'],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning(
            'Cannot read MaxArraySize from scontrol (%s); assuming %d',
            exc,
            DEFAULT_MAX_ARRAY_SIZE,
        )
        return DEFAULT_MAX_ARRAY_SIZE
    for line in out.splitlines():
        key, _, value = line.partition('=')
        if key.strip() == 'MaxArraySize':
            return int(value)
    log.warning('scontrol reports no MaxArraySize; assuming %d', DEFAULT_MAX_ARRAY_SIZE)
    return DEFAULT_MAX_ARRAY_SIZE


def array_layout(n_cases: int, cores_per_job: int, max_array: int) -> tuple[int, int]:
    """Return the number of array jobs and the number of cases each core runs in turn.

    Each job runs ``cores_per_job`` cases at a time. The array has one job per
    ``cores_per_job`` cases, at most ``max_array`` jobs; past that, each core of a job
    runs several cases one after the other (the waves).

    Parameters
    ----------
    n_cases : int
        Number of grid cases, at least 1.
    cores_per_job : int
        Cases run at the same time in one job, at least 1.
    max_array : int
        Largest number of tasks in one job array, at least 1.

    Returns
    -------
    tuple[int, int]
        ``(jobs, waves)``.
    """
    if min(n_cases, cores_per_job, max_array) < 1:
        raise ValueError(
            f'array_layout needs positive counts, got n_cases={n_cases}, '
            f'cores_per_job={cores_per_job}, max_array={max_array}'
        )
    jobs = min(math.ceil(n_cases / cores_per_job), max_array)
    waves = math.ceil(math.ceil(n_cases / jobs) / cores_per_job)
    return jobs, waves


def dispatch_script(
    *,
    cfgdir: str,
    logdir: str,
    n_cases: int,
    cores_per_job: int,
    jobs: int,
    throttle: int,
    days: float,
    mem_per_case_mb: int,
    command: str,
    casedir: str | None = None,
    env: str = '',
) -> str:
    """Return the sbatch script that runs the cases of a grid as a job array.

    Array task ``j`` runs the cases ``j, j + jobs, j + 2 jobs, ...``, ``cores_per_job``
    at a time with ``xargs -P``; a core takes the next case when its case ends. Each
    case is a job step of one core and its own memory, named after the case, so
    ``sacct`` reports its peak memory; it writes its output to
    ``<logdir>/case_NNNNNN.log`` and its temporary files to its own folder under
    ``$TMPDIR``, removed when the case ends. A case that fails, also by a signal, does
    not stop the other cases; the job then exits non-zero.

    A case that ends with its status file still at 0 or 1 (running), or without one,
    was killed before PROTEUS could write a final status (SIGKILL, out of memory, a
    crash in compiled code); the script then writes status 25 and counts the case as
    failed. Slurm sends SIGTERM to
    the cases 5 minutes before the wall time, a tenth of it for short jobs, so they can
    end with a status; from then on the job starts no further case, so those stay
    without a status folder, as not started. The job is not requeued, since a
    requeued job would start its finished cases again.

    Parameters
    ----------
    cfgdir, logdir : str
        Folders of the case configs (``case_NNNNNN.toml``) and of the logs.
    n_cases : int
        Number of grid cases.
    cores_per_job : int
        CPU cores per job, one case per core at a time.
    jobs : int
        Tasks in the job array.
    throttle : int
        Array tasks running at the same time.
    days : float
        Wall time per job [days], written in whole minutes, at least 1, since a zero
        Slurm time limit means no limit.
    mem_per_case_mb : int
        Memory per core, so per case [MB].
    command : str
        Command that runs one case; the config path follows it.
    casedir : str or None
        Folder that holds the case output folders; None writes no status.
    env : str
        Shell lines run before the cases, such as exports.
    """
    minutes = max(1, round(days * 24 * 60))
    signal_s = min(300, minutes * 6)  # a tenth of the wall time, at most 5 minutes
    return f"""#!/bin/bash
#SBATCH -J proteus.grid.array
#SBATCH --export=ALL
#SBATCH --no-requeue
#SBATCH --time={minutes}
#SBATCH --signal=TERM@{signal_s}
#SBATCH --cpus-per-task={cores_per_job}
#SBATCH --mem-per-cpu={mem_per_case_mb}M
#SBATCH -i /dev/null
#SBATCH -o {logdir}/proteus-%A_%a.log
#SBATCH --array=0-{jobs - 1}%{throttle}

{env}export cfgs={shlex.quote(cfgdir)} logs={shlex.quote(logdir)} cases={shlex.quote(casedir or '')}
export stop_at=$(( $(date +%s) + {minutes * 60 - signal_s} ))

# One case: $1 is its name, case_NNNNNN
run_case='
if [ "$(date +%s)" -ge "$stop_at" ]; then
    echo "$1 not started: the job is within {signal_s} s of its walltime"
    exit 1
fi
echo "running $1"
TMPDIR="${{TMPDIR:-/tmp}}/$1"
export TMPDIR
mkdir -p "$TMPDIR"
srun --job-name="$1" --ntasks=1 --cpus-per-task=1 --mem-per-cpu={mem_per_case_mb}M --exact \\
    {command} "$cfgs/$1.toml" > "$logs/$1.log" 2>&1
code=$?
rm -rf "$TMPDIR"
if [ -n "$cases" ]; then
    case "$(head -n 1 "$cases/$1/status" 2>/dev/null)" in
        ""|0|1)
            echo "$1 ended without a final status"
            mkdir -p "$cases/$1"
            printf "25\\nError (died, or exit requested by user)\\n" > "$cases/$1/status"
            exit 1 ;;
    esac
fi
[ "$code" -eq 0 ] || {{ echo "$1 failed with exit code $code"; exit 1; }}
'

printf 'case_%06d\\n' $(seq $SLURM_ARRAY_TASK_ID {jobs} {n_cases - 1}) |
    xargs -n 1 -P {cores_per_job} sh -c "$run_case" sh
"""
