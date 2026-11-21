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
    """Return ``(jobs, waves)``: one job per ``cores_per_job`` cases, at most ``max_array``
    jobs, and past that ``waves`` cases per core one after the other. Counts are >= 1.
    """
    if min(n_cases, cores_per_job, max_array) < 1:
        raise ValueError(
            f'array_layout needs positive counts, got n_cases={n_cases}, '
            f'cores_per_job={cores_per_job}, max_array={max_array}'
        )
    jobs = min(math.ceil(n_cases / cores_per_job), max_array)
    waves = math.ceil(math.ceil(n_cases / jobs) / cores_per_job)
    return jobs, waves


# A one-wave layout of at least 10 jobs idles fewer cores than one job has, so less than 10 %
IDLE_WARN_SHARE = 0.1


def idle_share(n_cases: int, cores_per_job: int, max_array: int) -> float:
    """Return the idle share of the allocated core time, with equal run times per case:
    a job of ``c`` cases holds its cores for ``ceil(c / cores_per_job)`` run times."""
    jobs, _ = array_layout(n_cases, cores_per_job, max_array)
    q, r = divmod(n_cases, jobs)
    runs = r * math.ceil((q + 1) / cores_per_job) + (jobs - r) * math.ceil(q / cores_per_job)
    return 1 - n_cases / (runs * cores_per_job)


def warn_idle_cores(n_cases: int, cores_per_job: int, max_array: int, max_jobs: int) -> None:
    """Log a warning when the job layout leaves more than ``IDLE_WARN_SHARE`` of the
    allocated core time idle, with nearby ``cores_per_job`` values that idle less and
    don't exceed ``max_jobs``, the number of cases that run at the same time."""
    share = idle_share(n_cases, cores_per_job, max_array)
    if share <= IDLE_WARN_SHARE:
        return
    better = sorted(
        (abs(k - cores_per_job), k, s)
        for k in range(1, min(2 * cores_per_job, max_jobs) + 1)  # 1 never idles
        if (s := idle_share(n_cases, k, max_array)) <= IDLE_WARN_SHARE
    )
    log.warning(
        'With equal run times per case, the jobs leave %.1f%% of their cores and memory '
        'idle; cores_per_job values that idle less: %s',
        share * 100,
        ', '.join(f'{k} ({s:.0%})' for _, k, s in better[:3]),
    )


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
    case_list: str | None = None,
    job_name: str = 'proteus.grid.array',
) -> str:
    """Return the sbatch script that runs the cases of a grid as a job array.

    Task ``j`` runs cases ``j, j + jobs, ...``, ``cores_per_job`` at a time, each as a
    one-core job step; "With Slurm" in docs/How-to/usage_grids.md describes the statuses,
    the early SIGTERM and the last case start. The early SIGTERM goes to the batch shell,
    which logs its time, starts no new case and sends it on to the running cases. As a
    fallback, no case starts within ``min(60 s, lead)`` of the time asked for the signal.

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
    case_list : str or None
        File with one case name per line, to run only those ``n_cases`` cases; task ``j``
        then runs the listed cases ``j, j + jobs, ...``. None runs every case of the grid.
    job_name : str
        Slurm name of the job array, as used by ``squeue -n`` and ``scancel -n``.
    """
    names = f"printf 'case_%06d\\n' $(seq $SLURM_ARRAY_TASK_ID {jobs} {n_cases - 1})"
    if case_list is not None:
        names = (
            f"awk -v t=$SLURM_ARRAY_TASK_ID '(NR - 1) % {jobs} == t' {shlex.quote(case_list)}"
        )
    minutes = max(1, round(days * 24 * 60))
    signal_s = min(900, minutes * 6)  # a tenth of the wall time, at most 15 minutes
    last_start_s = minutes * 60 - signal_s - min(60, signal_s)
    return f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --export=ALL
#SBATCH --no-requeue
#SBATCH --time={minutes}
#SBATCH --signal=B:TERM@{signal_s}
#SBATCH --cpus-per-task={cores_per_job}
#SBATCH --mem-per-cpu={mem_per_case_mb}M
#SBATCH -i /dev/null
#SBATCH -o {logdir}/proteus-%A_%a.log
#SBATCH --array=0-{jobs - 1}%{throttle}

{env}export cfgs={shlex.quote(cfgdir)} logs={shlex.quote(logdir)} cases={shlex.quote(casedir or '')}
export stop_at=$(( $(date +%s) + {last_start_s} )) stop="${{TMPDIR:-/tmp}}/${{SLURM_JOB_ID}}_stop"

# The early signal or a scancel: start no new case, send SIGTERM on to the running ones.
# After a scancel or at the walltime, Slurm signals the cases itself and this scancel fails.
on_term() {{
    echo "SIGTERM at $(date -u +%Y-%m-%dT%H:%M:%SZ), $SECONDS s into the job: no new case starts, and the running cases get SIGTERM"
    touch "$stop"
    scancel --signal=TERM "$SLURM_JOB_ID"
}}
trap on_term TERM

# One case: $1 is its name, case_NNNNNN
run_case='
if [ -e "$stop" ]; then
    echo "$1 not started: the job got SIGTERM"
    exit 1
fi
if [ "$(date +%s)" -ge "$stop_at" ]; then
    echo "$1 not started: the job is within {minutes * 60 - last_start_s} s of its walltime"
    exit 1
fi
echo "running $1"
TMPDIR="${{TMPDIR:-/tmp}}/${{SLURM_JOB_ID}}_$1"
export TMPDIR
mkdir -p "$TMPDIR"
srun --job-name="$1" --ntasks=1 --cpus-per-task=1 --mem-per-cpu={mem_per_case_mb}M --exact \\
    {command} "$cfgs/$1.toml" > "$logs/$1.log" 2>&1
code=$?
rm -rf "$TMPDIR"
if [ -n "$cases" ]; then
    status=$(head -n 1 "$cases/$1/status" 2>/dev/null)
    case "$status" in
        2[0-8]|30) [ "$code" -ne 0 ] || {{ echo "$1 ended with error status $status"; exit 1; }} ;;
        ""|0|1)
            echo "$1 ended without a final status"
            mkdir -p "$cases/$1"
            if grep -q "^srun: error: .*: task [0-9]*: Out Of Memory" "$logs/$1.log"; then
                printf "30\\nError (killed for memory)\\n" > "$cases/$1/status"
            else
                printf "25\\nError (died, or exit requested by user)\\n" > "$cases/$1/status"
            fi
            exit 1 ;;
    esac
fi
[ "$code" -eq 0 ] || {{ echo "$1 failed with exit code $code"; exit 1; }}
'

{names} |
    xargs -n 1 -P {cores_per_job} sh -c "$run_case" sh &
pid=$!
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
wait "$pid"
code=$?
rm -f "$stop"
exit "$code"
"""
