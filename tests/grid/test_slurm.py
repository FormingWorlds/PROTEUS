"""Unit tests for ``proteus.grid.slurm``, the job array that packs grid cases into jobs.

Checks: every case lands in exactly one array task and runs once; no job runs more
cases than its cores times its waves; the array never exceeds the cluster limit; a
failed case does not stop the other cases of its job; wall times convert to the
nearest minute.
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

import pytest

from proteus.grid import slurm

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize(
    ('n_cases', 'cores', 'max_array', 'expected'),
    [
        (10000, 16, 1000, (625, 1)),  # Interra ensemble on 1/8 rome nodes
        (10000, 128, 30001, (79, 1)),  # whole rome nodes; 79 * 128 = 10112 slots
        (2500, 1, 1001, (1001, 3)),  # FormingWorlds/PROTEUS#517: Habrok limit, one core
        (20000, 16, 1000, (1000, 2)),
        (17, 16, 1000, (2, 1)),  # one case over a full job needs a second job
        (1, 16, 1000, (1, 1)),
    ],
    ids=['rome-eighth', 'rome-node', 'habrok-limit', 'two-waves', 'one-over', 'single'],
)
def test_array_layout_covers_every_case_within_the_limit(n_cases, cores, max_array, expected):
    """The layout gives enough core slots for every case, stays within the array limit
    and adds a wave only when the limit is reached."""
    jobs, waves = slurm.array_layout(n_cases, cores, max_array)
    assert (jobs, waves) == expected
    assert jobs * cores * waves >= n_cases
    assert jobs <= max_array
    # The busiest job (strided cases) fits its cores times its waves
    assert -(-n_cases // jobs) <= cores * waves


@pytest.mark.parametrize(
    'args',
    [(0, 16, 1000), (10, 0, 1000), (10, 16, 0)],
    ids=['no-cases', 'no-cores', 'no-array'],
)
def test_array_layout_refuses_a_count_below_one(args):
    """A zero count has no layout; the error names the counts."""
    with pytest.raises(ValueError, match='array_layout needs positive counts') as exc:
        slurm.array_layout(*args)
    assert f'n_cases={args[0]}, cores_per_job={args[1]}, max_array={args[2]}' in str(exc.value)


def test_max_array_size_reads_scontrol():
    """The value of MaxArraySize is read from among the other scontrol keys."""
    out = 'MaxArraySize            = 30001\nMaxJobCount             = 100000\n'
    done = subprocess.CompletedProcess(['scontrol'], 0, stdout=out, stderr='')
    with mock.patch.object(slurm.subprocess, 'run', return_value=done) as run:
        assert slurm.max_array_size() == 30001
    assert run.call_args.args[0] == ['scontrol', 'show', 'config']


@pytest.mark.parametrize(
    'outcome',
    [FileNotFoundError('scontrol'), subprocess.CalledProcessError(1, 'scontrol'), 'MaxJob = 9'],
    ids=['no-scontrol', 'scontrol-fails', 'no-key'],
)
def test_max_array_size_falls_back_to_the_slurm_default(outcome):
    """Without a value from scontrol the limit is Slurm's own default of 1001."""
    if isinstance(outcome, str):
        kwargs = {'return_value': subprocess.CompletedProcess([], 0, stdout=outcome)}
    else:
        kwargs = {'side_effect': outcome}
    with mock.patch.object(slurm.subprocess, 'run', **kwargs):
        assert slurm.max_array_size() == slurm.DEFAULT_MAX_ARRAY_SIZE
    assert slurm.DEFAULT_MAX_ARRAY_SIZE == 1001


def _script(tmp_path, n_cases, cores, jobs, command, days=1.0):
    return slurm.dispatch_script(
        cfgdir=str(tmp_path / 'my grid' / 'cfgs'),
        logdir=str(tmp_path / 'my grid' / 'logs'),
        n_cases=n_cases,
        cores_per_job=cores,
        jobs=jobs,
        throttle=jobs,
        days=days,
        mem_per_case_mb=3072,
        command=command,
        casedir=str(tmp_path / 'my grid'),
    )


def test_dispatch_script_asks_for_the_cores_and_one_step_per_case(tmp_path):
    """The job asks for its cores and the memory per core; each case is a one-core job
    step with that memory, and the job is not requeued."""
    text = _script(tmp_path, n_cases=100, cores=16, jobs=7, command='proteus start --config')
    assert '#SBATCH --cpus-per-task=16\n' in text
    assert '#SBATCH --mem-per-cpu=3072M\n' in text
    assert '#SBATCH --array=0-6%7\n' in text
    assert '#SBATCH --no-requeue\n' in text
    assert 'xargs -n 1 -P 16 ' in text
    assert (
        'srun --job-name="$1" --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3072M --exact' in text
    )
    assert 'seq $SLURM_ARRAY_TASK_ID 7 99' in text


@pytest.mark.parametrize(
    ('days', 'minutes', 'lead', 'offset'),
    [
        (1.0, 1440, 300, 86040),
        (2.3, 3312, 300, 198360),  # 2.3 * 1440 is 3311.9999; truncation writes 3311
        (1 / 24, 60, 300, 3240),
        (0.02, 29, 174, 1506),  # 28.8 min rounds to 29; a tenth of 1740 s is 174 s
        (0.001, 1, 6, 48),  # the 60 s margin shrinks to the 6 s lead
        (0.0001, 1, 6, 48),  # 0.144 min rounds to 0, which Slurm reads as no limit
    ],
    ids=['day', 'float-below-minute', 'hour', 'half-hour', 'one-minute', 'below-a-minute'],
)
def test_walltime_signal_and_last_start_agree(tmp_path, days, minutes, lead, offset):
    """The wall time is written in whole minutes, at least one. The early SIGTERM comes
    300 s before it, or a tenth of it for short jobs. Slurm may send it up to 60 s
    early, so the script starts no case from 60 s before the signal on, or from the
    lead before it when the lead is shorter: the offset from the job start is the wall
    time minus the lead minus that margin."""
    text = _script(tmp_path, n_cases=4, cores=2, jobs=2, command='x', days=days)
    assert f'#SBATCH --time={minutes}\n' in text
    assert f'#SBATCH --signal=TERM@{lead}\n' in text
    assert f'export stop_at=$(( $(date +%s) + {offset} ))\n' in text
    assert '#SBATCH --time=0\n' not in text


def test_dispatch_script_runs_each_case_once_and_gives_killed_cases_a_status(tmp_path):
    """Run the script for every array task, with srun replaced by a pass-through.
    Fifteen cases in five jobs of two cores (two waves) run once each, in a folder with
    a space, each with its own temporary folder that is gone afterwards.

    Task 0 has a case that exits 255 and task 1 a case killed by SIGKILL; either stops
    a bare xargs, yet the case queued behind it runs. Both get status 25. Task 2's
    only failure is a case that returns 0 while its status still says running: it gets
    status 25 and fails its job. In task 3 one case writes the error status 22 and
    exits 1, as PROTEUS does on a model error, and one writes 22 and exits 255, which
    stops a bare xargs: both keep 22 and the case queued behind them runs. Task 4 has no
    failure and exits 0."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    # srun stand-in: drop the options, run the command
    (bindir / 'srun').write_text(
        '#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift; done\nexec "$@"\n'
    )
    record = tmp_path / 'ran.txt'
    # Normal cases sleep, so a failing case beside them ends first and the next case
    # of its job starts only after the failure
    (bindir / 'fakecase').write_text(
        '#!/bin/sh\n'
        'n=$(basename "$1" .toml); out="$(dirname "$(dirname "$1")")/$n"\n'
        f'[ -d "$TMPDIR" ] && echo "$n $TMPDIR" >> "{record}"\n'
        'mkdir -p "$out"\n'
        'case "$n" in\n'
        '  case_000000) rmdir "$out"; exit 255 ;;\n'
        '  case_000001) echo 1 > "$out/status"; kill -9 $$ ;;\n'
        '  case_000002) echo 1 > "$out/status"; exit 0 ;;\n'
        '  case_000003) printf "22\\nError\\n" > "$out/status"; sleep 0.3; exit 1 ;;\n'
        '  case_000008) printf "22\\nError\\n" > "$out/status"; exit 255 ;;\n'
        'esac\n'
        'sleep 0.3\n'
        'printf "10\\nCompleted\\n" > "$out/status"\n'
        'echo "output of $1"\n'
    )
    for tool in ('srun', 'fakecase'):
        os.chmod(bindir / tool, 0o755)
    os.makedirs(tmp_path / 'my grid' / 'logs')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    n_cases, cores = 15, 2
    jobs, waves = slurm.array_layout(n_cases, cores, max_array=5)
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases, cores, jobs, command='fakecase'))
    env = dict(
        os.environ, PATH=f'{bindir}{os.pathsep}{os.environ["PATH"]}', TMPDIR=str(scratch)
    )

    # The array tasks run at the same time, as on a cluster
    tasks = [
        subprocess.Popen(
            ['bash', str(script)],
            env={**env, 'SLURM_ARRAY_TASK_ID': str(task)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for task in range(jobs)
    ]
    codes = {task: proc.wait(timeout=20) for task, proc in enumerate(tasks)}

    # strided: task t runs cases t, t + 5 and t + 10, the last after a core frees up
    assert (jobs, waves) == (5, 2)
    ran = sorted(line.split(' ', 1) for line in record.read_text().splitlines())
    assert ran == [[f'case_{i:06d}', f'{scratch}/case_{i:06d}'] for i in range(n_cases)]
    assert list(scratch.iterdir()) == []
    assert [code != 0 for code in codes.values()] == [True, True, True, True, False]
    log10 = (tmp_path / 'my grid' / 'logs' / 'case_000010.log').read_text()
    assert 'output of' in log10 and 'cfgs/case_000010.toml' in log10
    first = {
        i: (tmp_path / 'my grid' / f'case_{i:06d}' / 'status').read_text().split('\n')[0]
        for i in range(n_cases)
    }
    assert first == {
        i: {0: '25', 1: '25', 2: '25', 3: '22', 8: '22'}.get(i, '10') for i in range(15)
    }


def test_dispatch_script_starts_no_case_after_the_early_signal(tmp_path):
    """Once the clock passes the time of the early SIGTERM, the job starts none of its
    remaining cases: they keep no status folder, as not started, and the job exits
    non-zero."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    marker = tmp_path / 'started'
    (bindir / 'srun').write_text(f'#!/bin/sh\ntouch "{marker}"\n')
    # date stand-in: the job starts at t = 1000 s, every later reading is two days on
    calls = tmp_path / 'date_called'
    (bindir / 'date').write_text(
        f'#!/bin/sh\nif [ -e "{calls}" ]; then echo 173800; else touch "{calls}"; echo 1000; fi\n'
    )
    for tool in ('srun', 'date'):
        os.chmod(bindir / tool, 0o755)
    os.makedirs(tmp_path / 'my grid' / 'logs')
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases=3, cores=2, jobs=1, command='x', days=1.0))
    env = dict(
        os.environ,
        PATH=f'{bindir}{os.pathsep}{os.environ["PATH"]}',
        TMPDIR=str(tmp_path),
        SLURM_ARRAY_TASK_ID='0',
    )
    done = subprocess.run(['bash', str(script)], env=env, capture_output=True, timeout=20)
    assert done.returncode != 0
    assert not marker.exists()
    assert not any((tmp_path / 'my grid').glob('case_*'))
    assert done.stdout.decode().count('not started') == 3
