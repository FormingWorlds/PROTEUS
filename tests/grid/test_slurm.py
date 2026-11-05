"""Unit tests for ``proteus.grid.slurm``, the job array that packs grid cases into jobs.

Checks: every case lands in exactly one array task and runs once; no job runs more
cases than its cores times its waves; the array never exceeds the cluster limit; a
failed case does not stop the other cases of its job; wall times convert to the
nearest minute.
"""

from __future__ import annotations

import os
import shutil
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


@pytest.mark.parametrize(
    ('days', 'expected'),
    [(2.3, '2-07:12:00'), (1.5, '1-12:00:00'), (5, '5-00:00:00'), (0.001, '0-00:01:00')],
    ids=['float-below-minute', 'half-day', 'snellius-limit', 'shortest'],
)
def test_slurm_time_rounds_to_the_nearest_minute(days, expected):
    """2.3 days is 3312 minutes; 2.3 * 1440 is 3311.9999 in floating point, which a
    rounding down would write as 2-07:11:00."""
    assert slurm.slurm_time(days) == expected
    assert slurm.slurm_time(days) != '2-07:11:00'


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
    step with that memory, the job is not requeued and gets SIGTERM 5 minutes early."""
    text = _script(tmp_path, n_cases=100, cores=16, jobs=7, command='proteus start --config')
    assert '#SBATCH --cpus-per-task=16\n' in text
    assert '#SBATCH --mem-per-cpu=3072M\n' in text
    assert '#SBATCH --array=0-6%7\n' in text
    assert '#SBATCH --no-requeue\n' in text
    assert '#SBATCH --time=1-00:00:00\n' in text
    assert '#SBATCH --signal=TERM@300\n' in text
    assert 'xargs -n 1 -P 16 ' in text
    assert (
        'srun --job-name="$1" --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3072M --exact' in text
    )
    assert 'seq $SLURM_ARRAY_TASK_ID 7 99' in text


@pytest.mark.parametrize(
    ('days', 'lead'),
    [(1.0, 300), (1 / 24, 300), (0.02, 172), (0.001, 8)],
    ids=['day', 'hour', 'half-hour', 'shortest'],
)
def test_signal_comes_five_minutes_early_or_at_nine_tenths_of_a_short_job(tmp_path, days, lead):
    """The early SIGTERM is 300 s before the wall time, but at most a tenth of it: a
    job of 0.02 days (1728 s) gets it 172 s early, not at its start."""
    text = _script(tmp_path, n_cases=4, cores=2, jobs=2, command='x', days=days)
    assert f'#SBATCH --signal=TERM@{lead}\n' in text
    assert lead <= 0.1 * days * 86400 + 1


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ('bash', 'seq', 'xargs')),
    reason='needs bash, seq and xargs',
)
def test_dispatch_script_runs_each_case_once_and_survives_failed_cases(tmp_path):
    """Run the script for every array task, with srun replaced by a pass-through. Nine
    cases in three jobs of two cores (two waves) run once each, in a folder with a
    space, each with its own temporary folder that is gone afterwards. The case that
    exits 255 and the one killed by a signal do not stop the case queued behind them
    in their job, and only those jobs exit non-zero. A case left at status 1 or
    without a status file gets status 25; a final status stays."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    # srun stand-in: drop the options, run the command
    (bindir / 'srun').write_text(
        '#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift; done\nexec "$@"\n'
    )
    record = tmp_path / 'ran.txt'
    # case 0 exits 255 before writing a status and case 4 is killed by a signal while
    # running, either of which stops a bare xargs; case 3 returns while still running
    (bindir / 'fakecase').write_text(
        '#!/bin/sh\n'
        'n=$(basename "$1" .toml); out="$(dirname "$(dirname "$1")")/$n"\n'
        f'[ -d "$TMPDIR" ] && echo "$n $TMPDIR" >> "{record}"\n'
        'case "$n" in\n'
        '  case_000000) exit 255 ;;\n'
        '  case_000003) mkdir -p "$out"; echo 1 > "$out/status"; exit 0 ;;\n'
        '  case_000004) mkdir -p "$out"; echo 1 > "$out/status"; kill -9 $$ ;;\n'
        'esac\n'
        'mkdir -p "$out"; printf "10\\nCompleted\\n" > "$out/status"\n'
        'echo "output of $1"\n'
    )
    for tool in ('srun', 'fakecase'):
        os.chmod(bindir / tool, 0o755)
    os.makedirs(tmp_path / 'my grid' / 'logs')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    n_cases, cores = 9, 2
    jobs, waves = slurm.array_layout(n_cases, cores, max_array=3)
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases, cores, jobs, command='fakecase'))
    env = dict(
        os.environ, PATH=f'{bindir}{os.pathsep}{os.environ["PATH"]}', TMPDIR=str(scratch)
    )

    codes = {}
    for task in range(jobs):
        env['SLURM_ARRAY_TASK_ID'] = str(task)
        codes[task] = subprocess.run(
            ['bash', str(script)], env=env, capture_output=True, timeout=20
        ).returncode

    ran = sorted(line.split(' ', 1) for line in record.read_text().splitlines())
    assert ran == [[f'case_{i:06d}', f'{scratch}/case_{i:06d}'] for i in range(n_cases)]
    assert list(scratch.iterdir()) == []
    assert (jobs, waves) == (3, 2)
    # strided: task 0 runs 0, 3, 6 (6 waits for a free core); task 1 runs 1, 4, 7
    assert codes[2] == 0
    assert codes[0] != 0 and codes[1] != 0
    log6 = (tmp_path / 'my grid' / 'logs' / 'case_000006.log').read_text()
    assert 'output of' in log6 and 'cfgs/case_000006.toml' in log6
    first = {
        i: (tmp_path / 'my grid' / f'case_{i:06d}' / 'status').read_text().split('\n')[0]
        for i in range(n_cases)
    }
    assert first == {
        0: '25',
        1: '10',
        2: '10',
        3: '25',
        4: '25',
        5: '10',
        6: '10',
        7: '10',
        8: '10',
    }
