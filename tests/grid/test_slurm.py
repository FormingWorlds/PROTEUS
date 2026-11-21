"""Unit tests for ``proteus.grid.slurm``, the job array that packs grid cases into jobs.

Checks: every case lands in exactly one array task and runs once; no job runs more
cases than its cores times its waves; the array never exceeds the cluster limit; a
failed case does not stop the other cases of its job; wall times convert to the
nearest minute.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import time
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
        (50, 4, 1001, (13, 1)),  # the check layouts of the grid layout explorer docs
        (50, 4, 5, (5, 3)),
    ],
    ids=[
        'rome-eighth',
        'rome-node',
        'habrok-limit',
        'two-waves',
        'one-over',
        'single',
        'docs',
        'docs-limit',
    ],
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
    ('n_cases', 'cores', 'max_array', 'expected'),
    [
        (1000, 16, 1001, 8 / 1008),  # 8 of 63 jobs have 15 cases on 16 cores
        (20, 16, 1001, 0.375),  # 2 jobs of 10 cases on 16 cores
        (20000, 16, 1000, 0.375),  # 20 cases per job: the second wave fills 4 cores
        (20000, 20, 1000, 0.0),
        (5, 16, 1001, 11 / 16),  # fewer cases than cores in the only job
        (50, 4, 1001, 2 / 52),  # the check layouts of the grid layout explorer docs
        (50, 4, 5, 10 / 60),
    ],
    ids=[
        'rome-eighth',
        'small-grid',
        'two-waves',
        'two-waves-full',
        'few-cases',
        'docs',
        'docs-limit',
    ],
)
def test_idle_share_counts_core_time_without_a_case(n_cases, cores, max_array, expected):
    """With equal run times, the idle share is the allocated core time in which a core
    of a job has no case, as a fraction of all allocated core time."""
    share = slurm.idle_share(n_cases, cores, max_array)
    assert share == pytest.approx(expected, abs=1e-12)
    assert 0 <= share < 1


@pytest.mark.parametrize(
    ('cores', 'expected'), [(1, 0.0), (2, 1 - 2500 / 2998)], ids=['one', 'two']
)
def test_idle_share_ends_a_job_with_fewer_cases_earlier(cores, expected):
    """2500 cases over 1001 jobs give jobs of 3 and of 2 cases. A job of 2 cases ends
    earlier and frees its cores, so counting every job as long as the longest one
    overestimates the idle share by more than 0.1."""
    jobs, waves = slurm.array_layout(2500, cores, 1001)
    share = slurm.idle_share(2500, cores, 1001)
    assert share == pytest.approx(expected, abs=1e-12)
    assert 1 - 2500 / (jobs * cores * waves) - share > 0.1  # 0.168 and 0.210


def test_one_wave_of_ten_jobs_or_more_stays_under_the_warning_share():
    """With one wave, the idle cores add up to fewer than one job has, so the idle share
    is below 1 / jobs: at least 10 jobs keep it below the 10 % warning share."""
    for cores in (1, 2, 3, 7, 16, 128):
        for n_cases in range(1, 3000, 7):
            jobs, waves = slurm.array_layout(n_cases, cores, 1001)
            share = slurm.idle_share(n_cases, cores, 1001)
            assert 0 <= share < 1
            if waves == 1:
                assert share < 1 / jobs
                if jobs >= 10:
                    assert share < slurm.IDLE_WARN_SHARE


def test_warn_idle_cores_names_the_share_and_nearer_layouts(caplog):
    """20 cases on 16 cores leave 37.5 % idle; the warning says so, says it assumes
    equal run times, and lists the three nearest core counts that idle at most 10 %."""
    with caplog.at_level(logging.WARNING, logger=slurm.log.name):
        slurm.warn_idle_cores(20, 16, 1001, max_jobs=64)
    (record,) = caplog.records
    text = record.getMessage()
    assert '37.5%' in text
    assert 'equal run times' in text
    # 20 (distance 4), then 11 and 21 (distance 5); 15 idles 33 % and is left out
    assert text.endswith('20 (0%), 11 (9%), 21 (5%)')


def test_warn_idle_cores_proposes_no_more_cores_than_max_jobs(caplog):
    """With max_jobs = 16 cases at a time, 20 and 21 cores would run more cases at once;
    the next nearest values that idle at most 10 % are 11, 10 and 7."""
    with caplog.at_level(logging.WARNING, logger=slurm.log.name):
        slurm.warn_idle_cores(20, 16, 1001, max_jobs=16)
    text = caplog.records[0].getMessage()
    assert '37.5%' in text
    assert text.endswith('11 (9%), 10 (0%), 7 (5%)')


@pytest.mark.parametrize(
    ('n_cases', 'cores', 'warns'),
    [(1000, 16, False), (9, 10, False), (8, 9, True), (1, 4, True)],
    ids=['rome-eighth', 'ten-percent', 'eleven-percent', 'one-case'],
)
def test_warn_idle_cores_only_above_ten_percent(caplog, n_cases, cores, warns):
    """The warning starts above 10 % idle: 9 cases on 10 cores (10 %) stay quiet, 8
    cases on 9 cores (11 %) warn. A single case on 4 cores still gets a suggestion."""
    with caplog.at_level(logging.WARNING, logger=slurm.log.name):
        slurm.warn_idle_cores(n_cases, cores, 1001, max_jobs=64)
    assert bool(caplog.records) is warns
    if warns:
        assert caplog.records[0].getMessage().split(': ')[-1].startswith(f'{n_cases} (0%)')


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


# srun stand-in: drop the options, run the command
_SRUN = 'while [ "${1#-}" != "$1" ]; do shift; done\nexec "$@"\n'


def _fake_tools(tmp_path, **tools):
    """Write each stand-in shell tool to tmp_path/bin and return an environment that
    finds them first, with TMPDIR at tmp_path."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for name, text in tools.items():
        (bindir / name).write_text('#!/bin/sh\n' + text)
        (bindir / name).chmod(0o755)
    return dict(
        os.environ, PATH=f'{bindir}{os.pathsep}{os.environ["PATH"]}', TMPDIR=str(tmp_path)
    )


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
        (1.0, 1440, 900, 85440),
        (2.3, 3312, 900, 197760),  # 2.3 * 1440 is 3311.9999; truncation writes 3311
        (1 / 24, 60, 360, 3180),  # a tenth of 3600 s is 360 s, below the cap
        (0.02, 29, 174, 1506),  # 28.8 min rounds to 29; a tenth of 1740 s is 174 s
        (0.001, 1, 6, 48),  # the 60 s margin shrinks to the 6 s lead
        (0.0001, 1, 6, 48),  # 0.144 min rounds to 0, which Slurm reads as no limit
    ],
    ids=['day', 'float-below-minute', 'hour', 'half-hour', 'one-minute', 'below-a-minute'],
)
def test_walltime_signal_and_last_start_agree(tmp_path, days, minutes, lead, offset):
    """The wall time is written in whole minutes, at least one. The early SIGTERM goes to
    the batch shell 900 s before it, or a tenth of it for short jobs. As a fallback for a
    signal that doesn't come, the script starts no case from 60 s before the signal on,
    or from the lead before it when the lead is shorter: the offset from the job start
    is the wall time minus the lead minus that margin."""
    text = _script(tmp_path, n_cases=4, cores=2, jobs=2, command='x', days=days)
    assert f'#SBATCH --time={minutes}\n' in text
    assert f'#SBATCH --signal=B:TERM@{lead}\n' in text
    assert f'export stop_at=$(( $(date +%s) + {offset} )) ' in text
    assert '#SBATCH --time=0\n' not in text


def test_dispatch_script_runs_each_case_once_and_gives_killed_cases_a_status(tmp_path):
    """Run the script for every array task, with srun replaced by a pass-through.
    Fifteen cases in five jobs of two cores (two waves) run once each, in a folder with
    a space, each with its own temporary folder that is gone afterwards.

    Task 0 has a case that exits 255 and task 1 a case killed by SIGKILL; either stops
    a bare xargs, yet the case queued behind it runs. Both get status 25. Each case's
    temporary folder carries its job id, so two jobs on one node don't share one. Task 2's
    only failure is a case that returns 0 while its status still says running: it gets
    status 25 and fails its job. In task 3 one case writes the error status 22 and
    exits 1, as PROTEUS does on a model error, and one writes 22 and exits 255, which
    stops a bare xargs: both keep 22 and the case queued behind them runs. Task 4 has no
    failure and exits 0."""
    record = tmp_path / 'ran.txt'
    # Normal cases sleep, so a failing case beside them ends first and the next case
    # of its job starts only after the failure
    fakecase = (
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
    os.makedirs(tmp_path / 'my grid' / 'logs')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    n_cases, cores = 15, 2
    jobs, waves = slurm.array_layout(n_cases, cores, max_array=5)
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases, cores, jobs, command='fakecase'))
    env = {**_fake_tools(tmp_path, srun=_SRUN, fakecase=fakecase), 'TMPDIR': str(scratch)}

    # The array tasks run at the same time, as on a cluster
    tasks = [
        subprocess.Popen(
            ['bash', str(script)],
            env={**env, 'SLURM_ARRAY_TASK_ID': str(task), 'SLURM_JOB_ID': str(100 + task)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for task in range(jobs)
    ]
    codes = {task: proc.wait(timeout=20) for task, proc in enumerate(tasks)}

    # strided: task t runs cases t, t + 5 and t + 10, the last after a core frees up
    assert (jobs, waves) == (5, 2)
    ran = sorted(line.split(' ', 1) for line in record.read_text().splitlines())
    assert ran == [
        [f'case_{i:06d}', f'{scratch}/{100 + i % jobs}_case_{i:06d}'] for i in range(n_cases)
    ]
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


def test_dispatch_script_starts_no_case_after_the_last_start_time(tmp_path):
    """Once the clock passes the last start time, the job starts none of its remaining
    cases, also without a signal: they keep no status folder, as not started, and the
    job exits non-zero."""
    marker, calls = tmp_path / 'started', tmp_path / 'date_called'
    # date stand-in: the job starts at t = 1000 s, every later reading is two days on
    env = _fake_tools(
        tmp_path,
        srun=f'touch "{marker}"\n',
        date=f'if [ -e "{calls}" ]; then echo 173800; else touch "{calls}"; echo 1000; fi\n',
    )
    os.makedirs(tmp_path / 'my grid' / 'logs')
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases=3, cores=2, jobs=1, command='x', days=1.0))
    done = subprocess.run(
        ['bash', str(script)],
        env={**env, 'SLURM_ARRAY_TASK_ID': '0'},
        capture_output=True,
        timeout=20,
    )
    assert done.returncode != 0
    assert not marker.exists()
    assert not any((tmp_path / 'my grid').glob('case_*'))
    assert done.stdout.decode().count('not started') == 3


def test_a_case_killed_for_memory_gets_status_30_and_other_kills_25(tmp_path):
    """srun reports a case the kernel killed for memory in the case log; the script then
    writes status 30 instead of the 25 of any other case that ended without a final status.
    A line that only mentions memory, not in srun's form, does not count. Both cases fail
    the job, and a case that finished keeps its own status."""
    oom = 'srun: error: tcn7: task 0: Out Of Memory'
    # srun stand-in: run the case, then report a kill for memory as srun does for case 0
    srun = _SRUN.replace(
        'exec "$@"', f'"$@"; c=$?; [ $c -eq 137 ] && echo "{oom}" >&2; exit $c'
    )
    fakecase = (
        'n=$(basename "$1" .toml); out="$(dirname "$(dirname "$1")")/$n"\n'
        'mkdir -p "$out"; echo 1 > "$out/status"\n'
        'case "$n" in\n'
        '  case_000000) kill -9 $$ ;;\n'
        '  case_000001) echo "the case ran Out Of Memory in a log line"; exit 1 ;;\n'
        'esac\n'
        'printf "10\\nCompleted\\n" > "$out/status"\n'
    )
    os.makedirs(tmp_path / 'my grid' / 'logs')
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases=3, cores=3, jobs=1, command='fakecase'))
    env = _fake_tools(tmp_path, srun=srun, fakecase=fakecase)
    done = subprocess.run(
        ['bash', str(script)],
        env={**env, 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_JOB_ID': '7'},
        capture_output=True,
        timeout=20,
    )
    status = {
        i: (tmp_path / 'my grid' / f'case_{i:06d}' / 'status').read_text().splitlines()
        for i in range(3)
    }
    assert done.returncode != 0
    assert status[0] == ['30', 'Error (killed for memory)']
    assert status[1] == ['25', 'Error (died, or exit requested by user)']
    assert status[2][0] == '10'
    assert oom in (tmp_path / 'my grid' / 'logs' / 'case_000000.log').read_text()


def _default_sigterm() -> None:
    """Give the batch shell the default SIGTERM action, unblocked: a CI runner can start
    the tests with SIGTERM ignored or blocked, and the shell and its cases inherit that."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGTERM])


@pytest.mark.parametrize('ending', [False, True], ids=['early-signal', 'scancel'])
def test_sigterm_to_the_batch_shell_stops_new_cases_and_reaches_running_ones(tmp_path, ending):
    """A SIGTERM to the batch shell, as the early signal or a scancel sends it, is
    logged with its UTC time and the seconds since the job started, and sent on with
    scancel to the job steps. After a scancel of the job, Slurm signals the steps itself
    and the forward fails; its error stays in the log, so a forward that fails for
    another reason shows too. The two running cases get the signal, write status 25 and
    exit 143, as PROTEUS does. The two queued cases don't start, and the shell waits for
    the running cases and exits non-zero."""
    pids, calls, ready = tmp_path / 'pids', tmp_path / 'scancel_args', tmp_path / 'ready'
    # scancel stand-in: signal every step, not the batch shell; on a job that is ending,
    # Slurm has signalled the steps and scancel fails
    fail = 'echo "scancel: error: Invalid job id specified" >&2; exit 1' if ending else 'exit 0'
    env = _fake_tools(
        tmp_path,
        srun=f'echo $$ >> "{pids}"\n' + _SRUN,
        scancel=f'echo "$@" > "{calls}"\nkill -TERM $(cat "{pids}") 2>/dev/null\n{fail}\n',
        fakecase='out="$(dirname "$(dirname "$1")")/$(basename "$1" .toml)"\n'
        'mkdir -p "$out"; echo 1 > "$out/status"\n'
        'trap \'kill $!; sleep 0.2; printf "25\\nError\\n" > "$out/status"; exit 143\' TERM\n'
        f'sleep 30 & echo >> "{ready}"; wait $!\n',
    )
    os.makedirs(tmp_path / 'my grid' / 'logs')
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases=4, cores=2, jobs=1, command='fakecase'))
    job = subprocess.Popen(
        ['bash', str(script)],
        env={**env, 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_JOB_ID': '4242'},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=_default_sigterm,
    )
    try:
        for _ in range(500):
            if ready.exists() and ready.read_text() == '\n\n':
                break
            time.sleep(0.01)
        job.send_signal(signal.SIGTERM)
        out = job.communicate(timeout=10)[0].decode()
    finally:
        if job.poll() is None:
            os.killpg(job.pid, signal.SIGKILL)

    assert job.returncode != 0
    assert re.search(
        r'^SIGTERM at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ, \d+ s into the job: ', out, re.M
    )
    assert calls.read_text().split() == ['--signal=TERM', '4242']
    assert ('Invalid job id' in out) is ending
    status = {
        i: (tmp_path / 'my grid' / f'case_{i:06d}' / 'status').read_text().split('\n')[0]
        for i in (0, 1)
    }
    assert status == {0: '25', 1: '25'}
    assert 'case_000000 failed with exit code 143' in out
    assert not (tmp_path / 'my grid' / 'case_000002').exists()
    assert out.count('not started: the job got SIGTERM') == 2


@pytest.mark.parametrize(
    ('status', 'fails'),
    [(20, True), (28, True), (29, False)],
    ids=['first-error', 'last-error', 'planet-evaporated'],
)
def test_error_status_fails_the_job_also_after_exit_code_0(tmp_path, status, fails):
    """A case that writes an error status (20 to 28) and exits with code 0 keeps its
    status and fails its job, so sacct doesn't show the job as completed. Status 29,
    "Completed (planet evaporated)", is no error: that case and its job succeed."""
    env = _fake_tools(
        tmp_path,
        srun=_SRUN,
        fakecase='out="$(dirname "$(dirname "$1")")/$(basename "$1" .toml)"\n'
        f'mkdir -p "$out"; printf "{status}\\nEnded\\n" > "$out/status"\n',
    )
    os.makedirs(tmp_path / 'my grid' / 'logs')
    script = tmp_path / 'slurm_dispatch.sh'
    script.write_text(_script(tmp_path, n_cases=1, cores=1, jobs=1, command='fakecase'))
    done = subprocess.run(
        ['bash', str(script)],
        env={**env, 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_JOB_ID': '7'},
        capture_output=True,
        timeout=20,
    )
    assert (done.returncode != 0) is fails
    assert (f'case_000000 ended with error status {status}' in done.stdout.decode()) is fails
    assert (
        (tmp_path / 'my grid' / 'case_000000' / 'status').read_text().startswith(f'{status}\n')
    )


def test_dispatch_script_runs_only_the_listed_cases_once_each(tmp_path):
    """With a case list, the tasks run the listed cases, gaps in the numbering included,
    each once, and no other case: five listed cases in two jobs of two cores, so task 0
    runs list lines 1, 3 and 5 and task 1 lines 2 and 4."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'srun').write_text(
        '#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift; done\nexec "$@"\n'
    )
    record = tmp_path / 'ran.txt'
    (bindir / 'fakecase').write_text(
        f'#!/bin/sh\necho "$SLURM_ARRAY_TASK_ID $(basename "$1" .toml)" >> "{record}"\n'
    )
    for tool in ('srun', 'fakecase'):
        os.chmod(bindir / tool, 0o755)
    listed = ['case_000000', 'case_000002', 'case_000003', 'case_000007', 'case_000011']
    listing = tmp_path / 'my grid' / 'cases.txt'
    os.makedirs(tmp_path / 'my grid' / 'logs')
    listing.write_text(''.join(f'{name}\n' for name in listed))
    jobs, _ = slurm.array_layout(len(listed), 2, max_array=2)
    script = tmp_path / 'dispatch.sh'
    script.write_text(
        slurm.dispatch_script(
            cfgdir=str(tmp_path / 'my grid' / 'cfgs'),
            logdir=str(tmp_path / 'my grid' / 'logs'),
            n_cases=len(listed),
            cores_per_job=2,
            jobs=jobs,
            throttle=jobs,
            days=1.0,
            mem_per_case_mb=1024,
            command='fakecase',
            case_list=str(listing),
        )
    )
    env = dict(os.environ, PATH=f'{bindir}{os.pathsep}{os.environ["PATH"]}')
    for task in range(jobs):
        subprocess.run(
            ['bash', str(script)], env={**env, 'SLURM_ARRAY_TASK_ID': str(task)}, check=True
        )

    ran = sorted(line.split() for line in record.read_text().splitlines())
    assert jobs == 2
    assert ran == sorted(
        [['0', 'case_000000'], ['0', 'case_000003'], ['0', 'case_000011']]
        + [['1', 'case_000002'], ['1', 'case_000007']]
    )
