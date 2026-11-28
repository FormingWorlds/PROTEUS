"""Julia thread and signal settings in a new process that imports PROTEUS and starts Julia."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import proteus
from proteus.utils.helper import ReadStatus

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(120)]

_SRC = str(Path(proteus.__file__).resolve().parents[1])

# The two ways to ask for 2 Julia threads: JULIA_NUM_THREADS gives 1 thread and Python
# handles SIGTERM; PYTHON_JULIACALL_THREADS gives 2 threads and Julia's signal handlers.
_ARMS = {
    'julia_num_threads': ('JULIA_NUM_THREADS', 1, None),
    'juliacall_threads': ('PYTHON_JULIACALL_THREADS', 2, 'yes'),
}

_STUB_RUN = """
import sys, time
import proteus.cli as cli
from proteus.proteus import _status_on_exit
from proteus.utils.helper import UpdateStatusfile

class StubProteus:
    def __init__(self, config_path):
        self.directories = {'output': sys.argv[1]}

    @_status_on_exit
    def start(self, resume, offline):
        UpdateStatusfile(self.directories, 1)
        print('running', flush=True)
        time.sleep(60)

cli.Proteus = StubProteus
cli.start.main(['-c', sys.argv[2]])
"""

_HOOK_RUN = """
import sys
from proteus.utils.helper import UpdateStatusfile
from proteus.utils.julia_env import register_status_hook

first, last, status = sys.argv[1:]
UpdateStatusfile({'output': first}, 1)
assert register_status_hook(first)
if status != 'none':
    UpdateStatusfile({'output': last}, int(status))
assert register_status_hook(last)
"""


_STRIPPED = ('PYTHON_JULIACALL_THREADS', 'PYTHON_JULIACALL_HANDLE_SIGNALS', 'JULIA_NUM_THREADS')


def _env(key):
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED}
    env[key] = '2'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [_SRC, env.get('PYTHONPATH')]))
    return env


@pytest.mark.parametrize('arm', _ARMS)
def test_two_thread_request_starts_julia_as_set_and_exits_0(arm):
    """A process that asks for 2 Julia threads and imports PROTEUS exits with 0: through
    JULIA_NUM_THREADS it runs 1 Julia thread, through PYTHON_JULIACALL_THREADS 2 threads
    with Julia's signal handlers on. With 2 threads and the handlers off, the exit of this
    process ends with SIGBUS (code 138) in some tries on macOS, so the exit code alone does
    not show the fix; the thread count and the signal setting do."""
    key, threads, signals = _ARMS[arm]
    code = (
        'import proteus, juliacall\n'
        "print(juliacall.Main.seval('Threads.nthreads()'), juliacall.CONFIG['opt_handle_signals'], flush=True)"
    )
    res = subprocess.run(
        [sys.executable, '-c', code], env=_env(key), capture_output=True, text=True, timeout=100
    )
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.split()[-2:] == [str(threads), str(signals)]


@pytest.mark.parametrize('arm', _ARMS)
def test_sigterm_records_status_25(arm, tmp_path):
    """SIGTERM (scancel, a SLURM time limit) to `proteus start` records status 25 on both
    ways: from the guard of `Proteus.start` when Python handles the signal, from the Julia
    exit hook when Julia does. The stub run writes status 1 and waits for the signal."""
    key, _, signals = _ARMS[arm]
    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')
    out = tmp_path / 'out'
    proc = subprocess.Popen(
        [sys.executable, '-c', _STUB_RUN, str(out), str(cfg)],
        env=_env(key),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == 'running'
        assert ReadStatus(out) == 1
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=60)
    finally:
        proc.kill()
    assert ReadStatus(out) == 25, err[-2000:]
    assert proc.returncode == (128 + signal.SIGTERM if signals is None else -signal.SIGTERM)


@pytest.mark.parametrize(
    'status, expected', [('0', 25), ('1', 25), ('10', 10), ('20', 20), ('none', None)]
)
def test_status_hook_keeps_a_final_status_and_follows_the_last_run(tmp_path, status, expected):
    """With Julia's signal handlers on, the exit hook runs at every exit, also a normal one.
    It writes 25 only while the last run reads as started (0) or running (1); it keeps a
    final status (10, 20), creates no status file and does not write into the folder of an
    earlier run of the same process, which still reads as running."""
    first, last = tmp_path / 'first', tmp_path / 'last'
    res = subprocess.run(
        [sys.executable, '-c', _HOOK_RUN, str(first), str(last), status],
        env=_env('PYTHON_JULIACALL_THREADS'),
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    assert ReadStatus(first) == 1
    if expected is None:
        assert not last.exists() or list(last.iterdir()) == []
    else:
        assert ReadStatus(last) == expected
        assert [p.name for p in last.iterdir()] == ['status']


_MOVE_RUN = """
import os, sys
from proteus.utils.helper import UpdateStatusfile
from proteus.utils.julia_env import register_status_hook

UpdateStatusfile({'output': 'run'}, 1)
print(os.stat('run/status').st_ino, flush=True)
assert register_status_hook('run')
os.makedirs('elsewhere')
os.chdir('elsewhere')
"""


def test_status_hook_writes_a_new_file_into_the_folder_it_was_given(tmp_path):
    """The exit hook takes the output folder as an absolute path, so a later change of the
    working folder does not move the status file. It writes a new file and renames it over
    the old one, so a reader never sees a truncated status."""
    res = subprocess.run(
        [sys.executable, '-c', _MOVE_RUN],
        cwd=tmp_path,
        env=_env('PYTHON_JULIACALL_THREADS'),
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    run = tmp_path / 'run'
    assert ReadStatus(run) == 25
    assert [p.name for p in run.iterdir()] == ['status']
    assert (run / 'status').stat().st_ino != int(res.stdout.split()[0])


_STOP_WORKER = """
import sys
import proteus.proteus
from proteus.inference import runner
from proteus.utils import julia_env

class StubProteus:
    def __init__(self, config_path):
        self.config_path = config_path

    def start(self, offline):
        open(self.config_path, 'a').write('ran\\n')
        if self.config_path.endswith('a.toml'):
            julia_env.STOP_SIGNAL = 15

proteus.proteus.Proteus = StubProteus
runner.serve(int(sys.argv[1]), int(sys.argv[2]))
"""


def test_a_stopped_worker_fails_its_own_job_and_the_next_gets_a_new_worker(
    monkeypatch, tmp_path
):
    """A serve worker whose job recorded a stop ends without a reply, so the parent books
    that job as crashed with 143 and runs the next job in a new worker, not in the one that
    is ending."""
    from proteus.inference.runner import _CRASHED, ProteusRunner

    script = tmp_path / 'worker.py'
    script.write_text(_STOP_WORKER)
    for key in _STRIPPED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PYTHONPATH', _env('JULIA_NUM_THREADS')['PYTHONPATH'])
    runner = ProteusRunner(tmp_path / 'console.log', command=[sys.executable, str(script)])
    try:
        first = runner.run(tmp_path / 'a.toml', timeout=100)
        second = runner.run(tmp_path / 'b.toml', timeout=100)
    except BrokenPipeError:
        pytest.fail('the parent wrote the next job into the worker that is ending')
    finally:
        try:
            runner.stop()
        except BrokenPipeError:
            pass
    assert first == (_CRASHED, 128 + signal.SIGTERM)
    assert second is None
    assert (tmp_path / 'b.toml').read_text() == 'ran\n'
