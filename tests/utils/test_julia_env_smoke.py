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


def _env(key):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('PYTHON_JULIACALL_', 'JULIA_NUM_THREADS'))
    }
    env[key] = '2'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [_SRC, env.get('PYTHONPATH')]))
    return env


@pytest.mark.parametrize('arm', _ARMS)
def test_two_thread_request_starts_julia_as_set_and_exits_0(arm):
    """A process that asks for 2 Julia threads and imports PROTEUS exits with 0: through
    JULIA_NUM_THREADS it runs 1 Julia thread, through PYTHON_JULIACALL_THREADS 2 threads
    with Julia's signal handlers on. With 2 threads and the handlers off, the exit of this
    process ends with SIGBUS (code 138) in most tries on macOS."""
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
