"""
Unit tests for the PROTEUS process that inference workers reuse.

Covers the contract of `proteus.inference.runner`: one process serving many
evaluations, the reason and exit code reported for a run that overran, for one
whose process died without replying and for one that raised, replacement after
`max_jobs`, and the console file a process shares across its simulations.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest

from proteus.inference.runner import ProteusRunner, handle_job

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Loop body recording the serving process, then answering.
_RECORD_PID = """
open(job['config'], 'a').write(str(os.getpid()) + '\\n')
reply.write(json.dumps({'error': None}) + '\\n')
reply.flush()
"""

# Loop body naming the run on the console, then answering.
_ECHO = """
print('ran ' + job['config'], flush=True)
reply.write(json.dumps({'error': None}) + '\\n')
reply.flush()
"""


def _stub_child(tmp_path, body: str) -> list[str]:
    """Command for a stand-in child speaking the runner protocol.

    The real child imports PROTEUS and Julia, far beyond a unit test's budget,
    so a small script answers on the same pipes instead. `body` runs once per
    request, with `job`, `reply` and `os` in scope.
    """
    lines = [
        'import json, os, sys, time',
        'requests = os.fdopen(int(sys.argv[1]), "r")',
        'reply = os.fdopen(int(sys.argv[2]), "w")',
        'for line in requests:',
        '    job = json.loads(line)',
    ]
    lines += ['    ' + line for line in textwrap.dedent(body).strip('\n').splitlines()]

    script = tmp_path / 'stub_child.py'
    script.write_text('\n'.join(lines) + '\n')
    return [sys.executable, str(script)]


def _reply(payload: str = 'None') -> str:
    """Loop body that answers one request and keeps the process alive."""
    return f"reply.write(json.dumps({{'error': {payload}}}) + '\\n')\nreply.flush()"


@pytest.fixture
def runner_factory(tmp_path):
    """Build runners against a stub child, and stop them when the test ends."""
    started = []

    def build(body: str, max_jobs: int = 0) -> ProteusRunner:
        runner = ProteusRunner(
            tmp_path / 'runner_console.log', max_jobs, _stub_child(tmp_path, body)
        )
        started.append(runner)
        return runner

    yield build
    for runner in started:
        runner.stop()


def test_one_process_serves_every_evaluation(runner_factory, tmp_path):
    """Consecutive evaluations reach the same process, so the Julia environment
    is loaded once rather than once per sample.
    """
    pidfile = tmp_path / 'pids.txt'
    runner = runner_factory(_RECORD_PID)

    for _ in range(3):
        assert runner.run(pidfile, timeout=20) is None

    pids = pidfile.read_text().split()
    assert len(pids) == 3
    # Discrimination: one process per evaluation would write three distinct
    # pids, which is exactly the cost this module exists to remove.
    assert len(set(pids)) == 1
    assert pids[0] == str(runner._proc.pid)


def test_finished_simulation_reports_no_failure(runner_factory, tmp_path):
    """A completed run reports nothing, which lets `run_proteus` fall through
    to reading the helpfile, and its process is kept for the next evaluation.
    """
    runner = runner_factory(_reply())

    assert runner.run(tmp_path / 'in.toml', timeout=20) is None
    assert runner._proc is not None
    assert runner._proc.poll() is None


def test_overrunning_simulation_is_killed_and_reported(runner_factory, tmp_path):
    """A run past its timeout cannot be interrupted, so the process is killed
    and the sample scored as failed.
    """
    runner = runner_factory('time.sleep(60)')
    started = time.perf_counter()
    reason, exit_code = runner.run(tmp_path / 'in.toml', timeout=0.5)
    elapsed = time.perf_counter() - started

    assert reason == 'exceeded the 0.5 s timeout'
    # No exit code: stopped rather than ended on its own.
    assert exit_code is None
    # Discrimination: waiting out the child's 60 s sleep, or failing to kill
    # it, would both take far longer than this bound.
    assert elapsed < 20
    assert runner._proc is None


def test_child_that_dies_without_replying_reports_its_exit_code(runner_factory, tmp_path):
    """A crash closes the pipe instead of answering; the exit code is carried
    through, and the next evaluation starts a fresh process.
    """
    runner = runner_factory('sys.exit(7)')
    cfg = tmp_path / 'in.toml'
    reason, exit_code = runner.run(cfg, timeout=20)

    assert reason == 'the simulator exited with an error'
    # Discrimination: hard-coding 0 or 1, or reporting None as the timeout
    # path does, would all fail here.
    assert exit_code == 7
    # A dead process does not wedge the worker for the rest of the study.
    assert runner.run(cfg, timeout=20)[1] == 7


def test_process_survives_a_simulation_that_raised(runner_factory, tmp_path):
    """A parameter combination the simulator cannot integrate is expected, so
    the process is kept and the next evaluation reuses it.
    """
    runner = runner_factory(
        """
        failed = getattr(sys.modules[__name__], 'failed', False)
        payload = None if failed else 'RuntimeError: boom'
        setattr(sys.modules[__name__], 'failed', True)
        reply.write(json.dumps({'error': payload}) + '\\n')
        reply.flush()
        """
    )
    cfg = tmp_path / 'in.toml'

    first = runner.run(cfg, timeout=20)
    pid_after_failure = runner._proc.pid
    second = runner.run(cfg, timeout=20)

    # Exit code 1, as the same failure gives as a one-shot child.
    assert first == ('the simulator exited with an error', 1)
    assert second is None
    # Discrimination: tearing the process down on a failed sample would give a
    # different pid here, and would pay the Julia load again.
    assert runner._proc.pid == pid_after_failure


def test_process_is_replaced_after_max_jobs(runner_factory, tmp_path):
    """`max_jobs` bounds how long one process is kept, guarding against state
    or memory accumulating across a long study.
    """
    runner = runner_factory(_reply(), max_jobs=2)
    cfg = tmp_path / 'in.toml'

    runner.run(cfg, timeout=20)
    first_pid = runner._proc.pid
    runner.run(cfg, timeout=20)
    # The second run reaches the limit, so it is stopped on the way out.
    assert runner._proc is None

    runner.run(cfg, timeout=20)
    assert runner._proc is not None
    # Discrimination: recycling after one or three jobs would put the boundary
    # elsewhere, and `first_pid` would be reused here.
    assert runner._proc.pid != first_pid


@pytest.mark.parametrize('max_jobs', [0, -5])
def test_max_jobs_at_or_below_zero_keeps_one_process(runner_factory, tmp_path, max_jobs):
    """Edge case at the boundary: 0 means no limit, and a negative value is
    clamped rather than becoming a truthy limit that recycles every time.
    """
    runner = runner_factory(_reply(), max_jobs=max_jobs)
    cfg = tmp_path / 'in.toml'

    assert runner.max_jobs == 0
    runner.run(cfg, timeout=20)
    first_pid = runner._proc.pid
    runner.run(cfg, timeout=20)

    assert runner._proc is not None
    assert runner._proc.pid == first_pid


def test_stop_is_safe_before_anything_has_run(runner_factory):
    """Edge case on teardown: workers call `close_worker_runner` in a `finally`,
    which can fire before any evaluation started a process.
    """
    runner = runner_factory(_reply())

    assert runner._proc is None
    runner.stop()
    runner.stop()
    assert runner._proc is None


def test_console_file_collects_output_from_every_simulation(runner_factory, tmp_path):
    """A reused process writes one console file for its whole life, because
    Julia's output stream cannot be re-pointed between simulations.
    """
    runner = runner_factory(_ECHO)

    assert runner.run(tmp_path / 'first.toml', timeout=20) is None
    assert runner.run(tmp_path / 'second.toml', timeout=20) is None
    runner.stop()

    console = runner.console.read_text()
    assert 'first.toml' in console
    # Discrimination: opening for writing rather than appending, or per
    # simulation rather than per process, would leave only the last run.
    assert 'second.toml' in console
    assert console.count('ran ') == 2


def test_console_file_survives_the_process_being_replaced(runner_factory, tmp_path):
    """Edge case at the `max_jobs` boundary: the replacement process appends to
    the same file, keeping the record of runs made before the recycle.
    """
    runner = runner_factory(_ECHO, max_jobs=1)

    runner.run(tmp_path / 'first.toml', timeout=20)
    runner.run(tmp_path / 'second.toml', timeout=20)
    runner.stop()

    console = runner.console.read_text()
    assert 'first.toml' in console
    assert 'second.toml' in console


def test_handle_job_reports_the_exception_type_and_message(capsys):
    """A simulation that raises is described by type and message, and its
    traceback is printed rather than swallowed.
    """

    def explode(config_path):
        raise RuntimeError('mantle did not converge')

    reply = handle_job({'config': 'x.toml'}, explode)
    printed = capsys.readouterr()

    assert reply == {'error': 'RuntimeError: mantle did not converge'}
    assert 'Traceback' in printed.err
    assert 'mantle did not converge' in printed.err
    # The run is named on the console, so a file holding a whole worker's
    # output can still be read one run at a time.
    assert 'x.toml' in printed.out


def test_handle_job_reports_nothing_for_a_simulation_that_finished():
    """The success reply carries an explicit null rather than an absent key, so
    the parent can tell a finished run from a malformed message.
    """
    seen = []

    reply = handle_job({'config': 'x.toml'}, seen.append)

    assert reply == {'error': None}
    assert seen == ['x.toml']
    # The reply travels the pipe, so it has to serialise.
    assert json.loads(json.dumps(reply)) == {'error': None}
