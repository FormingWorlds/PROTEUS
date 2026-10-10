"""One PROTEUS process reused across an inference worker's evaluations."""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import sys
import traceback
from pathlib import Path

log = logging.getLogger('fwl.' + __name__)

# Reached only if the kernel is yet to deliver SIGKILL, so a guard, not a budget.
_REAP_TIMEOUT_S = 10.0

_CRASHED = 'the simulator exited with an error'


class ProteusRunner:
    """One PROTEUS process, reused across the evaluations of one worker.

    Parameters
    ----------
    - console (str | Path): File collecting the process's console output.
    - max_jobs (int): Replace the process after this many simulations; 0 never.
    - command (list[str] | None): How to start it. Defaults to this module.
    """

    def __init__(
        self, console: str | Path, max_jobs: int = 0, command: list[str] | None = None
    ):
        self.console = Path(console)
        self.max_jobs = max(int(max_jobs), 0)
        self.command = command or [sys.executable, '-m', 'proteus.inference.runner']
        self._proc: subprocess.Popen | None = None
        self._requests = self._replies = self._out = None
        self._jobs = 0

    def run(
        self, config_path: str | Path, timeout: float | None
    ) -> tuple[str, int | None] | None:
        """Run one simulation, reusing the process left by the previous call.

        Returns None when it finished, otherwise the reason it did not and the
        exit code, worded as the one-process-per-run path words them.
        """
        if self._proc is None or self._proc.poll() is not None:
            self.stop()
            self._start()

        try:
            self._requests.write(json.dumps({'config': str(config_path)}) + '\n')
            self._requests.flush()
        except (BrokenPipeError, ValueError):
            # The process died between the check above and this write.
            self.stop()
            return _CRASHED, None

        # An overrun cannot be interrupted, so the process is killed instead.
        if not select.select([self._replies], [], [], timeout)[0]:
            self.stop()
            return f'exceeded the {timeout} s timeout', None

        # No reply means a closed pipe: a crash inside PROTEUS or Julia.
        line = self._replies.readline()
        if not line:
            exit_code = self._reap()
            self.stop()
            return _CRASHED, exit_code

        # A reply `serve` did not write fails this sample, not the whole worker.
        try:
            error = json.loads(line)['error']
        except (ValueError, KeyError, TypeError):
            log.warning(f'Reused PROTEUS process sent an unreadable reply: {line!r}')
            self.stop()
            return _CRASHED, None

        self._jobs += 1
        if self.max_jobs and self._jobs >= self.max_jobs:
            self.stop()

        # A simulation that raised leaves the process usable, so it is kept.
        # Exit code 1, as the same failure gives as a one-shot child.
        if error is None:
            return None
        log.debug(f'Reused PROTEUS process reported: {error}')
        return _CRASHED, 1

    def stop(self) -> None:
        """Kill the process. The next `run` starts a fresh one."""
        if self._proc is None:
            return
        self._proc.kill()
        if self._reap() is None:
            log.warning('Reused PROTEUS process did not exit after being killed')
        for stream in (self._requests, self._replies, self._out):
            if stream is not None:
                stream.close()
        self._proc = None
        self._requests = self._replies = self._out = None

    def _reap(self) -> int | None:
        """Exit code of the process, or None if it has yet to produce one.

        Waited for rather than sampled: end-of-file on the reply pipe can
        arrive before the process itself has been reaped.
        """
        try:
            return self._proc.wait(timeout=_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None

    def _start(self) -> None:
        """Start a fresh process, on pipes of its own for the reason above."""
        self.console.parent.mkdir(parents=True, exist_ok=True)
        self._out = open(self.console, 'a')
        request_r, request_w = os.pipe()
        reply_r, reply_w = os.pipe()
        self._proc = subprocess.Popen(
            [*self.command, str(request_r), str(reply_w)],
            stdin=subprocess.DEVNULL,
            stdout=self._out,
            stderr=subprocess.STDOUT,
            pass_fds=(request_r, reply_w),
            text=True,
            env=dict(os.environ, OMP_NUM_THREADS='1'),
        )
        # Closed here so each side sees end-of-file as soon as the other goes.
        os.close(request_r)
        os.close(reply_w)
        self._requests = os.fdopen(request_w, 'w')
        self._replies = os.fdopen(reply_r, 'r')
        self._jobs = 0


def handle_job(job: dict, run_simulation) -> dict:
    """Run one simulation; report nothing, or what it raised."""
    # Names the run in a console file that collects a whole worker's output.
    print(f'=== {job["config"]}', flush=True)
    try:
        run_simulation(job['config'])
    except Exception as exc:
        # The console file is where the failure report points the reader.
        traceback.print_exc()
        return {'error': f'{type(exc).__name__}: {exc}'}
    return {'error': None}


def serve(request_fd: int, reply_fd: int) -> None:
    """Serve simulation requests on the inherited pipes until they close."""
    from proteus.proteus import Proteus

    def run_simulation(config_path: str) -> None:
        Proteus(config_path=config_path).start(offline=True)

    reply = os.fdopen(reply_fd, 'w')
    for line in os.fdopen(request_fd, 'r'):
        reply.write(json.dumps(handle_job(json.loads(line), run_simulation)) + '\n')
        reply.flush()


if __name__ == '__main__':
    serve(int(sys.argv[1]), int(sys.argv[2]))
