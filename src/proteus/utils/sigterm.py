"""SIGTERM during a run: status 25 at once, and a flag that the solver loops check.

Slurm stops a job with SIGTERM. A ``SystemExit`` raised inside a scikits.odes callback
does not end the run cleanly (bmcage/odes#99): it becomes a ``SystemError`` with status 20,
or CVODE reports a failed step and the exception is lost. So the handler writes status 25
before it raises, and the Aragog retry ladder and the main loop raise again from the flag.
"""

from __future__ import annotations

import functools
import signal

from proteus.utils.helper import ReadStatus, UpdateStatusfile

# Signal number of the SIGTERM the current run received, or None.
_received: int | None = None


def stop_on_sigterm(start):
    """Run ``start`` with a SIGTERM handler that records status 25 and raises SystemExit.

    The handler replaces only a running status (0 or 1), and the exit code is
    ``128 + signum``, as for a process ended by the signal. After the call the handler
    from before it is back and the flag is clear.
    """

    @functools.wraps(start)
    def wrapper(self, *args, **kwargs):
        def handler(signum, _frame):
            global _received
            _received = signum
            if ReadStatus(self.directories['output']) in (0, 1):
                UpdateStatusfile(self.directories, 25)
            raise SystemExit(128 + signum)

        global _received
        before = signal.signal(signal.SIGTERM, handler)
        try:
            return start(self, *args, **kwargs)
        finally:
            signal.signal(signal.SIGTERM, before)
            _received = None

    return wrapper


def exit_if_terminated() -> None:
    """Raise SystemExit when the run received SIGTERM, also after the first one was lost."""
    if _received is not None:
        raise SystemExit(128 + _received)
