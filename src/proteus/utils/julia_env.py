"""Julia thread and signal settings that PROTEUS applies before it imports juliacall.

With more than one Julia thread and Julia's signal handlers off (the juliacall default), a
process can end with SIGBUS (exit code 138 on macOS) in a Julia worker thread at exit.
With the handlers on, Julia takes SIGTERM and SIGINT, so Python never records status 25.
PROTEUS therefore starts Julia with 1 thread unless the user asks for more through juliacall
itself; then it turns the handlers on and writes status 25 from a Julia exit hook.
"""

from __future__ import annotations

import os
import signal
import sys
from contextlib import contextmanager

# The line `print_system_configuration` logs when 1 thread runs instead of JULIA_NUM_THREADS
JULIA_THREADS_NOTE = ''

_status_hook = None

# The signal number once a Python stop handler (SIGTERM, SIGINT) has run
STOP_SIGNAL = None
_status_output = None
# True while a status write must finish: a stop handler then only records the signal
_writing = False


def _more_than_one(value: str) -> bool:
    """True when a thread setting such as '2', '1,1' or 'auto' asks for more than 1 thread.

    Parameters
    ----------
    value : str
        A value in the form of ``JULIA_NUM_THREADS``: ``N`` or ``N,M`` (M interactive
        threads), where ``N`` and ``M`` can also be ``auto``.

    Returns
    -------
    bool
        False for 1 thread and for a value Julia does not accept (left to Julia to report).
    """
    parts = [p.strip() for p in str(value).split(',')]
    if 'auto' in parts:
        return True
    try:
        counts = [int(p) for p in parts]
    except ValueError:
        return False
    return min(counts) >= 0 and sum(counts) > 1


def set_julia_threads(environ=os.environ, xoptions=None) -> None:
    """Choose the Julia threads and signal handling before juliacall starts Julia.

    A thread count from ``-X juliacall-threads`` or ``PYTHON_JULIACALL_THREADS`` is the
    user's choice and is kept; above 1 thread, ``PYTHON_JULIACALL_HANDLE_SIGNALS`` is set
    to ``yes`` unless already set. Without it, juliacall would take ``JULIA_NUM_THREADS``;
    above 1 thread, ``PYTHON_JULIACALL_THREADS`` is set to 1 instead. ``JULIA_THREADS_NOTE``
    is set whenever 1 thread runs while ``JULIA_NUM_THREADS`` asks for more, also in a
    child process that inherits the setting. Has no effect once juliacall is imported.

    Parameters
    ----------
    environ : MutableMapping[str, str]
        The environment to read and change.
    xoptions : Mapping or None
        The ``-X`` options; ``sys._xoptions`` when None.

    Notes
    -----
    With the Julia handlers on, a SLURM time limit under ``srun`` can leave status 1:
    2 SIGTERMs within a few ms can end the process before the Julia exit hook runs.
    Under ``exec`` in the batch script the process gets 1 SIGTERM and records status 25.
    """
    global JULIA_THREADS_NOTE
    if 'juliacall' in sys.modules:
        return
    xoptions = sys._xoptions if xoptions is None else xoptions
    julia = environ.get('JULIA_NUM_THREADS', '1')
    user = xoptions.get('juliacall-threads', environ.get('PYTHON_JULIACALL_THREADS'))
    if user is None and _more_than_one(julia):
        user = environ['PYTHON_JULIACALL_THREADS'] = '1'
    elif user is not None and _more_than_one(user):
        environ.setdefault('PYTHON_JULIACALL_HANDLE_SIGNALS', 'yes')
    if str(user).strip() == '1' and _more_than_one(julia):
        JULIA_THREADS_NOTE = (
            f'Julia threads     1 (JULIA_NUM_THREADS={julia} not used; '
            'set PYTHON_JULIACALL_THREADS for more)'
        )


def julia_handles_signals() -> bool:
    """True when juliacall started Julia with its signal handlers (``handle_signals = yes``)."""
    import juliacall

    return juliacall.CONFIG.get('opt_handle_signals') == 'yes'


def _record_stop(signum) -> bool:
    """Record the stop, and write status 25 while the run reads as started or running.

    The status is on disk before the handler raises, so a stop whose exception a finalizer
    or a C solver drops, or a SIGKILL that follows it, still leaves status 25. During a
    held status write (`sigterm_held`) only the signal is recorded.

    Returns
    -------
    bool
        True when the handler raises, False during a held status write.
    """
    global STOP_SIGNAL
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    STOP_SIGNAL = signum
    if _writing:
        return False
    if _status_output is not None and ReadStatus(_status_output) in (0, 1):
        with sigterm_held():
            UpdateStatusfile({'output': _status_output}, 25)
    return True


def _exit_on_sigterm(signum, _frame):
    if _record_stop(signum):
        sys.exit(128 + signum)


def _interrupt(signum, frame):
    if _record_stop(signum):
        signal.default_int_handler(signum, frame)


_HANDLERS = {signal.SIGTERM: _exit_on_sigterm, signal.SIGINT: _interrupt}


def exit_if_stopped() -> None:
    """End the process with 128 + signal when a stop is recorded in ``STOP_SIGNAL``.

    A C solver (CVODE) or a finalizer (a weakref callback, ``__del__``) can drop the
    SystemExit or KeyboardInterrupt that a stop handler raises. The main loop calls this once
    per iteration and after the final snapshots, code that retries or skips a failed step
    calls it first, and so do ``proteus start`` after the run and ``serve`` after each job.
    """
    if STOP_SIGNAL is not None:
        sys.exit(128 + STOP_SIGNAL)


def exit_on_sigterm() -> bool:
    """End the process on SIGTERM (scancel, a SLURM time limit) with SystemExit, so that
    ``Proteus.start`` records status 25.

    The handler, and a SIGINT handler that raises KeyboardInterrupt, record the signal in
    ``STOP_SIGNAL``, so the run reads as stopped also when a C solver (CVODE) turns the
    exception raised in its callback into its own error, or drops it; see `run_was_stopped`
    and `exit_if_stopped`. SIGTERM is unblocked, since a process
    started from one whose Julia handles the signals inherits it blocked. When Julia
    handles the signals, its handler stays: a Python one would also stop the Julia exit hook.

    Returns
    -------
    bool
        True when the Python handler is installed.
    """
    if julia_handles_signals():
        return False
    for signum, handler in _HANDLERS.items():
        signal.signal(signum, handler)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    return True


def run_was_stopped(exc: BaseException) -> bool:
    """True when ``exc`` ends a run that was stopped, not one that failed.

    A run was stopped when a Python stop handler has run, or when ``exc`` or an exception
    that its cause and context links reach is a KeyboardInterrupt or SystemExit.

    Parameters
    ----------
    exc : BaseException
        The exception that ends the run.

    Returns
    -------
    bool
        True for a stop (status 25), False for an error (status 20).
    """
    todo, seen = [exc], set()
    while todo:
        exc = todo.pop()
        if exc is None or id(exc) in seen:
            continue
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            return True
        seen.add(id(exc))
        todo += [exc.__cause__, exc.__context__]
    return STOP_SIGNAL is not None


@contextmanager
def sigterm_held():
    """Let a status write finish when SIGTERM or SIGINT comes: the Python stop handlers
    only record the signal in ``STOP_SIGNAL`` until the write ends.

    At a SLURM time limit srun and slurmstepd both send SIGTERM, and Ctrl-C can come twice;
    an exception would stop the status write. The caller ends the process afterwards: the
    handler raises after its own write, the status guard calls `exit_if_stopped`.
    """
    global _writing
    before, _writing = _writing, True
    try:
        yield
    finally:
        _writing = before


def register_status_hook(output_dir: str) -> bool:
    """Write status 25 at Julia exit while the run still reads as started or running.

    With ``handle_signals`` on, Julia ends the process on SIGTERM and SIGINT before
    Python can record the stop; a Julia ``atexit`` hook still runs. It writes status 25
    through a temporary file and a rename, so the status file is never absent. One hook
    per process; a later call moves it to the new output folder. The folder is also where
    the Python stop handler writes status 25.

    Parameters
    ----------
    output_dir : str
        Output folder of the run; the hook writes ``<output_dir>/status``.

    Returns
    -------
    bool
        True when the hook is set, False when Julia does not handle the signals.
    """
    global _status_hook, _status_output
    from proteus.utils.helper import CommentFromStatus

    _status_output = os.path.abspath(output_dir)
    if not julia_handles_signals():
        return False
    if _status_hook is None:
        import juliacall

        _status_hook = juliacall.Main.seval("""
            let path = Ref(""), text = Ref("")
                Base.atexit() do
                    s = try parse(Int, readline(path[])) catch; -1 end
                    if s in (0, 1)
                        write(path[] * ".tmp", text[])
                        Base.Filesystem.rename(path[] * ".tmp", path[])
                    end
                end
                (p, t) -> (path[] = p; text[] = t; nothing)
            end""")
    _status_hook(os.path.join(_status_output, 'status'), f'25\n{CommentFromStatus(25)}\n')
    return True
