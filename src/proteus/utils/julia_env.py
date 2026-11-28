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

# The line `print_system_configuration` logs when `set_julia_threads` changed the setting
JULIA_THREADS_NOTE = ''

_status_hook = None


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


def set_julia_threads(environ=os.environ, xoptions=None) -> str:
    """Choose the Julia threads and signal handling before juliacall starts Julia.

    A thread count from ``-X juliacall-threads`` or ``PYTHON_JULIACALL_THREADS`` is the
    user's choice and is kept; above 1 thread, ``PYTHON_JULIACALL_HANDLE_SIGNALS`` is set
    to ``yes`` unless already set. Without it, juliacall would take ``JULIA_NUM_THREADS``;
    above 1 thread, ``PYTHON_JULIACALL_THREADS`` is set to 1 instead. Has no effect in a
    process that has already started juliacall.

    Parameters
    ----------
    environ : MutableMapping[str, str]
        The environment to read and change.
    xoptions : Mapping or None
        The ``-X`` options; ``sys._xoptions`` when None.

    Returns
    -------
    str
        The line to log when the thread count was set to 1, else an empty string.

    Notes
    -----
    With the Julia handlers on, a SLURM time limit under ``srun`` can leave status 1:
    2 SIGTERMs within a few ms can end the process before the Julia exit hook runs.
    Under ``exec`` in the batch script the process gets 1 SIGTERM and records status 25.
    """
    global JULIA_THREADS_NOTE
    xoptions = sys._xoptions if xoptions is None else xoptions
    user = xoptions.get('juliacall-threads', environ.get('PYTHON_JULIACALL_THREADS'))
    if user is not None:
        if _more_than_one(user):
            environ.setdefault('PYTHON_JULIACALL_HANDLE_SIGNALS', 'yes')
        return ''
    julia = environ.get('JULIA_NUM_THREADS', '1')
    if not _more_than_one(julia):
        return ''
    environ['PYTHON_JULIACALL_THREADS'] = '1'
    JULIA_THREADS_NOTE = (
        f'Julia threads     1 (JULIA_NUM_THREADS={julia} not used; '
        'set PYTHON_JULIACALL_THREADS for more)'
    )
    return JULIA_THREADS_NOTE


def julia_handles_signals() -> bool:
    """True when juliacall started Julia with its signal handlers (``handle_signals = yes``)."""
    import juliacall

    return juliacall.CONFIG.get('opt_handle_signals') == 'yes'


def _exit_on_sigterm(signum, _frame):
    signal.signal(signum, signal.SIG_IGN)
    sys.exit(128 + signum)


def exit_on_sigterm() -> bool:
    """End the process on SIGTERM (scancel, a SLURM time limit) with SystemExit, so that
    ``Proteus.start`` records status 25.

    Later SIGTERMs are ignored: at a time limit srun and slurmstepd both send one, and a
    second SystemExit would stop the status write. SIGTERM is unblocked, since a process
    started from one whose Julia handles the signals inherits it blocked. When Julia
    handles the signals, its handler stays: a Python one would also stop the Julia exit hook.

    Returns
    -------
    bool
        True when the Python handler is installed.
    """
    if julia_handles_signals():
        return False
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    return True


def register_status_hook(output_dir: str, text: str) -> bool:
    """Write status 25 at Julia exit while the run still reads as started or running.

    With ``handle_signals`` on, Julia ends the process on SIGTERM and SIGINT before
    Python can record the stop; a Julia ``atexit`` hook still runs. It writes ``text``
    through a temporary file and a rename, so the status file is never absent. One hook
    per process; a later call moves it to the new output folder.

    Parameters
    ----------
    output_dir : str
        Output folder of the run; the hook writes ``<output_dir>/status``.
    text : str
        Content of the status file for status 25.

    Returns
    -------
    bool
        True when the hook is set, False when Julia does not handle the signals.
    """
    global _status_hook
    import juliacall

    if not julia_handles_signals():
        return False
    if _status_hook is None:
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
    _status_hook(os.path.join(os.path.abspath(output_dir), 'status'), text)
    return True
