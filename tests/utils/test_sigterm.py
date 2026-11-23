"""Tests for ``proteus.utils.sigterm``: the SIGTERM handler of a run and its flag.

Contract checked:
- the handler writes status 25 over a running status (0 or 1) and keeps any other one;
- it raises SystemExit with code 143, also when the run swallows the first one;
- a SystemError from a solver callback after the signal leaves status 25, not 20;
- the handler from before the run is back afterwards, and a new run clears the flag.

The tests call the installed handler instead of sending a real SIGTERM, because with
PYTHON_JULIACALL_HANDLE_SIGNALS=yes (as in CI) Julia takes the signal and ends pytest.
"""

from __future__ import annotations

import contextlib
import signal

import pytest

from proteus.proteus import _status_on_exit
from proteus.utils import sigterm
from proteus.utils.helper import ReadStatus, UpdateStatusfile

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _deliver_sigterm():
    """Call the installed SIGTERM handler, as Python does when the signal arrives."""
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)


class _Run:
    """A run whose ``start`` writes ``status`` and then calls ``body``."""

    def __init__(self, tmp_path, status, body):
        self.directories = {'output': str(tmp_path)}
        self.status, self.body = status, body

    @sigterm.stop_on_sigterm
    @_status_on_exit
    def start(self):
        if self.status is not None:
            UpdateStatusfile(self.directories, self.status)
        return self.body()


@pytest.mark.parametrize(('before', 'after'), [(0, 25), (1, 25), (22, 22), (11, 11)])
def test_sigterm_writes_status_25_over_a_running_status_only(tmp_path, before, after):
    """A run stopped by SIGTERM records status 25 when it was starting (0) or running (1).
    A final status the run wrote before the signal, an error (22) or a completion (11),
    stays. The exit code is 128 + 15."""
    with pytest.raises(SystemExit) as stop:
        _Run(tmp_path, before, _deliver_sigterm).start()
    assert stop.value.code == 143
    assert ReadStatus(tmp_path) == after


def test_sigterm_before_the_first_status_leaves_the_output_folder_alone(tmp_path):
    """A signal before the run wrote any status creates no status file."""
    with pytest.raises(SystemExit) as stop:
        _Run(tmp_path, None, _deliver_sigterm).start()
    assert stop.value.code == 143
    assert not (tmp_path / 'status').exists()


def test_a_systemerror_after_the_signal_keeps_status_25(tmp_path):
    """In a scikits.odes callback the SystemExit becomes a SystemError when the solve
    returns (bmcage/odes#99). The run still ends with status 25, not the 20 of an error."""

    def _callback_then_solver_error():
        try:
            _deliver_sigterm()
        except SystemExit as exc:
            raise SystemError('returned a result with an exception set') from exc

    with pytest.raises(SystemError):
        _Run(tmp_path, 1, _callback_then_solver_error).start()
    assert ReadStatus(tmp_path) == 25


def test_exit_if_terminated_stops_a_run_whose_first_systemexit_was_lost(tmp_path):
    """CVODE can swallow the SystemExit of a callback and report a failed step. The next
    check of the flag raises SystemExit 143 again; without a signal it does nothing."""
    checked = []

    def _lost_then_checked():
        sigterm.exit_if_terminated()
        checked.append('before')
        with contextlib.suppress(SystemExit):  # what CVODE does with it
            _deliver_sigterm()
        checked.append('after')
        sigterm.exit_if_terminated()
        checked.append('never')

    with pytest.raises(SystemExit) as stop:
        _Run(tmp_path, 1, _lost_then_checked).start()
    assert stop.value.code == 143
    assert checked == ['before', 'after']
    assert ReadStatus(tmp_path) == 25


def test_the_earlier_handler_and_a_clear_flag_are_back_after_the_run(tmp_path):
    """After a run, returned or stopped, SIGTERM goes to the handler from before it and
    the flag is clear, so a later run in the same process, as in the inference worker,
    or a solver call outside a run, does not stop on the signal of the stopped run."""

    def earlier(*_):
        return None

    signal.signal(signal.SIGTERM, earlier)
    with pytest.raises(SystemExit):
        _Run(tmp_path / 'a', 1, _deliver_sigterm).start()
    assert signal.getsignal(signal.SIGTERM) is earlier
    assert sigterm._received is None
    sigterm.exit_if_terminated()

    def _checks_flag():
        sigterm.exit_if_terminated()
        return 'done'

    assert _Run(tmp_path / 'b', 1, _checks_flag).start() == 'done'
    assert signal.getsignal(signal.SIGTERM) is earlier
    assert ReadStatus(tmp_path / 'b') == 1
