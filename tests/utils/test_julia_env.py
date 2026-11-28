"""Tests of the Julia thread and signal settings that PROTEUS applies before juliacall starts."""

from __future__ import annotations

import os
import signal
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip('hypothesis')
from hypothesis import example, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.utils import julia_env as je  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Julia 1.12 starts '-t 2' with 2 default threads and 1 interactive thread, '-t 1,1' with 1 and 1
_VALUES = [
    '1',
    '2',
    '1,1',
    '2,0',
    'auto',
    'auto,1',
    '1,auto',
    '0',
    '-1',
    '3,-1',
    'x',
    '',
    ' 2 ',
]
_MANY = {'2', '1,1', '2,0', 'auto', 'auto,1', '1,auto', ' 2 '}


@contextmanager
def _before_juliacall():
    """The state of a process that has not imported juliacall yet."""
    with patch.dict(sys.modules):
        sys.modules.pop('juliacall', None)
        yield


@pytest.mark.parametrize('value', _VALUES)
def test_julia_num_threads_above_one_sets_one_thread(value):
    """JULIA_NUM_THREADS with more than 1 thread (also interactive or auto) sets
    PYTHON_JULIACALL_THREADS to 1 and returns the log line; 1 thread and a value Julia
    refuses leave the environment as it is."""
    env = {'JULIA_NUM_THREADS': value}
    with _before_juliacall(), patch.object(je, 'JULIA_THREADS_NOTE', ''):
        je.set_julia_threads(env, {})
        note = je.JULIA_THREADS_NOTE
    many = value in _MANY
    assert env == (
        {'JULIA_NUM_THREADS': value, 'PYTHON_JULIACALL_THREADS': '1'}
        if many
        else {'JULIA_NUM_THREADS': value}
    )
    assert (f'JULIA_NUM_THREADS={value} not used' in note) is many


_SETTING = st.one_of(st.none(), st.sampled_from(_VALUES), st.text(max_size=5))


@example(julia='4', user='1', signals=None, xthreads=None)
@example(julia='2', user=' 1 ', signals=None, xthreads=None)
@given(
    julia=_SETTING,
    user=_SETTING,
    signals=st.sampled_from([None, 'yes', 'no']),
    xthreads=st.one_of(st.none(), st.sampled_from(['1', '2', 'auto']), st.just(True)),
)
def test_a_user_value_is_never_changed(julia, user, signals, xthreads):
    """Only an unset key is written, so every value the user set wins. A user thread
    count (-X juliacall-threads before PYTHON_JULIACALL_THREADS, as juliacall reads them)
    above 1 sets the signal handlers on; without one, a JULIA_NUM_THREADS above 1 sets 1
    thread and never the signal handlers. The note is logged whenever 1 thread runs while
    JULIA_NUM_THREADS asks for more."""
    keys = ('JULIA_NUM_THREADS', 'PYTHON_JULIACALL_THREADS', 'PYTHON_JULIACALL_HANDLE_SIGNALS')
    env = {k: v for k, v in zip(keys, (julia, user, signals)) if v is not None}
    before = dict(env)
    xoptions = {} if xthreads is None else {'juliacall-threads': xthreads}
    with _before_juliacall(), patch.object(je, 'JULIA_THREADS_NOTE', ''):
        je.set_julia_threads(env, xoptions)
        note = je.JULIA_THREADS_NOTE
    changed = {k for k in env if env.get(k) != before.get(k)}
    assert not changed & set(before)
    asked = xthreads if xthreads is not None else user
    if asked is None:
        many = julia is not None and je._more_than_one(julia)
        assert changed == ({'PYTHON_JULIACALL_THREADS'} if many else set())
        assert not many or env['PYTHON_JULIACALL_THREADS'] == '1'
        assert bool(note) is many
    else:
        sets_signals = signals is None and je._more_than_one(asked)
        assert changed == ({'PYTHON_JULIACALL_HANDLE_SIGNALS'} if sets_signals else set())
        assert env.get('PYTHON_JULIACALL_HANDLE_SIGNALS') == (
            'yes' if sets_signals else signals
        )
        one = str(asked).strip() == '1'
        assert bool(note) is (one and julia is not None and je._more_than_one(julia))


@pytest.mark.parametrize('key', ['JULIA_NUM_THREADS', 'PYTHON_JULIACALL_THREADS'])
def test_no_change_once_juliacall_has_started(monkeypatch, key):
    """After juliacall has started Julia a new setting has no effect, so the environment
    and the log note stay as they are; a note would claim 1 thread while Julia runs more."""
    monkeypatch.setitem(sys.modules, 'juliacall', MagicMock())
    env = {key: '2'}
    with patch.object(je, 'JULIA_THREADS_NOTE', ''):
        je.set_julia_threads(env, {})
        assert je.JULIA_THREADS_NOTE == ''
    assert env == {key: '2'}


def test_an_inherited_single_thread_logs_the_note():
    """A child process (a grid case, the --deterministic re-exec) inherits
    PYTHON_JULIACALL_THREADS=1 from its parent and logs the same note."""
    parent = {'JULIA_NUM_THREADS': '4'}
    with _before_juliacall(), patch.object(je, 'JULIA_THREADS_NOTE', ''):
        je.set_julia_threads(parent, {})
        note = je.JULIA_THREADS_NOTE
        je.JULIA_THREADS_NOTE = ''
        child = dict(parent)
        je.set_julia_threads(child, {})
        assert je.JULIA_THREADS_NOTE == note != ''
    assert child == parent


def test_status_hook_is_not_set_when_python_handles_the_signals(monkeypatch, tmp_path):
    """Without Julia's signal handlers, Python records status 25 itself, so no Julia exit
    hook is set and no Julia code runs."""
    import juliacall

    main = MagicMock()
    monkeypatch.setattr(juliacall, 'Main', main)
    monkeypatch.setattr(je, '_status_hook', None)
    monkeypatch.setattr(je, 'julia_handles_signals', lambda: False)
    assert je.register_status_hook(str(tmp_path)) is False
    main.seval.assert_not_called()
    assert je._status_hook is None


def test_status_hook_is_set_once_and_follows_the_last_run(monkeypatch, tmp_path):
    """With Julia's signal handlers on, the first call sets the Julia exit hook; every call
    points it to the status file of that run, with the text of status 25."""
    import juliacall

    main = MagicMock()
    monkeypatch.setattr(juliacall, 'Main', main)
    monkeypatch.setattr(je, '_status_hook', None)
    monkeypatch.setattr(je, 'julia_handles_signals', lambda: True)
    assert all(je.register_status_hook(str(tmp_path / d)) for d in ('a', 'b'))
    main.seval.assert_called_once()
    text = '25\nError (died, or exit requested by user)\n'
    assert main.seval.return_value.call_args_list == [
        call(str(tmp_path / d / 'status'), text) for d in ('a', 'b')
    ]


@pytest.mark.parametrize('julia', [False, True])
def test_python_sigterm_handler_only_without_the_julia_handlers(monkeypatch, julia):
    """Python takes SIGTERM only when Julia does not: its handler would replace Julia's and
    stop the Julia exit hook. The Python handler records the stop and stays installed, and
    SIGTERM blocked from the parent process is unblocked so that the handler runs."""
    import juliacall

    monkeypatch.setitem(juliacall.CONFIG, 'opt_handle_signals', 'yes' if julia else None)
    saved = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        assert je.exit_on_sigterm() is not julia
        handler = signal.getsignal(signal.SIGTERM)
        assert (handler is signal.SIG_DFL) is julia
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert (signal.SIGTERM in blocked) is julia
        if not julia:
            with pytest.raises(SystemExit) as stop:
                handler(signal.SIGTERM, None)
            assert stop.value.code == 143
            assert je.STOP_SIGNAL == signal.SIGTERM
            assert signal.getsignal(signal.SIGTERM) is handler
            with pytest.raises(KeyboardInterrupt):
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            assert je.STOP_SIGNAL == signal.SIGINT
    finally:
        signal.signal(signal.SIGTERM, saved[0])
        signal.signal(signal.SIGINT, saved[1])
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


@pytest.mark.usefixtures('signals_unblocked')
@pytest.mark.parametrize('signame', ['SIGTERM', 'SIGINT'])
def test_a_stop_in_a_held_status_write_is_recorded_not_raised(monkeypatch, tmp_path, signame):
    """A stop signal while a status write is held records the signal and lets the write
    finish; after the hold, also one that a raise ended, the handler writes 25 and raises."""
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    signum = getattr(signal, signame)
    signal.signal(
        signum, {signal.SIGTERM: je._exit_on_sigterm, signal.SIGINT: je._interrupt}[signum]
    )
    monkeypatch.setattr(je, '_status_output', str(tmp_path))
    UpdateStatusfile({'output': str(tmp_path)}, 1)
    ends = []
    with pytest.raises(ValueError):
        with je.sigterm_held():
            try:
                signal.raise_signal(signum)
                ends.append('returned')
            except (SystemExit, KeyboardInterrupt):
                ends.append('raised')
            ends.append((je.STOP_SIGNAL, ReadStatus(tmp_path)))
            raise ValueError
    assert ends == ['returned', (signum, 1)]
    with pytest.raises(SystemExit if signum == signal.SIGTERM else KeyboardInterrupt):
        signal.raise_signal(signum)
    assert ReadStatus(tmp_path) == 25


def test_a_status_write_over_files_left_by_a_killed_write(tmp_path):
    """Temporary files that a killed write left next to the status file do not stop the
    next write, and the status reads as written."""
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    UpdateStatusfile({'output': str(tmp_path)}, 1)
    for name in ('status.tmp', f'status.{os.getpid()}.0.tmp', f'status.{os.getpid()}.1.tmp'):
        (tmp_path / name).write_text('9\n')
    UpdateStatusfile({'output': str(tmp_path)}, 10)
    assert ReadStatus(tmp_path) == 10


@pytest.mark.usefixtures('signals_unblocked')
@pytest.mark.parametrize('signame', ['SIGTERM', 'SIGINT'])
@pytest.mark.parametrize('before', [0, 1])
def test_a_stop_inside_a_status_write_leaves_status_25(monkeypatch, tmp_path, before, signame):
    """A stop that comes while the status file is rewritten finds the old status on disk,
    writes 25 over it, and leaves no other file in the run folder."""
    import proteus.utils.helper as helper
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    signum = getattr(signal, signame)
    signal.signal(
        signum, {signal.SIGTERM: je._exit_on_sigterm, signal.SIGINT: je._interrupt}[signum]
    )
    UpdateStatusfile({'output': str(tmp_path)}, before)
    monkeypatch.setattr(je, '_status_output', str(tmp_path))
    real, calls = helper.CommentFromStatus, []

    def stop_in_write(status):
        calls.append(status)
        if len(calls) == 1:
            signal.raise_signal(signum)
        return real(status)

    monkeypatch.setattr(helper, 'CommentFromStatus', stop_in_write)
    with pytest.raises(SystemExit if signum == signal.SIGTERM else KeyboardInterrupt):
        UpdateStatusfile({'output': str(tmp_path)}, 13)
    assert calls == [13, 25]
    assert ReadStatus(tmp_path) == 25
    assert sorted(os.listdir(tmp_path)) == ['status']


def test_a_stop_is_read_from_the_handler_or_the_exception_chain(monkeypatch):
    """An error is a stop when the SIGTERM handler ran (a solver dropped its SystemExit) or
    when a KeyboardInterrupt or SystemExit is in its cause or context chain."""

    def chained(outer, inner, link='__context__'):
        setattr(outer, link, inner)
        return outer

    assert je.run_was_stopped(KeyboardInterrupt())
    assert je.run_was_stopped(chained(SystemError('callback'), SystemExit(143)))
    assert je.run_was_stopped(chained(RuntimeError('x'), KeyboardInterrupt(), '__cause__'))
    both = chained(
        chained(RuntimeError('y'), ValueError('cause'), '__cause__'), SystemExit(143)
    )
    assert je.run_was_stopped(both)
    assert not je.run_was_stopped(chained(RuntimeError('b'), ValueError('a')))
    loop = chained(ValueError('a'), ValueError('b'), '__cause__')
    loop.__cause__.__cause__ = loop
    assert not je.run_was_stopped(loop)
    monkeypatch.setattr(je, 'STOP_SIGNAL', signal.SIGTERM)
    assert je.run_was_stopped(RuntimeError('solver failed'))


def test_system_configuration_logs_the_thread_note():
    """The line of `set_julia_threads` goes into the run log with the system configuration,
    and no line is logged when the setting was left as it is."""
    from proteus.utils.coupler import print_system_configuration

    for note in ('Julia threads     1 (JULIA_NUM_THREADS=4 not used; ...)', ''):
        with (
            patch.object(je, 'JULIA_THREADS_NOTE', note),
            patch('proteus.utils.coupler.log') as log,
            patch('proteus.utils.coupler._get_julia_version', return_value='1.12'),
        ):
            print_system_configuration({'fwl': '/fwl'})
        lines = [c.args[0] for c in log.info.call_args_list]
        assert (note in lines) is bool(note)
        assert len(lines) == 8 + bool(note)


@pytest.mark.parametrize('stop', [None, signal.SIGTERM, signal.SIGINT])
def test_exit_if_stopped_exits_with_128_plus_the_recorded_signal(monkeypatch, stop):
    """No stop recorded: the call returns. A recorded SIGTERM or SIGINT ends the process with
    143 or 130, the code a shell reports for that signal."""
    monkeypatch.setattr(je, 'STOP_SIGNAL', stop)
    if stop is None:
        assert je.exit_if_stopped() is None
        return
    with pytest.raises(SystemExit) as end:
        je.exit_if_stopped()
    assert end.value.code == {signal.SIGTERM: 143, signal.SIGINT: 130}[stop]


@pytest.mark.parametrize('exc', [KeyboardInterrupt(), SystemExit(2)])
def test_the_status_guard_reraises_an_interrupt_or_exit_unchanged(monkeypatch, tmp_path, exc):
    """A KeyboardInterrupt or SystemExit leaves `Proteus.start` as raised, also while a stop
    is recorded: only an error that hides a dropped stop becomes the exit of that signal."""
    from proteus.proteus import _status_on_exit
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    class Stub:
        directories = {'output': str(tmp_path)}

        @_status_on_exit
        def start(self):
            UpdateStatusfile(self.directories, 1)
            raise exc

    monkeypatch.setattr(je, 'julia_handles_signals', lambda: False)
    monkeypatch.setattr(je, 'STOP_SIGNAL', signal.SIGTERM)
    with pytest.raises(BaseException) as end:
        Stub().start()
    assert end.value is exc
    assert ReadStatus(tmp_path) == 25


@pytest.mark.parametrize('signame', ['SIGTERM', 'SIGINT'])
@pytest.mark.parametrize('status', [0, 1, 10, 12, 20, 22, 25, None])
def test_a_stop_writes_status_25_only_over_a_started_or_running_run(
    monkeypatch, tmp_path, status, signame
):
    """The stop handlers write 25 when the run reads as started (0) or running (1), keep a
    final or error status and a missing file, and write nothing without a run folder."""
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    signum = getattr(signal, signame)
    handler = {signal.SIGTERM: je._exit_on_sigterm, signal.SIGINT: je._interrupt}[signum]
    run = tmp_path / 'run'
    if status is not None:
        UpdateStatusfile({'output': str(run)}, status)
    for folder in (None, str(run)):
        monkeypatch.setattr(je, '_status_output', folder)
        monkeypatch.setattr(je, 'STOP_SIGNAL', None)
        with pytest.raises(
            SystemExit if signum == signal.SIGTERM else KeyboardInterrupt
        ) as end:
            handler(signum, None)
        assert je.STOP_SIGNAL == signum
        if signum == signal.SIGTERM:
            assert end.value.code == 128 + signum
        if folder is None:
            assert ReadStatus(run) == (-1 if status is None else status)
    assert ReadStatus(run) == (25 if status in (0, 1) else -1 if status is None else status)


def test_a_status_written_after_a_stop_records_the_stop(monkeypatch, tmp_path):
    """Once a stop is recorded, every status write records 25, so a termination check or an
    error site later in the same step cannot hide the stop."""
    from proteus.utils.helper import ReadStatus, UpdateStatusfile

    UpdateStatusfile({'output': str(tmp_path)}, 13)
    assert ReadStatus(tmp_path) == 13
    monkeypatch.setattr(je, 'STOP_SIGNAL', signal.SIGTERM)
    for code in (1, 13, 21):
        UpdateStatusfile({'output': str(tmp_path)}, code)
        assert ReadStatus(tmp_path) == 25
