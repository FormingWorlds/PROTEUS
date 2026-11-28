"""Tests of the Julia thread and signal settings that PROTEUS applies before juliacall starts."""

from __future__ import annotations

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


@pytest.mark.parametrize('ours', [True, False])
def test_sigterm_is_held_only_over_its_own_handler(ours):
    """While a stop is recorded, SIGTERM is ignored and then the Python handler is back;
    any other handler (Julia's, the default) is left alone."""
    saved = signal.getsignal(signal.SIGTERM)
    before = je._exit_on_sigterm if ours else signal.SIG_DFL
    try:
        signal.signal(signal.SIGTERM, before)
        with je.sigterm_held():
            inside = signal.getsignal(signal.SIGTERM)
        assert inside is (signal.SIG_IGN if ours else before)
        assert signal.getsignal(signal.SIGTERM) is before
    finally:
        signal.signal(signal.SIGTERM, saved)


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
