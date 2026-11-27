"""Tests of the Julia thread and signal settings that PROTEUS applies before juliacall starts."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from proteus.utils import julia_env as je

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


@pytest.mark.parametrize('value', _VALUES)
def test_julia_num_threads_above_one_sets_one_thread(value):
    """JULIA_NUM_THREADS with more than 1 thread (also interactive or auto) sets
    PYTHON_JULIACALL_THREADS to 1 and returns the log line; 1 thread and a value Julia
    refuses leave the environment as it is."""
    env = {'JULIA_NUM_THREADS': value}
    with patch.object(je, 'JULIA_THREADS_NOTE', ''):
        note = je.set_julia_threads(env, {})
        assert je.JULIA_THREADS_NOTE == note
    many = value in _MANY
    assert env == (
        {'JULIA_NUM_THREADS': value, 'PYTHON_JULIACALL_THREADS': '1'}
        if many
        else {'JULIA_NUM_THREADS': value}
    )
    assert (f'JULIA_NUM_THREADS={value} not used' in note) is many


_SETTING = st.one_of(st.none(), st.sampled_from(_VALUES), st.text(max_size=5))


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
    thread and never the signal handlers."""
    keys = ('JULIA_NUM_THREADS', 'PYTHON_JULIACALL_THREADS', 'PYTHON_JULIACALL_HANDLE_SIGNALS')
    env = {k: v for k, v in zip(keys, (julia, user, signals)) if v is not None}
    before = dict(env)
    xoptions = {} if xthreads is None else {'juliacall-threads': xthreads}
    with patch.object(je, 'JULIA_THREADS_NOTE', ''):
        note = je.set_julia_threads(env, xoptions)
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
        assert note == ''


def test_status_hook_is_not_set_when_python_handles_the_signals(monkeypatch, tmp_path):
    """Without Julia's signal handlers, Python records status 25 itself, so no Julia exit
    hook is set and no Julia code runs."""
    import juliacall

    monkeypatch.setitem(juliacall.CONFIG, 'opt_handle_signals', None)
    monkeypatch.setattr(je, '_status_hook', None)
    with patch.object(juliacall.Main, 'seval') as seval:
        assert je.register_status_hook(str(tmp_path), '25\nstopped\n') is False
    seval.assert_not_called()
    assert je._status_hook is None


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
