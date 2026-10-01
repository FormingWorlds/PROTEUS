"""
Unit tests for proteus.utils.timing, the writer of ``<output>/timing.jsonl``.

Contract clauses: nothing is written without a started run; phases are the
roots, iterations sit under ``loop`` and module calls under their iteration,
each with a parent id lower than its own; a run that raises ends with a
``run_end`` naming the failure and its open spans marked ``ok: false``.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from proteus.utils import timing

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class _Clock:
    """perf_counter stand-in that moves only when told to [s]."""

    now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(timing, 'perf_counter', fake)
    return fake


def _config(**modules):
    sections = ('atmos_clim', 'atmos_chem', 'escape', 'interior_energetics')
    sections += ('orbit', 'outgas', 'star', 'interior_struct')
    return SimpleNamespace(**{s: SimpleNamespace(module=modules.get(s)) for s in sections})


def _events(path):
    return [json.loads(line) for line in (path / timing.FILENAME).read_text().splitlines()]


def test_nothing_is_written_without_a_started_run(tmp_path, clock):
    """Without start every call is a no-op, and step still returns the duration."""
    timing.mark('loop')
    timing.backend('aragog', 'solver', 'cvode')
    clock.now = 103.5
    assert timing.step('interior', 100.0) == pytest.approx(3.5)
    timing.end('ok')
    assert not (tmp_path / timing.FILENAME).exists()


def test_module_calls_nest_under_iterations_and_phases(tmp_path, clock):
    """Each call is a child of its iteration, each iteration of loop, each phase a root.

    The orbit module is unset, so its span names no submodule.
    """
    timing.start(tmp_path, _config(interior_energetics='aragog', atmos_clim='agni'))
    timing.mark('init')
    timing.mark('loop')
    for n, calls in ((1, [('interior', 3.5), ('orbit', 0.25)]), (2, [('atmos', 15.0)])):
        timing.mark('iter', iter=n)
        for component, secs in calls:
            t0 = clock.now
            clock.now += secs
            assert timing.step(component, t0) == pytest.approx(secs)
    timing.mark('shutdown')
    timing.end('ok')

    events = _events(tmp_path)
    spans = {e['id']: e for e in events if e['ev'] == 'span'}
    tree = [
        (s['name'], spans[s['parent']]['name'] if s['parent'] else None) for s in spans.values()
    ]
    assert tree == [
        ('setup', None),
        ('init', None),
        ('interior', 'iter'),
        ('orbit', 'iter'),
        ('iter', 'loop'),
        ('atmos', 'iter'),
        ('iter', 'loop'),
        ('loop', None),
        ('shutdown', None),
    ]
    assert all(s['parent'] is None or s['parent'] < s['id'] for s in spans.values())
    by_name = {s['name']: s for s in spans.values()}
    assert (by_name['interior']['submodule'], by_name['atmos']['submodule']) == (
        'aragog',
        'agni',
    )
    assert 'submodule' not in by_name['orbit'] and by_name['orbit']['component'] == 'orbit'
    assert [s['iter'] for s in spans.values() if s['name'] == 'iter'] == [1, 2]
    assert by_name['loop']['dur'] == pytest.approx(18.75)  # 3.5 + 0.25 + 15
    assert by_name['setup']['t0'] == pytest.approx(0.0, abs=1e-9)  # run_start is t0 = 0
    assert (events[0]['ev'], events[-1]['ev'], events[-1]['status']) == (
        'run_start',
        'run_end',
        'ok',
    )


@pytest.mark.parametrize(
    ('exc', 'status', 'error'),
    [
        (RuntimeError('AGNI did not converge'), 'error', 'RuntimeError: AGNI did not converge'),
        (KeyboardInterrupt(), 'interrupted', None),
    ],
    ids=['solver_error', 'user_interrupt'],
)
def test_a_run_that_raises_ends_with_its_status_and_failed_spans(tmp_path, exc, status, error):
    """record_run re-raises, closes the open spans with ok false and names the failure."""

    @timing.record_run
    def run():
        timing.start(tmp_path, _config())
        timing.mark('loop')
        timing.mark('iter', iter=1)
        raise exc

    with pytest.raises(type(exc)):
        run()

    events = _events(tmp_path)
    assert [(e['name'], e.get('ok', True)) for e in events if e['ev'] == 'span'] == [
        ('setup', True),
        ('iter', False),
        ('loop', False),
    ]
    assert (events[-1]['status'], events[-1].get('error')) == (status, error)
    # A later call outside a run writes nothing more
    timing.mark('loop')
    assert len(_events(tmp_path)) == len(events)


def test_group_span_holds_its_calls_and_fails_with_them(tmp_path, clock):
    """A group span contains the calls made inside it and is ok false when left by an exception.

    Outside a started run the group is a plain block: no file is written.
    """
    with timing.span('equilibrate'):
        pass
    assert not (tmp_path / timing.FILENAME).exists()

    timing.start(tmp_path, _config(outgas='calliope'))
    timing.mark('init')
    with timing.span('resume'):
        pass
    with pytest.raises(RuntimeError, match='structure'):
        with timing.span('equilibrate'):
            t0 = clock.now
            clock.now += 1.0
            timing.step('outgas', t0)
            clock.now += 420.0
            raise RuntimeError('structure solve failed')
    timing.end('error')

    spans = {e['name']: e for e in _events(tmp_path) if e['ev'] == 'span'}
    assert spans['outgas']['parent'] == spans['equilibrate']['id']
    assert spans['equilibrate']['parent'] == spans['init']['id']
    assert (spans['equilibrate']['ok'], 'ok' in spans['outgas']) == (False, False)
    assert spans['equilibrate']['dur'] == pytest.approx(421.0)
    assert 'ok' not in spans['resume']  # a group left normally
    assert (
        spans['outgas']['submodule'] == 'calliope' and 'component' not in spans['equilibrate']
    )
