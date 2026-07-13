"""Unit tests for tools/cache_badges.py (documentation badge snapshotting).

Covers the network-free logic: static shields URL escaping, the SVG-versus-HTML
validation that decides whether a fetched body is written, the placeholder
fallback, the run-conclusion to badge-style mapping, the fetch/last-good/
placeholder branch in cache_badge, the main-branch run query in
_latest_conclusion, and the workflow-to-badge mapping that reports main.

Testing standards: docs/How-to/test_infrastructure.md, test_categorization.md,
test_building.md
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / 'tools'


def _load_cache_badges():
    """Import tools/cache_badges.py by path.

    cache_badges imports a sibling module (generate_test_badges), so tools/ must
    be on sys.path for that import to resolve under any invocation context.
    """
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location('cache_badges', TOOLS / 'cache_badges.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_cache_badges()


class _FakeResp:
    """Minimal context-manager stand-in for a urlopen response body."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_static_shields_url_escapes_dash_underscore_space():
    """shields escaping doubles dashes and underscores and maps spaces to '_'.

    A discriminating label carrying all three characters distinguishes the
    correct combined escaping from any single missing or mis-ordered rule.
    """
    url = cb._static_shields_url('A-B C_D', '12', 'red')
    # '-' -> '--', '_' -> '__', ' ' -> '_'
    assert url == 'https://img.shields.io/badge/A--B_C__D-12-red'
    # a plain label keeps its text; colour and message survive verbatim
    assert cb._static_shields_url('tests', '2548') == (
        'https://img.shields.io/badge/tests-2548-blue'
    )


def test_looks_like_svg_accepts_svg_rejects_html():
    """Only bodies that OPEN as SVG/XML pass; HTML error pages are rejected.

    The discriminating case is an HTML error page embedding an inline <svg>
    logo: a substring check would accept it, the prefix check must reject it,
    otherwise a broken badge gets written as a different broken badge.
    """
    assert cb._looks_like_svg(b'<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    assert cb._looks_like_svg(b'  \n  <?xml version="1.0"?><svg></svg>')
    assert not cb._looks_like_svg(b'<!DOCTYPE html><html><body>504 Gateway</body></html>')
    assert not cb._looks_like_svg(
        b'<!DOCTYPE html><body><svg class="logo"></svg></body></html>'
    )


def test_placeholder_is_valid_svg_carrying_the_label():
    """The placeholder fallback opens as an SVG and embeds the badge name."""
    svg = cb._placeholder('unit')
    assert svg.lstrip().startswith(b'<svg')
    assert b'unit' in svg
    # exactly one root element, so it is a single well-formed badge
    assert svg.count(b'<svg') == 1


def test_conclusion_style_maps_failure_classes_to_red(monkeypatch):
    """Failure-class conclusions render red; an unreachable API yields no URL.

    A startup_failure is a failed run and must not be shown as neutral grey; an
    'unknown' (API unreachable) must return None so the caller keeps last-good.
    """
    monkeypatch.setattr(cb, '_latest_conclusion', lambda *a, **k: 'startup_failure')
    url = cb._status_badge_url('ci-pr-checks.yml', 'Unit Tests', 1, 1)
    assert url is not None and url.endswith('-red')
    monkeypatch.setattr(cb, '_latest_conclusion', lambda *a, **k: 'success')
    assert cb._status_badge_url('x.yml', 'X', 1, 1).endswith('-brightgreen')
    monkeypatch.setattr(cb, '_latest_conclusion', lambda *a, **k: 'unknown')
    assert cb._status_badge_url('x.yml', 'X', 1, 1) is None


def test_cache_badge_writes_placeholder_then_keeps_last_good(tmp_path, monkeypatch):
    """Fresh dir plus failed fetch writes a placeholder; an existing file is kept.

    This pins the fallback contract: a failed refresh over a pre-existing badge
    preserves the last-good SVG rather than overwriting it with a placeholder.
    """
    monkeypatch.setattr(cb, '_fetch_svg', lambda *a, **k: (None, None))
    result = cb.cache_badge(tmp_path, 'unit', ['http://example/x.svg'], 1, 1)
    assert result == 'placeholder'
    written = (tmp_path / 'unit.svg').read_bytes()
    assert written.lstrip().startswith(b'<svg')
    # a second failed fetch must NOT clobber the existing (last-good) file
    result2 = cb.cache_badge(tmp_path, 'unit', ['http://example/x.svg'], 1, 1)
    assert result2 == 'kept-last-good'
    assert (tmp_path / 'unit.svg').read_bytes() == written


def test_cache_badge_writes_fetched_svg(tmp_path, monkeypatch):
    """A successful fetch is written verbatim and reported as ok."""
    monkeypatch.setattr(cb, '_fetch_svg', lambda *a, **k: (b'<svg>ok</svg>', 'http://src'))
    result = cb.cache_badge(tmp_path, 'docs', ['http://src'], 1, 1)
    assert result == 'ok'
    assert (tmp_path / 'docs.svg').read_bytes() == b'<svg>ok</svg>'


def test_latest_conclusion_queries_completed_main_runs(monkeypatch):
    """Resolve the latest completed run on main and map its conclusion.

    The branch=main filter is what makes the badge report main: without it a
    transient feature-branch run would be the 'latest' and freeze the badge on
    an unrelated result. status=completed drops the in-progress docs deploy that
    invokes this script, whose null conclusion would render as 'no status'. An
    empty run history yields None (no status yet, not a fabricated one); an
    unreachable API yields the 'unknown' sentinel so the caller keeps last-good.
    """
    calls = {'url': None, 'attempts': 0}
    # No real sleeps: the retry backoff would otherwise blow the unit budget.
    monkeypatch.setattr(cb.time, 'sleep', lambda _s: None)

    def make_urlopen(payload=None, raises=False):
        def fake_urlopen(req, timeout=None):
            calls['url'] = req.full_url
            calls['attempts'] += 1
            if raises:
                raise cb.urllib.error.URLError('unreachable')
            return _FakeResp(json.dumps(payload).encode())

        return fake_urlopen

    # Happy path: the newest completed run on main is a success.
    monkeypatch.setattr(
        cb.urllib.request,
        'urlopen',
        make_urlopen({'workflow_runs': [{'conclusion': 'success'}]}),
    )
    assert cb._latest_conclusion('coverage-baseline.yml', 1, 1) == 'success'
    # Both filters must be present: branch scopes to main, status drops the
    # in-progress caller run. A missing branch filter is the exact regression
    # this pins against.
    assert 'branch=main' in calls['url']
    assert 'status=completed' in calls['url']
    assert 'coverage-baseline.yml/runs' in calls['url']

    # Boundary: no runs recorded yet returns None, never a made-up conclusion.
    monkeypatch.setattr(cb.urllib.request, 'urlopen', make_urlopen({'workflow_runs': []}))
    assert cb._latest_conclusion('coverage-baseline.yml', 1, 1) is None

    # Error contract: an unreachable API returns 'unknown' after exhausting the
    # requested retries. The attempt count discriminates a no-retry regression
    # (1) and an off-by-one (3) from the correct 2.
    calls['attempts'] = 0
    monkeypatch.setattr(cb.urllib.request, 'urlopen', make_urlopen(raises=True))
    assert cb._latest_conclusion('coverage-baseline.yml', 1, 2) == 'unknown'
    assert calls['attempts'] == 2


def test_status_workflows_map_unit_to_main_coverage_baseline():
    """The unit badge resolves the workflow that runs the unit suite on main.

    coverage-baseline.yml runs the unit suite on every push to main, so its run
    conclusion reflects main's unit-test health. The pull_request-only PR-check
    workflow never runs on main, so a badge pointed at it would never satisfy
    the branch=main query and would freeze; this pins the mapping against that
    regression and confirms the other two badges are unchanged.
    """
    mapping = {label: workflow for _name, workflow, label in cb._STATUS_WORKFLOWS}
    assert mapping['Unit Tests'] == 'coverage-baseline.yml'
    assert mapping['Integration Tests'] == 'ci-nightly.yml'
    assert mapping['Docs'] == 'docs.yaml'
    # No badge may point at the pull_request-only workflow: it never runs on
    # main and so would never satisfy the branch=main query.
    workflows = {workflow for _name, workflow, _label in cb._STATUS_WORKFLOWS}
    assert 'ci-pr-checks.yml' not in workflows
    # Badge names line up in order with the SVG files the docs pages embed.
    assert [name for name, _wf, _label in cb._STATUS_WORKFLOWS] == [
        'unit',
        'integration',
        'docs',
    ]
