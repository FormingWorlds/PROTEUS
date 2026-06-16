"""Unit tests for tools/cache_badges.py (documentation badge snapshotting).

Covers the network-free logic: static shields URL escaping, the SVG-versus-HTML
validation that decides whether a fetched body is written, the placeholder
fallback, the run-conclusion to badge-style mapping, and the
fetch/last-good/placeholder branch in cache_badge.

Testing standards: docs/How-to/test_infrastructure.md, test_categorization.md,
test_building.md
"""

from __future__ import annotations

import importlib.util
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
