"""Unit tests for ``tools/generate_version_badges.py``.

The module under test regenerates the version badge tables in
``docs/Reference/module_versions.md`` from the pyproject pins and verifies
that every fwl-* dependency is covered by a table or excused. These tests
exercise:

* the check/write round-trip against a stale page copy,
* idempotence of the write mode,
* the missing-marker error contract (exit 2, no partial write),
* the fwl-* coverage check in both failure directions,
* one badge row pinned against the committed pyproject requirement.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Load a tools/ script by path, registering it so sibling imports resolve."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Registered for its sys.modules side effect only; the script imports it by name.
_load('_docgen')
_gvb = _load('generate_version_badges')


_STALE_PAGE = """# Module versions

Prose introduction, hand-written.

<!-- BEGIN PYPI_TABLE -->
| outdated |
<!-- END PYPI_TABLE -->

<!-- BEGIN GIT_TABLE -->
| outdated |
<!-- END GIT_TABLE -->

<!-- BEGIN OPTIONAL_TABLE -->
| outdated |
<!-- END OPTIONAL_TABLE -->
"""


def _run(monkeypatch, target: Path, *argv: str) -> int:
    monkeypatch.setattr(_gvb, 'TARGET', target)
    monkeypatch.setattr(sys, 'argv', ['generate_version_badges.py', *argv])
    return _gvb.main()


def test_check_write_round_trip_and_idempotence(tmp_path, monkeypatch):
    """A stale page fails --check, --write repairs it while keeping the
    hand-written prose, a second --write changes nothing, and --check then
    passes (the full freshness contract)."""
    target = tmp_path / 'module_versions.md'
    target.write_text(_STALE_PAGE)
    assert _run(monkeypatch, target, '--check') == 1
    assert '| outdated |' in target.read_text()  # check mode must not write

    assert _run(monkeypatch, target, '--write') == 0
    first = target.read_text()
    assert 'Prose introduction, hand-written.' in first
    assert '| outdated |' not in first
    assert _run(monkeypatch, target, '--write') == 0
    assert target.read_text() == first  # idempotent
    assert _run(monkeypatch, target, '--check') == 0


def test_missing_marker_is_structural_error(tmp_path, monkeypatch):
    """Error contract: a page missing one marker pair exits 2 and is left
    untouched, rather than silently regenerating the other tables (edge
    case: partial marker loss from a bad hand edit)."""
    page = _STALE_PAGE.replace('<!-- BEGIN GIT_TABLE -->', '').replace(
        '<!-- END GIT_TABLE -->', ''
    )
    target = tmp_path / 'module_versions.md'
    target.write_text(page)
    assert _run(monkeypatch, target, '--write') == 2
    assert target.read_text() == page


def test_fwl_coverage_check_catches_both_directions():
    """An fwl-* dependency absent from the tables and a stale
    INTENTIONALLY_ABSENT entry are each reported; the committed pyproject
    passes clean."""
    cfg = tomllib.loads((_REPO_ROOT / 'pyproject.toml').read_text())
    deps = cfg['project']['dependencies']
    extras = cfg['project'].get('optional-dependencies', {})
    assert _gvb.check_fwl_coverage(deps, extras) == []

    problems = _gvb.check_fwl_coverage(deps + ['fwl-newmodule>=1.0'], extras)
    assert any('fwl-newmodule' in p and 'PYPI_META' in p for p in problems)

    # A stale excuse (nothing in pyproject matches it) is itself flagged.
    monkey_absent = dict(_gvb.INTENTIONALLY_ABSENT)
    monkey_absent['fwl-ghost'] = 'no longer exists'
    original = _gvb.INTENTIONALLY_ABSENT
    _gvb.INTENTIONALLY_ABSENT = monkey_absent
    try:
        problems = _gvb.check_fwl_coverage(deps, extras)
    finally:
        _gvb.INTENTIONALLY_ABSENT = original
    assert any('fwl-ghost' in p and 'matches no requirement' in p for p in problems)


def test_pypi_row_pins_the_committed_requirement():
    """The rendered fwl-aragog row carries the exact version floor written in
    pyproject.toml, read independently here so a parser regression in the
    table builder cannot agree with itself."""
    cfg = tomllib.loads((_REPO_ROOT / 'pyproject.toml').read_text())
    aragog_req = next(
        r for r in cfg['project']['dependencies'] if r.strip().startswith('fwl-aragog')
    )
    floor = re.search(r'>=([0-9.]+)', aragog_req).group(1)
    table = _gvb._build_pypi_table(cfg['project']['dependencies'])
    row = next(line for line in table.split('\n') if line.startswith('| fwl-aragog'))
    assert f'%3E%3D{floor}' in row  # URL-encoded >=floor inside the badge image
    assert f'/project/fwl-aragog/{floor}/' in row  # release link targets the floor
