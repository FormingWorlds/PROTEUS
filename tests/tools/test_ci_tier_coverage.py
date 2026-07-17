"""
Every test must be selected by exactly one CI tier.

The tier filters are mutually exclusive: the unit tier selects
``unit and not slow and not integration`` while the slow tier selects
``slow and not unit and not smoke and not integration``. A test carrying two
tier markers therefore matches neither filter and silently stops running, and
the failure is invisible: the suite stays green because the test is never
collected. Two tiers whose filters do not exclude each other (unit and smoke)
have the opposite failure, running the same test twice.

Reachability is the second half of the contract. The slow tier runs a targeted
file list from its matrix rather than the whole tree, and the integration tier
runs a fixed set of directories, so a correctly marked test in a file neither
one names is just as invisible as a doubly-marked one.

These are static checks over the test tree and the workflow, so they cost no
collection time and fail at the point the marker or the workflow is edited.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

TIERS = frozenset({'unit', 'smoke', 'integration', 'slow'})
REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / 'tests'
NIGHTLY = REPO_ROOT / '.github/workflows/ci-nightly.yml'


def _marker_names(decorators: list[ast.expr]) -> set[str]:
    """Return the ``pytest.mark.<name>`` names among a decorator list."""
    found = set()
    for dec in decorators:
        node = dec.func if isinstance(dec, ast.Call) else dec
        # pytest.mark.slow  ->  Attribute(attr='slow', value=Attribute(attr='mark'))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            if node.value.attr == 'mark':
                found.add(node.attr)
    return found


def _module_marker_names(tree: ast.Module) -> set[str]:
    """Return the tier names in a module-level ``pytestmark`` assignment."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, 'id', None) == 'pytestmark' for t in node.targets):
            continue
        value = node.value
        items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        return _marker_names(items)
    return set()


def _tiers_per_test(path: Path) -> dict[str, set[str]]:
    """Map every test function in ``path`` to the tier markers it carries.

    Markers are additive across module, class, and function scope, which is
    exactly why a function-level tier on top of a module-level one produces a
    test with two tiers rather than a re-tiered test.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    module_tiers = _module_marker_names(tree) & TIERS
    out: dict[str, set[str]] = {}

    def visit(body: list[ast.stmt], inherited: set[str], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                visit(
                    node.body,
                    inherited | (_marker_names(node.decorator_list) & TIERS),
                    f'{prefix}{node.name}::',
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('test_'):
                    continue
                own = _marker_names(node.decorator_list)
                if 'skip' in own:  # deliberately disabled, never selected
                    continue
                out[f'{prefix}{node.name}'] = inherited | (own & TIERS)

    visit(tree.body, module_tiers, '')
    return out


def _test_files() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob('test_*.py') if '__pycache__' not in p.parts)


def _nightly_text() -> str:
    # Read as text rather than parsed YAML: PyYAML is not a declared dependency,
    # and the PR image installs with --no-deps, so importing it here would make
    # this guard skip on CI instead of running.
    return NIGHTLY.read_text(encoding='utf-8')


def _slow_matrix_files() -> set[str]:
    """Test files the slow-tier shards name in their ``files:`` blocks."""
    return set(re.findall(r'^\s+(tests/\S+\.py)\s*$', _nightly_text(), re.M))


def _integration_paths() -> list[str]:
    """Directories the integration tier passes to pytest."""
    m = re.search(r'pytest ((?:tests/\S+ +)+)\\', _nightly_text())
    assert m, 'no pytest invocation with test paths found in the integration tier'
    return shlex.split(m.group(1))


def test_marker_extractor_detects_a_double_tier():
    """The extractor must see additive tiers, or the sweep below passes blind.

    A module mark plus a function mark is the exact shape that produces an
    untiered test, so the guard is worthless if the extractor misses it.
    """
    sample = (
        'import pytest\n'
        'pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]\n'
        '@pytest.mark.slow\n'
        'def test_double():\n'
        '    pass\n'
        '@pytest.mark.timeout(5)\n'
        'def test_single():\n'
        '    pass\n'
        'class TestGroup:\n'
        '    @pytest.mark.smoke\n'
        '    def test_in_class(self):\n'
        '        pass\n'
    )
    tmp = Path(__file__).parent / '_tier_sample_tmp.py'
    tmp.write_text(sample, encoding='utf-8')
    try:
        tiers = _tiers_per_test(tmp)
    finally:
        tmp.unlink()

    # The doubly-marked function is the shape this guard exists to catch.
    assert tiers['test_double'] == {'unit', 'slow'}
    # A non-tier marker (timeout) must not be mistaken for a tier.
    assert tiers['test_single'] == {'unit'}
    # Markers accumulate through a class, not just a function.
    assert tiers['TestGroup::test_in_class'] == {'unit', 'smoke'}


def test_every_test_carries_exactly_one_tier():
    """No test may carry two tier markers, or none.

    Two tiers means the mutually exclusive filters both reject it; zero means
    no filter selects it. Either way the test stops running and nothing goes
    red, so this is checked statically rather than trusted.
    """
    files = _test_files()
    assert len(files) > 100, f'expected the test tree, resolved {len(files)} files'

    untiered: list[str] = []
    multi: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        for name, tiers in _tiers_per_test(path).items():
            if not tiers:
                untiered.append(f'{rel}::{name}')
            elif len(tiers) > 1:
                multi.append(f'{rel}::{name} carries {sorted(tiers)}')

    assert not multi, (
        'these tests carry more than one tier marker, so the mutually exclusive '
        'tier filters select them either twice or not at all; give each test one '
        f'tier and split the file if it spans tiers: {multi}'
    )
    assert not untiered, (
        'these tests carry no tier marker and so are invisible to every CI tier; '
        f'add a module-level pytestmark: {untiered}'
    )


def test_every_tiered_file_is_reachable_by_its_ci_job():
    """A correctly marked test still never runs if no CI job names its file.

    The slow tier runs a targeted matrix file list and the integration tier a
    fixed set of directories, so marking alone does not put a test on CI.
    """
    slow_files = _slow_matrix_files()
    integ_paths = _integration_paths()
    assert slow_files, 'no files parsed from the slow-tier matrix'
    assert integ_paths, 'no directories parsed from the integration tier'

    missing_slow: list[str] = []
    missing_integ: list[str] = []
    for path in _test_files():
        rel = str(path.relative_to(REPO_ROOT))
        tiers = set().union(*_tiers_per_test(path).values()) if _tiers_per_test(path) else set()
        if 'slow' in tiers and rel not in slow_files:
            missing_slow.append(rel)
        if 'integration' in tiers and not any(rel.startswith(f'{p}/') for p in integ_paths):
            missing_integ.append(rel)

    assert not missing_slow, (
        'these files hold slow tests but no slow-tier shard names them, so the '
        f'tests never run; add them to the matrix in {NIGHTLY.name}: {missing_slow}'
    )
    assert not missing_integ, (
        'these files hold integration tests but sit outside the directories the '
        f'integration tier runs ({integ_paths}), so the tests never run: {missing_integ}'
    )
