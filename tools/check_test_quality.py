#!/usr/bin/env python3
"""AST-based test-quality linter for the PROTEUS test suite.

Enforces the rules in `.github/.claude/rules/proteus-tests.md` (sections 1 + 7):

- Every test file must declare a module-level ``pytestmark`` containing a tier
  marker (``unit`` / ``smoke`` / ``integration`` / ``slow``).
- Test functions must contain at least 2 assertion statements OR a discriminating
  property-based assertion. Single-assert tests are a known weak pattern.
- Forbidden standalone weak assertions: ``result is not None``, ``result > 0``,
  ``len(result) > 0``, ``isinstance(result, dict)``, ``result is None``.
- Every test function must have a docstring.
- ``==`` adjacent to a numeric literal in a test body is a likely float-comparison
  bug (use ``pytest.approx`` instead).

Two modes:

* ``--baseline``  Walk the test suite, write per-rule violation counts to
  ``tools/test_quality_baseline.json``. Run this only after a deliberate sweep
  that has reduced violations; commits should not raise the baseline.
* ``--check``     CI mode. Walk the suite, compare current violation counts to
  the baseline. Exit non-zero if any rule's violation count exceeds the
  baseline. Print the offending files + functions.

Optionally:

* ``--reference-pinned-audit``  Print the physics modules that lack at least
  one ``@pytest.mark.reference_pinned`` test. Does not exit non-zero on its
  own (advisory).
* ``--physics-invariant-audit``  Print physics-module tests that assert no
  invariant and are not tagged ``@pytest.mark.physics_invariant``. Advisory.

All exits in ``--check`` mode use exit code 1 on regression; 0 otherwise.

The script reads no configuration outside its own constants and the baseline
file. It is intentionally dependency-free (pure stdlib) so it can run in any
CI environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / 'tests'
BASELINE_PATH = REPO_ROOT / 'tools' / 'test_quality_baseline.json'

TIER_MARKERS = {'unit', 'smoke', 'integration', 'slow'}

# Test directories that contain at least one physics-required source file.
# Each directory must contain at least one @pytest.mark.reference_pinned test.
# The per-source-file carve-out (e.g. inference/BO.py is physics, inference/
# utils.py is utility) is specified in .github/.claude/rules/proteus-tests.md
# section 3; the audits here operate at directory granularity, which means
# inference/ is included because BO.py / async_BO.py / objective.py live there.
PHYSICS_MODULES = {
    'interior_struct',
    'interior_energetics',
    'atmos_clim',
    'atmos_chem',
    'escape',
    'outgas',
    'orbit',
    'star',
    'observe',
    'inference',
}

# Weak-assertion shapes flagged as standalone violations.
# Each entry is a callable on an ast.Assert node returning True if it matches.


def _is_weak_assert(node: ast.Assert) -> str | None:
    """Return a label if ``node`` is a forbidden weak standalone assertion."""
    test = node.test
    # `assert x is None` / `assert x is not None`
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        right = test.comparators[0]
        if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and right.value is None:
            return 'is_none_or_not_none'
        # `assert x > 0`
        if isinstance(op, ast.Gt) and isinstance(right, ast.Constant) and right.value == 0:
            return 'gt_zero'
        # `assert len(x) > 0`
        if (
            isinstance(op, ast.Gt)
            and isinstance(test.left, ast.Call)
            and isinstance(test.left.func, ast.Name)
            and test.left.func.id == 'len'
            and isinstance(right, ast.Constant)
            and right.value == 0
        ):
            return 'len_gt_zero'
    # `assert isinstance(x, T)` as the *only* assertion in the test is flagged
    # at the function level (see check_function), not here.
    return None


def _is_isinstance_assert(node: ast.Assert) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == 'isinstance'
    )


def _is_numpy_testing(node: ast.AST) -> bool:
    """True if ``node`` represents the ``numpy.testing`` or ``np.testing`` module
    (i.e. the value side of ``numpy.testing.assert_allclose``).

    Matches both the ``import numpy as np`` short form (``np.testing.X``) and
    the ``import numpy`` long form (``numpy.testing.X``).
    """
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr != 'testing':
        return False
    if isinstance(node.value, ast.Name) and node.value.id in ('np', 'numpy'):
        return True
    return False


def _is_exact_zero(value) -> bool:
    """True for the sentinel ``0.0`` / ``-0.0`` float comparand.

    Asserting an exact-zero result (e.g. radioactive heating disabled, escape
    rate at the no-atmosphere limit, eccentricity damping at ``e=0``) is a
    legitimate physics check and does not need ``pytest.approx``: there is no
    rounding error to absorb. Comparing against any other float literal is
    flagged.
    """
    return isinstance(value, float) and value == 0.0


def _has_float_eq(node: ast.AST) -> bool:
    """Return True if any descendant uses ``==`` against a non-zero float literal.

    Exact-zero comparisons are exempt; see :func:`_is_exact_zero`.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            for op, right in zip(child.ops, child.comparators):
                if isinstance(op, ast.Eq):
                    if isinstance(right, ast.Constant) and isinstance(right.value, float):
                        if not _is_exact_zero(right.value):
                            return True
                    if isinstance(child.left, ast.Constant) and isinstance(child.left.value, float):
                        if not _is_exact_zero(child.left.value):
                            return True
    return False


def _module_pytestmark_tier(tree: ast.Module) -> str | None:
    """Return the tier marker declared in a module-level ``pytestmark``, or None."""
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not (len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        if stmt.targets[0].id != 'pytestmark':
            continue
        # pytestmark = pytest.mark.<tier>  OR  pytestmark = [pytest.mark.<tier>, ...]
        marks = stmt.value
        nodes = marks.elts if isinstance(marks, (ast.List, ast.Tuple)) else [marks]
        for n in nodes:
            tier = _tier_of_mark_node(n)
            if tier is not None:
                return tier
    return None


def _tier_of_mark_node(n: ast.AST) -> str | None:
    """Given a ``pytest.mark.<x>`` or ``pytest.mark.<x>(...)`` node, return the tier name if it matches."""
    if isinstance(n, ast.Call):
        n = n.func
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute):
        if (
            isinstance(n.value.value, ast.Name)
            and n.value.value.id == 'pytest'
            and n.value.attr == 'mark'
            and n.attr in TIER_MARKERS
        ):
            return n.attr
    return None


def _func_markers(fn: ast.FunctionDef) -> set[str]:
    """All ``pytest.mark.<x>`` markers on a function definition."""
    out: set[str] = set()
    for dec in fn.decorator_list:
        n = dec
        if isinstance(n, ast.Call):
            n = n.func
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute):
            if (
                isinstance(n.value.value, ast.Name)
                and n.value.value.id == 'pytest'
                and n.value.attr == 'mark'
            ):
                out.add(n.attr)
    return out


def _docstring_of(fn: ast.FunctionDef) -> str | None:
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        v = fn.body[0].value.value
        if isinstance(v, str):
            return v
    return None


class Violations:
    """Aggregate counts and per-violation details for one rule scan."""

    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)
        self.details: dict[str, list[str]] = defaultdict(list)

    def add(self, rule: str, where: str) -> None:
        self.counts[rule] += 1
        self.details[rule].append(where)

    def to_baseline(self) -> dict[str, int]:
        return dict(self.counts)


def _count_implicit_assertions(node: ast.AST) -> int:
    """Count assertion-equivalents that are not bare ``assert`` statements.

    Recognized patterns:

    - ``with pytest.raises(...)`` blocks (each block counts as one assertion).
    - ``with pytest.warns(...)`` blocks.
    - ``mock.assert_called_with(...)`` / ``assert_called_once_with(...)`` /
      ``assert_not_called(...)`` etc. method calls on a Mock object.
    - ``pytest.fail(...)`` calls (used in conditional fail-mode tests).
    - ``np.testing.assert_*`` family.
    """
    count = 0
    for child in ast.walk(node):
        # `with pytest.raises(...)` / `with pytest.warns(...)`
        if isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
                    if (
                        isinstance(ctx.func.value, ast.Name)
                        and ctx.func.value.id == 'pytest'
                        and ctx.func.attr in ('raises', 'warns', 'deprecated_call')
                    ):
                        count += 1
        # Method calls: x.assert_called*, x.assert_not_called(),
        # np.testing.assert_*, numpy.testing.assert_*.
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            attr = child.func.attr
            if attr.startswith('assert_called') or attr == 'assert_not_called':
                count += 1
            elif attr.startswith('assert_') and _is_numpy_testing(child.func.value):
                # np.testing.assert_allclose, numpy.testing.assert_array_equal,
                # etc. Recognized for both `import numpy as np` and the bare
                # `import numpy` form.
                count += 1
            elif isinstance(child.func.value, ast.Name) and child.func.value.id == 'pytest':
                if attr == 'fail':
                    count += 1
    return count


def check_file(path: Path) -> Violations:
    v = Violations()
    try:
        rel = str(path.relative_to(REPO_ROOT))
    except ValueError:
        # Tests directory or individual test file lives behind a symlink that
        # points outside the repo root. Fall back to the absolute path so the
        # walk completes; this only affects display, not correctness.
        rel = str(path)
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        v.add('parse_error', f'{rel}: {e}')
        return v

    module_tier = _module_pytestmark_tier(tree)
    if module_tier is None:
        v.add('missing_module_pytestmark', rel)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith('test_'):
            continue
        where = f'{rel}::{node.name}'

        # Docstring required.
        if _docstring_of(node) is None:
            v.add('missing_docstring', where)

        # Float == literal.
        if _has_float_eq(node):
            v.add('float_eq_literal', where)

        # Count assertions in body, including pytest.raises / mock asserts / np.testing.assert_*.
        asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
        n_assert = len(asserts) + _count_implicit_assertions(node)
        if n_assert == 0:
            v.add('no_assertions', where)
        elif n_assert == 1:
            v.add('single_assert', where)

        # Weak-assertion shapes (flag every occurrence, not just sole-assertion).
        for a in asserts:
            label = _is_weak_assert(a)
            if label is not None:
                v.add(f'weak_assert_{label}', where)
        # Sole `assert isinstance(...)` as the only assert is its own violation.
        if n_assert == 1 and len(asserts) == 1 and _is_isinstance_assert(asserts[0]):
            v.add('weak_assert_only_isinstance', where)

    return v


def walk_tests() -> Violations:
    total = Violations()
    for p in sorted(TESTS_DIR.rglob('test_*.py')):
        if '__pycache__' in p.parts:
            continue
        v = check_file(p)
        for rule, n in v.counts.items():
            total.counts[rule] += n
        for rule, details in v.details.items():
            total.details[rule].extend(details)
    return total


def _file_has_decorator(path: Path, decorator_name: str) -> bool:
    """True if any function or class in ``path`` carries ``@pytest.mark.<decorator_name>``.

    AST scan; comments, docstrings, and import strings do not count.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            d = dec
            if isinstance(d, ast.Call):
                d = d.func
            if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Attribute):
                if (
                    isinstance(d.value.value, ast.Name)
                    and d.value.value.id == 'pytest'
                    and d.value.attr == 'mark'
                    and d.attr == decorator_name
                ):
                    return True
    return False


def reference_pinned_audit() -> list[str]:
    """Return module-relative paths of physics modules lacking a reference_pinned test."""
    missing = []
    for module in sorted(PHYSICS_MODULES):
        mod_dir = TESTS_DIR / module
        if not mod_dir.exists():
            continue
        has_ref = False
        for p in mod_dir.rglob('test_*.py'):
            if _file_has_decorator(p, 'reference_pinned'):
                has_ref = True
                break
        if not has_ref:
            missing.append(module)
    return missing


def physics_invariant_audit() -> list[str]:
    """Return physics-module tests that have neither an explicit physics_invariant
    marker nor any property-based assertion language in their body.

    The heuristic is intentionally simple: a physics-module test must either
    carry the marker OR contain at least one of the keywords
    {'approx', 'assert_allclose', 'monoton', 'conserve', 'symmetric'}
    in its body. Otherwise it is flagged for manual review. This is advisory,
    not blocking.
    """
    flagged = []
    keywords = {'approx', 'assert_allclose', 'monoton', 'conserve', 'symmetric', 'positive'}
    for module in sorted(PHYSICS_MODULES):
        mod_dir = TESTS_DIR / module
        if not mod_dir.exists():
            continue
        for p in sorted(mod_dir.rglob('test_*.py')):
            try:
                tree = ast.parse(p.read_text())
            except SyntaxError:
                continue
            rel = str(p.relative_to(REPO_ROOT))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not node.name.startswith('test_'):
                    continue
                markers = _func_markers(node)
                if 'physics_invariant' in markers:
                    continue
                body_src = ast.unparse(node)
                if any(kw in body_src for kw in keywords):
                    continue
                flagged.append(f'{rel}::{node.name}')
    return flagged


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text())


def cmd_baseline() -> int:
    v = walk_tests()
    # Guard against accidental regeneration that would raise the baseline and
    # mask a regression. If the new total exceeds the old, refuse unless the
    # caller explicitly opts in via PROTEUS_TEST_QUALITY_ALLOW_REGRESS=1.
    old = load_baseline()
    old_total = sum(old.values())
    new_total = sum(v.counts.values())
    import os
    allow = os.environ.get('PROTEUS_TEST_QUALITY_ALLOW_REGRESS') == '1'
    if old and new_total > old_total and not allow:
        print(
            f'Refusing to regenerate baseline: new total ({new_total}) exceeds '
            f'old total ({old_total}). The baseline should only ratchet downward.\n'
            f'If this is intentional (e.g. a new rule was added that surfaces '
            f'pre-existing violations), set PROTEUS_TEST_QUALITY_ALLOW_REGRESS=1.',
            file=sys.stderr,
        )
        return 2
    BASELINE_PATH.write_text(json.dumps(v.to_baseline(), indent=2, sort_keys=True) + '\n')
    print(f'Wrote baseline: {BASELINE_PATH.relative_to(REPO_ROOT)}')
    for rule in sorted(v.counts):
        print(f'  {rule}: {v.counts[rule]}')
    if old:
        delta = new_total - old_total
        print(f'  total: {new_total} ({delta:+d} vs previous baseline {old_total})')
    return 0


def cmd_check() -> int:
    baseline = load_baseline()
    v = walk_tests()
    failed = False
    print('Rule                                       Baseline   Current   Status')
    print('-' * 76)
    all_rules = sorted(set(baseline) | set(v.counts))
    for rule in all_rules:
        b = baseline.get(rule, 0)
        c = v.counts.get(rule, 0)
        status = 'OK' if c <= b else 'REGRESSION'
        if c > b:
            failed = True
        print(f'{rule:42} {b:>8} {c:>9}   {status}')
    # Cross-rule total-violation guard. A refactor that adds 5 single-asserts
    # and removes 5 missing-docstrings would pass the per-rule check while
    # degrading the suite. Catch that here.
    total_baseline = sum(baseline.values())
    total_current = sum(v.counts.values())
    print('-' * 76)
    total_status = 'OK' if total_current <= total_baseline else 'REGRESSION'
    print(f'{"TOTAL":42} {total_baseline:>8} {total_current:>9}   {total_status}')
    if total_current > total_baseline:
        failed = True
    if failed:
        print()
        print('New violations vs baseline:')
        for rule in all_rules:
            b = baseline.get(rule, 0)
            c = v.counts.get(rule, 0)
            if c > b:
                # Print the first 5 offending locations for context.
                offenders = v.details.get(rule, [])
                print(f'\n  {rule} (+{c - b}):')
                for offender in offenders[:5]:
                    print(f'    {offender}')
                if len(offenders) > 5:
                    print(f'    ... and {len(offenders) - 5} more')
        print()
        print('Reduce violations or, after a deliberate sweep, regenerate the baseline:')
        print('  python tools/check_test_quality.py --baseline')
        return 1
    return 0


def cmd_reference_pinned_audit() -> int:
    missing = reference_pinned_audit()
    if not missing:
        print('All physics modules contain at least one @pytest.mark.reference_pinned test.')
        return 0
    print('Physics modules missing a @pytest.mark.reference_pinned test:')
    for m in missing:
        print(f'  tests/{m}/')
    return 0


def cmd_physics_invariant_audit() -> int:
    flagged = physics_invariant_audit()
    if not flagged:
        print('All physics-module tests either carry @pytest.mark.physics_invariant or use property-based language.')
        return 0
    print('Physics-module tests without @pytest.mark.physics_invariant and no property-based assertion language:')
    for f in flagged:
        print(f'  {f}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--baseline',
        action='store_true',
        help='Regenerate tools/test_quality_baseline.json from current state.',
    )
    group.add_argument(
        '--check',
        action='store_true',
        help='CI mode: fail if violations exceed baseline.',
    )
    group.add_argument(
        '--reference-pinned-audit',
        action='store_true',
        help='Advisory: list physics modules missing a @pytest.mark.reference_pinned test.',
    )
    group.add_argument(
        '--physics-invariant-audit',
        action='store_true',
        help='Advisory: list physics-module tests without an invariant marker or property-based language.',
    )
    args = parser.parse_args()
    if args.baseline:
        return cmd_baseline()
    if args.check:
        return cmd_check()
    if args.reference_pinned_audit:
        return cmd_reference_pinned_audit()
    if args.physics_invariant_audit:
        return cmd_physics_invariant_audit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
