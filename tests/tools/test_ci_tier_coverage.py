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

# A heavy shard's job cap must clear the selected test's own pytest-timeout by
# at least this margin, because a slow checkout plus a SOCRATES cache-miss
# rebuild can approach this before the first test line even runs. Below the
# margin the runner cancels the job before the test's own timer can fire, which
# is the failure the one-test sharding exists to prevent.
SETUP_BUDGET_MIN = 25
# GitHub hosted-runner hard job limit. A timeout-minutes value at or above this
# is silently clamped, so a cap set here loses the "test times out first" order.
RUNNER_HARD_LIMIT_MIN = 360
# The slow-tier job cap a shard falls back to when it pins no ``job_minutes``.
# Mirrors ``timeout-minutes: ${{ matrix.job_minutes || 210 }}`` in the workflow
# and is verified against it, so the value cannot drift out of the assertion
# messages that cite it.
DEFAULT_JOB_MINUTES = 210


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


def _pytestmark_items(tree: ast.Module) -> list[ast.expr]:
    """Return the marker expressions in a module-level ``pytestmark``.

    Handles both a bare assignment (``pytestmark = [...]``) and an annotated one
    (``pytestmark: list = [...]``); the latter parses as ``ast.AnnAssign`` and
    would otherwise read as an empty marker list.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is None:  # an annotation with no assigned value
            continue
        if not any(getattr(t, 'id', None) == 'pytestmark' for t in targets):
            continue
        return value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return []


def _module_marker_names(tree: ast.Module) -> set[str]:
    """Return the tier names in a module-level ``pytestmark`` assignment."""
    return _marker_names(_pytestmark_items(tree))


def _timeout_seconds(items: list[ast.expr]) -> int | None:
    """Return the seconds from a ``pytest.mark.timeout(N)`` among marker items.

    ``items`` is either a module ``pytestmark`` list or a function's decorator
    list, so this resolves both the module-level and the function-level timeout.
    Both the positional ``timeout(N)`` and the keyword ``timeout(timeout=N)``
    forms are read, since pytest-timeout accepts either.
    """
    for item in items:
        node = item.func if isinstance(item, ast.Call) else item
        if not (isinstance(node, ast.Attribute) and node.attr == 'timeout'):
            continue
        if not (isinstance(node.value, ast.Attribute) and node.value.attr == 'mark'):
            continue
        if not isinstance(item, ast.Call):
            continue
        if item.args:
            arg = item.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                return int(arg.value)
        for kw in item.keywords:
            if kw.arg == 'timeout' and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, (int, float)):
                    return int(kw.value.value)
    return None


def _effective_timeout_minutes(path: Path, test_name: str) -> float | None:
    """Minutes of the effective ``pytest.mark.timeout`` on ``test_name``.

    ``test_name`` is a resolved non-skip test id from ``_tiers_per_test`` and is
    matched by exact function name, so a non-slow or skip-marked sibling whose
    name is a superset of the select substring can never supply its timeout in
    place of the real test. A function-level timeout overrides the module-level
    one (pytest resolves overlapping markers closest-first); the module
    ``pytestmark`` timeout is the fallback. Returns ``None`` when the resolved
    timeout is absent or non-positive, the latter because ``timeout(0)`` disables
    the timer entirely, which the caller must treat as an unbounded test rather
    than a zero-minute one.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    module_secs = _timeout_seconds(_pytestmark_items(tree))
    bare = test_name.split('::')[-1]
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != bare:
            continue
        secs = _timeout_seconds(node.decorator_list)
        if secs is None:
            secs = module_secs
        return secs / 60 if secs and secs > 0 else None
    return None


def _workflow_default_job_minutes() -> int | None:
    """The slow-tier ``timeout-minutes: ${{ matrix.job_minutes || N }}`` default."""
    m = re.search(r'matrix\.job_minutes\s*\|\|\s*(\d+)', _nightly_text())
    return int(m.group(1)) if m else None


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


def _slow_matrix_shards() -> list[dict]:
    """Parse the slow-tier matrix into one dict per shard entry.

    ``shard:`` appears only in the slow tier, so splitting the workflow text on
    it yields exactly the shard entries. Each dict carries the shard name, its
    ``os:`` runner, its ``select:`` value (or None), its ``job_minutes:`` cap (or
    None), and the test files it names. The ``os`` field lets a heavy shard run
    the same select on two runners for parity without reading as a duplicate.
    Parsed as text because PyYAML is not a dependency.
    """
    entries = re.split(r'\n\s+- shard:', _nightly_text())[1:]
    shards = []
    for entry in entries:
        # The last entry runs to end-of-file; stop it at the job's runs-on key
        # so trailing lines cannot leak file paths into this shard.
        entry = entry.split('\n    runs-on:')[0]
        name = entry.splitlines()[0].strip()
        os_m = re.search(r'^\s+os:\s*(\S+)\s*$', entry, re.M)
        sel = re.search(r'^\s+select:\s*(\S+)\s*$', entry, re.M)
        cap = re.search(r'^\s+job_minutes:\s*(\d+)\s*$', entry, re.M)
        shards.append(
            {
                'shard': name,
                'os': os_m.group(1) if os_m else None,
                'select': sel.group(1) if sel else None,
                'job_minutes': int(cap.group(1)) if cap else None,
                'files': re.findall(r'^\s+(tests/\S+\.py)\s*$', entry, re.M),
            }
        )
    return shards


def _integration_paths() -> list[str]:
    """Directories the integration tier passes to pytest."""
    m = re.search(r'pytest ((?:tests/\S+ +)+)\\', _nightly_text())
    assert m, 'no pytest invocation with test paths found in the integration tier'
    return shlex.split(m.group(1))


def _duplicate_select_os(shards: list[dict]) -> list[tuple[str, str | None]]:
    """Return ``(select, os)`` pairs that appear on more than one shard.

    Two shards sharing a select but differing in ``os`` are a legitimate dual-OS
    parity pattern and do not count. A repeat of the same pair means the same
    test is scheduled twice on the same runner, which is the real duplicate.
    """
    seen: set[tuple[str, str | None]] = set()
    dups: list[tuple[str, str | None]] = []
    for s in shards:
        pair = (s['select'], s['os'])
        if pair in seen:
            dups.append(pair)
        seen.add(pair)
    return dups


def _tests_selected_more_than_once(
    slow_tests: set[str], selects: set[str]
) -> dict[str, list[str]]:
    """Map each slow test to the distinct selects that match it as a substring,
    keeping only the tests matched by two or more.

    A ``-k`` select matches by substring, so two distinct selects
    (``solved_adiabat_pinned`` alongside ``test_m1_solved_adiabat_pinned``) can
    resolve to the same test and schedule it in two shards. This isolates that
    shape; an empty result means every test is claimed by at most one select.
    """
    cover: dict[str, set[str]] = {}
    for sel in selects:
        for name in slow_tests:
            if sel in name:
                cover.setdefault(name, set()).add(sel)
    return {name: sorted(sels) for name, sels in cover.items() if len(sels) > 1}


def _files_run_both_per_test_and_whole(
    select_shards: list[dict], whole_file_shards: list[dict]
) -> dict[str, list[str]]:
    """Map each select-sharded file that also runs whole in another shard to the
    whole-file shards that name it.

    A file run per-test under ``-k select`` and also whole in a non-select shard
    has its tests executed in both, so the mapping is empty exactly when the two
    shard sets are disjoint on files.
    """
    select_files = {s['files'][0] for s in select_shards if s['files']}
    out: dict[str, list[str]] = {}
    for s in whole_file_shards:
        for rel in sorted(set(s['files']).intersection(select_files)):
            out.setdefault(rel, []).append(s['shard'])
    return out


def test_marker_extractor_detects_a_double_tier(tmp_path):
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
    # Write the fixture into tmp_path, not the source tree: a crash before an
    # in-tree unlink would leave an untracked module behind, and parallel
    # workers would race on a shared path.
    tmp = tmp_path / '_tier_sample.py'
    tmp.write_text(sample, encoding='utf-8')
    tiers = _tiers_per_test(tmp)

    # The doubly-marked function is the shape this guard exists to catch.
    assert tiers['test_double'] == {'unit', 'slow'}
    # A non-tier marker (timeout) must not be mistaken for a tier.
    assert tiers['test_single'] == {'unit'}
    # Markers accumulate through a class, not just a function.
    assert tiers['TestGroup::test_in_class'] == {'unit', 'smoke'}

    # The two branches the sweep below relies on but the sample above does not
    # reach: a module with no pytestmark whose function carries no tier is the
    # runs-nowhere shape the sweep must report as untiered, and a skip-marked
    # function must be dropped entirely so the deliberately disabled tests do
    # not count as untiered.
    orphan_sample = (
        'import pytest\n'
        '@pytest.mark.skip\n'
        'def test_skipped():\n'
        '    pass\n'
        'def test_orphan():\n'
        '    pass\n'
    )
    orphan = tmp_path / '_tier_orphan.py'
    orphan.write_text(orphan_sample, encoding='utf-8')
    orphan_tiers = _tiers_per_test(orphan)

    # An untiered test resolves to the empty set, which the sweep flags.
    assert orphan_tiers['test_orphan'] == set()
    # A skip-marked test is excluded, not reported as untiered.
    assert 'test_skipped' not in orphan_tiers


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


def test_select_sharded_files_cover_every_slow_test():
    """A file run under per-shard ``-k select:`` runs only the tests those shards
    name, so file-level reachability no longer implies every test runs.

    A slow test added to such a file later would be selected by no shard and run
    nowhere, while ``test_every_tiered_file_is_reachable_by_its_ci_job`` stays
    green because the file is still named. This guard asserts instead that the
    shard selects cover every non-skip slow test in each select-sharded file
    exactly once, that no such file is also run whole in another shard, and that
    each shard's ``job_minutes`` cap sits below the 360 min hosted-runner hard
    limit and above the selected test's own ``pytest.mark.timeout`` plus a setup
    budget, so a hung test trips its own timer before the runner cancels the job.
    """
    all_shards = _slow_matrix_shards()
    select_shards = [s for s in all_shards if s['select']]
    assert select_shards, 'expected select-based heavy shards in the slow matrix'

    # The fallback cap the messages below cite is read from the workflow, not
    # trusted as prose, so editing the ``|| N`` default reddens here rather than
    # leaving a stale number in an assertion string.
    assert _workflow_default_job_minutes() == DEFAULT_JOB_MINUTES, (
        f'the slow-tier job_minutes default in {NIGHTLY.name} is '
        f'{_workflow_default_job_minutes()}, not the {DEFAULT_JOB_MINUTES} that '
        'DEFAULT_JOB_MINUTES records; update the constant with the workflow'
    )

    # A repeat of the same (select, os) means one test is scheduled twice on the
    # same runner; a shared select across two runners is legitimate OS parity.
    dup_pairs = _duplicate_select_os(select_shards)
    assert not dup_pairs, (
        f'these heavy shards repeat a (select, os) pair, so the same test runs '
        f'twice on one runner: {dup_pairs}'
    )

    # A select-sharded file must not also appear whole in a non-select shard, or
    # its tests run once per-test and again as part of the whole-file shard.
    whole_file_shards = [s for s in all_shards if not s['select']]
    both = _files_run_both_per_test_and_whole(select_shards, whole_file_shards)
    assert not both, (
        'these files run per-test in a select shard and also whole in another '
        f'shard, so their tests execute in two shards: {both}'
    )

    by_file: dict[str, list[str]] = {}
    for s in select_shards:
        assert s['job_minutes'] is not None, (
            f'shard {s["shard"]} selects a single test but pins no job_minutes; a '
            'heavy shard must set its cap explicitly so the cap-versus-timeout '
            f'check below has a value, because the {DEFAULT_JOB_MINUTES} min '
            'default can sit below the selected test timeout plus the setup budget'
        )
        assert s['job_minutes'] < RUNNER_HARD_LIMIT_MIN, (
            f'shard {s["shard"]} job_minutes={s["job_minutes"]} is at or above the '
            f'{RUNNER_HARD_LIMIT_MIN} min hosted-runner hard job limit'
        )
        assert len(s['files']) == 1, (
            f'select shard {s["shard"]} must name exactly one file, got {s["files"]}'
        )
        rel = s['files'][0]
        slow_tests = {n for n, t in _tiers_per_test(REPO_ROOT / rel).items() if 'slow' in t}
        matched = [n for n in slow_tests if s['select'] in n]
        assert len(matched) == 1, (
            f'select {s["select"]!r} in {rel} matches {matched}; it must name exactly '
            'one non-skip slow test (a substring select can silently match two)'
        )
        # Resolve the timeout against the matched test by exact name, not the
        # select substring, so a non-slow or skip-marked sibling cannot supply a
        # smaller timeout that would pass the cap check for the wrong reason.
        timeout_min = _effective_timeout_minutes(REPO_ROOT / rel, matched[0])
        assert timeout_min is not None, (
            f'shard {s["shard"]} selects {matched[0]!r} but it carries no positive '
            'pytest.mark.timeout and the file sets none either'
        )
        assert s['job_minutes'] >= timeout_min + SETUP_BUDGET_MIN, (
            f'shard {s["shard"]} job_minutes={s["job_minutes"]} leaves under '
            f'{SETUP_BUDGET_MIN} min above the selected test timeout of '
            f'{timeout_min:.0f} min, so a slow setup can cancel the job before the '
            'test times out; raise job_minutes or lower the test timeout'
        )
        by_file.setdefault(rel, []).append(s['select'])

    for rel, selects in by_file.items():
        slow_tests = {n for n, t in _tiers_per_test(REPO_ROOT / rel).items() if 'slow' in t}
        distinct = set(selects)
        # Coverage on the test axis, not the select-string axis: two distinct
        # selects resolving to the same test each pass the per-shard match check
        # above yet still schedule that test in two shards.
        over = _tests_selected_more_than_once(slow_tests, distinct)
        assert not over, (
            f'{rel}: these tests are matched by more than one distinct select, so '
            f'they run in more than one shard: {over}'
        )
        selected = {n for sel in distinct for n in slow_tests if sel in n}
        assert selected == slow_tests, (
            f'{rel}: the shards select {sorted(selected)} but the file holds non-skip '
            f'slow tests {sorted(slow_tests)}; the unselected ones run in no shard'
        )


def test_effective_timeout_reads_the_resolved_tests_own_timeout(tmp_path):
    """The cap guard reads the resolved test's own timeout, not a sibling's.

    A function-level ``pytest.mark.timeout`` overrides the module-level one, in
    either the positional or the keyword form; the module marker may be a bare or
    an annotated assignment; ``timeout(0)`` disables the timer and must read as
    unbounded, not zero. Matching by exact name keeps a non-slow or skip-marked
    sibling from supplying a smaller timeout that would pass the cap check for the
    wrong reason.
    """
    annotated = (
        'import pytest\n'
        'pytestmark: list = [pytest.mark.slow, pytest.mark.timeout(3600)]\n'
        '@pytest.mark.timeout(5400)\n'
        'def test_heavy_solve():\n'
        '    pass\n'
        '@pytest.mark.timeout(timeout=18000)\n'  # keyword form is valid pytest-timeout syntax
        'def test_keyword_form():\n'
        '    pass\n'
        'def test_inherits_module():\n'
        '    pass\n'
    )
    path = tmp_path / '_timeout_annotated.py'
    path.write_text(annotated, encoding='utf-8')

    # Function marker wins: 5400 s is 90 min, not the module's 3600 s / 60 min.
    assert _effective_timeout_minutes(path, 'test_heavy_solve') == pytest.approx(90.0)
    # The keyword form timeout(timeout=N) resolves like the positional form.
    assert _effective_timeout_minutes(path, 'test_keyword_form') == pytest.approx(300.0)
    # The AnnAssign module marker is the fallback for a function with no timeout;
    # a missed AnnAssign would read as an empty marker list and return None here.
    assert _effective_timeout_minutes(path, 'test_inherits_module') == pytest.approx(60.0)
    # A name no function carries resolves to None, a hard failure for the caller.
    assert _effective_timeout_minutes(path, 'test_absent') is None

    # A bare pytestmark assignment is the production shape; its module timeout is
    # the fallback when the selected function carries none of its own.
    bare = (
        'import pytest\n'
        'pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]\n'
        'def test_bare_fallback():\n'
        '    pass\n'
    )
    bpath = tmp_path / '_timeout_bare.py'
    bpath.write_text(bare, encoding='utf-8')
    assert _effective_timeout_minutes(bpath, 'test_bare_fallback') == pytest.approx(120.0)

    # A tier-only pytestmark with no timeout anywhere resolves to None, not zero.
    no_to = (
        'import pytest\n'
        'pytestmark = [pytest.mark.slow]\n'
        'def test_no_timeout_anywhere():\n'
        '    pass\n'
    )
    npath = tmp_path / '_timeout_none.py'
    npath.write_text(no_to, encoding='utf-8')
    assert _effective_timeout_minutes(npath, 'test_no_timeout_anywhere') is None

    # timeout(0) disables the timer, so it must read as unbounded (None): a test
    # with no self-cancel is always the runner-cancels-first shape the guard reds.
    disabled = (
        'import pytest\n'
        'pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]\n'
        '@pytest.mark.timeout(0)\n'
        'def test_timer_disabled():\n'
        '    pass\n'
    )
    dpath = tmp_path / '_timeout_disabled.py'
    dpath.write_text(disabled, encoding='utf-8')
    assert _effective_timeout_minutes(dpath, 'test_timer_disabled') is None


def test_file_run_both_per_test_and_whole_is_flagged():
    """A per-test-sharded file must not also run whole in another shard.

    Running a file both under ``-k select`` and as a whole-file shard executes
    its tests twice; the helper names the offending file and the whole-file shard
    and stays empty when the two shard sets are disjoint on files.
    """
    select_shards = [
        {
            'shard': 'heavy',
            'select': 'test_a',
            'os': 'ubuntu-latest',
            'files': ['tests/x/test_heavy.py'],
        },
    ]
    disjoint = [{'shard': 'misc', 'files': ['tests/x/test_other.py']}]
    assert _files_run_both_per_test_and_whole(select_shards, disjoint) == {}

    colliding = [
        {'shard': 'misc', 'files': ['tests/x/test_heavy.py', 'tests/x/test_other.py']},
    ]
    assert _files_run_both_per_test_and_whole(select_shards, colliding) == {
        'tests/x/test_heavy.py': ['misc']
    }


def test_shard_dedup_allows_dual_os_but_flags_repeats():
    """Dual-OS parity is allowed; a same-runner repeat is not.

    The dedup axis is ``(select, os)``: the same select on two runners is a
    legitimate parity pattern, while the same pair twice schedules one test twice
    on one runner. A substring select resolving to two distinct tests, or two
    selects to one test, is the double-execution shape on the coverage axis.
    """
    parity = [
        {'select': 'test_x', 'os': 'ubuntu-latest'},
        {'select': 'test_x', 'os': 'macos-latest'},
    ]
    assert _duplicate_select_os(parity) == []
    repeated = parity + [{'select': 'test_x', 'os': 'ubuntu-latest'}]
    assert _duplicate_select_os(repeated) == [('test_x', 'ubuntu-latest')]

    slow_tests = {'test_m1_solved_adiabat_pinned', 'test_solved_entropy_is_mass_independent'}
    # Two distinct selects collapsing onto one test is the shape the string-only
    # dedup missed: 'solved_adiabat_pinned' is a substring of the m1 test name.
    collide = _tests_selected_more_than_once(
        slow_tests, {'solved_adiabat_pinned', 'test_m1_solved_adiabat_pinned'}
    )
    assert collide == {
        'test_m1_solved_adiabat_pinned': [
            'solved_adiabat_pinned',
            'test_m1_solved_adiabat_pinned',
        ]
    }
    # Distinct selects that each name a different test raise nothing.
    clean = _tests_selected_more_than_once(
        slow_tests, {'test_m1_solved_adiabat_pinned', 'mass_independent'}
    )
    assert clean == {}
