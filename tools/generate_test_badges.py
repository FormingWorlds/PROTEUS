"""Generate shields.io endpoint-badge JSON files for PROTEUS test counts.

The script invokes ``pytest --collect-only -q`` per marker expression to
count tests without executing them, then writes one JSON file per badge
under the ``--out`` directory in the shields.io endpoint-badge schema:

    {"schemaVersion": 1, "label": "<text>", "message": "<count>", "color": "blue"}

Public-surface output (three files, two categories):

- ``tests-total.json`` (label "tests"): count of ``not skip`` tests.
- ``tests-unit.json`` (label "unit tests"): count of ``unit and not skip``.
- ``tests-integration.json`` (label "integration tests"): count of
  ``(smoke or integration or slow) and not skip``.

The internal pytest marker scheme has four tiers (``unit``, ``smoke``,
``integration``, ``slow``); the public badge surface intentionally
collapses ``smoke + integration + slow`` into a single
"Integration Tests" category because a four-way taxonomy is confusing to
non-developer readers. Internal CI granularity is unaffected; the four
markers are still registered in ``pyproject.toml`` and used by
``ci-pr-checks.yml`` and ``ci-nightly.yml`` directly.

The documentation deploy writes the JSON files into the published site
under ``badges/`` so shields.io can fetch them; nothing is pushed to a
branch.

Usage
-----
    python tools/generate_test_badges.py --out site/badges/

Notes
-----
Running the script does not execute the test suite; only collection is
triggered. Collection runs with ``--continue-on-collection-errors`` so a
single test module that fails to import (for example, a version mismatch
in an optional submodule) does not break the badge or the documentation
deploy; the tests in that module are simply excluded from the count and a
warning is emitted. Pytest exit code 5 ("no tests collected") is treated
as a successful zero count. A run is a hard failure only when no
collection summary line can be parsed at all.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_COLLECT_RE = re.compile(r'^(\d+)(?:/\d+)?\s+tests?\s+collected\b', re.MULTILINE)
_ERRORS_RE = re.compile(r',\s*(\d+)\s+errors?\b')

_BADGES: tuple[tuple[str, str, str], ...] = (
    ('total', 'tests', 'not skip'),
    ('unit', 'unit tests', 'unit and not skip'),
    (
        'integration',
        'integration tests',
        '(smoke or integration or slow) and not skip',
    ),
)

# Filenames not in the public-surface set; removed from the output
# directory at the end of every run so the badges directory stays in
# sync with the three-file scheme above.
_PRUNE_FILES: tuple[str, ...] = ('tests-smoke.json', 'tests-slow.json')


def count_tests(marker_expr: str) -> int:
    """Run pytest collection and return the number of selected tests.

    Parameters
    ----------
    marker_expr : str
        Pytest marker expression passed via ``-m``.

    Returns
    -------
    int
        Number of tests pytest collected for the given marker. Exit
        code 5 ("no tests collected") is mapped to 0. Modules that fail
        to import are skipped and excluded from the count.

    Raises
    ------
    RuntimeError
        If no collection summary line can be parsed from stdout.
    """
    proc = subprocess.run(
        [
            'pytest',
            '--collect-only',
            '-q',
            '--continue-on-collection-errors',
            '-m',
            marker_expr,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 5:
        return 0
    match = _COLLECT_RE.search(proc.stdout)
    if match is None:
        raise RuntimeError(
            f'pytest collection summary not found for marker {marker_expr!r} '
            f'(exit code {proc.returncode})\n'
            f'--- stdout ---\n{proc.stdout}\n'
            f'--- stderr ---\n{proc.stderr}'
        )
    errors = _ERRORS_RE.search(proc.stdout)
    if errors is not None and int(errors.group(1)) > 0:
        print(
            f'warning: {errors.group(1)} test module(s) failed to import for '
            f'marker {marker_expr!r}; those tests are excluded from the count.',
            file=sys.stderr,
        )
    return int(match.group(1))


def write_badge(out_dir: Path, name: str, label: str, count: int) -> Path:
    """Write a shields.io endpoint-badge JSON file.

    Parameters
    ----------
    out_dir : Path
        Directory to write the JSON file into. Must already exist.
    name : str
        Suffix used in the filename ``tests-<name>.json``.
    label : str
        Badge label rendered on the left side of the shield.
    count : int
        Badge message count rendered on the right side of the shield.

    Returns
    -------
    Path
        Path of the written JSON file.
    """
    payload = {
        'schemaVersion': 1,
        'label': label,
        'message': str(count),
        'color': 'blue',
    }
    out_path = out_dir / f'tests-{name}.json'
    out_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return out_path


def prune_extra_files(out_dir: Path) -> list[str]:
    """Remove any badge JSON files outside the public-surface set.

    Parameters
    ----------
    out_dir : Path
        Directory the JSON files live in.

    Returns
    -------
    list[str]
        Names of files that were removed, in the order encountered.
    """
    removed: list[str] = []
    for name in _PRUNE_FILES:
        path = out_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def main() -> int:
    """Entry point.

    Returns
    -------
    int
        Process exit code (always 0 on success; failures raise).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Directory to write tests-<name>.json badge files into.',
    )
    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, label, expr in _BADGES:
        count = count_tests(expr)
        write_badge(out_dir, name, label, count)
        print(f'{label}: {count}')

    for removed_name in prune_extra_files(out_dir):
        print(f'pruned: {removed_name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
