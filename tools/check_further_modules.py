#!/usr/bin/env python3
"""Check that the four further modules of PROTEUS are usable.

Tests import of the three Python packages (Morrigan, VULCAN, atmodeller),
verifies that the doctor diagnostics report passes for all four further
modules (including Obliqua), and loads Obliqua through the PROTEUS orbit
wrapper.

Exit codes: 0 when all four modules are usable, 1 when any check fails.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable

from proteus.doctor import PASS, run_all_checks

MODULES: tuple[str, ...] = ('morrigan', 'vulcan', 'atmodeller')
DOCTOR_NAMES: tuple[str, ...] = (
    'fwl-morrigan',
    'fwl-vulcan',
    'atmodeller',
    'Obliqua',
)


def check_imports(
    importer: Callable[[str], Any] = importlib.import_module,
) -> list[str]:
    """Check that each of the three Python modules imports.

    Parameters
    ----------
    importer : callable, default=importlib.import_module
        Import function taking a module name string.

    Returns
    -------
    list of str
        Problem descriptions; empty when all three modules import.
    """
    problems: list[str] = []
    for name in MODULES:
        try:
            importer(name)
        except Exception as exc:
            problems.append(f'{name} does not import: {type(exc).__name__}: {exc}')
    return problems


def check_doctor(results: list[Any]) -> list[str]:
    """Check that the doctor report includes each of the four modules with pass status.

    Parameters
    ----------
    results : list of CheckResult
        Diagnostic results as returned by ``proteus.doctor.run_all_checks()``.

    Returns
    -------
    list of str
        Problem descriptions; empty when every module has a pass status.
    """
    problems: list[str] = []
    for name in DOCTOR_NAMES:
        matches = [r for r in results if getattr(r, 'name', None) == name]
        if len(matches) != 1:
            problems.append(f'{name}: not in the doctor report')
            continue
        res = matches[0]
        if res.status != PASS:
            problems.append(f'{name}: doctor says {res.status}: {res.message}')
    return problems


def check_obliqua(
    root: Path, loader: Callable[[dict[str, str]], Any] | None = None
) -> list[str]:
    """Load Obliqua the way a PROTEUS run does.

    Parameters
    ----------
    root : Path
        PROTEUS repository root folder.
    loader : callable or None, default=None
        Function accepting a dictionary mapping of directory paths. When
        None, imports ``proteus.orbit.obliqua.import_obliqua``.

    Returns
    -------
    list of str
        Problem descriptions; empty when Obliqua loads cleanly.
    """
    if loader is None:
        from proteus.orbit.obliqua import import_obliqua

        loader = import_obliqua
    try:
        loader({'obliqua': str(root / 'Obliqua')})
    except Exception as exc:
        return [f'Obliqua does not load through PROTEUS: {type(exc).__name__}: {exc}']
    return []


def main(argv: list[str] | None = None) -> int:
    """Run the checks for the four further modules.

    Parameters
    ----------
    argv : list of str or None, default=None
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 if all checks pass, 1 if any check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='PROTEUS repository root folder',
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    problems_imports = check_imports()
    problems_doctor = check_doctor(run_all_checks())
    problems_obliqua = check_obliqua(root)

    checks = [
        ('imports', problems_imports),
        ('doctor', problems_doctor),
        ('obliqua', problems_obliqua),
    ]

    all_problems: list[str] = []
    for label, problems in checks:
        if problems:
            for p in problems:
                print(p, file=sys.stderr)
            all_problems.extend(problems)
        else:
            print(f'{label}: ok')

    if all_problems:
        return 1

    print('All 4 further modules are usable.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
