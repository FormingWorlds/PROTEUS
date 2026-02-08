#!/usr/bin/env python3
"""Automatically ratchet coverage thresholds for fast and full test suites.

This script implements a coverage ratcheting mechanism: the required coverage
threshold for a given suite can only increase or stay the same, never
decrease. It supports two modes:

* full  – updates `[tool.coverage.report].fail_under` (the global threshold)
* fast  – updates `[tool.proteus.coverage_fast].fail_under` (unit/smoke gate)

Usage examples::

    # Ratchet full-suite threshold using coverage.json
    python tools/update_coverage_threshold.py --coverage-file coverage.json --target full

    # Ratchet fast (unit/smoke) threshold using a separate coverage JSON
    python tools/update_coverage_threshold.py --coverage-file coverage-unit.json --target fast

Exit codes:
    0 -> threshold updated (increased)
    1 -> no update needed
    2 -> validation failure (e.g., missing keys)

This script is intended to run in CI after coverage is computed for the
corresponding suite; it can also be used locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError as e:
        raise ImportError(
            'tomllib (Python 3.11+) or tomli package is required. '
            'Install with: pip install tomli'
        ) from e

import tomlkit


def read_current_coverage(coverage_file: Path) -> float:
    """Read the current test coverage percentage from a coverage JSON file.

    Args:
        coverage_file: Path to coverage JSON (from `coverage json`)

    Returns:
        Current coverage as a float (e.g., 69.23 for 69.23%)

    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If the JSON format is unexpected
    """
    if not coverage_file.exists():
        raise FileNotFoundError(
            f"{coverage_file} not found. Run 'coverage json -o {coverage_file}' first."
        )

    with coverage_file.open() as f:
        data = json.load(f)

    return float(data['totals']['percent_covered'])


def read_threshold_from_pyproject(target: str) -> float:
    """Read the current coverage threshold from pyproject.toml for a target.

    Args:
        target: "full" or "fast"

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        ValueError: If the required setting is missing
    """
    pyproject_file = Path('pyproject.toml')
    if not pyproject_file.exists():
        raise FileNotFoundError('pyproject.toml not found')

    data = tomllib.loads(pyproject_file.read_text())
    try:
        if target == 'full':
            return float(data['tool']['coverage']['report']['fail_under'])
        if target == 'fast':
            return float(data['tool']['proteus']['coverage_fast']['fail_under'])
    except KeyError as exc:
        raise ValueError(
            f"fail_under setting not found in pyproject.toml for target '{target}'"
        ) from exc

    raise ValueError(f"Unknown target '{target}'")


def update_threshold_in_pyproject(target: str, new_threshold: float) -> bool:
    """Update the fail_under threshold in pyproject.toml for a target.

    Args:
        target: "full" or "fast"
        new_threshold: New threshold value (rounded to 2 decimals)

    Returns:
        True if file was updated, False if no change was needed
    """
    pyproject_file = Path('pyproject.toml')
    if not pyproject_file.exists():
        raise FileNotFoundError('pyproject.toml not found')

    document = tomlkit.parse(pyproject_file.read_text())

    if target == 'full':
        report_section = document.get('tool', {}).get('coverage', {}).get('report')
        if report_section is None:
            raise ValueError('[tool.coverage.report] section not found in pyproject.toml')
        section = report_section
    elif target == 'fast':
        proteus_section = document.get('tool', {}).get('proteus')
        if proteus_section is None:
            raise ValueError('[tool.proteus] section not found in pyproject.toml')
        coverage_fast = proteus_section.get('coverage_fast')
        if coverage_fast is None:
            raise ValueError('[tool.proteus.coverage_fast] section not found in pyproject.toml')
        section = coverage_fast
    else:
        raise ValueError(f"Unknown target '{target}'")

    current_value = float(section.get('fail_under', 0))
    new_value = float(f'{new_threshold:.2f}')

    # Ratchet: only update if strictly higher
    if new_value <= current_value:
        return False

    section['fail_under'] = new_value
    pyproject_file.write_text(tomlkit.dumps(document))
    print(f'[+] Updated pyproject.toml: {target} fail_under = {new_value:.2f}')
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ratcheted coverage thresholds')
    parser.add_argument(
        '--coverage-file',
        default='coverage.json',
        help="Path to coverage JSON (from 'coverage json'). Default: coverage.json",
    )
    parser.add_argument(
        '--target',
        choices=['full', 'fast'],
        default='full',
        help="Which threshold to ratchet: 'full' (global) or 'fast' (unit/smoke)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entrypoint for threshold ratcheting."""
    args = parse_args()

    try:
        coverage_path = Path(args.coverage_file)
        target = args.target

        current_coverage = read_current_coverage(coverage_path)
        current_threshold = read_threshold_from_pyproject(target)

        print(f'Target: {target}')
        print(f'Current coverage: {current_coverage:.2f}%')
        print(f'Current threshold: {current_threshold:.2f}%')

        new_threshold = round(current_coverage, 2)

        if new_threshold > current_threshold:
            print(
                f'[+] Coverage increased! Updating threshold: '
                f'{current_threshold:.2f}% -> {new_threshold:.2f}%'
            )
            update_threshold_in_pyproject(target, new_threshold)
            return 0

        if new_threshold == current_threshold:
            print(f'[=] Threshold already at {current_threshold:.2f}% (no update needed)')
            return 1

        # Coverage decreased - pytest fail-under should have caught this already
        print(f'[!] Coverage decreased: {new_threshold:.2f}% < {current_threshold:.2f}%')
        print('    Threshold not updated.')
        return 2

    except FileNotFoundError as e:
        print(f'[x] Error: Required file not found: {e}', file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print(
            f'[x] Error: Invalid coverage data or configuration ({type(e).__name__}): {e}',
            file=sys.stderr,
        )
        return 2
    except Exception as e:
        print(
            f'[x] Error updating coverage threshold ({type(e).__name__}): {e}',
            file=sys.stderr,
        )
        return 2


if __name__ == '__main__':
    sys.exit(main())
