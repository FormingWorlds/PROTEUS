#!/usr/bin/env python3
"""Automatically update test coverage threshold based on current coverage.

This script implements a coverage ratcheting mechanism: the required coverage
threshold can only increase or stay the same, never decrease. This ensures
that as new tests are added, the baseline coverage is automatically raised,
preventing coverage regression in future commits.

Usage:
    python tools/update_coverage_threshold.py

The script:
1. Reads current coverage from coverage.json
2. Reads current threshold from pyproject.toml
3. If current coverage >= threshold, updates pyproject.toml
4. Returns exit code 0 if update made, 1 if no update needed

This is typically run automatically by CI on successful test runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore

import tomlkit


def read_current_coverage() -> float:
    """Read the current test coverage percentage from coverage.json.

    Returns:
        Current coverage as a float (e.g., 69.23 for 69.23%)

    Raises:
        FileNotFoundError: If coverage.json doesn't exist
        KeyError: If coverage.json format is unexpected
    """
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        raise FileNotFoundError(
            "coverage.json not found. Run 'coverage json' first."
        )

    with open(coverage_file) as f:
        data = json.load(f)

    # Extract total coverage percentage
    current_coverage = float(data["totals"]["percent_covered"])
    return current_coverage


def read_threshold_from_pyproject() -> float:
    """Read the current coverage threshold from pyproject.toml.

    Returns:
        Current fail_under threshold as a float (e.g., 69.0 for 69%)

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        ValueError: If fail_under setting not found
    """
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        raise FileNotFoundError("pyproject.toml not found")

    data = tomllib.loads(pyproject_file.read_text())
    try:
        return float(data["tool"]["coverage"]["report"]["fail_under"])
    except KeyError as exc:
        raise ValueError("fail_under setting not found in pyproject.toml") from exc


def update_threshold_in_pyproject(new_threshold: float) -> bool:
    """Update the fail_under threshold in pyproject.toml.

    Args:
        new_threshold: New threshold value to set (will be rounded to 2 decimals)

    Returns:
        True if file was updated, False if no change needed
    """
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        raise FileNotFoundError("pyproject.toml not found")

    document = tomlkit.parse(pyproject_file.read_text())
    report_section = document.get("tool", {}).get("coverage", {}).get("report")
    if report_section is None:
        raise ValueError("[tool.coverage.report] section not found in pyproject.toml")

    current_value = float(report_section.get("fail_under", 0))
    new_value = float(f"{new_threshold:.2f}")

    if new_value <= current_value:
        return False

    report_section["fail_under"] = new_value
    pyproject_file.write_text(tomlkit.dumps(document))
    print(f"✅ Updated pyproject.toml: fail_under = {new_value:.2f}")
    return True


def main() -> int:
    """Main function to update coverage threshold.

    Returns:
        0 if threshold was updated, 1 if no update needed
    """
    try:
        # Read current state
        current_coverage = read_current_coverage()
        current_threshold = read_threshold_from_pyproject()

        print(f"Current coverage: {current_coverage:.2f}%")
        print(f"Current threshold: {current_threshold:.2f}%")

        # Round current coverage down to 2 decimal places for threshold
        # This ensures we don't set a threshold higher than what we achieved
        new_threshold = round(current_coverage, 2)

        # Only update if new threshold is higher than current
        if new_threshold > current_threshold:
            print(f"📈 Coverage increased! Updating threshold: {current_threshold:.2f}% → {new_threshold:.2f}%")
            update_threshold_in_pyproject(new_threshold)
            return 0
        elif new_threshold == current_threshold:
            print(f"✓ Coverage threshold already at {current_threshold:.2f}% (no update needed)")
            return 1
        else:
            # Coverage decreased - this should trigger a test failure via pytest-cov
            print(f"⚠️  Coverage decreased: {new_threshold:.2f}% < {current_threshold:.2f}%")
            print("    Tests should have failed. Threshold not updated.")
            return 1

    except Exception as e:
        print(f"❌ Error updating coverage threshold: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
