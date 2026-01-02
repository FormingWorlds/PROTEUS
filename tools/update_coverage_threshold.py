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

    content = pyproject_file.read_text()

    # Find the fail_under line in [tool.coverage.report] section
    in_coverage_section = False
    for line in content.split("\n"):
        if "[tool.coverage.report]" in line:
            in_coverage_section = True
            continue

        if in_coverage_section:
            if line.strip().startswith("["):
                # Entered a new section, stop looking
                break
            if "fail_under" in line:
                # Extract value: "fail_under = 69" -> 69.0
                value = line.split("=")[1].strip()
                return float(value)

    raise ValueError("fail_under setting not found in pyproject.toml")


def update_threshold_in_pyproject(new_threshold: float) -> bool:
    """Update the fail_under threshold in pyproject.toml.

    Args:
        new_threshold: New threshold value to set (will be rounded to 2 decimals)

    Returns:
        True if file was updated, False if no change needed
    """
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()
    lines = content.split("\n")

    # Find and update the fail_under line
    updated = False
    in_coverage_section = False
    for i, line in enumerate(lines):
        if "[tool.coverage.report]" in line:
            in_coverage_section = True
            continue

        if in_coverage_section:
            if line.strip().startswith("["):
                # Entered a new section
                in_coverage_section = False
                continue
            if "fail_under" in line:
                # Update the line with new threshold (rounded to 2 decimals)
                old_line = line
                # Preserve indentation and format
                indent = len(line) - len(line.lstrip())
                new_line = " " * indent + f"fail_under = {new_threshold:.2f}"
                lines[i] = new_line
                updated = (old_line != new_line)
                break

    if updated:
        pyproject_file.write_text("\n".join(lines))
        print(f"✅ Updated pyproject.toml: fail_under = {new_threshold:.2f}")

    return updated


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
