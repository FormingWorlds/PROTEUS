"""Unit tests for ``proteus.atmos_chem.common``.

Exercises the ``read_result`` function, which loads atmospheric
chemistry output CSV files from the output directory.

Invariants tested:
  - Correct file path construction from module name and optional filename
  - Guard: returns None for 'none' or None module
  - Guard: returns None when the CSV file does not exist
  - Successful read of well-formed whitespace-delimited CSV

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import os

import pytest

from proteus.atmos_chem.common import read_result

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# -----------------------------------------------------------------------
# Guard: disabled module
# -----------------------------------------------------------------------


def test_read_result_none_module_returns_none():
    """When module is None, read_result returns None without touching
    the filesystem.

    Edge case: chemistry is disabled in the config.
    """
    result = read_result('/nonexistent/path', module=None)
    assert result is None
    # Adjacent check: the string 'none' is also a disabled sentinel
    result2 = read_result('/nonexistent/path', module='none')
    assert result2 is None


# -----------------------------------------------------------------------
# Guard: missing file
# -----------------------------------------------------------------------


def test_read_result_missing_file_returns_none(tmp_path):
    """When the CSV file does not exist on disk, read_result returns
    None and does not raise.

    Edge case: the output directory exists but the chemistry run
    did not produce output (solver failure, wrong module name).
    """
    # Create the offchem subdirectory but no CSV file
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir()

    result = read_result(str(tmp_path), module='vulcan')
    assert result is None
    # The expected path would be <outdir>/offchem/vulcan.csv
    assert not os.path.exists(str(offchem_dir / 'vulcan.csv'))


# -----------------------------------------------------------------------
# Successful read
# -----------------------------------------------------------------------


def test_read_result_reads_whitespace_delimited_csv(tmp_path):
    """read_result parses a whitespace-delimited CSV into a DataFrame
    with correct column names and values.

    Uses a minimal 2-column, 3-row file to verify parsing. The
    delimiter is whitespace (consistent with VULCAN output format).
    """
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir()

    # Write a whitespace-delimited CSV (VULCAN output format)
    csv_content = 'pressure   temperature\n1.0e5   300.0\n1.0e4   250.0\n1.0e3   200.0\n'
    csv_path = offchem_dir / 'vulcan.csv'
    csv_path.write_text(csv_content, encoding='utf-8')

    result = read_result(str(tmp_path), module='vulcan')
    assert result is not None
    assert len(result) == 3
    assert 'pressure' in result.columns
    assert 'temperature' in result.columns
    # Verify numeric values were parsed correctly
    assert result['pressure'].iloc[0] == pytest.approx(1.0e5, rel=1e-12)
    assert result['temperature'].iloc[2] == pytest.approx(200.0, rel=1e-12)


def test_read_result_custom_filename(tmp_path):
    """When filename is provided, read_result uses it instead of
    constructing '<module>.csv'.

    This path is used by online mode to read per-snapshot files
    (e.g. 'vulcan_5000.csv').
    """
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir()

    csv_content = 'species   vmr\nH2O   0.01\nCO2   0.001\n'
    csv_path = offchem_dir / 'vulcan_5000.csv'
    csv_path.write_text(csv_content, encoding='utf-8')

    result = read_result(str(tmp_path), module='vulcan', filename='vulcan_5000.csv')
    assert result is not None
    assert len(result) == 2
    assert result['species'].iloc[0] == 'H2O'
    # Default filename would have looked for 'vulcan.csv' and returned None
    result_default = read_result(str(tmp_path), module='vulcan')
    assert result_default is None
