"""Shared fixtures for the interior_chem tests."""

from __future__ import annotations

import pytest

from proteus.interior_chem import eos_deng

# Resolved at collection, before tests/conftest.py points an unset FWL_DATA at an
# empty data root for each test.
_KEANE_TABLE_FILE = eos_deng.table_path()


@pytest.fixture
def keane_table(monkeypatch):
    """Point eos_deng at the Keane + Deng-thermal table in FWL_DATA, or skip:
    the table is kept out of the repository."""
    if not _KEANE_TABLE_FILE.is_file():
        pytest.skip('Keane + Deng-thermal table not in FWL_DATA/interior_chem')
    monkeypatch.setattr(eos_deng, 'table_path', lambda: _KEANE_TABLE_FILE)
    return _KEANE_TABLE_FILE


_CONSTDV_TABLE_FILE = eos_deng.table_path('keane_table_constdV')
_FLAT_TABLE_FILE = eos_deng.table_path('keane_table_flat')


@pytest.fixture
def keane_tables(monkeypatch):
    """All table models (extrapolated and the two capped test variants), or skip."""
    files = {
        'keane_table': _KEANE_TABLE_FILE,
        'keane_table_flat': _FLAT_TABLE_FILE,
        'keane_table_constdV': _CONSTDV_TABLE_FILE,
    }
    if not all(f.is_file() for f in files.values()):
        pytest.skip('Keane tables not in FWL_DATA/interior_chem')
    monkeypatch.setattr(eos_deng, 'table_path', lambda model='keane_table': files[model])
    monkeypatch.setattr(eos_deng, '_TABLES', {})
    return files
