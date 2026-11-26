"""Shared fixtures for the interior_chem tests."""

from __future__ import annotations

import pytest

from proteus.interior_chem import eos_deng


@pytest.fixture
def keane_table():
    """The Keane + Deng-thermal table, built in memory once per process."""
    return eos_deng._keane_table()


# Resolved at collection, before tests/conftest.py points an unset FWL_DATA at an
# empty data root for each test.
_CONSTDV_TABLE_FILE = eos_deng.table_path('keane_table_constdV')
_FLAT_TABLE_FILE = eos_deng.table_path('keane_table_flat')


@pytest.fixture
def keane_tables(monkeypatch):
    """The table files of the two capped test variants, or skip."""
    files = {
        'keane_table_flat': _FLAT_TABLE_FILE,
        'keane_table_constdV': _CONSTDV_TABLE_FILE,
    }
    if not all(f.is_file() for f in files.values()):
        pytest.skip('Keane tables not in FWL_DATA/interior_chem')
    monkeypatch.setattr(eos_deng, 'table_path', lambda model: files[model])
    monkeypatch.setattr(eos_deng, '_TABLES', {})
    return files


_KOMAFIT_FILE = eos_deng.komafit_path()


@pytest.fixture
def komafit_table(monkeypatch, keane_tables):
    """The Keane tables plus the komafit extension of the Komabayashi liquids, or skip."""
    if not _KOMAFIT_FILE.is_file():
        pytest.skip('Komabayashi extension table not in FWL_DATA/interior_chem')
    monkeypatch.setattr(eos_deng, 'komafit_path', lambda: _KOMAFIT_FILE)
    monkeypatch.setattr(eos_deng, '_KOMAFIT', None)
    return _KOMAFIT_FILE
