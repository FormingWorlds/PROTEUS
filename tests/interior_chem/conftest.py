"""Shared fixtures for the interior_chem tests."""

from __future__ import annotations

import pytest

from proteus.interior_chem import eos_deng


@pytest.fixture
def keane_table():
    """The Keane + Deng-thermal table, built in memory once per process."""
    return eos_deng._keane_table()
