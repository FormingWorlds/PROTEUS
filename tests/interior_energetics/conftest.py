"""Shared fixtures for the interior energetics tests."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _clear_superliquidus_memo():
    """Clear the process-wide super-liquidus anchor memo around each test."""

    def _clear():
        zal = sys.modules.get('proteus.interior_struct.zalmoxis')
        if zal is not None:
            zal._clear_superliquidus_cache()

    _clear()
    yield
    _clear()


@pytest.fixture
def _no_core_table(monkeypatch):
    """Stub the core temperature table builder for rows without a structure (M_core, P_center,
    R_core, P_cmb), on which the real builder raises; the solve gets no table. Opt in with
    usefixtures; tests/interior_energetics/test_structure_core_table.py tests the real builder."""
    import proteus.interior_energetics.wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod, '_structure_core_table', lambda *a, **k: None)
