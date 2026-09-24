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
