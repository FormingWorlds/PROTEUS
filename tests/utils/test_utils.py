"""
Tests for proteus.utils module
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_placeholder():
    """Smoke check that the proteus.utils package imports cleanly.

    Acts as the floor-coverage entry until per-module test files under
    tests/utils/ fully replace it. Asserts the package is import-clean
    so a future regression in proteus/utils/__init__.py surfaces here.
    """
    import types

    from proteus import utils

    assert isinstance(utils, types.ModuleType)
    assert utils.__name__ == 'proteus.utils'
