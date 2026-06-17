"""Shared fixtures for the outgassing test suite.

See ``docs/How-to/test_infrastructure.md`` for the fixture conventions.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_atmodeller_model_cache():
    """Reset the atmodeller equilibrium-model cache around every test.

    ``proteus.outgas.atmodeller`` reuses one ``EquilibriumModel`` per species
    network (so its JIT solver compiles once instead of every solve). That cache
    is module-level and would otherwise persist across tests: a model built
    under a patched ``EquilibriumModel`` in one test would be reused in the next
    test with the same species signature, ignoring that test's own patch. Clear
    it before and after each test to keep the tests independent.
    """
    from proteus.outgas.atmodeller import _MODEL_CACHE

    _MODEL_CACHE.clear()
    yield
    _MODEL_CACHE.clear()
