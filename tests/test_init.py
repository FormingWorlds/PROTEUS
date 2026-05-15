# Test importing PROTEUS as a python library
from __future__ import annotations

import pytest

from proteus import __version__

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_version():
    """``proteus.__version__`` resolves to a truthy string at import time
    (smoke check that setuptools-scm produced a version, not an empty default).
    """
    assert __version__
