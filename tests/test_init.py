# Test importing PROTEUS as a python library
from __future__ import annotations

import pytest

from proteus import __version__

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_version():
    assert __version__
