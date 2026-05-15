from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from helpers import PROTEUS_ROOT

from proteus.config import Config
from proteus.plot._cpl_helpers import get_handler_from_argv


@pytest.mark.unit
def test_get_handler_from_argv():
    """get_handler_from_argv creates a Proteus handler with parsed config."""
    config_path = str(PROTEUS_ROOT / 'input' / 'minimal.toml')

    with (
        patch.object(sys, 'argv', [None, config_path]),
        patch('proteus.proteus.Proteus.init_directories'),
    ):
        handler = get_handler_from_argv()

    assert isinstance(handler.config, Config)
    # directories not populated (init_directories mocked)
    assert handler.directories is None
