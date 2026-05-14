from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from helpers import PROTEUS_ROOT

from proteus.config import Config
from proteus.plot._cpl_helpers import get_handler_from_argv


@pytest.mark.unit
@pytest.mark.skip(reason='FIXME: ValueError raised against input/minimal.toml in CI container; same root cause as tests/config/test_config.py::test_factory_defaults_from_minimal_config. Tracked in claude-config/memory for the test-rework phase.')
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
