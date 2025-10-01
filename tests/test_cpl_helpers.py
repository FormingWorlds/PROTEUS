from __future__ import annotations

import sys
from unittest.mock import patch

from helpers import PROTEUS_ROOT

from proteus.config import Config
from proteus.plot._cpl_helpers import get_handler_from_argv


def test_get_handler_from_argv():
    config_path = str(PROTEUS_ROOT / 'input' / 'minimal.toml')

    with patch.object(sys, 'argv', [None, config_path]):
        handler = get_handler_from_argv()

    assert isinstance(handler.config, Config)
    assert 'output' in handler.directories
