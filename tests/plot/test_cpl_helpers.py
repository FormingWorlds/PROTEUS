from __future__ import annotations

import sys
from unittest.mock import patch

from helpers import PROTEUS_ROOT

from proteus.plot._cpl_helpers import get_options_dirs_from_argv


def test_get_options_dirs_from_argv():
    config_path = str(PROTEUS_ROOT / 'input' / 'default.toml')

    with patch.object(sys, 'argv', [None, config_path]):
        options, dirs = get_options_dirs_from_argv()

    assert 'star_model' in options
    assert 'output' in dirs