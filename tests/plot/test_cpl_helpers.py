from __future__ import annotations

from helpers import PROTEUS_ROOT

from proteus.plot._cpl_helpers import get_options_dirs_from_argv


def test_get_options_dirs_from_argv():
    default = str(PROTEUS_ROOT / 'input' / 'default.toml')

    options, dirs = get_options_dirs_from_argv(default=default)

    assert 'star_model' in options
    assert 'output' in dirs
