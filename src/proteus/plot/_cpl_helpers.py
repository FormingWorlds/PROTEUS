from __future__ import annotations

import sys

from proteus import Proteus


def get_options_dirs_from_argv(default: str = 'init_coupler.toml') -> tuple[dict, dict]:
    """Attempts to get config from argv, otherwise defaults to default

    Returns config dict and directories
    """
    config_path = sys.argv[1] if len(sys.argv) == 2 else default
    handler = Proteus(config_path=config_path)

    return handler.config, handler.directories
