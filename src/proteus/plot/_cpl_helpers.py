from __future__ import annotations

import sys

from proteus import Proteus


def get_handler_from_argv(default: str = 'init_coupler.toml') -> Proteus:
    """Attempts to get handler from argv, otherwise defaults to default"""
    config_path = sys.argv[1] if len(sys.argv) == 2 else default
    handler = Proteus(config_path=config_path)

    return handler
