from __future__ import annotations

import sys
from pathlib import Path

import cattrs

from ._config import Config

if sys.version_info < (3, 11):
    import toml

    read_config = toml.load
else:
    import tomllib

    def read_config(path: Path | str) -> dict:
        """Read config file from path"""
        with open(path, 'rb') as f:
            config = tomllib.load(f)
        return config


def read_config_object(path: Path | str) -> Config:
    """Read and validate config into Config object."""
    cfg = read_config(path)
    return cattrs.structure(cfg, Config)


__all__ = [
    "Config",
    "read_config_object",
    "read_config",
]
