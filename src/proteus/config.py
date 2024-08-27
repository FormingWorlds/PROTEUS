from __future__ import annotations

import sys
from pathlib import Path

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
