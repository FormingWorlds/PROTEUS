from __future__ import annotations

import tomllib
from pathlib import Path

import cattrs

from ._config import Config


def read_config(path: Path | str) -> dict:
    """Read config file from path"""
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    return config


def read_config_object(path: Path | str) -> Config:
    """Read and validate config into Config object."""
    cfg = read_config(path)
    try:
        return cattrs.structure(cfg, Config)
    except cattrs.errors.ClassValidationError as e:
        # Extract actionable error messages from the nested exception group
        messages = []
        for exc in e.exceptions:
            if hasattr(exc, 'exceptions'):
                for sub in exc.exceptions:
                    messages.append(str(sub))
            else:
                messages.append(str(exc))
        detail = '\n  '.join(messages)
        raise ValueError(
            f'Invalid configuration in {path}:\n  {detail}\n'
            f'See input/all_options.toml for the full parameter reference.'
        ) from None


__all__ = ['Config', 'read_config_object', 'read_config']
