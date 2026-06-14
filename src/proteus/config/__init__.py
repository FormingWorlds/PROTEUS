from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import cattrs

from ._config import Config
from .orphans import check_config_orphan_free

log = logging.getLogger('fwl.' + __name__)


def read_config(path: Path | str) -> dict:
    """Read config file from path"""
    log.debug('Reading config from %s', path)
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    log.debug('TOML sections loaded: %s', sorted(config.keys()))
    return config


def read_config_object(path: Path | str) -> Config:
    """Read and validate config into Config object."""

    # Read config from TOML file in path as a raw dict.
    cfg = read_config(path)

    # Attempt to structure config with cattrs.
    try:
        # Structure the config
        obj = cattrs.structure(cfg, Config)
        log.debug(
            'Config structured: star.module=%s, interior_energetics.module=%s, '
            'outgas.module=%s, atmos_clim.module=%s, escape.module=%s',
            obj.star.module,
            obj.interior_energetics.module,
            obj.outgas.module,
            obj.atmos_clim.module,
            obj.escape.module,
        )

        # Looks good! Return the structured config object.
        return obj

    # Catch validation exceptions
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


__all__ = ['Config', 'read_config_object', 'read_config', 'check_config_orphan_free']
