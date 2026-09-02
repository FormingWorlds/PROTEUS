from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import cattrs

from ._config import Config
from ._interior import _STEP_CAP_FIELDS
from .orphans import UnknownConfigKeyError, find_key_problems, format_orphan_message

log = logging.getLogger('fwl.' + __name__)


def _is_explicit_zero(value: object) -> bool:
    """True for a TOML int or float value of zero, false for a bool or 0.0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0.0


def _check_step_cap_zeros(raw: dict, path: Path | str) -> None:
    """Reject an explicit 0.0 for an Aragog step-cap field.

    An absent key resolves to the schema default and, on a zalmoxis
    interior structure, to the Aragog wrapper's built-in cap; only a
    value the file actually sets is checked here, and only when
    ``interior_energetics.module`` resolves to ``aragog``.

    Parameters
    ----------
    raw:
        Raw TOML dict as returned by `read_config`.
    path:
        Config file the dict came from, quoted back in any error.

    Raises
    ------
    ValueError
        If a step-cap field is explicitly set to 0.0.
    """
    interior_energetics = raw.get('interior_energetics', {})
    if not isinstance(interior_energetics, dict):
        return
    if interior_energetics.get('module', 'aragog') != 'aragog':
        return
    aragog = interior_energetics.get('aragog', {})
    if not isinstance(aragog, dict):
        return
    zeroed = [f for f in _STEP_CAP_FIELDS if _is_explicit_zero(aragog.get(f))]
    if zeroed:
        names = ', '.join(f'interior_energetics.aragog.{f}' for f in zeroed)
        raise ValueError(
            f'Invalid configuration in {path}:\n'
            f'  {names} set to 0.0, which is ambiguous with the unset default.\n'
            f'  Omit the key to use the default, set it to -1.0 to disable the cap, '
            f'or set a positive value for a custom cap.'
        )


def read_config(path: Path | str) -> dict:
    """Read config file from path"""
    log.debug('Reading config from %s', path)
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    log.debug('TOML sections loaded: %s', sorted(config.keys()))
    return config


def structure_config(raw: dict, path: Path | str) -> Config:
    """Structure a raw config dict into a Config object.

    This performs no key checking: cattrs discards anything the schema does not
    map, so a caller that uses this directly is responsible for having checked
    the keys itself. `read_config_object` is the checked entry point and is
    what almost every caller wants. The step is separate so the runner can
    obtain a configuration, resolve the output directory named inside it, and
    record a refusal there before raising.

    Parameters
    ----------
    raw:
        Raw TOML dict as returned by `read_config`.
    path:
        Config file the dict came from, quoted back in any error.

    Returns
    -------
    Config
        The structured configuration.

    Raises
    ------
    ValueError
        If a value fails validation.
    """

    _check_step_cap_zeros(raw, path)

    try:
        obj = cattrs.structure(raw, Config)
        log.debug(
            'Config structured: star.module=%s, interior_energetics.module=%s, '
            'outgas.module=%s, atmos_clim.module=%s, escape.module=%s',
            obj.star.module,
            obj.interior_energetics.module,
            obj.outgas.module,
            obj.atmos_clim.module,
            obj.escape.module,
        )
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


def read_config_object(path: Path | str) -> Config:
    """Read and validate config into Config object.

    Parameters
    ----------
    path:
        Path to the TOML config file.

    Returns
    -------
    Config
        The structured configuration.

    Raises
    ------
    UnknownConfigKeyError
        If the file carries keys the schema cannot accept.
    ValueError
        If a value fails validation.
    """

    # Read config from TOML file in path as a raw dict.
    cfg = read_config(path)

    # Reject unusable keys before structuring, so that a typo is reported as a
    # typo rather than as whatever the resulting default happens to break
    # further downstream.
    orphans, mistyped = find_key_problems(cfg)
    if orphans or mistyped:
        raise UnknownConfigKeyError(format_orphan_message(orphans, path, mistyped))

    return structure_config(cfg, path)


__all__ = [
    'Config',
    'UnknownConfigKeyError',
    'read_config_object',
    'read_config',
    'structure_config',
    'find_key_problems',
    'format_orphan_message',
]
