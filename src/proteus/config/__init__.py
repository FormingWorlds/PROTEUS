from __future__ import annotations

import linecache
import logging
import re
import tomllib
import traceback
from pathlib import Path
from typing import Literal, Union

import cattrs
from attr import validators
from cattrs.errors import AttributeValidationNote, IterableValidationNote

from ._config import Config
from ._interior import _STEP_CAP_FIELDS
from .orphans import (
    UnknownConfigKeyError,
    field_path_kind,
    find_key_problems,
    format_orphan_message,
)

log = logging.getLogger('fwl.' + __name__)


def structure_k_val(val, _cls):
    if isinstance(val, bool) or not isinstance(val, (int, str)):
        raise ValueError(f'Expected int or "none", got {val!r}')
    if val == 'none':
        return 'none'
    return int(val)


# Register this for the specific Union type
cattrs.register_structure_hook(Union[int, Literal['none']], structure_k_val)


def _is_explicit_zero(value: object) -> bool:
    """True for a TOML int or float value of zero, false for a bool or 0.0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0.0


def _check_step_cap_zeros(raw: dict, path: Path | str) -> None:
    """Reject an explicit 0.0 for an Aragog step-cap field.

    -1.0 is the canonical explicit off spelling for a step cap, so an explicit
    0.0 is refused and the user is pointed at -1.0. An absent key resolves to
    the schema default 0.0, which the wrapper reads as off, so only a value the
    file actually sets is checked here. Checked only when
    ``interior_energetics.module`` resolves to ``aragog`` (an absent module key
    included, since aragog is the schema default); for any other module the
    step-cap fields are inert.

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


def _checked_field(exc: BaseException) -> str:
    """Name of the one field *exc* is about, or '' when it can concern several.

    The attrs-generated ``__init__`` line that raised calls a converter, which sees only
    its own value, or a validator, which checks only its own field when it is an attrs
    built-in; a PROTEUS validator can compare several fields.
    """
    frames = list(traceback.walk_tb(exc.__traceback__))
    init = [(f, n) for f, n in frames if f.f_code.co_filename.startswith('<attrs generated')]
    if not init:
        return ''
    line = linecache.getline(init[-1][0].f_code.co_filename, init[-1][1])
    if match := re.search(r'__attr_converter_(\w+)\(', line):
        return match[1]
    match = re.search(r'__attr_validator_(\w+)\(', line)
    return match[1] if match and frames[-1][0].f_code.co_filename == validators.__file__ else ''


def _error_lines(exc: BaseException, path: str = '') -> list[str]:
    """One ``key.path: reason`` line per leaf of a cattrs structuring error.

    Parameters
    ----------
    exc:
        The error cattrs raised, usually a nested exception group.
    path:
        Dotted key path of *exc* in the config.

    Returns
    -------
    list[str]
        The leaf errors in the order cattrs raised them.
    """
    if not isinstance(exc, cattrs.errors.BaseValidationError):
        reason = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
        return [f'{path}: {reason}' if path and not reason.startswith(path) else reason]
    lines = []
    for sub in exc.exceptions:
        notes = getattr(sub, '__notes__', ())
        step = next(
            (n.name for n in notes if isinstance(n, AttributeValidationNote)), ''
        ) or _checked_field(sub)
        index = next((n.index for n in notes if isinstance(n, IterableValidationNote)), None)
        sub_path = '.'.join(p for p in (path, step) if p)
        if index is not None:
            sub_path += f'[{index}]'
        lines += _error_lines(sub, sub_path)
    return lines


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
        detail = '\n  '.join(_error_lines(e))
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
    'field_path_kind',
    'find_key_problems',
    'format_orphan_message',
]
