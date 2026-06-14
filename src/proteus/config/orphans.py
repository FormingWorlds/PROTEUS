"""Config key validation: detect TOML keys that the parser does not consume.

This module provides helpers that recursively compare the raw dict, read from
the PROTEUS toml-formatted config files, against the Config schema. It checks
for keys in the raw dict that are not declared in the Config schema. These
are important to identify because a user could set these with intent, but they
will be ignored if not mapped by the parser.
"""

from __future__ import annotations

import logging
import typing

import attrs

from ._config import Config

log = logging.getLogger('fwl.' + __name__)


def _extract_attrs_class(hint: type) -> type | None:
    """Return the attrs class from a type hint, unwrapping union types.

    This is required for handling recursion (nested classes) in the config
    schema. It is called by `_collect_orphan_keys` below.

    Parameters
    ----------
    hint:
        Type hint to inspect.

    Returns
    -------
    type | None
        The attrs-decorated class if found, else None.
    """
    if isinstance(hint, type) and attrs.has(hint):
        return hint
    args = getattr(hint, '__args__', None)
    if args:
        for arg in args:
            if isinstance(arg, type) and attrs.has(arg):
                return arg
    return None


def _collect_orphan_keys(data: dict, cls: type, path: str = '') -> list[str]:
    """Recursively collect TOML keys that have no matching field in *cls*.

    Parameters
    ----------
    data:
        Raw TOML sub-dict to inspect.
    cls:
        attrs-decorated class to compare against.
    path:
        Dotted key prefix for building human-readable paths in error messages.

    Returns
    -------
    list[str]
        Dotted key paths (e.g. ``"planet.orphan_field"``) that appear in
        *data* but are not declared fields of *cls* or any nested attrs class.
    """

    # If the class is not an attrs class, then we cannot inspect it.
    if not attrs.has(cls):
        return []

    # Get the field names of the attrs class
    field_names = {f.name for f in attrs.fields(cls)}

    # Get the type hints of the attrs class, if possible.
    # If not, log a warning but attempt to continue.
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        log.warning(f'Config validator failed to get type hints for {cls}.')
        hints = {}

    orphans: list[str] = []

    # Loop through keys in the raw dict.
    for key, value in data.items():
        # Build a dotted path
        full_path = f'{path}.{key}' if path else key

        # Check if the key is in the attrs class. If not, add to orphans.
        if key not in field_names:
            orphans.append(full_path)

        # If this is a dict, then we need to go deeper.
        elif isinstance(value, dict):
            nested_cls = _extract_attrs_class(hints.get(key))

            # Recursion on this function
            if nested_cls is not None:
                orphans.extend(_collect_orphan_keys(value, nested_cls, full_path))

    return orphans


def check_config_orphan_free(raw_dict: dict, outdir: str | None = None) -> bool:
    """Detect if *raw_dict* contains keys that Config schema doesn't define.

    Parameters
    ----------
    raw_dict:
        Raw TOML dict as returned by `tomllib.load`.
    outdir:
        Output directory path.

    Returns
    ------
    bool
        True if looks good. False if unrecognised keys are found.
    """

    # Identify orphaned keys in the raw_dict
    orphans = _collect_orphan_keys(raw_dict, Config)

    # No orphans -> looks good
    if not orphans:
        return True

    # If didn't return, then we have some orphans to deal with
    # Construct a message for the user.
    log.error('Configuration contains "orphan" keys that are not recognised.')
    log.error('\tPerhaps you have a typo or are using an outdated option name.')
    log.error('\tCheck input/all_options.toml for parameter reference.')

    # List the keys
    msg = '\tUnrecognised keys: ' + ', '.join(f'"{key}"' for key in orphans)
    log.error(msg)

    # Return false if orphans
    return False
