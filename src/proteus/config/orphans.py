"""Config key validation: detect TOML keys that the parser does not consume.

This module provides helpers that recursively compare the raw dict, read from
the PROTEUS toml-formatted config files, against the Config schema. It checks
for keys in the raw dict that are not declared in the Config schema. These
are important to identify because a user could set these with intent, but they
will be ignored if not mapped by the parser.
"""

from __future__ import annotations

import logging
import types
import typing
from pathlib import Path

import attrs

from ._config import Config

log = logging.getLogger('fwl.' + __name__)


class UnknownConfigKeyError(ValueError):
    """Raised when a config file carries keys outside the schema.

    A distinct type so that callers can present this to the user as a
    configuration problem without also catching unrelated failures. It derives
    from ValueError, which is what every other config rejection raises, so a
    caller that does not care about the distinction needs no change.
    """


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


def _expects_single_table(hint: type) -> bool:
    """Whether *hint* puts exactly one table at a field, rather than several.

    A field holding one nested class is written as ``[name]`` and nothing else
    is valid there. A field holding a container of them would be written as
    repeated ``[[name]]`` tables, where a list is the intended shape rather
    than a mistake. The schema declares no such field today, so this only
    keeps a later one from being refused.

    Parameters
    ----------
    hint:
        Type hint to inspect.

    Returns
    -------
    bool
        True when a single table is the only shape the field accepts.
    """
    origin = typing.get_origin(hint)
    if origin is None:
        return isinstance(hint, type) and attrs.has(hint)
    if origin in (types.UnionType, typing.Union):
        return any(isinstance(arg, type) and attrs.has(arg) for arg in typing.get_args(hint))
    return False


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

    return _collect_key_problems(data, cls, path)[0]


def _collect_key_problems(data: dict, cls: type, path: str = '') -> tuple[list[str], list[str]]:
    """Recursively collect the keys in *data* that *cls* cannot accept.

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
    tuple[list[str], list[str]]
        Two lists of dotted key paths, in file order. The first holds names the
        schema does not declare. The second holds sections the schema declares
        as a table but which the file supplies as something else.
    """

    # If the class is not an attrs class, then we cannot inspect it.
    if not attrs.has(cls):
        return [], []

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
    mistyped: list[str] = []

    # Loop through keys in the raw dict.
    for key, value in data.items():
        # Build a dotted path
        full_path = f'{path}.{key}' if path else key

        # Check if the key is in the attrs class. If not, add to orphans.
        if key not in field_names:
            orphans.append(full_path)
            continue

        # Nothing further to check unless the schema puts a nested class here.
        nested_cls = _extract_attrs_class(hints.get(key))
        if nested_cls is None:
            continue

        # If this is a dict, then we need to go deeper.
        if isinstance(value, dict):
            sub_orphans, sub_mistyped = _collect_key_problems(value, nested_cls, full_path)
            orphans.extend(sub_orphans)
            mistyped.extend(sub_mistyped)
        elif _expects_single_table(hints.get(key)):
            # The schema puts one table here and the file supplies something
            # else, most often because the section was written as [[name]]
            # rather than [name]. The name still matches a field, so nothing
            # above notices, while structuring discards the section whole and
            # every parameter inside it falls back to its default.
            mistyped.append(full_path)

    return orphans, mistyped


def find_orphan_keys(raw_dict: dict) -> list[str]:
    """List the keys in *raw_dict* that the Config schema does not define.

    Parameters
    ----------
    raw_dict:
        Raw TOML dict as returned by `tomllib.load`.

    Returns
    -------
    list[str]
        Dotted key paths (e.g. ``"planet.orphan_field"``) in file order.
        Empty when every key maps onto a Config field.
    """
    return _collect_orphan_keys(raw_dict, Config)


def find_key_problems(raw_dict: dict) -> tuple[list[str], list[str]]:
    """List the keys in *raw_dict* that the Config schema cannot accept.

    Both kinds leave the file saying one thing and the run doing another, so
    both are reported together.

    Parameters
    ----------
    raw_dict:
        Raw TOML dict as returned by `tomllib.load`.

    Returns
    -------
    tuple[list[str], list[str]]
        Dotted key paths in file order: names the schema does not define, and
        sections the schema declares as a table but which the file supplies as
        something else. Both are empty for a conforming file.
    """
    return _collect_key_problems(raw_dict, Config)


def format_orphan_message(
    orphans: list[str], path: Path | str, mistyped: list[str] | None = None
) -> str:
    """Build the user-facing message naming the keys that were refused.

    Parameters
    ----------
    orphans:
        Dotted key paths the schema does not define.
    path:
        Config file the keys came from, quoted back to the user.
    mistyped:
        Dotted paths of sections the schema declares as a table but which the
        file supplies as something else. Omit when there are none.

    Returns
    -------
    str
        Multi-line message listing the keys and how to resolve them.
    """
    blocks: list[str] = []

    if orphans:
        keys = ', '.join(f'"{key}"' for key in orphans)
        single = len(orphans) == 1
        heading = (
            'Unrecognised configuration key' if single else 'Unrecognised configuration keys'
        )
        subject = 'This key is' if single else 'These keys are'
        setting = 'Setting it' if single else 'Setting them'
        blocks.append(
            f'{heading} in {path}:\n'
            f'  {keys}\n'
            f'  {subject} not part of the configuration schema. {setting} '
            f'has no effect, so the file is refused rather than run on defaults '
            f'that were not asked for. Check for a typo or an outdated option '
            f'name.'
        )

    if mistyped:
        sections = ', '.join(f'"{key}"' for key in mistyped)
        single = len(mistyped) == 1
        heading = (
            'Misdeclared configuration section'
            if single
            else 'Misdeclared configuration sections'
        )
        subject = 'This section is' if single else 'These sections are'
        blocks.append(
            f'{heading} in {path}:\n'
            f'  {sections}\n'
            f'  {subject} declared as a table by the schema, but the file gives '
            f'another kind of value. Writing a section as [[name]] rather than '
            f'[name] is the usual cause. As written the whole section is '
            f'discarded and every parameter inside it falls back to its default.'
        )

    blocks.append('See input/all_options.toml for the full parameter reference.')
    return '\n'.join(blocks)
