"""Config key validation: detect TOML keys that the parser does not consume.

This module provides helpers that recursively compare the raw dict, read from
the PROTEUS toml-formatted config files, against the Config schema. It checks
for keys in the raw dict that are not declared in the Config schema. These
are important to identify because a user could set these with intent, but they
will be ignored if not mapped by the parser.
"""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=256)
def _type_hints_for(cls: type) -> types.MappingProxyType[str, typing.Any]:
    """Resolve the type hints of a schema class, once per class.

    The config modules postpone annotation evaluation, so each annotation is
    a string that `typing.get_type_hints` recompiles on every call. The walk
    below asks for the hints of every nested class on every config load, and
    the schema classes do not change once imported, so one resolution per
    class gives the same answer for a fraction of the work. The cache holds
    the schema comfortably; 46 classes declare the config today.

    Parameters
    ----------
    cls:
        attrs-decorated class whose annotations are to be resolved.

    Returns
    -------
    types.MappingProxyType[str, typing.Any]
        Field name to resolved type hint. Every caller receives the same
        object, so it is a read-only view: a write raises rather than
        corrupting the schema view held by every later config load.

    Raises
    ------
    NameError
        If an annotation names something the defining module cannot resolve.
        Failures are not cached, so a repeat call raises again.

    Notes
    -----
    A class whose annotations are rewritten after its first resolution keeps
    the first result.
    """
    return types.MappingProxyType(typing.get_type_hints(cls))


def _extract_attrs_class(hint: type) -> type | None:
    """Return the attrs class from a type hint, unwrapping union types.

    This is required for handling recursion (nested classes) in the config
    schema. It is called by `find_key_problems` below.

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


def find_key_problems(
    data: dict, cls: type = Config, path: str = ''
) -> tuple[list[str], list[str]]:
    """Recursively collect the keys in *data* that the schema cannot accept.

    Both kinds of fault leave the file saying one thing and the run doing
    another, so both are collected in one walk and reported together.

    Parameters
    ----------
    data:
        Raw TOML dict as returned by `tomllib.load`, or a sub-dict of one.
    cls:
        attrs-decorated class to compare against. Defaults to the whole Config
        schema; the parameter exists so the walk can descend into nested
        classes and is not normally passed by callers.
    path:
        Dotted key prefix for the paths in the returned lists. Carried by the
        recursion; callers start at the root and leave it empty.

    Returns
    -------
    tuple[list[str], list[str]]
        Two lists of dotted key paths, in file order. The first holds names the
        schema does not declare. The second holds sections the schema declares
        as a table but which the file supplies as something else. Both are
        empty for a conforming file.
    """

    # If the class is not an attrs class, then we cannot inspect it.
    if not attrs.has(cls):
        return [], []

    # Get the field names of the attrs class
    field_names = {f.name for f in attrs.fields(cls)}

    # Get the type hints of the attrs class, if possible.
    # If not, log a warning but attempt to continue.
    try:
        hints = _type_hints_for(cls)
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
            sub_orphans, sub_mistyped = find_key_problems(value, nested_cls, full_path)
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
        setting = 'setting it' if single else 'setting them'
        blocks.append(
            f'{heading} in {path}:\n'
            f'  {keys}\n'
            f'  {subject} unrecognised, so {setting} will have no effect. '
            f'Check for typos or outdated option names.'
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
            f'another kind of value. Check for typos or double-brackets.'
        )

    blocks.append('See input/all_options.toml for reference, or read the docs:')
    blocks.append('https://proteus-framework.org/PROTEUS/Reference/config/params.html')
    return '\n'.join(blocks)
