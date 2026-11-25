#!/usr/bin/env python3
"""Translate a PROTEUS configuration file from format version 3.0 to 3.1.

Version 3.1 changed three defaults:

- ``escape.zephyrus.scheme`` is new and defaults to ``"dispatcher"``, the
  ZEPHYRUS regime dispatcher. Under it an unset ``escape.reservoir`` resolves to
  ``"pxuv"``, ``escape.hill_clamp`` to off, and ``escape.zephyrus.tidal`` to on.
  3.0 computed energy-limited escape with ``"outgas"``, the Hill clip on, and the
  tidal factor off, which ``scheme = "el"`` reproduces.
- ``atmos_clim.p_top`` went from 1e-6 to 1e-10 bar, so that the XUV wind base
  of the dispatcher lies inside the atmosphere column.
- ``atmos_clim.agni.conduction`` went from off to on.

A 3.0 file that leaves these unset would run differently under 3.1 without
saying so, so the loader refuses it. This tool sets each of them to its 3.0
value where the file leaves it unset, keeps every value the file states, and
writes ``config_version = "3.1"``. Comments and layout are kept. The pins are
written whatever modules the file selects, so a grid that switches a module
per case still reproduces 3.0.

A grid file (one with ``ref_config``) only has its ``config_version`` updated,
since the grid manager does not read the key and the pins belong in the
reference config.

Usage
-----
    python tools/migrate_config_v3_0_to_v3_1.py old.toml -o new.toml
    python tools/migrate_config_v3_0_to_v3_1.py old.toml      # writes old.v3_1.toml
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import tomlkit

OLD_VERSION = '3.0'
NEW_VERSION = '3.1'

# Dotted path -> the 3.0 default that 3.1 changed.
PINS = {
    'escape.zephyrus.scheme': 'el',
    'atmos_clim.p_top': 1.0e-6,
    'atmos_clim.agni.conduction': False,
}


def _table(doc, keys: list[str]):
    """Return the table at ``keys``, creating any that are missing."""
    node = doc
    for key in keys:
        if key not in node:
            node[key] = tomlkit.table()
        node = node[key]
    return node


def _has(doc, path: str) -> bool:
    node = doc
    for key in path.split('.'):
        if not hasattr(node, 'keys') or key not in node:
            return False
        node = node[key]
    return True


def migrate(doc) -> list[str]:
    """Translate a parsed 3.0 document in place and return what was pinned.

    Raises
    ------
    ValueError
        If the document is not a 3.0 file.
    """
    version = str(doc.get('config_version', OLD_VERSION))
    if version != OLD_VERSION:
        raise ValueError(f'expected config_version = "{OLD_VERSION}", found "{version}"')
    doc['config_version'] = NEW_VERSION
    if 'ref_config' in doc:
        return []
    pinned = []
    for path, value in PINS.items():
        if _has(doc, path):
            continue
        *parents, leaf = path.split('.')
        _table(doc, parents)[leaf] = value
        pinned.append(f'{path} = {tomlkit.item(value).as_string()}')
    return pinned


def _flat(data: dict, prefix: str = '') -> dict:
    out = {}
    for key, value in data.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            out.update(_flat(value, path + '.'))
        else:
            out[path] = value
    return out


def check(before: str, after: str) -> list[str]:
    """Problems with a translation: what changed beyond the version and the pins.

    Both texts are read back with ``tomllib``, so a pin that tomlkit placed
    where it changes the meaning of other keys (after dotted keys, or inside an
    inline table) is caught here rather than by the loader later.
    """
    try:
        old, new = _flat(tomllib.loads(before)), _flat(tomllib.loads(after))
    except tomllib.TOMLDecodeError as exc:
        return [f'the output is not valid TOML ({exc})']
    want = {**old, 'config_version': NEW_VERSION}
    if 'ref_config' not in old:
        want.update({k: v for k, v in PINS.items() if k not in old})
    problems = [f'{k} is missing or changed' for k in want if new.get(k) != want[k]]
    problems += [f'{k} was added' for k in new if k not in want]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input', type=Path, help='3.0 config or grid file')
    parser.add_argument(
        '-o', '--output', type=Path, help='output path (default: <input>.v3_1.toml)'
    )
    args = parser.parse_args(argv)

    out = args.output or args.input.with_suffix('.v3_1.toml')
    if out.resolve() == args.input.resolve():
        print('Refusing to overwrite the input file. Choose another --output.', file=sys.stderr)
        return 1
    if out.exists():
        print(f'{out} exists. Choose another --output.', file=sys.stderr)
        return 1

    text = args.input.read_text()
    doc = tomlkit.parse(text)
    try:
        pinned = migrate(doc)
    except ValueError as exc:
        print(f'{args.input}: {exc}', file=sys.stderr)
        return 1
    problems = check(text, tomlkit.dumps(doc))
    if problems:
        print(
            f'{args.input}: the translation would change the file, so nothing was written. '
            'Write escape, escape.zephyrus, atmos_clim, and atmos_clim.agni as [section] '
            'tables, not dotted keys or inline tables, and put config_version first. '
            + '. '.join(problems),
            file=sys.stderr,
        )
        return 1
    out.write_text(tomlkit.dumps(doc))

    print(f'Wrote {out} (config_version = "{NEW_VERSION}")')
    for line in pinned:
        print(f'  pinned {line}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
