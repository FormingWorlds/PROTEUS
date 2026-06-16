"""Read module pins from pyproject.toml.

Single entry point for tools/get_*.sh and the CI composite action to look
up the URL and ref of an external module (AGNI, SOCRATES, SPIDER, VULCAN,
LovePy, PETSc). The source of truth is pyproject.toml's
``[tool.proteus.modules.<name>]`` table.

Usage from a shell script:

    URL=$(python tools/_module_pins.py agni url)
    REF=$(python tools/_module_pins.py agni ref)

Exits non-zero with a clear message if the module is unknown or the
requested field is missing.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib


def _repo_root() -> pathlib.Path:
    """Return the PROTEUS repo root, identified by ``pyproject.toml``.

    Walks up from this file's directory so the helper works whether it
    is called from the repo root, ``tools/``, or a CI checkout copy.
    """
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    raise SystemExit('Could not locate pyproject.toml above _module_pins.py')


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print('usage: python tools/_module_pins.py <module> <field>', file=sys.stderr)
        print('  fields: url, ref, sha256', file=sys.stderr)
        return 2

    _, module, field = argv
    data = tomllib.loads((_repo_root() / 'pyproject.toml').read_text())
    modules = data.get('tool', {}).get('proteus', {}).get('modules', {})

    if module not in modules:
        known = ', '.join(sorted(modules)) or '(none configured)'
        print(
            f"Unknown module '{module}'. Known: {known}",
            file=sys.stderr,
        )
        return 1

    entry = modules[module]
    if field not in entry:
        print(
            f"Module '{module}' has no field '{field}'. Available: {', '.join(sorted(entry))}",
            file=sys.stderr,
        )
        return 1

    print(entry[field])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
