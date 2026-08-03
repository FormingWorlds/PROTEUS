"""Guard on where ruff looks to decide that an import is first party.

The ecosystem modules are installed as clones beside ``src/`` at the repository
root. Whatever ruff treats as an import search path is also where it looks for
first-party names, so with the root searched, a clone directory carrying its
own module name makes that module read as part of this project and its imports
sort into the group that holds ``proteus``.

The damage is not symmetric between a contributor's machine and the checks,
because only the machine with the module installed carries the clone. The
formatter runs with ``--fix``, so on that machine it rewrites the import groups
of any file it touches, and the rewrite it commits is what the checks then
reject. Keeping the root off the search path closes both halves at once, for
every module rather than one name at a time.

The checks below read the configuration and the source, so they hold on a
machine that installed no modules at all, which is the case the checks
themselves run in.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from functools import cache
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_ROOTS = (REPO_ROOT / 'src', REPO_ROOT / 'tests')

# Spellings of the repository root as a search path entry. Anything resolving
# to the root puts the module clones back in front of ruff.
ROOT_SPELLINGS = frozenset({'.', './', '', str(REPO_ROOT)})

# Directories that belong to this repository rather than to a module beside it.
REPO_OWN_DIRECTORIES = frozenset({'src', 'tests', 'tools', 'docs', 'examples', 'input'})


def _read_config() -> dict:
    """Return the parsed ``pyproject.toml``."""
    return tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


@cache
def _imported_root_names() -> frozenset[str]:
    """Return every top-level module name the repository imports.

    Imports nested in functions and classes are included. ruff sorts a nested
    import block exactly as it sorts one at module scope, so a nested import is
    misread in the same way, and the defect this file guards against was a
    nested import. Reading the whole source tree costs about a second, so the
    result is cached for the file.
    """
    names: set[str] = set()
    for root in SCANNED_ROOTS:
        for path in root.rglob('*.py'):
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - none today
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(alias.name.split('.', 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names.add(node.module.split('.', 1)[0])
    return frozenset(names)


@cache
def _cloned_module_directories() -> frozenset[str]:
    """Return the root directories that are clones of a module, lowercased.

    A clone carries its own ``.git``; this repository's own directories do not.
    Empty on a machine that installed no modules.
    """
    found = set()
    for path in REPO_ROOT.iterdir():
        if not path.is_dir() or path.name in REPO_OWN_DIRECTORIES:
            continue
        if (path / '.git').exists():
            found.add(path.name.lower())
    return frozenset(found)


def _declared_module_names() -> set[str]:
    """Return the ecosystem module names, lowercased.

    Two sources, both read from the configuration rather than the filesystem,
    so this says which modules can stand at the root on any machine, including
    one with none installed: the modules pinned for a clone step, and the
    ecosystem distributions, whose module is the distribution name without its
    ``fwl-`` prefix. A stripped name that belongs to the standard library is
    dropped, since ``fwl-io`` provides ``fwl_io`` rather than ``io``.
    """
    config = _read_config()
    names = {name.lower() for name in config['tool']['proteus']['modules']}

    requirements = list(config['project'].get('dependencies', []))
    for extra in config['project'].get('optional-dependencies', {}).values():
        requirements.extend(extra)
    for requirement in requirements:
        distribution = re.split(r'[<>=!~\[; ]', requirement.strip())[0].lower()
        if distribution.startswith('fwl-'):
            module = distribution[len('fwl-') :]
            if module not in sys.stdlib_module_names:
                names.add(module)
    return names


def test_the_repository_root_is_not_searched_for_first_party_imports():
    """Import grouping is decided from the source tree alone.

    Contract clause: the root holds module clones, so searching it for
    first-party names makes an ecosystem module read as part of this project.
    The formatter then rewrites import groups on a machine that has the module
    and the checks reject the result on a machine that does not.

    Verifies:
    - A search path is configured at all, rather than left at the default,
      which searches the root.
    - No entry on it resolves to the repository root, under any of the
      spellings that would name it.
    - Only the source tree is searched, so a directory added at the root later
      cannot change how anything is sorted.
    """
    ruff = _read_config()['tool']['ruff']

    assert 'src' in ruff, (
        'ruff has no src setting, so it falls back to searching the repository '
        'root, where the ecosystem module clones are installed'
    )

    configured = [str(entry) for entry in ruff['src']]
    on_the_root = sorted(entry for entry in configured if entry.strip() in ROOT_SPELLINGS)
    assert not on_the_root, (
        f'{on_the_root} put the repository root on the import search path, where the '
        f'module clones are installed. Clones present here: {sorted(_cloned_module_directories())}'
    )
    assert configured == ['src'], (
        f'the import search path is {configured} rather than the source tree alone; '
        'any other entry is another directory whose contents decide how imports sort'
    )


def test_the_modules_beside_the_source_are_the_ones_that_would_be_misread():
    """The setting above is guarding a name that really can collide.

    Contract clause: keeping the root off the search path is only worth
    asserting while a module that the repository imports can appear there. If
    that ever stops being true the guard above is inert, and this says so
    rather than leaving it passing on nothing.

    Verifies:
    - The scan reads the source, finding names imported only inside function
      bodies, which is the shape the sorting defect took.
    - Several ecosystem modules are imported by this repository, so a clone of
      any of them at the root would be misread. This reads the configuration
      rather than the filesystem and so holds where none is installed.
    - Every clone standing at the root right now is one the configuration
      already names, so the two ways of arriving at the set agree wherever both
      have something to say.
    - An ordinary dependency is not mistaken for one of those modules, so the
      reasoning separates the ecosystem from the rest of what the repository
      imports rather than sweeping both together.
    """
    imported = _imported_root_names()

    assert len(imported) > 50, (
        f'only {len(imported)} imported module names were found, so the scan is no '
        'longer reading the source and the checks here would pass on nothing'
    )
    assert 'boreas' in imported, (
        'boreas is imported inside function bodies and the scan no longer sees it, '
        'so nested imports are escaping the check'
    )

    declared = _declared_module_names()
    collidable = sorted(declared & imported)
    assert len(collidable) >= 5, (
        f'only {collidable} of the ecosystem modules are imported by this repository, '
        'so the search path guard is protecting far less than it appears to and the '
        'reasoning behind it should be revisited rather than left in place'
    )

    # Both routes to the set should agree: a clone on this machine that the
    # configuration does not name would mean one of them has gone stale.
    unaccounted = sorted(_cloned_module_directories() - declared)
    assert not unaccounted, (
        f'{unaccounted} stand at the repository root but are named nowhere in the '
        'configuration, so the set read from it no longer covers what is installed'
    )

    # A dependency that never stands at the root cannot collide, so finding one
    # here would mean the derivation has started over-reaching.
    assert not {'numpy', 'scipy', 'io'} & declared, (
        'an ordinary dependency now looks like a module that can stand at the '
        'repository root, so the derivation no longer separates the ecosystem from '
        'the rest of what the repository imports'
    )
