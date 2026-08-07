"""Guard on where ruff looks to decide that an import is first party.

The ecosystem modules are installed as clones beside ``src/`` at the repository
root. Whatever ruff treats as an import search path is also where it looks for
first-party names, so with the root searched, a clone directory carrying its
own module name makes that module read as part of this project and its imports
sort into the group that holds ``proteus``.

The damage is not symmetric between a contributor's machine and the checks,
because only the machine with the module installed carries the clone. The lint
hook runs ``ruff check --fix``, so on that machine it rewrites the import
groups of any file it touches, and the rewrite it commits is what the checks
then reject. Keeping the root off the search path closes both halves at once,
for every module rather than one name at a time.

The checks below read the configuration and the source, so they hold on a
machine that installed no modules at all, which is the case the checks
themselves run in. Each derivation carries a canary as well as an assertion,
so a check that stops reaching its subject fails rather than passing on
nothing.

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

# Ecosystem modules this repository imports only from inside a function body.
# Nothing names them at the start of a line, so they are the witnesses that a
# scan reading module scope alone would lose.
NESTED_ONLY_IMPORTS = frozenset({'mors', 'zephyrus'})


def _read_config() -> dict:
    """Return the parsed ``pyproject.toml``."""
    return tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


@cache
def _python_files() -> tuple[Path, ...]:
    """Return every Python file under the scanned roots."""
    return tuple(sorted(path for root in SCANNED_ROOTS for path in root.rglob('*.py')))


def _import_line_pattern(candidates: frozenset[str]) -> re.Pattern[str]:
    """Return a pattern matching an import statement that names a candidate.

    Indentation is allowed, so a nested import matches as readily as one at
    module scope, and the whole statement is searched rather than the position
    after the keyword, so a name reached through a comma list or a dotted path
    still matches. Matching without case keeps the filter wider than the parse
    behind it. Callers join continued lines first, since the name of a module
    imported across a backslash stands on a line of its own.
    """
    alternation = '|'.join(re.escape(name) for name in sorted(candidates))
    return re.compile(rf'(?m)^\s*(?:import|from)\b.*\b(?:{alternation})\b', re.IGNORECASE)


@cache
def _imported_module_names(candidates: frozenset[str]) -> frozenset[str]:
    """Return the members of ``candidates`` that the repository imports.

    Only a name that can stand at the root as a clone decides how imports sort,
    so a file whose import statements never spell one cannot contribute and is
    not parsed. That leaves about a fifth of the tree, and what remains is read
    with ``ast`` rather than by pattern, so a name written in a comment or a
    docstring is not mistaken for an import. ``ast`` also sees imports nested in
    functions and classes: ruff sorts a nested import block exactly as it sorts
    one at module scope, and the defect this file guards against was nested.

    The scan costs a few tenths of a second, over the 100 ms that
    ``docs/How-to/testing.md`` sets for the unit marker and well inside the
    ``timeout`` ceiling above. What is left is the parse itself, on the files
    that really do import an ecosystem module.
    """
    if not candidates:
        return frozenset()
    pattern = _import_line_pattern(candidates)
    names: set[str] = set()
    for path in _python_files():
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:  # pragma: no cover - none today
            continue
        if not pattern.search(source.replace('\\\n', ' ')):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - none today
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split('.', 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split('.', 1)[0])
    return frozenset(names & candidates)


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
    ``ruff check --fix`` then rewrites import groups on a machine that has the
    module and the checks reject the result on a machine that does not.

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
        f'the import search path is {configured} rather than the source tree alone. '
        'Every entry on it is a directory whose contents decide how imports sort, so '
        'the path is held to one; a second entry that is genuinely wanted belongs in '
        'the same change as the contract here'
    )


def test_the_modules_beside_the_source_are_the_ones_that_would_be_misread():
    """The setting above is guarding a name that really can collide.

    Contract clause: keeping the root off the search path is only worth
    asserting while a module that the repository imports can appear there. If
    that ever stops being true the guard above is inert, and this says so
    rather than leaving it passing on nothing.

    Verifies:
    - The walk reaches the source tree, and the scan finds a name imported only
      inside function bodies, which is the shape the sorting defect took.
    - Several ecosystem modules are imported by this repository, so a clone of
      any of them at the root would be misread. This reads the configuration
      rather than the filesystem and so holds where none is installed.
    - Every clone standing at the root that this repository imports is one the
      configuration already names, so the two ways of arriving at the set agree
      wherever both have something to say. A directory nothing imports cannot
      change how imports sort and so is not the subject here.
    - An ordinary dependency is not mistaken for one of those modules, so the
      reasoning separates the ecosystem from the rest of what the repository
      imports rather than sweeping both together.
    """
    declared = _declared_module_names()
    cloned = _cloned_module_directories()
    imported = _imported_module_names(frozenset(declared | cloned))

    scanned = _python_files()
    assert len(scanned) > 100, (
        f'only {len(scanned)} Python files were found under the source tree, so the '
        'walk is no longer reaching it and the checks here would pass on nothing'
    )
    # Nothing imports these at module scope, so they are what says the scan
    # reaches an import nested in a function body rather than only the ones
    # standing at the start of a line.
    missing = sorted(NESTED_ONLY_IMPORTS - imported)
    assert not missing, (
        f'{missing} are imported only inside function bodies and the scan no longer '
        'sees them, so either nested imports are escaping it or those imports were '
        'renamed or dropped; which of the two decides whether anything below holds'
    )

    collidable = sorted(declared & imported)
    assert len(collidable) >= 5, (
        f'only {collidable} of the ecosystem modules are imported by this repository, '
        'so the search path guard is protecting far less than it appears to and the '
        'reasoning behind it should be revisited rather than left in place'
    )

    # Both routes to the set should agree: a clone this repository imports that
    # the configuration does not name would mean one of them has gone stale.
    # A root directory nothing imports is left alone, since a fork or a working
    # tree kept beside the clones cannot decide how any import sorts.
    unaccounted = sorted((cloned & imported) - declared)
    assert not unaccounted, (
        f'{unaccounted} stand at the repository root and are imported here but are '
        'named nowhere in the configuration, so the set read from it no longer covers '
        'what is installed'
    )

    # A dependency that never stands at the root cannot collide, so finding one
    # here would mean the derivation has started over-reaching.
    assert not {'numpy', 'scipy', 'io'} & declared, (
        'an ordinary dependency now looks like a module that can stand at the '
        'repository root, so the derivation no longer separates the ecosystem from '
        'the rest of what the repository imports'
    )


def test_a_scan_with_no_candidate_names_finds_nothing():
    """An empty candidate set reads as no names to look for, not as all of them.

    Contract clause: the scan builds its filter by alternating the candidate
    names, and an empty alternation matches at every position, so an empty set
    would sweep in every import in the tree rather than none. The guard returns
    ahead of that, which is the reading a configuration naming no modules
    should get.

    Verifies:
    - The empty case returns nothing, so the boundary is handled rather than
      falling through to a pattern that matches indiscriminately.
    - A single-name scan still finds that name, so the empty result above comes
      from the guard rather than from a scan that has stopped working.
    """
    assert _imported_module_names(frozenset()) == frozenset()

    witness = sorted(NESTED_ONLY_IMPORTS)[0]
    assert _imported_module_names(frozenset({witness})) == frozenset({witness})
