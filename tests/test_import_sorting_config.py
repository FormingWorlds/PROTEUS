"""Guard on where ruff looks to decide that an import is first party.

The ecosystem modules are installed as clones beside ``src/`` at the repository
root. Whatever ruff treats as an import search path is also where it looks for
first-party names, so with the root searched, a clone directory carrying its
own module name makes that module read as part of this project and its imports
sort into the group that holds ``proteus``. Naming a module in the first-party
list of the isort settings arrives at the same place by a shorter route.

The damage is not symmetric between a contributor's machine and the checks,
because only the machine with the module installed carries the clone. The lint
hook runs ``ruff check --fix``, so on that machine it rewrites the import
groups of any file it touches, and the rewrite it commits is what the checks
then reject. Keeping the root off the search path closes both halves at once,
for every module rather than one name at a time.

The checks below read the configuration and the source, so they hold on a
machine that installed no modules at all, which is the case the checks
themselves run in. Each derivation carries a canary as well as an assertion, so
a check that stops reaching its subject fails rather than passing on nothing.

What they conclude rests on how ruff reads a search path, which is described
here rather than consulted. ``tests/test_import_sorting_ruff.py`` puts that
description to the tool itself, on a project built for the question, and runs
in the nightly tier because it carries a real binary. The shapes
``_entries_holding_a_module`` counts and the spellings
``_entries_on_the_repository_root`` resolves are the ones pinned over there, so
a change to either belongs in both files.

Whether a clone at the root is read as a module follows the filesystem, since
ruff looks the name up as a path: a case-folding filesystem answers to
``MORS/`` for ``import mors`` and a case-sensitive one does not. The clone names
are compared folded here, which matches the machines where the rewrite happens
and is wider than the machines where it does not. The first-party list is a
different matter and is compared as written, because ruff matches those names
itself rather than through the filesystem.

``docs/Explanations/test_framework.md`` asks every check for an edge case and a
path through the error contract, in wording written for the physics modules.
Read here, the edge cases are the limits a configuration and a scan can reach:
a spelling of the root nobody wrote down, an import that does not begin its
line, a candidate set with nothing in it. The contract a failure reports is the
assertion message, which says what the setting defends rather than only that it
changed.

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

# The directories holding Python this repository lints. The hook that rewrites
# import groups reaches all of them rather than the two the checks run over, so
# an ecosystem import added to any of them is one the scan should see.
SCANNED_ROOTS = (REPO_ROOT / 'src', REPO_ROOT / 'tests', REPO_ROOT / 'tools')

# Both suffixes the lint hook takes, since a stub carries imports and is sorted
# exactly as a module is.
PYTHON_SUFFIXES = ('*.py', '*.pyi')

# Directories that belong to this repository rather than to a module beside it.
REPO_OWN_DIRECTORIES = frozenset({'src', 'tests', 'tools', 'docs', 'examples', 'input'})

# Ecosystem modules this repository imports only from inside a function body.
# Nothing names them at the start of a line, so they are the witnesses that a
# scan reading module scope alone would lose.
NESTED_ONLY_IMPORTS = frozenset({'mors', 'zephyrus'})

# Ecosystem modules the configuration names only through an optional extra.
# Neither the pinned clone list nor the base requirements reach them, so they
# are what says that reading the extras is still part of the derivation.
OPTIONAL_EXTRA_ONLY = frozenset({'vulcan'})


def _read_config() -> dict:
    """Return the parsed ``pyproject.toml``."""
    return tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


@cache
def _python_files() -> tuple[Path, ...]:
    """Return every Python file under the scanned roots and at the root itself."""
    found = {
        path
        for root in SCANNED_ROOTS
        for suffix in PYTHON_SUFFIXES
        for path in root.rglob(suffix)
    }
    found.update(path for suffix in PYTHON_SUFFIXES for path in REPO_ROOT.glob(suffix))
    return tuple(sorted(found))


@cache
def _candidate_pattern(candidates: frozenset[str]) -> re.Pattern[str] | None:
    """Return a pattern selecting the files worth parsing, or ``None`` for none.

    An import keyword and the name it brings in stand on one line, so a line
    holding both is the narrowest thing that can be required without losing an
    import: a clause header, a semicolon or indentation in front of the keyword
    all leave the pair together. Callers join continued lines first, since a
    backslash can carry the name onto the next line. Matching without case
    keeps the filter wider than the parse behind it. An empty candidate set
    would alternate to a pattern matching at every position, so it gets no
    pattern at all.
    """
    if not candidates:
        return None
    alternation = '|'.join(re.escape(name) for name in sorted(candidates))
    return re.compile(rf'(?m)^.*\b(?:import|from)\b.*\b(?:{alternation})\b', re.IGNORECASE)


def _imported_names_in_source(source: str, candidates: frozenset[str]) -> frozenset[str]:
    """Return the members of ``candidates`` that ``source`` imports.

    A file whose lines never hold an import keyword beside a candidate name
    cannot contribute one and is not parsed. What survives the filter is read
    with ``ast`` rather than by pattern, so a name written in a comment or a
    string is not mistaken for an import. ``ast`` also sees imports nested in
    functions and classes: ruff sorts a nested import block exactly as it sorts
    one at module scope, and the defect this file guards against was nested.

    Line endings are normalised before the continued lines are joined, so the
    filter reads a source handed to it as text the same way it reads one that
    arrived through a file.
    """
    pattern = _candidate_pattern(candidates)
    joined = source.replace('\r\n', '\n').replace('\\\n', ' ')
    if pattern is None or not pattern.search(joined):
        return frozenset()
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - none today
        return frozenset()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.', 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split('.', 1)[0])
    return frozenset(names & candidates)


@cache
def _python_sources() -> tuple[str, ...]:
    """Return the text of every Python file under the scanned roots.

    Held for the file, since several checks ask the same tree about different
    names and the reading is the part none of them need to repeat.
    """
    sources = []
    for path in _python_files():
        try:
            sources.append(path.read_text(encoding='utf-8'))
        except UnicodeDecodeError:  # pragma: no cover - none today
            continue
    return tuple(sources)


@cache
def _imported_module_names(candidates: frozenset[str]) -> frozenset[str]:
    """Return the members of ``candidates`` the repository imports anywhere.

    Two costs sit behind this. The filter runs over every source and stands
    whatever is asked for; the parse runs only on what the filter keeps, and
    that is where asking for fewer names pays, since the two witnesses below
    reach about a twentieth of the tree where the whole declared set reaches a
    quarter of it. Asking for a set the configuration already accounts for,
    which is the usual case for the staleness check, builds no filter and
    parses nothing.

    The result is a couple of tenths of a second, over the 100 ms that
    ``docs/How-to/testing.md`` sets for the unit marker and well inside the
    ``timeout`` ceiling above. Answering what the repository imports means
    reading the repository, so the cost is taken rather than traded for a
    narrower question.
    """
    names: set[str] = set()
    for source in _python_sources():
        names |= _imported_names_in_source(source, candidates)
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


def _entries_on_the_repository_root(entries: list[str]) -> list[str]:
    """Return the search path entries that lead to the repository root.

    Each entry is resolved against the directory holding the configuration, the
    way ruff reads it, so entries are separated by where they arrive rather
    than by how they are written and a spelling nobody listed is caught with
    the rest.
    """
    return sorted(entry for entry in entries if (REPO_ROOT / entry).resolve() == REPO_ROOT)


def _entries_holding_a_module(entries: list[str], modules: frozenset[str]) -> list[str]:
    """Return the search path entries that a module could stand in.

    This is what makes a clone read as first party: not that an entry is the
    root, but that a module stands inside it. ruff reads a directory, a bare
    ``.py`` file and a bare ``.pyi`` stub alike as a module, so all three count.
    An entry naming a directory that is not there holds nothing and is left
    alone.
    """
    holding = []
    for entry in entries:
        directory = (REPO_ROOT / entry).resolve()
        if not directory.is_dir():
            continue
        held = {
            child.name.lower() if child.is_dir() else child.stem.lower()
            for child in directory.iterdir()
            if child.is_dir() or child.suffix in {'.py', '.pyi'}
        }
        if held & modules:
            holding.append(entry)
    return sorted(holding)


def test_the_repository_root_is_not_searched_for_first_party_imports():
    """Import grouping is decided from the source tree alone.

    Contract clause: the root holds the module clones, so a search path entry
    that reaches them makes an ecosystem module read as part of this project.
    ``ruff check --fix`` then rewrites import groups on a machine that has the
    module and the checks reject the result on a machine that does not.

    Verifies:
    - A search path is configured at all, rather than left at the default,
      which searches the root.
    - No entry leads to the repository root, under any spelling of it.
    - No entry is one a module could stand in, which covers an entry that
      reaches the clones without being the root itself. An entry outside
      ``src/`` is not a defect in itself: a package kept elsewhere in the
      repository can join the search path as long as no module can stand in it.
    """
    ruff = _read_config()['tool']['ruff']

    assert 'src' in ruff, (
        'ruff has no src setting, so it falls back to searching the repository '
        'root, where the ecosystem module clones are installed'
    )

    configured = [str(entry) for entry in ruff['src']]
    on_the_root = _entries_on_the_repository_root(configured)
    assert not on_the_root, (
        f'{on_the_root} put the repository root on the import search path, where the '
        f'module clones are installed. Clones present here: {sorted(_cloned_module_directories())}'
    )

    modules = frozenset(_declared_module_names()) | _cloned_module_directories()
    holding = _entries_holding_a_module(configured, modules)
    assert not holding, (
        f'{holding} are on the import search path and an ecosystem module stands in them, '
        'so ruff reads that module as part of this project and sorts its imports into the '
        'group that holds proteus'
    )


def test_no_ecosystem_module_is_declared_first_party():
    """The isort settings do not name a module the search path keeps out.

    Contract clause: ``known-first-party`` overrides whatever the search path
    would have decided, so a module listed there sorts with ``proteus`` on
    every machine, installed or not. That is the same misreading the search
    path guard prevents, reached by a shorter route.

    The names are compared as written. ruff matches this list against the
    imported name exactly, so ``Mors`` leaves ``import mors`` where it was and
    only ``mors`` moves it, which holds whatever the filesystem does about
    case. Folding here would report a spelling that changes nothing.

    Verifies:
    - This project is named first party, so the settings are being read at the
      key that decides the question rather than at one that no longer exists.
    - No ecosystem module is named alongside it.
    """
    lint = _read_config()['tool']['ruff'].get('lint', {})
    first_party = set(lint.get('isort', {}).get('known-first-party', []))

    assert 'proteus' in first_party, (
        f'known-first-party is {sorted(first_party)} and does not name this project, so '
        'either the key moved or the setting was dropped, and the check below no longer '
        'reads what decides whether a module sorts as first party'
    )

    ecosystem = sorted(first_party & _declared_module_names())
    assert not ecosystem, (
        f'{ecosystem} are ecosystem modules named as first party, so their imports sort '
        'into the group that holds proteus even on a machine that keeps the module clones '
        'off the import search path'
    )


def test_the_repository_root_is_recognised_under_every_spelling():
    """A search path entry is judged by where it leads, not by how it reads.

    Contract clause: the entry that puts the clones in front of ruff is the one
    that arrives at the repository root, and a configuration can arrive there
    several ways. Comparing the text would leave whichever spellings nobody
    listed unguarded.

    Verifies:
    - Every spelling that arrives at the root is reported: the bare dot, a
      trailing separator, an empty entry, the absolute path, and a path that
      climbs back out of a subdirectory.
    - The source tree is not reported, so the check separates the root from an
      ordinary entry rather than reporting whatever it is given.
    """
    spellings = ['.', './', '', str(REPO_ROOT), 'src/..']

    assert _entries_on_the_repository_root(spellings) == sorted(spellings), (
        'a spelling of the repository root went unreported, so a configuration writing '
        'the root that way would put the module clones on the import search path'
    )
    assert _entries_on_the_repository_root(['src', 'src/proteus']) == [], (
        'an entry inside the source tree was reported as the repository root, so the '
        'check no longer separates the two and would fail on a configuration that is fine'
    )


def test_a_path_entry_a_module_could_stand_in_is_reported(tmp_path):
    """A module standing inside a path entry is what makes it first party.

    Contract clause: ruff reads what sits inside a search path entry as module
    names, so an entry is a hazard when a module can stand in it, whatever the
    entry itself is called.

    Verifies:
    - An entry holding a module as a directory is reported, and the match
      ignores case, since the clone directories are capitalised and the module
      names are not.
    - An entry holding a module as a bare ``.py`` file is reported too, and so
      is one holding it as a bare ``.pyi`` stub, since ruff reads either as a
      module name exactly as it reads a package directory.
    - An entry holding unrelated directories and files is not reported, so a
      path entry is not condemned for existing.
    - An entry naming a directory that is not there is not reported, so a
      configuration written ahead of a directory does not fail on its absence.
    """
    packaged = tmp_path / 'with_package'
    (packaged / 'AGNI').mkdir(parents=True)
    single = tmp_path / 'with_single_file'
    single.mkdir()
    (single / 'agni.py').write_text('', encoding='utf-8')
    stubbed = tmp_path / 'with_stub'
    stubbed.mkdir()
    (stubbed / 'agni.pyi').write_text('', encoding='utf-8')
    plain = tmp_path / 'without_module'
    (plain / 'notes').mkdir(parents=True)
    (plain / 'notes.py').write_text('', encoding='utf-8')

    modules = frozenset({'agni'})
    assert _entries_holding_a_module([str(packaged)], modules) == [str(packaged)], (
        'a directory named like an ecosystem module was not seen inside a search path '
        'entry, so the guard above would pass on a configuration that reaches the clones'
    )
    assert _entries_holding_a_module([str(single)], modules) == [str(single)], (
        'a module standing as a single file was not seen inside a search path entry, and '
        'ruff reads that as a first-party name exactly as it reads a package directory'
    )
    assert _entries_holding_a_module([str(stubbed)], modules) == [str(stubbed)], (
        'a module standing as a stub file was not seen inside a search path entry, and '
        'ruff reads a bare .pyi as a first-party name just as it reads a .py'
    )
    assert _entries_holding_a_module([str(plain)], modules) == [], (
        'a search path entry holding nothing but unrelated names was reported, so the '
        'check condemns any entry rather than the ones a module can stand in'
    )
    assert _entries_holding_a_module([str(tmp_path / 'absent')], modules) == [], (
        'an entry naming a directory that does not exist was reported, so the check reads '
        'the absence of a directory as a module standing in it'
    )


def test_an_import_away_from_the_start_of_its_line_is_counted():
    """A module imported from inside a compound statement counts as imported.

    Contract clause: the scan skips a file whose lines never hold an import
    keyword beside a candidate name. A statement following a clause header or a
    semicolon is an import like any other and ruff sorts its group the same
    way, so a skip that hid one would leave every derivation below reading a
    smaller repository than the real one.

    Verifies:
    - Each shape that keeps an import away from the start of its line is found:
      after a clause header, in an else branch, and after a semicolon.
    - A name carried onto the next line by a backslash is found, so continued
      lines are joined before the filter runs, whichever ending the source
      separates its lines with.
    - A name written in a comment or inside a string is not counted, so the
      filter still leaves the decision to the parse behind it.
    """
    candidates = frozenset({'mors', 'zephyrus'})

    compound = {
        'clause header': 'try: import mors\nexcept ImportError: mors = None\n',
        'else branch': 'if True:\n    pass\nelse: import zephyrus\n',
        'after a semicolon': 'started = True; import mors\n',
        'continued line': 'import \\\n    mors\n',
        'continued line, other endings': 'import \\\r\n    mors\r\n',
    }
    found = {
        label: _imported_names_in_source(src, candidates) for label, src in compound.items()
    }
    missed = sorted(label for label, names in found.items() if not names)
    assert not missed, (
        f'an import reached {missed} was not counted, so a file whose only import of an '
        'ecosystem module takes that shape reads as importing nothing'
    )

    quoted = '# import mors for the stellar track\nEXAMPLE = "from zephyrus import escape"\n'
    assert _imported_names_in_source(quoted, candidates) == frozenset(), (
        'a name written in a comment or a string was counted as an import, so the scan '
        'reports modules this repository does not import'
    )


def test_a_scan_with_no_candidate_names_looks_for_nothing(monkeypatch):
    """An empty candidate set reads as no names to look for, not as all of them.

    Contract clause: the filter alternates the candidate names, and an empty
    alternation matches at every position, so an empty set would select the
    whole tree to parse. The scan intersects what it finds with the candidates,
    so the answer that comes back is empty either way and only the parse that
    never happens shows the guard doing anything.

    Verifies:
    - No pattern is built for an empty set, so the indiscriminate one is never
      reached, while a one-name set still builds one that selects a file
      importing that name.
    - A source that plainly imports a candidate is never handed to the parser
      under an empty candidate set, with every parse recorded to say so.
    - The same source is parsed and found once the name is a candidate, so the
      silence above is the guard rather than a scan that reads nothing at all.
    """
    assert _candidate_pattern(frozenset()) is None, (
        'a pattern was built from an empty candidate set, and an empty alternation '
        'matches everywhere, so the scan would parse the whole tree to find nothing'
    )
    probe = _candidate_pattern(frozenset({'mors'}))
    assert probe is not None and probe.search('import mors'), (
        'the filter built for a single name no longer selects a file importing it, so '
        'the empty result above says nothing about the guard'
    )

    parsed: list[str] = []
    real_parse = ast.parse

    def record(source, *args, **kwargs):
        parsed.append(source)
        return real_parse(source, *args, **kwargs)

    monkeypatch.setattr(ast, 'parse', record)

    witness = sorted(NESTED_ONLY_IMPORTS)[0]
    source = f'import {witness}\n'
    assert _imported_names_in_source(source, frozenset()) == frozenset()
    assert parsed == [], (
        'the scan parsed a file although it had been given no names to look for, so the '
        'empty-set guard no longer stands in front of a pattern that matches everywhere'
    )

    assert _imported_names_in_source(source, frozenset({witness})) == frozenset({witness})
    assert parsed == [source], (
        f'the same source was not parsed with {witness!r} among the candidates, so the '
        'refusal above comes from a scan that parses nothing rather than from the guard'
    )


def test_the_modules_beside_the_source_are_the_ones_that_would_be_misread():
    """The setting above is guarding a name that really can collide.

    Contract clause: keeping the root off the search path is only worth
    asserting while a module that the repository imports can appear there. If
    that ever stops being true the guard above is inert, and this says so
    rather than leaving it passing on nothing.

    Verifies:
    - The walk reaches the source tree, and the scan finds both names imported
      only inside function bodies, which is the shape the sorting defect took.
    - No clone standing at the root that this repository imports is missing
      from the configuration, so the two ways of arriving at the set agree
      wherever both have something to say. A directory nothing imports cannot
      change how imports sort and so is not the subject here.
    - Those two names are themselves modules the configuration names, so what
      the setting above covers is a family rather than a single import. This
      reads the configuration rather than the filesystem and so holds where no
      module is installed.
    - An ordinary dependency is not mistaken for one of those modules, so the
      reasoning separates the ecosystem from the rest of what the repository
      imports rather than sweeping both together.
    """
    declared = _declared_module_names()
    cloned = _cloned_module_directories()

    scanned = _python_files()
    assert len(scanned) > 100, (
        f'only {len(scanned)} Python files were found under the source tree, so the '
        'walk is no longer reaching it and the checks here would pass on nothing'
    )
    # Nothing imports these at module scope, so they are what says the scan
    # reaches an import nested in a function body rather than only the ones
    # standing at the start of a line.
    missing = sorted(NESTED_ONLY_IMPORTS - _imported_module_names(NESTED_ONLY_IMPORTS))
    assert not missing, (
        f'{missing} are imported only inside function bodies and the scan no longer '
        'sees them, so either nested imports are escaping it or those imports were '
        'renamed or dropped; which of the two decides whether anything below holds'
    )

    # Both routes to the set should agree: a clone this repository imports that
    # the configuration does not name would mean one of them has gone stale.
    # Only the names the configuration leaves unaccounted for are looked for, so
    # a fork or a working tree kept beside the clones costs nothing and, being
    # imported by nothing, decides nothing.
    unaccounted = sorted(_imported_module_names(frozenset(cloned - declared)))
    assert not unaccounted, (
        f'{unaccounted} stand at the repository root and are imported here but are '
        'named nowhere in the configuration, so the set read from it no longer covers '
        'what is installed'
    )

    uncovered = sorted(NESTED_ONLY_IMPORTS - declared)
    assert not uncovered, (
        f'{uncovered} are what says the scan reaches a nested import, but the '
        'configuration has stopped naming them as ecosystem modules, so they no longer '
        'say that the setting above covers a family rather than a single import'
    )

    # An extra is the only route to these, so they say the requirements are read
    # past the ones installed by default rather than stopping at them.
    from_extras = sorted(OPTIONAL_EXTRA_ONLY - declared)
    assert not from_extras, (
        f'{from_extras} are named only in an optional extra and the derivation no longer '
        'reaches them, so a module installed through an extra can stand at the root '
        'without the configuration accounting for it'
    )

    # A dependency that never stands at the root cannot collide, so finding one
    # here would mean the derivation has started over-reaching.
    assert not {'numpy', 'scipy', 'io'} & declared, (
        'an ordinary dependency now looks like a module that can stand at the '
        'repository root, so the derivation no longer separates the ecosystem from '
        'the rest of what the repository imports'
    )
