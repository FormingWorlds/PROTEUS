"""Ask ruff itself how it decides that an import is first party.

``tests/test_import_sorting_config.py`` reads the configuration and reasons
about what ruff does with it, and everything it concludes rests on three
behaviours of the tool: a directory standing in a search path entry names a
first-party module, a bare module file does the same, and the first-party list
is matched as written. A ruff release that changed any of them would leave
every assertion over there green while the repository broke in the way that
file exists to prevent.

The checks here build a project for the question rather than reading the one
around them, so nothing they report depends on which modules happen to be
installed. They run the real binary, which puts them in the tier that does, and
they fail rather than skip when it is absent: ruff is a requirement of this
project rather than an optional tool, and a skipped check and a passing one
read the same in a run log.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

RUFF = shutil.which('ruff')

PROBE = 'import mors\nimport os\nimport proteus\nimport requests\n'


def _require_ruff() -> None:
    """Fail with the reason rather than skipping when ruff is not installed."""
    assert RUFF is not None, (
        'ruff is not on PATH. It is a requirement of this project rather than an '
        'optional tool, so its absence means the environment is incomplete; these '
        'checks fail here rather than skipping, because a skip and a pass read the '
        'same in a run log and these are the only checks that read the real tool'
    )


def _build_probe_project(root: Path) -> Path:
    """Lay out a project holding a module beside the source and one inside it."""
    (root / 'mors').mkdir()
    (root / 'mors' / '__init__.py').write_text('', encoding='utf-8')
    package = root / 'src' / 'proteus'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('', encoding='utf-8')
    return package / 'probe.py'


def _sorted_probe(
    project: Path,
    probe: Path,
    search_path: list[str],
    source: str = PROBE,
    first_party: tuple[str, ...] = (),
) -> tuple[list[list[str]], str]:
    """Return the groups ruff sorts a probe into, and whatever it reported.

    The configuration is written fresh each call, so two calls differ only in
    what they are given. ruff sorts the probe in place and the blank lines it
    leaves behind are the group boundaries. Anything it wrote to its error
    stream comes back too, so a run that failed says why rather than arriving
    as an unexplained grouping.
    """
    entries = ', '.join(f'"{entry}"' for entry in search_path)
    isort = ''
    if first_party:
        names = ', '.join(f'"{name}"' for name in first_party)
        isort = f'[tool.ruff.lint.isort]\nknown-first-party = [{names}]\n'
    (project / 'pyproject.toml').write_text(
        '[project]\nname = "probe"\nversion = "0"\n'
        f'[tool.ruff]\nsrc = [{entries}]\n'
        f'[tool.ruff.lint]\nselect = ["I"]\n{isort}',
        encoding='utf-8',
    )
    probe.write_text(source, encoding='utf-8')
    completed = subprocess.run(
        [RUFF, 'check', '--no-cache', '--fix', '--quiet', str(probe)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in probe.read_text(encoding='utf-8').splitlines():
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks, completed.stderr.strip()


def _block_holding(blocks: list[list[str]], name: str) -> list[str]:
    """Return the group holding ``name``, or an empty group if none does."""
    return next((block for block in blocks if f'import {name}' in block), [])


def _assert_ruff_sorted(blocks: list[list[str]], reported: str, label: str) -> None:
    """Fail unless ruff rearranged the probe, so a grouping below means something."""
    assert _block_holding(blocks, 'os') == ['import os'], (
        f'with {label}, ruff left the standard library import beside the others, so it '
        f'did not sort the probe and the grouping read from it says nothing. Groups: '
        f'{blocks}. ruff reported: {reported or "nothing"}'
    )


def test_ruff_reads_a_module_beside_the_source_as_first_party(tmp_path):
    """A clone standing in a searched directory is read as part of this project.

    Contract clause: the ecosystem modules are installed beside the source, so
    a search path that reaches them makes their imports sort into the group
    that holds this project. Keeping the path off the root is what prevents it,
    and that is only true while ruff still behaves this way.

    Verifies:
    - ruff sorts the probe at all, under both search paths, which is what says
      the groupings below describe its reading rather than an untouched file.
    - With the root searched, the module beside the source joins the group that
      holds the package standing inside it.
    - With the source tree alone searched, it does not.
    """
    _require_ruff()
    probe = _build_probe_project(tmp_path)

    searched, searched_report = _sorted_probe(tmp_path, probe, ['.', 'src'])
    _assert_ruff_sorted(searched, searched_report, 'the root searched')
    source_only, source_report = _sorted_probe(tmp_path, probe, ['src'])
    _assert_ruff_sorted(source_only, source_report, 'the source tree alone searched')

    assert 'import proteus' in _block_holding(searched, 'mors'), (
        'with the repository root on the import search path, ruff no longer sorts a '
        'module standing beside the source into this project, so the misreading the '
        'configuration guards against has stopped reproducing'
    )
    assert 'import proteus' not in _block_holding(source_only, 'mors'), (
        'with the source tree alone searched, ruff still sorts a module standing beside '
        'it into this project, so keeping the root off the search path is no longer '
        'enough to prevent the misreading'
    )


def test_ruff_matches_the_first_party_list_as_written(tmp_path):
    """The first-party list is compared by spelling, not by fold.

    Contract clause: the configuration check reads ``known-first-party`` and
    reports an ecosystem module named there. It compares the names as written,
    which is only right while ruff does the same; were ruff to fold case, a
    module named in another spelling would sort with this project and go
    unreported.

    Verifies:
    - A name spelled exactly as the imported module moves that import into the
      group holding this project.
    - The same name in another case leaves it where it was, so the spelling is
      what decides and a fold would be reporting something that changes
      nothing.
    """
    _require_ruff()
    probe = _build_probe_project(tmp_path)

    exact, exact_report = _sorted_probe(tmp_path, probe, ['src'], first_party=('mors',))
    _assert_ruff_sorted(exact, exact_report, 'the module named as written')
    folded, folded_report = _sorted_probe(tmp_path, probe, ['src'], first_party=('Mors',))
    _assert_ruff_sorted(folded, folded_report, 'the module named in another case')

    assert 'import proteus' in _block_holding(exact, 'mors'), (
        'naming a module in known-first-party no longer sorts its imports with this '
        'project, so the configuration check is reading a setting that has stopped '
        'deciding anything'
    )
    assert 'import proteus' not in _block_holding(folded, 'mors'), (
        'ruff now matches known-first-party without regard to case, so the configuration '
        'check has to fold the names it reads or it will pass over a module named in a '
        'spelling that does sort with this project'
    )


def test_ruff_reads_a_bare_module_file_as_first_party(tmp_path):
    """A module standing as a single file counts as one standing as a directory.

    Contract clause: the check that looks for a module inside a search path
    entry counts a directory, a ``.py`` file and a ``.pyi`` stub alike. That is
    only worth doing while ruff reads all three as module names.

    Verifies:
    - A bare ``.py`` file in a searched directory sorts its import into the
      group that holds this project.
    - A bare ``.pyi`` stub does the same, so the stub shape is not a gap.
    - Neither is read that way once the directory holding them is off the
      search path, so the file is what carries the name rather than something
      else about the project.
    """
    _require_ruff()
    probe = _build_probe_project(tmp_path)
    (tmp_path / 'filemodule.py').write_text('', encoding='utf-8')
    (tmp_path / 'stubmodule.pyi').write_text('value: int\n', encoding='utf-8')
    source = 'import filemodule\nimport os\nimport proteus\nimport stubmodule\n'

    searched, searched_report = _sorted_probe(tmp_path, probe, ['.', 'src'], source=source)
    _assert_ruff_sorted(searched, searched_report, 'the root searched')
    source_only, source_report = _sorted_probe(tmp_path, probe, ['src'], source=source)
    _assert_ruff_sorted(source_only, source_report, 'the source tree alone searched')

    assert 'import proteus' in _block_holding(searched, 'filemodule'), (
        'a module standing as a single file in a searched directory is no longer read as '
        'part of this project, so looking for one is guarding against nothing'
    )
    assert 'import proteus' in _block_holding(searched, 'stubmodule'), (
        'a module standing as a stub file in a searched directory is no longer read as '
        'part of this project, so counting stubs is guarding against nothing'
    )
    assert 'import proteus' not in _block_holding(source_only, 'filemodule'), (
        'a module file outside every search path entry is still read as part of this '
        'project, so which directories are searched has stopped deciding the question'
    )
