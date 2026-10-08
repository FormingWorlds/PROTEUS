"""Ask ruff itself how it decides that an import is first party.

``tests/test_import_sorting_config.py`` reads the configuration and reasons
about what ruff does with it, and what it concludes rests on how ruff reads a
search path: a directory standing in a searched entry names a first-party
module whether or not it holds an ``__init__.py``, a bare module file or stub
does the same, an entry reaching the root by an oblique spelling reaches it,
and the first-party list is matched as written. Those are the shapes
``_entries_holding_a_module`` and ``_entries_on_the_repository_root`` count
over there, and they are pinned against the tool here.

Three ruffs meet this repository and none is coordinated with the others. The
one that rewrites import groups belongs to the pre-commit hook and is held at a
pinned revision in ``.pre-commit-config.yaml``. The one that rejects the
rewrite belongs to the checks and is installed unpinned. The one resolved here
is whatever ``ruff`` stands on the path, which this project requires without
pinning. So what these checks read is a real ruff rather than the one doing the
rewriting, which is deliberate: reaching for the hook's binary would tie this
file to the hook's internals.

They belong to the nightly tier rather than the pull request checks, because
they carry a real binary. A change in how ruff reads a search path therefore
surfaces within a day of the release that brings it rather than on the pull
request that installs it.

Each check builds its own project, so nothing it reports depends on which
modules are installed. That project carries its own ``pyproject.toml`` above
the probe, which is what ruff reads, and the environment handed to it is
stripped of the home and configuration paths as well, so a setting belonging to
whoever runs the checks cannot reach the probe. Whether a directory is matched
with regard to case is left out: ruff looks the name up as a path, so the
answer follows the filesystem, and an assertion about a capitalised directory
would hold on a case-folding machine and fail on a case-sensitive one.

They fail rather than skip when ruff is absent, since it is a requirement of
this project rather than an optional tool, and a skipped check and a passing
one read the same in a run log.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
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

    The home and configuration paths are pointed inside the project, so the
    only settings ruff can reach are the ones written here.
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
    sandboxed = project / 'settings_of_no_one'
    completed = subprocess.run(
        [RUFF, 'check', '--no-cache', '--fix', '--quiet', str(probe)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, 'HOME': str(sandboxed), 'XDG_CONFIG_HOME': str(sandboxed)},
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
    - ruff sorts the probe at all, under every search path read, which is what
      says the groupings below describe its reading rather than an untouched
      file.
    - With the root searched, the module beside the source joins the group that
      holds the package standing inside it.
    - With the source tree alone searched, it does not.
    - An entry that arrives at the root by climbing back out of the source tree
      reaches it as surely as the bare dot does, which is the reading the
      configuration check takes when it resolves an entry rather than comparing
      how it is spelled.
    """
    _require_ruff()
    probe = _build_probe_project(tmp_path)

    searched, searched_report = _sorted_probe(tmp_path, probe, ['.', 'src'])
    _assert_ruff_sorted(searched, searched_report, 'the root searched')
    source_only, source_report = _sorted_probe(tmp_path, probe, ['src'])
    _assert_ruff_sorted(source_only, source_report, 'the source tree alone searched')
    oblique, oblique_report = _sorted_probe(tmp_path, probe, ['src/..', 'src'])
    _assert_ruff_sorted(oblique, oblique_report, 'the root reached obliquely')

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
    assert 'import proteus' in _block_holding(oblique, 'mors'), (
        'an entry climbing back out of the source tree no longer reaches the repository '
        'root, so resolving an entry rather than reading its spelling reports a hazard '
        'ruff would not act on'
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
    - A directory carrying no ``__init__.py`` does the same, which is the
      breadth the configuration check takes when it counts every directory
      standing in an entry rather than only the ones laid out as packages.
    - None of them is read that way once the directory holding them is off the
      search path, so what carries the name is the entry being searched rather
      than something else about the project.
    """
    _require_ruff()
    probe = _build_probe_project(tmp_path)
    (tmp_path / 'filemodule.py').write_text('', encoding='utf-8')
    (tmp_path / 'stubmodule.pyi').write_text('value: int\n', encoding='utf-8')
    (tmp_path / 'baredirectory').mkdir()
    source = (
        'import baredirectory\nimport filemodule\nimport os\n'
        'import proteus\nimport stubmodule\n'
    )

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
    assert 'import proteus' in _block_holding(searched, 'baredirectory'), (
        'a directory carrying no __init__.py in a searched directory is no longer read '
        'as part of this project, so counting every directory rather than the ones laid '
        'out as packages reports a hazard ruff would not act on'
    )
    assert 'import proteus' not in _block_holding(source_only, 'filemodule'), (
        'a module file outside every search path entry is still read as part of this '
        'project, so which directories are searched has stopped deciding the question'
    )
