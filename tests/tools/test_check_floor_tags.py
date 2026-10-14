"""Unit tests for tools/check_floor_tags.sh, the release check that each module floor has one tag.

The script runs as shipped, from a scratch PROTEUS copy, against local bare
remotes that ``PROTEUS_MODULE_GIT_BASE`` points it at; no test reaches the
network. Contract checked:
- each of the 8 modules gets one line, with the tag of its floor, and the
  exit status is 0 when every floor has exactly one tag;
- a floor without a tag, a floor with two tags of its version and a remote
  that cannot be read each give exit status 1 and name the module;
- the module table of the script names the repository and the distribution
  that each setup script clones.

See also:
- docs/How-to/update_module_pins.md
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import git

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

TOOLS_DIR = Path(__file__).resolve().parents[2] / 'tools'

# Repository, distribution and floor of each module; the floors differ, so a line cannot borrow
# another module's tag.
_MODULES = [
    ('MORS', 'fwl-mors', '26.10.6'),
    ('JANUS', 'fwl-janus', '26.10.8'),
    ('CALLIOPE', 'fwl-calliope', '26.6.1'),
    ('ZEPHYRUS', 'fwl-zephyrus', '26.5.13'),
    ('aragog', 'fwl-aragog', '26.7.4'),
    ('Zalmoxis', 'fwl-zalmoxis', '26.7.17'),
    ('VULCAN', 'fwl-vulcan', '26.4.22'),
    ('Morrigan', 'fwl-morrigan', '26.9.1'),
]


def _padded(floor: str) -> str:
    """The zero-padded tag of a floor, for example ``26.06.01`` for ``26.6.1``."""
    return '.'.join(f'{int(part):02d}' for part in floor.split('.'))


@pytest.fixture
def site(tmp_path) -> dict:
    """A PROTEUS copy with the script, the helper library and the pin reader below a
    pyproject.toml with the 8 floors (2 as optional dependencies), and a local bare remote per
    module with an annotated tag of its floor on the first of 2 commits and a newer tag on the
    tip."""
    root = tmp_path / 'PROTEUS'
    (root / 'tools').mkdir(parents=True)
    for name in ('check_floor_tags.sh', '_get_common.sh', '_module_pins.py'):
        shutil.copy2(TOOLS_DIR / name, root / 'tools' / name)
    required = ', '.join(f'"{d}>={f}"' for _, d, f in _MODULES[:6])
    (root / 'pyproject.toml').write_text(
        f'[project]\nname = "fixture"\ndependencies = [{required}]\n'
        '[project.optional-dependencies]\n'
        f'vulcan = ["fwl-vulcan>={_MODULES[6][2]}"]\nmorrigan = ["fwl-morrigan>={_MODULES[7][2]}"]\n'
    )
    remotes = tmp_path / 'remotes' / 'FormingWorlds'
    remotes.mkdir(parents=True)
    for repo, _, floor in _MODULES:
        work = tmp_path / 'work' / repo
        work.mkdir(parents=True)
        git(work, 'init', '-q', '-b', 'main')
        for i in range(2):
            (work / 'version.txt').write_text(f'{i}\n')
            git(work, 'add', 'version.txt')
            git(work, 'commit', '-q', '-m', f'commit {i}')
            if i == 0:
                git(work, 'tag', '-a', '-m', 'release', _padded(floor))
        git(work, 'tag', '27.01.01')
        git(tmp_path, 'clone', '-q', '--bare', str(work), str(remotes / f'{repo}.git'))
    env = {**os.environ, 'PROTEUS_MODULE_GIT_BASE': f'{tmp_path / "remotes"}/'}
    env.pop('GIT_DIR', None)
    return {'root': root, 'remotes': remotes, 'env': env}


def _run(site: dict) -> subprocess.CompletedProcess:
    """Run the shipped check from the scratch copy, from outside it."""
    return subprocess.run(
        ['bash', str(site['root'] / 'tools' / 'check_floor_tags.sh')],
        cwd=site['root'].parent,
        env=site['env'],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_every_floor_with_one_tag_passes_with_one_line_per_module(site):
    """With one tag of each floor (annotated, so ls-remote also lists its ^{} line) and a newer
    tag on every tip, the check exits 0 and prints 8 lines, one per module in the order of the
    table, each with its own floor and its zero-padded tag, not the newer one."""
    result = _run(site)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f'{dist} floor {floor}: tag {_padded(floor)}' for _, dist, floor in _MODULES
    ]
    assert '27.01.01' not in result.stdout
    assert '^{}' not in result.stdout


def test_floor_without_a_tag_fails_and_names_the_module(site):
    """A JANUS remote without the tag of its floor makes the check exit 1 with the line of
    JANUS and the cause from the pin reader; the other 7 modules keep their tag line."""
    git(site['remotes'] / 'JANUS.git', 'tag', '-d', '26.10.08')
    result = _run(site)
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 8
    janus = [ln for ln in lines if ln.startswith('fwl-janus ')]
    assert janus == [
        "fwl-janus floor 26.10.8: NO SINGLE TAG ('fwl-janus' floor 26.10.8: no tag matches it;"
        ' release tags seen: 1, highest: 27.01.01)'
    ]
    assert sum(': tag ' in ln for ln in lines) == 7


def test_two_tags_of_one_version_fail(site):
    """A MORS remote with the tags 26.10.6 and 26.10.06, which name one version, has no single
    tag of its floor: exit 1, and the line names both tags."""
    remote = site['remotes'] / 'MORS.git'
    git(remote, 'tag', '26.10.6', '26.10.06^{commit}')
    result = _run(site)
    assert result.returncode == 1
    mors = [ln for ln in result.stdout.splitlines() if ln.startswith('fwl-mors ')]
    assert len(mors) == 1
    assert mors[0].startswith('fwl-mors floor 26.10.6: NO SINGLE TAG (')
    assert '2 tags match it: 26.10.06, 26.10.6' in mors[0]


def test_unreachable_remote_fails_and_names_the_module(site):
    """A Zalmoxis remote that is gone makes the check exit 1 with the line of Zalmoxis, which
    names the URL it could not read; the other modules are still checked."""
    remote = site['remotes'] / 'Zalmoxis.git'
    remote.rename(remote.with_name('Zalmoxis.moved'))
    result = _run(site)
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    zalmoxis = [ln for ln in lines if ln.startswith('fwl-zalmoxis ')]
    assert len(zalmoxis) == 1
    assert zalmoxis[0].startswith(
        f'fwl-zalmoxis floor 26.7.17: NO SINGLE TAG (cannot read the tags of {remote}:'
    )
    assert sum(': tag ' in ln for ln in lines) == 7


def test_module_table_matches_the_setup_scripts():
    """Each repository of the table is the one its setup script clones under FormingWorlds, with
    the distribution it installs; the 4 of get_python_module.sh are those it accepts."""
    text = (TOOLS_DIR / 'check_floor_tags.sh').read_text()
    block = re.search(r'^modules="([^"]*)"', text, re.MULTILINE | re.DOTALL).group(1)
    table = dict(entry.split(':') for entry in block.split())
    assert table == {repo: dist for repo, dist, _ in _MODULES}
    accepted = re.search(
        r'^modules="([^"]*)"$', (TOOLS_DIR / 'get_python_module.sh').read_text(), re.MULTILINE
    ).group(1)
    shared = {repo for repo in table if repo not in accepted.split()}
    assert shared == {'aragog', 'Zalmoxis', 'VULCAN', 'Morrigan'}
    for repo in shared:
        script = (TOOLS_DIR / f'get_{repo.lower()}.sh').read_text()
        assert f'https://github.com/FormingWorlds/{repo}.git' in script, repo
        assert re.search(rf'checkout_(floor_tag|paired_or_floor) {table[repo]} ', script), repo
