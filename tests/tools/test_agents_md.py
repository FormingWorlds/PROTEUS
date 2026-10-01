"""Unit tests for ``tools/agents/check_agents_md.py`` and ``tools/agents/sync_core.py``.

The checker runs in every ecosystem repository and must fail on a local edit of
a shared block, on a file above its byte cap, on a ``CLAUDE.md`` that is a
symlink or holds more than the import, and on a long ``copilot-instructions.md``.
The sync script must rewrite only the shared blocks, be idempotent, and report
lag in ``--check`` mode without writing.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_AGENTS_DIR = Path(__file__).resolve().parents[2] / 'tools' / 'agents'


def _load(name: str):
    """Load a script from ``tools/agents/`` without putting it on ``sys.path`` for good."""
    sys.path.insert(0, str(_AGENTS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f'{name}_uut', _AGENTS_DIR / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_AGENTS_DIR))
    return module


chk = _load('check_agents_md')
sync = _load('sync_core')


def _block(name: str, body: str, digest: str | None = None) -> str:
    """Return a shared block with the hash of ``body`` unless ``digest`` is given."""
    digest = digest or chk.block_hash(body)
    return f'<!-- fwl-{name}:begin sha256={digest} -->\n{body}\n<!-- fwl-{name}:end -->\n'


ROOT = _block('core', 'shared rule')
NESTED = _block('tests-core', 'tier rule')


def _repo(tmp_path: Path, agents: str, nested: str | None = None) -> Path:
    """Create a minimal non-git repository with the given AGENTS.md text."""
    (tmp_path / 'AGENTS.md').write_text(agents, encoding='utf-8')
    (tmp_path / 'CLAUDE.md').write_text('@AGENTS.md\n')
    if nested is not None:
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests' / 'AGENTS.md').write_text(nested, encoding='utf-8')
        (tmp_path / 'tests' / 'CLAUDE.md').write_text('@AGENTS.md\n')
    return tmp_path


def test_clean_repo_passes_and_hash_ignores_outer_blank_lines(tmp_path):
    """A repository whose blocks match their hashes passes; outer newlines do not change the hash."""
    repo = _repo(
        tmp_path, '# Repo\n' + _block('core', 'shared rule'), _block('tests-core', 'tier rule')
    )
    assert chk.check(repo) == []
    assert chk.block_hash('\nshared rule\n\n') == chk.block_hash('shared rule')
    assert chk.block_hash('shared rule.') != chk.block_hash('shared rule')


def test_local_edit_of_a_shared_block_fails(tmp_path):
    """Editing the shared text in a module repository is reported with the block name."""
    edited = _block('core', 'shared rule', digest=chk.block_hash('shared rule')).replace(
        'shared rule\n', 'shared rule, edited\n'
    )
    errors = chk.check(_repo(tmp_path, edited))
    assert len(errors) == 1
    assert 'fwl-core' in errors[0] and 'differs from its hash' in errors[0]


def test_malformed_marker_is_reported(tmp_path):
    """A begin marker without its end marker is not silently skipped."""
    errors = chk.check(_repo(tmp_path, ROOT + '<!-- fwl-voice:begin sha256=0123 -->\ntext\n'))
    assert len(errors) == 1
    assert 'malformed' in errors[0]


@pytest.mark.parametrize(
    ('where', 'size', 'n_errors'),
    [('root', 16_000, 0), ('root', 16_001, 1), ('nested', 12_000, 0), ('nested', 12_001, 1)],
)
def test_byte_caps_at_the_boundary(tmp_path, where, size, n_errors):
    """The cap is inclusive: 16,000 B root and 12,000 B nested pass, one byte more fails."""
    block = ROOT if where == 'root' else NESTED
    text = block + 'x' * (size - len(block))
    repo = _repo(tmp_path, text) if where == 'root' else _repo(tmp_path, ROOT, text)
    errors = chk.check(repo)
    assert len(errors) == n_errors
    assert all(str(size) in e for e in errors)


def test_byte_cap_counts_bytes_not_characters(tmp_path):
    """A root file of 15,000 characters that encodes to more than 16,000 bytes fails."""
    text = ROOT + 'é' * (15_000 - len(ROOT))
    assert len(text) <= 16_000 < len(text.encode())
    errors = chk.check(_repo(tmp_path, text))
    assert errors == [f'AGENTS.md: {len(text.encode())} B exceeds the 16000 B cap']


def test_checker_main_reads_the_root_from_argv_or_the_working_directory(
    tmp_path, monkeypatch, capsys
):
    """main() checks argv[1] or, without it, the working directory, and prints each failure."""
    good, bad = tmp_path / 'good', tmp_path / 'bad'
    good.mkdir()
    bad.mkdir()
    _repo(good, ROOT)
    (bad / 'CLAUDE.md').write_text('@AGENTS.md\n')
    assert chk.main(['check_agents_md.py', str(good)]) == 0
    assert chk.main(['check_agents_md.py', str(bad)]) == 1
    assert capsys.readouterr().out == 'ERROR: AGENTS.md: missing at the repository root\n'
    monkeypatch.chdir(good)
    assert chk.main(['check_agents_md.py']) == 0
    monkeypatch.chdir(bad)
    assert chk.main(['check_agents_md.py']) == 1


def test_claude_md_symlink_or_extra_text_fails(tmp_path):
    """CLAUDE.md must be a regular file with the import only; a symlink or extra text fails."""
    repo = _repo(tmp_path, ROOT)
    (repo / 'CLAUDE.md').write_text('@AGENTS.md\nextra rule\n')
    assert len(chk.check(repo)) == 1
    (repo / 'CLAUDE.md').unlink()
    (repo / 'import.md').write_text('@AGENTS.md\n')
    (repo / 'CLAUDE.md').symlink_to('import.md')
    assert any('CLAUDE.md' in e for e in chk.check(repo))
    (repo / 'CLAUDE.md').unlink()
    assert chk.check(repo) == ['CLAUDE.md: must be a regular file containing only @AGENTS.md']
    (repo / 'CLAUDE.md').write_text('@AGENTS.md\n')
    assert chk.check(repo) == []


def test_nested_agents_md_needs_its_own_claude_md(tmp_path):
    """A nested AGENTS.md without a sibling CLAUDE.md import is reported by path."""
    repo = _repo(tmp_path, ROOT, NESTED)
    assert chk.check(repo) == []
    (repo / 'tests' / 'CLAUDE.md').unlink()
    assert chk.check(repo) == [
        'tests/CLAUDE.md: must be a regular file containing only @AGENTS.md'
    ]


def test_missing_root_file_and_missing_directory_fail(tmp_path):
    """A repository without a root AGENTS.md, or a path that is not a directory, fails."""
    (tmp_path / 'CLAUDE.md').write_text('@AGENTS.md\n')
    assert chk.check(tmp_path) == ['AGENTS.md: missing at the repository root']
    assert chk.check(tmp_path / 'nope') == [f'{tmp_path / "nope"}: not a directory']
    assert chk.main(['check', str(tmp_path / 'nope')]) == 1


def test_tests_directory_without_its_agents_md_fails(tmp_path):
    """A repository with a tests/ directory must carry tests/AGENTS.md."""
    repo = _repo(tmp_path, ROOT, NESTED)
    assert chk.check(repo) == []
    (repo / 'tests' / 'AGENTS.md').unlink()
    (repo / 'tests' / 'CLAUDE.md').unlink()
    assert chk.check(repo) == [
        'tests/AGENTS.md: missing although the repository has a tests/ directory'
    ]


def test_marker_named_inside_a_line_is_not_counted(tmp_path):
    """Prose that quotes the marker syntax mid-line is not taken for a broken block."""
    text = 'Blocks start with `<!-- fwl-<name>:begin sha256=<hash> -->`.\n' + _block(
        'core', 'x'
    )
    assert chk.check(_repo(tmp_path, text)) == []
    assert len(chk.MARKER_RE.findall(text)) == 2


def test_git_checkout_sees_untracked_and_skips_deleted_files(tmp_path):
    """In a git checkout an untracked AGENTS.md is checked and a deleted tracked one is skipped."""
    repo = _repo(tmp_path, ROOT, NESTED)
    git = ['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t']
    git += ['-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null']
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '-A'], check=True)
    subprocess.run([*git, 'commit', '-qm', 'init'], check=True)
    (repo / 'tests' / 'AGENTS.md').unlink()
    (repo / 'src dir').mkdir()
    (repo / 'src dir' / 'AGENTS.md').write_text('x' * 12_001)
    errors = chk.check(repo)
    assert 'src dir/AGENTS.md: 12001 B exceeds the 12000 B cap' in errors
    assert 'src dir/CLAUDE.md: must be a regular file containing only @AGENTS.md' in errors
    assert 'tests/AGENTS.md: missing although the repository has a tests/ directory' in errors
    assert len(errors) == 3


def test_directory_walk_outside_git_skips_the_git_directory(tmp_path):
    """Without a usable git checkout the walk ignores an AGENTS.md under ``.git/``."""
    repo = _repo(tmp_path, ROOT)
    (repo / '.git').mkdir()
    (repo / '.git' / 'AGENTS.md').write_text('x' * 12_001)
    assert chk.agents_files(repo) == [repo / 'AGENTS.md']
    assert chk.check(repo) == []


def test_missing_shared_block_fails_in_root_and_tests(tmp_path):
    """A root AGENTS.md without fwl-core, or a tests/AGENTS.md without fwl-tests-core, fails."""
    repo = _repo(tmp_path, '# Repo\n' + _block('voice', 'v'), 'tests only\n')
    assert chk.check(repo) == [
        'AGENTS.md: missing the shared fwl-core block',
        'tests/AGENTS.md: missing the shared fwl-tests-core block',
    ]
    (repo / 'src').mkdir()
    (repo / 'src' / 'AGENTS.md').write_text('no block needed here\n')
    (repo / 'src' / 'CLAUDE.md').write_text('@AGENTS.md\n')
    assert len(chk.check(repo)) == 2


def test_block_hash_is_the_full_16_digit_prefix():
    """The marker hash is pinned: a shorter prefix or another digest would not match it."""
    assert chk.block_hash('shared rule') == 'f27e484383878999'
    assert len(chk.block_hash('')) == 16


def test_copilot_instructions_line_cap(tmp_path):
    """The Copilot file passes at 60 lines and fails at 61."""
    repo = _repo(tmp_path, ROOT)
    (repo / '.github').mkdir()
    (repo / '.github' / 'copilot-instructions.md').write_text('line\n' * 60)
    assert chk.check(repo) == []
    (repo / '.github' / 'copilot-instructions.md').write_text('line\n' * 61)
    assert chk.check(repo) == ['.github/copilot-instructions.md: 61 lines exceeds 60']


@pytest.fixture
def canon(tmp_path, monkeypatch):
    """Point the sync script at a temporary canonical directory."""
    d = tmp_path / 'canon'
    d.mkdir()
    (d / 'core.md').write_text('new shared rule\n')
    (d / 'voice-neutral.md').write_text('neutral voice\n')
    monkeypatch.setattr(sync, 'HERE', d)
    return d


def test_sync_rewrites_only_blocks_and_is_idempotent(tmp_path, canon):
    """Sync replaces block bodies and hashes, keeps all other text, and is idempotent."""
    old = '# Top\n' + _block('core', 'old rule') + 'repo text\n' + _block('voice', 'old voice')
    new = '# Top\n' + _block('core', 'new shared rule') + 'repo text\n'
    new += _block('voice', 'neutral voice') + 'tail text\n'
    repo = _repo(tmp_path, old + 'tail text\n')
    assert sync.main([str(repo)]) == 0
    text = (repo / 'AGENTS.md').read_text()
    assert text == new
    assert chk.check(repo) == []
    assert sync.sync_file(repo / 'AGENTS.md', write=True) == []
    assert (repo / 'AGENTS.md').read_text() == text


def test_sync_leaves_a_file_in_sync_byte_for_byte(tmp_path, canon):
    """A block that matches the canonical hash is left byte for byte, blank lines included."""
    text = '<!-- fwl-core:begin sha256=' + chk.block_hash('new shared rule') + ' -->\n'
    text += 'new shared rule\n\n<!-- fwl-core:end -->\n'
    repo = _repo(tmp_path, text)
    assert sync.main([str(repo)]) == 0
    assert (repo / 'AGENTS.md').read_text() == text


def test_check_mode_reports_lag_without_writing(tmp_path, canon, capsys, monkeypatch):
    """``--check`` exits 1 on a lagging block, leaves the file unchanged, and passes after a sync."""
    repo = _repo(tmp_path, _block('core', 'old rule'))
    before = (repo / 'AGENTS.md').read_text()
    assert sync.main(['--check', str(repo)]) == 1
    assert (repo / 'AGENTS.md').read_text() == before
    assert 'core differs' in capsys.readouterr().out
    assert sync.main([str(repo)]) == 0
    assert 'core updated' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['sync_core.py', '--check', str(repo)])
    assert sync.main() == 0
    (repo / 'AGENTS.md').write_text(before)
    assert sync.main() == 1


def test_sync_restores_a_body_edited_under_the_current_hash(tmp_path, canon):
    """A block whose marker carries the canonical hash but whose body was edited is rewritten."""
    canonical_hash = chk.block_hash('new shared rule')
    edited = _block('core', 'edited rule', digest=canonical_hash)
    repo = _repo(tmp_path, edited)
    assert sync.sync_file(repo / 'AGENTS.md', write=True) == ['core']
    assert 'new shared rule' in (repo / 'AGENTS.md').read_text()
    assert chk.check(repo) == []


def test_hash_mismatch_after_the_eighth_digit_is_detected(tmp_path, canon):
    """A marker hash equal to the body hash in its first 8 digits only fails the check and the sync."""
    repo = _repo(
        tmp_path, _block('core', 'shared rule', chk.block_hash('shared rule')[:8] + '0' * 8)
    )
    errors = chk.check(repo)
    assert len(errors) == 1 and 'fwl-core differs from its hash' in errors[0]
    lagging = chk.block_hash('new shared rule')[:8] + '0' * 8
    (repo / 'AGENTS.md').write_text(_block('core', 'new shared rule', lagging))
    assert sync.sync_file(repo / 'AGENTS.md', write=False) == ['core']


def test_every_block_the_sync_can_request_exists_in_tools_agents():
    """Every shared block name here, and the voice block, has a canonical file."""
    repo = _AGENTS_DIR.parents[1]
    names = {'core', 'tests-core', 'voice'}
    for path in ('AGENTS.md', 'tests/AGENTS.md'):
        names |= {m['name'] for m in chk.BLOCK_RE.finditer((repo / path).read_text())}
    assert sync.HERE == _AGENTS_DIR
    for name in sorted(names):
        assert sync.canonical(name).strip(), name
    with pytest.raises(FileNotFoundError):
        sync.canonical('no-such-block')


@pytest.mark.parametrize(
    'prefix',
    ['<!--fwl-voice:', '  <!-- fwl-voice:', '<!-- FWL-voice:', '<!-- fwl_voice:'],
    ids=['no-space', 'indented', 'uppercase', 'underscore'],
)
def test_a_misspelt_marker_prefix_fails_both_scripts(tmp_path, canon, prefix):
    """A block whose two markers share a misspelt prefix is reported, not skipped as prose."""
    text = ROOT + _block('voice', 'old voice').replace('<!-- fwl-voice:', prefix)
    repo = _repo(tmp_path, text)
    assert chk.check(repo) == ['AGENTS.md: unmatched or malformed fwl- block marker']
    assert sync.main(['--check', str(repo)]) == 1
    assert (repo / 'AGENTS.md').read_text() == text


def test_sync_reports_a_block_without_a_canonical_file(tmp_path, canon, capsys):
    """A block name with no file in tools/agents/ fails without a traceback or a write."""
    text = _block('bogus', 'module text')
    repo = _repo(tmp_path, text)
    assert sync.main([str(repo)]) == 1
    assert 'no canonical file bogus.md' in capsys.readouterr().out
    assert (repo / 'AGENTS.md').read_text() == text


@pytest.mark.parametrize(
    'broken',
    [
        ROOT.replace('fwl-core:end -->', 'fwl-core:end  -->'),
        ROOT.replace('\n', '\r\n'),
        ROOT.replace(chk.block_hash('shared rule'), chk.block_hash('shared rule').upper()),
    ],
    ids=['typo-in-end-marker', 'crlf-line-ends', 'uppercase-hash'],
)
def test_sync_reports_a_marker_it_cannot_parse(tmp_path, canon, capsys, broken):
    """A block marker the sync cannot parse fails both modes and leaves the file unchanged."""
    repo = _repo(tmp_path, broken)
    (repo / 'AGENTS.md').write_bytes(broken.encode())
    assert sync.main(['--check', str(repo)]) == 1
    assert sync.main([str(repo)]) == 1
    assert (repo / 'AGENTS.md').read_bytes() == broken.encode()
    assert capsys.readouterr().out.count('malformed fwl- block marker') == 2


def test_sync_fails_on_a_path_that_is_not_a_directory(tmp_path, canon, capsys):
    """A mistyped repository path fails in both modes instead of passing silently."""
    assert sync.main([str(tmp_path / 'nope')]) == 1
    assert sync.main(['--check', str(tmp_path / 'nope')]) == 1
    assert 'not a directory' in capsys.readouterr().out
