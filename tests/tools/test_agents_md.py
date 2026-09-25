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
    (tmp_path / 'AGENTS.md').write_text(agents)
    (tmp_path / 'CLAUDE.md').write_text('@AGENTS.md\n')
    if nested is not None:
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests' / 'AGENTS.md').write_text(nested)
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
    (d / 'voice-list.md').write_text('list voice\n')
    monkeypatch.setattr(sync, 'HERE', d)
    return d


def test_sync_rewrites_only_blocks_and_is_idempotent(tmp_path, canon):
    """Sync replaces the block bodies and hashes, keeps the repo text, and a second run changes nothing."""
    (tmp_path / 'r').mkdir()
    agents = (
        '# Top\n' + _block('core', 'old rule') + 'repo text\n' + _block('voice', 'old voice')
    )
    repo = _repo(tmp_path / 'r', agents)
    assert sync.main([str(repo)]) == 0
    text = (repo / 'AGENTS.md').read_text()
    assert 'new shared rule' in text and 'neutral voice' in text and 'old rule' not in text
    assert text.startswith('# Top\n') and '\nrepo text\n' in text
    assert chk.check(repo) == []
    assert sync.sync_file(repo / 'AGENTS.md', 'neutral', write=True) == []
    assert (repo / 'AGENTS.md').read_text() == text


def test_check_mode_reports_lag_without_writing(tmp_path, canon, capsys):
    """``--check`` exits 1 on a lagging block, leaves the file unchanged, and passes after a sync."""
    (tmp_path / 'r').mkdir()
    repo = _repo(tmp_path / 'r', _block('core', 'old rule'))
    before = (repo / 'AGENTS.md').read_text()
    assert sync.main(['--check', str(repo)]) == 1
    assert (repo / 'AGENTS.md').read_text() == before
    assert 'core differs' in capsys.readouterr().out
    sync.main([str(repo), '--voice', 'list'])
    assert sync.main(['--check', str(repo)]) == 0


def test_sync_restores_a_body_edited_under_the_current_hash(tmp_path, canon):
    """A block whose marker carries the canonical hash but whose body was edited is rewritten."""
    (tmp_path / 'r').mkdir()
    canonical_hash = chk.block_hash('new shared rule')
    edited = _block('core', 'edited rule', digest=canonical_hash)
    repo = _repo(tmp_path / 'r', edited)
    assert sync.sync_file(repo / 'AGENTS.md', 'neutral', write=True) == ['core']
    assert 'new shared rule' in (repo / 'AGENTS.md').read_text()
    assert chk.check(repo) == []


def _flip_tail(digest: str) -> str:
    """Return ``digest`` with its last 8 hex digits changed and its first 8 kept."""
    tail = ''.join('0' if c != '0' else '1' for c in digest[8:])
    return digest[:8] + tail


def test_hash_mismatch_after_the_eighth_digit_is_detected(tmp_path, canon):
    """A marker hash equal to the body hash in its first 8 digits only fails the check and the sync."""
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    marker = _flip_tail(chk.block_hash('shared rule'))
    assert marker[:8] == chk.block_hash('shared rule')[:8] and marker != chk.block_hash(
        'shared rule'
    )
    errors = chk.check(_repo(tmp_path / 'a', _block('core', 'shared rule', digest=marker)))
    assert len(errors) == 1 and 'fwl-core differs from its hash' in errors[0]
    lagging = _block(
        'core', 'new shared rule', digest=_flip_tail(chk.block_hash('new shared rule'))
    )
    repo = _repo(tmp_path / 'b', lagging)
    assert sync.sync_file(repo / 'AGENTS.md', 'neutral', write=False) == ['core']


def test_voice_variant_selects_the_canonical_file(tmp_path, canon):
    """``--voice list`` writes the list variant; checking against neutral then reports lag."""
    (tmp_path / 'r').mkdir()
    repo = _repo(tmp_path / 'r', _block('voice', 'old voice'))
    sync.main([str(repo), '--voice', 'list'])
    assert 'list voice' in (repo / 'AGENTS.md').read_text()
    assert sync.sync_file(repo / 'AGENTS.md', 'neutral', write=False) == ['voice']
    with pytest.raises(FileNotFoundError):
        sync.canonical('no-such-block', 'neutral')


def test_sync_fails_on_a_path_that_is_not_a_directory(tmp_path, canon, capsys):
    """A mistyped repository path fails in both modes instead of passing silently."""
    assert sync.main([str(tmp_path / 'nope')]) == 1
    assert sync.main(['--check', str(tmp_path / 'nope')]) == 1
    assert 'not a directory' in capsys.readouterr().out
