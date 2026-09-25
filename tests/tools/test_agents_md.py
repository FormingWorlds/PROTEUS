"""Unit tests for ``tools/agents/check_agents_md.py`` and ``tools/agents/sync_core.py``.

The checker runs in every ecosystem repository and must fail on a local edit of
a shared block, on a file above its byte cap, on a ``CLAUDE.md`` that is a
symlink or holds more than the import, and on a long ``copilot-instructions.md``.
The sync script must rewrite only the shared blocks, be idempotent, and report
lag in ``--check`` mode without writing.
"""

from __future__ import annotations

import importlib.util
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


def _repo(tmp_path: Path, agents: str, nested: str | None = None) -> Path:
    """Create a minimal non-git repository with the given AGENTS.md text."""
    (tmp_path / 'AGENTS.md').write_text(agents)
    (tmp_path / 'CLAUDE.md').write_text('@AGENTS.md\n')
    if nested is not None:
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests' / 'AGENTS.md').write_text(nested)
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
    errors = chk.check(_repo(tmp_path, '<!-- fwl-core:begin sha256=0123 -->\ntext\n'))
    assert len(errors) == 1
    assert 'malformed' in errors[0]


@pytest.mark.parametrize(
    ('where', 'size', 'n_errors'),
    [('root', 16_000, 0), ('root', 16_001, 1), ('nested', 12_000, 0), ('nested', 12_001, 1)],
)
def test_byte_caps_at_the_boundary(tmp_path, where, size, n_errors):
    """The cap is inclusive: 16,000 B root and 12,000 B nested pass, one byte more fails."""
    text = 'x' * size
    repo = _repo(tmp_path, text) if where == 'root' else _repo(tmp_path, 'root', text)
    errors = chk.check(repo)
    assert len(errors) == n_errors
    assert all(str(size) in e for e in errors)


def test_claude_md_symlink_or_extra_text_fails(tmp_path):
    """CLAUDE.md must be a regular file with the import only; a symlink or extra text fails."""
    repo = _repo(tmp_path, 'root')
    (repo / 'CLAUDE.md').write_text('@AGENTS.md\nextra rule\n')
    assert len(chk.check(repo)) == 1
    (repo / 'CLAUDE.md').unlink()
    (repo / 'import.md').write_text('@AGENTS.md\n')
    (repo / 'CLAUDE.md').symlink_to('import.md')
    assert any('CLAUDE.md' in e for e in chk.check(repo))
    (repo / 'CLAUDE.md').unlink()
    assert chk.check(repo) == []


def test_copilot_instructions_line_cap(tmp_path):
    """The Copilot file passes at 60 lines and fails at 61."""
    repo = _repo(tmp_path, 'root')
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


def test_voice_variant_selects_the_canonical_file(tmp_path, canon):
    """``--voice list`` writes the list variant; checking against neutral then reports lag."""
    (tmp_path / 'r').mkdir()
    repo = _repo(tmp_path / 'r', _block('voice', 'old voice'))
    sync.main([str(repo), '--voice', 'list'])
    assert 'list voice' in (repo / 'AGENTS.md').read_text()
    assert sync.sync_file(repo / 'AGENTS.md', 'neutral', write=False) == ['voice']
    with pytest.raises(FileNotFoundError):
        sync.canonical('no-such-block', 'neutral')
