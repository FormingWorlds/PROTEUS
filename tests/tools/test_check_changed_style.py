"""Unit tests for ``tools/check_changed_style.py``.

The script checks only the Python lines a branch changes. These tests build a
small git repository per case, commit a base version on ``main``, change it on a
branch and run the check against ``main``. They exercise:

* the comment-block limit (a block of 5 lines fails, 4 passes, a header at
  line 1 is exempt, a touched legacy block fails, ``#`` inside a string is
  not a comment),
* the docstring rules with the NumPy convention on changed nodes only (an
  untouched legacy violation passes, a method edit does not flag its class,
  a new file needs a module docstring),
* the error contract (bad base ref exits 2, a syntax error is a finding).

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md`` for
the test framework.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_module():
    """Load ``tools/check_changed_style.py`` directly; ``tools/`` is not a package."""
    script = Path(__file__).resolve().parents[2] / 'tools' / 'check_changed_style.py'
    spec = importlib.util.spec_from_file_location('check_changed_style_uut', script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ccs = _load_module()

LEGACY = '''"""Legacy module."""


def legacy(x):
    return x


def helper(x):
    """Return x doubled."""
    return 2 * x
'''


def _git(repo: Path, *args: str) -> None:
    """Run a git command in ``repo`` with a fixed identity."""
    subprocess.run(
        ['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _base(repo: Path, files: dict[str, str]) -> None:
    """Commit ``files`` on ``main``, the base of the branch under check."""
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'base')


def _check(repo: Path, monkeypatch, capsys, files: dict[str, str]) -> tuple[int, list[str]]:
    """Write ``files`` on a branch off ``main`` and run the check against ``main``."""
    _git(repo, 'checkout', '-q', '-B', 'feature')
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, 'add', '-A')
    monkeypatch.chdir(repo)
    code = _ccs.main(['--base', 'main'])
    return code, capsys.readouterr().out.splitlines()


@pytest.fixture
def repo(tmp_path):
    """Repository with ``mod.py`` committed on ``main``."""
    _git(tmp_path, 'init', '-q', '-b', 'main')
    _base(tmp_path, {'mod.py': LEGACY})
    return tmp_path


@pytest.mark.parametrize(('n_lines', 'expected'), [(4, 0), (5, 1)])
def test_added_comment_block_limit(repo, monkeypatch, capsys, n_lines, expected):
    """A new block of 5 comment lines fails and a block of 4 passes."""
    block = ''.join(f'    # note {i}\n' for i in range(n_lines))
    text = LEGACY.replace('    return 2 * x\n', block + '    return 2 * x\n')
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': text})
    assert code == expected
    assert len(out) == expected
    if expected:
        assert out[0] == 'mod.py:10: CMT comment block of 5 lines (max 4)'


def test_untouched_legacy_violations_pass(repo, monkeypatch, capsys):
    """A legacy function without docstring and a legacy long block pass when untouched."""
    base = LEGACY + ''.join(f'# old {i}\n' for i in range(6))
    _base(repo, {'mod.py': base})
    text = base.replace('return 2 * x', 'return x + x')
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': text})
    assert code == 0
    assert out == []
    # The same file with the legacy function touched fails, so the pass above is not vacuous.
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': text.replace('return x\n', 'return +x\n')}
    )
    assert code == 1
    assert out == ['mod.py:4: D103 Missing docstring in public function']


def test_touched_legacy_comment_block_fails(repo, monkeypatch, capsys):
    """Editing or deleting one line of a legacy block longer than 4 lines fails."""
    base = LEGACY + ''.join(f'# old {i}\n' for i in range(7))
    _base(repo, {'mod.py': base})
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': base.replace('# old 3', '# new 3')}
    )
    assert (code, out) == (1, ['mod.py:11: CMT comment block of 7 lines (max 4)'])
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': base.replace('# old 3\n', '')})
    assert (code, out) == (1, ['mod.py:11: CMT comment block of 6 lines (max 4)'])


def test_header_block_and_string_hashes_are_exempt(repo, monkeypatch, capsys):
    """A header block at line 1 and ``#`` lines inside a string do not count."""
    header = ''.join(f'# licence line {i}\n' for i in range(6))
    text = header + LEGACY + '\nTEXT = """\n' + '# not a comment\n' * 6 + '"""\n'
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': text})
    assert code == 0
    assert out == []


def test_method_edit_does_not_flag_legacy_class(repo, monkeypatch, capsys):
    """A method edit leaves a class without docstring alone; a class-level edit flags it."""
    base = (
        LEGACY
        + '\n\nclass Legacy:\n    size = 1\n\n    def get(self):\n        """Return 1."""\n        return 1\n'
    )
    _base(repo, {'mod.py': base})
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': base.replace('return 1\n', 'return 2\n')}
    )
    assert (code, out) == (0, [])
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': base.replace('size = 1', 'size = 2')}
    )
    assert (code, out) == (1, ['mod.py:13: D101 Missing docstring in public class'])


def test_changed_method_without_docstring_fails(repo, monkeypatch, capsys):
    """A changed method without docstring fails inside a documented class."""
    base = LEGACY + '\n\nclass Doc:\n    """Doc."""\n\n    def get(self):\n        return 1\n'
    _base(repo, {'mod.py': base})
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': base.replace('return 1\n', 'return 2\n')}
    )
    assert code == 1
    assert out == ['mod.py:16: D102 Missing docstring in public method']


def test_trailing_comments_do_not_form_a_block(repo, monkeypatch, capsys):
    """Comments after code on consecutive lines are not a comment block."""
    body = ''.join(f'    x{i} = {i}  # note {i}\n' for i in range(6))
    text = LEGACY.replace('    return 2 * x\n', body + '    return 2 * x\n')
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': text})
    assert code == 0
    assert out == []


def test_numpy_convention_and_deleted_neighbour(repo, monkeypatch, capsys):
    """NumPy convention skips ``__init__`` (D107); deleting a function flags no neighbour."""
    gone = '\n\ndef gone():\n    pass\n'
    init = '\n\nclass Doc:\n    """Doc."""\n\n    def __init__(self):\n        pass\n'
    base = LEGACY.replace('    return x\n', '    return x\n' + gone, 1) + init
    _base(repo, {'mod.py': base})
    text = LEGACY + init.replace('        pass\n', '        self.a = 1\n')
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': text})
    assert code == 0
    assert out == []


def test_new_file_needs_module_docstring(repo, monkeypatch, capsys):
    """A new file without a module docstring fails; an edited legacy file without one passes."""
    _base(repo, {'old.py': 'X = 1\n'})
    code, out = _check(repo, monkeypatch, capsys, {'new.py': 'Y = 2\n', 'old.py': 'X = 3\n'})
    assert code == 1
    assert out == ['new.py:1: D100 Missing docstring in public module']


def test_paths_limit_the_check(repo, monkeypatch, capsys):
    """Given paths, only those changed files are checked, from any directory."""
    _base(repo, {'other.py': LEGACY})
    edit = LEGACY.replace('return x\n', 'return +x\n')
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': edit, 'other.py': edit})
    assert (code, len(out)) == (1, 2)
    assert _ccs.main(['--base', 'main', 'other.py']) == 1
    assert capsys.readouterr().out.splitlines() == [
        'other.py:4: D103 Missing docstring in public function'
    ]
    (repo / 'sub').mkdir()
    monkeypatch.chdir(repo / 'sub')
    assert _ccs.main(['--base', 'main', '../mod.py']) == 1
    assert capsys.readouterr().out.startswith('mod.py:4: D103')


def test_error_contract(repo, monkeypatch, capsys):
    """A bad base ref exits 2 and a syntax error in a changed file is a finding."""
    monkeypatch.chdir(repo)
    assert _ccs.main(['--base', 'no-such-ref']) == 2
    assert 'merge-base' in capsys.readouterr().err
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': LEGACY + '\ndef broken(:\n'})
    assert code == 1
    assert len(out) == 1 and out[0].startswith('mod.py:12: E999 syntax error')
    (repo / 'bad.py').write_bytes(b'"""M."""\nX = "\xe9"\n')
    code, out = _check(repo, monkeypatch, capsys, {})
    assert code == 2
    assert out == []


def test_boundary_deletions_touch_their_block_or_node(repo, monkeypatch, capsys):
    """Deleting the first or last line of a long block, or a function's last line, counts."""
    block = ''.join(f'# old {i}\n' for i in range(7))
    base = LEGACY.replace('    return x\n', '    y = x\n    return y\n') + block
    _base(repo, {'mod.py': base})
    for gone in ('# old 0\n', '# old 6\n'):
        code, out = _check(repo, monkeypatch, capsys, {'mod.py': base.replace(gone, '')})
        assert (code, out) == (1, ['mod.py:12: CMT comment block of 6 lines (max 4)'])
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': base.replace('    return y\n', '')}
    )
    assert (code, out) == (1, ['mod.py:4: D103 Missing docstring in public function'])


def test_decorator_deletion_and_rename(repo, monkeypatch, capsys):
    """Removing a decorator changes the function; a renamed file keeps its history."""
    base = LEGACY.replace('def legacy(x):', '@staticmethod\ndef legacy(x):')
    _base(repo, {'mod.py': base})
    code, out = _check(repo, monkeypatch, capsys, {'mod.py': LEGACY})
    assert (code, out) == (1, ['mod.py:4: D103 Missing docstring in public function'])
    _git(repo, 'checkout', '-q', '--', 'mod.py')
    _git(repo, 'mv', 'mod.py', 'moved.py')
    edit = base.replace('return 2 * x', 'return x + x')
    code, out = _check(repo, monkeypatch, capsys, {'moved.py': edit})
    assert (code, out) == (0, [])


def test_module_docstring_rules(repo, monkeypatch, capsys):
    """D100 follows the module docstring, not a function on line 1."""
    _base(repo, {'first.py': 'def f(x):\n    """F."""\n    return x\n'})
    code, out = _check(
        repo, monkeypatch, capsys, {'first.py': 'def f(x):\n    """F."""\n    return +x\n'}
    )
    assert (code, out) == (0, [])
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': LEGACY.replace('"""Legacy module."""\n', '')}
    )
    assert code == 1
    assert out[0] == 'mod.py:1: D100 Missing docstring in public module'


def test_diff_parsing_is_robust(repo, monkeypatch, capsys):
    """Content lines like diff headers, git diff.noprefix and line order do not break the check."""
    _git(repo, 'config', 'diff.noprefix', 'true')
    text = (
        LEGACY
        + '\nT = """\n++ not a header\n-- nor this\n"""\n'
        + '\n\ndef late():\n    pass\n'
    )
    code, out = _check(
        repo, monkeypatch, capsys, {'mod.py': text.replace('return x\n', 'return +x\n')}
    )
    assert code == 1
    assert out == [
        'mod.py:4: D103 Missing docstring in public function',
        'mod.py:18: D103 Missing docstring in public function',
    ]
