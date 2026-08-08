"""Unit tests for ``tools/_docgen.py``.

The module under test carries the shared machinery of the reference-doc
generators: marker-delimited replacement, output normalisation, deterministic
JSON serialisation, the checkout-scoped ``proteus.config`` import, the
check/write driver, and the docs sanity checks. These tests exercise:

* the marker contract (replacement preserves surrounding prose; absent or
  duplicated markers raise instead of silently no-opping),
* the output hygiene contract (no trailing whitespace, single final newline),
* drift detection semantics of the check/write driver,
* the nav-path extraction and marker-balance sanity checks.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_module():
    """Load ``tools/_docgen.py`` directly.

    The ``tools/`` directory is not a Python package, and adding the repo root
    to ``sys.path`` would shadow installed ecosystem packages with the local
    checkout. ``importlib.util`` loads the script in isolation.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / 'tools' / '_docgen.py'
    spec = importlib.util.spec_from_file_location('docgen_uut', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_dg = _load_module()


_PAGE = """# Title

Hand-written intro prose.

<!-- BEGIN GENERATED: demo-table -->
| old | table |
<!-- END GENERATED: demo-table -->

Hand-written closing prose.
"""


def test_replace_between_markers_preserves_surrounding_prose():
    """Replacement swaps only the delimited region; prose on both sides and
    the marker lines themselves survive byte-for-byte."""
    out = _dg.replace_between_markers(_PAGE, 'GENERATED: demo-table', '| new | table |')
    assert 'Hand-written intro prose.' in out
    assert 'Hand-written closing prose.' in out
    assert '| new | table |' in out
    assert '| old | table |' not in out
    # The marker pair itself must survive so the next run can find it again.
    assert out.count('<!-- BEGIN GENERATED: demo-table -->') == 1
    assert out.count('<!-- END GENERATED: demo-table -->') == 1


def test_replace_between_markers_at_end_of_file():
    """Edge case: a marker pair terminating the file (no trailing prose) is
    replaced without corrupting the closing marker."""
    page = '# T\n\n<!-- BEGIN X -->\nold\n<!-- END X -->'
    out = _dg.replace_between_markers(page, 'X', 'new')
    assert out.endswith('<!-- END X -->')
    assert '\nnew\n' in out
    assert 'old' not in out


def test_replace_between_markers_missing_marker_raises():
    """Error contract: a marker absent from the page raises DocgenError
    rather than returning the input unchanged (the silent-no-op bug class)."""
    with pytest.raises(_dg.DocgenError, match='not found'):
        _dg.replace_between_markers(_PAGE, 'GENERATED: no-such-table', 'x')
    # A duplicated marker pair is equally structural: refuse, do not guess.
    twice = (
        _PAGE + '\n<!-- BEGIN GENERATED: demo-table -->\n<!-- END GENERATED: demo-table -->\n'
    )
    with pytest.raises(_dg.DocgenError, match='appears 2 times'):
        _dg.replace_between_markers(twice, 'GENERATED: demo-table', 'x')


def test_normalize_enforces_hygiene_contract():
    """Trailing whitespace is stripped per line and the text ends with exactly
    one newline; already-clean input is a fixed point (idempotent)."""
    dirty = 'a  \nb\t\n\nc'
    clean = _dg.normalize(dirty)
    assert clean == 'a\nb\n\nc\n'
    assert _dg.normalize(clean) == clean
    # Edge case: many trailing newlines collapse to exactly one.
    assert _dg.normalize('x\n\n\n') == 'x\n'


def test_escape_link_brackets_protects_units_but_not_real_links():
    """Bracketed units stop being shortcut reference syntax, while both
    markdown link forms survive untouched."""
    assert _dg.escape_link_brackets('pressure [bar].') == 'pressure \\[bar\\].'
    assert _dg.escape_link_brackets('k [W m-1 K-1]') == 'k \\[W m-1 K-1\\]'
    inline = 'See [documentation](https://example.org/a.html).'
    assert _dg.escape_link_brackets(inline) == inline
    # A URL carrying parentheses must not defeat the link protection.
    nested = 'See [entry](https://example.org/Foo_(bar)#x).'
    assert _dg.escape_link_brackets(nested) == nested
    reference = 'A ref link [text][label] stays.'
    assert _dg.escape_link_brackets(reference) == reference
    # A closing bracket with no opener is interval notation, not a link.
    assert _dg.escape_link_brackets('within (0, 10] bar') == 'within (0, 10\\] bar'


def test_escape_link_brackets_leaves_code_spans_alone_and_is_idempotent():
    """A backslash renders literally inside a code span, so brackets there must
    survive; escaping an already-escaped string is a fixed point."""
    code = "gravity from ``hf_row['gravity']`` before"
    assert _dg.escape_link_brackets(code) == code
    mixed = 'Distinct from ``[a.b].c`` unlike [K] here.'
    assert _dg.escape_link_brackets(mixed) == 'Distinct from ``[a.b].c`` unlike \\[K\\] here.'
    once = _dg.escape_link_brackets('pressure [bar] and [K]')
    assert _dg.escape_link_brackets(once) == once
    assert once.count('\\\\') == 0  # never doubles an existing escape
    assert _dg.escape_link_brackets('') == ''


def test_dump_json_is_deterministic_and_newline_terminated():
    """Two serialisations of the same object are byte-identical, insertion
    order is preserved (the generators sort upstream), and non-ASCII survives
    unescaped."""
    obj = {'b': 1, 'a': [2, 3], 'unit': 'kg m s-2'}
    one, two = _dg.dump_json(obj), _dg.dump_json(obj)
    assert one == two
    assert one.endswith('}\n')
    assert list(obj) == ['b', 'a', 'unit']  # dump must not reorder keys
    assert '\\u' not in _dg.dump_json({'sym': 'microµ'})


def test_run_generator_check_mode_detects_drift_and_missing(tmp_path, capsys):
    """Check mode exits 1 for a stale file and a missing file, names the
    regeneration command, and never writes; write mode repairs both."""
    fresh = tmp_path / 'fresh.md'
    fresh.write_text('current\n')
    stale = tmp_path / 'stale.md'
    stale.write_text('outdated\n')
    missing = tmp_path / 'missing.json'
    targets = [(fresh, 'current\n'), (stale, 'current\n'), (missing, '{}\n')]

    rc = _dg.run_generator(targets, check=True, regen_cmd='python tools/x.py')
    out = capsys.readouterr().out
    assert rc == 1
    assert 'python tools/x.py' in out
    assert stale.read_text() == 'outdated\n'  # check mode must not write
    assert not missing.exists()

    rc = _dg.run_generator(targets, check=False, regen_cmd='python tools/x.py')
    assert rc == 0
    assert stale.read_text() == 'current\n'
    assert missing.read_text() == '{}\n'
    # After the write pass, a re-check comes back clean.
    assert _dg.run_generator(targets, check=True, regen_cmd='python tools/x.py') == 0


def test_check_markers_balanced_discriminates_pairings():
    """Balanced pairs (including the legacy PYPI_TABLE style) pass; an
    unclosed BEGIN, an orphan END, and a name mismatch are each reported."""
    good = '<!-- BEGIN A -->\nx\n<!-- END A -->\n<!-- BEGIN PYPI_TABLE -->\n<!-- END PYPI_TABLE -->'
    assert _dg.check_markers_balanced(good) == []
    assert _dg.check_markers_balanced('no markers at all') == []  # edge: none
    assert any('never closed' in p for p in _dg.check_markers_balanced('<!-- BEGIN A -->'))
    assert any('without a matching' in p for p in _dg.check_markers_balanced('<!-- END A -->'))
    mismatch = '<!-- BEGIN A -->\n<!-- END B -->'
    assert any('does not match' in p for p in _dg.check_markers_balanced(mismatch))


def test_parse_nav_paths_extracts_pages_and_rejects_empty():
    """Nav extraction returns the .md paths in order, ignores non-page lines
    (section headers, URLs), and raises on a nav-less file (error contract)."""
    yml = (
        'site_name: X\n'
        'nav:\n'
        '  - Home: index.md\n'
        '  - Guides:\n'
        '      - Setup: How-to/setup.md\n'
        '  - External: https://example.org/page\n'
        'plugins:\n'
        '  - search\n'
    )
    assert _dg.parse_nav_paths(yml) == ['index.md', 'How-to/setup.md']
    with pytest.raises(_dg.DocgenError, match='no nav entries'):
        _dg.parse_nav_paths('site_name: X\nplugins:\n  - search\n')


def test_parse_nav_paths_on_committed_mkdocs_yml():
    """The committed mkdocs.yml yields a plausible nav (dozens of pages, all
    ending in .md) without YAML-loading it, and every page exists on disk."""
    repo_root = Path(__file__).resolve().parents[2]
    paths = _dg.parse_nav_paths((repo_root / 'mkdocs.yml').read_text())
    assert len(paths) > 20  # the real nav is large; a tiny result means the block parse broke
    assert all(p.endswith('.md') for p in paths)
    missing = [p for p in paths if not (repo_root / 'docs' / p).is_file()]
    assert missing == []


def test_import_proteus_config_skips_package_init():
    """The stub-parent import loads proteus.config from the checkout without
    executing the package __init__ (whose import chain pulls juliacall)."""
    saved = {k: sys.modules[k] for k in list(sys.modules) if k.split('.')[0] == 'proteus'}
    for k in saved:
        del sys.modules[k]
    had_juliacall = 'juliacall' in sys.modules
    try:
        cfg_module = _dg.import_proteus_config()
        assert hasattr(cfg_module, 'Config')
        # The stub path must not have pulled in the runtime stack.
        assert ('juliacall' in sys.modules) == had_juliacall
        src_root = str(_dg.REPO_ROOT / 'src')
        assert cfg_module.__file__ is not None and cfg_module.__file__.startswith(src_root)
    finally:
        for k in list(sys.modules):
            if k.split('.')[0] == 'proteus':
                del sys.modules[k]
        sys.modules.update(saved)
