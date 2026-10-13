"""Unit tests for ``tools/figures/code_links.py``.

The script resolves the name-based code links of the architecture diagram to
line ranges. These tests check:

* a section banner resolves to its closing banner and wins over a comment
  that mentions the same words,
* a ``#`` inside a string is not taken for a comment,
* a comment resolves to itself and the whole statement after it,
* a method resolves by qualified name,
* the error contract: a missing, duplicated or unclosed target raises,
* the deploy rewrite pins links to a commit and falls back to the file,
* every target of the committed diagram resolves in the committed tree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIGURES = _REPO_ROOT / 'tools' / 'figures'

SAMPLE = '''\
"""Docstring that quotes a banner: ############### ESCAPE"""


class Runner:
    def step(self):
        x = 1
        ############### ESCAPE
        # escape is cheap here
        y = 2
        z = 3
        ############### / ESCAPE

        # Apply impacts now. Then carry on.
        if x:
            y = 4
            z = 5
        w = 6
        return w
'''


@pytest.fixture
def links(tmp_path, monkeypatch):
    """The module with its source root pointed at a sample tree."""
    spec = importlib.util.spec_from_file_location('code_links', _FIGURES / 'code_links.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'mod.py').write_text(SAMPLE)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    mod.source.cache_clear()
    return mod


def test_banner_resolves_to_closing_banner_not_to_mentions(links):
    """The banner spans lines 7 to 11; the docstring and the lower-case comment do not count."""
    assert links.resolve_target('pkg/mod.py', 'escape') == (7, 11)
    # the comment on line 8 slugs to 'escape-is-cheap-here', not 'escape'
    assert links.resolve_target('pkg/mod.py', 'escape-is-cheap-here') == (8, 9)


def test_comment_covers_the_whole_following_statement(links):
    """A comment before an if block resolves to the end of the block, not its first line."""
    first, last = links.resolve_target('pkg/mod.py', 'apply-impacts-now')
    assert (first, last) == (13, 16)
    assert last != 14  # the if header alone
    assert links.resolve_target('pkg/mod.py', 'apply-impacts-now-then-carry-on') == (13, 16)


def test_method_resolves_by_qualified_name(links):
    """Runner.step resolves to its def and its last line; the bare name does not."""
    assert links.resolve_target('pkg/mod.py', 'Runner.step') == (5, 18)
    with pytest.raises(links.LinkError, match='no function or comment'):
        links.resolve_target('pkg/mod.py', 'Runner.stop')


def test_missing_duplicate_and_unclosed_targets_raise(links, tmp_path):
    """A target that matches zero or several times, or a banner never closed, is an error."""
    (tmp_path / 'pkg' / 'dup.py').write_text('# Same note\nx = 1\n# Same note\ny = 2\n')
    (tmp_path / 'pkg' / 'open.py').write_text('############### OPEN\nx = 1\n')
    with pytest.raises(links.LinkError, match='2 comments match'):
        links.resolve_target('pkg/dup.py', 'same-note')
    with pytest.raises(links.LinkError, match='never closed'):
        links.resolve_target('pkg/open.py', 'open')
    with pytest.raises(links.LinkError, match='does not exist'):
        links.resolve_target('pkg/gone.py', 'escape')


def test_line_anchors_and_foreign_links_pass_through(links):
    """Only name-based links into the repository are resolved."""
    hard = links.REPO_URL + 'pkg/mod.py#L12'
    other = 'https://example.org/page#section'
    assert links.resolve(hard) == hard
    assert links.resolve(other) == other
    pinned = links.resolve(links.REPO_URL + 'pkg/mod.py#escape', commit='abc123')
    assert pinned == f'{links.REPO}/blob/abc123/pkg/mod.py#L7-L11'


def test_deploy_rewrite_pins_commit_and_falls_back_to_file(links, tmp_path, monkeypatch):
    """--commit rewrites resolvable links to the commit and a broken one to the file."""
    good = links.REPO_URL + 'pkg/mod.py#escape'
    bad = links.REPO_URL + 'pkg/mod.py#renamed-away'
    svg = tmp_path / 'fig.svg'
    svg.write_text(
        f'<svg><a xlink:href="x" data-target="{good}"/>'
        f'<a xlink:href="y" data-target="{bad}"/></svg>'
    )
    monkeypatch.setattr(links, 'SVGS', [svg])
    problems = links.rewrite(None, 'abc123', lenient=True)
    text = svg.read_text()
    assert f'{links.REPO}/blob/abc123/pkg/mod.py#L7-L11' in text
    assert f'xlink:href="{links.REPO}/blob/abc123/pkg/mod.py" data-target="{bad}"' in text
    assert len(problems) == 1 and 'renamed-away' in problems[0]


def test_committed_diagram_targets_all_resolve():
    """Every link in arch_final.json resolves exactly once in this checkout."""
    spec = importlib.util.spec_from_file_location('code_links_real', _FIGURES / 'code_links.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    errors = mod.validate(None)
    assert errors == []
    assert len(mod.model_links()) > 10
