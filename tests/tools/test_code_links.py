"""Unit tests for ``tools/figures/code_links.py``.

Checks that banners, comments and methods resolve to their own line, that a
missing or ambiguous target raises, that other links pass through, that the
deploy drops a broken link while a strict run keeps it, and that every link of
the committed diagram resolves.
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


def test_banner_resolves_to_its_line_not_to_mentions(links):
    """The banner is line 7; the docstring (line 1) and closing banner (line 11) do not count."""
    assert links.resolve_target('pkg/mod.py', 'escape') == 7
    # the comment on line 8 slugs to 'escape-is-cheap-here', not 'escape'
    assert links.resolve_target('pkg/mod.py', 'escape-is-cheap-here') == 8


def test_comment_resolves_to_its_own_line(links):
    """A comment before an if block is line 13, not the if header on line 14."""
    assert links.resolve_target('pkg/mod.py', 'apply-impacts-now') == 13
    assert links.resolve_target('pkg/mod.py', 'apply-impacts-now-then-carry-on') == 13
    with pytest.raises(links.LinkError, match='no function or comment'):
        links.resolve_target('pkg/mod.py', 'then-carry-on')


def test_method_resolves_by_qualified_name(links):
    """Runner.step resolves to its def on line 5; a wrong or bare name does not."""
    assert links.resolve_target('pkg/mod.py', 'Runner.step') == 5
    assert links.resolve_target('pkg/mod.py', 'Runner') == 4
    with pytest.raises(links.LinkError, match='no function or comment'):
        links.resolve_target('pkg/mod.py', 'Runner.stop')
    with pytest.raises(links.LinkError, match='no function or comment'):
        links.resolve_target('pkg/mod.py', 'step')


def test_missing_and_duplicate_targets_raise(links, tmp_path):
    """A target that matches zero or several times, or a missing file, is an error."""
    (tmp_path / 'pkg' / 'dup.py').write_text('# Same note\nx = 1\n# Same note\ny = 2\n')
    (tmp_path / 'pkg' / 'twice.py').write_text('def f():\n    pass\n\n\ndef f():\n    pass\n')
    with pytest.raises(links.LinkError, match='2 comments match'):
        links.resolve_target('pkg/dup.py', 'same-note')
    with pytest.raises(links.LinkError, match='2 definitions share'):
        links.resolve_target('pkg/twice.py', 'f')
    with pytest.raises(links.LinkError, match='does not exist'):
        links.resolve_target('pkg/gone.py', 'escape')


def test_unclosed_banner_still_resolves(links, tmp_path):
    """A link points at one line, so a banner without its closing banner is not an error."""
    text = 'x = 0\n############### OPEN\nx = 1\n'
    (tmp_path / 'pkg' / 'open.py').write_text(text)
    line = links.resolve_target('pkg/open.py', 'open')
    assert line == 2
    assert text.splitlines()[line - 1] == '############### OPEN'


def test_line_anchors_and_foreign_links_pass_through(links):
    """Only name-based links into the repository are resolved, and they point at main."""
    hard = links.REPO_URL + 'pkg/mod.py#L12'
    other = 'https://example.org/page#section'
    assert links.resolve(hard) == hard
    assert links.resolve(other) == other
    assert links.resolve(links.REPO_URL + 'pkg/mod.py#escape') == (
        f'{links.REPO}/blob/main/pkg/mod.py#L7'
    )


def test_file_link_must_name_an_existing_file(links):
    """A link to a repository file passes through only while the file exists."""
    assert links.resolve(links.REPO_URL + 'pkg/mod.py') == links.REPO_URL + 'pkg/mod.py'
    with pytest.raises(links.LinkError, match='pkg/gone.py: file does not exist'):
        links.resolve(links.REPO_URL + 'pkg/gone.py')


def test_deploy_rewrite_drops_link_of_broken_target(links, tmp_path, monkeypatch):
    """--deploy links a good target to main and leaves a broken one without an href."""
    good = links.REPO_URL + 'pkg/mod.py#escape'
    bad = links.REPO_URL + 'pkg/mod.py#renamed-away'
    svg = tmp_path / 'fig.svg'
    svg.write_text(
        f'<svg><a xlink:href="x" data-target="{good}"/>'
        f'<a xlink:href="y" data-target="{bad}"/></svg>'
    )
    monkeypatch.setattr(links, 'SVGS', [svg])
    problems = links.rewrite(None, lenient=True)
    text = svg.read_text()
    assert f'xlink:href="{links.REPO}/blob/main/pkg/mod.py#L7" data-target="{good}"' in text
    assert f'<a data-target="{bad}"/>' in text
    assert 'xlink:href="y"' not in text
    assert len(problems) == 1 and 'renamed-away' in problems[0]

    # once the target exists again, a later run restores the link
    (tmp_path / 'pkg' / 'mod.py').write_text(SAMPLE + '# Renamed away\n')
    links.source.cache_clear()
    assert links.rewrite(None, lenient=True) == []
    assert f'xlink:href="{links.REPO}/blob/main/pkg/mod.py#L19" data-target="{bad}"' in (
        svg.read_text()
    )


def test_strict_rewrite_keeps_broken_link_unchanged(links, tmp_path, monkeypatch):
    """Without --deploy a broken target is reported and its old href kept."""
    good = links.REPO_URL + 'pkg/mod.py#Runner.step'
    bad = links.REPO_URL + 'pkg/mod.py#renamed-away'
    svg = tmp_path / 'fig.svg'
    svg.write_text(
        f'<svg><a xlink:href="x" data-target="{good}"/>'
        f'<a xlink:href="y" data-target="{bad}"/></svg>'
    )
    monkeypatch.setattr(links, 'SVGS', [svg])
    problems = links.rewrite(None, lenient=False)
    text = svg.read_text()
    assert f'xlink:href="{links.REPO}/blob/main/pkg/mod.py#L5"' in text
    assert f'<a xlink:href="y" data-target="{bad}"/>' in text
    assert len(problems) == 1 and 'renamed-away' in problems[0]


def test_committed_diagram_targets_all_resolve():
    """Every link in arch_final.json resolves exactly once in this checkout."""
    spec = importlib.util.spec_from_file_location('code_links_real', _FIGURES / 'code_links.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    errors = mod.validate(None)
    assert errors == []
    assert len(mod.model_links()) > 10
