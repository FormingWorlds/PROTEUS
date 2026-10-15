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


def _commit_tree(root):
    """Make ``root`` a git repository with one commit of everything in it."""
    import subprocess

    git = ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t']
    for step in (
        ['init', '-q'],
        ['add', '.'],
        ['-c', 'commit.gpgsign=false', 'commit', '-qm', 's'],
    ):
        subprocess.run([*git, *step], check=True)


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
    assert links.resolve_target('pkg/mod.py', 'comment-escape-is-cheap-here') == 8


def test_comment_resolves_to_its_own_line(links):
    """A comment before an if block is line 13, not the if header on line 14."""
    assert links.resolve_target('pkg/mod.py', 'comment-apply-impacts-now') == 13
    assert links.resolve_target('pkg/mod.py', 'comment-apply-impacts-now-then-carry-on') == 13
    with pytest.raises(links.LinkError, match='no comment matches'):
        links.resolve_target('pkg/mod.py', 'comment-then-carry-on')


def test_banner_and_comment_targets_never_stand_in_for_each_other(links, tmp_path):
    """A renamed banner fails, naming the comment that would have caught the link; a
    comment target does not resolve to a banner of the same name."""
    renamed = SAMPLE.replace('############### ESCAPE\n', '############### LOSSES\n', 1)
    (tmp_path / 'pkg' / 'renamed.py').write_text(renamed + '# Escape\n')
    with pytest.raises(links.LinkError) as err:
        links.resolve_target('pkg/renamed.py', 'escape')
    assert str(err.value) == (
        'pkg/renamed.py#escape: no function or banner matches this target; '
        'line 19 is a comment, not a banner: link it as #comment-escape'
    )
    with pytest.raises(links.LinkError, match='line 7 is a banner: link it as #escape'):
        links.resolve_target('pkg/mod.py', 'comment-escape')


def test_method_resolves_by_qualified_name(links):
    """Runner.step resolves to its def on line 5; a wrong or bare name does not."""
    assert links.resolve_target('pkg/mod.py', 'Runner.step') == 5
    assert links.resolve_target('pkg/mod.py', 'Runner') == 4
    with pytest.raises(links.LinkError, match='no function or banner'):
        links.resolve_target('pkg/mod.py', 'Runner.stop')
    with pytest.raises(links.LinkError, match='no function or banner'):
        links.resolve_target('pkg/mod.py', 'step')


def test_missing_and_duplicate_targets_raise(links, tmp_path):
    """A target that matches zero or several times, or a missing file, is an error."""
    (tmp_path / 'pkg' / 'dup.py').write_text('# Same note\nx = 1\n# Same note\ny = 2\n')
    (tmp_path / 'pkg' / 'twice.py').write_text('def f():\n    pass\n\n\ndef f():\n    pass\n')
    with pytest.raises(links.LinkError, match='2 comments match'):
        links.resolve_target('pkg/dup.py', 'comment-same-note')
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


def test_file_link_checks_existence_without_reading_the_file(links, tmp_path):
    """A text or binary file passes and a query is ignored; a directory, a missing
    file or a path that is not plain fails; in the work tree and at a git ref alike."""
    (tmp_path / 'pkg' / 'logo.png').write_bytes(b'\x89PNG\r\n\x1a\n\x00\xff')
    _commit_tree(tmp_path)
    for ref in (None, 'HEAD'):
        for ok in ('pkg/mod.py', 'pkg/logo.png', 'pkg/mod.py?plain=1'):
            assert links.resolve(links.REPO_URL + ok, ref) == links.REPO_URL + ok
        named = links.REPO_URL + 'pkg/mod.py?plain=1#escape'
        assert links.resolve(named, ref) == links.REPO_URL + 'pkg/mod.py#L7'
        for bad, path in (
            ('pkg', 'pkg'),
            ('pkg/gone.py', 'pkg/gone.py'),
            ('pkg#escape', 'pkg'),
        ):
            with pytest.raises(links.LinkError, match=f'^{path}: file does not exist'):
                links.resolve(links.REPO_URL + bad, ref)
        with pytest.raises(links.LinkError, match='pkg/logo.png: not a text file'):
            links.resolve(links.REPO_URL + 'pkg/logo.png#escape', ref)
        for odd in ('/etc/hosts', 'pkg/../pkg/mod.py', 'pkg/mod.py/', 'pkg//mod.py'):
            with pytest.raises(links.LinkError, match='not a plain path'):
                links.resolve(links.REPO_URL + odd, ref)


def test_banner_and_comment_of_one_slug_resolve_by_kind(links, tmp_path):
    """With a banner and a comment of the same slug each kind finds its own line;
    two banners of one slug, or a banner that reads as a comment link, are errors."""
    (tmp_path / 'pkg' / 'both.py').write_text('############### ESCAPE\nx = 1\n# Escape\n')
    assert links.resolve_target('pkg/both.py', 'escape') == 1
    assert links.resolve_target('pkg/both.py', 'comment-escape') == 3
    (tmp_path / 'pkg' / 'two.py').write_text('############### A\nx = 1\n############### A\n')
    with pytest.raises(links.LinkError, match='2 banners match'):
        links.resolve_target('pkg/two.py', 'a')
    (tmp_path / 'pkg' / 'clash.py').write_text('############### COMMENT NOTES\n# Notes\n')
    with pytest.raises(links.LinkError, match='banner on line 1 reads as a comment link'):
        links.resolve_target('pkg/clash.py', 'comment-notes')
    (tmp_path / 'pkg' / 'closing.py').write_text('############### / COMMENT NOTES\n# Notes\n')
    assert links.resolve_target('pkg/closing.py', 'comment-notes') == 2


def test_main_checks_the_ref_once(links, tmp_path, monkeypatch, capsys):
    """An unknown ref gives one error before any link is read (with the fetch hint for
    an origin/ ref); a deploy then warns, leaves the SVGs as they are and succeeds; any
    other git failure shows git's message; an empty --ref is refused."""
    svg = tmp_path / 'fig.svg'
    svg.write_text(
        f'<svg><a xlink:href="x" data-target="{links.REPO_URL}pkg/mod.py#escape"/></svg>'
    )
    monkeypatch.setattr(links, 'SVGS', [svg])
    assert links.ref_problem('HEAD').startswith('git rev-parse failed for HEAD: fatal')
    _commit_tree(tmp_path)
    assert links.ref_problem('HEAD') is None
    assert links.main(['--check', '--ref', 'nonexist']) == 1
    assert capsys.readouterr().err == 'error: nonexist is not a commit of this repository\n'
    assert links.main(['--ref', 'origin/nope']) == 1
    assert capsys.readouterr().err.endswith('; run git fetch origin main\n')
    assert links.main(['--deploy', '--ref', 'nonexist']) == 0
    assert capsys.readouterr().out == (
        '::warning::nonexist is not a commit of this repository; the SVG links are left as they are\n'
    )
    assert 'xlink:href="x"' in svg.read_text()
    with pytest.raises(SystemExit):
        links.main(['--ref', ''])


def test_validate_reports_missing_file_and_svg_out_of_step(links, tmp_path, monkeypatch):
    """--check names a link to a missing file, and per SVG the model links it lacks,
    the links it has beyond the model (escaped as in the SVG), or no links at all."""
    good, gone = links.REPO_URL + 'pkg/mod.py#escape', links.REPO_URL + 'pkg/mod.py?a=1&b=2'
    monkeypatch.setattr(links, 'model_links', lambda: sorted([good, gone]))
    svgs = [tmp_path / name for name in ('a.svg', 'b.svg', 'c.svg')]
    svgs[0].write_text(f'<svg><a xlink:href="x" data-target="{good}"/></svg>')
    extra = links.REPO_URL + 'pkg/mod.py#Runner'
    svgs[1].write_text(
        f'<svg><a data-target="{good}"/><a data-target="{gone.replace("&", "&amp;")}"/>'
        f'<a data-target="{extra}"/></svg>'
    )
    svgs[2].write_text('<svg></svg>')
    monkeypatch.setattr(links, 'SVGS', svgs)
    (tmp_path / 'pkg' / 'mod.py').unlink()
    links.source.cache_clear()
    before = [svg.read_text() for svg in svgs]
    assert links.validate(None) == [
        f'pkg/mod.py: file does not exist [{good}]',
        f'pkg/mod.py: file does not exist [{gone}]',
        f'a.svg: links missing: {gone}',
        f'b.svg: links not in arch_final.json: {extra}',
        'c.svg: no name-based links; rebuild the figure',
    ]
    assert [svg.read_text() for svg in svgs] == before  # --check writes nothing


def test_deploy_rewrite_drops_link_of_broken_target(links, tmp_path, monkeypatch):
    """--deploy links a good target to main and leaves a broken one without an href."""
    good = links.REPO_URL + 'pkg/mod.py#escape'
    bad = links.REPO_URL + 'pkg/mod.py#comment-renamed-away'
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
    assert len(problems) == 1 and problems[0].endswith(f' [{bad}]')

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
