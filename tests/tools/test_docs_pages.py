"""Unit tests for tools/_docs_pages.py: the tutorial pages against the figure manifest.

Each test builds a small repository under `tmp_path` with 1 page and the 2 files of 1 pair.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_script = Path(__file__).resolve().parents[2] / 'tools' / '_docs_pages.py'
_spec = importlib.util.spec_from_file_location('_docs_pages', _script)
pages = sys.modules.setdefault('_docs_pages', importlib.util.module_from_spec(_spec))
if not hasattr(pages, 'check_pages'):
    _spec.loader.exec_module(pages)

PAIR = (
    '![Alt](../assets/tutorials/fig_a.avif#only-light){ width="100%" }\n'
    '![Alt](../assets/tutorials/fig_a_dark.avif#only-dark){ width="100%" }\n'
)


def _entry(**changes) -> dict:
    """The part of a manifest entry that the page check reads."""
    entry = {'id': 'fig_a', 'page': 'page_a.md', 'asset': 'docs/assets/tutorials/fig_a'}
    return {**entry, 'variants': ['light', 'dark'], **changes}


def _manifest(figures: list[dict] | None = None, exempt: tuple[str, ...] = ()) -> dict:
    """A manifest as `load_manifest` of the figure tool returns it."""
    figures = [_entry()] if figures is None else figures
    return {'pages': 'docs/Tutorials', 'exempt_pages': list(exempt), 'figure': figures}


@pytest.fixture
def site(tmp_path):
    """A repository root with 1 page that embeds the pair of 1 entry."""
    repo = tmp_path / 'repo'
    (repo / 'docs' / 'Tutorials').mkdir(parents=True)
    (repo / 'docs' / 'Tutorials' / 'page_a.md').write_text(f'# Page A\n\n{PAIR}')
    assets = repo / 'docs' / 'assets' / 'tutorials'
    assets.mkdir(parents=True)
    for name in ('fig_a.avif', 'fig_a_dark.avif'):
        (assets / name).write_bytes(b'placeholder')
    return SimpleNamespace(repo=repo)


def test_check_passes_on_a_page_with_a_pair(site):
    """A page that embeds both files of its entry next to each other has no problem."""
    assert pages.check_pages(_manifest(), site.repo) == []
    # A link to a page or a file is not an embedded image
    page = site.repo / 'docs' / 'Tutorials' / 'page_a.md'
    page.write_text(page.read_text() + '\nSee [the data](../assets/tutorials/data.png).\n')
    assert pages.check_pages(_manifest(), site.repo) == []


@pytest.mark.parametrize(
    'page_text, problems',
    [
        (
            PAIR + '![Other](../assets/tutorials/other.png)\n',
            ['page_a.md: image docs/assets/tutorials/other.png has no manifest entry'],
        ),
        (
            PAIR + '![Web](https://example.org/a.png)\n',
            ['page_a.md: image https://example.org/a.png has no manifest entry'],
        ),
        (
            PAIR.splitlines()[0] + '\n',
            ['fig_a: page_a.md does not embed'],
        ),
        (
            '\n'.join(reversed(PAIR.splitlines())) + '\n',
            ['fig_a: page_a.md does not embed'],
        ),
        (
            PAIR.replace('\n', '\n![Other](../assets/tutorials/fig_a.avif#only-light)\n', 1),
            ['fig_a: page_a.md does not embed'],
        ),
        (
            PAIR.replace('#only-light', '').replace('#only-dark', ''),
            [
                'page_a.md: image docs/assets/tutorials/fig_a.avif has no manifest entry',
                'page_a.md: image docs/assets/tutorials/fig_a_dark.avif has no manifest entry',
                'fig_a: page_a.md does not embed',
            ],
        ),
        ('No figure here.\n', ['fig_a: page_a.md does not embed']),
    ],
)
def test_check_names_each_page_fault(site, page_text, problems):
    """An image without an entry, a single variant, a wrong order and a split pair are named."""
    (site.repo / 'docs' / 'Tutorials' / 'page_a.md').write_text(page_text)
    found = pages.check_pages(_manifest(), site.repo)
    assert len(found) == len(problems)
    assert all(expected in line for expected, line in zip(problems, found))


@pytest.mark.parametrize(
    'line, image',
    [
        (
            '![Other](../assets/tutorials/other.png "A title")',
            'docs/assets/tutorials/other.png',
        ),
        (
            '![T [K] against t]( ../assets/tutorials/other.png )',
            'docs/assets/tutorials/other.png',
        ),
        (
            '<img alt="x" src="../assets/tutorials/other.png" width="50%">',
            'docs/assets/tutorials/other.png',
        ),
        ("<IMG SRC='../assets/tutorials/other.png'>", 'docs/assets/tutorials/other.png'),
        (
            '<img data-src="lazy.png" src=../assets/tutorials/other.png>',
            'docs/assets/tutorials/other.png',
        ),
        (
            '![Other][ref]\n\n[ref]: ../assets/tutorials/other.png#only-light',
            'docs/assets/tutorials/other.png#only-light',
        ),
        ('![Ref][]\n\n[REF]: <../assets/tutorials/other>', 'docs/assets/tutorials/other'),
        ('![Other](<../assets/tutorials/other.png>)', 'docs/assets/tutorials/other.png'),
        ('![Other](<../assets/tutorials/my other.png>)', 'docs/assets/tutorials/my other.png'),
        ('![Other](../assets/tutorials/other(1).png)', 'docs/assets/tutorials/other(1).png'),
        (
            '  ```\n  code\n  ```\n![Other](../assets/tutorials/other.png)',
            'docs/assets/tutorials/other.png',
        ),
    ],
    ids=[
        'title',
        'brackets in the alt text',
        'img tag',
        'capitals',
        'unquoted src after data-src',
        'reference',
        'collapsed reference without a file extension',
        'angle',
        'angle with a space',
        'parentheses',
        'after an indented fence',
    ],
)
def test_check_sees_an_image_in_every_form_that_a_page_can_use(site, line, image):
    """An image with a title, in an HTML tag or behind a reference needs a manifest entry too."""
    page = site.repo / 'docs' / 'Tutorials' / 'page_a.md'
    page.write_text(f'{PAIR}\n{line}\n')
    manifest = _manifest()
    assert pages.check_pages(manifest, site.repo) == [
        f'page_a.md: image {image} has no manifest entry'
    ]
    (page.parent / 'more').mkdir()
    (page.parent / 'more' / 'deep.md').write_text('![Deep](../../assets/tutorials/deep.png)\n')
    assert pages.check_pages(manifest, site.repo) == [
        'more/deep.md: image docs/assets/tutorials/deep.png has no manifest entry',
        f'page_a.md: image {image} has no manifest entry',
    ]


@pytest.mark.parametrize(
    'text',
    [
        '```markdown\n![Example](../assets/tutorials/example.png)\n```',
        '~~~\n<img src="x.png">\n~~~',
        '    ```markdown\n    ![In a list](../assets/tutorials/example.png)\n    ```',
        '````\n```\n![Inner fence](../assets/tutorials/example.png)\n```\n````',
        '~~~\n```\n![Other fence character](../assets/tutorials/example.png)\n~~~',
        '```\n```python\n![After a fence line with text](../assets/tutorials/example.png)\n```',
        'Write `![x](../assets/tutorials/example.png)` for an image.',
        '<!-- ![x](../assets/tutorials/example.png)\n<img src="y.png"> -->',
        '<img data-src="lazy.png" alt="no src">',
        'See ![a][missing] and [the guide](other_page.md), or [ref2].\n\n[ref2]: other_page.md',
    ],
    ids=[
        'fence',
        'tilde fence',
        'indented fence',
        'longer fence',
        'fence of the other character',
        'fence line with text does not close',
        'inline code',
        'comment',
        'data-src only',
        'links and an undefined reference',
    ],
)
def test_check_reads_no_image_from_code_a_comment_or_a_link(site, text):
    """Code, a comment, a link and an image reference without a definition embed no image."""
    page = site.repo / 'docs' / 'Tutorials' / 'page_a.md'
    page.write_text(f'{text}\n\n{PAIR}')
    assert pages.check_pages(_manifest(), site.repo) == []
    assert pages._page_images(text) == []


def test_check_requires_that_the_files_of_a_pair_occur_only_as_a_pair(site):
    """The same pair twice is still a pair; a third light file alone is not. A fence
    without an end hides the rest of the page, as the site renders it."""
    page = site.repo / 'docs' / 'Tutorials' / 'page_a.md'
    page.write_text(PAIR + PAIR)
    assert pages.check_pages(_manifest(), site.repo) == []
    page.write_text(PAIR + PAIR.splitlines()[0] + '\n')
    problems = pages.check_pages(_manifest(), site.repo)
    assert len(problems) == 1 and 'does not embed' in problems[0]
    page.write_text(f'```\ncode without an end\n{PAIR}')
    assert pages._page_images(page.read_text()) == []
    assert 'does not embed' in pages.check_pages(_manifest(), site.repo)[0]


def test_check_names_a_missing_file_and_a_missing_page(site):
    """An entry without its dark file, or with a page that does not exist, is a problem."""
    (site.repo / 'docs' / 'assets' / 'tutorials' / 'fig_a_dark.avif').unlink()
    lost = _entry(id='fig_b', page='page_b.md', asset='docs/assets/tutorials/fig_b')
    (site.repo / 'docs' / 'assets' / 'tutorials' / 'fig_b.avif').write_bytes(b'x')
    found = pages.check_pages(_manifest([_entry(), lost]), site.repo)
    assert found == [
        'fig_a: file docs/assets/tutorials/fig_a_dark.avif is missing',
        'fig_b: file docs/assets/tutorials/fig_b_dark.avif is missing',
        'fig_b: page page_b.md is exempt or does not exist',
    ]
    (site.repo / 'docs' / 'assets' / 'tutorials' / 'fig_b_dark.avif').write_bytes(b'x')
    (site.repo / 'docs' / 'assets' / 'tutorials' / 'fig_a_dark.avif').write_bytes(b'x')
    assert pages.check_pages(_manifest([_entry(), lost]), site.repo) == [
        'fig_b: page page_b.md is exempt or does not exist'
    ]


def test_check_skips_an_exempt_page_and_requires_that_it_exists(site):
    """An exempt page can embed anything; an exempt page that does not exist is a problem."""
    tutorials = site.repo / 'docs' / 'Tutorials'
    (tutorials / 'old.md').write_text('![Old](../assets/tutorials/old.png)\n')
    assert pages.check_pages(_manifest(exempt=('old.md',)), site.repo) == []
    assert pages.check_pages(_manifest(), site.repo) == [
        'old.md: image docs/assets/tutorials/old.png has no manifest entry'
    ]
    manifest = _manifest(exempt=('old.md', 'gone.md', 'page_a.md'))
    assert pages.check_pages(manifest, site.repo) == [
        'exempt page gone.md does not exist',
        'fig_a: page page_a.md is exempt or does not exist',
    ]


def test_single_variant_entry_needs_its_one_file_only(site):
    """An entry with the dark variant alone is met by the dark embed and the dark file."""
    page = site.repo / 'docs' / 'Tutorials' / 'page_a.md'
    page.write_text(PAIR.splitlines()[1] + '\n')
    (site.repo / 'docs' / 'assets' / 'tutorials' / 'fig_a.avif').unlink()
    assert pages.check_pages(_manifest([_entry(variants=['dark'])]), site.repo) == []
    assert pages.check_pages(_manifest(), site.repo) == [
        'fig_a: file docs/assets/tutorials/fig_a.avif is missing',
        (
            'fig_a: page_a.md does not embed docs/assets/tutorials/fig_a.avif#only-light and '
            'docs/assets/tutorials/fig_a_dark.avif#only-dark as a pair'
        ),
    ]
