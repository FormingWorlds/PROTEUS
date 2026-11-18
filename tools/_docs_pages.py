"""Compare the images that the tutorial pages embed with the figure manifest.

`check_pages` is the `--check` mode of `tools/make_docs_figures.py`: every image of a
page under the pages folder of the manifest needs an entry, every entry needs its files,
and a page embeds the files of an entry as a pair, `<asset>.avif#only-light` next to
`<asset>_dark.avif#only-dark`. `_page_images` reads the Markdown forms that the pages
use; it is not a full Markdown parser.
"""

from __future__ import annotations

import re
from pathlib import Path

# File name suffix of each variant; a page embeds a variant with `#only-<variant>`.
SUFFIX = {'light': '', 'dark': '_dark'}
# Alt text of an image, with one level of brackets inside it
ALT = r'(?:[^\[\]\n]|\[[^\[\]\n]*\])*'
# An inline image, a reference-style image and an <img> tag, in page order
IMAGE = re.compile(
    rf'!\[{ALT}\]\(\s*(?:<([^<>\n]+)>|((?:[^\s<>()]|\([^\s<>()]*\))+))'
    rf'|!\[({ALT})\]\[([^\]\n]*)\]'
    r'|<img\b[^>]*?(?<![\w-])src\s*=\s*["\']?([^"\'\s>]+)',
    re.IGNORECASE,
)
DEFINITION = re.compile(r'^ {0,3}\[([^\]\n]+)\]:\s*<?([^\s<>]+)', re.MULTILINE)
NOT_CONTENT = re.compile(r'<!--.*?-->|`[^`\n]*`', re.DOTALL)
FENCE = re.compile(r'\s*(`{3,}|~{3,})')


def _repo_target(page: Path, target: str, repo: Path) -> str:
    """Path of an embedded image from the repository root, with its `#fragment` kept.

    A web address or a path that leaves the repository is returned as written.
    """
    path, _, fragment = target.partition('#')
    resolved = (page.parent / path).resolve()
    if '://' in path or not resolved.is_relative_to(repo.resolve()):
        return target
    return resolved.relative_to(repo.resolve()).as_posix() + (
        f'#{fragment}' if fragment else ''
    )


def _page_images(text: str) -> list[str]:
    """Targets of the images that a Markdown page embeds, in page order.

    Reads inline images, reference-style images and `<img>` tags. Fenced code (a fence
    without an end runs to the end of the page), inline code and HTML comments are not
    page content.
    """
    lines, fence = [], None
    for line in text.splitlines():
        mark = FENCE.match(line)
        if fence is None and mark:
            fence = mark.group(1)
        elif fence and mark and mark.group(1).startswith(fence) and not line.strip(' \t`~'):
            fence = None
        elif fence is None:
            lines.append(line)
    text = NOT_CONTENT.sub('', '\n'.join(lines))
    definitions = {label.lower(): target for label, target in DEFINITION.findall(text)}
    targets = []
    for angle, plain, alt, label, src in IMAGE.findall(text):
        reference = definitions.get((label or alt).lower()) if alt or label else None
        targets.append((angle or plain or src or reference or '').strip())
    return [target for target in targets if target]


def _in_pairs(embeds: list[str], pair: list[str]) -> bool:
    """True when `pair` occurs at least once in `embeds` and its files occur nowhere else."""
    size = len(pair)
    pairs = sum(embeds[i : i + size] == pair for i in range(len(embeds)))
    return pairs > 0 and sum(embeds.count(name) for name in pair) == size * pairs


def check_pages(manifest: dict, repo: Path) -> list[str]:
    """Compare the manifest with the images that the tutorial pages embed.

    Parameters
    ----------
    manifest : dict
        Manifest from `load_manifest`.
    repo : Path
        Repository root.

    Returns
    -------
    list[str]
        One line per problem: an exempt page that does not exist, an embedded image
        without an entry, an entry without its files, or an entry that its page does
        not embed as a light and dark pair on adjacent embeds. Empty when all agree.
    """
    pages_dir = repo / manifest['pages']
    problems = [
        f'exempt page {name} does not exist'
        for name in manifest['exempt_pages']
        if not (pages_dir / name).is_file()
    ]
    wanted = {}
    for fig in manifest['figure']:
        pair = [f'{fig["asset"]}{SUFFIX[v]}.avif#only-{v}' for v in fig['variants']]
        wanted.setdefault(fig['page'], []).append((fig['id'], pair))
        for target in pair:
            if not (repo / target.split('#')[0]).is_file():
                problems.append(f'{fig["id"]}: file {target.split("#")[0]} is missing')
        if fig['page'] in manifest['exempt_pages'] or not (pages_dir / fig['page']).is_file():
            problems.append(f'{fig["id"]}: page {fig["page"]} is exempt or does not exist')
    for page in sorted(pages_dir.rglob('*.md')):
        name = page.relative_to(pages_dir).as_posix()
        if name in manifest['exempt_pages']:
            continue
        embeds = [_repo_target(page, t, repo) for t in _page_images(page.read_text())]
        entries = wanted.get(name, [])
        known = {target for _, pair in entries for target in pair}
        problems += [
            f'{name}: image {e} has no manifest entry' for e in embeds if e not in known
        ]
        for fig_id, pair in entries:
            if not _in_pairs(embeds, pair):
                problems.append(
                    f'{fig_id}: {name} does not embed {" and ".join(pair)} as a pair'
                )
    return problems
