"""Resolve the name-based code links of the architecture diagram to line links.

A link names its target instead of a line, as
``https://github.com/FormingWorlds/PROTEUS/blob/main/<path>#<target>``, where
``<target>`` is a function, class or method (``Proteus.start``), the slug of a
section banner (``interior`` for ``############### INTERIOR``) or, with the
prefix ``comment-``, the slug of another comment. It must match exactly once.

Usage:
    python3 tools/figures/code_links.py --check    # every target resolves (PR CI)
    python3 tools/figures/code_links.py --deploy   # resolve against this checkout (docs deploy)
    python3 tools/figures/code_links.py            # resolve against origin/main
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from functools import lru_cache
from html import escape, unescape
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
REPO = 'https://github.com/FormingWorlds/PROTEUS'
REPO_URL = f'{REPO}/blob/main/'
SVGS = [
    ROOT / 'docs/assets/proteus_architecture.svg',
    ROOT / 'docs/assets/proteus_architecture_darkmode.svg',
]
LINE_ANCHOR = re.compile(r'^L\d+(-L\d+)?$')
BANNER = re.compile(r'^#{5,}\s*(.*?)\s*$')
SVG_LINK = re.compile(r'<a(?: xlink:href="([^"]*)")? data-target="([^"]*)"')


class LinkError(Exception):
    pass


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def ref_problem(ref: str) -> str | None:
    """Why git does not know ``ref`` as a commit, None when it does."""
    res = subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}'],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return None
    if res.returncode == 1:
        hint = '; run git fetch origin main' if ref.startswith('origin/') else ''
        return f'{ref} is not a commit of this repository{hint}'
    return f'git rev-parse failed for {ref}: {res.stderr.strip()}'


def require_file(path: str, ref: str | None) -> None:
    """Raise LinkError unless ``path`` is a plain path to a file in the work tree
    (``ref`` None) or at ``ref``."""
    if {'', '.', '..'} & set(path.split('/')):
        raise LinkError(f'{path}: not a plain path inside the repository')
    if ref is None:
        found = (ROOT / path).is_file()
    else:
        res = subprocess.run(
            ['git', '-C', str(ROOT), 'cat-file', '-t', f'{ref}:{path}'],
            capture_output=True,
            text=True,
        )
        found = res.stdout.strip() == 'blob'
    if not found:
        raise LinkError(f'{path}: file does not exist' + (f' at {ref}' if ref else ''))


@lru_cache(maxsize=None)
def source(path: str, ref: str | None) -> str:
    require_file(path, ref)
    try:
        if ref is None:
            return (ROOT / path).read_text()
        git = ['git', '-C', str(ROOT), 'show', f'{ref}:{path}']
        return subprocess.run(git, capture_output=True, text=True).stdout
    except UnicodeDecodeError:
        raise LinkError(f'{path}: not a text file') from None


def definitions(tree: ast.AST):
    """Yield (qualified name, first line, last line) of every def and class."""
    stack = [(tree, '')]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                yield name, child.lineno, child.end_lineno
                stack.append((child, name + '.'))
            else:
                stack.append((child, prefix))


def comments(text: str):
    """Yield (line, text) of every comment, from the tokenizer, not inside strings."""
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.start[0], tok.string


def resolve_target(path: str, target: str, ref: str | None = None) -> int:
    """Line of the definition, banner or (with ``comment-``) comment that ``target`` names."""
    text = source(path, ref)
    tree = ast.parse(text)
    defs = [a for name, a, _ in definitions(tree) if name == target]
    if len(defs) == 1:
        return defs[0]
    if len(defs) > 1:
        raise LinkError(f'{path}#{target}: {len(defs)} definitions share this name')

    name = target.removeprefix('comment-')
    banners, others = [], []
    for i, tok in comments(text):
        m = BANNER.match(tok)
        if m:
            if not m.group(1).startswith('/') and slug(m.group(1)) == name:
                banners.append(i)
            if not m.group(1).startswith('/') and slug(m.group(1)) == target != name:
                raise LinkError(
                    f'{path}#{target}: the banner on line {i} reads as a comment link'
                )
            continue
        body = tok.lstrip('#').strip()
        first_sentence = re.split(r'(?<=\.)\s', body)[0]
        if name in (slug(body), slug(first_sentence)):
            others.append(i)
    hits = others if name != target else banners
    if len(hits) == 1:
        return hits[0]
    if hits:
        kind = 'comments' if name != target else 'banners'
        raise LinkError(f'{path}#{target}: {len(hits)} {kind} match this target')
    hint = ''
    if name == target and others:
        hint = f'; line {others[0]} is a comment, not a banner: link it as #comment-{name}'
    elif name != target and banners:
        hint = f'; line {banners[0]} is a banner: link it as #{name}'
    what = 'comment' if name != target else 'function or banner'
    raise LinkError(f'{path}#{target}: no {what} matches this target{hint}')


def split(href: str) -> tuple[str, str] | None:
    """(path, target) of a name-based link, None for any other link."""
    if not href.startswith(REPO_URL) or '#' not in href:
        return None
    path, target = href[len(REPO_URL) :].split('#', 1)
    return None if LINE_ANCHOR.match(target) else (path.partition('?')[0], target)


def resolve(href: str, ref: str | None = None) -> str:
    """Line link for a name-based link, read from ``ref`` (None: the work tree);
    other links pass through, a link to a repository file only when the file exists."""
    parts = split(href)
    if parts is None:
        if href.startswith(REPO_URL) and '#' not in href:
            require_file(href[len(REPO_URL) :].partition('?')[0], ref)
        return href
    path, target = parts
    return f'{REPO_URL}{path}#L{resolve_target(path, target, ref)}'


def model_links() -> list[str]:
    items = json.loads((HERE / 'arch_final.json').read_text())['items']
    return sorted({it['href'] for it in items if it.get('href')})


def validate(ref: str | None) -> list[str]:
    errors, model = [], set(model_links())
    for href in sorted(model):
        if href.startswith(REPO) and LINE_ANCHOR.match(href.partition('#')[2]):
            errors.append(f'hard-coded line anchor in arch_final.json: {href}')
            continue
        try:
            resolve(href, ref)
        except (LinkError, SyntaxError) as e:
            errors.append(f'{e} [{href}]')
    for svg in SVGS:
        targets = {unescape(m.group(2)) for m in SVG_LINK.finditer(svg.read_text())}
        if not targets:
            errors.append(f'{svg.name}: no name-based links; rebuild the figure')
            continue
        for what, links in (
            ('missing', model - targets),
            ('not in arch_final.json', targets - model),
        ):
            if links:
                errors.append(f'{svg.name}: links {what}: {", ".join(sorted(links))}')
    return errors


def rewrite(ref: str | None, lenient: bool) -> list[str]:
    """Re-resolve the links in the SVGs. A broken target loses its link when
    ``lenient`` and is left as it was otherwise."""
    problems = []
    for svg in SVGS:
        text = svg.read_text()

        def fix(m):
            target = unescape(m.group(2))
            try:
                new = resolve(target, ref)
            except (LinkError, SyntaxError) as e:
                problems.append(f'{e} [{target}]')
                if not lenient:
                    return m.group(0)
                return f'<a data-target="{m.group(2)}"'  # no href: not clickable
            return f'<a xlink:href="{escape(new, quote=True)}" data-target="{m.group(2)}"'

        new_text = SVG_LINK.sub(fix, text)
        if new_text != text:
            svg.write_text(new_text)
            print(f'updated links in {svg.relative_to(ROOT)}')
    return sorted(set(problems))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='fail if a target does not resolve')
    mode.add_argument(
        '--deploy',
        action='store_true',
        help='resolve against the work tree (a main checkout); a missing target loses its link',
    )
    ap.add_argument('--ref', help='git ref to read the code from (default origin/main)')
    args = ap.parse_args(argv)
    if args.ref == '':
        ap.error('--ref must name a commit')
    lenient = args.deploy
    ref = args.ref if args.check or lenient else (args.ref or 'origin/main')
    if ref is not None and (problem := ref_problem(ref)):
        if lenient:
            print(f'::warning::{problem}; the SVG links are left as they are')
            return 0
        print(f'error: {problem}', file=sys.stderr)
        return 1

    if args.check:
        errors = validate(args.ref)
        for e in errors:
            print(f'error: {e}', file=sys.stderr)
        if errors:
            print(
                'Update the link in draw.io and arch_final.json (tools/figures/README.md).',
                file=sys.stderr,
            )
        return 1 if errors else 0

    problems = rewrite(ref, lenient)
    for p in problems:
        if lenient:
            print(f'::warning::{p}; this box has no link until the diagram is updated')
        else:
            print(f'error: {p}', file=sys.stderr)
    return 0 if lenient or not problems else 1


if __name__ == '__main__':
    sys.exit(main())
