"""Resolve name-based code links in the architecture diagram to line ranges.

A diagram link names its target instead of a line number::

    https://github.com/FormingWorlds/PROTEUS/blob/main/<path>#<target>

where ``<target>`` is one of

* a function, class or method, by qualified name (``run_zephyrus``,
  ``Proteus.start``): resolves to its definition;
* a comment, as a slug of its text or of its first sentence
  (``interior`` for ``############### INTERIOR``, ``run-offline-chemistry``
  for ``# Run offline chemistry``).  A section banner resolves to the range up
  to its closing ``############### /`` banner and takes precedence over other
  comments; any other comment resolves to the comment and the statement that
  follows it.  Only real comments count, not ``#`` inside strings.

Each target must match exactly once.  Line numbers are never stored: the
documentation deploy resolves every link against the commit it publishes and
pins the link to that commit, so the published diagram always points at the
lines of the code it was built with.

Usage:
    python3 tools/figures/code_links.py --check         # every target resolves (PR CI)
    python3 tools/figures/code_links.py --commit SHA    # pin links to SHA (docs deploy)
    python3 tools/figures/code_links.py                 # refresh the committed SVGs
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
SVG_LINK = re.compile(r'<a xlink:href="([^"]*)" data-target="([^"]*)"')


class LinkError(Exception):
    pass


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


@lru_cache(maxsize=None)
def source(path: str, ref: str | None) -> str:
    if ref is None:
        file = ROOT / path
        if not file.is_file():
            raise LinkError(f'{path}: file does not exist')
        return file.read_text()
    res = subprocess.run(
        ['git', '-C', str(ROOT), 'show', f'{ref}:{path}'], capture_output=True, text=True
    )
    if res.returncode:
        raise LinkError(f'{path} not found at {ref}: {res.stderr.strip()}')
    return res.stdout


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


def statement_after(tree: ast.AST, line: int) -> int:
    """Last line of the outermost statement that starts first below ``line``."""
    best = None
    for node in ast.walk(tree):  # breadth first: an outer statement is seen first
        if isinstance(node, ast.stmt) and node.lineno > line:
            if best is None or node.lineno < best.lineno:
                best = node
    if best is None:
        raise LinkError(f'no statement follows line {line}')
    return best.end_lineno


def resolve_target(path: str, target: str, ref: str | None = None) -> tuple[int, int]:
    text = source(path, ref)
    tree = ast.parse(text)
    defs = [(a, b) for name, a, b in definitions(tree) if name == target]
    if len(defs) == 1:
        return defs[0]
    if len(defs) > 1:
        raise LinkError(f'{path}#{target}: {len(defs)} definitions share this name')

    found = list(comments(text))
    banners, others = [], []
    for i, tok in found:
        m = BANNER.match(tok)
        if m:
            if not m.group(1).startswith('/') and slug(m.group(1)) == target:
                banners.append(i)
            continue
        body = tok.lstrip('#').strip()
        first_sentence = re.split(r'(?<=\.)\s', body)[0]
        if target in (slug(body), slug(first_sentence)):
            others.append(i)
    hits = banners or others  # a section banner wins over a passing mention
    if len(hits) != 1:
        what = 'no function or comment matches' if not hits else f'{len(hits)} comments match'
        raise LinkError(f'{path}#{target}: {what} this target')
    start = hits[0]
    if banners:
        for i, tok in found:
            m = BANNER.match(tok)
            if i > start and m:
                if m.group(1).startswith('/'):
                    return start, i
                raise LinkError(f'{path}#{target}: next banner opens before this one closes')
        raise LinkError(f'{path}#{target}: section banner is never closed')
    return start, statement_after(tree, start)


def split(href: str) -> tuple[str, str] | None:
    """(path, target) of a name-based link, None for any other link."""
    if not href.startswith(REPO_URL) or '#' not in href:
        return None
    path, target = href[len(REPO_URL) :].split('#', 1)
    return None if LINE_ANCHOR.match(target) else (path, target)


def is_symbolic(href: str) -> bool:
    return split(href) is not None


def resolve(href: str, ref: str | None = None, commit: str = 'main') -> str:
    """Turn a name-based link into a line-range link; other links pass through."""
    parts = split(href)
    if parts is None:
        return href
    path, target = parts
    a, b = resolve_target(path, target, ref)
    return f'{REPO}/blob/{commit}/{path}#L{a}-L{b}'


def model_links() -> list[str]:
    items = json.loads((HERE / 'arch_final.json').read_text())['items']
    return sorted({it['href'] for it in items if it.get('href')})


def validate(ref: str | None) -> list[str]:
    errors = []
    for href in model_links():
        if href.startswith(REPO) and LINE_ANCHOR.match(href.partition('#')[2]):
            errors.append(f'hard-coded line anchor in arch_final.json: {href}')
            continue
        try:
            resolve(href, ref)
        except (LinkError, SyntaxError) as e:
            errors.append(str(e))
    for svg in SVGS:
        if not SVG_LINK.search(svg.read_text()):
            errors.append(f'{svg.name}: no name-based links; rebuild the figure')
    return errors


def rewrite(ref: str | None, commit: str, lenient: bool) -> list[str]:
    """Rewrite the links in the SVGs; with ``lenient``, an unresolvable target
    falls back to the file itself instead of failing."""
    problems = []
    for svg in SVGS:
        text = svg.read_text()

        def fix(m):
            target = unescape(m.group(2))
            try:
                new = resolve(target, ref, commit)
            except (LinkError, SyntaxError) as e:
                problems.append(str(e))
                if not lenient:
                    return m.group(0)
                path = split(target)[0]
                new = f'{REPO}/blob/{commit}/{path}'
            return f'<a xlink:href="{escape(new, quote=True)}" data-target="{m.group(2)}"'

        new_text = SVG_LINK.sub(fix, text)
        if new_text != text:
            svg.write_text(new_text)
            print(f'updated links in {svg.relative_to(ROOT)}')
    return sorted(set(problems))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='fail if a target does not resolve')
    mode.add_argument('--commit', help='pin the links to this commit; never fails')
    ap.add_argument('--ref', help='resolve against this git ref instead of the work tree')
    args = ap.parse_args()

    if args.check:
        errors = validate(args.ref)
        for e in errors:
            print(f'error: {e}', file=sys.stderr)
        if errors:
            print(
                'Fix the target in draw.io and arch_final.json, or restore the '
                'renamed function or comment (see tools/figures/README.md).',
                file=sys.stderr,
            )
        return 1 if errors else 0

    lenient = args.commit is not None
    problems = rewrite(args.ref, args.commit or 'main', lenient)
    for p in problems:
        # a GitHub annotation in the docs deploy, a plain message elsewhere
        print(f'::warning::{p}' if lenient else f'error: {p}', file=sys.stderr)
    return 0 if lenient or not problems else 1


if __name__ == '__main__':
    sys.exit(main())
