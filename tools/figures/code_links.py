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
  follows it.

The resolved link is ``...#L<first>-L<last>`` against the given source tree.
Each target must match exactly once, so a renamed function or comment fails
the build instead of producing a link to the wrong lines.

Usage:
    python3 tools/figures/code_links.py            # rewrite links in docs/assets
    python3 tools/figures/code_links.py --check    # fail if any link is stale
    python3 tools/figures/code_links.py --ref origin/main
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from functools import lru_cache
from html import escape, unescape
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
REPO_URL = 'https://github.com/FormingWorlds/PROTEUS/blob/main/'
SVGS = [
    ROOT / 'docs/assets/proteus_architecture.svg',
    ROOT / 'docs/assets/proteus_architecture_darkmode.svg',
]
LINE_ANCHOR = re.compile(r'^L\d+(-L\d+)?$')
BANNER = re.compile(r'^\s*#{5,}\s*(.*?)\s*$')
COMMENT = re.compile(r'^\s*#\s?(.*?)\s*$')


class LinkError(Exception):
    pass


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


@lru_cache(maxsize=None)
def source(path: str, ref: str | None) -> str:
    if ref is None:
        return (ROOT / path).read_text()
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


def statement_after(tree: ast.AST, line: int) -> int:
    """Last line of the first statement that starts below ``line``."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and node.lineno > line:
            if best is None or node.lineno < best.lineno:
                best = node
    if best is None:
        raise LinkError(f'no statement follows line {line}')
    return best.end_lineno


def resolve_target(path: str, target: str, ref: str | None) -> tuple[int, int]:
    text = source(path, ref)
    tree = ast.parse(text)
    defs = [(a, b) for name, a, b in definitions(tree) if name == target]
    if len(defs) == 1:
        return defs[0]
    if len(defs) > 1:
        raise LinkError(f'{path}#{target}: {len(defs)} definitions share this name')

    lines = text.splitlines()
    hits = []
    for i, ln in enumerate(lines, start=1):
        m = BANNER.match(ln)
        if m:
            if slug(m.group(1)) == target and not m.group(1).startswith('/'):
                hits.append(('banner', i))
            continue
        m = COMMENT.match(ln)
        if m:
            first_sentence = re.split(r'(?<=\.)\s', m.group(1))[0]
            if target in (slug(m.group(1)), slug(first_sentence)):
                hits.append(('comment', i))
    banners = [h for h in hits if h[0] == 'banner']
    hits = banners or hits  # a section banner wins over a passing mention
    if len(hits) != 1:
        what = 'no function or comment matches' if not hits else f'{len(hits)} comments match'
        raise LinkError(f'{path}#{target}: {what} this target')
    kind, start = hits[0]
    if kind == 'banner':
        for j in range(start + 1, len(lines) + 1):
            m = BANNER.match(lines[j - 1])
            if m and m.group(1).startswith('/'):
                return start, j
        raise LinkError(f'{path}#{target}: section banner is never closed')
    return start, statement_after(tree, start)


def is_symbolic(href: str) -> bool:
    if not href.startswith(REPO_URL) or '#' not in href:
        return False
    return not LINE_ANCHOR.match(href.split('#', 1)[1])


def resolve(href: str, ref: str | None = None) -> str:
    """Turn a name-based link into a line-range link; other links pass through."""
    if not is_symbolic(href):
        return href
    path, target = href[len(REPO_URL) :].split('#', 1)
    a, b = resolve_target(path, target, ref)
    return f'{REPO_URL}{path}#L{a}-L{b}'


def model_links() -> list[str]:
    items = json.loads((HERE / 'arch_final.json').read_text())['items']
    return sorted({it['href'] for it in items if it.get('href')})


SVG_LINK = re.compile(r'<a xlink:href="([^"]*)" data-target="([^"]*)"')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='fail if a link is stale')
    ap.add_argument('--ref', help='resolve against this git ref instead of the work tree')
    args = ap.parse_args()

    errors = []
    for href in model_links():
        if href.startswith(REPO_URL) and LINE_ANCHOR.match(href.partition('#')[2]):
            errors.append(f'hard-coded line anchor in arch_final.json: {href}')
        try:
            resolve(href, args.ref)
        except LinkError as e:
            errors.append(str(e))

    for svg in SVGS:
        text = svg.read_text()
        found = SVG_LINK.findall(text)
        if not found:
            errors.append(f'{svg.name}: no name-based links; rebuild the figure')

        def fix(m, svg=svg):
            target = unescape(m.group(2))
            try:
                new = resolve(target, args.ref)
            except LinkError:
                return m.group(0)  # already reported above
            if new != unescape(m.group(1)):
                if args.check:
                    errors.append(f'{svg.name}: stale link for {target}')
                return f'<a xlink:href="{escape(new, quote=True)}" data-target="{m.group(2)}"'
            return m.group(0)

        new_text = SVG_LINK.sub(fix, text)
        if not args.check and new_text != text:
            svg.write_text(new_text)
            print(f'updated links in {svg.relative_to(ROOT)}')

    for e in errors:
        print(f'error: {e}', file=sys.stderr)
    if errors and args.check:
        print('run: python3 tools/figures/code_links.py', file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
