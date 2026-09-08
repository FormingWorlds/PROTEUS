#!/usr/bin/env python3
"""Shared helpers for the reference-documentation generators.

The generator scripts (generate_config_reference.py, generate_module_map.py,
generate_output_reference.py, generate_version_badges.py) render committed
reference pages and JSON files from the source tree. This module carries the
pieces they share: marker-delimited replacement that fails loudly when a
marker is missing, a check/write driver with a common exit-code contract,
deterministic JSON serialisation, and an import helper that loads
``proteus.config`` from the checkout without executing the heavyweight
package ``__init__``.

Usage:
    python tools/_docgen.py --sanity

The ``--sanity`` mode verifies documentation structure without building the
site: every page referenced in the mkdocs.yml nav exists under ``docs/``, and
every BEGIN/END marker pair in the docs tree is balanced.

Exit codes:
    0  clean
    1  a sanity problem was found
    2  structural error (unreadable file, malformed nav)
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / 'docs'
MKDOCS_YML = REPO_ROOT / 'mkdocs.yml'

# Matches both the generic generated-region markers used by the reference
# generators and the older PYPI_TABLE-style markers in module_versions.md.
_MARKER_RE = re.compile(r'<!--\s*(BEGIN|END)\s+(.+?)\s*-->')


class DocgenError(Exception):
    """Structural failure while generating documentation (exit code 2)."""


def replace_between_markers(content: str, marker: str, replacement: str) -> str:
    """Replace the region between ``BEGIN <marker>`` and ``END <marker>``.

    Raises ``DocgenError`` when the marker pair is absent or appears more
    than once, so a typo in a page never silently produces a no-op.
    """
    pattern = (
        rf'(<!--\s*BEGIN {re.escape(marker)}\s*-->)(.*?)(<!--\s*END {re.escape(marker)}\s*-->)'
    )
    new, count = re.subn(pattern, rf'\1\n{replacement}\n\3', content, flags=re.DOTALL)
    if count == 0:
        raise DocgenError(f'marker pair "{marker}" not found')
    if count > 1:
        raise DocgenError(f'marker pair "{marker}" appears {count} times; expected once')
    return new


def normalize(text: str) -> str:
    """Strip trailing whitespace per line and end with exactly one newline.

    docs/ is excluded from the pre-commit whitespace hooks, so generated
    content must arrive clean rather than relying on a fixer.
    """
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).rstrip('\n') + '\n'


# A code span first, so a bracket inside one is never read as a link, then
# the two real link forms. The URL part tolerates one level of nested
# parentheses. Anything outside these is free text.
_PROTECTED = re.compile(
    r'(?P<ticks>`+).+?(?P=ticks)'
    r'|\[[^\[\]]*\]\([^()]*(?:\([^()]*\)[^()]*)*\)'
    r'|\[[^\[\]]*\]\[[^\[\]]*\]',
    re.DOTALL,
)
_BARE_BRACKET = re.compile(r'(?<!\\)([\[\]])')


_BARE_PIPE_BRACKET = re.compile(r'(?<!\\)([\[\]|])')


def escape_link_brackets(text: str, *, escape_pipes: bool = False) -> str:
    """Escape square brackets that markdown would read as a link reference.

    Descriptions carry units in brackets ("[bar]", "[W m-1 K-1]"), which is
    correct scientific style and also markdown's shortcut reference syntax.
    Real links keep working, and code spans are left alone because a
    backslash renders literally inside one. With ``escape_pipes`` the same
    protected regions also keep their ``|`` while free-text pipes are escaped
    for table cells. Call this only in the markdown render path; the JSON
    sidecars carry the raw text.
    """
    bare = _BARE_PIPE_BRACKET if escape_pipes else _BARE_BRACKET
    out: list[str] = []
    pos = 0
    for match in _PROTECTED.finditer(text):
        out.append(bare.sub(r'\\\1', text[pos : match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(bare.sub(r'\\\1', text[pos:]))
    return ''.join(out)


def dump_json(obj) -> str:
    """Serialise deterministically: stable key order as inserted, no ASCII escaping."""
    return json.dumps(obj, indent=2, ensure_ascii=False) + '\n'


def import_proteus_config():
    """Import ``proteus.config`` from the checkout's ``src/`` tree.

    Registers a stub parent package so ``src/proteus/__init__.py`` (which
    imports the full runtime stack, including juliacall) never executes.
    Guarantees the generators read the checkout, not an installed copy.
    """
    if 'proteus' not in sys.modules:
        stub = types.ModuleType('proteus')
        stub.__path__ = [str(REPO_ROOT / 'src' / 'proteus')]
        sys.modules['proteus'] = stub
    return importlib.import_module('proteus.config')


def add_check_write_cli(parser: argparse.ArgumentParser) -> None:
    """Attach the mutually exclusive ``--check`` / ``--write`` options."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--check',
        action='store_true',
        help='verify the committed output is current; exit 1 on drift, write nothing',
    )
    group.add_argument(
        '--write',
        action='store_true',
        help='regenerate the committed output in place (default)',
    )


def run_generator(targets: list[tuple[Path, str]], *, check: bool, regen_cmd: str) -> int:
    """Write or verify each (path, content) target.

    In check mode, compares the rendered content against the file on disk and
    reports drift without writing; the failure message names ``regen_cmd`` so
    the remediation is one copy-paste away. In write mode, rewrites only the
    files whose content changed.
    """
    stale = 0
    for path, content in targets:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        on_disk = path.read_text() if path.exists() else None
        if content == on_disk:
            print(f'[=] {rel} is up to date')
            continue
        if check:
            state = 'is stale' if on_disk is not None else 'is missing'
            print(f'[x] {rel} {state}; regenerate with: {regen_cmd}')
            stale += 1
        else:
            path.write_text(content)
            print(f'[+] wrote {rel}')
    return 1 if (check and stale) else 0


def check_markers_balanced(text: str) -> list[str]:
    """Return one problem string per marker-pairing violation in ``text``.

    Markers must alternate strictly: each BEGIN is closed by an END carrying
    the same name before the next BEGIN opens. Nesting is not supported.
    """
    problems: list[str] = []
    open_name: str | None = None
    for kind, name in _MARKER_RE.findall(text):
        if kind == 'BEGIN':
            if open_name is not None:
                problems.append(f'BEGIN "{name}" opened before END "{open_name}"')
            open_name = name
        elif open_name is None:
            problems.append(f'END "{name}" without a matching BEGIN')
        elif name != open_name:
            problems.append(f'END "{name}" does not match BEGIN "{open_name}"')
            open_name = None
        else:
            open_name = None
    if open_name is not None:
        problems.append(f'BEGIN "{open_name}" is never closed')
    return problems


def parse_nav_paths(mkdocs_text: str) -> list[str]:
    """Extract the .md paths listed in the ``nav:`` block of mkdocs.yml.

    A plain regex walk over the indented block; mkdocs.yml carries custom
    YAML tags that a safe YAML load rejects, so it is never yaml-parsed here.
    """
    paths: list[str] = []
    in_nav = False
    for line in mkdocs_text.split('\n'):
        if line.startswith('nav:'):
            in_nav = True
            continue
        if in_nav:
            if line.strip() and not line.startswith((' ', '\t')):
                break
            match = re.search(r':\s*([A-Za-z0-9_][A-Za-z0-9_/.\-]*\.md)\s*$', line)
            if match:
                paths.append(match.group(1))
    if not paths:
        raise DocgenError('no nav entries found in mkdocs.yml')
    return paths


def run_sanity() -> int:
    """Check nav targets exist and all docs marker pairs are balanced."""
    failures = 0
    for nav_path in parse_nav_paths(MKDOCS_YML.read_text()):
        if not (DOCS_DIR / nav_path).is_file():
            print(f'[x] mkdocs.yml nav references missing page docs/{nav_path}')
            failures += 1
    for page in sorted(DOCS_DIR.rglob('*.md')):
        for problem in check_markers_balanced(page.read_text()):
            print(f'[x] {page.relative_to(REPO_ROOT)}: {problem}')
            failures += 1
    if failures == 0:
        print('[+] docs sanity checks passed')
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sanity',
        action='store_true',
        help='verify nav targets exist and docs marker pairs are balanced',
    )
    args = parser.parse_args()
    if not args.sanity:
        parser.error('nothing to do; pass --sanity')
    try:
        return run_sanity()
    except DocgenError as exc:
        print(f'[x] {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
