#!/usr/bin/env python3
"""Write the canonical shared blocks into the AGENTS.md files of ecosystem repos.

The canonical text lives next to this script: ``core.md`` (block ``fwl-core``),
``tests-core.md`` (``fwl-tests-core``) and ``voice-neutral.md`` (``fwl-voice``).
The script replaces the body of every shared block it finds in a repository's
``AGENTS.md`` files and writes the new hash into the begin marker; it never adds
or removes a block. A file with a marker it cannot parse, or with a block that has
no canonical file here, is reported and left unchanged.

Usage::

    python tools/agents/sync_core.py <repo> [<repo> ...]
    python tools/agents/sync_core.py --check <repo> [<repo> ...]

``--check`` writes nothing and exits 1 when any block differs from the
canonical text, so a lagging repository is visible from PROTEUS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from check_agents_md import BLOCK_RE, MARKER_RE, agents_files, block_hash

HERE = Path(__file__).resolve().parent


def canonical(name: str) -> str:
    """Return the canonical body of block ``fwl-<name>``."""
    stem = 'voice-neutral' if name == 'voice' else name
    return (HERE / f'{stem}.md').read_text().strip('\n')


def sync_file(path: Path, write: bool) -> list[str] | None:
    """Bring the shared blocks of one file up to date.

    Parameters
    ----------
    path : Path
        An ``AGENTS.md`` file.
    write : bool
        Rewrite the file when True; only report when False.

    Returns
    -------
    list of str or None
        Names of the blocks that differed from the canonical text, or None when
        a marker does not parse and the file was left unchanged.
    """
    text = path.read_bytes().decode()
    blocks = list(BLOCK_RE.finditer(text))
    if len(MARKER_RE.findall(text)) != 2 * len(blocks):
        return None
    stale, parts, pos = [], [], 0
    for m in blocks:
        body = canonical(m['name'])
        if m['hash'] != block_hash(body) or block_hash(m['body']) != m['hash']:
            stale.append(m['name'])
        parts += [
            text[pos : m.start()],
            f'<!-- fwl-{m["name"]}:begin sha256={block_hash(body)} -->\n{body}\n',
            f'<!-- fwl-{m["name"]}:end -->',
        ]
        pos = m.end()
    if write and stale:
        path.write_bytes((''.join(parts) + text[pos:]).encode())
    return stale


def main(argv: list[str] | None = None) -> int:
    """Sync or check the given repositories; return the exit status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('repos', nargs='+', type=Path)
    ap.add_argument('--check', action='store_true', help='report only, exit 1 on lag')
    args = ap.parse_args(argv)

    lagging = failed = False
    for repo in args.repos:
        root = repo.resolve()
        if not root.is_dir():
            print(f'{repo}: not a directory')
            failed = True
            continue
        for path in agents_files(root):
            try:
                stale = sync_file(path, write=not args.check)
            except FileNotFoundError as e:
                failed = True
                print(
                    f'{path.relative_to(root.parent)}: no canonical file {Path(e.filename).name}'
                )
                continue
            if stale is None:
                failed = True
                print(
                    f'{path.relative_to(root.parent)}: unmatched or malformed fwl- block marker'
                )
            elif stale:
                lagging = True
                verb = 'differs' if args.check else 'updated'
                print(f'{path.relative_to(root.parent)}: {", ".join(stale)} {verb}')
    return 1 if failed or (args.check and lagging) else 0


if __name__ == '__main__':
    sys.exit(main())
