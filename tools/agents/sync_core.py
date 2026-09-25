#!/usr/bin/env python3
"""Write the canonical shared blocks into the AGENTS.md files of ecosystem repos.

The canonical text lives next to this script: ``core.md`` (block ``fwl-core``),
``tests-core.md`` (``fwl-tests-core``), and ``voice-neutral.md`` or
``voice-list.md`` (``fwl-voice``). The script replaces the body of every
shared block it finds in a repository's ``AGENTS.md`` files and writes the new
hash into the begin marker; it never adds or removes a block.

Usage::

    python tools/agents/sync_core.py <repo> [<repo> ...] [--voice neutral|list]
    python tools/agents/sync_core.py --check <repo> [<repo> ...]

``--check`` writes nothing and exits 1 when any block differs from the
canonical text, so a lagging repository is visible from PROTEUS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from check_agents_md import agents_files, block_hash, find_blocks

HERE = Path(__file__).resolve().parent


def canonical(name: str, voice: str) -> str:
    """Return the canonical body of block ``fwl-<name>``."""
    stem = f'voice-{voice}' if name == 'voice' else name
    return (HERE / f'{stem}.md').read_text().strip('\n')


def sync_file(path: Path, voice: str, write: bool) -> list[str]:
    """Bring the shared blocks of one file up to date.

    Parameters
    ----------
    path : Path
        An ``AGENTS.md`` file.
    voice : str
        ``'neutral'`` or ``'list'``, the variant of the ``fwl-voice`` block.
    write : bool
        Rewrite the file when True; only report when False.

    Returns
    -------
    list of str
        Names of the blocks that differed from the canonical text.
    """
    text = path.read_text()
    stale, parts, pos = [], [], 0
    for m in find_blocks(text):
        body = canonical(m['name'], voice)
        if m['hash'] != block_hash(body) or block_hash(m['body']) != m['hash']:
            stale.append(m['name'])
        parts += [
            text[pos : m.start()],
            f'<!-- fwl-{m["name"]}:begin sha256={block_hash(body)} -->\n{body}\n',
            f'<!-- fwl-{m["name"]}:end -->',
        ]
        pos = m.end()
    if write and stale:
        path.write_text(''.join(parts) + text[pos:])
    return stale


def main(argv: list[str] | None = None) -> int:
    """Sync or check the given repositories; return the exit status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('repos', nargs='+', type=Path)
    ap.add_argument('--voice', choices=('neutral', 'list'), default='neutral')
    ap.add_argument('--check', action='store_true', help='report only, exit 1 on lag')
    args = ap.parse_args(argv)

    lagging = False
    for repo in args.repos:
        root = repo.resolve()
        for path in agents_files(root):
            stale = sync_file(path, args.voice, write=not args.check)
            if stale:
                lagging = True
                verb = 'differs' if args.check else 'updated'
                print(f'{path.relative_to(root.parent)}: {", ".join(stale)} {verb}')
    return 1 if (args.check and lagging) else 0


if __name__ == '__main__':
    sys.exit(main())
