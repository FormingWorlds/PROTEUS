#!/usr/bin/env python3
"""Check the agent instruction files of a PROTEUS-ecosystem repository.

The same file is used in every ecosystem repository; the canonical copy is
``tools/agents/check_agents_md.py`` in FormingWorlds/PROTEUS. It checks:

* every shared block (``<!-- fwl-<name>:begin sha256=<hash> -->`` to
  ``<!-- fwl-<name>:end -->``) still matches the hash written by
  ``sync_core.py``, which catches a local edit of the shared text (whether the
  text matches PROTEUS is checked there, by ``sync_core.py --check``);
* a root ``AGENTS.md`` exists, and the byte caps: 16,000 B for the root file,
  12,000 B for a nested one;
* next to every ``AGENTS.md`` a ``CLAUDE.md`` is a regular file that imports
  ``AGENTS.md`` and nothing else; with a root ``CLAUDE.md`` present, Claude Code
  reads a nested ``AGENTS.md`` only through its own ``CLAUDE.md``;
* ``.github/copilot-instructions.md`` has at most 60 lines.

Usage: ``python tools/agents/check_agents_md.py [repo_root]``. Exit status 1
lists every failure.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT_MAX_BYTES = 16_000
NESTED_MAX_BYTES = 12_000
COPILOT_MAX_LINES = 60
CLAUDE_MD_TEXT = '@AGENTS.md'

BLOCK_RE = re.compile(
    r'^<!-- fwl-(?P<name>[a-z-]+):begin sha256=(?P<hash>[0-9a-f]+) -->\n'
    r'(?P<body>.*?)'
    r'^<!-- fwl-(?P=name):end -->$',
    re.MULTILINE | re.DOTALL,
)
MARKER_RE = re.compile(r'^<!-- fwl-', re.MULTILINE)


def block_hash(body: str) -> str:
    """Return the 16-hex-digit SHA-256 prefix of a block body.

    Parameters
    ----------
    body : str
        Text between the begin and end markers; outer blank lines are ignored.

    Returns
    -------
    str
        The hash written into the begin marker.
    """
    return hashlib.sha256(body.strip('\n').encode()).hexdigest()[:16]


def find_blocks(text: str) -> list[re.Match]:
    """Return the shared blocks of a markdown file, in file order."""
    return list(BLOCK_RE.finditer(text))


def agents_files(root: Path) -> list[Path]:
    """Return the ``AGENTS.md`` files under ``root`` that git tracks or would add.

    Untracked files count unless git ignores them; files deleted from the
    working tree are skipped. Falls back to a directory walk outside a git
    checkout.
    """
    try:
        out = subprocess.run(
            [
                'git',
                'ls-files',
                '--cached',
                '--others',
                '--exclude-standard',
                '--',
                'AGENTS.md',
                '*/AGENTS.md',
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = sorted({root / p for p in out.splitlines()})
    except (OSError, subprocess.CalledProcessError):
        paths = sorted(p for p in root.rglob('AGENTS.md') if '.git' not in p.parts)
    return [p for p in paths if p.is_file()]


def check(root: Path) -> list[str]:
    """Return one message per failed check for the repository at ``root``."""
    if not root.is_dir():
        return [f'{root}: not a directory']
    errors = []
    if not (root / 'AGENTS.md').is_file():
        errors.append('AGENTS.md: missing at the repository root')
    for path in agents_files(root):
        rel = path.relative_to(root)
        data = path.read_bytes()
        cap = ROOT_MAX_BYTES if rel == Path('AGENTS.md') else NESTED_MAX_BYTES
        if len(data) > cap:
            errors.append(f'{rel}: {len(data)} B exceeds the {cap} B cap')
        text = data.decode()
        blocks = find_blocks(text)
        for m in blocks:
            if block_hash(m['body']) != m['hash']:
                errors.append(
                    f'{rel}: shared block fwl-{m["name"]} differs from its hash; '
                    'edit tools/agents/ in PROTEUS and run sync_core.py'
                )
        if len(MARKER_RE.findall(text)) != 2 * len(blocks):
            errors.append(f'{rel}: unmatched or malformed fwl- block marker')
        claude = path.with_name('CLAUDE.md')
        if (
            claude.is_symlink()
            or not claude.is_file()
            or claude.read_text().strip() != CLAUDE_MD_TEXT
        ):
            errors.append(
                f'{claude.relative_to(root)}: must be a regular file containing only {CLAUDE_MD_TEXT}'
            )

    copilot = root / '.github' / 'copilot-instructions.md'
    if copilot.is_file():
        n = len(copilot.read_text().splitlines())
        if n > COPILOT_MAX_LINES:
            errors.append(f'{copilot.relative_to(root)}: {n} lines exceeds {COPILOT_MAX_LINES}')
    return errors


def main(argv: list[str]) -> int:
    """Run the checks and print the failures; return the exit status."""
    root = Path(argv[1] if len(argv) > 1 else '.').resolve()
    errors = check(root)
    for e in errors:
        print(f'ERROR: {e}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
