"""Migrate TOML configs to set planet.elements.O_mode and O_budget.

Issue #677 fix (whole-planet oxygen accounting): every config that uses
`[planet.elements]` mode (i.e. not `volatile_mode = "gas_prs"`) must now
explicitly declare an O_mode. The hard-cutover policy (Tim's choice 2026-05-14)
makes a missing O_mode a config-load error.

This script does a one-shot rewrite: every `[planet.elements]` block that
lacks `O_mode` gets `O_mode = "ic_chemistry"` and `O_budget = 0.0` inserted.
The "ic_chemistry" choice preserves legacy behaviour: PROTEUS defers the
O_kg_total IC value to CALLIOPE (or atmodeller), which is exactly what
the pre-fix code did implicitly.

Usage:
    python tools/migrate_oxygen_mode.py [--dry-run] [paths...]

Without paths the script walks `input/` and `tests/` and rewrites every TOML
that contains a `[planet.elements]` block. With `--dry-run`, prints planned
edits without writing.

This script is a one-shot helper, not part of the production code path.
After the migration lands and is committed, it can be deleted.
"""

from __future__ import annotations

import pathlib
import sys


def migrate_file(path: pathlib.Path, dry_run: bool = False) -> bool:
    """Add O_mode/O_budget to the planet.elements block in `path` if missing.

    Returns True iff the file was modified (or would be modified in dry-run).
    """
    text = path.read_text()

    if '[planet.elements]' not in text:
        return False
    if 'O_mode' in text:
        return False

    # Find the [planet.elements] line and insert after its block header.
    lines = text.splitlines()
    out_lines = []
    inserted = False
    in_block = False
    for line in lines:
        out_lines.append(line)
        stripped = line.strip()
        if stripped == '[planet.elements]':
            in_block = True
            continue
        if in_block and not inserted:
            # Insert at the first non-comment, non-blank line within the block
            # (or before any next section). We choose to insert immediately
            # after the [planet.elements] header for visibility.
            out_lines.insert(
                len(out_lines) - 1,
                '        O_mode   = "ic_chemistry"  # Issue #677: '
                'use CALLIOPE IC equilibrium to derive O budget',
            )
            out_lines.insert(
                len(out_lines) - 1,
                '        O_budget = 0.0           # ignored for ic_chemistry mode',
            )
            inserted = True

    if not inserted:
        return False

    new_text = '\n'.join(out_lines)
    if not new_text.endswith('\n'):
        new_text += '\n'

    if dry_run:
        print(f'[DRY] would modify {path}')
        return True

    path.write_text(new_text)
    print(f'migrated {path}')
    return True


def main() -> int:
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    args = [a for a in args if not a.startswith('--')]

    root = pathlib.Path(__file__).resolve().parent.parent
    if args:
        paths = [pathlib.Path(a) for a in args]
    else:
        paths = list((root / 'input').rglob('*.toml')) + list((root / 'tests').rglob('*.toml'))

    n_modified = 0
    for p in sorted(paths):
        if migrate_file(p, dry_run=dry_run):
            n_modified += 1

    print(f'\n{"would modify" if dry_run else "modified"} {n_modified} file(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
