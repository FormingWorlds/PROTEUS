"""One-shot migration: rename aragog backend selector in TOML configs.

Schema change:
    [interior_energetics.aragog] jax = false        ->  removed (research-only via code flag)
    [interior_energetics.aragog] use_jax_jacobian = true/false  ->  backend = "jax"/"numpy"

This script rewrites:
  1. input/chili/*.toml             (committed configs)
  2. output/*/init_coupler.toml     (per-run snapshots, used by plot scripts)

It is idempotent: running twice is a no-op.
Run from the PROTEUS repo root with the proteus conda env active.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Keep the [interior_energetics.aragog] block intact line-by-line; only
# rewrite the two fields. Use line-anchored regexes so we don't touch any
# `jax = ` or `use_jax_jacobian` strings outside that block.

RE_JAX_LINE   = re.compile(r'^jax = (true|false)\s*$', re.M)
RE_UJJ_LINE   = re.compile(r'^use_jax_jacobian = (true|false)\s*$', re.M)
RE_BACKEND    = re.compile(r'^backend = "(numpy|jax)"\s*$', re.M)


def migrate(text: str) -> tuple[str, dict]:
    """Apply the schema migration to one TOML's text. Returns (new_text, stats)."""
    stats = {'jax_removed': 0, 'ujj_renamed': 0, 'already_migrated': 0}

    if RE_BACKEND.search(text) and not RE_UJJ_LINE.search(text):
        stats['already_migrated'] = 1
        return text, stats

    # Remove the `jax = ...` line entirely (and its trailing newline).
    new_text, n = RE_JAX_LINE.subn('', text)
    stats['jax_removed'] = n
    # Drop any double-blank-line that the removal may have left behind.
    new_text = re.sub(r'\n\n\n+', '\n\n', new_text)

    # Rename use_jax_jacobian -> backend with mapped value.
    def _ujj_to_backend(m: re.Match) -> str:
        val = m.group(1)
        return f'backend = "{"jax" if val == "true" else "numpy"}"'

    new_text, n = RE_UJJ_LINE.subn(_ujj_to_backend, new_text)
    stats['ujj_renamed'] = n
    return new_text, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true',
                    help='Write changes. Without this, dry-run only.')
    ap.add_argument('--include-snapshots', action='store_true',
                    help='Also rewrite output/*/init_coupler.toml.')
    args = ap.parse_args()

    targets: list[Path] = list((REPO / 'input/chili').glob('*.toml'))
    if args.include_snapshots:
        targets.extend((REPO / 'output').glob('*/init_coupler.toml'))

    total = {'files': 0, 'changed': 0, 'jax_removed': 0,
             'ujj_renamed': 0, 'already_migrated': 0}
    for p in sorted(targets):
        total['files'] += 1
        old = p.read_text()
        new, stats = migrate(old)
        for k in ('jax_removed', 'ujj_renamed', 'already_migrated'):
            total[k] += stats[k]
        if new != old:
            total['changed'] += 1
            rel = p.relative_to(REPO)
            print(f'  {"[apply]" if args.apply else "[dry  ]"} {rel}  '
                  f'-jax={stats["jax_removed"]}  ujj->backend={stats["ujj_renamed"]}')
            if args.apply:
                p.write_text(new)

    print()
    print(f'files scanned        : {total["files"]}')
    print(f'files changed        : {total["changed"]}')
    print(f'jax lines removed    : {total["jax_removed"]}')
    print(f'use_jax_jacobian->backend: {total["ujj_renamed"]}')
    print(f'already migrated     : {total["already_migrated"]}')
    if not args.apply:
        print('\n(dry-run; pass --apply to write)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
