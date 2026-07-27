#!/usr/bin/env python3
"""Record the reference trajectory the golden-run parity check compares against.

The check in ``tests/integration/test_integration_golden_run.py`` runs a fixed
all-dummy configuration and requires every helpfile column to reproduce a
recorded trajectory. A change to the physics, to the coupling loop or to the
configuration moves that trajectory, and the reference then has to be recorded
again in the same commit as the change that moved it.

Usage::

    # Show what a change did to the trajectory, without touching the file
    python tools/record_golden_run.py --dry-run

    # Record the trajectory this working tree produces
    python tools/record_golden_run.py

The same comparison serves a refactor that is meant to change nothing: record
the reference before the work, and run the check as the work proceeds.

Exit codes:
    0 -> reference written, or (with --dry-run) the run reproduces it
    1 -> the run differs from the stored reference and --dry-run was given
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from proteus.utils.trajectory import (
    compare_trajectories,
    config_digest,
    read_reference,
    run_trajectory,
    write_reference,
)

PROTEUS_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROTEUS_ROOT / 'tests' / 'integration' / 'golden_run.toml'
REFERENCE_PATH = PROTEUS_ROOT / 'tests' / 'integration' / 'golden_run.tsv'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='report how the run differs from the stored reference, and leave it alone',
    )
    parser.add_argument(
        '--reference',
        type=Path,
        default=REFERENCE_PATH,
        help=f'reference file to write or compare against (default: {REFERENCE_PATH})',
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix='proteus_golden_') as scratch:
        frame = run_trajectory(CONFIG_PATH, Path(scratch) / 'run')

    if args.dry_run:
        if not args.reference.is_file():
            print(f'no reference at {args.reference}; run without --dry-run to write one')
            return 1
        stored = read_reference(args.reference)
        # Reported before the comparison: a reference taken from a different
        # configuration explains any number of differing columns, and a reader
        # who meets the column list first will look for the cause in the code.
        if stored.config_digest != config_digest(CONFIG_PATH):
            print(
                f'the reference was recorded from a different {CONFIG_PATH.name}; '
                'record it again to bring the two back together'
            )
            return 1
        comparison = compare_trajectories(stored.frame, frame)
        print(comparison.report())
        return 0 if comparison.agrees else 1

    previous = read_reference(args.reference) if args.reference.is_file() else None
    write_reference(frame, args.reference, config_path=CONFIG_PATH)
    print(f'wrote {args.reference} ({len(frame)} rows)')
    if previous is not None:
        comparison = compare_trajectories(previous.frame, frame)
        print('against the reference it replaces:')
        print(comparison.report())
    return 0


if __name__ == '__main__':
    sys.exit(main())
