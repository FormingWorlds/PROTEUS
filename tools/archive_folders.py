#!/usr/bin/env python3
"""Create an individual .tar.xz archive of each sub-folder in a root directory.

Given a root directory, every immediate sub-folder is archived to
``<name>.tar.xz`` next to it (or into a chosen output directory). Files
directly in the root directory are ignored; only sub-folders are archived.

Each sub-folder is treated as a branch of the tree containing one or more
PROTEUS simulations, identified by an ``init_coupler.toml`` config file. Before
the branch is packed, ``proteus create-archives`` is run on each such
simulation, which packs its many per-iteration output files into a handful of
tar archives. This drastically reduces the file count the final .tar.xz has to
compress, which is otherwise the dominant cost for simulation output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

# Config file that marks a directory as a PROTEUS simulation output folder.
SIM_CONFIG_NAME = 'init_coupler.toml'

# Sub-folders whose name starts with this prefix are left untouched.
NOARCHIVE_PREFIX = 'noarchive'


def archive_folder(folder: Path, out_dir: Path, overwrite: bool = False) -> Path:
    """Archive a single folder to ``<out_dir>/<folder.name>.tar.xz``.

    The folder itself becomes the top-level entry inside the archive so it
    extracts back to a directory of the same name.
    """
    archive_path = out_dir / f'{folder.name}.tar.xz'
    if archive_path.exists() and not overwrite:
        raise FileExistsError(archive_path)

    # Write to a temporary name and rename on success, so an interrupted run
    # never leaves a truncated archive that looks complete.
    tmp_path = archive_path.with_suffix(archive_path.suffix + '.partial')
    with tarfile.open(tmp_path, mode='w:xz') as tar:
        tar.add(folder, arcname=folder.name)
    tmp_path.replace(archive_path)
    return archive_path


def iter_subfolders(root: Path):
    """Yield immediate sub-directories of ``root`` in sorted order."""
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def find_simulations(branch: Path) -> list[Path]:
    """Return every ``init_coupler.toml`` found anywhere under ``branch``.

    Each match identifies the config of one PROTEUS simulation. The list is
    sorted for deterministic ordering and reproducible logs.
    """
    return sorted(branch.rglob(SIM_CONFIG_NAME))


def create_simulation_archives(configs: list[Path]) -> int:
    """Run ``proteus create-archives -c <config>`` on each simulation config.

    Returns the number of simulations for which the command failed. Failures
    are reported but do not abort the sweep, so a single broken simulation
    does not prevent the rest of the branch from being packed.
    """
    failures = 0
    for config in configs:
        print(f'        create-archives {config.parent.name} ...', end=' ', flush=True)
        result = subprocess.run(
            ['proteus', 'create-archives', '-c', str(config)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print('ok')
        else:
            failures += 1
            print('FAIL')
            # Surface the tail of stderr so the cause is visible without
            # re-running the command by hand.
            tail = (result.stderr or result.stdout).strip().splitlines()[-5:]
            for line in tail:
                print(f'            {line}', file=sys.stderr)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root',
        type=Path,
        help='Root directory whose sub-folders will each be archived.',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=None,
        help='Where to write the archives (default: alongside each folder, i.e. the root).',
    )
    parser.add_argument(
        '-f',
        '--overwrite',
        action='store_true',
        help='Overwrite existing archives instead of skipping them.',
    )
    parser.add_argument(
        '--no-create-archives',
        action='store_true',
        help=(
            'Skip running "proteus create-archives" on the simulations before '
            'packing each branch (pack the raw output files as-is).'
        ),
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        parser.error(f'root is not a directory: {root}')

    out_dir: Path = args.output_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = list(iter_subfolders(root))
    if not folders:
        print(f'No sub-folders found in {root}', file=sys.stderr)
        return 0
    print(f'Found {len(folders)} sub-folders in {root} to archive into {out_dir}')

    failures = 0
    for folder in folders:
        print(f'  branch {folder.name}')

        # Honour the opt-out prefix so hand-curated folders are never packed.
        if folder.name.startswith(NOARCHIVE_PREFIX):
            print(f'    skip   (name starts with "{NOARCHIVE_PREFIX}")')
            continue

        # Skip early if the archive already exists and we are not overwriting,
        # so we do not spend time on create-archives for nothing.
        archive_path = out_dir / f'{folder.name}.tar.xz'
        if archive_path.exists() and not args.overwrite:
            print(f'    skip   (exists: {archive_path}; use --overwrite)')
            continue

        # Pack each simulation's output files first: this is what makes the
        # subsequent whole-branch .tar.xz fast on PROTEUS output.
        if not args.no_create_archives:
            configs = find_simulations(folder)
            print(f'    found {len(configs)} simulation(s) to create-archives')
            failures += create_simulation_archives(configs)

        print(f'    archiving {folder.name} ...', end=' ', flush=True)
        try:
            archive_path = archive_folder(folder, out_dir, overwrite=args.overwrite)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures += 1
            print('FAIL')
            print(f'    FAIL   {folder.name}  ({exc})', file=sys.stderr)
            continue
        size_mb = archive_path.stat().st_size / 1e6
        print(f'ok  -> {archive_path.name}  ({size_mb:.1f} MB)')

    print('Finished archiving sub-folders')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
