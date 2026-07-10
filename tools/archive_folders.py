#!/usr/bin/env python3
"""Create an individual .tar.xz archive of each sub-folder in a root directory.

Given a root directory, every immediate sub-folder is archived to
``<name>.tar.xz`` next to it (or into a chosen output directory). Files sitting
directly in the root are ignored; only directories are archived.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path


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

    failures = 0
    for folder in folders:
        try:
            archive_path = archive_folder(folder, out_dir, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f'skip   {folder.name}  (exists: {exc}; use --overwrite)')
            continue
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures += 1
            print(f'FAIL   {folder.name}  ({exc})', file=sys.stderr)
            continue
        size_mb = archive_path.stat().st_size / 1e6
        print(f'ok     {folder.name} -> {archive_path.name}  ({size_mb:.1f} MB)')

    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
