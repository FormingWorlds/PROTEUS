"""Reuse of prepared spectral files across runs that share a stellar spectrum.

A run's `runtime.sf` is the base spectral file from FWL_DATA with that run's
stellar spectrum inserted. This module keeps one copy per distinct input set so the
second and later runs copy it instead.

The cache is seeded into the run's output folder as `runtime.sf`, which is
where the atmosphere wrapper already expects to manage it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger('fwl.' + __name__)

# Prepared spectral files come in a pair, and the existence check in the AGNI
# module only tests the first. Seeding one without the other would leave the
# module using a file whose companion is absent.
SPECTRAL_SUFFIXES = ('', '_k')

# Bytes read per chunk when fingerprinting a file.
_CHUNK = 1 << 20


def _file_digest(path: Path) -> str:
    """Hash a file's contents."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(base_sf: Path | str, star_spectrum: Path | str, group: str, bands: str) -> str:
    """Name the cache entry for a prepared spectral file. The stellar spectrum is hashed by content.

    Parameters
    ----------
    - base_sf (Path | str): Base spectral file, from FWL_DATA.
    - star_spectrum (Path | str): Stellar spectrum (`.sflux`) to be inserted.
    - group (str): Spectral file group.
    - bands (str): Number of wavenumber bands.

    Returns
    ----------
    - str: Hex digest naming this combination.
    """
    base = Path(base_sf)
    stat = base.stat()
    parts = (
        group,
        bands,
        base.name,
        str(stat.st_size),
        str(int(stat.st_mtime)),
        _file_digest(Path(star_spectrum)),
    )
    return hashlib.sha256('\0'.join(parts).encode()).hexdigest()[:32]


def _entry_paths(cache_dir: Path | str, key: str) -> list[Path]:
    """Paths of the cached pair for one key."""
    return [Path(cache_dir) / f'{key}.sf{suffix}' for suffix in SPECTRAL_SUFFIXES]


def _output_paths(out_dir: Path | str) -> list[Path]:
    """Paths of the prepared pair inside a run's output folder."""
    return [Path(out_dir) / f'runtime.sf{suffix}' for suffix in SPECTRAL_SUFFIXES]


def seed_from_cache(cache_dir: Path | str, key: str, out_dir: Path | str) -> bool:
    """Copy a cached spectral file pair into a run's output folder.

    Parameters
    ----------
    - cache_dir (Path | str): Folder holding cached entries.
    - key (str): Key from `cache_key`.
    - out_dir (Path | str): The run's output folder.

    Returns
    ----------
    - bool: True if the run can now use `runtime.sf` as-is. False on a miss, or
      on any error: a cache that cannot be read is a slower run, never a failed
      one, so the caller falls back to building the file.
    """
    entries = _entry_paths(cache_dir, key)
    if not all(entry.is_file() for entry in entries):
        return False

    targets = _output_paths(out_dir)
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for entry, target in zip(entries, targets):
            shutil.copyfile(entry, target)
    except OSError as err:
        log.warning(f'Could not seed spectral file from cache: {err}')
        # A half-copied pair would be read as a prepared file. Clear it so the
        # run rebuilds from scratch instead of starting from a truncated file.
        for target in targets:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        return False

    log.debug(f'Seeded spectral file from cache entry {key}')
    return True


def store_in_cache(cache_dir: Path | str, key: str, out_dir: Path | str) -> bool:
    """Copy a run's prepared spectral file pair into the cache.

    Written to a temporary name and renamed into place, so concurrent runs
    racing to populate the same key never expose a half-written entry.

    Parameters
    ----------
    - cache_dir (Path | str): Folder holding cached entries.
    - key (str): Key from `cache_key`.
    - out_dir (Path | str): The run's output folder, holding the prepared pair.

    Returns
    ----------
    - bool: True if the entry was stored. False if the run produced no prepared
      file, or the cache could not be written; neither is a fault of the run.
    """
    sources = _output_paths(out_dir)
    if not all(source.is_file() for source in sources):
        return False

    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for source, entry in zip(sources, _entry_paths(cache_dir, key)):
            tmp = entry.with_name(f'{entry.name}.{os.getpid()}.tmp')
            try:
                shutil.copyfile(source, tmp)
                os.replace(tmp, entry)
            except BaseException:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
    except OSError as err:
        log.warning(f'Could not store spectral file in cache: {err}')
        return False

    log.debug(f'Stored spectral file in cache as entry {key}')
    return True
