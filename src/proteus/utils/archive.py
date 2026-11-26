# Archival functions and utilities

# Import utils-specific modules
from __future__ import annotations

import glob
import logging
import os
import tarfile

from proteus.utils.helper import parse_subyear_time, safe_rm

log = logging.getLogger('fwl.' + __name__)

# Snapshot times a resume unpacks: the resume row, the rows it may fall back to
# when the latest pair is incomplete, and the mesh copy of an earlier structure.
RESUME_SNAPSHOT_TIMES = 10


def _tarfile_from_dir(dir: str) -> str:
    name = os.path.split(dir)[-1]
    return os.path.join(dir, f'{name}.tar')


def _snapshot_time(name: str) -> float | None:
    """Parse the simulated time from a timestamped snapshot filename.

    A timestamped snapshot is a file ending in ``.nc``, ``.json`` or
    ``_zalmoxis.dat`` whose leading token is the simulated time in years, e.g.
    ``1000p000_int.nc``, ``1000p000_atm.nc``, ``0p200_int.nc``, or
    ``5000.json``. Whole-year names without a fractional part
    (``1000_int.nc``) parse to the same value.

    Arguments
    ---------
    name : str
        The basename to parse (not a full path).

    Returns
    -------
    float or None
        The simulated time [years] for a timestamped snapshot, or None
        for any other entry: the fixed-name runtime files that the
        interior modules re-read between structure re-solves
        (``zalmoxis_output.dat`` and its ``.prev`` backup,
        ``zalmoxis_output_temp.txt``, ``spider_mesh.dat``), the EOS and
        other runtime table directories, the timestamped ``.sflux``
        stellar-spectrum files (which stay loose during the run and are
        captured only by the final full archive), and the tar archive
        itself.
    """

    if not name.endswith(('.nc', '.json', '_zalmoxis.dat')):
        return None
    try:
        return parse_subyear_time(name.rsplit('.', 1)[0].split('_')[0])
    except ValueError:
        return None


def archive_exists(dir: str, ignore_warnings: bool = False) -> bool:
    """
    Check if the archive tar file exists inside a directory.

    Arguments
    ---------
    dir : str
        The directory to check.

    Returns
    -------
    bool
        Whether the tar file exists.
    """

    # Tar file path
    tar = _tarfile_from_dir(os.path.abspath(dir))

    # Exists?
    exists = os.path.exists(tar)

    if (not exists) and (not ignore_warnings):
        log.warning(f'The archive tar file does not exist: {tar}')

    return exists


def create(dir: str, remove_files: bool = True, snapshots_only: bool = False) -> str:
    """
    Create a new tar archive from a directory of files, placing the tar inside that directory.

    Optionally removes all files in that directory (other than the tar file).

    Arguments
    ---------
    dir : str
        The directory to archive.
    remove_files:bool
        Whether to remove the appended files from the directory.
    snapshots_only : bool
        When True, archive (and, if remove_files, remove) only timestamped
        snapshots as recognised by :func:`_snapshot_time`. The fixed-name
        runtime files and the runtime table directories are left in place.

    Returns
    -------
    str
        The path to the tar file created.
    """

    # Tar file path
    dir = os.path.abspath(dir)
    log.debug(f'Creating new archive of {dir}')

    # Check if the directory exists
    if not os.path.exists(dir):
        log.error(f'Directory {dir} does not exist. Cannot archive it.')
        return

    # Check if the tar file exists
    if archive_exists(dir, ignore_warnings=True):
        log.error(f'Archive tar file for {dir} already exists. Will not create a new one.')
        return

    # List files in directory
    files = glob.glob(os.path.join(dir, '*'))
    files = [os.path.abspath(f) for f in files]

    # Add files to new tar file
    tar = _tarfile_from_dir(dir)
    with tarfile.open(tar, 'w') as tar_file:
        for f in files:
            if snapshots_only and _snapshot_time(os.path.basename(f)) is None:
                continue
            tar_file.add(f, arcname=os.path.basename(f))

    # Remove the files that were archived (never the tar file itself)
    if remove_files:
        for f in files:
            if f == tar:
                continue
            if snapshots_only and _snapshot_time(os.path.basename(f)) is None:
                continue
            safe_rm(f)

    # Return path to the tar file
    return tar


def append(dir: str, remove_files: bool = True, snapshots_only: bool = False) -> str:
    """
    Add files within `dir` into `dir/dir.tar`.

    The tar file must already exist. A file whose last archived copy has the same
    size and modification time is not added again.

    Arguments
    ---------
    dir:str
        Path to the archived directory.
    remove_files:bool
        Whether to remove the appended files from the directory.
    snapshots_only : bool
        When True, append (and, if remove_files, remove) only timestamped
        snapshots as recognised by :func:`_snapshot_time`. The fixed-name
        runtime files and the runtime table directories are left in place,
        so repeated appends do not accumulate duplicate copies of them.

    Returns
    -------
    str
        The path to the tar file which was updated.
    """

    # Paths
    dir = os.path.abspath(dir)
    log.debug(f'Appending files to archive in {dir}')

    # Check if the tar file exists
    if not archive_exists(dir, ignore_warnings=False):
        log.error('Cannot append to archive.')
        return

    # List files in directory
    files = glob.glob(os.path.join(dir, '*'))
    files = [os.path.abspath(f) for f in files]

    # Append files to existing tar file
    tar = _tarfile_from_dir(dir)
    with tarfile.open(tar, 'a') as tar_file:
        archived = {m.name: m for m in tar_file.getmembers()}

        def changed(info):
            """Leave out a file whose last archived copy has its size and mtime."""
            last = archived.get(info.name)
            if info.isfile() and last and (last.size, last.mtime) == (info.size, info.mtime):
                return None
            return info

        for f in files:
            if snapshots_only and _snapshot_time(os.path.basename(f)) is None:
                continue
            tar_file.add(f, arcname=os.path.basename(f), filter=changed)

    # Remove the files that were appended (never the tar file itself)
    if remove_files:
        for f in files:
            if f == tar:
                continue
            if snapshots_only and _snapshot_time(os.path.basename(f)) is None:
                continue
            safe_rm(f)

    return tar


def _extract_missing(tar_file, dir: str, keep=None) -> None:
    """Extract the last archived copy of each name that `keep` accepts and that is
    not loose in `dir`; a loose file is newer than its archived copy."""
    members = {m.name: m for m in tar_file.getmembers()}
    wanted = [
        m
        for n, m in members.items()
        if (keep is None or keep(n)) and not os.path.lexists(os.path.join(dir, n))
    ]
    log.info(f'Extracting {len(wanted)} of {len(members)} archived files in {dir}')
    tar_file.extractall(dir, members=wanted, filter='data')


def extract(dir: str, remove_tar: bool = False, ignore_warnings: bool = False) -> str:
    """
    Extract the tar file contained within a directory, placing the content within that directory.

    A file that is already loose in the directory keeps its loose copy.
    Removes the tar file afterwards if `remove_tar` is set to True.
    Does not complain if the tar file does not exist, if `ignore_warnings` is set to True.

    Arguments
    ---------
    dir : str
        The directory to extract.
    remove_tar : bool
        Whether to remove the tar file after extraction.
    ignore_warnings : bool
        Whether to complain if the tar file does not exist.

    Returns
    -------
    str
        The path to the tar file which was extracted.
    """

    # Paths
    dir = os.path.abspath(dir)
    tar = _tarfile_from_dir(dir)
    log.debug(f'Extracting tar file inside {dir}')

    # Check if the directory exists
    if not os.path.exists(dir):
        log.error(f'Directory {dir} does not exist.')
        log.error('Cannot extract archive.')
        return

    # Check if the tar file exists
    if not archive_exists(dir, ignore_warnings=ignore_warnings):
        return

    # Extract tar file
    with tarfile.open(tar, 'r') as tar_file:
        _extract_missing(tar_file, dir)

    # Remove tar file
    if remove_tar:
        safe_rm(tar)


def extract_latest(dir: str, count: int = RESUME_SNAPSHOT_TIMES) -> None:
    """
    Extract the entries a resume reads from the tar file in `dir`, leaving the tar file whole.

    These are every entry that is not a snapshot (see :func:`_snapshot_time`) and
    the snapshots of the latest `count` snapshot times. A name that is already
    loose in `dir` keeps its loose copy. The tar file is only read, so a resume
    that is stopped part way loses no data, and :func:`append` skips the
    extracted files that are archived again unchanged.

    Arguments
    ---------
    dir : str
        The directory holding the tar file.
    count : int
        Number of latest snapshot times to extract.
    """

    dir = os.path.abspath(dir)
    if not archive_exists(dir):
        return

    with tarfile.open(_tarfile_from_dir(dir), 'r') as tar_file:
        times = {n: _snapshot_time(n) for n in tar_file.getnames()}
        latest = sorted({t for t in times.values() if t is not None})[-count:]
        _extract_missing(tar_file, dir, lambda n: times[n] is None or times[n] >= latest[0])


def update(dir: str, remove_files: bool = True, snapshots_only: bool = False) -> None:
    """
    Create and/or update a data file archive.

    This function archives the folder located at `dir/`. The files inside `dir/`
    are stored inside a tar file called `dir.tar` located in `dir/`.
    The tar file is created if it does not already exist, otherwise it is
    updated with the contents of `dir/`.

    All paths should be absolute.

    Arguments
    ---------
    dir : str
        The directory to archive.
    remove_files : bool
        Whether to remove the archived files from the directory.
    snapshots_only : bool
        When True, archive only timestamped snapshots and leave the
        fixed-name runtime files and runtime table directories in place.
        The rolling in-loop archive sets this so that the fixed-name
        files, which must stay on disk for the interior modules and for
        resume, are not re-appended on every archive cycle.
    """

    # Paths
    dir = os.path.abspath(dir)

    # Update archive
    if archive_exists(dir, ignore_warnings=True):
        append(dir, remove_files=remove_files, snapshots_only=snapshots_only)

    # Create new archive
    else:
        create(dir, remove_files=remove_files, snapshots_only=snapshots_only)


def remove_old(dir: str, before: float) -> None:
    """
    Prune archived snapshot files older than a cutoff time.

    Only timestamped snapshot files are removed: names ending in ``.nc``,
    ``.json`` or ``_zalmoxis.dat`` whose leading token parses as a simulated time (e.g.
    ``1000p000_int.nc``), and only when that time is below `before`.
    Every other entry is kept, notably the tar
    archive itself and the fixed-name runtime files that the interior
    modules re-read between structure re-solves (``zalmoxis_output.dat``
    and its ``.prev`` backup, ``zalmoxis_output_temp.txt``,
    ``spider_mesh.dat``, and the EOS table directories). Pruned
    snapshots remain recoverable from the tar archive written by
    :func:`update` before this function runs.

    Arguments
    ---------
    dir : str
        The directory to remove old files from.
    before : float
        Remove snapshot files corresponding to simulated times before
        this time [years].
    """

    # Paths
    dir = os.path.abspath(dir)

    # Files
    files = glob.glob(os.path.join(dir, '*'))

    # Remove only recognized timestamped snapshots older than the cutoff
    for f in files:
        age = _snapshot_time(os.path.basename(f))
        if age is not None and age < before:
            safe_rm(f)
