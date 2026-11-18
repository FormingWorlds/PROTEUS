# Archival functions and utilities

# Import utils-specific modules
from __future__ import annotations

import glob
import logging
import os
import tarfile

from proteus.utils.helper import parse_subyear_time, safe_rm

log = logging.getLogger('fwl.' + __name__)


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


def create(dir: str, remove_files: bool = True) -> str:
    """
    Create a new tar archive from a directory of files, placing the tar inside that directory.

    Optionally removes all files in that directory (other than the tar file).

    Arguments
    ---------
    dir : str
        The directory to archive.
    remove_files:bool
        Whether to remove the appended files from the directory.

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
            tar_file.add(f, arcname=os.path.basename(f))

    # Remove the files that were archived (never the tar file itself)
    if remove_files:
        for f in files:
            if f == tar:
                continue
            safe_rm(f)

    # Return path to the tar file
    return tar


def append(dir: str, remove_files: bool = True) -> str:
    """
    Add files within `dir` into `dir/dir.tar`.

    The tar file must already exist.

    Arguments
    ---------
    dir:str
        Path to the archived directory.
    remove_files:bool
        Whether to remove the appended files from the directory.

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
        for f in files:
            tar_file.add(f, arcname=os.path.basename(f))

    # Remove the files that were appended (never the tar file itself)
    if remove_files:
        for f in files:
            if f == tar:
                continue
            safe_rm(f)

    return tar


def extract(dir: str, remove_tar: bool = False, ignore_warnings: bool = False) -> str:
    """
    Extract the tar file contained within a directory, placing the content within that directory.

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
        tar_file.extractall(dir, filter='data')

    # Remove tar file
    if remove_tar:
        safe_rm(tar)


def update(dir: str, remove_files: bool = True) -> None:
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
    """

    # Paths
    dir = os.path.abspath(dir)

    # Update archive
    if archive_exists(dir, ignore_warnings=True):
        append(dir, remove_files=remove_files)

    # Create new archive
    else:
        create(dir, remove_files=remove_files)


def archive_old(dir: str, before: float) -> None:
    """
    Move timestamped snapshots older than a cutoff time into the archive.

    Only snapshots recognised by :func:`_snapshot_time` whose time is below
    `before` are appended to the tar file (created if missing) and then
    removed from `dir`. Each snapshot therefore enters the tar once, when
    it leaves the disk. Every other entry stays in place, notably the
    fixed-name runtime files that the interior modules re-read between
    structure re-solves (``zalmoxis_output.dat`` and its ``.prev`` backup,
    ``zalmoxis_output_temp.txt``, ``spider_mesh.dat``, and the EOS table
    directories); the end-of-run :func:`update` archives those once.

    Arguments
    ---------
    dir : str
        The directory holding the snapshots and the tar file.
    before : float
        Archive snapshot files corresponding to simulated times before
        this time [years].
    """

    dir = os.path.abspath(dir)
    old = []
    for f in glob.glob(os.path.join(dir, '*')):
        age = _snapshot_time(os.path.basename(f))
        if age is not None and age < before:
            old.append(f)
    if not old:
        return

    log.debug(f'Moving {len(old)} snapshot files into the archive in {dir}')
    with tarfile.open(_tarfile_from_dir(dir), 'a') as tar_file:
        for f in old:
            tar_file.add(f, arcname=os.path.basename(f))
    for f in old:
        safe_rm(f)
