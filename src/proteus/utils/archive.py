# Archival functions and utilities

# Import utils-specific modules
from __future__ import annotations

import logging
import os
import glob
import tarfile

log = logging.getLogger("fwl."+__name__)

def _tarfile_from_dir(dir:str) -> str:
    name = os.path.split(dir)[-1]
    return os.path.join(dir, f"{name}.tar")

def create(dir:str, remove_files:bool=True) -> str:
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
    tar = _tarfile_from_dir(dir)
    log.debug(f"Creating new archive of {dir}")

    # Check if the directory exists
    if not os.path.exists(dir):
        log.error(f"Directory {dir} does not exist. Cannot archive.")
        return

    # Check if the tar file exists
    if os.path.exists(tar):
        log.warning(f"Tar file {tar} already exists. Overwriting.")
        os.remove(tar)

    # List files in directory
    files = glob.glob(os.path.join(dir, "*"))
    files = [os.path.abspath(f) for f in files]

    # Add files to new tar file
    with tarfile.open(tar, "w") as tar_file:
        for f in files:
            tar_file.add(f, arcname=os.path.split(f)[-1])

    # Remove files in that directory other than the tar file
    if remove_files:
        for f in files:
            if f != tar:  # do not remove the tar file itself
                os.remove(os.path.join(dir, f))

    # Return path to the tar file
    return tar

def append(dir:str, remove_files:bool=True) -> str:
    """
    Add files within `dir`, into `dir/dir.tar` excluding those in `exclude`.

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
    tar = _tarfile_from_dir(dir)
    log.debug(f"Appending files to archive in {dir}")

    # Check if the tar file exists
    if not os.path.exists(tar):
        log.error(f"Tar file {tar} does not exist. Cannot append to it.")
        return

    # List files in directory
    files = glob.glob(os.path.join(dir, "*"))
    files = [os.path.abspath(f) for f in files]

    # Append file to existing tar file
    with tarfile.open(tar, "a") as tar_file:
        for f in files:
            tar_file.add(f, arcname=os.path.split(f)[-1])

    # Remove appended files
    if remove_files:
        for f in files:
            if f != tar:  # do not remove the tar file itself
                os.remove(f)

    return tar

def extract(dir:str, remove_tar:bool=True) -> str:
    """
    Extract the tar file contained within a directory, placing the content within that directory.

    Removes the tar file afterwards if `remove_tar` is set to True.

    Arguments
    ---------
    dir : str
        The directory to extract.
    remove_tar : bool
        Whether to remove the tar file after extraction.

    Returns
    -------
    str
        The path to the tar file which was extracted.
    """

    # Paths
    dir = os.path.abspath(dir)
    tar = _tarfile_from_dir(dir)
    log.debug(f"Extracting tar file inside {dir}")

    # Check if the directory exists
    if not os.path.exists(dir):
        log.error(f"Directory {dir} does not exist. Cannot extract into it.")
        return

    # Check if the tar file exists
    if not os.path.exists(tar):
        log.error(f"Tar file {tar} does not exist. Cannot extract it.")
        return

    # Extract tar file
    with tarfile.open(tar, "r") as tar_file:
        tar_file.extractall(dir)

    # Remove tar file
    if remove_tar:
        os.remove(tar)

def update(dir:str, remove_files:bool=True) -> None:
    """
    Create and/or update a data file archive.

    This function archives the folder located at `dir/`. The files inside `dir/`
    are stored inside a tar file called `dir.tar` located in `dir/`.
    The tar files is created if it  does not already exist.
    Otherwise, it is updated with the contents of `dir/`, except for the files listed in exclude.

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
    tar = _tarfile_from_dir(dir)

    # Update archive
    if os.path.exists(tar):
        append(dir, remove_files=remove_files)
    # Create new archive
    else:
        create(dir, remove_files=remove_files)
