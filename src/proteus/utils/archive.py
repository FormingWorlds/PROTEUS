# Archival functions and utilities

# Import utils-specific modules
from __future__ import annotations

import logging
import os
import tarfile

log = logging.getLogger("fwl."+__name__)


def _new(dir:str, tar:str) -> None:
    """
    Archive a directory into a tar file.

    Assumes that the directory contains loose files. Does not store subdirectories.

    Arguments
    ---------
    dir : str
        The directory to archive.
    tar : str
        The name of the tar file to create.
    """
    log.debug(f"Archiving {dir} to {tar}")

    # Check if the directory exists
    if not os.path.exists(dir):
        log.error(f"Directory {dir} does not exist. Cannot archive.")
        return

    # Check if the tar file exists
    if os.path.exists(tar):
        log.warning(f"Tar file {tar} already exists. Overwriting.")
        os.remove(tar)

    # List files in directory
    files = os.listdir(dir)

    # Add files to new tar file
    with tarfile.open(tar, "w") as tar_file:
        for f in files:
            tar_file.add(os.path.join(dir,f))


def archive(dir:str, protect:list = []) -> str:
    """
    Create a new archive from a directory of files, placing it inside that directory.

    Removes all files in that directory other than the tar file created, and those provided
    in the list protect.

    Arguments
    ---------
    dir : str
        The directory to archive.
    protect : list
        A list of file names to protect from deletion.

    Returns
    -------
    str
        The path to the tar file created.
    """

    # Tar file
    name = f"{os.path.basename(dir)}.tar"
    tar = os.path.join(dir, name)

    # Protect tar file
    protect.append(name)

    # Create a new archive in that directory
    _new(dir, tar)

    # Remove files in that directory other than the tar file
    for f in os.listdir(dir):
        if f not in protect:
            os.remove(os.path.join(dir, f))

    # Return path to the tar file
    return tar

def append(tar:str,file:str) -> None:
    """
    Add file to an existing tar file.

    Arguments
    ---------
    tar : str
        The name of the tar file to add to.
    file : str
        The file to add to the tar file.
    """

    log.debug(f"Appending {file} to {tar}")

    # Check if the tar file exists
    if not os.path.exists(tar):
        log.error(f"Tar file {tar} does not exist. Cannot append to it.")
        return

    # Check if the file exists
    if not os.path.exists(file):
        log.error(f"File {file} does not exist. Cannot append to tar.")
        return

    # Append file to existing tar file
    with tarfile.open(tar, "a") as tar_file:
        tar_file.add(file)

def extract(dir:str, remove_after:bool=True) -> None:
    """
    Extract the tar file contained within a directory, placing the content within that directory.

    Removes the tar file afterwards.

    Arguments
    ---------
    dir : str
        The directory to extract.
    remove_after : bool
        Whether to remove the tar file after extraction.
    """

    dir = os.path.abspath(dir)

    tar = os.path.join(dir, f"{os.path.basename(dir)}.tar")
    log.debug(f"Extracting {tar} to {dir}/")

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
        tar_file.extractall()

    # Remove tar file
    if remove_after:
        os.remove(tar)
