from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import subprocess as sp
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

import numpy as np
import platformdirs
from osfclient.api import OSF
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.helper import safe_rm

log = logging.getLogger('fwl.' + __name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))
MAX_ATTEMPTS = 3
MAX_DLTIME = 120.0  # seconds
RETRY_WAIT = 5.0  # seconds

ARAGOG_BASIC = (
    '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa',
    'Melting_curves/Wolf_Bower+2018',
)

log.debug(f'FWL data location: {FWL_DATA_DIR}')


def download_zenodo_folder(zenodo_id: str, folder_dir: Path) -> bool:
    """
    Download a specific Zenodo record into specified folder

    Inputs :
        - zenodo_id : str
            Zenodo record ID to download
        - folder_dir : Path
            Local directory where the Zenodo record will be downloaded

    Returns :
        - zenodo_ok : bool
            Did the download/request complete successfully?
    """
    # Sanitize zenodo_id to prevent command injection
    # Zenodo IDs should only contain digits
    if not re.match(r'^[0-9]+$', zenodo_id):
        log.error(f'Invalid Zenodo ID format: {zenodo_id}. Must contain only digits.')
        return False

    # Check if zenodo_get is available
    try:
        sp.run(['zenodo_get', '--version'], capture_output=True, check=True, timeout=10)
    except (FileNotFoundError, sp.TimeoutExpired, sp.CalledProcessError) as e:
        log.error(f'zenodo_get command not available or not working: {e}')
        return False

    out = os.path.join(GetFWLData(), 'zenodo_download.log')
    log.debug(f'    zenodo_get, logging to {out}')

    # Use exponential backoff for retries
    for attempt in range(MAX_ATTEMPTS):
        # remove folder
        safe_rm(folder_dir)
        folder_dir.mkdir(parents=True, exist_ok=True)

        # try making request with timeout
        try:
            with open(out, 'w') as hdl:
                # Use Python's subprocess timeout for robust timeout handling
                # (zenodo_get's -t flag is not always respected by all versions)
                proc = sp.run(
                    ['zenodo_get', '-o', str(folder_dir), zenodo_id],
                    stdout=hdl,
                    stderr=sp.STDOUT,  # Combine stderr into stdout for better logging
                    timeout=MAX_DLTIME,  # Python's timeout will kill the process if it hangs
                    check=False,  # Don't raise on non-zero exit
                )

            # Check if command succeeded and folder has content
            if proc.returncode == 0:
                # Verify folder exists and has files
                if folder_dir.exists():
                    # Check if folder has any files (not just empty directory)
                    files = list(folder_dir.rglob('*'))
                    if files and any(f.is_file() for f in files):
                        log.info(f'Successfully downloaded Zenodo record {zenodo_id}')
                        return True
                    else:
                        log.warning(
                            f'Zenodo download completed but folder is empty (ID {zenodo_id})'
                        )
                else:
                    log.warning(
                        f'Zenodo download completed but folder does not exist (ID {zenodo_id})'
                    )
            else:
                # Read error from log file for better diagnostics
                error_msg = 'Unknown error'
                try:
                    with open(out, 'r') as f:
                        error_lines = f.readlines()[-10:]  # Last 10 lines
                        error_msg = ''.join(error_lines).strip()
                except Exception:
                    pass
                log.warning(
                    f'Failed to get data from Zenodo (ID {zenodo_id}, attempt {attempt + 1}/{MAX_ATTEMPTS}): '
                    f'exit code {proc.returncode}. Error: {error_msg[:500]}'
                )

        except sp.TimeoutExpired:
            log.warning(
                f'zenodo_get timed out after {MAX_DLTIME:.1f}s (ID {zenodo_id}, '
                f'attempt {attempt + 1}/{MAX_ATTEMPTS})'
            )
        except Exception as e:
            log.warning(f'Unexpected error during Zenodo download (ID {zenodo_id}): {e}')

        # Exponential backoff: wait longer between retries
        if attempt < MAX_ATTEMPTS - 1:
            wait_time = RETRY_WAIT * (2**attempt)  # Exponential backoff
            log.debug(f'Waiting {wait_time:.1f}s before retry...')
            sleep(wait_time)

    # Return status indicating that file/folder is invalid, if failed
    log.error(
        f'Could not obtain data for Zenodo record {zenodo_id} after {MAX_ATTEMPTS} attempts'
    )
    return False

def download_zenodo_file(zenodo_id: str, folder_dir: Path, record_path: str) -> bool:
    """
    Download a specific file from a Zenodo record into specified folder

    Inputs :
        - zenodo_id : str
            Zenodo record ID to download
        - folder_dir : Path
            Local directory where the Zenodo file will be downloaded
        - record_path : str
            Record-internal path/name of the file to download (passed to zenodo_get -g)

    Returns :
        - zenodo_ok : bool
            Did the download/request complete successfully?
    """
    # Sanitize zenodo_id to prevent command injection
    # Zenodo IDs should only contain digits
    if not re.match(r'^[0-9]+$', zenodo_id):
        log.error(f'Invalid Zenodo ID format: {zenodo_id}. Must contain only digits.')
        return False

    # Check if zenodo_get is available
    try:
        sp.run(['zenodo_get', '--version'], capture_output=True, check=True, timeout=10)
    except (FileNotFoundError, sp.TimeoutExpired, sp.CalledProcessError) as e:
        log.error(f'zenodo_get command not available or not working: {e}')
        return False

    out = os.path.join(GetFWLData(), 'zenodo_download.log')
    log.debug(f'    zenodo_get, logging to {out}')

    # Use exponential backoff for retries
    for attempt in range(MAX_ATTEMPTS):
        # remove folder
        safe_rm(folder_dir)
        folder_dir.mkdir(parents=True, exist_ok=True)

        # try making request with timeout
        try:
            with open(out, 'w') as hdl:
                # Use Python's subprocess timeout for robust timeout handling
                # (zenodo_get's -t flag is not always respected by all versions)
                proc = sp.run(
                    ['zenodo_get', '-o', str(folder_dir), '-g', record_path, zenodo_id],
                    stdout=hdl,
                    stderr=sp.STDOUT,  # Combine stderr into stdout for better logging
                    timeout=MAX_DLTIME,  # Python's timeout will kill the process if it hangs
                    check=False,  # Don't raise on non-zero exit
                )

            # Check if command succeeded and folder has content
            if proc.returncode == 0:
                # Verify folder exists and has files
                if folder_dir.exists():
                    # Check if folder has any files (not just empty directory)
                    files = list(folder_dir.rglob('*'))
                    if files and any(f.is_file() for f in files):
                        log.info(
                            f'Successfully downloaded Zenodo record {zenodo_id} (file: {record_path})'
                        )
                        return True
                    else:
                        log.warning(
                            f'Zenodo download completed but folder is empty (ID {zenodo_id}, file {record_path})'
                        )
                else:
                    log.warning(
                        f'Zenodo download completed but folder does not exist (ID {zenodo_id}, file {record_path})'
                    )
            else:
                # Read error from log file for better diagnostics
                error_msg = 'Unknown error'
                try:
                    with open(out, 'r') as f:
                        error_lines = f.readlines()[-10:]  # Last 10 lines
                        error_msg = ''.join(error_lines).strip()
                except Exception:
                    pass
                log.warning(
                    f'Failed to get data from Zenodo (ID {zenodo_id}, attempt {attempt + 1}/{MAX_ATTEMPTS}): '
                    f'exit code {proc.returncode}. Error: {error_msg[:500]}'
                )

        except sp.TimeoutExpired:
            log.warning(
                f'zenodo_get timed out after {MAX_DLTIME:.1f}s (ID {zenodo_id}, '
                f'attempt {attempt + 1}/{MAX_ATTEMPTS})'
            )
        except Exception as e:
            log.warning(f'Unexpected error during Zenodo download (ID {zenodo_id}): {e}')

        # Exponential backoff: wait longer between retries
        if attempt < MAX_ATTEMPTS - 1:
            wait_time = RETRY_WAIT * (2**attempt)  # Exponential backoff
            log.debug(f'Waiting {wait_time:.1f}s before retry...')
            sleep(wait_time)

    # Return status indicating that file/folder is invalid, if failed
    log.error(
        f'Could not obtain data for Zenodo record {zenodo_id} after {MAX_ATTEMPTS} attempts'
    )
    return False

def md5(_fname):
    """Return the md5 hash of a file."""

    # https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(_fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_zenodo_folder(zenodo_id: str, folder_dir: Path, hash_maxfilesize=100e6) -> bool:
    """
    Validate the content of a specific Zenodo-provided folder by checking md5 hashes

    Inputs :
        - zenodo_id : str
            Zenodo record ID to compare
        - folder_dir : Path
            Local directory where the Zenodo record has already been downloaded
        - hash_maxfilesize
            Don't validate the md5 hash of files greater than this size (bytes)

    Returns :
        - valid : bool
            Is folder valid?
    """
    # Sanitize zenodo_id to prevent command injection
    # Zenodo IDs should only contain digits
    if not re.match(r'^[0-9]+$', zenodo_id):
        log.error(f'Invalid Zenodo ID format: {zenodo_id}. Must contain only digits.')
        return False

    # Check if zenodo_get is available
    try:
        sp.run(['zenodo_get', '--version'], capture_output=True, check=True, timeout=10)
    except (FileNotFoundError, sp.TimeoutExpired, sp.CalledProcessError):
        # If zenodo_get not available, skip validation but warn
        log.warning('zenodo_get not available for validation - skipping hash check')
        # If folder exists and has files, assume it's valid
        if folder_dir.exists() and any(f.is_file() for f in folder_dir.rglob('*')):
            return True
        return False

    # Use zenodo_get to obtain md5 hashes
    #     They will be saved to a txt file in folder_dir
    md5sums_path = os.path.join(folder_dir, 'md5sums.txt')
    out = os.path.join(GetFWLData(), 'zenodo_validate.log')
    zenodo_ok = False

    for attempt in range(MAX_ATTEMPTS):
        # remove file
        safe_rm(md5sums_path)

        # try making request with timeout
        try:
            with open(out, 'w') as hdl:
                proc = sp.run(
                    ['zenodo_get', '-m', zenodo_id],
                    stdout=hdl,
                    stderr=sp.STDOUT,
                    cwd=str(folder_dir),
                    timeout=60,  # Shorter timeout for validation
                    check=False,
                )

            # process exited fine and file exists?
            zenodo_ok = (proc.returncode == 0) and os.path.isfile(md5sums_path)

            # try again?
            if zenodo_ok:
                break
            else:
                log.warning(
                    f'Failed to get checksum from Zenodo (ID {zenodo_id}, '
                    f'attempt {attempt + 1}/{MAX_ATTEMPTS})'
                )
        except sp.TimeoutExpired:
            log.warning(f'zenodo_get validation timed out (ID {zenodo_id})')
        except Exception as e:
            log.warning(f'Unexpected error during Zenodo validation (ID {zenodo_id}): {e}')

        if attempt < MAX_ATTEMPTS - 1:
            sleep(RETRY_WAIT * (2**attempt))  # Exponential backoff

    # Return status indicating that file/folder is invalid, if failed
    if not zenodo_ok:
        log.warning(
            f'Could not obtain checksum for Zenodo record {zenodo_id} - skipping validation'
        )
        # If we can't validate but folder exists with files, assume it's valid
        if folder_dir.exists() and any(f.is_file() for f in folder_dir.rglob('*')):
            log.info(f'Folder exists with files - assuming valid (ID {zenodo_id})')
            return True
        return False

    # Read hashes file
    try:
        with open(md5sums_path, 'r') as hdl:
            md5sums = hdl.readlines()
    except Exception as e:
        log.warning(f'Could not read md5sums file: {e}')
        # If folder has files, assume valid
        if folder_dir.exists() and any(f.is_file() for f in folder_dir.rglob('*')):
            return True
        return False

    # Check each item in the record...
    for line in md5sums:
        if not line.strip():
            continue
        try:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sum_expect, name = parts[0], parts[1]
            file = os.path.join(folder_dir, name)

            # exit here if file does not exist
            if not os.path.exists(file):
                log.warning(f'Detected missing file {name} (Zenodo record {zenodo_id})')
                return False

            # skip symbolic links for security
            if os.path.islink(file):
                log.debug(f'Skipping symbolic link: {name}')
                continue

            # don't check the hashes of very large files, because it's slow
            if os.path.getsize(file) > hash_maxfilesize:
                continue  # Skip hash check but file exists

            # check the actual hash of the file on disk, compare to expected
            sum_actual = md5(file).strip()
            if sum_actual != sum_expect:
                log.warning(f'Detected invalid file {name} (Zenodo record {zenodo_id})')
                log.warning(f'    expected hash {sum_expect}, got {sum_actual}')
                return False
        except Exception as e:
            log.warning(f'Error validating file from md5sums line: {e}')
            continue

    return True


# Unified mapping of folder names to both Zenodo and OSF identifiers
# Structure: folder_name -> {'zenodo_id': str, 'osf_id': str, 'osf_project': str}
DATA_SOURCE_MAP: dict[str, dict[str, str]] = {
    # Spectral files (OSF project: vehxg)
    'Frostflow/16': {'zenodo_id': '15799743', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Frostflow/48': {'zenodo_id': '15696415', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Frostflow/256': {'zenodo_id': '15799754', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Frostflow/4096': {'zenodo_id': '15799776', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Dayspring/16': {'zenodo_id': '15799318', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Dayspring/48': {'zenodo_id': '15721749', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Dayspring/256': {'zenodo_id': '15799474', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Dayspring/4096': {'zenodo_id': '15799495', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Honeyside/16': {'zenodo_id': '15799607', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Honeyside/48': {'zenodo_id': '15799652', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Honeyside/256': {'zenodo_id': '15799731', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Honeyside/4096': {'zenodo_id': '15696457', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    'Oak/318': {'zenodo_id': '15743843', 'osf_id': 'vehxg', 'osf_project': 'vehxg'},
    # Interior lookup tables (OSF project: phsxf)
    '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018': {
        'zenodo_id': '15877374',
        'osf_id': 'phsxf',
        'osf_project': 'phsxf',
    },
    '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_400GPa': {
        'zenodo_id': '15877424',
        'osf_id': 'phsxf',
        'osf_project': 'phsxf',
    },
    '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa': {
        'zenodo_id': '17417017',
        'osf_id': 'phsxf',
        'osf_project': 'phsxf',
    },
    # Melting curves (OSF project: phsxf)
    'Melting_curves/Monteux+600': {
        'zenodo_id': '15728091',
        'osf_id': 'phsxf',
        'osf_project': 'phsxf',
    },
    'Melting_curves/Monteux-600': {
        'zenodo_id': '15728138',
        'osf_id': 'phsxf',
        'osf_project': 'phsxf',
    },
    'Melting_curves/Wolf_Bower+2018': {
        'zenodo_id': '15728072',
        'osf_id': 'phsxf',
        'osf_project': 'phsxf',
    },
    # Surface albedos (OSF project: 2gcd9)
    'Hammond24': {'zenodo_id': '15880455', 'osf_id': '2gcd9', 'osf_project': '2gcd9'},
    # Stellar spectra (OSF project: 8r2sw)
    'Named': {'zenodo_id': '15721440', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Stellar spectra - PHOENIX (OSF project: 8r2sw)
    'PHOENIX': {'zenodo_id': '17674612', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Stellar spectra - MUSCLES (OSF project: 8r2sw)
    'MUSCLES': {'zenodo_id': '17802209', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Stellar spectra - solar (OSF project: 8r2sw)
    'solar': {'zenodo_id': '17981836', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Exoplanet data (OSF project: fzwr4)
    'Exoplanets': {'zenodo_id': '15727878', 'osf_id': 'fzwr4', 'osf_project': 'fzwr4'},
    # Mass-radius data (OSF project: xge8t)
    'Zeng2019': {'zenodo_id': '15727899', 'osf_id': 'xge8t', 'osf_project': 'xge8t'},
    # Population data (OSF project: dpkjb)
    # NOTE: Population and EOS_Seager2007 currently share Zenodo ID '15727998'.
    'Population': {'zenodo_id': '15727998', 'osf_id': 'dpkjb', 'osf_project': 'dpkjb'},
    # EOS material properties (OSF project: dpkjb)
    'EOS_Seager2007': {'zenodo_id': '15727998', 'osf_id': 'dpkjb', 'osf_project': 'dpkjb'},
}


def get_data_source_info(folder: str) -> dict[str, str] | None:
    """
    Get both Zenodo and OSF identifiers for a given folder.

    Parameters
    ----------
    folder : str
        Folder name to get identifiers for

    Returns
    -------
    dict[str, str] | None
        Dictionary with 'zenodo_id', 'osf_id', and 'osf_project' keys, or None if not found
    """
    return DATA_SOURCE_MAP.get(folder, None)


def get_zenodo_record(folder: str) -> str | None:
    """
    Get Zenodo record ID for a given folder.

    Inputs :
        - folder : str
            Folder name to get the Zenodo record ID for

    Returns :
        - str | None : Zenodo record ID or None if not found
    """
    info = get_data_source_info(folder)
    return info.get('zenodo_id') if info else None


def get_osf_project(folder: str) -> str | None:
    """
    Get OSF project ID for a given folder.

    Parameters
    ----------
    folder : str
        Folder name to get the OSF project ID for

    Returns
    -------
    str | None
        OSF project ID or None if not found
    """
    info = get_data_source_info(folder)
    return info.get('osf_project') if info else None


def get_zenodo_from_osf(osf_id: str) -> list[str]:
    """
    Get all Zenodo record IDs associated with an OSF project.

    Parameters
    ----------
    osf_id : str
        OSF project ID

    Returns
    -------
    list[str]
        List of Zenodo record IDs in that OSF project
    """
    return [
        info['zenodo_id']
        for info in DATA_SOURCE_MAP.values()
        if info.get('osf_project') == osf_id and 'zenodo_id' in info
    ]


def get_osf_from_zenodo(zenodo_id: str) -> str | None:
    """
    Get OSF project ID associated with a Zenodo record.

    Parameters
    ----------
    zenodo_id : str
        Zenodo record ID

    Returns
    -------
    str | None
        OSF project ID or None if not found
    """
    for info in DATA_SOURCE_MAP.values():
        if info.get('zenodo_id') == zenodo_id:
            return info.get('osf_project')
    return None


def download_OSF_folder(*, storage, folders: list[str], data_dir: Path):
    """
    Download a specific folder in the OSF repository

    Inputs :
        - storage : OSF storage name
        - folders : folder names to download
        - data_dir : local repository where data are saved
    """
    downloaded_files = 0
    total_size = 0

    try:
        # Iterate through all files in OSF storage
        for file in storage.files:
            for folder in folders:
                # Check if file path matches folder (handle both with and without leading slash)
                file_path = file.path.lstrip('/')
                folder_path = folder.lstrip('/')

                if not file_path.startswith(folder_path):
                    continue

                # Extract relative path parts
                parts = file.path.lstrip('/').split('/')
                target = Path(data_dir, *parts)
                target.parent.mkdir(parents=True, exist_ok=True)

                # Skip if file already exists and is not empty
                if target.exists() and target.stat().st_size > 0:
                    log.debug(f'Skipping existing file: {file.path}')
                    continue

                try:
                    log.info(f'Downloading {file.path} ({file.size / 1024 / 1024:.1f} MB)...')
                    with open(target, 'wb') as f:
                        file.write_to(f)
                    downloaded_files += 1
                    total_size += target.stat().st_size
                except Exception as e:
                    log.warning(f'Failed to download {file.path}: {e}')
                    # Remove partial file
                    if target.exists():
                        try:
                            target.unlink()
                        except Exception:
                            pass
                    continue
                break

        if downloaded_files > 0:
            log.info(
                f'Downloaded {downloaded_files} file(s) from OSF '
                f'({total_size / 1024 / 1024:.1f} MB total)'
            )
        else:
            log.warning(f'No files downloaded from OSF for folders: {folders}')

    except Exception as e:
        log.error(f'Error accessing OSF storage: {e}')
        raise


def GetFWLData() -> Path:
    """
    Get path to FWL data directory on the disk
    """
    return Path(FWL_DATA_DIR).absolute()


@functools.cache
def get_osf(id: str):
    """
    Generate an object to access OSF storage
    """
    osf = OSF()
    project = osf.project(id)
    return project.storage('osfstorage')


def check_needs_update(dir, zenodo):
    """
    Check whether the folder 'dir' needs to be re-downloaded.

    This is the case when it is missing, outdated, or corrupted.

    Inputs :
        - dir : folder path
        - zenodo : zenodo record ID
    """

    log.debug(f'Checking whether {dir} needs updating (record {zenodo})')

    # Trivial case where folder is missing
    if not os.path.isdir(dir):
        return True

    # Folder exists but cannot check hashes, so exit here
    if not zenodo:
        return False  # don't update

    # Folder exists... use Zenodo to check MD5 hashes
    return not validate_zenodo_folder(zenodo, dir)


def download(
    *,
    folder: str,
    target: str,
    osf_id: str | None = None,
    zenodo_id: str | None = None,
    desc: str,
    force: bool = False,
) -> bool:
    """
    Generic download function with automatic source mapping.

    This function can automatically look up OSF and Zenodo IDs from the unified
    DATA_SOURCE_MAP if they are not provided. If both are provided, uses them directly.

    Attributes
    ----------
    folder: str
        Filename to download
    target: str
        name of target directory
    osf_id: str | None
        OSF project id (optional, will be looked up from mapping if not provided)
    zenodo_id: str | None
        Zenodo record id (optional, will be looked up from mapping if not provided)
    desc: str
        Description for logging
    force: bool
        Force a re-download even if valid

    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise
    """
    log.debug(f'Get {desc}?')

    # Try to get source info from mapping if IDs not provided
    source_info = get_data_source_info(folder)
    if source_info:
        # Use mapping values if not explicitly provided
        if zenodo_id is None:
            zenodo_id = source_info.get('zenodo_id')
        if osf_id is None:
            osf_id = source_info.get('osf_project')
        log.debug(f'Using mapped source info: Zenodo={zenodo_id}, OSF={osf_id}')
    elif zenodo_id is None and osf_id is None:
        log.warning(f'No source mapping found for {folder} and no IDs provided')
        log.warning(f'  Cannot download {desc} without source identifiers')
        return False

    # Check that target FWL_DATA folder exists
    data_dir = GetFWLData() / target
    data_dir.mkdir(parents=True, exist_ok=True)

    # Path to specific folder (download) within the data_dir folder
    folder_dir = data_dir / folder

    # Check if the folder needs updating
    folder_invalid = check_needs_update(folder_dir, zenodo_id) or force

    # Update the folder
    if folder_invalid:
        log.info(f'Downloading {desc} to {data_dir}')
        success = False

        # Try Zenodo in the first instance
        if zenodo_id is not None:
            try:
                # download the folder
                if download_zenodo_folder(zenodo_id=zenodo_id, folder_dir=folder_dir):
                    # files validated ok?
                    success = validate_zenodo_folder(zenodo_id, folder_dir)
            except RuntimeError as e:
                log.warning(f'    Zenodo download failed: {e}')
                if folder_dir.exists():
                    try:
                        folder_dir.rmdir()
                    except Exception:
                        pass  # Ignore cleanup errors
        else:
            log.debug('    No Zenodo ID provided, skipping Zenodo download')

        if success:
            return True

        # If Zenodo fails or not available, try OSF
        if osf_id:
            try:
                log.info(f'Attempting OSF fallback download (project {osf_id})...')
                storage = get_osf(osf_id)
                download_OSF_folder(storage=storage, folders=[folder], data_dir=data_dir)

                # Verify OSF download succeeded
                if folder_dir.exists() and any(f.is_file() for f in folder_dir.rglob('*')):
                    log.info(f'Successfully downloaded {desc} from OSF (project {osf_id})')
                    success = True
                else:
                    log.warning(f'OSF download completed but folder is empty: {folder_dir}')
                    success = False
            except Exception as e:
                log.warning(f'    OSF download failed: {e}')
                import traceback

                log.debug(f'OSF download traceback: {traceback.format_exc()}')
                success = False
        else:
            log.warning(f'No OSF project ID available for {desc}')

        if success:
            return True

        log.error(f'    Failed to download {desc} from IDs: Zenodo {zenodo_id}, OSF {osf_id}')
        return False

    else:
        log.debug(f'    {desc} already exists')
    return True


def download_surface_albedos():
    """
    Download reflectance data for various surface materials
    """
    folder = 'Hammond24'
    source_info = get_data_source_info(folder)
    if not source_info:
        raise ValueError(f'No data source mapping found for folder: {folder}')

    download(
        folder=folder,
        target='surface_albedos',
        osf_id=source_info['osf_project'],
        zenodo_id=source_info['zenodo_id'],
        desc='surface reflectance data',
    )


def download_spectral_file(name: str, bands: str):
    """
    Download spectral file.

    Inputs :
        - name : str
            folder name (e.g. "Dayspring")
        - bands : str
            number of bands (e.g. "256")
    """
    # Check name and bands
    if not isinstance(name, str) or (len(name) < 1):
        raise Exception('Must provide name of spectral file')
    if not isinstance(bands, str) or (len(bands) < 1):
        raise Exception('Must provide number of bands in spectral file')

    folder = f'{name}/{bands}'
    source_info = get_data_source_info(folder)
    if not source_info:
        raise ValueError(f'No data source mapping found for folder: {folder}')

    download(
        folder=folder,
        target='spectral_files',
        osf_id=source_info['osf_project'],
        zenodo_id=source_info['zenodo_id'],
        desc=f'{name}{bands} spectral file',
    )


def download_phoenix(*, alpha: float = 0.0, FeH: float = 0.0, force: bool = False) -> bool:
    """Download PHOENIX stellar spectra into `FWL_DATA`.

    Used by `proteus.star.phoenix`. The current implementation downloads the
    PHOENIX bundle via the unified `download()` mechanism.
    """
    desc = f'PHOENIX stellar spectra (alpha={alpha:+0.1f}, [Fe/H]={FeH:+0.1f})'
    return download(folder='PHOENIX', target='stellar_spectra', desc=desc, force=force)


def download_interior_lookuptables(clean=False):
    """
    Download basic interior lookup tables
    """
    log.debug('Download basic interior lookup tables')

    data_dir = GetFWLData() / 'interior_lookup_tables'
    data_dir.mkdir(parents=True, exist_ok=True)

    for dir in ARAGOG_BASIC:
        folder_dir = data_dir / dir
        if clean:
            safe_rm(folder_dir.as_posix())
        source_info = get_data_source_info(dir)
        if not source_info:
            log.warning(f'No data source mapping found for {dir}, skipping')
            continue

        download(
            folder=dir,
            target=data_dir,
            osf_id=source_info['osf_project'],
            zenodo_id=source_info['zenodo_id'],
            desc=f'Interior lookup tables: {dir}',
        )


def download_melting_curves(config: Config, clean=False):
    """
    Download melting curve data
    """
    log.debug('Download melting curve data')
    dir = 'Melting_curves/' + config.interior.melting_dir

    data_dir = GetFWLData() / 'interior_lookup_tables'
    data_dir.mkdir(parents=True, exist_ok=True)

    folder_dir = data_dir / dir
    if clean:
        safe_rm(folder_dir.as_posix())
    source_info = get_data_source_info(dir)
    if not source_info:
        raise ValueError(f'No data source mapping found for folder: {dir}')

    download(
        folder=dir,
        target=data_dir,
        osf_id=source_info['osf_project'],
        zenodo_id=source_info['zenodo_id'],
        desc=f'Melting curve data: {dir}',
    )


def download_stellar_spectra(*, folders: tuple[str, ...] | None = None):
    """
    Download stellar spectra folders into ``FWL_DATA/stellar_spectra``.

    Notes
    -----
    PROTEUS expects different stellar spectra collections in subfolders like:
    - ``stellar_spectra/solar/``
    - ``stellar_spectra/MUSCLES/``
    - ``stellar_spectra/Named/``

    Parameters
    ----------
    folders:
        Specific folders to download. If None, downloads a minimal set that covers
        common configurations.
    """
    if folders is None:
        # Minimal set for most configurations:
        # - Named: general named spectra
        # - solar: solar spectra (e.g., sun.txt, Sun0.6Ga.txt, ...)
        # - MUSCLES: observed stellar spectra catalogue
        folders = ('Named', 'solar', 'MUSCLES')

    for folder in folders:
        source_info = get_data_source_info(folder)
        if not source_info:
            raise ValueError(f'No data source mapping found for folder: {folder}')

        download(
            folder=folder,
            target='stellar_spectra',
            osf_id=source_info['osf_project'],
            zenodo_id=source_info['zenodo_id'],
            desc=f'stellar spectra ({folder})',
        )


def download_exoplanet_data():
    """
    Download exoplanet data
    """
    folder = 'Exoplanets'
    source_info = get_data_source_info(folder)
    if not source_info:
        raise ValueError(f'No data source mapping found for folder: {folder}')

    download(
        folder=folder,
        target='planet_reference',
        osf_id=source_info['osf_project'],
        zenodo_id=source_info['zenodo_id'],
        desc='exoplanet data',
    )


def download_massradius_data():
    """
    Download mass-radius data
    """
    folder = 'Zeng2019'
    source_info = get_data_source_info(folder)
    if not source_info:
        raise ValueError(f'No data source mapping found for folder: {folder}')

    download(
        folder=folder,
        target='mass_radius',
        osf_id=source_info['osf_project'],
        zenodo_id=source_info['zenodo_id'],
        desc='mass radius data',
    )


def download_stellar_tracks(track: str, use_osf_fallback: bool = True):
    """
    Download stellar evolution tracks

    Uses the function built-into MORS. Falls back to OSF if MORS download fails.

    Parameters
    ----------
    track : str
        Track name ('Spada' or 'Baraffe')
    use_osf_fallback : bool
        If True, attempt OSF download if MORS download fails
    """
    from mors.data import DownloadEvolutionTracks

    log.debug(f'Downloading stellar evolution tracks: {track}')

    # Try MORS download first
    try:
        DownloadEvolutionTracks(track)
        # Verify download succeeded by checking if tracks directory exists
        fwl_data = GetFWLData()
        tracks_path = fwl_data / 'stellar_evolution_tracks' / track
        if tracks_path.exists() and any(tracks_path.iterdir()):
            log.info(f'Successfully downloaded {track} tracks via MORS')
            return
        else:
            log.warning(f'MORS download completed but tracks not found at {tracks_path}')
            raise FileNotFoundError(f'Tracks directory empty or missing: {tracks_path}')
    except Exception as e:
        log.warning(f'MORS download failed for {track} tracks: {e}')

        if not use_osf_fallback:
            raise

        # Fallback to OSF if available
        # Note: OSF project ID for stellar tracks may need to be determined
        # For now, log that we're attempting OSF fallback
        log.info(f'Attempting OSF fallback for {track} tracks...')

        # Try to find OSF project with stellar tracks
        # Common OSF project IDs in PROTEUS: 'phsxf' (ARAGOG), '8r2sw' (stellar spectra)
        # Stellar tracks might be in a different project - would need to check MORS docs
        try:
            # Try common OSF projects that might have stellar data
            osf_projects = ['8r2sw']  # Stellar spectra project - might also have tracks

            fwl_data = GetFWLData()
            tracks_dir = fwl_data / 'stellar_evolution_tracks'
            tracks_dir.mkdir(parents=True, exist_ok=True)
            target_dir = tracks_dir / track

            for osf_id in osf_projects:
                try:
                    storage = get_osf(osf_id)
                    # Try to download from OSF
                    # Note: Folder structure on OSF may differ - this is a best-effort attempt
                    download_OSF_folder(
                        storage=storage,
                        folders=[f'stellar_evolution_tracks/{track}', track],
                        data_dir=tracks_dir,
                    )
                    if target_dir.exists() and any(target_dir.iterdir()):
                        log.info(
                            f'Successfully downloaded {track} tracks via OSF (project {osf_id})'
                        )
                        return
                except Exception as osf_e:
                    log.debug(f'OSF project {osf_id} did not have {track} tracks: {osf_e}')
                    continue

            log.error(
                f'Could not download {track} tracks via MORS or OSF fallback. '
                f'You may need to download manually or check network connectivity.'
            )
            raise RuntimeError(
                f'Failed to download {track} tracks: MORS failed, OSF fallback unavailable'
            )

        except Exception as osf_fallback_error:
            log.error(f'OSF fallback also failed for {track} tracks: {osf_fallback_error}')
            raise RuntimeError(
                f'Failed to download {track} tracks: MORS error ({e}), OSF fallback error ({osf_fallback_error})'
            )


def _get_sufficient(config: Config, clean: bool = False):
    # Star stuff
    if config.star.module == 'mors':
        # Download only the spectra collections that this config may require.
        # (PHOENIX is handled separately via download_phoenix().)
        spec_src = config.star.mors.spectrum_source
        folders: list[str] = ['Named']
        if spec_src in (None, 'solar'):
            folders.append('solar')
        if spec_src in (None, 'muscles'):
            folders.append('MUSCLES')
        download_stellar_spectra(folders=tuple(dict.fromkeys(folders)))
        if config.star.mors.tracks == 'spada':
            download_stellar_tracks('Spada')
        else:
            download_stellar_tracks('Baraffe')

    # Spectral files
    if config.atmos_clim.module in ('janus', 'agni'):
        # High-res file often used for post-processing
        download_spectral_file('Honeyside', '4096')

        # Get the spectral file we need for this simluation
        from proteus.atmos_clim.common import get_spfile_name_and_bands

        group, bands = get_spfile_name_and_bands(config)
        download_spectral_file(group, bands)

    # Surface single-scattering data
    if config.atmos_clim.module == 'agni':
        download_surface_albedos()

    # Exoplanet population data
    download_exoplanet_data()

    # Mass-radius reference data
    download_massradius_data()

    # Interior look up tables
    if config.interior.module == 'aragog':
        download_interior_lookuptables(clean=clean)
        download_melting_curves(config, clean=clean)


def download_sufficient_data(config: Config, clean: bool = False):
    """
    Download the required data based on the current options
    """

    log.info('Getting physical and reference data')

    if config.params.offline:
        # Don't try to get data
        log.warning('Running in offline mode. Will not check for reference data.')

    else:
        # Try to get data
        try:
            _get_sufficient(config, clean=clean)

        # Some issue. Usually due to lack of internet connection, but print the error
        #     anyway so that the user knows what happened.
        except OSError as e:
            log.warning('Problem when downloading/checking reference data')
            log.warning(str(e))

    log.info(' ')


def _none_dirs():
    from proteus.utils.helper import get_proteus_dir

    dirs = {'proteus': get_proteus_dir()}
    dirs['tools'] = os.path.join(dirs['proteus'], 'tools')
    return dirs


def get_socrates(dirs=None):
    """
    Download and install SOCRATES
    """

    log.info('Setting up SOCRATES')

    # None dirs
    if dirs is None:
        dirs = _none_dirs()

    # Get path
    workpath = os.path.join(dirs['proteus'], 'SOCRATES')
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        log.debug('    already set up')
        return

    # Download, configure, and build
    log.debug('Running get_socrates.sh')
    cmd = [os.path.join(dirs['tools'], 'get_socrates.sh'), workpath]
    out = os.path.join(dirs['proteus'], 'nogit_setup_socrates.log')
    log.debug('    logging to %s' % out)
    with open(out, 'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    # Set environment
    os.environ['RAD_DIR'] = workpath
    log.debug('    done')


def get_petsc(dirs=None):
    """
    Download and install PETSc
    """

    log.info('Setting up PETSc')

    if dirs is None:
        dirs = _none_dirs()

    # Get path
    workpath = os.path.join(dirs['proteus'], 'petsc')
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        # already downloaded
        log.debug('    already set up')
        return

    # Download, configure, and build
    log.debug('Running get_petsc.sh')
    cmd = [os.path.join(dirs['tools'], 'get_petsc.sh'), workpath]
    out = os.path.join(dirs['proteus'], 'nogit_setup_petsc.log')
    log.debug('    logging to %s' % out)
    with open(out, 'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    log.debug('    done')


def get_spider(dirs=None):
    """
    Download and install SPIDER
    """

    if dirs is None:
        dirs = _none_dirs()

    # Need to install PETSc first
    get_petsc(dirs)

    log.info('Setting up SPIDER')

    # Get path
    workpath = os.path.join(dirs['proteus'], 'SPIDER')
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        # already downloaded
        log.debug('    already set up')
        return

    # Download, configure, and build
    log.debug('Running get_spider.sh')
    cmd = [os.path.join(dirs['tools'], 'get_spider.sh'), workpath]
    out = os.path.join(dirs['proteus'], 'nogit_setup_spider.log')
    log.debug('    logging to %s' % out)
    with open(out, 'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    log.debug('    done')


def download_Seager_EOS():
    """
    Download EOS material properties from Seager et al. (2007)
    """
    folder = 'EOS_Seager2007'
    source_info = get_data_source_info(folder)
    if not source_info:
        raise ValueError(f'No data source mapping found for folder: {folder}')

    download(
        folder=folder,
        target='EOS_material_properties',
        osf_id=source_info['osf_project'],
        zenodo_id=source_info['zenodo_id'],
        desc='EOS Seager2007 material files',
    )


def load_melting_curve(melt_file):
    """
    Loads melting curve data for MgSiO3 from a text file.
    Parameters:
        melt_file: Path to the melting curve data file
    Returns:
        interp_func: Interpolation function for T(P)
    """
    try:
        data = np.loadtxt(melt_file, comments='#')
        pressures = data[:, 0]  # in Pa
        temperatures = data[:, 1]  # in K
        interp_func = interp1d(
            pressures, temperatures, kind='linear', bounds_error=False, fill_value=np.nan
        )
        return interp_func
    except Exception as e:
        print(f'Error loading melting curve data: {e}')
        return None


def get_zalmoxis_melting_curves(config: Config):
    """
    Loads and returns the solidus and liquidus melting curves for temperature-dependent silicate mantle EOS. This is for use with Zalmoxis.
    Zalmoxis currently only supports the 'Monteux-600' melting curves directory.
    Parameters:
        config: Configuration object containing interior settings
    Returns: A tuple containing the solidus and liquidus data files.
    """
    if config.interior.melting_dir != 'Monteux-600':
        raise ValueError(
            f"Zalmoxis currently only supports 'Monteux-600' for melting_dir. You have configured '{config.interior.melting_dir}'."
        )
    melting_curves_folder = (
        FWL_DATA_DIR / 'interior_lookup_tables' / 'Melting_curves' / config.interior.melting_dir
    )
    solidus_func = load_melting_curve(melting_curves_folder / 'solidus.dat')
    liquidus_func = load_melting_curve(melting_curves_folder / 'liquidus.dat')
    melting_curves_functions = (solidus_func, liquidus_func)
    return melting_curves_functions


def get_zalmoxis_EOS():
    """
    Build and return material properties dictionaries for Seager et al. (2007) EOS data. This is for use with Zalmoxis.
    Returns:
        tuple: A tuple containing three dictionaries for iron/silicate planets, iron/Tdep_silicate planets, and water planets.
    """
    # Define the EOS folder paths
    Seager_eos_folder = FWL_DATA_DIR / 'EOS_material_properties' / 'EOS_Seager2007'
    Wolf_Bower_eos_folder = (
        FWL_DATA_DIR
        / 'interior_lookup_tables'
        / '1TPa-dK09-elec-free'
        / 'MgSiO3_Wolf_Bower_2018_1TPa'
    )

    # Download the EOS material properties if not already present
    if not Seager_eos_folder.exists():
        log.debug('Get EOS material properties from Seager et al. (2007)')
        download_Seager_EOS()

    # Build the material_properties_iron_silicate_planets dictionary for iron/silicate planets according to Seager et al. (2007)
    material_properties_iron_silicate_planets = {
        'core': {
            # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
            'eos_file': Seager_eos_folder / 'eos_seager07_iron.txt'
        },
        'mantle': {
            # Silicate, modeled in Seager et al. (2007) using the fourth-order Birch-Murnaghan EOS fit to MgSiO3 perovskite and DFT calculations
            'eos_file': Seager_eos_folder / 'eos_seager07_silicate.txt'
        },
    }

    # Build the material_properties_iron_Tdep_silicate_planets dictionary for iron/silicate planets with temperature-dependent silicate mantle EOS from Wolf & Bower (2018)
    material_properties_iron_Tdep_silicate_planets = {
        'core': {
            # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
            'eos_file': Seager_eos_folder / 'eos_seager07_iron.txt'
        },
        'melted_mantle': {
            # MgSiO3 in melt state, modeled in Wolf & Bower (2018) using their developed high P–T RTpress EOS
            'eos_file': Wolf_Bower_eos_folder / 'density_melt.dat'
        },
        'solid_mantle': {
            # MgSiO3 in solid state, modeled in Wolf & Bower (2018) using their developed high P–T RTpress EOS
            'eos_file': Wolf_Bower_eos_folder / 'density_solid.dat'
        },
    }

    # Build the material_properties_water_planets dictionary for water planets according to Seager et al. (2007)
    material_properties_water_planets = {
        'core': {
            # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
            'eos_file': Seager_eos_folder / 'eos_seager07_iron.txt'
        },
        'mantle': {
            # Silicate, modeled in Seager et al. (2007) using the fourth-order Birch-Murnaghan EOS fit to MgSiO3 perovskite and DFT calculations
            'eos_file': Seager_eos_folder / 'eos_seager07_silicate.txt'
        },
        'water_ice_layer': {
            # Water ice, modeled in Seager et al. (2007) using experimental data, DFT predictions for water ice in phases VIII and X, and DFT calculations
            'eos_file': Seager_eos_folder / 'eos_seager07_water.txt'
        },
    }
    return (
        material_properties_iron_silicate_planets,
        material_properties_iron_Tdep_silicate_planets,
        material_properties_water_planets,
    )


def get_Seager_EOS():
    """Backward-compatible Seager EOS helper.

    Returns the Seager et al. (2007) EOS material property dictionaries for
    iron/silicate and water planets. This mirrors the original
    ``get_Seager_EOS`` API that older code and tests expect.

    The implementation reuses :func:`get_zalmoxis_EOS` and returns only the
    iron/silicate and water dictionaries.
    """

    iron_silicate, _iron_Tdep, water = get_zalmoxis_EOS()
    return iron_silicate, water
