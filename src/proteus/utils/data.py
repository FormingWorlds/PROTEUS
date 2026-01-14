from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess as sp
import time
import zipfile
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

import platformdirs
from osfclient.api import OSF

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.helper import safe_rm
from proteus.utils.phoenix_helper import phoenix_param, phoenix_to_grid

log = logging.getLogger("fwl."+__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))
MAX_ATTEMPTS = 3
MAX_DLTIME   = 120.0 # seconds
RETRY_WAIT   = 5.0   # seconds
ZENODO_COOLDOWN = 2.0  # seconds between Zenodo API requests to avoid overwhelming server

ARAGOG_BASIC = (
    "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa",
    "Melting_curves/Wolf_Bower+2018",
    )

log.debug(f'FWL data location: {FWL_DATA_DIR}')

# Rate limiting for Zenodo API requests
_last_zenodo_request_time = 0.0


def _zenodo_cooldown():
    """
    Ensure cooldown between Zenodo API requests to avoid overwhelming the server.

    This function enforces a minimum delay (ZENODO_COOLDOWN seconds) between
    consecutive Zenodo API requests to prevent rate limiting and server overload.
    """
    global _last_zenodo_request_time
    current_time = time.time()
    time_since_last = current_time - _last_zenodo_request_time
    if time_since_last < ZENODO_COOLDOWN:
        wait_time = ZENODO_COOLDOWN - time_since_last
        log.debug(f'Zenodo cooldown: waiting {wait_time:.2f}s before next request')
        sleep(wait_time)
    _last_zenodo_request_time = time.time()


def _has_zenodo_token() -> bool:
    """
    Check if Zenodo API token is available.

    Checks for Zenodo API token in environment variables or PyStow config file.
    The zenodo_client library requires an API token for authentication.

    Returns
    -------
    bool
        True if token is configured, False otherwise.
    """
    # Check environment variable
    if os.environ.get('ZENODO_API_TOKEN'):
        return True

    # Check PyStow config file
    config_file = Path.home() / ".config" / "zenodo.ini"
    if config_file.exists():
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file)
            if 'zenodo' in config and config['zenodo'].get('api_token'):
                return True
        except Exception:
            pass

    return False


def download_zenodo_folder_client(zenodo_id: str, folder_dir: Path) -> bool:
    """
    Download a Zenodo record using zenodo_client library (primary method).

    Note: zenodo_client requires an API token. For public repositories without
    a token, this function will return False and fallback methods will be used.

    Inputs :
        - zenodo_id : str
            Zenodo record ID to download
        - folder_dir : Path
            Local directory where the Zenodo record will be downloaded

    Returns :
        - success : bool
            Did the download complete successfully?
    """
    # Skip zenodo_client if no API token is available (required by library)
    # This avoids wasting time on operations that will fail
    if not _has_zenodo_token():
        log.debug('zenodo_client requires API token - skipping (will use fallback methods)')
        return False

    try:
        from zenodo_client import Zenodo
    except ImportError:
        log.debug('zenodo_client library not available - will use fallback methods')
        return False
    except Exception as e:
        log.debug(f'Error importing zenodo_client: {e} - will use fallback methods')
        return False

    _zenodo_cooldown()

    try:
        # Try to create Zenodo client (now we know token should be available)
        try:
            zenodo = Zenodo()
        except Exception as e:
            # Token check may have been wrong, or token is invalid
            log.debug(f'zenodo_client configuration issue: {e}')
            log.debug('  Will use fallback download methods')
            return False

        # Get the record metadata
        _zenodo_cooldown()
        try:
            # get_latest_record() returns a string (record ID), not a dict
            record_id = zenodo.get_latest_record(zenodo_id)
            # Get the actual record object
            record = zenodo.get_record(record_id)
            # Access the JSON data with files
            record_data = record.json()
        except Exception as e:
            log.debug(f'Error getting record from zenodo_client: {e}')
            log.debug('  Will use fallback download methods')
            return False

        if not record_data or 'files' not in record_data:
            log.warning(f'No files found in Zenodo record {zenodo_id}')
            return False

        folder_dir.mkdir(parents=True, exist_ok=True)
        downloaded_count = 0

        # Download each file in the record
        for file_info in record_data['files']:
            file_key = file_info.get('key', '')
            if not file_key:
                continue

            # Determine output path (preserve directory structure)
            if '/' in file_key:
                file_path = folder_dir / file_key
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path = folder_dir / file_key

            # Always download files when this function is called
            # (check_needs_update() already determined update is needed)
            # Remove any existing file to ensure fresh download
            if file_path.exists():
                safe_rm(file_path)

            try:
                # Download the file using zenodo_client
                _zenodo_cooldown()
                downloaded_path = zenodo.download_latest(zenodo_id, file_key)

                # zenodo_client downloads to pystow-managed location
                # Handle both string and Path return types, or None
                actual_downloaded_path = None
                if downloaded_path:
                    if isinstance(downloaded_path, Path):
                        actual_downloaded_path = downloaded_path
                    else:
                        actual_downloaded_path = Path(downloaded_path)

                # If no path returned or file doesn't exist, try pystow's default location
                if not actual_downloaded_path or not actual_downloaded_path.exists():
                    try:
                        import pystow
                        module = pystow.module("zenodo", zenodo_id)
                        # Try common pystow path patterns
                        possible_paths = [
                            module.join(zenodo_id, file_key),
                            module.join(file_key),
                            Path.home() / ".data" / "zenodo" / zenodo_id / file_key,
                        ]
                        for pp in possible_paths:
                            pp_path = Path(pp) if not isinstance(pp, Path) else pp
                            if pp_path.exists():
                                actual_downloaded_path = pp_path
                                break
                    except ImportError:
                        pass  # pystow not available, skip

                if actual_downloaded_path and actual_downloaded_path.exists():
                    # Move to target location if needed
                    if actual_downloaded_path.resolve() != file_path.resolve():
                        import shutil
                        # Ensure target directory exists
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(actual_downloaded_path), str(file_path))

                    # Verify file was moved/copied successfully
                    if file_path.exists() and file_path.stat().st_size > 0:
                        downloaded_count += 1
                        file_size = file_path.stat().st_size
                        log.debug(f'  Downloaded: {file_key} ({file_size} bytes)')
                    else:
                        log.warning(f'  File {file_key} moved but verification failed')
                else:
                    log.warning(
                        f'  Failed to download: {file_key} '
                        f'(file not found after download)'
                    )
                    log.debug(f'    downloaded_path: {downloaded_path}')
                    log.debug(f'    actual_downloaded_path: {actual_downloaded_path}')
            except Exception as e:
                log.warning(f'  Error downloading {file_key}: {e}')
                import traceback
                log.debug(f'  Traceback: {traceback.format_exc()}')
                continue

        # Verify download succeeded
        if folder_dir.exists():
            files = list(folder_dir.rglob('*'))
            file_count = sum(1 for f in files if f.is_file())
            if file_count > 0:
                log.info(
                    f'Successfully downloaded Zenodo record {zenodo_id} '
                    f'using zenodo_client ({downloaded_count} files downloaded, '
                    f'{file_count} total files)'
                )
                # If downloaded_count doesn't match file_count, some files may have been skipped or already existed
                if downloaded_count < file_count:
                    skipped = file_count - downloaded_count
                    log.debug(f'  Note: {skipped} files were already present or skipped')
                return True
            else:
                log.warning(
                    f'Zenodo download completed but folder is empty '
                    f'(ID {zenodo_id})'
                )
                return False
        else:
            log.warning(
                f'Zenodo download completed but folder does not exist '
                f'(ID {zenodo_id})'
            )
            return False

    except Exception as e:
        log.debug(f'zenodo_client download failed: {e}')
        import traceback
        log.debug(f'Traceback: {traceback.format_exc()}')
        return False


def download_zenodo_file_client(zenodo_id: str, folder_dir: Path, zenodo_path: str) -> bool:
    """
    Download a single file from a Zenodo record using zenodo_client library (primary method).

    Note: zenodo_client requires an API token. For public repositories without
    a token, this function will return False and fallback methods will be used.

    Inputs
    ------
    zenodo_id : str
        Zenodo record ID to download from.
    folder_dir : Path
        Local directory where the file will be downloaded.
    zenodo_path : str
        Path/filename inside the Zenodo record, e.g. "subdir/file.txt" or "file.txt".

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    # Skip zenodo_client if no API token is available (required by library)
    # This avoids wasting time on operations that will fail
    if not _has_zenodo_token():
        log.debug('zenodo_client requires API token - skipping (will use fallback methods)')
        return False

    try:
        from zenodo_client import Zenodo
    except ImportError:
        log.debug('zenodo_client library not available - will use fallback methods')
        return False
    except Exception as e:
        log.debug(f'Error importing zenodo_client: {e} - will use fallback methods')
        return False

    _zenodo_cooldown()

    try:
        # Try to create Zenodo client (now we know token should be available)
        try:
            zenodo = Zenodo()
        except Exception as e:
            # Token check may have been wrong, or token is invalid
            log.debug(f'zenodo_client configuration issue: {e}')
            log.debug('  Will use fallback download methods')
            return False

        # Make sure local base directory exists
        folder_dir.mkdir(parents=True, exist_ok=True)

        # Local target path (preserves any subdirectory structure in zenodo_path)
        target_path = folder_dir / zenodo_path

        # Always download when this function is called
        # (validation already determined update is needed)
        # Remove any existing file to ensure fresh download
        if target_path.exists():
            safe_rm(target_path)

        # Make sure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        _zenodo_cooldown()
        downloaded_path = zenodo.download_latest(zenodo_id, zenodo_path)

        # zenodo_client downloads to pystow-managed location
        # Handle both string and Path return types, or None
        actual_downloaded_path = None
        if downloaded_path:
            if isinstance(downloaded_path, Path):
                actual_downloaded_path = downloaded_path
            else:
                actual_downloaded_path = Path(downloaded_path)

        # If no path returned or file doesn't exist, try pystow's default location
        if not actual_downloaded_path or not actual_downloaded_path.exists():
            try:
                import pystow
                module = pystow.module("zenodo", zenodo_id)
                # Try common pystow path patterns
                possible_paths = [
                    module.join(zenodo_id, zenodo_path),
                    module.join(zenodo_path),
                    Path.home() / ".data" / "zenodo" / zenodo_id / zenodo_path,
                ]
                for pp in possible_paths:
                    pp_path = Path(pp) if not isinstance(pp, Path) else pp
                    if pp_path.exists():
                        actual_downloaded_path = pp_path
                        break
            except ImportError:
                pass  # pystow not available, skip

        if actual_downloaded_path and actual_downloaded_path.exists():
            # Move to target location if needed
            if actual_downloaded_path.resolve() != target_path.resolve():
                import shutil
                shutil.move(str(actual_downloaded_path), str(target_path))

            if target_path.exists() and target_path.stat().st_size > 0:
                log.debug(f'Successfully downloaded file {zenodo_path} using zenodo_client')
                return True
            else:
                log.warning(f'Downloaded file is empty: {zenodo_path}')
                return False
        else:
            log.warning(
                f'Failed to download file: {zenodo_path} '
                f'(file not found after download)'
            )
            return False

    except Exception as e:
        log.debug(f'zenodo_client file download failed: {e}')
        return False


def download_zenodo_folder(zenodo_id: str, folder_dir: Path) -> bool:
    """
    Download a specific Zenodo record into specified folder.

    Uses zenodo_client library as primary method, with fallbacks to zenodo_get
    command-line tool and web-based download.

    Inputs :
        - zenodo_id : str
            Zenodo record ID to download
        - folder_dir : Path
            Local directory where the Zenodo record will be downloaded

    Returns :
        - zenodo_ok : bool
            Did the download/request complete successfully?
    """
    # Primary method: Try zenodo_client library
    log.debug(f'Attempting to download Zenodo record {zenodo_id} using zenodo_client...')
    if download_zenodo_folder_client(zenodo_id, folder_dir):
        return True

    # Fallback 1: Try zenodo_get command-line tool
    log.debug('zenodo_client failed, trying zenodo_get command-line tool...')
    try:
        sp.run(['zenodo_get', '--version'], capture_output=True, check=True, timeout=10)
    except (FileNotFoundError, sp.TimeoutExpired, sp.CalledProcessError) as e:
        log.debug(f'zenodo_get command not available: {e}')
    else:
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
                    # Add timeout to subprocess (MAX_DLTIME + buffer for overhead)
                    proc = sp.run(
                        [
                            'zenodo_get', '-o', str(folder_dir),
                            '-t', f'{MAX_DLTIME:.1f}', zenodo_id
                        ],
                        stdout=hdl,
                        stderr=sp.STDOUT,  # Combine stderr into stdout for better logging
                        timeout=MAX_DLTIME + 30,  # Add buffer for subprocess overhead
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
                                f'Successfully downloaded Zenodo record '
                                f'{zenodo_id} using zenodo_get'
                            )
                            return True
                        else:
                            log.warning(
                                f'Zenodo download completed but folder is empty '
                                f'(ID {zenodo_id})'
                            )
                    else:
                        log.warning(
                            f'Zenodo download completed but folder does not exist '
                            f'(ID {zenodo_id})'
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
                    attempt_num = attempt + 1
                    log.warning(
                        f'Failed to get data from Zenodo '
                        f'(ID {zenodo_id}, attempt {attempt_num}/{MAX_ATTEMPTS}): '
                        f'exit code {proc.returncode}. Error: {error_msg[:200]}'
                    )

            except sp.TimeoutExpired:
                log.warning(
                    f'zenodo_get timed out after {MAX_DLTIME + 30:.1f}s (ID {zenodo_id}, '
                    f'attempt {attempt + 1}/{MAX_ATTEMPTS})'
                )
            except Exception as e:
                log.warning(
                    f'Unexpected error during Zenodo download '
                    f'(ID {zenodo_id}): {e}'
                )

            # Exponential backoff: wait longer between retries
            if attempt < MAX_ATTEMPTS - 1:
                wait_time = RETRY_WAIT * (2 ** attempt)  # Exponential backoff
                log.debug(f'Waiting {wait_time:.1f}s before retry...')
                sleep(wait_time)

    # Fallback 2: Try web-based download
    log.debug('zenodo_get failed, trying web-based download...')
    if download_zenodo_folder_web(zenodo_id, folder_dir):
        return True

    # All methods failed
    log.error(
        f'Could not obtain data for Zenodo record {zenodo_id} '
        f'after trying all methods'
    )
    return False


def download_zenodo_folder_web(zenodo_id: str, folder_dir: Path) -> bool:
    """
    Download a Zenodo record using web interface (fallback method).

    Uses direct download URLs from Zenodo web interface via curl. This is a
    fallback method when zenodo_client and zenodo_get are unavailable or fail.
    Downloads all files from the Zenodo record and preserves directory structure.

    Inputs :
        - zenodo_id : str
            Zenodo record ID to download
        - folder_dir : Path
            Local directory where the Zenodo record will be downloaded

    Returns :
        - success : bool
            Did the download complete successfully?
    """
    log.debug(f'  Downloading Zenodo {zenodo_id} via web interface...')

    # Get the actual file URLs from Zenodo API
    import json
    import urllib.request

    _zenodo_cooldown()

    try:
        api_url = f"https://zenodo.org/api/records/{zenodo_id}"
        with urllib.request.urlopen(api_url, timeout=30) as response:
            data = json.loads(response.read())

        files = data.get('files', [])
        if not files:
            log.warning("  ✗ No files found in Zenodo record")
            return False

    except Exception as e:
        log.warning(f"  ⚠ Could not get metadata: {e}")
        return False

    # Download all files from the record
    folder_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download all files
        downloaded_files = 0
        for f in files:
            file_url = f['links']['self']
            file_key = f.get('key', 'unknown')
            file_size = f.get('size', 0) / 1024 / 1024  # MB

            # Determine output path
            if '/' in file_key:
                # Preserve directory structure
                file_path = folder_dir / file_key
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path = folder_dir / file_key

            # Skip if already exists
            if file_path.exists() and file_path.stat().st_size > 0:
                log.debug(f"  ⏭  Skipping (exists): {file_key}")
                continue

            log.debug(f"  Downloading: {file_key} ({file_size:.2f} MB)...")

            # Apply cooldown before each download
            _zenodo_cooldown()

            result = sp.run(
                ['curl', '--fail', '-L', '-o', str(file_path), file_url],
                timeout=600,
                capture_output=True,
                text=True
            )

            if (result.returncode == 0 and file_path.exists() and
                    file_path.stat().st_size > 0):
                downloaded_files += 1
                size_mb = file_path.stat().st_size / 1024 / 1024
                log.debug(f"    ✓ Downloaded {size_mb:.2f} MB")
            else:
                log.warning(f"    ✗ Failed: {result.stderr[:100] if result.stderr else 'Unknown error'}")

        # Count total files
        all_files = list(folder_dir.rglob('*'))
        file_count = sum(1 for f in all_files if f.is_file())
        if file_count > 0:
            log.info(f"  ✓ Total: {file_count} files in directory (web download)")
            return True
        else:
            log.warning("  ✗ No files downloaded")
            return False

    except Exception as e:
        log.warning(f"  ✗ Error: {e}")
        return False

def get_zenodo_file(zenodo_id: str, folder_dir: Path, zenodo_path: str) -> bool:
    """
    Download a single file from a Zenodo record into the specified local folder.

    Uses zenodo_client library as primary method, with fallbacks to zenodo_get
    command-line tool and web-based download.

    Inputs
    ------
    zenodo_id : str
        Zenodo record ID to download from.
    folder_dir : Path
        Local directory where the file will be downloaded.
    zenodo_path : str
        Path/filename inside the Zenodo record, e.g. "subdir/file.txt" or "file.txt".

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    # Primary method: Try zenodo_client library
    log.debug(f"Attempting to download file '{zenodo_path}' from Zenodo {zenodo_id} using zenodo_client...")
    if download_zenodo_file_client(zenodo_id, folder_dir, zenodo_path):
        return True

    # Fallback 1: Try zenodo_get command-line tool
    log.debug("zenodo_client failed, trying zenodo_get command-line tool...")
    # Where to log zenodo_get output
    out = os.path.join(GetFWLData(), "zenodo_get_file.log")
    log.debug(f"    zenodo_get (file {zenodo_path}), logging to {out}")

    # Make sure local base directory exists
    folder_dir.mkdir(parents=True, exist_ok=True)

    # Local target path (preserves any subdirectory structure in zenodo_path)
    target_path = folder_dir / zenodo_path

    for i in range(MAX_ATTEMPTS):

        # Remove any existing copy of the file
        safe_rm(target_path)

        # Make sure parent directory exists (in case zenodo_path contains subdirs)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Try making request: use -g to select just this path/pattern
        try:
            with open(out, 'w') as hdl:
                proc = sp.run(
                    ["zenodo_get", zenodo_id, "-o", str(folder_dir), "-g", zenodo_path],
                    stdout=hdl,
                    stderr=hdl,
                    timeout=MAX_DLTIME + 30,
                )

            # Worked ok?
            if (proc.returncode == 0 and target_path.exists() and
                    target_path.stat().st_size > 0):
                log.debug(f"Successfully downloaded file '{zenodo_path}' using zenodo_get")
                return True
        except FileNotFoundError as e:
            log.debug(f"zenodo_get not available: {e}")
            break  # Don't retry if command doesn't exist
        except sp.TimeoutExpired as e:
            log.warning(
                f"zenodo_get timed out after {MAX_DLTIME + 30:.1f}s (ID {zenodo_id}, "
                f"attempt {i+1}/{MAX_ATTEMPTS})"
            )
            # Continue to retry with exponential backoff
        except Exception as e:
            log.debug(f"zenodo_get error: {e}")

        # Exponential backoff: wait longer between retries
        if i < MAX_ATTEMPTS - 1:
            wait_time = RETRY_WAIT * (2 ** i)  # Exponential backoff
            log.debug(f"Waiting {wait_time:.1f}s before retry...")
            sleep(wait_time)

    # Fallback 2: Try web-based download
    log.debug("zenodo_get failed, trying web-based download...")
    if get_zenodo_file_web(zenodo_id, folder_dir, zenodo_path):
        return True

    # All methods failed
    log.error(
        f"Could not obtain file '{zenodo_path}' from Zenodo record {zenodo_id} after trying all methods"
    )
    return False


def get_zenodo_file_web(zenodo_id: str, folder_dir: Path, zenodo_path: str) -> bool:
    """
    Download a single file from a Zenodo record using web interface (fallback method).

    Uses direct download URL from Zenodo web interface via curl. This is a
    fallback method when zenodo_client and zenodo_get are unavailable or fail.
    Preserves any subdirectory structure in the zenodo_path.

    Inputs
    ------
    zenodo_id : str
        Zenodo record ID to download from.
    folder_dir : Path
        Local directory where the file will be downloaded.
    zenodo_path : str
        Path/filename inside the Zenodo record, e.g. "subdir/file.txt" or "file.txt".

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    import json
    import urllib.request

    _zenodo_cooldown()

    try:
        # Get record metadata
        api_url = f"https://zenodo.org/api/records/{zenodo_id}"
        with urllib.request.urlopen(api_url, timeout=30) as response:
            data = json.loads(response.read())

        files = data.get('files', [])
        if not files:
            log.warning(f"  ✗ No files found in Zenodo record {zenodo_id}")
            return False

        # Find the specific file
        target_file = None
        for f in files:
            if f.get('key', '') == zenodo_path:
                target_file = f
                break

        if not target_file:
            log.warning(f"  ✗ File '{zenodo_path}' not found in Zenodo record {zenodo_id}")
            return False

        # Make sure local base directory exists
        folder_dir.mkdir(parents=True, exist_ok=True)
        target_path = folder_dir / zenodo_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if target_path.exists() and target_path.stat().st_size > 0:
            log.debug(f"  ⏭  File already exists: {zenodo_path}")
            return True

        # Download the file
        file_url = target_file['links']['self']
        file_size = target_file.get('size', 0) / 1024 / 1024  # MB
        log.debug(f"  Downloading: {zenodo_path} ({file_size:.2f} MB)...")

        _zenodo_cooldown()

        result = sp.run(
            ['curl', '--fail', '-L', '-o', str(target_path), file_url],
            timeout=600,
            capture_output=True,
            text=True
        )

        if (result.returncode == 0 and target_path.exists() and
                target_path.stat().st_size > 0):
            size_mb = target_path.stat().st_size / 1024 / 1024
            log.debug(f"  ✓ Downloaded {size_mb:.2f} MB (web download)")
            return True
        else:
            log.warning(f"  ✗ Failed: {result.stderr[:100] if result.stderr else 'Unknown error'}")
            return False

    except Exception as e:
        log.warning(f"  ✗ Error downloading file via web: {e}")
        return False


def md5(_fname):
    """Return the md5 hash of a file."""

    # https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(_fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
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
                    cwd=folder_dir,
                    timeout=60,  # Reasonable timeout for checksum request
                    check=False,
                )

            # process exited fine and file exists?
            zenodo_ok = (proc.returncode == 0) and os.path.isfile(md5sums_path)

            # try again?
            if zenodo_ok:
                break
            else:
                log.warning(f'Failed to get checksum from Zenodo (ID {zenodo_id}, attempt {attempt + 1}/{MAX_ATTEMPTS})')
        except sp.TimeoutExpired:
            log.warning(f'zenodo_get validation timed out (ID {zenodo_id}, attempt {attempt + 1}/{MAX_ATTEMPTS})')
        except Exception as e:
            log.warning(f'Unexpected error during Zenodo validation (ID {zenodo_id}): {e}')

        # Exponential backoff
        if attempt < MAX_ATTEMPTS - 1:
            wait_time = RETRY_WAIT * (2 ** attempt)
            sleep(wait_time)

    # Return status indicating that file/folder is invalid, if failed
    if not zenodo_ok:
        log.warning(f'Could not obtain checksum for Zenodo record {zenodo_id} - skipping validation')
        # Gracefully degrade: if folder exists and has files, assume valid
        if folder_dir.exists() and any(f.is_file() for f in folder_dir.rglob('*')):
            return True
        return False

    # Read hashes file
    with open(md5sums_path,'r') as hdl:
        md5sums = hdl.readlines()

    # Check each item in the record...
    for line in md5sums:
        try:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip malformed lines
            sum_expect, name = parts[0], parts[1]
        except Exception as e:
            log.warning(f'Error parsing md5sums line: {line.strip()}: {e}')
            continue

        file = os.path.join(folder_dir, name)

        # exit here if file does not exist
        if not os.path.exists(file):
            log.warning(f'Detected missing file {name} (Zenodo record {zenodo_id})')
            return False

        # don't check the hashes of very large files, because it's slow
        if os.path.getsize(file) > hash_maxfilesize:
            continue  # Skip hash check for large files, but continue validation

        # check the actual hash of the file on disk, compare to expected
        sum_actual = md5(file).strip()
        if sum_actual != sum_expect:
            log.warning(f'Detected invalid file {name} (Zenodo record {zenodo_id})')
            log.warning(f'    expected hash {sum_expect}, got {sum_actual}')
            return False

    return True


# Unified mapping of data sources to Zenodo and OSF identifiers
# Each entry maps a folder name to its Zenodo record ID, OSF storage ID,
# and OSF project ID. This enables automatic fallback from Zenodo to OSF
# when downloads fail.
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
    'Melting_curves/Monteux+600': {'zenodo_id': '15728091', 'osf_id': 'phsxf', 'osf_project': 'phsxf'},
    'Melting_curves/Monteux-600': {'zenodo_id': '15728138', 'osf_id': 'phsxf', 'osf_project': 'phsxf'},
    'Melting_curves/Wolf_Bower+2018': {'zenodo_id': '15728072', 'osf_id': 'phsxf', 'osf_project': 'phsxf'},
    # Surface albedos (OSF project: 2gcd9)
    'Hammond24': {'zenodo_id': '15880455', 'osf_id': '2gcd9', 'osf_project': '2gcd9'},
    # Stellar spectra (OSF project: 8r2sw)
    'Named': {'zenodo_id': '15721440', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Stellar spectra - PHOENIX (OSF project: 8r2sw)
    'PHOENIX': {'zenodo_id': '17674612', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Stellar spectra - MUSCLES (OSF project: 8r2sw)
    'MUSCLES': {'zenodo_id': '17802209', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Stellar spectra - Solar (OSF project: 8r2sw)
    'Solar': {'zenodo_id': '17981836', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
    # Exoplanet data (OSF project: fzwr4)
    'Exoplanets': {'zenodo_id': '15727878', 'osf_id': 'fzwr4', 'osf_project': 'fzwr4'},
    # Mass-radius data (OSF project: xge8t)
    'Zeng2019': {'zenodo_id': '15727899', 'osf_id': 'xge8t', 'osf_project': 'xge8t'},
    # Population data (OSF project: dpkjb)
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
        if info.get('osf_project') == osf_id and info.get('zenodo_id')
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

def download_OSF_folder(*, storage, folders: list[str], data_dir: Path, force: bool = False):
    """
    Download a specific folder in the OSF repository

    Inputs :
        - storage : OSF storage name
        - folders : folder names to download
        - data_dir : local repository where data are saved
        - force : bool
            If True, re-download files even if they already exist
    """
    downloaded_files = 0
    total_size = 0

    for file in storage.files:
        for folder in folders:
            if not file.path[1:].startswith(folder):
                continue
            parts = file.path.split('/')[1:]
            target = Path(data_dir, *parts)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file already exists and has content (unless force=True)
            if not force and target.exists() and target.stat().st_size > 0:
                log.debug(f'Skipping existing file: {file.path}')
                continue

            # If force=True, remove existing file to ensure fresh download
            if force and target.exists():
                safe_rm(target)

            try:
                log.info(f'Downloading {file.path} ({file.size / 1024 / 1024:.1f} MB)...')
                with open(target, 'wb') as f:
                    file.write_to(f)
                downloaded_files += 1
                total_size += file.size
            except Exception as e:
                log.warning(f'Failed to download {file.path}: {e}')
                # Clean up partial file
                if target.exists():
                    try:
                        target.unlink()
                    except Exception:
                        pass
                continue
            break

    if downloaded_files > 0:
        log.info(f'Downloaded {downloaded_files} files ({total_size / 1024 / 1024:.1f} MB) from OSF')

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

    log.debug(f"Checking whether {dir} needs updating (record {zenodo})")

    # Trivial case where folder is missing
    if not os.path.isdir(dir):
        return True

    # Folder exists but cannot check hashes, so exit here
    if not zenodo:
        return False # don't update

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
    zenodo_path: str | None = None,
) -> bool:
    """
    Generic download function with automatic source mapping.

    This function can automatically look up OSF and Zenodo IDs from the unified
    DATA_SOURCE_MAP if they are not provided. If both are provided, uses them
    directly. Downloads are attempted from Zenodo first, with automatic
    fallback to OSF if Zenodo fails.

    The download process:
    1. Looks up source IDs from DATA_SOURCE_MAP if not provided
    2. Checks if data already exists and is valid (unless force=True)
    3. Attempts download from Zenodo (with multiple fallback methods)
    4. Falls back to OSF if Zenodo fails
    5. Validates downloaded files

    Parameters
    ----------
    folder: str
        Folder name to download (must match key in DATA_SOURCE_MAP if IDs not
        provided)
    target: str
        Name of target directory within FWL_DATA
    osf_id: str | None
        OSF project id (optional, will be looked up from mapping if not
        provided)
    zenodo_id: str | None
        Zenodo record id (optional, will be looked up from mapping if not
        provided)
    desc: str
        Description for logging purposes
    force: bool
        Force a re-download even if valid data already exists
    zenodo_path: str | None
        Optional path to a specific file within the Zenodo record.
        If provided, downloads only this file instead of the entire folder.
        Example: "subdir/file.txt" or "file.txt"

    Returns
    -------
    bool
        True if the file/folder was downloaded successfully, False otherwise
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

    # Handle single file download vs folder download
    if zenodo_path is not None:
        # Single file download mode
        target_file_path = folder_dir / zenodo_path

        # Check if file needs updating
        file_invalid = force or not (target_file_path.exists() and target_file_path.stat().st_size > 0)

        if file_invalid:
            log.info(f'Downloading {desc} ({zenodo_path}) to {data_dir}')
            success = False

            # Try Zenodo in the first instance
            if zenodo_id is not None:
                try:
                    # Download the specific file
                    if get_zenodo_file(zenodo_id=zenodo_id, folder_dir=folder_dir, zenodo_path=zenodo_path):
                        # Verify file exists and has content
                        if target_file_path.exists() and target_file_path.stat().st_size > 0:
                            success = True
                        else:
                            log.warning(f'File {zenodo_path} downloaded but verification failed')
                except RuntimeError as e:
                    log.warning(f'    Zenodo download failed: {e}')
            else:
                log.debug('    No Zenodo ID provided, skipping Zenodo download')

            if success:
                return True

            # If Zenodo fails or not available, try OSF
            if osf_id:
                try:
                    log.info(f'Attempting OSF fallback download for file {zenodo_path} (project {osf_id})...')
                    storage = get_osf(osf_id)
                    # For single file, we need to download from OSF
                    # Find the file in OSF storage
                    osf_file_path = f'{folder}/{zenodo_path}'
                    found = False
                    for file in storage.files:
                        if file.path[1:] == osf_file_path or file.path[1:].endswith(zenodo_path):
                            target_file_path.parent.mkdir(parents=True, exist_ok=True)
                            if force and target_file_path.exists():
                                safe_rm(target_file_path)
                            with open(target_file_path, 'wb') as f:
                                file.write_to(f)
                            found = True
                            log.info(f'Successfully downloaded {desc} from OSF (project {osf_id})')
                            success = True
                            break
                    if not found:
                        log.warning(f'File {zenodo_path} not found in OSF project {osf_id}')
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

            log.error(f'    Failed to download {desc} ({zenodo_path}) from IDs: Zenodo {zenodo_id}, OSF {osf_id}')
            return False
        else:
            log.debug(f'    {desc} ({zenodo_path}) already exists')
            return True
    else:
        # Folder download mode (original behavior)
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
                    download_OSF_folder(storage=storage, folders=[folder], data_dir=data_dir, force=force)

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
    source_info = get_data_source_info('Hammond24')
    return download(
        folder='Hammond24',
        target='surface_albedos',
        osf_id=(source_info or {}).get('osf_project', '2gcd9'),
        zenodo_id=(source_info or {}).get('zenodo_id', '15880455'),
        desc='surface reflectance data',
    )

def download_spectral_file(name:str, bands:str):
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
        raise Exception("Must provide name of spectral file")
    if not isinstance(bands, str) or (len(bands) < 1):
        raise Exception("Must provide number of bands in spectral file")

    folder_name = f'{name}/{bands}'
    source_info = get_data_source_info(folder_name)
    return download(
        folder=folder_name,
        target='spectral_files',
        osf_id=(source_info or {}).get('osf_project', 'vehxg'),
        zenodo_id=(source_info or {}).get('zenodo_id') or get_zenodo_record(folder_name),
        desc=f'{name}{bands} spectral file',
    )


def download_interior_lookuptables(clean=False):
    """
    Download basic interior lookup tables
    """
    log.debug("Download basic interior lookup tables")

    data_dir = GetFWLData() / "interior_lookup_tables"
    data_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for dir in ARAGOG_BASIC:
        folder_dir = data_dir / dir
        if clean:
            safe_rm(folder_dir.as_posix())
        source_info = get_data_source_info(dir)
        if not download(
            folder=dir,
            target=data_dir,
            osf_id=(source_info or {}).get('osf_project', 'phsxf'),
            zenodo_id=(source_info or {}).get('zenodo_id') or get_zenodo_record(dir),
            desc=f'Interior lookup tables: {dir}',
        ):
            success = False
    return success

def download_melting_curves(config:Config, clean=False):
    """
    Download melting curve data
    """
    log.debug("Download melting curve data")
    dir = "Melting_curves/" + config.interior.melting_dir

    data_dir = GetFWLData() / "interior_lookup_tables"
    data_dir.mkdir(parents=True, exist_ok=True)

    folder_dir = data_dir / dir
    if clean:
        safe_rm(folder_dir.as_posix())
    source_info = get_data_source_info(dir)
    return download(
        folder=dir,
        target=data_dir,
        osf_id=(source_info or {}).get('osf_project', 'phsxf'),
        zenodo_id=(source_info or {}).get('zenodo_id') or get_zenodo_record(dir),
        desc=f'Melting curve data: {dir}',
    )

def download_stellar_spectra():
    """
    Download stellar spectra
    """
    source_info = get_data_source_info('Named')
    return download(
        folder='Named',
        target='stellar_spectra',
        osf_id=(source_info or {}).get('osf_project', '8r2sw'),
        zenodo_id=(source_info or {}).get('zenodo_id', '15721440'),
        desc='stellar spectra',
    )

def download_solar_spectrum():
    """
    Download NREL solar spectrum
    """
    named_zenodo_id = "15721440"
    data_dir   = GetFWLData() / "stellar_spectra"
    folder_dir = data_dir / "solar"
    filename = "sun.txt"
    log.info(f"Downloading solar spectrum {filename}")

    return get_zenodo_file(
        zenodo_id=named_zenodo_id,
        folder_dir=folder_dir,
        zenodo_path=filename,
    )

def download_all_solar_spectra():
    """
    Download all solar spectra (nrel, VPL past, VPL present, VPL future)
    """
    source_info = get_data_source_info('Solar')
    return download(
        folder='Solar',
        target='stellar_spectra',
        osf_id=(source_info or {}).get('osf_project', '8r2sw'),
        zenodo_id=(source_info or {}).get('zenodo_id', '17981836'),
        desc='all solar spectra',
    )

def download_phoenix(alpha: float | int | str, FeH: float | int | str) -> bool:
    """
    Download and unpack a PHOENIX spectra ZIP like
    FeH+0.5_alpha+0.0_phoenixMedRes_R05000.zip
    into <FWL_DATA>/stellar_spectra/PHOENIX.

    NOTE: Assumes alpha and FeH are already mapped to nearest grid point.
    """

    feh_str   = phoenix_param(FeH,   kind="FeH")
    alpha_str = phoenix_param(alpha, kind="alpha")
    zip_name  = f"FeH{feh_str}_alpha{alpha_str}_phoenixMedRes_R05000.zip"

    data_dir = GetFWLData() / "stellar_spectra"
    folder_dir = data_dir / "PHOENIX" / f"FeH{feh_str}_alpha{alpha_str}"

    if folder_dir.exists() and any(folder_dir.iterdir()):
        log.info(f"PHOENIX spectra already present in {folder_dir}")
        return True

    folder_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading PHOENIX spectra {zip_name}")
    log.info("This may take a while.")

    source_info = get_data_source_info('PHOENIX')
    phoenix_zenodo_id = (source_info or {}).get('zenodo_id', '17674612')
    phoenix_osf_id = (source_info or {}).get('osf_project', '8r2sw')

    # first download readme using download() function for OSF fallback
    download(
        folder='PHOENIX',
        target='stellar_spectra',
        osf_id=phoenix_osf_id,
        zenodo_id=phoenix_zenodo_id,
        desc='PHOENIX readme',
        zenodo_path="_readme.md",
    )

    # then download zip using download() function for OSF fallback
    if not download(
        folder='PHOENIX',
        target='stellar_spectra',
        osf_id=phoenix_osf_id,
        zenodo_id=phoenix_zenodo_id,
        desc=f'PHOENIX spectra {zip_name}',
        zenodo_path=zip_name,
    ):
        log.error(
            f"Failed to download PHOENIX ZIP {zip_name} "
            f"from Zenodo {phoenix_zenodo_id} or OSF {phoenix_osf_id}"
        )
        return False

    # The file should be in PHOENIX/zip_name, but we need it in the subfolder
    zip_path_source = data_dir / "PHOENIX" / zip_name
    zip_path = folder_dir / zip_name
    if zip_path_source.exists() and not zip_path.exists():
        import shutil
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(zip_path_source), str(zip_path))

    zip_path = folder_dir / zip_name

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(folder_dir)
    except zipfile.BadZipFile as e:
        log.error(f"Downloaded PHOENIX ZIP is corrupted: {zip_path}")
        log.error(str(e))
        safe_rm(zip_path)
        return False
    else:
        safe_rm(zip_path)
        log.info(f"PHOENIX spectra unpacked to {folder_dir}")
        return True


def download_muscles(star_name: str) -> bool:
    star_filename = f"{star_name.strip().lower().replace(' ', '-').replace('gj-', 'gj')}.txt" # lowercase, and; "trappist 1" -> "trappist-1", but "gj 876" or "gj-876" -> "gj876"
    log.info(f"Downloading MUSCLES file {star_filename}")

    source_info = get_data_source_info('MUSCLES')
    muscles_zenodo_id = (source_info or {}).get('zenodo_id', '17802209')
    muscles_osf_id = (source_info or {}).get('osf_project', '8r2sw')

    # first download readme using download() function for OSF fallback
    download(
        folder='MUSCLES',
        target='stellar_spectra',
        osf_id=muscles_osf_id,
        zenodo_id=muscles_zenodo_id,
        desc='MUSCLES readme',
        zenodo_path="_readme.md",
    )

    # then download star file using download() function for OSF fallback
    return download(
        folder='MUSCLES',
        target='stellar_spectra',
        osf_id=muscles_osf_id,
        zenodo_id=muscles_zenodo_id,
        desc=f'MUSCLES spectrum {star_filename}',
        zenodo_path=star_filename,
    )

def download_exoplanet_data():
    """
    Download exoplanet data
    """
    source_info = get_data_source_info('Exoplanets')
    return download(
        folder='Exoplanets',
        target='planet_reference',
        osf_id=(source_info or {}).get('osf_project', 'fzwr4'),
        zenodo_id=(source_info or {}).get('zenodo_id', '15727878'),
        desc='exoplanet data',
    )


def download_massradius_data():
    """
    Download mass-radius data
    """
    source_info = get_data_source_info('Zeng2019')
    return download(
        folder='Zeng2019',
        target='mass_radius',
        osf_id=(source_info or {}).get('osf_project', 'xge8t'),
        zenodo_id=(source_info or {}).get('zenodo_id', '15727899'),
        desc='mass-radius data',
    )


def download_stellar_tracks(track:str):
    """
    Download stellar evolution tracks

    Uses the function built-into MORS.
    """
    from mors.data import DownloadEvolutionTracks
    log.debug("Get evolution tracks")
    DownloadEvolutionTracks(track)



def _get_sufficient(config:Config, clean:bool=False):
    # Star stuff
    if config.star.module == "mors":

        if config.star.mors.star_path is None:
            src = config.star.mors.spectrum_source

            if src == "solar":
                log.info("Spectrum source set to 'solar'.")
                solar_dir = GetFWLData() / "stellar_spectra" / "solar"
                sun_now   = solar_dir / "sun.txt"
                sun_06ga  = solar_dir / "Sun0.6Ga.txt"

                if (not sun_now.exists()) or (not sun_06ga.exists()):
                    download_all_solar_spectra()

            elif src is None:
                if config.star.mors.star_name.lower() != "sun":
                    muscles_ok =  download_muscles(config.star.mors.star_name)
                    if muscles_ok:
                        log.info("Spectrum source not set. MUSCLES spectrum found and downloaded.")
                        log.info("To always use MUSCLES, set star.mors.spectrum_source = 'muscles'.")
                    else:
                        log.info("Spectrum source not set. No MUSCLES spectrum found; downloading solar spectrum by default.")
                        log.info("To use a MUSCLES spectrum, check the available MUSCLES spectra at https://proteus-framework.org/PROTEUS/data.html#stellar-spectra and set star.mors.spectrum_source = 'muscles'.")
                        download_solar_spectrum()
                else:
                    download_solar_spectrum()

            elif src == "muscles":
                log.info("Spectrum source set to 'muscles'. Downloading MUSCLES spectrum.")
                muscles_ok = download_muscles(config.star.mors.star_name)
                if not muscles_ok:
                    log.error(f"Could not download MUSCLES spectrum for star {config.star.mors.star_name}.")
                    log.error("Check the MUSCLES available MUSCLES spectra at https://proteus-framework.org/PROTEUS/data.html#stellar-spectra to verify the star name.")
                    log.error("If no observed spectrum is available, consider using a PHOENIX synthetic spectrum by setting star.mors.spectrum_source = 'phoenix'.")

            elif src == "phoenix":
                log.info("Spectrum source set to 'phoenix'.")
                FeH   = config.star.mors.phoenix_FeH
                alpha = config.star.mors.phoenix_alpha

                # make sure what is downloaded matches nearest grid point
                Teff_override = getattr(config.star.mors, "phoenix_Teff", None) # Optional Teff override in the config -> relevant for mapping alpha fraction to grid
                grid   = phoenix_to_grid(FeH=FeH, alpha=alpha, Teff=Teff_override)
                FeH_g  = grid["FeH"]
                alpha_g = grid["alpha"]

                log.info(f"Downloading PHOENIX spectra with [Fe/H]={FeH_g:.2f}, [alpha/M]={alpha_g:.2f}.")
                log.info("Note that the requested values are mapped to the nearest grid point.")
                download_phoenix(alpha=alpha_g, FeH=FeH_g)

        if config.star.mors.tracks == 'spada':
            download_stellar_tracks("Spada")
        else:
            download_stellar_tracks("Baraffe")

    # Spectral files
    if config.atmos_clim.module in ('janus', 'agni'):
        # High-res file often used for post-processing
        download_spectral_file("Honeyside","4096")

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
    if config.interior.module == "aragog":
        download_interior_lookuptables(clean=clean)
        download_melting_curves(config, clean=clean)

def download_sufficient_data(config:Config, clean:bool=False):
    """
    Download the required data based on the current options
    """

    log.info("Getting physical and reference data")
    log.info("")

    if config.params.offline:
        # Don't try to get data
        log.warning("Running in offline mode. Will not check for reference data.")

    else:
        # Try to get data
        try:
            _get_sufficient(config, clean=clean)

        # Some issue. Usually due to lack of internet connection, but print the error
        #     anyway so that the user knows what happened.
        except OSError as e:
            log.warning("Problem when downloading/checking reference data")
            log.warning(str(e))

    log.info(" ")


def _none_dirs():
    from proteus.utils.helper import get_proteus_dir

    dirs = {"proteus":get_proteus_dir()}
    dirs["tools"] = os.path.join(dirs["proteus"],"tools")
    return dirs


def get_socrates(dirs=None):
    """
    Download and install SOCRATES
    """

    log.info("Setting up SOCRATES")

    # None dirs
    if dirs is None:
        dirs = _none_dirs()

    # Get path
    workpath = os.path.join(dirs["proteus"], "SOCRATES")
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        log.debug("    already set up")
        return

    # Download, configure, and build
    log.debug("Running get_socrates.sh")
    cmd = [os.path.join(dirs["tools"],"get_socrates.sh"), workpath]
    out = os.path.join(dirs["proteus"], "nogit_setup_socrates.log")
    log.debug("    logging to %s"%out)
    with open(out,'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    # Set environment
    os.environ["RAD_DIR"] = workpath
    log.debug("    done")


def get_petsc(dirs=None):
    """
    Download and install PETSc
    """

    log.info("Setting up PETSc")

    if dirs is None:
        dirs = _none_dirs()

    # Get path
    workpath = os.path.join(dirs["proteus"], "petsc")
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        # already downloaded
        log.debug("    already set up")
        return

    # Download, configure, and build
    log.debug("Running get_petsc.sh")
    cmd = [os.path.join(dirs["tools"],"get_petsc.sh"), workpath]
    out = os.path.join(dirs["proteus"], "nogit_setup_petsc.log")
    log.debug("    logging to %s"%out)
    with open(out,'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    log.debug("    done")


def get_spider(dirs=None):
    """
    Download and install SPIDER
    """

    if dirs is None:
        dirs = _none_dirs()

    # Need to install PETSc first
    get_petsc(dirs)

    log.info("Setting up SPIDER")

    # Get path
    workpath = os.path.join(dirs["proteus"], "SPIDER")
    workpath = os.path.abspath(workpath)
    if os.path.isdir(workpath):
        # already downloaded
        log.debug("    already set up")
        return

    # Download, configure, and build
    log.debug("Running get_spider.sh")
    cmd = [os.path.join(dirs["tools"],"get_spider.sh"), workpath]
    out = os.path.join(dirs["proteus"], "nogit_setup_spider.log")
    log.debug("    logging to %s"%out)
    with open(out,'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

    log.debug("    done")

def download_Seager_EOS():
    """
    Download EOS material properties from Seager et al. (2007)
    """
    source_info = get_data_source_info('EOS_Seager2007')
    return download(
        folder='EOS_Seager2007',
        target='EOS_material_properties',
        osf_id=(source_info or {}).get('osf_project', 'dpkjb'),
        zenodo_id=(source_info or {}).get('zenodo_id', '15727998'),
        desc='EOS Seager2007 material files',
    )

def get_Seager_EOS():
    """
    Build and return material properties dictionaries for Seager et al. (2007) EOS data.

    This function constructs dictionaries containing material properties for iron/silicate planets
    and water planets based on the Seager et al. (2007) equation of state (EOS) data. The data files
    are expected to be located in the FWL_DATA/EOS_material_properties/EOS_Seager2007 folder.

    Returns:
        tuple: A tuple containing two dictionaries:
            - material_properties_iron_silicate_planets: Material properties for iron/silicate planets.
            - material_properties_water_planets: Material properties for water planets.
    """
    # Define the EOS folder path
    eos_folder = FWL_DATA_DIR / "EOS_material_properties" / "EOS_Seager2007"

    # Download the EOS material properties if not already present
    if not eos_folder.exists():
        log.debug("Get EOS material properties from Seager et al. (2007)")
        download_Seager_EOS()

    # Build the material_properties_iron_silicate_planets dictionary for iron/silicate planets according to Seager et al. (2007)
    material_properties_iron_silicate_planets = {
        "mantle": {
            # Mantle properties based on bridgmanite
            "rho0": 4100,  # From Table 1 of Seager et al. (2007) for bridgmanite
            "eos_file": eos_folder / "eos_seager07_silicate.txt"  # Path to silicate mantle file
        },
        "core": {
            # For liquid iron alloy outer core
            "rho0": 8300,  # From Table 1 of Seager et al. (2007) for the epsilon phase of iron of Fe
            "eos_file": eos_folder / "eos_seager07_iron.txt"  # Path to iron core file
        }
    }
    # Build the material_properties_water_planets dictionary for water planets according to Seager et al. (2007)
    material_properties_water_planets = {
        "core": {
            # For liquid iron alloy outer core
            "rho0": 8300,  # From Table 1 of Seager et al. (2007) for the epsilon phase of iron of Fe in kg/m^3
            "eos_file": eos_folder / "eos_seager07_iron.txt"  # Name of the file with tabulated EOS data
        },
        "bridgmanite_shell": {
                # Inner mantle properties based on bridgmanite
                "rho0": 4100,  # From Table 1 of Seager et al. (2007) for bridgmanite in kg/m^3
                "eos_file": eos_folder / "eos_seager07_silicate.txt"  # Name of the file with tabulated EOS data
        },
        "water_ice_layer": {
            # Outer water ice layer in ice VII phase
                "rho0": 1460,  # From Table 1 of Seager et al. (2007) for H2O in ice VII phase in kg/m^3
                "eos_file": eos_folder / "eos_seager07_water.txt"  # Name of the file with tabulated EOS data
        }
    }
    return material_properties_iron_silicate_planets, material_properties_water_planets
