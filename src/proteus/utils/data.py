from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess as sp
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

ARAGOG_BASIC = (
    "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa",
    "Melting_curves/Wolf_Bower+2018",
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
                # Add timeout to subprocess (MAX_DLTIME + buffer for overhead)
                proc = sp.run(
                    ['zenodo_get', '-o', str(folder_dir), '-t', f'{MAX_DLTIME:.1f}', zenodo_id],
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
                        log.info(f'Successfully downloaded Zenodo record {zenodo_id}')
                        return True
                    else:
                        log.warning(f'Zenodo download completed but folder is empty (ID {zenodo_id})')
                else:
                    log.warning(f'Zenodo download completed but folder does not exist (ID {zenodo_id})')
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
                    f'exit code {proc.returncode}. Error: {error_msg[:200]}'
                )

        except sp.TimeoutExpired:
            log.warning(
                f'zenodo_get timed out after {MAX_DLTIME + 30:.1f}s (ID {zenodo_id}, '
                f'attempt {attempt + 1}/{MAX_ATTEMPTS})'
            )
        except Exception as e:
            log.warning(f'Unexpected error during Zenodo download (ID {zenodo_id}): {e}')

        # Exponential backoff: wait longer between retries
        if attempt < MAX_ATTEMPTS - 1:
            wait_time = RETRY_WAIT * (2 ** attempt)  # Exponential backoff
            log.debug(f'Waiting {wait_time:.1f}s before retry...')
            sleep(wait_time)

    # Return status indicating that file/folder is invalid, if failed
    log.error(f'Could not obtain data for Zenodo record {zenodo_id} after {MAX_ATTEMPTS} attempts')
    return False

def get_zenodo_file(zenodo_id: str, folder_dir: Path, zenodo_path: str) -> bool:
    """
    Download a single file from a Zenodo record into the specified local folder.

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
        with open(out, 'w') as hdl:
            proc = sp.run(
                ["zenodo_get", zenodo_id, "-o", str(folder_dir), "-g", zenodo_path],
                stdout=hdl,
                stderr=hdl,
            )

        # Worked ok?
        if (proc.returncode == 0) and target_path.exists():
            return True

        log.warning(
            f"Failed to get file '{zenodo_path}' from Zenodo (ID {zenodo_id}), "
            f"attempt {i+1}/{MAX_ATTEMPTS}"
        )
        sleep(RETRY_WAIT)

    # Return status indicating failure
    log.error(
        f"Could not obtain file '{zenodo_path}' from Zenodo record {zenodo_id}"
    )
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


def get_zenodo_record(folder: str) -> str | None:
    """
    Get Zenodo record ID for a given folder.

    Inputs :
        - folder : str
            Folder name to get the Zenodo record ID for

    Returns :
        - str | None : Zenodo record ID or None if not found
    """
    zenodo_map = {
        "Frostflow/16"  : "15799743",
        "Frostflow/48"  : "15696415",
        "Frostflow/256" : "15799754",
        "Frostflow/4096": "15799776",
        "Dayspring/16"  : "15799318",
        "Dayspring/48"  : "15721749",
        "Dayspring/256" : "15799474",
        "Dayspring/4096": "15799495",
        "Honeyside/16"  : "15799607",
        "Honeyside/48"  : "15799652",
        "Honeyside/256" : "15799731",
        "Honeyside/4096": "15696457",
        "Oak/318"       : "15743843",

        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018': '15877374',
        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_400GPa': '15877424',
        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa': '17417017',
        'Melting_curves/Monteux+600': '15728091',
        'Melting_curves/Monteux-600': '15728138',
        'Melting_curves/Wolf_Bower+2018': '15728072',
    }
    return zenodo_map.get(folder, None)

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

    for file in storage.files:
        for folder in folders:
            if not file.path[1:].startswith(folder):
                continue
            parts = file.path.split('/')[1:]
            target = Path(data_dir, *parts)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file already exists and has content
            if target.exists() and target.stat().st_size > 0:
                log.debug(f'Skipping existing file: {file.path}')
                continue

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
    osf_id: str,
    zenodo_id: str | None = None,
    desc: str,
    force: bool = False
) -> bool:
    """
    Generic download function.

    Attributes
    ----------
    folder: str
        Filename to download
    target: str
        name of target directory
    osf_id: str
        OSF project id
    zenodo_id: str
        Zenodo record id
    desc: str
        Description for logging
    force: bool
        Force a re-download even if valid

    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise
    """
    log.debug(f"Get {desc}?")

    # Check that target FWL_DATA folder exists
    data_dir = GetFWLData() / target
    data_dir.mkdir(parents=True, exist_ok=True)

    # Path to specific folder (download) within the data_dir folder
    folder_dir = data_dir / folder

    # Check if the folder needs updating
    folder_invalid = check_needs_update(folder_dir, zenodo_id) or force

    # Update the folder
    if folder_invalid:
        log.info(f"Downloading {desc} to {data_dir}")
        success = False

        # Try Zenodo in the first instance
        try:
            if zenodo_id is not None:
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
        if success:
            return True

        # If Zenodo fails, try OSF
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

        log.error(f"    Failed to download {desc} from IDs: Zenodo {zenodo_id}, OSF {osf_id}")
        return False

    else:
        log.debug(f"    {desc} already exists")
    return True


def download_surface_albedos():
    """
    Download reflectance data for various surface materials
    """
    download(
        folder = 'Hammond24',
        target = "surface_albedos",
        osf_id = '2gcd9',
        zenodo_id = '15880455',
        desc = 'surface reflectance data'
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

    download(
        folder = f'{name}/{bands}',
        target = "spectral_files",
        osf_id = 'vehxg',
        zenodo_id= get_zenodo_record(f'{name}/{bands}'),
        desc = f'{name}{bands} spectral file',
    )


def download_interior_lookuptables(clean=False):
    """
    Download basic interior lookup tables
    """
    log.debug("Download basic interior lookup tables")

    data_dir = GetFWLData() / "interior_lookup_tables"
    data_dir.mkdir(parents=True, exist_ok=True)

    for dir in ARAGOG_BASIC:
        folder_dir = data_dir / dir
        if clean:
            safe_rm(folder_dir.as_posix())
        download(
            folder = dir,
            target = data_dir,
            osf_id = "phsxf",
            zenodo_id = get_zenodo_record(dir),
            desc = f"Interior lookup tables: {dir}"
            )

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
    download(
        folder = dir,
        target = data_dir,
        osf_id = "phsxf",
        zenodo_id = get_zenodo_record(dir),
        desc = f"Melting curve data: {dir}"
        )

def download_stellar_spectra():
    """
    Download stellar spectra
    """
    download(
        folder = 'Named',
        target = "stellar_spectra",
        osf_id = '8r2sw',
        zenodo_id= '15721440',
        desc = 'stellar spectra'
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
    log.info("Downloading solar spectra to 'solar' folder")

    return download_zenodo_folder(
        zenodo_id='17981836',
        folder_dir=GetFWLData() / "stellar_spectra" / "solar")

def download_phoenix(alpha: float | int | str, FeH: float | int | str) -> bool:
    """
    Download and unpack a PHOENIX spectra ZIP like
    FeH+0.5_alpha+0.0_phoenixMedRes_R05000.zip
    into <FWL_DATA>/stellar_spectra/PHOENIX.

    NOTE: Assumes alpha and FeH are already mapped to nearest grid point.
    """

    phoenix_zenodo_id = "17674612"

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

    # first download readme
    get_zenodo_file(
        zenodo_id=phoenix_zenodo_id,
        folder_dir= data_dir / "PHOENIX",
        zenodo_path="_readme.md",)

    # then download zip
    if not get_zenodo_file(
        zenodo_id=phoenix_zenodo_id,
        folder_dir=folder_dir,
        zenodo_path=zip_name,
    ):
        log.error(f"Failed to download PHOENIX ZIP {zip_name} from Zenodo {phoenix_zenodo_id}")
        return False

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
    muscles_zenodo_id = "17802209"
    data_dir   = GetFWLData() / "stellar_spectra"
    folder_dir = data_dir / "MUSCLES"
    star_filename = f"{star_name.strip().lower().replace(' ', '-').replace('gj-', 'gj')}.txt" # lowercase, and; "trappist 1" -> "trappist-1", but "gj 876" or "gj-876" -> "gj876"
    log.info(f"Downloading MUSCLES file {star_filename}")

    # first download readme
    get_zenodo_file(
        zenodo_id=muscles_zenodo_id,
        folder_dir= data_dir / "MUSCLES",
        zenodo_path="_readme.md",)

    return get_zenodo_file(
        zenodo_id=muscles_zenodo_id,
        folder_dir=folder_dir,
        zenodo_path=star_filename,
    )

def download_exoplanet_data():
    """
    Download exoplanet data
    """
    download(
        folder = 'Exoplanets',
        target = "planet_reference",
        osf_id = 'fzwr4',
        zenodo_id= '15727878',
        desc = 'exoplanet data'
    )


def download_massradius_data():
    """
    Download mass-radius data
    """
    download(
        folder = 'Zeng2019',
        target = "mass_radius",
        osf_id = 'xge8t',
        zenodo_id= '15727899',
        desc = 'mass radius data'
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

    download(
    folder='EOS_Seager2007',
    target='EOS_material_properties',
    osf_id='dpkjb',
    zenodo_id= '15727998',
    desc='EOS Seager2007 material files'
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
