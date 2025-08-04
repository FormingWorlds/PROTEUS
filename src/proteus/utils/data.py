from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import subprocess as sp
from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs
from osfclient.api import OSF

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

log.debug(f'FWL data location: {FWL_DATA_DIR}')

def download_zenodo_folder(zenodo_id: str, folder_dir: Path):
    """
    Download a specific Zenodo record into specified folder

    Inputs :
        - zenodo_id : str
            Zenodo record ID to download
        - folder_dir : Path
            Local directory where the Zenodo record will be downloaded
    """

    shutil.rmtree(str(folder_dir), ignore_errors=True)
    folder_dir.mkdir(parents=True)
    cmd = [
            "zenodo_get", zenodo_id,
            "-o", folder_dir
        ]
    out = os.path.join(GetFWLData(), "zenodo_download.log")
    log.debug("    zenodo_get, logging to %s"%out)
    with open(out,'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

def md5(_fname):
    """Return the md5 hash of a file."""

    # https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(_fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_zenodo_folder(zenodo_id: str, folder_dir: Path, hash_maxfilesize=100e6):
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

    # Use zenodo_get to obtain md5 hashes
    # They will be saved to a txt file in folder_dir
    cmd = [ "zenodo_get", zenodo_id, "-m" ]
    out = os.path.join(GetFWLData(), "zenodo_validate.log")
    log.debug("    zenodo_get, logging to %s"%out)
    with open(out,'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl, cwd=folder_dir)

    # Check that hashes file exists
    md5sums_path = os.path.join(folder_dir, "md5sums.txt")
    if not os.path.isfile(md5sums_path):
        return False

    # Read hashes file
    with open(md5sums_path,'r') as hdl:
        md5sums = hdl.readlines()

    # Check each item in the record...
    for line in md5sums:
        sum_expect, name = line.strip().split()
        file = os.path.join(folder_dir, name)

        # exit here if file does not exist
        if not os.path.exists(file):
            log.warning(f"Detected missing file {name} (Zenodo record {zenodo_id})")
            return False

        # don't check the hashes of very large files, because it's slow
        if os.path.getsize(file) > hash_maxfilesize:
            return True

        # check the actual hash of the file on disk, compare to expected
        sum_actual = md5(file).strip()
        if sum_actual != sum_expect:
            log.warning(f"Detected invalid file {name} (Zenodo record {zenodo_id})")
            log.warning(f"    expected hash {sum_expect}, got {sum_actual}")
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
    for file in storage.files:
        for folder in folders:
            if not file.path[1:].startswith(folder):
                continue
            parts = file.path.split('/')[1:]
            target = Path(data_dir, *parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            log.info(f'Downloading {file.path}...')
            with open(target, 'wb') as f:
                file.write_to(f)
            break

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
                download_zenodo_folder(zenodo_id=zenodo_id, folder_dir=folder_dir)

                # validate files ok?
                success = validate_zenodo_folder(zenodo_id, folder_dir)
        except RuntimeError as e:
            log.warning(f"    Zenodo download failed: {e}")
            folder_dir.rmdir()
        if success:
            return True

        # If Zenodo fails, try OSF
        try:
            storage = get_osf(osf_id)
            download_OSF_folder(storage=storage, folders=[folder], data_dir=data_dir)
            success = True
        except RuntimeError as e:
            log.warning(f"    OSF download failed: {e}")
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

def download_interior_lookuptables():
    """
    Download interior lookup tables
    """
    from aragog.data import DownloadLookupTableData
    log.debug("Get interior lookup tables")
    DownloadLookupTableData()

def download_melting_curves(config:Config):
    """
    Download melting curve data
    """
    from aragog.data import DownloadLookupTableData
    log.debug("Get melting curve data")
    dir = (
        "Melting_curves/"
        + config.interior.melting_dir
    )
    DownloadLookupTableData(dir)

def _get_sufficient(config:Config):
    # Star stuff
    if config.star.module == "mors":
        download_stellar_spectra()
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
        download_interior_lookuptables()
        download_melting_curves(config)


def download_sufficient_data(config:Config):
    """
    Download the required data based on the current options
    """

    log.info("Getting physical and reference data")

    if config.params.offline:
        # Don't try to get data
        log.warning("Running in offline mode. Will not check for reference data.")

    else:
        # Try to get data
        try:
            _get_sufficient(config)

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
