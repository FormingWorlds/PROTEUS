from __future__ import annotations

import functools
import logging
import os
import subprocess as sp
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

import platformdirs
from osfclient.api import OSF

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

log.debug(f'FWL data location: {FWL_DATA_DIR}')

def download_folder(*, storage, folders: list[str], data_dir: Path):
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


def download(
    *,
    folder: str,
    target: str,
    osf_id: str,
    desc: str,
    max_tries: int = 3,
    wait_time: float = 5,
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
    desc: str
        Description for logging
    max_tries: int
        Number of tries to download the file
    wait_time: float
        Time to wait between tries

    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise
    """
    log.debug(f"Get {desc}?")

    data_dir = GetFWLData() / target
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / folder).exists():
        storage = get_osf(osf_id)
        log.info(f"Downloading {desc} to {data_dir}")
        for i in range(max_tries):
            log.debug(f"    attempt {i+1}")
            try:
                download_folder(storage=storage, folders=[folder], data_dir=data_dir)
                break
            except RuntimeError as e:
                log.warning(f"    {desc} download failed: {e}")
                if i < max_tries - 1:
                    log.info(f"    Retrying in {wait_time} seconds...")
                    sleep(wait_time)
                else:
                    log.error(f"    Failed to download {desc} after {max_tries} attempts")
                    return False
    else:
        log.debug(f"    {desc} already exists")
    return True


def download_surface_albedos():
    """
    Download surface optical properties
    """
    download(
        folder = 'Hammond24',
        target = "surface_albedos",
        osf_id = '2gcd9',
        desc = 'surface albedos'
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
        desc = 'exoplanet data'
    )


def download_massradius_data():
    """
    Download mass-radius data
    """
    download(
        folder = 'Mass-radius',
        target = "mass_radius",
        osf_id = 'fzwr4',
        desc = 'mass radius data'
    )


def download_evolution_tracks(track:str):
    """
    Download evolution tracks
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

def download_melting_curves():
    """
    Download melting curve data
    """
    download(
        folder = 'Melting_curves',
        target = "interior_lookup_tables",
        osf_id = 'phsxf',
        desc = 'melting curve data'
    )

def _get_sufficient(config:Config):
    # Star stuff
    if config.star.module == "mors":
        download_stellar_spectra()
        if config.star.mors.tracks == 'spada':
            download_evolution_tracks("Spada")
        else:
            download_evolution_tracks("Baraffe")

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
        download_melting_curves()


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
