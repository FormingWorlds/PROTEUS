from __future__ import annotations

import logging
import os
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

def get_osf(id:str):
    """
    Generate an object to access OSF storage
    """
    osf = OSF()
    project = osf.project(id)
    return project.storage('osfstorage')


def download_surface_albedos():
    """
    Download surface optical properties
    """
    log.debug("Get surface albedos?")
    storage = get_osf('2gcd9')

    folder_name = 'Hammond24'
    data_dir = GetFWLData() / "surface_albedos"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / folder_name).exists():
        log.info(f"Downloading surface albedos to {data_dir}")
        download_folder(storage=storage, folders=[folder_name], data_dir=data_dir)

def download_spectral_files(fname: str="", nband: int=256):
    """
    Download spectral files data

    Inputs :
        - fname (optional) :    folder name, i.e. "/Dayspring"
                                if not provided download all the basic list
        - nband (optional) :    number of band = 16, 48, 256, 4096
                                (only relevant for Dayspring, Frostflow and Honeyside)
    """
    log.debug("Get spectral files?")

    #Create spectral file data repository if not existing
    data_dir = GetFWLData() / "spectral_files"
    data_dir.mkdir(parents=True, exist_ok=True)

    #Link with OSF project repository
    storage = get_osf('vehxg')

    basic_list = (
        "Dayspring/48",
        "Dayspring/256",
        "Frostflow/256",
        "Honeyside/4096",
        )

    #If no folder specified download all basic list
    if not fname:
        folder_list = basic_list
    else:
        folder_list = [fname + "/" + str(nband)]

    folders = [folder for folder in folder_list if not (data_dir / folder).exists()]

    if folders:
        log.info(f"Downloading spectral files to {data_dir}")
        log.debug("\t"+str(folders))
        download_folder(storage=storage, folders=folders, data_dir=data_dir)


def download_stellar_spectra():
    """
    Download stellar spectra
    """
    log.debug("Get stellar spectra?")

    folder_name = 'Named'
    storage = get_osf('8r2sw')

    data_dir = GetFWLData() / "stellar_spectra"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / folder_name).exists():
        log.info(f"Downloading stellar spectra to {data_dir}")
        download_folder(storage=storage, folders=[folder_name], data_dir=data_dir)

def download_exoplanet_data():
    """
    Download exoplanet data
    """
    log.debug("Get exoplanet data?")

    folder_name = 'Exoplanets'
    storage = get_osf('fzwr4')

    data_dir = GetFWLData() / "exoplanets"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / folder_name).exists():
        log.info(f"Downloading exoplanet population data to {data_dir}")
        download_folder(storage=storage, folders=[folder_name], data_dir=data_dir)

def download_evolution_tracks(track:str):
    """
    Download evolution tracks
    """
    from mors.data import DownloadEvolutionTracks
    log.debug("Get evolution tracks")
    DownloadEvolutionTracks(track)

def download_sufficient_data(config:Config):
    """
    Download the required data based on the current options
    """

    # Star stuff
    if config.star.mors.tracks in ('spada', 'baraffe'):
        download_stellar_spectra()
        if config.star.mors.tracks == 'spada':
            download_evolution_tracks("Spada")
        else:
            download_evolution_tracks("Baraffe")

    # Atmosphere stuff
    if config.atmos_clim.module in ('janus', 'agni'):
        download_spectral_files()
    if config.atmos_clim.module == 'agni':
        download_surface_albedos()

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
