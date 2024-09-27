from __future__ import annotations

import logging

from osfclient.api import OSF

log = logging.getLogger("fwl."+__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

log.info(f'FWL data location: {FWL_DATA_DIR}')

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

def download_albedos():
    """
    Download surface optical properties
    """
    log.debug("Get surface albedos")
    #project ID of the stellar spectra on OSF
    project_id = '2gcd9'
    folder_name = 'Hammond24'

    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')

    data_dir = GetFWLData() / "surface_albedos"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / folder_name).exists():
        log.info(f"Downloading surface albedos to {data_dir}")
        download_folder(storage=storage, folders=[folder_name], data_dir=data_dir)

def download_spectral_files():
    """
    Download spectral files
    """
    from janus.utils.data import DownloadSpectralFiles
    log.debug("Get spectral files")
    DownloadSpectralFiles()

def download_stellar_spectra():
    """
    Download stellar spectra
    """
    from janus.utils.data import DownloadStellarSpectra
    log.debug("Get stellar spectra")
    DownloadStellarSpectra()

def download_evolution_tracks():
    """
    Download evolution tracks
    """
    from mors.utils.data import DownloadEvolutionTracks
    log.debug("Get evolution tracks")
    DownloadEvolutionTracks()


def download_sufficient(OPTIONS:dict):
    """
    Download the required data based on the current options
    """

    # Star stuff
    if OPTIONS["star_model"] in [0,1]:
        download_stellar_spectra()
        download_evolution_tracks()

    # Atmosphere stuff
    if OPTIONS["atmosphere_model"] in [0,1]:
        download_spectral_files()
    if OPTIONS["atmosphere_model"] == 1:
        download_surface_albedos()

    
