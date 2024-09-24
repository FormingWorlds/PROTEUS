from __future__ import annotations

import logging

from janus.utils.data import (
    DownloadSpectralFiles,
    DownloadStellarSpectra,
    GetFWLData,
    download_folder,
)
from osfclient.api import OSF

log = logging.getLogger("fwl."+__name__)

def DownloadSurfaceAlbedos():
    """
    Download surface optical properties
    """
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


def download_basic():
    """
    Download all basic data
    """

    log.debug("Downloading basic data")

    log.debug("    spectral files")
    DownloadSpectralFiles()

    log.debug("    stellar spectra")
    DownloadStellarSpectra()

    log.debug("    surface albedos")
    DownloadSurfaceAlbedos()
