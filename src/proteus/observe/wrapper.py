# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.observe.common import OBS_SOURCES

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def _source_available(source: str, config: Config) -> bool:
    if (source == 'profile') and (config.atmos_clim.module == 'dummy'):
        return False
    if (source == 'offchem') and (config.atmos_chem.module is None):
        return False
    return True


def _get_observe_sources(config: Config) -> list[str]:
    selected = getattr(config.observe, 'source', 'all')
    sources = list(OBS_SOURCES) if selected == 'all' else [selected]

    available: list[str] = []
    for source in sources:
        if _source_available(source, config):
            available.append(source)
            continue

        if selected == 'all':
            continue

        if source == 'profile':
            raise ValueError("observe.source = 'profile' requires atmos_clim.module != 'dummy'")
        if source == 'offchem':
            raise ValueError("observe.source = 'offchem' requires atmos_chem.module != none")

    return available


def _get_observe_spectrum_type(config: Config) -> str:
    return getattr(config.observe, 'spectrum_type', 'both')


def calc_synthetic_spectra(hf_row: dict, config: Config, dirs: dict[str, str]):
    """
    Calculate "perfect" synthetic spectra. Does not model instrumentation.

    Results are saved to files on the disk.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    config : Config
        PROTEUS config object.
    dirs : dict[str, str]
        Directories dictionary created during startup.
    """

    if config.observe.module == 'petitRADTRANS':
        from proteus.observe.petitRADTRANS import eclipse_depth, transit_depth
    else:
        raise ValueError(f"Unknown synthesis module '{config.observe.module}'")

    # First, run synthetic observations
    spectrum_type = _get_observe_spectrum_type(config)
    for source in _get_observe_sources(config):
        log.debug(f"Synthesising observations for atmosphere set by '{source}'")

        # Compute selected transit and/or eclipse spectra.
        if spectrum_type in ('both', 'transit'):
            transit_depth(hf_row, config, source, dirs)
        if spectrum_type in ('both', 'eclipse'):
            eclipse_depth(hf_row, config, source, dirs)


def run_observe(hf_row: dict, config: Config, dirs: dict[str, str]):
    """
    Observe the planet!

    First, run the synthetic observations with "perfect" viewing conditions.
    These results are processed further by specific telescope simulators.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    config : Config
        PROTEUS config object.
    dirs : dict[str, str]
        Directories dictionary created during startup.
    """

    log.info('Observing the planet...')

    # Synthetic spectra
    calc_synthetic_spectra(hf_row, config, dirs)

    # Telescope simulators go here
    # TODO: add telescope simulators
