# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.observe.common import OBS_SOURCES

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def calc_synthetic_spectra(hf_row:dict, outdir:str, config:Config):
    '''
    Calculate "perfect" synthetic spectra. Does not model instrumentation.

    Results are saved to files on the disk.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    outdir : str
        The output directory for the PROTEUS run.
    config : Config
        PROTEUS config object.
    '''

    if config.observe.synthesis == "platon":
        from proteus.observe.platon import eclipse_depth, transit_depth
    else:
        raise ValueError(f"Unknown synthesis module '{config.observe.synthesis}'")

    # First, run synthetic observations
    for source in OBS_SOURCES:

        # Can we use this source?
        if (source == "profile") and (config.atmos_clim.module == "dummy"):
            continue
        if (source == "offchem") and (config.atmos_chem.module is None):
            continue

        log.debug(f"Synthesising observations for atmosphere set by '{source}'")

        # Compute transit and eclipse depth spectra
        transit_depth(hf_row, outdir, config, source)
        eclipse_depth(hf_row, outdir, config, source)

def run_observe(hf_row:dict, outdir:str, config:Config):
    '''
    Observe the planet!

    First, run the synthetic observations with "perfect" viewing conditions.
    These results are processed further by specific telescope simulators.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    outdir : str
        The output directory for the PROTEUS run.
    config : Config
        PROTEUS config object.
    '''

    log.info("Observing the planet...")

    # Synthetic spectra
    calc_synthetic_spectra(hf_row, outdir, config)

    # Telescope simulators go here
    # TODO: add telescope simulators
