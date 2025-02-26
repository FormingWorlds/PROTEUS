# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def transit_depth_synth(config:Config, hf_row:dict, outdir:str):
    '''
    Calculate transit depth spectrum from the simulation.
    '''

    log.info("Calculating transit depth spectrum")
    if config.observe.synthesis == "platon":
        from proteus.observe.platon import transit_depth
        transit_depth(hf_row, outdir)

def eclipse_depth_synth(config:Config, hf_row:dict, outdir:str):
    '''
    Calculate eclipse depth from the simulation
    '''

    log.info("Calculating eclipse depth spectrum")
    if config.observe.synthesis == "platon":
        from proteus.observe.platon import eclipse_depth
        eclipse_depth(hf_row, outdir)
