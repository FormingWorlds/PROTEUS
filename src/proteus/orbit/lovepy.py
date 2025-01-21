# Lovepy tidal heating module
from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING
from juliacall import Main as jl

from proteus.interior.common import Interior_t

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def import_lovepy(lovepy_dir:str):
    log.debug("Import lovepy...")
    lib = os.path.join(os.path.abspath(lovepy_dir), "lovepy.jl")
    jl.seval('include("%s")'%lib)

def run_lovepy(hf_row:dict, interior_o:Interior_t):
    '''
    Run the lovepy tidal heating module
    '''

    # Default case; zero heating throughout the mantle
    H_tide = np.zeros(len(interior_o.phi))

    # Calculate orbital properties
    omega = 2 * np.pi / hf_row["period"]
    ecc   = hf_row["eccentricity"]

    power = jl.calculate_heating(omega, ecc)
    print(power)

    return H_tide
