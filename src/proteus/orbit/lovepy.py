# Lovepy tidal heating module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from juliacall import Main as jl

from proteus.interior.common import Interior_t

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def activate_julia():
    jl.Pkg.add("DoubleFloats")

def run_lovepy(config:Config, interior_o:Interior_t):
    '''
    Lovepy tidal heating module
    '''

    # Default case; zero heating throughout the mantle
    H_tide = np.zeros(len(interior_o.phi))

    return H_tide
