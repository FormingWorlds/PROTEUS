# Lovepy tidal heating module
from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING
import juliacall
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

def _jlarr(arr:np.array):
    # Make copy of array and convert to Julia type
    return juliacall.convert(jl.Array[jl.Float64, 1], np.array(arr, dtype=float))

def run_lovepy(hf_row:dict, interior_o:Interior_t):
    '''
    Run the lovepy tidal heating module
    '''

    # Default case; zero heating throughout the mantle
    H_tide = np.zeros(len(interior_o.phi))

    # Other arrays (FIX ME)
    interior_o.shear = np.ones(len(H_tide)) * 60e9
    interior_o.bulk  = np.ones(len(H_tide)) * 100e9

    # Get orbital properties
    omega = 2 * np.pi / hf_row["period"]
    ecc   = hf_row["eccentricity"]

    # Calculate heating using lovepy
    power = jl.calculate_heating(omega, ecc,
                                    _jlarr(interior_o.rho),
                                    _jlarr(interior_o.radius),
                                    _jlarr(interior_o.visc),
                                    _jlarr(interior_o.shear),
                                    _jlarr(interior_o.bulk),
                                )
    print(power)

    return H_tide
