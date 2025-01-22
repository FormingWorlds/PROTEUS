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
    # Make copy of array, reverse order, and convert to Julia type
    return juliacall.convert(jl.Array[jl.Float64, 1], arr[::-1])

def run_lovepy(hf_row:dict, config:Config, interior_o:Interior_t):
    '''
    Run the lovepy tidal heating module.

    Sets the interior tidal heating and returns Im(k2) love number.
    '''

    # Default case; zero heating throughout the mantle
    interior_o.tides = np.zeros_like(interior_o.phi)
    nlev = len(interior_o.phi)

    # Get orbital properties
    omega = 2 * np.pi / hf_row["period"]
    ecc   = hf_row["eccentricity"]

    # Other arrays (FIX ME)
    interior_o.shear = np.ones_like(interior_o.phi) * config.orbit.lovepy.shear_modulus
    interior_o.bulk  = np.ones_like(interior_o.phi) * config.orbit.lovepy.bulk_modulus

    # Truncate arrays based on valid viscosity range
    visc_thresh = 1e9
    i_top = nlev-1
    for i in range(nlev-1,0,-1):
        if interior_o.visc[i] > visc_thresh:
            i_top = i

    # Check if fully liquid
    if i_top == nlev-1:
        return 0.0

    lov_rho    = interior_o.rho[i_top:]
    lov_visc   = interior_o.visc[i_top:]
    lov_shear  = interior_o.shear[i_top:]
    lov_bulk   = interior_o.bulk[i_top:]
    lov_mass   = interior_o.mass[i_top:]
    lov_radius = interior_o.radius[i_top:] # radius array length N+1

    # Calculate heating using lovepy
    power_prf, power_blk, k2_love = jl.calculate_heating(omega, ecc,
                                                            _jlarr(lov_rho),
                                                            _jlarr(lov_radius),
                                                            _jlarr(lov_visc),
                                                            _jlarr(lov_shear),
                                                            _jlarr(lov_bulk),
                                                            )
    interior_o.tides[i_top:] = np.array(power_prf)[::-1]
    interior_o.tides[-1] = interior_o.tides[-2]
    power_blk /= np.sum(lov_mass)

    # Return imaginary part of k2 love number
    return float(k2_love)
