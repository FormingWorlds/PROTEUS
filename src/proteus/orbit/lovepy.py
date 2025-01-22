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

def run_lovepy(hf_row:dict, config:Config, interior_o:Interior_t):
    '''
    Run the lovepy tidal heating module
    '''

    # Default case; zero heating throughout the mantle
    H_tide = np.zeros_like(interior_o.phi)

    # Get orbital properties
    omega = 2 * np.pi / hf_row["period"]
    ecc   = hf_row["eccentricity"]

    # Other arrays (FIX ME)
    interior_o.shear = np.ones_like(interior_o.phi) * config.orbit.lovepy.shear_modulus
    interior_o.bulk  = np.ones_like(interior_o.phi) * config.orbit.lovepy.bulk_modulus

    # Truncated arrays based on valid viscosity range
    visc_crit = 1e9
    mask = interior_o.visc > visc_crit
    lov_rho    = interior_o.rho[mask]
    lov_radius = interior_o.radius[mask]
    lov_visc   = interior_o.visc[mask]
    lov_shear  = interior_o.shear[mask]
    lov_bulk   = interior_o.bulk[mask]

    # Viscosity is small everywhere?
    #    Return default value (H_tide = 0)
    if len(lov_rho)  == 0:
        return H_tide

    # Calculate heating using lovepy
    power = jl.calculate_heating(omega, ecc,
                                    _jlarr(lov_rho),
                                    _jlarr(lov_radius),
                                    _jlarr(lov_visc),
                                    _jlarr(lov_shear),
                                    _jlarr(lov_bulk),
                                )
    print("Power from lovepy: %.3e W"%power)
    H_tide[mask] = power / np.sum(interior_o.mass[mask])

    # Return array
    return H_tide
