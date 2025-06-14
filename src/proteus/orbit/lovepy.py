# Lovepy tidal heating module
from __future__ import annotations

import logging

import juliacall
import numpy as np
from juliacall import Main as jl

from proteus.interior.common import Interior_t
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger("fwl."+__name__)

def import_lovepy():
    log.debug("Import lovepy...")
    jl.seval("using LovePy")

def _jlarr(arr:np.array):
    # Make copy of array, reverse order, and convert to Julia type
    return juliacall.convert(jl.Array[jl.LovePy.prec, 1], arr[::-1])

def _jlsca(sca:float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.LovePy.prec, sca)

def run_lovepy(hf_row:dict, dirs:dict, interior_o:Interior_t,
                    visc_thresh:float, ncalc:int=1200) -> float:
    """Run the lovepy tidal heating module.

    Sets the interior tidal heating and returns Im(k2) love number.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        dirs: dict
            Dictionary of directories.
        interior_o: Interior_t
            Struct containing interior arrays at current time.
        visc_thresh: float
            Minimum viscosity required for heating [Pa s]
        ncalc: int
            Number of layers to use for tidal calculation, optional.
    Returns
    ----------
        Imk2_love: float
    """

    # Calculate angular frequency of rotation
    omega = 2 * np.pi / hf_row["orbital_period"]

    # Truncate arrays based on valid viscosity range
    i_top = interior_o.nlev_s-1
    for i in range(interior_o.nlev_s-1,0,-1):
        if interior_o.visc[i] >= visc_thresh:
            i_top = i

    # Check if fully liquid
    if i_top == interior_o.nlev_s-1:
        return 0.0

    # Construct arrays for lovepy
    lov_density = interior_o.density[i_top:]
    lov_visc    = interior_o.visc[i_top:]
    lov_shear   = interior_o.shear[i_top:]
    lov_bulk    = interior_o.bulk[i_top:]
    lov_mass    = interior_o.mass[i_top:]
    lov_radius  = interior_o.radius[i_top:] # radius array length N+1

    # Calculate heating using lovepy
    try:
        power_prf, power_blk, Imk2 = \
            jl.calc_lovepy_tides(_jlsca(omega),
                                 _jlsca(hf_row["eccentricity"]),
                                 _jlarr(lov_density),
                                 _jlarr(lov_radius),
                                 _jlarr(lov_visc),
                                 _jlarr(lov_shear),
                                 _jlarr(lov_bulk),
                                 ncalc=int(ncalc)
                                 )
    except juliacall.JuliaError as e:
        UpdateStatusfile(dirs, 26)
        log.error(e)
        raise RuntimeError("Encountered problem when running lovepy module")

    # Store result
    interior_o.tides[i_top:] = np.array(power_prf)[::-1]
    interior_o.tides[-1] = interior_o.tides[-2]

    # Verify result
    power_blk /= np.sum(lov_mass)
    log.debug("    power from bulk calc: %.3e W kg-1"%power_blk)

    # Return imaginary part of k2 love number
    return float(Imk2)
