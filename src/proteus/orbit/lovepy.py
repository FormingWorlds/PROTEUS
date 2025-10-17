# Lovepy tidal heating module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import juliacall
import numpy as np
from juliacall import Main as jl

from proteus.interior.common import Interior_t
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def import_lovepy():
    log.debug("Import lovepy...")
    jl.seval("using LovePy")

def _jlarr(arr:np.array):
    # Make copy of array, reverse order, and convert to Julia type
    cop = np.array(arr, copy=True, dtype=float).flatten()
    return juliacall.convert(jl.Array[jl.LovePy.prec, 1], cop)

def _jlsca(sca:float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.LovePy.prec, sca)

def run_lovepy(hf_row:dict, dirs:dict, interior_o:Interior_t, config:Config) -> float:
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
        config: Config
            PROTEUS config object
    Returns
    ----------
        Imk2_love: float
    """

    # Calculate angular frequency of rotation
    omega = _jlsca(2 * np.pi / hf_row["orbital_period"])
    ecc   = _jlsca(hf_row["eccentricity"])

    # Copy arrays
    arr_keys = ("density", "visc", "shear", "bulk", "mass", "radius")
    lov = {k:np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior.module == "spider":
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    # Get viscous region and check if fully liquid
    i_top = 0  # index of topmost cell which has visc>visc_thresh
    if config.interior.module == "dummy":
        if lov["visc"][0] < config.orbit.lovepy.visc_thresh:
            return 0.0

        # Construct arrays for lovepy (we need two cells, three edges here)
        for k in arr_keys:
            if k == "radius":
                rmid = np.median(lov["radius"])
                arr = [lov["radius"][0], rmid, lov["radius"][1]]
                lov[k] = _jlarr(arr[:])
            else:
                arr = [lov[k][0],lov[k][0]]
                lov[k] = _jlarr(arr[:])

    else:
        # for spider/aragog, loop from bottom upwards
        for i in range(interior_o.nlev_s):
            if lov["visc"][i] >= config.orbit.lovepy.visc_thresh:
                i_top = i

        # fully liquid
        if i_top <= 1:
            return 0.0

        # Construct arrays for lovepy
        for k in arr_keys:
            if k == "radius":
                i = i_top + 1
            else:
                i = i_top
            lov[k] = _jlarr(lov[k][:i])

    # Calculate heating using lovepy
    try:
        power_prf, power_blk, Imk2 = \
            jl.calc_lovepy_tides(omega, ecc,
                                 lov["density"],
                                 lov["radius"],
                                 lov["visc"],
                                 lov["shear"],
                                 lov["bulk"],
                                 ncalc=int(config.orbit.lovepy.ncalc)
                                 )
    except juliacall.JuliaError as e:
        UpdateStatusfile(dirs, 26)
        log.error(e)
        raise RuntimeError("Encountered problem when running lovepy module")


    # Extract result and store
    if config.interior.module == "dummy":
        interior_o.tides[0] = power_prf[1]

    else:
        # Pad power array to make it full length
        tides = np.zeros(np.shape(interior_o.tides), dtype=float)
        tides[:i_top] = power_prf[:]
        tides[0] = tides[1]

        # Store result, flipping for SPIDER
        if config.interior.module == "spider":
            interior_o.tides[:] = tides[::-1]
        else:
            interior_o.tides[:] = tides[:]

        # Verify result against bulk calculation
        power_blk /= np.sum(lov["mass"])
        log.debug("    power from bulk calc: %.3e W kg-1"%power_blk)

    # Return imaginary part of k2 love number
    return float(Imk2)
