# Dummy star module
from __future__ import annotations

import logging

import numpy as np

from proteus.utils.constants import AU, const_sigma
from proteus.utils.phys import planck_wav

log = logging.getLogger("fwl."+__name__)

PLANCK_MIN_TEMPERATURE:float = 0.1

def generate_spectrum(tmp:float, R_star:float):
    '''
    Get stellar spectrum at 1 AU, assuming that the star emits like a blackbody.

    Parameters
    -----------
        tmp : float
            Temperature [K]
        R_star : float
            Stellar radius [m]

    Returns
    -----------
        wl_arr : list
            Wavelength bin centres [nm]
        fl_arr : list
            Stellar spectral flux density at 1 AU from star [erg s-1 cm-2 nm-1]
    '''

    log.debug("Generating stellar spectrum at Teff=%.0f K"%tmp)

    # Allocate wavelength array
    wl_min = 1e-10  # 0.1 nm
    wl_max = 1e-1   # 10 cm
    wl_pts = 600    # number of points to sample
    wl_arr = np.logspace(np.log10(wl_min), np.log10(wl_max), wl_pts)

    # Allocate flux array with zeros
    fl_arr = np.zeros(np.shape(wl_arr))

    # If Teff=0, keep fluxes at zero. This is equivalent to turning the star 'off'.

    # Else, for non-zero effective temperatures...
    if tmp > PLANCK_MIN_TEMPERATURE:

        # Evaluate planck function in each bin
        for i,wav in enumerate(wl_arr):
            fl_arr[i] = planck_wav(tmp, wav) # W m-2 m-1 at stellar surface

        # Scale from stellar surface to 1 AU
        fl_arr *= (R_star/AU)**2

        # Convert m-1 to nm-1
        fl_arr *= 1e-9

        # Convert W m-2 to erg s-1 cm-2
        fl_arr *= 1e3

    # Convert wavelengths to nm
    wl_arr *= 1e9

    # Return as lists
    return list(wl_arr), list(fl_arr)

def calc_star_luminosity(tmp:float, R_star:float):
    '''
    Calculate star's bolometric luminosity.

    Assumes that the star emits like a blackbody.
    '''

    # Evaluate stefan-boltzmann law at Teff
    lum = 0.0
    if tmp > PLANCK_MIN_TEMPERATURE:
        lum = 4 * np.pi * R_star * R_star * const_sigma * (tmp**4)

    return lum


def calc_instellation(tmp:float, R_star:float, sep:float):
    '''
    Calculate planet's instellation based on the star's luminosity.

    Parameters
    -----------
        tmp : float
            Star's effective temperature
        R_star : float
            Star's radius [m]
        sep : float
            Planet-star separation [m]

    Returns
    -----------
        S_0 : float
            Instellation [W m-2]
    '''

    # Get luminosity
    L_bol = calc_star_luminosity(tmp, R_star)

    # Return flux at planet-star distance
    return L_bol / (4 * np.pi * sep * sep)
