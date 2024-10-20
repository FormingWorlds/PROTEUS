# Dummy star module
from __future__ import annotations

import logging

import numpy as np

from proteus.utils.constants import AU
from proteus.utils.phys import planck_wav

log = logging.getLogger("fwl."+__name__)

def generate_spectrum(wl_arr, tmp:float, R_star:float, lowcut:float=0.0):
    '''
    Get stellar spectrum at 1 AU.

    Can specify a short wavelength cutoff. Flux density is constant below this value,
    to represent a toy model of the "Balmer break".

    Parameters
    -----------
        wl_arr : list | np.ndarray
            Wavelengths [nm]
        tmp : float
            Temperature [K]
        R_star : float
            Stellar radius [m]

        lowcut : float
            Short wavelength cutoff [nm]

    Returns
    -----------
        fl_arr : list
            Stellar spectral flux density [erg s-1 cm-2 nm-1]
    '''

    # Allocate zero array
    fl_arr = np.zeros(np.shape(wl_arr))

    # Evaluate planck function in each bin
    for i,wav in enumerate(wl_arr):
        wav = max(wav, lowcut)
        fl_arr[i] = planck_wav(tmp, wav) # W m-2 m-1 at stellar surface

    # Scale from stellar surface to 1 AU
    fl_arr *= (R_star/AU)**2

    # Convert m-1 to nm-1
    fl_arr *= 1e-9

    # Convert W m-2 to erg s-1 cm-2
    fl_arr *= 1e3

    # Return as list
    return list(fl_arr)
