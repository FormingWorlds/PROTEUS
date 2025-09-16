# Small physics functions that can be used universally
# This file should not depend on too many other files, as this can cause circular import issues

from __future__ import annotations

import logging

import numpy as np

from proteus.utils.constants import const_c, const_h, const_k

log = logging.getLogger("fwl."+__name__)

def planck_wav(tmp:float, wav:float):
    '''
    Evaluate the planck function at a given temperature and wavelength.
    All in SI units.

    Parameters
    -----------
        tmp : float
            Temperature [K]
        wav : float
            Wavelength [m]

    Returns
    -----------
        flx : float
            Spectral flux density [W m-2 m-1]
    '''

    # Optimisation variables
    wav5 = wav*wav*wav*wav*wav
    hc   = const_h * const_c

    # Calculate planck function value [W m-2 sr-1 m-1]
    # http://spiff.rit.edu/classes/phys317/lectures/planck.html
    with np.errstate(all='raise'):
        # Catch overflow error, which can occur when evaluating at short wavelengths
        try:
            flx = 2.0 * hc * (const_c / wav5) / ( np.exp(hc / (wav * const_k * tmp)) - 1.0)
        except FloatingPointError:
            flx = 0.0

    # Integrate solid angle (hemisphere)
    flx = flx * np.pi

    # Do not allow zero or near-zero fluxes
    flx = max(flx, 1e-40)

    return flx
