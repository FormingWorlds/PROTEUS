from __future__ import annotations

import numpy as np
from proteus.utils.constants import const_sigma, const_c, const_h, R_sun, AU

def planck(Teff:float, wave:float):
    """Evalulate the planck function.

    Parameters
    ----------
        Teff : float
            Effective temperature [K]
        wave : float
            Wavelength [nm]

    Returns
    ----------
        flux : float
            Spectral flux density [W m-2 nm-1]
    """
    # Convert nm to m
    wave = wave * 1.0e-9

    # Optimisation variables
    hc = const_h * const_c

    # Calculate planck function value [W m-2 sr-1 m-1]
    # http://spiff.rit.edu/classes/phys317/lectures/planck.html
    flx = 2.0 * hc * (c_vac / wave**5) / ( exp(hc / (wav * k_B * tmp)) - 1.0)

    # Integrate solid angle (hemisphere), convert units
    flx = flx * pi * 1.0e-9 # [W m-2 nm-1]

    return flx

def calc_instellation(Teff:float, sep:float, Rstar:float):
    """Get the planetary instellation flux.

    Parameters
    ----------
        Teff : float
            Effective temperature [K]
        sep : float
            Planet-star separation [m]
        Rstar : float
            Stellar radius [m]

    Returns
    ----------
        flux : float
            Bolometric instellation flux [W m-2]
    """


    # Get the flux at the stellar surface
    F_surf = const_sigma * Teff**4

    # Scale from stellar surface to planet
    return F_surf * (Rstar / sep)**2


def calc_spectrum(Teff:float, Rstar:float, bin_c:list):
    """Calculate stellar spectrum at 1 AU

    Parameters
    ----------
        Teff : float
            Effective temperature [K]
        Rstar : float
            Star radius [R_sun]
        bin_c : list
            List of wavelength bin centres(ascending) [nm]

    Returns
    ----------
        flux : np.ndarray
            Spectral flux density [erg s-1 cm-2 nm-1]
    """

    # Convert to numpy array
    bin_c = np.array(bin_c, dtype=float)

    # Output array
    fluxes = np.zeros(np.shape(bin_c), dtype=float)

    # Calculate fluxes in each bin
    for i,w in enumerate(bin_c):
        fluxes[i] = planck(Teff, w)

    # Convert [W m-2 nm-1] -> [erg s-1 cm-2 nm-1]
    fluxes *= 1000.0

    # Scale from stellar surface to 1 AU
    fluxes *= (Rsun * Rstar / AU )**2

    return fluxes

