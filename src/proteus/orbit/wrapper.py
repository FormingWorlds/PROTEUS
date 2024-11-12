# Generic orbital dynamics stuff
from __future__ import annotations

import logging
import numpy as np

from proteus.utils.constants import AU, const_G

log = logging.getLogger("fwl."+__name__)

def update_separation(hf_row:dict, sma:float, ecc:float):
    '''
    Calculate time-averaged orbital separation on an elliptical path.

    Converts from AU to metres.
    https://physics.stackexchange.com/a/715749

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
        sma: float
            Semimajor axis [AU]
        ecc: float
            Eccentricity
    '''

    hf_row["separation"] = sma * AU *  (1 + 0.5*ecc*ecc)

def update_period(hf_row:dict, sma:float):
    '''
    Calculate orbital period on an elliptical path.

    Assuming that M_volatiles << M_star + M_mantle + M_core.
    https://en.wikipedia.org/wiki/Elliptic_orbit#Orbital_period

    Parameters
    -------------
        hf_row: dict
            Current helpfile row
        sma: float
            Semimajor axis [AU]
    '''

    # Standard gravitational parameter neglecting volatile mass.
    mu = const_G * (hf_row["M_star"] + hf_row["M_mantle"] + hf_row["M_core"])

    # Orbital period [seconds]
    hf_row["period"] = 2 * np.pi * (sma*sma*sma/mu)**0.5
