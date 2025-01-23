# Dummy orbit module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior.common import Interior_t

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_dummy_orbit(config:Config, interior_o:Interior_t):
    '''
    Dummy orbit module.

    Sets interior tidal heating, return Im(k2) love number value of zero.
    '''

    # Default case; zero heating throughout the mantle
    interior_o.tides = np.zeros_like(interior_o.phi)

    # Inequality (less than, value)
    Pt_lt = bool(config.orbit.dummy.Phi_tide[0] == "<")
    Pt_va = float(config.orbit.dummy.Phi_tide[1:])

    # For regions with melt fraction in the appropriate range, set heating to be non-zero
    for i,p in enumerate(interior_o.phi):
        if Pt_lt:
            if p < Pt_va:
                interior_o.tides[i] = config.orbit.dummy.H_tide
        else:
            if p > Pt_va:
                interior_o.tides[i] = config.orbit.dummy.H_tide

    return 0.0
