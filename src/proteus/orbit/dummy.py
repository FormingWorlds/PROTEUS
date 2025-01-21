# Dummy orbit module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_dummy_tides(config:Config, phi:np.ndarray):
    '''
    Dummy tides module.
    '''

    # Default case; zero heating throughout the mantle
    H_tide = np.zeros(len(phi))

    # Inequality (less than, value)
    Pt_lt = bool(config.orbit.dummy.Phi_tide[0] == "<")
    Pt_va = float(config.orbit.dummy.Phi_tide[1:])

    # For regions with melt fraction in the appropriate range, set heating to be non-zero
    for i,p in enumerate(phi):
        if Pt_lt:
            if p < Pt_va:
                H_tide[i] = config.orbit.dummy.H_tide
        else:
            if p > Pt_va:
                H_tide[i] = config.orbit.dummy.H_tide

    return H_tide
