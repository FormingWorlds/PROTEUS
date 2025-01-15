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
    Dummy tides module
    '''

    H_tide = np.zeros(len(phi))

    for i,p in enumerate(phi):
        if p > config.orbit.dummy.Phi_tide:
            H_tide[i] = 0.0
        else:
            H_tide[i] = config.orbit.dummy.H_tide

    return H_tide





