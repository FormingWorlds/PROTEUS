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
    """Run the dummy orbit module.

    Sets interior tidal heating, returns Im(k2) value from config.

    Parameters
    ----------
        config : Config
            Model configuration.
        interior_o: Interior_t
            Struct containing interior arrays at current time.

    Returns
    ----------
        Imk2_love: float
            Always returns the value provided in the config.
    """

    # Default case; zero heating throughout the mantle
    interior_o.tides = np.zeros_like(interior_o.phi)

    # Inequality (less than, value)
    Pt_lt = bool(config.orbit.dummy.Phi_tide[0] == "<")
    Pt_va = float(config.orbit.dummy.Phi_tide[1:])

    # Heating scales with melt fraction when inequality is satisfied.
    #   Elsewhere, heating is zero.
    for i,p in enumerate(interior_o.phi):
        if Pt_lt:
            if p < Pt_va:
                # Heating maximised when fully solid.
                interior_o.tides[i] = config.orbit.dummy.H_tide * (1-p/Pt_va)
        else:
            if p > Pt_va:
                # Heating maximised when fully liquid.
                interior_o.tides[i] = config.orbit.dummy.H_tide * (p-Pt_va) / (1-Pt_va)

    return config.orbit.dummy.Imk2
