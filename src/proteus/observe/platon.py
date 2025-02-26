# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import os

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def transit_depth(hf_row:dict, outdir:str):
    from platon.transit_depth_calculator import TransitDepthCalculator

    # All planet quantities in SI
    Rs = hf_row["R_star"]     # Radius of star [m]
    Mp = hf_row["M_tot"]      # Mass of planet [kg]
    Rp = hf_row["R_int"]      # Radius of planet [m]
    Ts = hf_row["T_surf"]     # Surface temperature [K]

    # create a TransitDepthCalculator object
    depth_calculator = TransitDepthCalculator()

    # compute wavelength dependent transit depths
    log.debug("Calculate transmission spectrum")
    wl, td, _ = depth_calculator.compute_depths(Rs, Mp, Rp, Ts,
                                                    logZ=3, CO_ratio=0.5,
                                                    T_star=hf_row["T_star"]
                                                    cloudtop_pressure=hf_row["P_surf"])

    wl = np.array(wl) * 1e6 # convert to um
    td = np.array(td) * 1e6 # convert to ppm

    # write file
    log.debug("Writing transmission spectrum")
    header = "Wavelength [um], transit depth [ppm]"
    fpath = os.path.join(outdir, "data", "obs_synth_transit.csv")
    np.savetxt(fpath, np.array([wl, td]).T, fmt="%.6e", header=header)
