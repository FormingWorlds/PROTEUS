# Functions used to run the BOREAS escape module
from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING
import numpy as np

from proteus.utils.constants import element_list

# Import BOREAS from local path
BOREAS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..","..","..","boreas"))
print(BOREAS_PATH)
sys.path.append(BOREAS_PATH)
from boreas import Main as bm # noqa

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_boreas(config:Config, hf_row:dict):
    """Run BOREAS escape model.

    Calculates the mass loss rate of each element.
    Updates the quantities in hf_row as appropriate.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    log.info("Running fractionated escape (BOREAS) ...")

    # Set parameters for escape
    params          = bm.ModelParams()

    # Pass elemental mixing ratios from Rxuv
    for e in element_list:
        setattr(params, f"X_{e}",  hf_row[f"{e}_mmr_xuv"])

    # Set parameters from config provided by user
    params.kappa_p      = config.escape.kappa_p
    params.sigma_EUV    = config.escape.boreas.sigma_XUV
    params.alpha_rec    = config.escape.boreas.alpha_rec
    params.light_major  = config.escape.boreas.light_major
    params.heavy_major  = config.escape.boreas.heavy_major
    params.heavy_minor  = config.escape.boreas.heavy_minor
    params.eff          = config.escape.boreas.efficiency

    # Set parameters from atmosphere calculation
    params.Teq       = hf_row["T_obs"]              # K
    params.FEUV      = hf_row["F_xuv"] * 1e7        # XUV flux, converted to ergs cm-2 s-1
    params.rplanet   = hf_row["R_obs"] * 1e2        # convert m to cm
    params.mplanet   = hf_row["M_planet"] * 1e3     # convert kg to g

    # Initalise objects
    mass_loss       = bm.MassLoss(params)
    fractionation   = bm.Fractionation(params)

    # Run bulk mass loss calculation
    ml_result = mass_loss.compute_mass_loss_parameters()

    # Run fractionation calculation
    fr_result = fractionation.execute_fractionation(mass_loss, ml_result)

    # Store bulk outputs
    hf_row["esc_rate_total"] = fr_result["Mdot"]  * 1e-3    # g/s   ->  kg/s
    hf_row["R_xuv"]          = fr_result["REUV"]  * 1e-2    # cm    ->  m
    hf_row["P_xuv"]          = fr_result["PEUV"] ; raise  # check this
    hf_row["cs_xuv"]         = fr_result["cs"]    * 1e-2    # cm/s  ->  m/s

    # Convert escape fluxes to rates, and store
    for e in element_list:
        # default is zero
        key = "Phi_"+e
        hf_row["esc_rate_"+e] = 0.0

        # set escape rate if we have result from BOREAS
        if key in fr_result.keys():
            # convert g/cm2/s  ->  kg/m^2/s
            flx = fr_result[key] * 10

            # get global rate [kg/s] from flux through Rxuv
            hf_row["esc_rate_O"] = flx * 4 * np.pi * (hf_row["R_xuv"])**2

    # Get fractionation factors with respect to `light_major`
    # x_O/C/N        dimensioness, wrt H
