# Functions used to run the BOREAS escape module
from __future__ import annotations

import logging
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import boreas
import numpy as np

from proteus.escape.common import calc_unfract_fluxes
from proteus.utils.constants import AU, element_list, gas_list
from proteus.utils.helper import UpdateStatusfile, eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)
log.write = lambda msg: log.info(msg.rstrip('\n')) if msg != '\n' else None # print() redirect

# Get set of shared-supported gases
BOREAS_GASES = set(boreas.ModelParams().kappa.keys()) & set(gas_list)

# Get set of shared-supported elements
BOREAS_ELEMS = set(boreas.ModelParams().sigma_XUV.keys()) & set(element_list)

def run_boreas(config:Config, hf_row:dict, dirs:dict):
    """Run BOREAS escape model.

    Calculates the mass loss rate of each element.
    Updates the quantities in hf_row as appropriate, including R_xuv.
    P_xuv must then be calculated from R_xuv as required.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dirs: dict
            Dictionary of directories.
    """

    log.info("Running escape...")

    # Set parameters for escape
    params          = boreas.ModelParams()

    # These should not matter, but set them to be sure
    params.albedo   = 0.0
    params.beta     = 1.0
    params.epsilon  = 1.0
    params.aplau    = hf_row["semimajorax"] / AU

    # Set parameters from config provided by user
    for g in BOREAS_GASES:
        params.kappa[g] = getattr(config.escape.boreas,f"kappa_{g}")
    for e in BOREAS_ELEMS:
        params.sigma_XUV[e] = getattr(config.escape.boreas,f"sigma_{e}")
    params.eff = config.escape.boreas.efficiency

    # Set gas MASS mixing ratios from VOLUME mixing ratios
    #    first, get MMW of relevant gases at this layer
    mmw_xuv = 0.0
    for g in BOREAS_GASES:
        mmw_xuv += hf_row[f"{g}_vmr_xuv"] * eval_gas_mmw(g)
    for g in BOREAS_GASES:
        mmr = hf_row[f"{g}_vmr_xuv"] * eval_gas_mmw(g) / mmw_xuv
        setattr(params, f"X_{g}", mmr)

    # Set parameters from atmosphere calculation
    params.Teq       = hf_row["T_obs"]              # K
    params.FXUV      = hf_row["F_xuv"] * 1e3        # XUV flux, converted to ergs cm-2 s-1
    params.rplanet   = hf_row["R_obs"] * 1e2        # convert m to cm
    params.mplanet   = hf_row["M_planet"] * 1e3     # convert kg to g
    params.mmw_outflow_eff = None

    # Finalise parameters
    params._recompute_composites()
    params._init_opacities()

    # Do escape computation with BOREAS
    # Redirect print() calls inside BOREAS to -> PROTEUS' log.info()
    with redirect_stdout(log):
        try:
            mass_loss       = boreas.MassLoss(params)
            fractionation   = boreas.Fractionation(params)

            # Run bulk mass loss calculation
            ml_result = mass_loss.compute_mass_loss_parameters(
                            [params.mplanet], [params.rplanet], [params.Teq])

            # Run fractionation calculation
            fr_result = fractionation.execute(ml_result, mass_loss)[0]

        # Safely capture errors
        except Exception as e:
            UpdateStatusfile(dirs, 28)
            log.error(e)
            raise RuntimeError("Encountered problem when running BOREAS module") from e

    # Print info
    regime_map = {"RL":"recomb-limited", "EL":"energy-limited", "DL":"diffusion-limited"}
    log.info("Escape regime is "+regime_map[fr_result['regime']])

    # Store bulk outputs (rate, sound speed, escape level)
    hf_row["esc_rate_total"] = fr_result["Mdot"]  * 1e-3    # g/s   ->  kg/s
    hf_row["cs_xuv"]         = fr_result["cs"]    * 1e-2    # cm/s  ->  m/s
    hf_row["R_xuv"]          = fr_result["RXUV"]  * 1e-2    # cm    ->  m
    hf_row["p_xuv"]          = 0.0  # to be calc'd by atmosphere module

    # If not doing fractionation, overwrite fluxes...
    if not config.escape.boreas.fractionate:
        calc_unfract_fluxes(hf_row, reservoir=config.escape.reservoir,
                                    min_thresh=config.outgas.mass_thresh)
        return

    # If we ARE doing fractionation, parse results from BOREAS...

    # Convert escape fluxes to rates, and store
    for e in element_list:
        if e in BOREAS_ELEMS:
            # convert atoms/cm2/s  ->  kg/m^2/s
            flx = fr_result[f"phi_{e}_num"] * getattr(params, 'm_'+e) * 1e-3 * 100**2
            log.debug(f"Escape flux of {e}: {flx:.3e} kg m-2 s-1")

            # get global rate [kg/s] from flux through Rxuv
            hf_row["esc_rate_"+e] = flx * 4 * np.pi * (hf_row["R_xuv"])**2

        else:
            hf_row["esc_rate_"+e] = 0.0

    # Print info to user
    log.info("Fractionation coefficients:")
    for e in BOREAS_ELEMS:
        if e != 'H':
            log.info(f"    {e:2s} = {fr_result['x_'+e]:.6f}")
