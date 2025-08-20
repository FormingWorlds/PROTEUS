# Functions used to run the BOREAS escape module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

import boreas
from proteus.utils.constants import element_list
from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Supported gases
BOREAS_GASES = ("H2O", "H2" , "O2" , "CO2", "CO" , "CH4", "N2" , "NH3")
BOREAS_ELEMS = ('H','C','O','N')

def run_boreas(config:Config, hf_row:dict):
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
    """

    log.info("Running fractionated escape (BOREAS)...")

    # Set parameters for escape
    params          = boreas.ModelParams()

    # Set gas MASS mixing ratios from VOLUME mixing ratios
    #    first, get MMW of relevant gases at this layer
    mmw_xuv = 0.0
    for g in BOREAS_GASES:
        mmw_xuv += hf_row[g+"_xuv"] * eval_gas_mmw(g)
    for g in BOREAS_GASES:
        mmr = hf_row[g+"_xuv"] * eval_gas_mmw(g) / mmw_xuv
        setattr(params, "X_"+g, mmr)

    # Set parameters from config provided by user
    for g in BOREAS_GASES:
        params.kappa[g] = getattr(config.escape.boreas,"kappa_"+g)
    params.sigma_EUV    = config.escape.boreas.sigma_XUV
    params.alpha_rec    = config.escape.boreas.alpha_rec
    params.eff          = config.escape.boreas.efficiency

    # Set parameters from atmosphere calculation
    params.Teq       = hf_row["T_obs"]              # K
    params.FEUV      = hf_row["F_xuv"] * 1e3        # XUV flux, converted to ergs cm-2 s-1
    params.rplanet   = hf_row["R_obs"] * 1e2        # convert m to cm
    params.mplanet   = hf_row["M_planet"] * 1e3     # convert kg to g

    # Initalise objects
    mass_loss       = boreas.MassLoss(params)
    fractionation   = boreas.Fractionation(params)

    # Run bulk mass loss calculation
    ml_result = mass_loss.compute_mass_loss_parameters(
                    [params.mplanet], [params.rplanet], [params.Teq])

    # Run fractionation calculation
    fr_result = fractionation.execute(ml_result, mass_loss)[0]

    # Store bulk outputs (rate, sound speed, escape level)
    hf_row["esc_rate_total"] = fr_result["Mdot"]  * 1e-3    # g/s   ->  kg/s
    hf_row["cs_xuv"]         = fr_result["cs"]    * 1e-2    # cm/s  ->  m/s
    hf_row["R_xuv"]          = fr_result["REUV"]  * 1e-2    # cm    ->  m
    hf_row["p_xuv"]          = 0.0  # to be calc'd by atmosphere module

    # Convert escape fluxes to rates, and store
    for e in element_list:
        if e in BOREAS_ELEMS:
            # convert g/cm2/s  ->  kg/m^2/s
            flx = fr_result["phi_"+e] * 10
            log.debug(f"Escape flux of {e}: {flx:.3e} kg m-2 s-1")

            # get global rate [kg/s] from flux through Rxuv
            hf_row["esc_rate_"+e] = flx * 4 * np.pi * (hf_row["R_xuv"])**2

        else:
            hf_row["esc_rate_"+e] = 0.0

    # Print info to user
    regime_map = {"RL":"recomb-limited", "EL":"energy-limited", "DL":"diffusion-limited"}
    log.info("Escape regime is "+regime_map[fr_result['regime']])
    log.info("Fractionation coefficients:")
    for e in ('O','C','N'):
        log.info(f"    {e:2s} = {fr_result['x_'+e]:.6f}")

