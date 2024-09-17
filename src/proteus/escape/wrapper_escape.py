# Functions used to handle escape
from __future__ import annotations

import logging

import mors
import numpy as np

from proteus.utils.constants import element_list, ergcm2stoWm2, secs_per_year
from proteus.utils.helper import PrintHalfSeparator
from zephyrus.escape import EL_escape

log = logging.getLogger("fwl."+__name__)

# Define global variables
Fxuv_star_SI_full = None
age_star_mors = None


def RunEscape(config, hf_row, dt, solvevol_target):
    """Run Escape submodule.

    Generic function to run escape calculation using ZEPHYRUS or dummy.
    Writes into the hf_row generic and solvevol_target variable passed as an arguement.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time interval over which escape is occuring [yr]
        solvevol_target : dict
            Dictionary of elemental mass inventories [kg]
    """

    PrintHalfSeparator()

    if self.config["escape_model"] == 0:
        pass

    elif self.config["escape_model"] == 1:
        hf_row["esc_rate_total"] = RunZEPHYRUS(config, hf_row)

    elif self.config["escape_model"] == 2:
        hf_row["esc_rate_total"] = config["escape_dummy_rate"]

    else:
        raise Exception("Invalid escape model")

    if (config["escape_model"] ==1 or config["escape_model"] ==2):
        log.info(
                "Bulk escape rate: %.2e kg yr-1 = %.2e kg s-1"
                % (hf_row["esc_rate_total"] * secs_per_year, hf_row["esc_rate_total"])
                )

        solvevol_target = SpeciesEscapeFromTotalEscape(hf_row, dt)

    return

def RunZEPHYRUS(config, hf_row):
    """Run energy-limited escape (for now) model.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only

    Returns
    -------
        mlr : float
            Bulk escape rate [kg s-1]
    """
    global Fxuv_star_SI_full
    global age_star_mors

    log.info("Running EL escape (ZEPHYRUS) ...")

    ## Load stellar evolution track + compute EL escape
    log.info("Load stellar evolution track + compute EL escape ")

    # Get the age of the star at time t to compute XUV flux at that time
    age_star = hf_row["age_star"] # [years]

    if (Fxuv_star_SI_full is None):
        star = mors.Star(Mstar=config["star_mass"],
                         Age=age_star/1e6,
                         Omega=config["star_omega"])
        age_star_mors = star.Tracks['Age']
        Fxuv_star_SI_full = ((star.Tracks['Lx'] + star.Tracks['Leuv'])
                             / (4 * np.pi * (semi_major_axis * 1e2)**2)) * ergcm2stoWm2

    # Interpolating the XUV flux at the age of the star
    Fxuv_star_SI = np.interp(age_star, age_star_mors * 1e6, Fxuv_star_SI_full)
    log.info(f"Interpolated Fxuv_star_SI at age_star = {age_star} years is {Fxuv_star_SI}")

    # Compute energy-limited escape
    mlr = EL_escape(config["escape_el_tidal_correction"], #tidal contribution (True/False)
                    config["mean_distance"] * AU, #planetary semi-major axis [m]
                    config["eccentricity"], #eccentricity
                    hf_row["M_planet"], #planetary mass [kg]
                    config["star_mass"], #stellar mass [kg]
                    config["efficiency_factor"], #efficiency factor
                    hf_row["R_planet"], #planetary radius [m]
                    hf_row["R_planet"], #XUV optically thick planetary radius [m]
                    Fxuv_star_SI)   # [kg s-1]

    log.info('Zephyrus escape computation done :)')

    return mlr

def SpeciesEscapeFromTotalEscape(hf_row:dict, dt:float):

    out = {}
    esc = hf_row["esc_rate_total"]

    # calculate total mass of volatiles (except oxygen, which is set by fO2)
    M_vols = 0.0
    for e in element_list:
        if e=='O':
            continue
        M_vols += hf_row[e+"_kg_total"]

    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio

    for e in element_list:
        if e=='O':
            continue

        # current elemental mass ratio in total
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new total mass of element e, keeping a constant mixing ratio of that element
        out[e] = emr * (M_vols - esc * dt * secs_per_year)

        # do not allow negative masses
        out[e] = max(0.0, out[e])

    return out