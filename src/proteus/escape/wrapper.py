# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from zephyrus.escape import EL_escape

from proteus.utils.constants import AU, element_list, ergcm2stoWm2, secs_per_year
from proteus.utils.helper import PrintHalfSeparator

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def RunEscape(config:Config, hf_row:dict, dt:float, stellar_track):
    """Run Escape submodule.

    Generic function to run escape calculation using ZEPHYRUS or dummy.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time interval over which escape is occuring [yr]
        stellar_track : mors star object
            Mors star object storing spada track data.
    """

    PrintHalfSeparator()

    if not config.escape.module:
        # solvevol_target is undefined?
        pass
    elif config.escape.module == 'zephyrus':
        hf_row["esc_rate_total"] = RunZEPHYRUS(config, hf_row, stellar_track)
    elif config.escape.module == 'dummy':
        hf_row["esc_rate_total"] = config.escape.dummy.rate
    else:
        raise ValueError(f"Invalid escape model: {config.escape.module}")

    log.info(
        "Bulk escape rate: %.2e kg yr-1 = %.2e kg s-1"
        % (hf_row["esc_rate_total"] * secs_per_year, hf_row["esc_rate_total"])
        )

    solvevol_target = SpeciesEscapeFromTotalEscape(hf_row, dt)

    # store in hf_row as elements
    for e in element_list:
        if e == "O":
            continue
        hf_row[e + "_kg_total"] = solvevol_target[e]

def RunZEPHYRUS(config, hf_row, stellar_track):
    """Run energy-limited escape (for now) model.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        stellar_track : mors star object
            Mors star object storing spada track data.
    Returns
    -------
        mlr : float
            Bulk escape rate [kg s-1]
    """

    log.info("Running EL escape (ZEPHYRUS) ...")

    # Get the age of the star at time t to compute XUV flux at that time
    age_star = hf_row["age_star"] / 1e6 # [Myrs]

    # Interpolating the XUV flux at the age of the star
    Fxuv_star_SI = ((stellar_track.Value(age_star, 'Lx') + stellar_track.Value(age_star, 'Leuv'))
                             / (4 * np.pi * (config.orbit.semimajoraxis * AU * 1e2)**2)) * ergcm2stoWm2

    log.info(f"Interpolated Fxuv_star_SI at age_star = {age_star} Myr is {Fxuv_star_SI} [W/m2]")

    # Compute energy-limited escape
    mlr = EL_escape(config.escape.zephyrus.tidal, #tidal contribution (True/False)
                    config.orbit.semimajoraxis * AU, #planetary semi-major axis [m]
                    config.orbit.eccentricity, #eccentricity
                    hf_row["M_planet"], #planetary mass [kg]
                    config.star.mass, #stellar mass [kg]
                    config.escape.zephyrus.efficiency, #efficiency factor
                    hf_row["R_int"], #planetary radius [m]
                    hf_row["R_xuv"], #XUV optically thick planetary radius [m]
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
