# Generic stellar evolution wrapper
from __future__ import annotations

import logging
import os
import numpy as np
from typing import TYPE_CHECKING

from proteus.utils.constants import AU, R_sun, L_sun, const_sigma

log = logging.getLogger("fwl."+__name__)

if TYPE_CHECKING:
    from proteus.config import Config

def scale_spectrum_to_toa(fl_arr, sep:float):
    '''
    Scale stellar fluxes from 1 AU to top of atmosphere
    '''
    return np.array(fl_arr) * ( (AU / sep)**2 )

def write_spectrum(wl_arr, fl_arr, hf_row:dict, output_dir:str):
    '''
    Write stellar spectrum to file.
    '''

    # Header information
    header = (
        "# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = %.2e yr"
        % hf_row["age_star"]
    )

    # Write to TSV file
    np.savetxt(
        os.path.join(output_dir, "data", "%d.sflux" % hf_row["Time"]),
        np.array([wl_arr, fl_arr]).T,
        header=header,
        comments="",
        fmt="%.8e",
        delimiter="\t",
    )

def update_stellar_radius(hf_row:dict, config:Config, baraffe=None):
    '''
    Update stellar radius in hf_row, stored in SI units.
    '''

    # Dummy case
    if config.outgas.module == 'dummy':
        R_star = config.star.radius

    # Mors cases
    elif config.outgas.module == 'mors':

        import mors

        # which track?
        match config.star.mors.tracks:
            case 'spada':
                R_star = mors.Value(config.star.mass, hf_row["age_star"] / 1e6, "Rstar")
            case 'baraffe':
                R_star = baraffe.BaraffeStellarRadius(hf_row["age_star"])

    # Dimensionalise and store in dictionary
    hf_row["R_star"] = R_star * R_sun

def update_instellation(hf_row:dict, config:Config, baraffe_track=None):
    '''
    Update hf_row value of bolometric stellar flux impinging upon the planet.
    '''

    # Dummy case
    if config.outgas.module == 'dummy':
        from proteus.star.dummy import calc_instellation
        S_0 = calc_instellation(config.star.Teff, hf_row["R_star"], hf_row["separation"])

    # Mors cases
    elif config.outgas.module == 'mors':

        import mors

        # which track?
        match config.star.mors.tracks:
            case 'spada':
                S_0 = mors.Value(config.star.mass, hf_row["age_star"] / 1e6, "Lbol") \
                        * L_sun  / (4.0 * np.pi * hf_row["separation"]**2.0 )

            case 'baraffe':
                S_0 = baraffe_track.BaraffeSolarConstant(hf_row["age_star"],
                                                    hf_row["separation"]/AU)

    # Update hf_row dictionary
    hf_row["F_ins"] = S_0

def update_equilibrium_temperature(hf_row:dict, config:Config):
    '''
    Calculate planetary equilibrium temperature.
    '''

    # Absorbed stellar flux
    F_asf = hf_row["F_ins"] * config.orbit.s0_factor * (1-config.atmos_clim.albedo_pl)

    # Planetary equilibrium temperature
    hf_row["T_eqm"] = (F_asf / const_sigma)**(1.0/4.0)
