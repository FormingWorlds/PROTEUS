# Generic atmosphere wrapper
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
from numpy import pi

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.atmos_clim.common import Atmos_t, get_spfile_path
from proteus.utils.constants import const_R
from proteus.utils.helper import UpdateStatusfile, safe_rm

log = logging.getLogger("fwl."+__name__)

def run_atmosphere(atmos_o:Atmos_t, config:Config, dirs:dict, loop_counter:dict,
                     wl: list, fl:list,
                     update_stellar_spectrum:bool, hf_all:pd.DataFrame, hf_row:dict):
    """Run Atmosphere submodule.

    Generic function to run an atmospheric simulation with either JANUS, AGNI or dummy.
    Writes into the hf_row generic variable passed as an arguement.

    Parameters
    ----------
        atmos_o: Atmos_t
            Atmosphere struct
        config : Config
            Configuration options and other variables
        dirs : dict
            Dictionary containing paths to directories
        loop_counter : dict
            Dictionary containing iteration information
        wl : list
            Wavelength [nm]
        fl : list
            Flux at 1 AU [erg s-1 cm-2 nm-1]
        update_stellar_spectrum : bool
            Spectral file path
        hf_all : pd.DataFrame
            Dataframe containing simulation variables (now and historic)
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    """

    log.info("Solving atmosphere...")

    # Warnings
    if config.atmos_clim.albedo_pl > 1.0e-9:
        if config.atmos_clim.rayleigh:
            log.warning("Physically inconsistent options selected: "
                        "`albedo_pl > 0` and `rayleigh = True`")
        if config.atmos_clim.cloud_enabled:
            log.warning("Physically inconsistent options selected: "
                        "`albedo_pl > 0` and `cloud_enabled = True`")

    # Handle new surface temperature
    if config.atmos_clim.surf_state == 'mixed_layer':
        hf_row["T_surf"] = ShallowMixedOceanLayer(hf_all.iloc[-1].to_dict(), hf_row)

    elif config.atmos_clim.surf_state == 'fixed':
        hf_row["T_surf"] = hf_row["T_magma"]

    # elif surf_state=='skin':
    #    Don't do anything here, because this will be handled by the atmosphere model.

    if config.atmos_clim.module == 'janus':
        # Import
        from proteus.atmos_clim.janus import InitAtm, InitStellarSpectrum, RunJANUS

        # Run JANUS
        no_atm = bool(atmos_o._atm is None)
        #Enforce surface temperature at first iteration
        if no_atm:
            hf_row["T_surf"] = hf_row["T_magma"]

        #Init atm object if first iteration or change in stellar spectrum
        if no_atm or update_stellar_spectrum:
            spectral_file_nostar = get_spfile_path(dirs["fwl"], config)
            if not os.path.exists(spectral_file_nostar):
                UpdateStatusfile(dirs, 20)
                raise FileNotFoundError(
                            "Spectral file does not exist at '%s'" % spectral_file_nostar)
            InitStellarSpectrum(dirs, wl, fl, spectral_file_nostar)
            atmos_o._atm = InitAtm(dirs, config)

        atm_output = RunJANUS(atmos_o._atm, dirs, config, hf_row, hf_all)

    elif config.atmos_clim.module == 'agni':
        # Import
        from proteus.atmos_clim.agni import (
            activate_julia,
            deallocate_atmos,
            init_agni_atmos,
            run_agni,
            update_agni_atmos,
        )

        # Run AGNI
        # Initialise atmosphere struct
        spfile_path = os.path.join(dirs["output"] , "runtime.sf")
        no_atm = bool(atmos_o._atm is None)
        if no_atm or update_stellar_spectrum:
            log.debug("Initialise new atmosphere struct")

            # first run?
            if no_atm:
                activate_julia(dirs)
                # surface temperature guess
                hf_row["T_surf"] = hf_row["T_magma"]
            else:
                # Remove old spectral file if it exists
                safe_rm(spfile_path)
                safe_rm(spfile_path+"_k")

                # deallocate old atmosphere
                deallocate_atmos(atmos_o._atm)

            # allocate new
            atmos_o._atm = init_agni_atmos(dirs, config, hf_row)

            # Check allocation was ok
            if not bool(atmos_o._atm.is_alloc):
                UpdateStatusfile(dirs, 22)
                raise RuntimeError("Atmosphere struct not allocated")

        # Check if atmosphere is transparent
        transparent = bool(hf_row["P_surf"] < config.atmos_clim.agni.psurf_thresh)  # bar

        # Update profile
        atmos_o._atm = update_agni_atmos(atmos_o._atm, hf_row, dirs, transparent)

        # Run solver
        atmos_o._atm, atm_output = run_agni(atmos_o._atm,
                                            loop_counter["total"], dirs, config, hf_row,
                                            transparent)

    elif config.atmos_clim.module == 'dummy':
        # Import
        from proteus.atmos_clim.dummy import RunDummyAtm
        # Run dummy atmosphere model
        atm_output = RunDummyAtm(dirs, config, hf_row["T_magma"], hf_row["F_ins"],
                                    hf_row["R_int"], hf_row["M_int"], hf_row["P_surf"])

    # Store atmosphere module output variables
    #    observables
    hf_row["rho_obs"]= atm_output["rho_obs"] # [kg m-3]
    hf_row["p_obs"]  = atm_output["p_obs"]   # [bar]
    hf_row["R_obs"]  = atm_output["R_obs"]   # [m]
    #    fluxes
    hf_row["F_atm"]  = atm_output["F_atm"]
    hf_row["F_olr"]  = atm_output["F_olr"]
    hf_row["F_sct"]  = atm_output["F_sct"]
    hf_row["F_net"]  = hf_row["F_int"] - hf_row["F_atm"]
    hf_row["bond_albedo"]= atm_output["albedo"]
    hf_row["T_surf"] = atm_output["T_surf"]
    #    escape
    hf_row["p_xuv"]  = atm_output["p_xuv"]    # Closest pressure to Pxuv    [bar]
    hf_row["R_xuv"]  = atm_output["R_xuv"]    # Radius at p_xuv [m]

    # Calculate bolometric observables (measured at infinite distance)
    update_bolometry(hf_row)

    # Estimate WTG parameter
    update_wtg_surf(hf_row)


def update_wtg_surf(hf_row:dict):
    '''
    Update WTG parameter.

    https://royalsocietypublishing.org/doi/full/10.1098/rspa.2016.0107
    '''

    omega = 2 * pi / hf_row["axial_period"]     # Angular rotation rate
    R_mix = const_R / hf_row["atm_kg_per_mol"]  # Specific gas constant
    hf_row["wtg_surf"] = (R_mix * hf_row["T_surf"])**0.5 / (omega * hf_row["R_int"])


def update_bolometry(hf_row:dict):
    '''
    Update bolometric observables (transit depth, contrast ratio.)

    https://link.springer.com/content/pdf/10.1007/978-3-319-30648-3_40-1.pdf
    '''

    # Transit depth
    hf_row["transit_depth"] =  (hf_row["R_obs"]  / hf_row["R_star"])**2.0

    # Eclipse depth
    #    Accounting for fact that F_ins is scaled to TOA, not to stellar surface.
    hf_row["eclipse_depth"] = ((hf_row["F_olr"]+hf_row["F_sct"])/hf_row["F_ins"]) * \
                                 (hf_row["R_obs"] / hf_row["separation"])**2.0

def ShallowMixedOceanLayer(hf_cur:dict, hf_pre:dict):
    # This scheme is not typically used, but it maintained here from legacy code
    from scipy.integrate import solve_ivp

    log.info(">>>>>>>>>> Flux convergence scheme <<<<<<<<<<<")

    # For SI conversion
    yr          = 3.154e+7      # s

    # Last T_surf and time from atmosphere, K
    t_cur  = hf_cur["Time"]*yr
    t_pre  = hf_pre["Time"]*yr
    Ts_pre = hf_pre["T_surf"]

    # Properties of the shallow mixed ocean layer
    c_p_layer   = 1000          # J kg-1 K-1
    rho_layer   = 3000          # kg m-3
    depth_layer = 1000          # m

    def ocean_evolution(t, y):
        # Specific heat of mixed ocean layer
        mu      = c_p_layer * rho_layer * depth_layer # J K-1 m-2
        # RHS of ODE
        RHS     = - hf_cur["F_net"] / mu
        return RHS

    # Solve ODE
    sol_curr  = solve_ivp(ocean_evolution, [t_pre, t_cur], [Ts_pre])

    # New current surface temperature from shallow mixed layer
    Ts_cur = sol_curr.y[0][-1] # K

    return Ts_cur
