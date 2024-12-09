# Generic atmosphere wrapper
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.atmos_clim.common import get_spfile_path
from proteus.utils.helper import PrintHalfSeparator, UpdateStatusfile, safe_rm

atm = None
log = logging.getLogger("fwl."+__name__)

def RunAtmosphere(config:Config, dirs:dict, loop_counter:dict,
                  wl: list, fl:list,
                  update_stellar_spectrum:bool, hf_all:pd.DataFrame, hf_row:dict):
    """Run Atmosphere submodule.

    Generic function to run an atmospheric simulation with either JANUS, AGNI or dummy.
    Writes into the hf_row generic variable passed as an arguement.

    Parameters
    ----------
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

    #Warning! Find a way to store atm object for AGNI
    global atm

    PrintHalfSeparator()

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
        no_atm = bool(atm is None)
        #Enforce surface temperature at first iteration
        if no_atm:
            hf_row["T_surf"] = hf_row["T_magma"]

        #Init atm object if first iteration or change in stellar spectrum
        if no_atm or update_stellar_spectrum:
            spectral_file_nostar = get_spfile_path(dirs["fwl"], config)
            if not os.path.exists(spectral_file_nostar):
                UpdateStatusfile(dirs, 20)
                raise Exception("Spectral file does not exist at '%s'" % spectral_file_nostar)
            InitStellarSpectrum(dirs, wl, fl, spectral_file_nostar)
            atm = InitAtm(dirs, config)

        atm_output = RunJANUS(atm, dirs, config, hf_row, hf_all)

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
        no_atm = bool(atm is None)
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
                deallocate_atmos(atm)

            # allocate new
            atm = init_agni_atmos(dirs, config, hf_row)

            # Check allocation was ok
            if not bool(atm.is_alloc):
                log.error("Failed to allocate atmos struct")
                UpdateStatusfile(dirs, 22)
                exit(1)

        # Update profile
        atm = update_agni_atmos(atm, hf_row, dirs)

        # Run solver
        atm, atm_output = run_agni(atm, loop_counter["total"], dirs, config, hf_row)

    elif config.atmos_clim.module == 'dummy':
        # Import
        from proteus.atmos_clim.dummy import RunDummyAtm
        # Run dummy atmosphere model
        atm_output = RunDummyAtm(dirs, config, hf_row["T_magma"], hf_row["F_ins"],
                                    hf_row["R_int"], hf_row["M_int"], hf_row["P_surf"])

    # Store atmosphere module output variables
    hf_row["rho_obs"]= atm_output["rho_obs"] # [kg m-3]
    hf_row["p_obs"]  = atm_output["p_obs"]   # [bar]
    hf_row["z_obs"]  = atm_output["z_obs"]   # [m]
    hf_row["F_atm"]  = atm_output["F_atm"]
    hf_row["F_olr"]  = atm_output["F_olr"]
    hf_row["F_sct"]  = atm_output["F_sct"]
    hf_row["T_surf"] = atm_output["T_surf"]
    hf_row["F_net"]  = hf_row["F_int"] - hf_row["F_atm"]
    hf_row["bond_albedo"]= atm_output["albedo"]
    hf_row["p_xuv"]  = atm_output["p_xuv"]                  # Closest pressure from Pxuv    [bar]
    hf_row["z_xuv"]  = atm_output["z_xuv"]                  # Height at Pxuv                [m]
    hf_row["R_xuv"]  = hf_row["R_int"] + hf_row["z_xuv"]    # Radius at z_xuv [m]

    # Calculate observables (measured at infinite distance)
    R_obs = hf_row["z_obs"] + hf_row["R_int"] # observed radius [m]
    hf_row["transit_depth"] =  (R_obs / hf_row["R_star"])**2.0
    hf_row["contrast_ratio"] = ((hf_row["F_olr"]+hf_row["F_sct"])/hf_row["F_ins"]) * \
                                 (R_obs / hf_row["separation"])**2.0

def ShallowMixedOceanLayer(hf_cur:dict, hf_pre:dict):

    # This scheme is not typically used, but it maintained here from legacy code
    # We could consider removing it in the future.

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
