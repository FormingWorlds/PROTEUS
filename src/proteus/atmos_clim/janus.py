# Functions used to handle atmosphere thermodynamics (running JANUS, etc.)
from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import TYPE_CHECKING

import janus.set_socrates_env  # noqa
import numpy as np
import pandas as pd

from proteus.atmos_clim.common import get_radius_from_pressure
from proteus.utils.constants import vap_list, vol_list
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def InitStellarSpectrum(dirs:dict, wl:list, fl:list, spectral_file_nostar):

    from janus.utils import InsertStellarSpectrum, PrepareStellarSpectrum

    log.debug("Prepare spectral file for JANUS")

    # Generate a new SOCRATES spectral file containing this new spectrum
    star_spec_src = dirs["output"]+"socrates_star.txt"

    # Spectral file stuff
    PrepareStellarSpectrum(wl,fl,star_spec_src)

    log.debug("Insert stellar spectrum into spectral file")
    InsertStellarSpectrum(spectral_file_nostar,
                          star_spec_src,
                          dirs["output"]
                          )
    os.remove(star_spec_src)

    return

def InitAtm(dirs:dict, config:Config):

    from janus.utils import ReadBandEdges, atmos

    log.debug("Create new JANUS atmosphere object")

    vol_dict = {}
    for vol in vol_list:
        vol_dict[vol] = 1.0/len(vol_list)

    # Spectral bands
    band_edges = ReadBandEdges(dirs["output"]+"star.sf")

    # Make object
    # The var flag indicates variable parameters
    # to be set at each PROTEUS iteration
    # through the routine UpdateStateAtm
    atm = atmos(0.0, #var
                1e5, #var
                config.atmos_clim.janus.p_top*1e5,
                6.371e6, #var
                5.972e24, #var
                band_edges,
                vol_mixing = vol_dict, #var
                req_levels = config.atmos_clim.janus.num_levels,
                water_lookup = False,
                alpha_cloud=config.atmos_clim.cloud_alpha,
                trppT = config.atmos_clim.tmp_minimum,
                minT = config.atmos_clim.tmp_minimum,
                maxT = config.atmos_clim.tmp_maximum,
                do_cloud = config.atmos_clim.cloud_enabled,
                re = 1.0e-5, # Effective radius of the droplets [m] (drizzle forms above 20 microns)
                lwm = 0.8, # Liquid water mass fraction [kg/kg]
                clfr = 0.8, # Water cloud fraction
                albedo_s = config.atmos_clim.surf_greyalbedo,
                albedo_pl = config.atmos_clim.albedo_pl,
                zenith_angle = config.orbit.zenith_angle,
                )

    atm.inst_sf = config.orbit.s0_factor
    atm.skin_d = config.atmos_clim.surface_d
    atm.skin_k = config.atmos_clim.surface_k

    match config.atmos_clim.janus.overlap_method:
        case "ro":
            atm.overlap_type = 2
        case "ee":
            atm.overlap_type = 4
        case "rorr":
            atm.overlap_type = 8
        case _:
            raise ValueError("Invalid overlap method selected for SOCRATES/JANUS!")

    return atm

def UpdateStateAtm(atm, hf_row:dict, tropopause):
    """UpdateStateAtm

    Update the atm object state with current iteration variables

    Parameters
    ----------
        atm : atmos
            Atmosphere object
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        tropopause
            Tropopause type (None, "skin", "dynamic")
    """

    atm.setSurfaceTemperature(hf_row["T_surf"])
    atm.setSurfacePressure(hf_row["P_surf"]*1e5)
    atm.setPlanetProperties(hf_row["R_int"], hf_row["M_int"])

    # Warn about rock vapours
    has_vapours = False
    for gas in vap_list:
        has_vapours = has_vapours or (hf_row[gas+"_vmr"] > 1e-5)
    if has_vapours:
        log.warning("Atmosphere contains rock vapours which will be neglected by JANUS")

    # Store volatiles only
    vol_mixing = {}
    for vol in vol_list:
        vol_mixing[vol] = hf_row[vol+"_vmr"]
    atm.setVolatiles(vol_mixing)

    atm.instellation = hf_row["F_ins"]
    atm.tmp_magma = hf_row["T_magma"]
    if tropopause == "skin":
        atm.trppT = hf_row["T_skin"]
    else:
        atm.trppT = 0.5
    log.debug("Setting stratosphere to %.2f K"%atm.trppT)

    return

def RunJANUS(atm, dirs:dict, config:Config, hf_row:dict, hf_all:pd.DataFrame,
             write_in_tmp_dir=True, search_method=0, rtol=1.0e-4):
    """Run JANUS.

    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. Limits flux
    change if required.

    Parameters
    ----------
        atm : atmos
            Atmosphere object
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        hf_all : pd.DataFrame
            Dataframe containing simulation variables (now and historic)
        write_in_tmp_dir : bool
            Write temporary files in a local folder within /tmp, rather than in the output folder
        search_method : int
            Root finding method used by JANUS
        rtol : float
            Relative tolerance on solution for root finding method
    Returns
    ----------
        atm : atmos
            Updated atmos object
        output : dict
            Output variables, as a dict

    """

    from janus.utils.observed_rho import calc_observed_rho

    # Runtime info
    log.debug("Running JANUS...")
    time = hf_row["Time"]


    #Update atmosphere with current variables
    UpdateStateAtm(atm, hf_row, config.atmos_clim.janus.tropopause)

    # Change dir
    cwd = os.getcwd()
    tmp_dir = dirs["output"]
    if write_in_tmp_dir:
        tmp_dir = create_tmp_folder()
    log.debug("Will run socrates inside '%s'"%tmp_dir)
    os.chdir(tmp_dir)

    # Prepare to calculate temperature structure w/ General Adiabat
    trppD = config.atmos_clim.janus.tropopause == 'dynamic'
    rscatter = config.atmos_clim.rayleigh

    # Run JANUS
    if config.atmos_clim.surf_state == 'fixed':  # fixed T_Surf
        from janus.modules import MCPA
        atm = MCPA(dirs, atm, False, trppD, rscatter)

    elif config.atmos_clim.surf_state == 'skin': # conductive lid
        from janus.modules import MCPA_CBL

        T_surf_max = -1
        T_surf_old = -1
        atol       = 1.0e-5

        # Done with initial loops
        if time > 0:

            # Get previous temperature as initial guess
            T_surf_old = hf_all.iloc[-1]["T_surf"]

            # Prevent heating of the interior
            if config.atmos_clim.prevent_warming:
                T_surf_max = T_surf_old

            # calculate tolerance
            tol = rtol * abs(hf_all.iloc[-1]["F_atm"]) + atol
        else:
            tol = 0.1

        # run JANUS
        atm = MCPA_CBL(dirs, atm, trppD, rscatter, method=search_method, atol=tol,
                        atm_bc=int(config.atmos_clim.janus.F_atm_bc), T_surf_guess=float(T_surf_old)-1.0, T_surf_max=float(T_surf_max))

    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid surface state chosen for JANUS")

    # Clean up run directory
    for file in glob.glob(tmp_dir+"/current??.????"):
        os.remove(file)
    for file in glob.glob(tmp_dir+"/profile.*"):
        os.remove(file)
    os.chdir(cwd)
    if write_in_tmp_dir:
        shutil.rmtree(tmp_dir,ignore_errors=True)

    any_cloud = np.any(np.array(atm.clfr) > 1.0e-20)
    log.info("Water clouds have formed = %s"%(str(any_cloud)))
    log.info("SOCRATES fluxes (net@surf, net@TOA, OLR): %.5e, %.5e, %.5e W m-2" %
             (atm.net_flux[-1], atm.net_flux[0] , atm.LW_flux_up[0]))

    # Save atm data to disk
    nc_fpath = dirs["output"]+"/data/"+str(int(time))+"_atm.nc"
    atm.write_ncdf(nc_fpath)

    # Check for NaNs
    if not np.isfinite(atm.net_flux).all():
        UpdateStatusfile(dirs, 22)
        raise Exception("JANUS output array contains NaN or Inf values")

    # Store new flux
    if (config.atmos_clim.janus.F_atm_bc == 0):
        F_atm_new = atm.net_flux[0]
    else:
        F_atm_new = atm.net_flux[-1]

    # Require that the net flux must be upward
    F_atm_lim = F_atm_new
    if config.atmos_clim.prevent_warming:
        F_atm_lim = max( 1.0e-8 , F_atm_new )

    # Print if a limit was applied
    if not np.isclose(F_atm_lim , F_atm_new ):
        log.warning("Change in F_atm [W m-2] limited in this step!")
        log.warning("    %g  ->  %g" % (F_atm_new , F_atm_lim))

    # observables
    p_obs = float(config.atmos_clim.janus.p_obs)*1e5 # converted to Pa
    r_arr = np.array(atm.z[:]) + hf_row["R_int"]
    rho_obs = -1.0
    if atm.height_error:
        log.error("Hydrostatic integration failed in JANUS!")
    else:
        # find observed level [m] at p ~ p_obs
        _, r_obs = get_radius_from_pressure(atm.p, r_arr, p_obs)

        # calc observed density [kg m-3]
        rho_obs = calc_observed_rho(atm)

    # XUV height in atm
    if config.escape.module == 'zephyrus':
        # escape level set by zephyrus config
        p_xuv = config.escape.zephyrus.Pxuv # [bar]
    else:
        # escape level set to surface
        p_xuv = hf_row["P_surf"] # [bar]
    p_xuv, r_xuv = get_radius_from_pressure(atm.p, r_arr, p_xuv*1e5) # [Pa], [m]

    # final things to store
    output={}
    output["T_surf"] = atm.ts            # Surface temperature [K]
    output["F_atm"]  = F_atm_lim         # Net flux at TOA
    output["F_olr"]  = atm.LW_flux_up[0] # OLR
    output["F_sct"]  = atm.SW_flux_up[0] # Scattered SW flux
    output["albedo"] = atm.SW_flux_up[0] / atm.SW_flux_down[0]
    output["p_obs"]  = p_obs/1e5        # observed level [bar]
    output["R_obs"]  = r_obs            # observed level [m]
    output["rho_obs"]= rho_obs          # observed density [kg m-3]
    output["p_xuv"]  = p_xuv/1e5        # Closest pressure from Pxuv    [bar]
    output["R_xuv"]  = r_xuv            # Radius at Pxuv                [m]

    return output
