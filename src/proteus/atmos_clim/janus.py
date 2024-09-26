# Functions used to handle atmosphere thermodynamics (running JANUS, etc.)
from __future__ import annotations

import glob
import logging
import os
import shutil

import janus.set_socrates_env  # noqa
import numpy as np
import pandas as pd
import netCDF4 as nc

from proteus.utils.constants import volatile_species
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder, find_nearest

log = logging.getLogger("fwl."+__name__)

def read_ncdf_profile(nc_fpath:str):
    # open file
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables["p"][:])
    pl = np.array(ds.variables["pl"][:])

    t = np.array(ds.variables["tmp"][:])
    tl = np.array(ds.variables["tmpl"][:])

    z = np.array(ds.variables["z"][:])
    zl = np.array(ds.variables["zl"][:])

    nlev = len(p)

    # read pressure, temperature, height data into dictionary values
    out = {}
    out["p"] = [pl[0]]
    out["t"] = [tl[0]]
    out["z"] = [zl[0]]
    for i in range(nlev):
        out["p"].append(p[i])
        out["p"].append(pl[i+1])

        out["t"].append(t[i])
        out["t"].append(tl[i+1])

        out["z"].append(z[i])
        out["z"].append(zl[i+1])

    # close file
    ds.close()

    # convert to np arrays
    for k in out.keys():
        out[k] = np.array(out[k], dtype=float)

    return out

def read_ncdfs(output_dir:str, times:list):
    return [read_ncdf_profile(os.path.join(output_dir, "data", "%d_atm.nc"%t)) for t in times]

def InitStellarSpectrum(dirs:dict, wl:list, fl:list, spectral_file_nostar):

    from janus.utils import InsertStellarSpectrum, PrepareStellarSpectrum

    # Generate a new SOCRATES spectral file containing this new spectrum
    star_spec_src = dirs["output"]+"socrates_star.txt"

    # Spectral file stuff
    PrepareStellarSpectrum(wl,fl,star_spec_src)
    InsertStellarSpectrum(spectral_file_nostar,
                          star_spec_src,
                          dirs["output"]
                          )
    os.remove(star_spec_src)

    return

def InitAtm(dirs:dict, OPTIONS:dict):

    from janus.utils import ReadBandEdges, atmos

    vol_list = {}
    for vol in volatile_species:
        vol_list[vol] = 1.0/len(volatile_species)

    match OPTIONS["tropopause"]:
        case 0 | 1: # 0: none 1: skin temperature set in UpdateStateAtm
            trppT = 0.0
        case 2: # dynamically, based on heating rate
            trppT = OPTIONS["min_temperature"]
        case _:
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid tropopause option '%d'" % OPTIONS["tropopause"])

    # Spectral bands
    band_edges = ReadBandEdges(dirs["output"]+"star.sf")

    # Make object
    # The var flag indicates variable parameters
    # to be set at each PROTEUS iteration
    # through the routine UpdateStateAtm
    atm = atmos(0.0, #var
                1e5, #var
                OPTIONS["P_top"]*1e5,
                6.371e6, #var
                5.972e24, #var
                band_edges,
                vol_mixing = vol_list, #var
                req_levels = OPTIONS["atmosphere_nlev"],
                water_lookup = False,
                alpha_cloud=float(OPTIONS["alpha_cloud"]),
                trppT = trppT, #var if tropopause option is set to 1
                minT = OPTIONS["min_temperature"],
                maxT = OPTIONS["max_temperature"],
                do_cloud = bool(OPTIONS["water_cloud"] == 1),
                re = 1.0e-5, # Effective radius of the droplets [m] (drizzle forms above 20 microns)
                lwm = 0.8, # Liquid water mass fraction [kg/kg]
                clfr = 0.8, # Water cloud fraction
                albedo_s = OPTIONS["albedo_s"],
                albedo_pl = OPTIONS["albedo_pl"],
                zenith_angle = OPTIONS["zenith_angle"],
                )

    atm.inst_sf = OPTIONS["asf_scalefactor"]
    atm.skin_d = OPTIONS["skin_d"]
    atm.skin_k = OPTIONS["skin_k"]

    return atm

def UpdateStateAtm(atm, hf_row:dict, trppT:int):
    """UpdateStateAtm

    Update the atm object state with current iteration variables

    Parameters
    ----------
        atm : atmos
            Atmosphere object
        hf_row : dict
            Dictionary containing simulation variables for current iteration
    """

    atm.setSurfaceTemperature(hf_row["T_surf"])
    atm.setSurfacePressure(hf_row["P_surf"]*1e5)
    atm.setPlanetProperties(hf_row["R_planet"], hf_row["M_planet"])

    vol_mixing = {}
    for vol in volatile_species:
        vol_mixing[vol] = hf_row[vol+"_vmr"]
    atm.setVolatiles(vol_mixing)

    atm.instellation = hf_row["F_ins"]
    atm.tmp_magma = hf_row["T_magma"]
    if (trppT == 1):
        atm.trppT = hf_row["T_skin"]
        log.debug("Setting stratosphere to T_skin=%.2f K"%atm.trppT)

    return

def RunJANUS(atm, dirs:dict, OPTIONS:dict, hf_row:dict, hf_all:pd.DataFrame,
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
        OPTIONS : dict
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
    log.info("Running JANUS...")
    time = hf_row["Time"]


    #Update atmosphere with current variables
    UpdateStateAtm(atm, hf_row, OPTIONS["tropopause"])

    # Change dir
    cwd = os.getcwd()
    tmp_dir = dirs["output"]
    if write_in_tmp_dir:
        tmp_dir = create_tmp_folder()
    log.debug("Will run socrates inside '%s'"%tmp_dir)
    os.chdir(tmp_dir)

    # Prepare to calculate temperature structure w/ General Adiabat
    trppD = bool(OPTIONS["tropopause"] == 2)
    rscatter = bool(OPTIONS["rayleigh"] == 1)

    # Run JANUS
    if OPTIONS["atmosphere_surf_state"] == 1:  # fixed T_Surf
        from janus.modules import MCPA
        atm = MCPA(dirs, atm, False, trppD, rscatter)

    elif OPTIONS["atmosphere_surf_state"] == 2: # conductive lid
        from janus.modules import MCPA_CBL

        T_surf_max = -1
        T_surf_old = -1
        atol       = 1.0e-5

        # Done with initial loops
        if time > 0:

            # Get previous temperature as initial guess
            T_surf_old = hf_all.iloc[-1]["T_surf"]

            # Prevent heating of the interior
            if (OPTIONS["prevent_warming"] == 1):
                T_surf_max = T_surf_old

            # calculate tolerance
            tol = rtol * abs(hf_all.iloc[-1]["F_atm"]) + atol
        else:
            tol = 0.1

        # run JANUS
        atm = MCPA_CBL(dirs, atm, trppD, rscatter, method=search_method, atol=tol,
                        atm_bc=int(OPTIONS["F_atm_bc"]), T_surf_guess=float(T_surf_old)-1.0, T_surf_max=float(T_surf_max))

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
    if (OPTIONS["F_atm_bc"] == 0):
        F_atm_new = atm.net_flux[0]
    else:
        F_atm_new = atm.net_flux[-1]

    # Require that the net flux must be upward
    F_atm_lim = F_atm_new
    if (OPTIONS["prevent_warming"] == 1):
        F_atm_lim = max( 1.0e-8 , F_atm_new )

    # Print if a limit was applied
    if not np.isclose(F_atm_lim , F_atm_new ):
        log.warning("Change in F_atm [W m-2] limited in this step!")
        log.warning("    %g  ->  %g" % (F_atm_new , F_atm_lim))

    # observables
    z_obs = -1.0
    rho_obs = -1.0
    if not atm.height_error:
        # find 1 mbar level
        idx = find_nearest(atm.p, 1e2)[1]
        z_obs = atm.z[idx] + atm.planet_radius
        # calc observed density
        rho_obs = calc_observed_rho(atm)

    # final things to store
    output={}
    output["T_surf"] = atm.ts            # Surface temperature [K]
    output["F_atm"]  = F_atm_lim         # Net flux at TOA
    output["F_olr"]  = atm.LW_flux_up[0] # OLR
    output["F_sct"]  = atm.SW_flux_up[0] # Scattered SW flux
    output["z_obs"]  = z_obs
    output["rho_obs"]= rho_obs

    return output
