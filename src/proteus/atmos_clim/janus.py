# Functions used to handle atmosphere thermodynamics (running JANUS, etc.)
from __future__ import annotations

import glob
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd

from proteus.utils.constants import volatile_species
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder, find_nearest
from proteus.utils.logs import StreamToLogger

log = logging.getLogger("fwl."+__name__)

# Generate atmosphere from input files
def StructAtm( dirs:dict, hf_row:dict, OPTIONS:dict ):

    from janus.utils import ReadBandEdges, atmos

    # Create atmosphere object and set parameters
    pl_radius = hf_row["R_planet"]
    pl_mass   = hf_row["M_planet"]

    vol_list = {}
    for vol in volatile_species:
        vol_list[vol] = hf_row[vol+"_vmr"]

    match OPTIONS["tropopause"]:
        case 0:
            trppT = 0.0  # none
        case 1:
            trppT = hf_row["T_skin"]  # skin temperature (grey stratosphere)
        case 2:
            trppT = OPTIONS["min_temperature"]  # dynamically, based on heating rate
        case _:
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid tropopause option '%d'" % OPTIONS["tropopause"])

    # Spectral bands
    band_edges = ReadBandEdges(dirs["output"]+"star.sf")

    # Cloud properties
    re   = 1.0e-5 # Effective radius of the droplets [m] (drizzle forms above 20 microns)
    lwm  = 0.8    # Liquid water mass fraction [kg/kg] - how much liquid vs. gas is there upon cloud formation? 0 : saturated water vapor does not turn liquid ; 1 : the entire mass of the cell contributes to the cloud
    clfr = 0.8    # Water cloud fraction - how much of the current cell turns into cloud? 0 : clear sky cell ; 1 : the cloud takes over the entire area of the cell (just leave at 1 for 1D runs)
    do_cloud = bool(OPTIONS["water_cloud"] == 1)
    alpha_cloud = float(OPTIONS["alpha_cloud"])

    # Make object
    atm = atmos(hf_row["T_surf"], hf_row["P_surf"]*1e5,
                OPTIONS["P_top"]*1e5, pl_radius, pl_mass,
                band_edges,
                vol_mixing=vol_list,
                minT = OPTIONS["min_temperature"],
                maxT = OPTIONS["max_temperature"],
                trppT=trppT,
                water_lookup=False,
                req_levels=OPTIONS["atmosphere_nlev"], alpha_cloud=alpha_cloud,
                re=re, lwm=lwm, clfr=clfr, do_cloud=do_cloud
                )

    atm.zenith_angle    = OPTIONS["zenith_angle"]
    atm.albedo_pl       = OPTIONS["albedo_pl"]
    atm.inst_sf         = OPTIONS["asf_scalefactor"]
    atm.albedo_s        = OPTIONS["albedo_s"]
    atm.skin_d          = OPTIONS["skin_d"]
    atm.skin_k          = OPTIONS["skin_k"]

    atm.instellation    = hf_row["F_ins"]
    atm.tmp_magma       = hf_row["T_magma"]

    return atm

def RunJANUS( atm, time:float, dirs:dict, OPTIONS:dict, hf_all:pd.DataFrame,
             write_in_tmp_dir=True, search_method=0, rtol=1.0e-4):
    """Run JANUS.

    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. Limits flux
    change if required.

    Parameters
    ----------
        atm : atmos
            Atmosphere object
        time : float
            Model time [yrs]
        dirs : dict
            Dictionary containing paths to directories
        OPTIONS : dict
            Configuration options and other variables
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

    # Runtime info
    log.info("Running JANUS...")

    output={}

    # Change dir
    cwd = os.getcwd()
    tmp_dir = dirs["output"]
    if write_in_tmp_dir:
        tmp_dir = create_tmp_folder()
    log.debug("Will run socrates inside '%s'"%tmp_dir)
    os.chdir(tmp_dir)

    # Prepare to calculate temperature structure w/ General Adiabat
    trppD = bool(OPTIONS["tropopause"] == 2 )
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
                        atm_bc=int(OPTIONS["F_atm_bc"]), T_surf_guess=float(T_surf_old)-0.5, T_surf_max=float(T_surf_max))

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

    # find 1 mbar level
    idx = find_nearest(atm.p*1e5, 1e-3)[1]
    z_obs = atm.z[idx]

    output["T_surf"] = atm.ts            # Surface temperature [K]
    output["F_atm"]  = F_atm_lim         # Net flux at TOA
    output["F_olr"]  = atm.LW_flux_up[0] # OLR
    output["F_sct"]  = atm.SW_flux_up[0] # Scattered SW flux
    output["z_obs"]  = z_obs + atm.planet_radius

    return output
