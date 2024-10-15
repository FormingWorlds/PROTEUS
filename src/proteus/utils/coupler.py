# Functions used to help run PROTEUS which are mostly submodule agnostic.

# Import utils-specific modules
from __future__ import annotations

import glob
import logging
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.atmos_clim.common import read_ncdfs
from proteus.interior.spider import read_jsons
from proteus.plot.cpl_atmosphere import plot_atmosphere
from proteus.plot.cpl_elements import plot_elements
from proteus.plot.cpl_emission import plot_emission
from proteus.plot.cpl_escape import plot_escape
from proteus.plot.cpl_fluxes_atmosphere import plot_fluxes_atmosphere
from proteus.plot.cpl_fluxes_global import plot_fluxes_global
from proteus.plot.cpl_global import plot_global
from proteus.plot.cpl_interior import plot_interior
from proteus.plot.cpl_interior_cmesh import plot_interior_cmesh
from proteus.plot.cpl_observables import plot_observables
from proteus.plot.cpl_sflux import plot_sflux
from proteus.plot.cpl_sflux_cross import plot_sflux_cross
from proteus.plot.cpl_stacked import plot_stacked
from proteus.utils.constants import (
    const_sigma,
    element_list,
    volatile_species,
)
from proteus.utils.helper import UpdateStatusfile, get_proteus_dir, safe_rm
from proteus.utils.plot import sample_times

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)


def GitRevision(dir:str) -> str:
    '''
    Get git hash for repository in `dir`.
    '''
    # change dir
    cwd = os.getcwd()
    os.chdir(dir)

    # get hash (https://stackoverflow.com/a/21901260)
    hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    # change dir back
    os.chdir(cwd)

    return hash

def CalculateEqmTemperature(I_0, ASF_sf, A_B):
    '''
    Calculate planetary equilibrium temperature.
    Params: Stellar flux, ASF scale factor, and bond albedo.
    '''
    return (I_0 * ASF_sf * (1.0 - A_B) / const_sigma)**(1.0/4.0)

def PrintCurrentState(hf_row:dict):
    '''
    Print the current state of the model to the logger
    '''
    log.info("Runtime info...")
    log.info("    System time  :   %s  "         % str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    log.info("    Model time   :   %.2e   yr"    % float(hf_row["Time"]))
    log.info("    T_surf       :   %4.3f   K"    % float(hf_row["T_surf"]))
    log.info("    T_magma      :   %4.3f   K"    % float(hf_row["T_magma"]))
    log.info("    P_surf       :   %.2e   bar"   % float(hf_row["P_surf"]))
    log.info("    Phi_global   :   %.2e   "      % float(hf_row["Phi_global"]))
    log.info("    Instellation :   %.2e   W m-2" % float(hf_row["F_ins"]))
    log.info("    F_int        :   %.2e   W m-2" % float(hf_row["F_int"]))
    log.info("    F_atm        :   %.2e   W m-2" % float(hf_row["F_atm"]))
    log.info("    |F_net|      :   %.2e   W m-2" % abs(float(hf_row["F_net"])))

def CreateLockFile(output_dir:str):
    '''
    Create a lock file which, if removed, will signal for the simulation to stop.
    '''
    keepalive_file = os.path.join(output_dir,"keepalive")
    safe_rm(keepalive_file)
    with open(keepalive_file, 'w') as fp:
        fp.write("Removing this file will be interpreted by PROTEUS as a request to stop the simulation loop\n")
    return keepalive_file

def GetHelpfileKeys():
    '''
    Variables to be held in the helpfile.

    All dimensional quantites should be stored in SI units, except those noted below.
    * Pressure is in units of [bar].
    * Time is in units of [years].
    '''

    # Basic keys
    keys = [
            # Model tracking and basic parameters
            "Time", "R_planet", "M_planet", "separation", # [yr], [m], [kg], [m]

            # Temperatures
            "T_surf", "T_magma", "T_eqm", "T_skin", # all [K]

            # Energy fluxes
            "F_int", "F_atm", "F_net", "F_olr", "F_sct", "F_ins", # all [W m-2]

            # Interior properties
            "gravity", "Phi_global", "RF_depth", # [m s-2] , [1] , [1]
            "M_core", "M_mantle", "M_mantle_solid", "M_mantle_liquid", # all [kg]

            # Stellar
            "R_star", "age_star", # [m], [yr]

            # Observational (from infinity)
            "z_obs", "rho_obs", "transit_depth", "contrast_ratio", # [m], [kg m-3], [1], [1]

            # Escape
            "esc_rate_total", # [kg s-1]

            # Atmospheric composition
            "M_atm", "P_surf", "atm_kg_per_mol", # [kg], [bar], [kg mol-1]
            ]

    # gases
    for s in volatile_species:
        keys.append(s+"_mol_atm")
        keys.append(s+"_mol_solid")
        keys.append(s+"_mol_liquid")
        keys.append(s+"_mol_total")
        keys.append(s+"_kg_atm")
        keys.append(s+"_kg_solid")
        keys.append(s+"_kg_liquid")
        keys.append(s+"_kg_total")
        keys.append(s+"_vmr")
        keys.append(s+"_bar")

    # element masses
    for e in element_list:
        keys.append(e+"_kg_atm")
        keys.append(e+"_kg_solid")
        keys.append(e+"_kg_liquid")
        keys.append(e+"_kg_total")

    # elemental ratios
    for e1 in element_list:
        for e2 in element_list:
            if e1==e2:
                continue

            # reversed ratio
            k = "%s/%s_atm"%(e2,e1)
            if k in keys:
                # skip this, since it's redundant to store (for example) the
                # ratio of H/C when we already have C/H.
                continue

            # intended ratio to be stored
            k = "%s/%s_atm"%(e1,e2)
            if k in keys:
                continue
            keys.append(k)

    return keys

def CreateHelpfileFromDict(d:dict):
    '''
    Create helpfile to hold output variables.
    '''
    log.debug("Creating new helpfile from dict")
    return pd.DataFrame([d], columns=GetHelpfileKeys(), dtype=float)

def ZeroHelpfileRow():
    '''
    Get a dictionary with same keys as helpfile but with values of zero
    '''
    out = {}
    for k in GetHelpfileKeys():
        out[k] = 0.0
    return out

def ExtendHelpfile(current_hf:pd.DataFrame, new_row:dict):
    '''
    Extend helpfile with new row of variables
    '''
    log.debug("Extending helpfile with new row")

    # validate keys
    missing_keys = set(GetHelpfileKeys()) - set(new_row.keys())
    if len(missing_keys)>0:
        raise Exception("There are mismatched keys in helpfile: %s"%missing_keys)

    # convert row to df
    new_row = pd.DataFrame([new_row], columns=GetHelpfileKeys(), dtype=float)

    # concatenate and return
    return pd.concat([current_hf, new_row], ignore_index=True)


def WriteHelpfileToCSV(output_dir:str, current_hf:pd.DataFrame):
    '''
    Write helpfile to a CSV file
    '''
    log.debug("Writing helpfile to CSV file")

    # check for invalid or missing keys
    difference = set(GetHelpfileKeys()) - set(current_hf.keys())
    if len(difference) > 0:
        raise Exception("There are mismatched keys in helpfile: "+str(difference))

    # remove old file
    fpath = os.path.join(output_dir , "runtime_helpfile.csv")
    if os.path.exists(fpath):
        os.remove(fpath)

    # write new file
    current_hf.to_csv(fpath, index=False, sep="\t", float_format="%.6e")
    return fpath

def ReadHelpfileFromCSV(output_dir:str):
    '''
    Read helpfile from disk CSV file to DataFrame
    '''
    fpath = os.path.join(output_dir , "runtime_helpfile.csv")
    if not os.path.exists(fpath):
        raise Exception("Cannot find helpfile at '%s'"%fpath)
    return pd.read_csv(fpath, sep=r"\s+")


def ValidateInitFile(dirs:dict, config: Config):
    '''
    Validate configuration file, checking for invalid options
    '''
    # Ensure that all volatiles are all tracked
    for s in volatile_species:
        key_pp = str(s+"_initial_bar")
        key_in = str(s+"_included")

        if (config[key_pp] > 0.0) and (config[key_in] == 0):
            UpdateStatusfile(dirs, 20)
            raise RuntimeError(f"Volatile {s} has non-zero pressure but is disabled in cfg")

        if (config[key_pp] > 0.0) and (config["solvevol_use_params"] > 0):
            UpdateStatusfile(dirs, 20)
            raise RuntimeError(f"Volatile {s} has non-zero pressure but outgassing parameters are enabled")

    # Required vols
    for s in ["H2O","CO2","N2","S2"]:
        if config[s+"_included"] == 0:
            UpdateStatusfile(dirs, 20)
            raise RuntimeError(f"Missing required volatile {s}")

    return True

def UpdatePlots( hf_all:pd.DataFrame, output_dir:str, config:Config, end=False, num_snapshots=7):
    """Update plots during runtime for analysis

    Calls various plotting functions which show information about the interior/atmosphere's energy and composition.

    Parameters
    ----------
        output_dir : str
            Output directory containing simulation information
        config : Config
            PROTEUS options dictionary.
        end : bool
            Is this function being called at the end of the simulation?
    """

    # Check model configuration
    dummy_atm = config["atmosphere_model"] == 'dummy'
    dummy_int = config["interior_model"] == 'dummy'
    escape    = config["escape_model"] is not None

    # Get all output times
    if dummy_int:
        output_times = []
    else:
        from proteus.interior.spider import get_all_output_times
        output_times = get_all_output_times( output_dir )

    # Global properties for all timesteps
    plot_global(hf_all, output_dir, config)

    # Elemental mass inventory
    if escape:
        plot_elements(hf_all, output_dir, config["plot_format"])
        plot_escape(hf_all, output_dir, escape_model=config['escape_model'], plot_format=config["plot_format"])

    # Which times do we have atmosphere data for?
    if not dummy_atm:
        ncs = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
        nc_times = [int(f.split("/")[-1].split("_atm")[0]) for f in ncs]

        # Check intersection of atmosphere and interior data
        if dummy_int:
            output_times = nc_times
        else:
            output_times = sorted(list(set(output_times) & set(nc_times)))

    # Samples for plotting profiles
    if len(output_times) > 0:
        nsamp = 7
        tmin = 1.0
        if np.amax(output_times) > 1e3:
            tmin = 1e3
        plot_times, _ = sample_times(output_times, nsamp, tmin=tmin)
        log.debug("Snapshots to plot:" + str(plot_times))

    # Interior profiles
    if not dummy_int:
        jsons = read_jsons(output_dir, plot_times)
        plot_interior(output_dir, plot_times, jsons, config["plot_format"])

    # Temperature profiles
    if not dummy_atm:
        ncdfs = read_ncdfs(output_dir, plot_times)

        # Atmosphere only
        plot_atmosphere(output_dir, plot_times, ncdfs, config["plot_format"])

        # Atmosphere and interior, stacked
        if not dummy_int:
            plot_stacked(output_dir, plot_times, jsons, ncdfs, config["plot_format"])

        # Flux profiles
        if config["atmosphere_model"] == 0:
            # only do this for JANUS, AGNI does it automatically
            plot_fluxes_atmosphere(output_dir, config["plot_format"])

    # Only at the end of the simulation
    if end:
        plot_global(hf_all,         output_dir, config, logt=False)
        plot_fluxes_global(hf_all,  output_dir, config)
        plot_observables(hf_all,    output_dir, plot_format=config["plot_format"])
        plot_sflux(output_dir,          plot_format=config["plot_format"])
        plot_sflux_cross(output_dir,    plot_format=config["plot_format"])

        if not dummy_int:
            plot_interior_cmesh(output_dir, plot_format=config["plot_format"])

        if not dummy_atm:
            plot_emission(output_dir, plot_times, plot_format=config["plot_format"])

    # Close all figures
    plt.close()

def SetDirectories(config: Config):
    """Set directories dictionary

    Sets paths to the required directories, based on the configuration provided
    by the options dictionary.

    Parameters
    ----------
        config : Config
            PROTEUS options dictionary

    Returns
    ----------
        dirs : dict
            Dictionary of paths to important directories
    """

    proteus_dir = get_proteus_dir()
    proteus_src = os.path.join(proteus_dir,"src","proteus")

    # PROTEUS folders
    dirs = {
            "output":   os.path.join(proteus_dir,"output",config['dir_output']),
            "input":    os.path.join(proteus_dir,"input"),
            "proteus":  proteus_dir,
            "agni":     os.path.join(proteus_dir,"AGNI"),
            "vulcan":   os.path.join(proteus_dir,"VULCAN"),
            "spider":   os.path.join(proteus_dir,"SPIDER"),
            "utils":    os.path.join(proteus_src,"utils"),
            "tools":    os.path.join(proteus_dir,"tools"),
            }

    # FWL data folder
    if os.environ.get('FWL_DATA') is None:
        UpdateStatusfile(dirs, 20)
        raise Exception("The FWL_DATA environment variable needs to be set up!"
                        "Did you source PROTEUS.env?")
    else:
        dirs["fwl"] = os.environ.get('FWL_DATA')

    # SOCRATES directory
    if config["atmosphere_model"] in [0,1]:
        # needed for atmosphere models 0 and 1

        if os.environ.get('RAD_DIR') is None:
            UpdateStatusfile(dirs, 20)
            raise Exception("The RAD_DIR environment variable has not been set")
        else:
            dirs["rad"] = os.environ.get('RAD_DIR')

    # Get abspaths
    for key in dirs.keys():
        dirs[key] = os.path.abspath(dirs[key])+"/"

    return dirs


# End of file
