# Functions used to help run PROTEUS which are mostly submodule agnostic.

# Import utils-specific modules
from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.plot.cpl_atmosphere import plot_atmosphere
from proteus.plot.cpl_elements import plot_elements
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
from proteus.utils.helper import UpdateStatusfile, find_nearest, safe_rm
from proteus.utils.spider import get_all_output_times

log = logging.getLogger("PROTEUS")


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

def parse_console_arguments()->dict:
    '''
    Handle command line arguments for PROTEUS
    '''
    parser = argparse.ArgumentParser(description='PROTEUS command line arguments')

    parser.add_argument('--cfg', type=str,
                        default="input/default.cfg", help='Path to configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume simulation from disk')

    args = vars(parser.parse_args())
    return args

# https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

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
    Variables to be held in the helpfile
    '''

    # Basic keys
    keys = [
            # Model tracking and basic parameters
            "Time", "R_planet", "M_planet",

            # Temperatures
            "T_surf", "T_magma", "T_eqm", "T_skin",

            # Energy fluxes
            "F_int", "F_atm", "F_net", "F_olr", "F_sct", "F_ins",

            # Interior properties
            "gravity", "Phi_global", "RF_depth",
            "M_core", "M_mantle", "M_mantle_solid", "M_mantle_liquid",

            # Stellar
            "R_star", "age_star",

            # Observational
            "z_obs", "transit_depth", "contrast_ratio", # observed from infinity

            # Escape
            "esc_rate_total",

            # Atmospheric composition
            "M_atm", "P_surf", "atm_kg_per_mol", # more keys added below
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

def ReadInitFile(init_file_passed:str, verbose=False):
    '''
    Read configuration file into a dictionary
    '''
    log.debug("Reading configuration file")

    # Read in input file as dictionary
    OPTIONS  = {}
    if verbose:
        log.info("Read in init file:" + init_file)

    if os.path.isfile(init_file_passed):
        init_file = init_file_passed
    else:
        raise Exception("Init file provided is not a file or does not exist (%s)" % init_file_passed)

    if verbose: log.info("Settings:")

    # Open file and fill dict
    with open(init_file) as f:

        # Filter blank lines
        lines = list(filter(None, (line.rstrip() for line in f)))

        for line in lines:

            # Skip comments
            if not line.startswith("#"):

                # Skip inline comments
                line = line.split("#")[0]
                line = line.split(",")[0]

                if verbose: log.info(line)

                # Assign key and value
                (key, val) = line.split("=")
                key = key.strip()
                val = val.strip()

                # Some parameters are int
                if key in [ "solid_stop", "steady_stop", "iter_max", "emit_stop",
                            "escape_model", "atmosphere_surf_state", "water_cloud",
                            "plot_iterfreq", "stellar_heating", "mixing_length",
                            "atmosphere_chemistry", "solvevol_use_params",
                            "tropopause", "F_atm_bc",
                            "dt_dynamic", "prevent_warming", "atmosphere_model",
                            "atmosphere_nlev", "rayleigh",
                            "shallow_ocean_layer"]:
                    val = int(val)

                # Some are str
                elif key in [ 'star_spectrum', 'dir_output', 'plot_format',
                                'spectral_file' , 'log_level']:
                    val = str(val)

                # Most are float
                else:
                    val = float(val)

                # Set option
                OPTIONS[key] = val

    return OPTIONS

def ValidateInitFile(dirs:dict, OPTIONS:dict):
    '''
    Validate configuration file, checking for invalid options
    '''

    if OPTIONS["atmosphere_surf_state"] == 2: # Not all surface treatments are mutually compatible
        if OPTIONS["shallow_ocean_layer"] == 1:
            UpdateStatusfile(dirs, 20)
            raise Exception("Shallow mixed layer scheme is incompatible with the conductive lid scheme! Turn one of them off")

    surf_state = int(OPTIONS["atmosphere_surf_state"])
    if not (0 <= surf_state <= 3):
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid surface state %d" % surf_state)

    if OPTIONS["atmosphere_model"] == 1:  # Julia required for AGNI
        if shutil.which("julia") is None:
            UpdateStatusfile(dirs, 20)
            raise Exception("Could not find Julia in current environment")

    if OPTIONS["atmosphere_nlev"] < 15:
        UpdateStatusfile(dirs, 20)
        raise Exception("Atmosphere must have at least 15 levels")

    if OPTIONS["interior_nlev"] < 40:
        UpdateStatusfile(dirs, 20)
        raise Exception("Interior must have at least 40 levels")

    # Ensure that all volatiles are all tracked
    for s in volatile_species:
        key_pp = str(s+"_initial_bar")
        key_in = str(s+"_included")
        if (OPTIONS[key_pp] > 0.0) and (OPTIONS[key_in] == 0):
            UpdateStatusfile(dirs, 20)
            raise Exception("Volatile %s has non-zero pressure but is disabled in cfg"%s)
        if (OPTIONS[key_pp] > 0.0) and (OPTIONS["solvevol_use_params"] > 0):
            UpdateStatusfile(dirs, 20)
            raise Exception("Volatile %s has non-zero pressure but outgassing parameters are enabled"%s)

    # Required vols
    for s in ["H2O","CO2","N2","S2"]:
        if OPTIONS[s+"_included"] == 0:
            UpdateStatusfile(dirs, 20)
            raise Exception("Missing required volatile '%s'"%s)

    return True

def UpdatePlots( output_dir:str, OPTIONS:dict, end=False, num_snapshots=7):
    """Update plots during runtime for analysis

    Calls various plotting functions which show information about the interior/atmosphere's energy and composition.

    Parameters
    ----------
        output_dir : str
            Output directory containing simulation information
        OPTIONS : dict
            PROTEUS options dictionary.
        end : bool
            Is this function being called at the end of the simulation?
    """

    # Check model configuration
    dummy_atm = bool(OPTIONS["atmosphere_model"] == 2)
    escape    = bool(OPTIONS["escape_model"] > 0)

    # Get all JSON files
    output_times = get_all_output_times( output_dir )

    # Do not plot if there's insufficient data
    if np.amax(output_times) < 2:
        return

    # Global properties for all timesteps
    if len(output_times) > 2:
        plot_global(output_dir, OPTIONS)

        # Elemental mass inventory
        if escape:
            plot_elements(output_dir, OPTIONS["plot_format"])
            plot_escape(output_dir, escape_model=OPTIONS['escape_model'], plot_format=OPTIONS["plot_format"])
    # Filter to only include steps with corresponding NetCDF files
    if not dummy_atm:
        ncs = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
        nc_times = [int(f.split("/")[-1].split("_atm")[0]) for f in ncs]
        output_times = sorted(list(set(output_times) & set(nc_times)))

    # Work out which times we want to plot
    if len(output_times) <= num_snapshots:
        plot_times = output_times

    else:
        plot_times = []
        tmin = max(1,np.amin(output_times))
        tmax = max(tmin+1, np.amax(output_times))

        for s in np.logspace(np.log10(tmin),np.log10(tmax),num_snapshots): # Sample on log-scale

            remaining = list(set(output_times) - set(plot_times))
            if len(remaining) == 0:
                break

            v,_ = find_nearest(remaining,s) # Find next new sample
            plot_times.append(int(v))

    plot_times = sorted(set(plot_times)) # Remove any duplicates + resort
    log.debug("Snapshots to plot:" + str(plot_times))

    # Temperature profiles
    plot_interior(output_dir, plot_times, OPTIONS["plot_format"])
    if not dummy_atm:
        plot_atmosphere(output_dir, plot_times, OPTIONS["plot_format"])
        plot_stacked(output_dir, plot_times, OPTIONS["plot_format"])

        if OPTIONS["atmosphere_model"] != 1:
            # don't make this plot for AGNI, since it will do it itself
            plot_fluxes_atmosphere(output_dir, OPTIONS["plot_format"])

    # Only at the end of the simulation
    if end:
        plot_global(output_dir, OPTIONS, logt=False)
        plot_interior_cmesh(output_dir, plot_format=OPTIONS["plot_format"])
        plot_sflux(output_dir, plot_format=OPTIONS["plot_format"])
        plot_sflux_cross(output_dir, plot_format=OPTIONS["plot_format"])
        plot_fluxes_global(output_dir, OPTIONS)
        plot_observables(output_dir, plot_format=OPTIONS["plot_format"])

    # Close all figures
    plt.close()

def SetDirectories(OPTIONS: dict):
    """Set directories dictionary

    Sets paths to the required directories, based on the configuration provided
    by the options dictionary.

    Parameters
    ----------
        OPTIONS : dict
            PROTEUS options dictionary

    Returns
    ----------
        dirs : dict
            Dictionary of paths to important directories
    """

    if os.environ.get('PROTEUS_DIR') == None:
        raise Exception("Environment variables not set! Have you sourced PROTEUS.env?")
    proteus_dir = os.path.abspath(os.getenv('PROTEUS_DIR'))
    proteus_src = os.path.join(proteus_dir,"src/proteus")

    # PROTEUS folders
    dirs = {
            "output":   os.path.join(proteus_dir,"output",OPTIONS['dir_output']),
            "input":    os.path.join(proteus_dir,"input"),
            "proteus":  proteus_dir,
            "janus":    os.path.join(proteus_dir,"JANUS"),
            "agni":     os.path.join(proteus_dir,"AGNI"),
            "vulcan":   os.path.join(proteus_dir,"VULCAN"),
            "spider":   os.path.join(proteus_dir,"SPIDER"),
            "utils":    os.path.join(proteus_src,"utils")
            }

    # FWL data folder
    if os.environ.get('FWL_DATA') == None:
        UpdateStatusfile(dirs, 20)
        raise Exception("The FWL_DATA environment variable where spectral"
                        "and evolution tracks data will be downloaded needs to be set up!"
                        "Did you source PROTEUS.env?")
    else:
        dirs["fwl"] = os.environ.get('FWL_DATA')


    # SOCRATES directory
    if OPTIONS["atmosphere_model"] in [0,1]:
        # needed for atmosphere models 0 and 1

        if os.environ.get('RAD_DIR') == None:
            UpdateStatusfile(dirs, 20)
            raise Exception("The RAD_DIR environment variable has not been set")
        else:
            dirs["rad"] = os.environ.get('RAD_DIR')

    # Get abspaths
    for key in dirs.keys():
        dirs[key] = os.path.abspath(dirs[key])+"/"

    return dirs


# End of file
