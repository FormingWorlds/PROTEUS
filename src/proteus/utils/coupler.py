# Functions used to help run PROTEUS which are mostly submodule agnostic.

# Import utils-specific modules
from __future__ import annotations

import glob
import logging
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib as mpl  # noqa

mpl.use('Agg') # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import element_list, gas_list, secs_per_hour, secs_per_minute
from proteus.utils.helper import UpdateStatusfile, get_proteus_dir, safe_rm
from proteus.utils.plot import sample_times

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

LOCKFILE_NAME="keepalive"

def _get_current_time():
    '''
    Get the current system time as a formatted string.
    '''
    return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z'))

def _get_git_revision(dir:str) -> str:
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

def _get_socrates_version():
    '''
    Get the installed SOCRATES version.
    '''
    verpath = os.path.join(os.environ.get("RAD_DIR"),"version")
    with open(verpath, 'r') as hdl:
        ver = hdl.read().replace("\n","")
    return str(ver)

def _get_spider_version():
    '''
    Get the installed SPIDER version.
    '''

    # This is the only one that we use, and it won't be updated in the future.
    return "0.2.0"

def _get_petsc_version():
    '''
    Get the installed PETSc version.
    '''

    # Like SPIDER, this is the only one that we use
    return "1.3.19.0"

def _get_agni_version(dirs:dict):
    '''
    Get the installed AGNI version
    '''
    from tomllib import load as tomlload
    with open(os.path.join(dirs["agni"],"Project.toml"),'rb') as hdl:
        agni_meta = tomlload(hdl)
    return agni_meta["version"]

def _get_julia_version():
    '''
    Get the installed Julia version
    '''
    return subprocess.check_output(["julia","--version"]).decode("utf-8").split()[-1]

def print_system_configuration(dirs:dict):
    '''
    Print the current system configuration.
    '''
    import platform
    import pwd
    import sys

    # Try to get the login name using os.getlogin()
    try:
        username = os.getlogin()
    except OSError:
        username = pwd.getpwuid(os.getuid()).pw_name

    log.info("Current time      " + _get_current_time())
    log.info("Python version    " + sys.version.split(" ")[0])
    log.info("System hostname   " + str(os.uname()[1]))
    log.info("System username   " + str(username))
    log.info("Platform type     " + str(platform.system()))
    log.info("FWL data path     " + dirs["fwl"])
    log.info(" ")

def print_module_configuration(dirs:dict, config:Config, config_path:str):
    '''
    Print the current module configuration, with versions.
    '''

    # PROTEUS
    from proteus import __version__ as proteus_version
    log.info("PROTEUS version   " + proteus_version)
    log.info("PROTEUS location  " + dirs["proteus"])
    log.info("PROTEUS git hash  " + _get_git_revision(dirs["proteus"]))
    log.info("Config file       " + str(config_path))
    log.info("Output path       " + dirs["output"])
    log.info(" ")

    # Interior module
    write = "Interior module   %s" % config.interior.module
    match config.interior.module:
        case 'spider':
            write += " version " + _get_spider_version()
        case 'aragog':
            from aragog import __version__ as aragog_version
            write += " version " + aragog_version
    log.info(write)
    if config.interior.module == "spider":
        log.info("  - PETSc         version " + _get_petsc_version())

    # Atmosphere module
    write = "Atmos_clim module %s" % config.atmos_clim.module
    match config.atmos_clim.module:
        case 'janus':
            from janus import __version__ as janus_version
            write += " version " + janus_version
        case 'agni':
            write += " version " + _get_agni_version(dirs)
    log.info(write)
    if config.atmos_clim.module in ['janus', 'agni']:
        log.info("  - SOCRATES      version %s at %s" %(_get_socrates_version(), dirs["rad"]))
        if config.atmos_clim.module == 'agni':
            log.info("  - Julia         version " + _get_julia_version())

    # Outgassing module
    write = "Outgas module     %s" % config.outgas.module
    if config.outgas.module == 'calliope':
        from calliope import __version__ as calliope_version
        write += " version " + calliope_version
    log.info(write)

    # Escape module
    write = "Escape module     %s" % config.escape.module
    if config.escape.module == 'zephyrus':
        from zephyrus import __version__ as zephyrus_version
        write += " version " + zephyrus_version
    log.info(write)

    # Star module
    write = "Star module       %s" % config.star.module
    if config.star.module == 'mors':
        from mors import __version__ as mors_version
        write += " version " + mors_version
    log.info(write)

    # Orbit module
    log.info("Orbit module      %s" % config.orbit.module)
    if config.orbit.module == 'lovepy':
        log.info("  - Julia         version " + _get_julia_version())

    # Delivery module
    log.info("Delivery module   %s" % config.delivery.module)

    # Atmospheric chemistry module
    log.info("Atmos_chem module %s" % config.atmos_chem.module)

    # Observations synthesis module
    write = "Observe module    %s" % config.observe.synthesis
    if config.observe.synthesis == "platon":
        from platon import __version__ as platon_version
        write += " version " + platon_version
    log.info(write)

    # End spacer
    log.info(" ")

def print_citation(config:Config):
    '''
    Print information on which papers should be cited.
    '''

    log.info("If you use these results in a publication, please cite:")

    def _cite(key:str, url:str):
        log.info("  - "+key+", "+url)

    # Core PROTEUS papers
    _cite("Lichtenberg et al. (2021)",
            "https://doi.org/10.1029/2020JE006711")
    _cite("Nicholls et al. (2024)",
            "https://doi.org/10.1029/2024JE008576")

    # Atmosphere module
    match config.atmos_clim.module:
        case 'janus':
            _cite("Graham et al. (2021)",
                    "https://doi.org/10.3847/PSJ/ac214c")
        case 'agni':
            _cite("Nicholls et al. (2025)",
                    "https://doi.org/10.1093/mnras/stae2772")
        case _:
            pass

    # Interior module
    match config.interior.module:
        case 'spider':
            _cite("Bower et al. (2021)",
                    "https://doi.org/10.3847/PSJ/ac5fb1")
        case 'aragog':
            # _cite("Bower et al. (2025)", "in prep")
            pass
        case _:
            pass

    # Outgassing module
    match config.outgas.module:
        case 'calliope':
            # Covered by Nicholls et al. (2024, 2025)
            pass
        case 'atmodeller':
            # _cite("Bower et al. (2025)", "in prep")
            pass
        case _:
            pass

    # Escape module
    match config.outgas.module:
        case 'zephyrus':
            # _cite("Postolec et al. (2025)", "in prep")
            pass
        case _:
            pass

    # Star module
    match config.star.module:
        case 'mors':
            _cite("Johnstone et al. (2021)",
                    "https://doi.org/10.1051/0004-6361/202038407")
        case _:
            pass

    # Orbit module
    match config.orbit.module:
        case 'lovepy':
            _cite("Hay & Matsuyama (2019)",
                    "https://doi.org/10.3847/1538-4357/ab0c21")
        case _:
            pass

    # Delivery module
    match config.delivery.module:
        case _:
            pass

    # Observations synthesis module
    match config.observe.synthesis:
        case "platon":
            _cite("Zhang et al. (2024)",
                    "https://doi.org/10.48550/arXiv.2410.22398")
        case _:
            pass

    # Atmospheric chemistry module
    if config.atmos_chem.when != "manually":
        match config.atmos_chem.module:
            case "vulcan":
                _cite("Tsai et al. (2021)",
                        "https://doi.org/10.3847/1538-4357/ac29bc")
            case _:
                pass

def print_header():
    log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    log.info("                   PROTEUS framework                   ")
    log.info("            Copyright (C) %4d Forming Worlds          "%(datetime.now().year))
    log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    log.info(" ")

def print_stoptime(start_time):
    end_time = datetime.now()
    log.info("Simulation stopped at: " + _get_current_time())

    run_time = end_time - start_time
    run_time = run_time.total_seconds()
    if run_time > secs_per_hour:
        log.info("Total runtime: %.2f hours" % (run_time / secs_per_hour))
    elif run_time > secs_per_minute:
        log.info("Total runtime: %.2f minutes" % (run_time / secs_per_minute))
    else:
        log.info("Total runtime: %.2f seconds" % run_time)

    log.info(" ")

def PrintCurrentState(hf_row:dict):
    '''
    Print the current state of the model to the logger
    '''
    log.info("Runtime info...")
    log.info("    Wall time  = %s  "         % _get_current_time())
    log.info("    Model time = %.2e   yr"    % float(hf_row["Time"]))
    log.info("    T_surf     = %8.3f   K"    % float(hf_row["T_surf"]))
    log.info("    T_magma    = %8.3f   K"    % float(hf_row["T_magma"]))
    log.info("    P_surf     = %.2e   bar"   % float(hf_row["P_surf"]))
    log.info("    Phi_global = %.2e   "      % float(hf_row["Phi_global"]))
    log.info("    F_atm      = %.2e   W m-2" % float(hf_row["F_atm"]))
    log.info("    F_int      = %.2e   W m-2" % float(hf_row["F_int"]))

def CreateLockFile(output_dir:str):
    '''
    Create a lock file which, if removed, will signal for the simulation to stop.
    '''
    keepalive_file = os.path.join(output_dir,LOCKFILE_NAME)
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
            # Model tracking
            "Time", # [yr]

            # Orbit semi-major axis and time-averaged separation
            "semimajorax", "separation", # [m], [m],

            # Orbital period and eccentricity
            "orbital_period", "eccentricity", # [s], [1]

            # Day length
            "axial_period", # [s]

            # Dry interior radius (calculated) and mass (from config)
            "R_int", "M_int", # [m], [kg]

            # Total planet mass (mantle + core + volatiles)
            "M_tot", # [kg]

            # Temperatures
            "T_surf", "T_magma", "T_eqm", "T_skin", # all [K]

            # Planet energy fluxes, all in units of [W m-2]
            "F_int", "F_atm", "F_net", "F_olr", "F_sct", "F_ins", "F_xuv",
            "F_tidal", "F_radio",

            # Interior properties
            "gravity", "Phi_global", "RF_depth", # [m s-2] , [1] , [1]
            "M_core", "M_mantle", "M_planet",    # all [kg]
            "M_mantle_solid", "M_mantle_liquid", # all [kg]

            # Stellar
            "M_star", "R_star", "age_star", # [kg], [m], [yr]
            "T_star", # [K]

            # Observational (from infinity)
            "p_obs",    # observered radius [bar]
            "R_obs",    # observed radius [m]
            "rho_obs",  # observed bulk density [kg m-3]
            "transit_depth", "eclipse_depth", # [1], [1]
            "bond_albedo", # bolometric bond albedo [1]

            # Imaginary part of k2 Love Number
            "Imk2", # [1]

            # Escape
            "esc_rate_total", "p_xuv", "R_xuv", # [kg s-1], [bar], [m]

            # Atmospheric composition
            "M_atm", "P_surf", "atm_kg_per_mol", # [kg], [bar], [kg mol-1]
            ]

    # gases
    for s in gas_list:
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

    # Diagnostic variables...

    # Weak temperature gradient parameter at the surface
    keys.append("wtg_surf") # [1]

    # Roche limit (orbital separation at which planet will disintegrate)
    keys.append("roche_limit") # [m]

    # Hill radius (radius from planet at which the atmosphere becomes unbound)
    keys.append("hill_radius") # [m]

    # Wall-clock runtime
    keys.append("runtime") # [s]

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
    current_hf.to_csv(fpath, index=False, sep="\t", float_format="%.10e")
    return fpath

def ReadHelpfileFromCSV(output_dir:str):
    '''
    Read helpfile from disk CSV file to DataFrame
    '''
    fpath = os.path.join(output_dir , "runtime_helpfile.csv")
    if not os.path.exists(fpath):
        raise Exception("Cannot find helpfile at '%s'"%fpath)
    return pd.read_csv(fpath, sep=r"\s+")

def UpdatePlots( hf_all:pd.DataFrame, dirs:dict, config:Config, end=False, num_snapshots=7):
    """Update plots during runtime for analysis

    Calls various plotting functions which show information about the
    interior/atmosphere's energy and composition.

    Parameters
    ----------
    hf_all : pd.DataFrame
        Dataframe containing all the output data
    dirs : dict
        Dictionary of directories
    config : Config
        PROTEUS configuration object
    end : bool
        Is this function being called at the end of the simulation?
    num_snapshots : int
        Number of snapshots to include in each plot.
    """

    # Import utilities
    from proteus.atmos_clim.common import read_atmosphere_data
    from proteus.interior.wrapper import read_interior_data

    # Import plotting functions
    from proteus.plot.cpl_atmosphere import plot_atmosphere
    from proteus.plot.cpl_bolometry import plot_bolometry
    from proteus.plot.cpl_chem_atmosphere import plot_chem_atmosphere
    from proteus.plot.cpl_emission import plot_emission
    from proteus.plot.cpl_escape import plot_escape
    from proteus.plot.cpl_fluxes_atmosphere import plot_fluxes_atmosphere
    from proteus.plot.cpl_fluxes_global import plot_fluxes_global
    from proteus.plot.cpl_global import plot_global
    from proteus.plot.cpl_interior import plot_interior
    from proteus.plot.cpl_interior_cmesh import plot_interior_cmesh
    from proteus.plot.cpl_population import (
        plot_population_mass_radius,
        plot_population_time_density,
    )
    from proteus.plot.cpl_sflux import plot_sflux
    from proteus.plot.cpl_sflux_cross import plot_sflux_cross
    from proteus.plot.cpl_spectra import plot_spectra
    from proteus.plot.cpl_structure import plot_structure
    from proteus.plot.cpl_visual import plot_visual

    # Directories
    output_dir = dirs["output"]
    fwl_dir    = dirs["fwl"]

    # Check model configuration
    dummy_atm = config.atmos_clim.module == 'dummy'
    dummy_int = config.interior.module == 'dummy'
    agni      = config.atmos_clim.module == 'agni'
    spider    = config.interior.module == 'spider'
    aragog    = config.interior.module == 'aragog'
    observed  = bool(config.observe.synthesis is not None)

    # Get all output times
    output_times = []
    if spider:
        from proteus.interior.spider import get_all_output_times
        output_times = get_all_output_times( output_dir )
    if aragog:
        from proteus.interior.aragog import get_all_output_times
        output_times = get_all_output_times( output_dir )

    # Global properties for all timesteps
    plot_global(hf_all, output_dir, config)

    # Elemental mass inventory
    plot_escape(hf_all, output_dir, plot_format=config.params.out.plot_fmt)

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
        int_data = read_interior_data(output_dir, config.interior.module, plot_times)
        plot_interior(output_dir, plot_times, int_data,
                              config.interior.module, config.params.out.plot_fmt)

    # Atmosphere profiles
    if not dummy_atm:
        atm_data = read_atmosphere_data(output_dir, plot_times)

        # Atmosphere temperature/height profiles
        plot_atmosphere(output_dir, plot_times, atm_data, config.params.out.plot_fmt)

        # Atmospheric chemistry
        plot_chem_atmosphere(output_dir, config.atmos_chem.module,
                                plot_format=config.params.out.plot_fmt,
                                plot_offchem=False)

        # Atmosphere and interior, stacked radially
        if not dummy_int:
            plot_structure(hf_all, output_dir, plot_times, int_data, atm_data,
                            config.interior.module, config.params.out.plot_fmt)

        # Energy flux profiles
        plot_fluxes_atmosphere(output_dir, config.params.out.plot_fmt)

    # Only at the end of the simulation
    if end:

        # Global plot with linear-time axis
        plot_global(hf_all,         output_dir, config, logt=False)

        # Energy flux balance
        plot_fluxes_global(hf_all,  output_dir, config)

        # Bolometric observables
        plot_bolometry(hf_all,      output_dir, plot_format=config.params.out.plot_fmt)

        # Spectral observables
        if observed:
            plot_spectra(output_dir, plot_format=config.params.out.plot_fmt)

        # Chemical profiles
        if not dummy_atm:
            plot_chem_atmosphere(output_dir, config.atmos_chem.module,
                                    plot_format=config.params.out.plot_fmt)

        # Visualise planet and star
        if agni:
            plot_visual(hf_all, output_dir, idx=-1, plot_format=config.params.out.plot_fmt)

        # Check that the simulation ran for long enough to make useful plots
        if len(hf_all["Time"]) >= 3:
            plot_population_mass_radius (hf_all, output_dir, fwl_dir,
                                            config.params.out.plot_fmt)
            plot_population_time_density(hf_all, output_dir, fwl_dir,
                                            config.params.out.plot_fmt)

            plt_modern = bool(config.star.module == "mors")
            if plt_modern:
                modern_age = config.star.mors.age_now * 1e9
            else:
                modern_age = -1
            plot_sflux(output_dir, plt_modern=plt_modern,
                            plot_format=config.params.out.plot_fmt)
            plot_sflux_cross(output_dir,modern_age=modern_age,
                            plot_format=config.params.out.plot_fmt)

            if not dummy_int:
                plot_interior_cmesh(output_dir, plot_times, int_data,
                                        config.interior.module,
                                        plot_format=config.params.out.plot_fmt)

            if not dummy_atm:
                plot_emission(output_dir, plot_times,
                                    plot_format=config.params.out.plot_fmt)

    # Close all figures
    plt.close("all")

def remove_excess_files(outdir:str, rm_spectralfiles:bool=False):
    """Remove excess files from the output directory

    Parameters
    ----------
    outdir: str
        Path to the simulation's output directory
    rm_spectralfiles: bool
        Whether to remove spectral files
    """

    # Files to remove, relative to outdir
    rm_paths = [
        "agni_recent.log",
        "data/.spider_tmp",
        LOCKFILE_NAME
    ]

    # Remove spectral files if requested
    if rm_spectralfiles:
        rm_paths.append("runtime.sf")
        rm_paths.append("runtime.sf_k")

    # Loop over files
    for f in rm_paths:
        f = os.path.join(outdir, f)

        # Remove the file
        log.debug(f"Removing {f}")
        safe_rm(f)

def get_proteus_directories(outdir="_unset") -> dict[str, str]:
    """Create dict of proteus directories from root dir.

    Parameters
    ----------
    outdir : str
        Name of the simulation's output directory

    Returns
    -------
    dirs : dict[str, str]
        Proteus directories dict
    """
    root_dir = get_proteus_dir()

    return {
        "proteus":  root_dir,
        "agni":     os.path.join(root_dir, "AGNI"),
        "lovepy":   os.path.join(root_dir, "lovepy"),
        "input":    os.path.join(root_dir, "input"),
        "spider":   os.path.join(root_dir, "SPIDER"),
        "tools":    os.path.join(root_dir, "tools"),
        "vulcan":   os.path.join(root_dir, "VULCAN"),
        "utils":    os.path.join(root_dir, "src", "proteus", "utils"),
        "output":           os.path.join(root_dir, "output", outdir),
        "output/data":      os.path.join(root_dir, "output", outdir, "data"),
        "output/observe":   os.path.join(root_dir, "output", outdir, "observe"),
        "output/offchem":   os.path.join(root_dir, "output", outdir, "offchem"),
        "output/plots":     os.path.join(root_dir, "output", outdir, "plots"),
    }


def set_directories(config: Config) -> dict[str, str]:
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
    dirs = get_proteus_directories(outdir=config.params.out.path)

    # FWL data folder
    if os.environ.get('FWL_DATA') is None:
        UpdateStatusfile(dirs, 20)
        raise EnvironmentError("The FWL_DATA environment variable has not been set")
    else:
        dirs["fwl"] = os.environ.get('FWL_DATA')

    # SOCRATES directory
    if config.atmos_clim.module in ('janus', 'agni'):
        # needed for atmosphere models 0 and 1

        if os.environ.get('RAD_DIR') is None:
            UpdateStatusfile(dirs, 20)
            raise EnvironmentError("The RAD_DIR environment variable has not been set")
        else:
            dirs["rad"] = os.environ.get('RAD_DIR')

    # Get abspaths
    for key in dirs.keys():
        dirs[key] = os.path.abspath(dirs[key])+"/"

    return dirs
