# Run VULCAN chemical kinetics module
from __future__ import annotations

import glob
import logging
import os
import pickle
import shutil
import sys
from typing import TYPE_CHECKING

import numpy as np

# Import PROTEUS
from proteus.atmos_clim.common import read_atmosphere_data
from proteus.utils.constants import AU, R_sun, element_list, vol_list
from proteus.utils.helper import find_nearest

# Import VULCAN
# This is horrible, and should be changed when VULCAN is converted into a Python package
VULCAN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..","..","..","VULCAN"))
sys.path.append(VULCAN_PATH)
import vulcan # noqa

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

VULCAN_NAME = "vulcan.pkl"
OUTPUT_NAME = "vulcan.csv"

def run_vulcan_offline(dirs:dict, config:Config, hf_row:dict) -> bool:
    """
    Run VULCAN as a subprocess, postprocessing the final PROTEUS output state.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Configuration object.
    hf_row : dict
        Dictionary of current helpfile row.

    Returns
    ----------
    success : bool
        Did VULCAN run successfully?
    """

    log.debug(f"Using VULCAN from: {vulcan.__file__}")
    log.debug(f"    version {vulcan.__version__}")

    success = True

    # ------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------

    if config.atmos_clim.module != 'agni':
        log.warning("VULCAN offline chemistry only supported with AGNI")
        return False

    # make folder
    shutil.rmtree(dirs["output/offchem"], ignore_errors=True)
    os.makedirs(dirs["output/offchem"])

    # ------------------------------------------------------------
    # READ DATA FROM PROTEUS RUN
    # ------------------------------------------------------------

    # Get all times for which we have data
    year = hf_row["Time"]
    log.debug("Reading data for t=%.2e yr"%year)

    # Read atmosphere data
    atmos = read_atmosphere_data(dirs["output"], [year],
                                    extra_keys=["pl","tmpl", "x_gas", "Kzz"])[0]

    # ------------------------------------------------------------
    # WRITE VULCAN INPUT FILES
    # ------------------------------------------------------------

    # Output folder
    vulcan_out = dirs["output/offchem"]
    vulcan_plt = dirs["output/offchem"]
    log.debug("Writing VULCAN input data")

    # Find a reasonable file for the stellar flux
    ls = glob.glob(dirs["output"]+"/data/*.sflux")
    years = [int(f.split("/")[-1].split(".")[0]) for f in ls]
    sflux_fpath = dirs["output"]+"/data/%d.sflux"%find_nearest(years,year)[0]

    # Read spectrum and scale to surface of the star
    sflux_data = np.loadtxt(sflux_fpath, skiprows=1).T
    star_wl = np.array(sflux_data[0])
    star_fl = np.array(sflux_data[1]) * hf_row["separation"]**2 / hf_row["R_star"]**2

    # Remove small values
    star_wl = star_wl[star_fl > config.atmos_chem.vulcan.clip_fl]
    star_fl = star_fl[star_fl > config.atmos_chem.vulcan.clip_fl]

    # Write spectrum
    star_write = dirs["output/offchem"] + "star.dat"
    np.savetxt(star_write, np.array([star_wl, star_fl]).T,
                header="# WL(nm)    Flux(ergs/cm**2/s/nm)",
                fmt=["%.4f","%.4e"])

    # Get TP profile
    p_arr = np.array(atmos["pl"]) * 10.0 # dyne/cm^2
    t_arr = np.clip(atmos["tmpl"], a_min=180.0, a_max=None)

    # Get Kzz profile
    k_arr = np.array(atmos["Kzz"]) * 1e4 # cm^2/s

    # Write TPK profile
    header  = "#(dyne/cm2)\t(K)\t(cm2/s)\n"
    header += "Pressure\tTemp\tKzz"
    prof_write = dirs["output/offchem"] + "profile.dat"
    np.savetxt(prof_write, np.array([p_arr[::-1], t_arr[::-1], k_arr[::-1]]).T,
               delimiter="\t", header=header, comments='', fmt="%1.5e")

    # Write mixing ratios
    vmr_write = dirs["output/offchem"] + "vmrs.dat"
    x_gas = [list(p_arr)]
    header = "#Input composition arrays \nPressure\t"
    for key in atmos.keys():
        if "_vmr" in key:
            gas = key.split("_")[0]
            arr = list(atmos[key])
            arr.append(arr[-1]) # extend by 1 level
            x_gas.append(arr)
            header += gas + " "
    x_gas = np.array(x_gas).T
    np.savetxt(vmr_write, x_gas, delimiter='\t', comments='', header=header)

    # ------------------------------------------------------------
    # CREATE VULCAN CONFIG
    # ------------------------------------------------------------

    # Base config
    log.debug("Creating VULCAN config")
    vcfg = vulcan.Config()

    # Which elements are included?
    e_incl = []
    for e in element_list:
        if hf_row[e+"_kg_atm"] > 1e10:
            e_incl.append(e)
    log.debug(f"Found elements: {e_incl}")

    # Determine network and species of interest
    match config.atmos_chem.vulcan.network:
        case "CHO":
            ele_arr = ['H', 'O', 'C']
            plt_arr = ['H2',  'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2']
            sct_arr = ['H2', 'O2', 'CO2']
            if config.atmos_chem.photo_on:
                network_file = "CHO_photo_network.txt"
            else:
                network_file = "CHO_thermo_network.txt"

        case "NCHO":
            plt_arr = ['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'N2']
            sct_arr = ['H2', 'O2', 'N2', 'CO2']
            ele_arr = ['H', 'O', 'C', 'N']
            if config.atmos_chem.photo_on:
                network_file = "NCHO_photo_network.txt"
            else:
                network_file = "NCHO_thermo_network.txt"

        case "SNCHO":
            plt_arr = ['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'SO2', 'H2S', 'S2', 'S8']
            sct_arr = ['H2', 'O2', 'N2', 'CO2']
            ele_arr = ['H', 'O', 'C', 'N', 'S']
            network_file = "SNCHO_photo_network.txt"

        case _:
            log.error(f"Network not recognised: '{config.atmos_chem.vulcan.network}'")
            return False

    log.debug(f"Using '{network_file}' ")
    vcfg.network    = os.path.join(vulcan.paths.THERMO_DIR, network_file)
    vcfg.atom_list  = ele_arr
    vcfg.use_photo  = config.atmos_chem.photo_on
    vcfg.use_ion    = False

    # Parse atmospheric composition into string, work out background gas
    backs = {'H2':0.0, 'N2':0.0, 'O2':0.0, 'CO2':0.0}
    vcfg.const_mix = {}
    for gas in vol_list:
        vmr = hf_row[gas+"_vmr"]
        if gas in backs.keys():
            backs[gas] = vmr
        if vmr > config.atmos_chem.vulcan.clip_vmr:
            vcfg.const_mix[gas] = vmr

    # Translate initial composition to VULCAN's string
    if config.atmos_chem.vulcan.ini_mix == "profile":
        vcfg.ini_mix = "table"
    else:
        vcfg.ini_mix = "const_mix"
    vcfg.vul_ini = vmr_write
    vcfg.scat_sp = sct_arr
    vcfg.T_cross_sp = []

    # Surface boundary condition
    vcfg.use_botflux = False
    if config.atmos_chem.vulcan.fix_surf:
        vcfg.use_fix_sp_bot = vcfg.const_mix
    else:
        vcfg.use_fix_sp_bot = {}

    # TOA boundary condition
    vcfg.use_topflux = False
    vcfg.diff_esc    = [] # species for diffusion-limit escape at TOA

    # Set background gas
    vcfg.atm_base = max(backs, key=backs.get)
    log.debug(f"Background gas is '{vcfg.atm_base}'")

    # TP profile
    vcfg.nz         =   len(p_arr)
    vcfg.P_b        =   np.amax(atmos["p"])*10
    vcfg.P_t        =   np.amin(atmos["p"])*10
    vcfg.atm_type   =   "file"
    vcfg.atm_file   =   prof_write
    vcfg.rocky      =   True
    vcfg.Rp         =   hf_row["R_int"] * 100.0  # Planetary radius (cm)
    vcfg.gs         =   hf_row["gravity"] * 100  # surface gravity (cm/s^2)

    # Spectrum
    vcfg.sflux_file   = star_write
    vcfg.r_star       = hf_row["R_star"] / R_sun     # stellar radius (R_sun)
    vcfg.orbit_radius = hf_row["separation"] / AU    # planet-star distance in A.U.
    vcfg.sl_angle     = config.orbit.zenith_angle * np.pi / 180.0   # the zenith angle
    vcfg.f_diurnal    = config.orbit.s0_factor

    # Mixing processes
    vcfg.update_frq  = 50    # frequency for updating dz and dzi due to change of mu
    vcfg.use_moldiff = config.atmos_chem.moldiff_on
    vcfg.use_vz      = True
    vcfg.vz_prof     = 'const'  # Options: 'const' or 'file'
    vcfg.const_vz    = config.atmos_chem.updraft_const # (cm/s)
    vcfg.use_Kzz     = config.atmos_chem.Kzz_on
    if config.atmos_chem.Kzz_const is not None:
        vcfg.Kzz_prof = "const"
        vcfg.const_Kzz = config.atmos_chem.Kzz_const
    else:
        # if None, will get Kzz from profile
        vcfg.Kzz_prof = "file"
        vcfg.const_Kzz = 1e5 # <- dummy value, will be replaced when VULCAN is run

    # Condensation
    vcfg.use_condense        = False
    vcfg.use_settling        = False
    vcfg.start_conden_time   = 1e10
    vcfg.condense_sp         = []
    vcfg.non_gas_sp          = []
    vcfg.fix_species         = []
    vcfg.fix_species_time    = 0

    # Convergence
    vcfg.st_factor          = 0.5
    vcfg.conv_step          = 100
    vcfg.yconv_cri          = config.atmos_chem.vulcan.yconv_cri  # check steady-state
    vcfg.slope_cri          = config.atmos_chem.vulcan.slope_cri  # check steady-state
    vcfg.yconv_min          = 0.5

    # Time-stepping
    vcfg.ode_solver         = 'Ros2'
    vcfg.trun_min           = 1e2
    vcfg.runtime            = 1.E22
    vcfg.use_print_prog     = True
    vcfg.use_print_delta    = False
    vcfg.print_prog_num     = 1    # print the progress every x steps
    vcfg.dttry              = 1.E-8
    vcfg.dt_min             = 1.E-9
    vcfg.dt_max             = vcfg.runtime*1e-4
    vcfg.dt_var_max         = 2.
    vcfg.dt_var_min         = 0.5
    vcfg.count_min          = 120
    vcfg.count_max          = int(3E4)
    vcfg.atol               = 1.E-1 # Decrease this if the solutions are not stable
    vcfg.rtol               = 0.9   # relative tolerence for adjusting the stepsize
    vcfg.pos_cut            = 0
    vcfg.nega_cut           = -1.
    vcfg.loss_eps           = 1e-1
    vcfg.flux_cri           = 0.1

    # Folders
    vcfg.output_dir         =   vulcan_out
    vcfg.plot_dir           =   vulcan_plt
    vcfg.movie_dir          =   vulcan_plt+"/frames/"
    vcfg.out_name           =   VULCAN_NAME

    # Plotting
    vcfg.plot_TP            = config.atmos_chem.vulcan.save_frames
    vcfg.use_live_plot      = config.atmos_chem.vulcan.save_frames
    vcfg.use_save_movie     = config.atmos_chem.vulcan.save_frames
    vcfg.save_movie_rate    = 20
    vcfg.plot_spec          = plt_arr
    vcfg.plot_height        = False
    vcfg.save_evolution     = False
    vcfg.output_humanread   = False

    # for k in vars(vcfg).keys():
    #     print(f"{k}: ", vars(vcfg)[])

    # ------------------------------------------------------------
    # RUN VULCAN
    # ------------------------------------------------------------

    # Make chemical network
    if config.atmos_chem.vulcan.make_funs:
        log.debug("Performing `make_chem_funs` step...")
        vulcan.make_all(vcfg)
        log.debug("    done")

    # Call the solver
    vulcan.main(vcfg)

    # ------------------------------------------------------------
    # READ AND PARSE OUTPUT FILES
    # ------------------------------------------------------------

    result_file = vulcan_out + VULCAN_NAME
    if not os.path.exists(result_file):
        log.warning(f"Could not find output file {result_file}")
        return False

    # load data from file
    with open(result_file,'rb') as hdl:
        result = pickle.load(hdl)
    result_dict = {}

    # read t, p, z, kzz
    result_dict["tmp"] = np.array(result["atm"]["Tco"])
    result_dict["p"]   = np.array(result["atm"]["pco"]) / 10         # convert to Pa
    result_dict["z"]   = np.array(result["atm"]["zco"][:-1]) / 100   # convert to m
    result_dict["Kzz"] = np.array(result["atm"]["Kzz"])
    result_dict["Kzz"] = np.insert(result_dict["Kzz"],0,result_dict["Kzz"][0])

    # read mixing ratios
    result_gas = result["variable"]["species"]
    for i,gas in enumerate(result_gas):
        if "_" not in gas:
            result_dict[gas] = np.array(result["variable"]["ymix"][:,i])

    # write csv file, in a human-readable format
    csv_file = vulcan_out + OUTPUT_NAME
    log.debug(f"Writing to {csv_file}")
    header = ""
    Xarr = []
    for key in result_dict.keys():
        header += str(key).ljust(14, ' ') + "\t"  # compose header
        Xarr.append(list(result_dict[key]))
    Xarr = np.array(Xarr).T  # transpose such that each column is a single variable
    Xarr = Xarr[::-1]        # flip arrays to match AGNI format
    np.savetxt(csv_file, Xarr, delimiter='\t', fmt="%.8e",
                header=header, comments="")

    log.info("    done")
    return success
