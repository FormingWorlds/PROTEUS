# Run VULCAN chemical kinetics module
from __future__ import annotations

import glob
import logging
import os
import pickle
import shutil
import subprocess as sp
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

import numpy as np

# Import PROTEUS
from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.constants import AU, R_sun, element_list, vol_list
from proteus.utils.helper import find_nearest

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

VULCAN_NAME = "recent.vul"

def run_vulcan_offline(dirs:dict, config:Config, hf_row:dict) -> bool:
    """
    Run VULCAN as a subprocess, postprocessing the final PROTEUS output state.
    """

    success = True

    # ------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------

    if config.atmos_clim.module != 'agni':
        log.warning("VULCAN offline chemistry only supported with AGNI")
        return False

    # make folder
    work_dir = os.path.join(dirs["output"],"offchem") + "/"
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)

    # ------------------------------------------------------------
    # READ DATA FROM PROTEUS RUN
    # ------------------------------------------------------------

    # Get all times for which we have data
    year = hf_row["Time"]
    log.debug("Reading data for t=%.2e yr"%year)

    # Read atmosphere data
    ncdf_path = os.path.join(dirs["output"], "data", "%d_atm.nc"%year)
    atmos = read_ncdf_profile(ncdf_path, extra_keys=["pl","tmpl", "x_gas", "Kzz"])

    # ------------------------------------------------------------
    # WRITE VULCAN INPUT FILES
    # ------------------------------------------------------------

    # Output folder
    vulcan_out = work_dir
    vulcan_plt = work_dir
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
    star_write = work_dir + "star.dat"
    np.savetxt(star_write, np.array([star_wl, star_fl]).T,
                header="# WL(nm)    Flux(ergs/cm**2/s/nm)",
                fmt=["%.4f","%.4e"])

    # Get TP profile
    p_arr = np.array(atmos["pl"]) * 10.0 # dyne/cm^2
    t_arr = np.clip(atmos["tmpl"], a_min=180.0, a_max=None)

    # Get Kzz profile
    k_arr = np.array(atmos["Kzz"]) * 1e4 # cm^2/s

    # Write TPK profile
    num_levels = len(p_arr)
    header  = "#(dyne/cm2)\t(K)\t(cm2/s)\n"
    header += "Pressure\tTemp\tKzz"
    prof_write = work_dir + "profile.dat"
    np.savetxt(prof_write, np.array([p_arr[::-1], t_arr[::-1], k_arr[::-1]]).T,
               delimiter="\t", header=header, comments='', fmt="%1.5e")

    # Write mixing ratios
    vmr_write = work_dir + "vmrs.dat"
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

    # Which elements are included?
    e_incl = []
    for e in element_list:
        if hf_row[e+"_kg_atm"] > 1e10:
            e_incl.append(e)
    log.debug(f"Found elements: {e_incl}")

    # Determine network and species of interest
    match config.atmos_chem.vulcan.network:
        case "CHO":
            ele_str = "['H', 'O', 'C']"
            plt_str = "['H2',  'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2']"
            sct_str = "['H2', 'O2', 'CO2']"
            if config.atmos_chem.photo_on:
                network_file = "CHO_photo_network.txt"
            else:
                network_file = "CHO_thermo_network.txt"

        case "NCHO":
            plt_str = "['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'N2']"
            sct_str = "['H2', 'O2', 'N2', 'CO2']"
            ele_str = "['H', 'O', 'C', 'N']"
            if config.atmos_chem.photo_on:
                network_file = "NCHO_photo_network.txt"
            else:
                network_file = "NCHO_thermo_network.txt"

        case "SNCHO":
            plt_str = "['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'SO2', 'H2S', 'S2', 'S8']"
            sct_str = "['H2', 'O2', 'N2', 'CO2']"
            ele_str = "['H', 'O', 'C', 'N', 'S']"
            if config.atmos_chem.photo_on:
                network_file = "SNCHO_photo_network.txt"
            else:
                raise ValueError("Photochemistry is required when using sulfur network")

        case _:
            log.error(f"Network not recognised: '{config.atmos_chem.vulcan.network}'")
            return False

    log.debug(f"Using '{network_file}' ")

    # Parse atmospheric composition into string, work out background gas
    backs = {'H2':0.0, 'N2':0.0, 'O2':0.0, 'CO2':0.0}
    outgas_str = ""
    for gas in vol_list:
        vmr = hf_row[gas+"_vmr"]
        if gas in backs.keys():
            backs[gas] = vmr
        if vmr > config.atmos_chem.vulcan.clip_vmr:
            outgas_str += "'%s':%.8e, "%(gas, vmr)
    outgas_str = outgas_str[:-2]

    # Surface boundary condition
    if config.atmos_chem.vulcan.fix_surf:
        fixsurf_str = outgas_str
    else:
        fixsurf_str = ""

    # Set background gas
    background = max(backs, key=backs.get)
    log.debug(f"Background gas is '{background}'")

    # Kzz value if constant

    if config.atmos_chem.Kzz_const is not None:
        Kzz_src = "const"
        Kzz_val = config.atmos_chem.Kzz_const
    else:
        # if None, will get Kzz from profile
        Kzz_src = "file"
        Kzz_val = 1e5 # <- dummy value, will be replaced by profile

    log.debug("Writing VULCAN config")
    vulcan_config = f"""\
        # VULCAN CONFIGURATION FILE
        # CREATED AUTOMATICALLY BY PROTEUS

        atom_list               = {ele_str}
        network                 = 'thermo/{network_file}'
        use_lowT_limit_rates    = False
        gibbs_text              = 'thermo/gibbs_text.txt' # (all the nasa9 files must be placed in the folder: thermo/NASA9/)
        cross_folder            = 'thermo/photo_cross/'
        com_file                = 'thermo/all_compose.txt'

        atm_base                = '{background}'
        rocky                   = True           # for the surface gravity
        nz                      = {num_levels}   # number of vertical layers
        P_b                     = {np.amax(atmos["p"])*10}  # pressure at the bottom (dyne/cm^2)
        P_t                     = {np.amin(atmos["p"])*10}  # pressure at the top (dyne/cm^2)
        atm_type                = 'file'
        atm_file                = '{prof_write}'

        sflux_file              = '{star_write}'
        top_BC_flux_file        = 'atm/BC_top.txt' # the file for the top boundary conditions
        bot_BC_flux_file        = 'atm/BC_bot.txt' # the file for the lower boundary conditions

        output_dir              = '{vulcan_out}'
        plot_dir                = '{vulcan_plt}'
        movie_dir               = '{vulcan_plt}/frames/'
        out_name                = '{VULCAN_NAME}'

        # ====== Setting up the elemental abundance ======
        ini_mix = '{config.atmos_chem.vulcan.ini_mix}'
        const_mix = {{ {outgas_str} }}
        vul_ini = '{vmr_write}'


        # ====== Setting up photochemistry ======
        use_ion         = False
        use_photo       = {config.atmos_chem.photo_on}
        r_star          = {hf_row["R_star"] / R_sun}     # stellar radius (R_sun)
        Rp              = {hf_row["R_int"] * 100.0}      # Planetary radius (cm)
        orbit_radius    = {hf_row["separation"] / AU}    # planet-star distance in A.U.
        gs              = {hf_row["gravity"] * 100}      # surface gravity (cm/s^2)  (HD189:2140  HD209:936)
        sl_angle        = {config.orbit.zenith_angle * np.pi / 180.0}   # the zenith angle
        f_diurnal       = {config.orbit.s0_factor}
        scat_sp         = {sct_str}
        T_cross_sp      = []

        edd             = 0.5 # the Eddington coefficient
        dbin1           = 0.1  # the uniform bin width < dbin_12trans (nm)
        dbin2           = 2.   # the uniform bin width > dbin_12trans (nm)
        dbin_12trans    = 240. # the wavelength switching from dbin1 to dbin2 (nm)

        # the frequency to update the actinic flux and optical depth
        ini_update_photo_frq    = 100
        final_update_photo_frq  = 5

        # ====== Mixing processes ======
        use_moldiff = {config.atmos_chem.moldiff_on}

        use_vz      = True
        vz_prof     = 'const'  # Options: 'const' or 'file'
        const_vz    = {config.atmos_chem.updraft_const} # (cm/s)

        use_Kzz     = {config.atmos_chem.Kzz_on}
        Kzz_prof    = '{Kzz_src}' # Options: 'const','file'
        const_Kzz   = {Kzz_val} # Only reads when Kzz_prof = 'const'
        K_max       = 1e5        # for Kzz_prof = 'Pfunc'
        K_p_lev     = 0.1      # for Kzz_prof = 'Pfunc'

        update_frq  = 50    # frequency for updating dz and dzi due to change of mu

        # ====== Setting up the boundary conditions ======
        use_topflux     = False
        use_botflux     = False
        use_fix_sp_bot  = {{ {fixsurf_str} }} # fixed mixing ratios at the lower boundary
        diff_esc        = [] # species for diffusion-limit escape at TOA
        max_flux        = 1e13  # upper limit for the diffusion-limit fluxes

        # ====== Reactions to be switched off  ======
        remove_list = [] # in pairs e.g. [1,2]

        # == Condensation ======
        use_condense        = False
        use_settling        = False
        start_conden_time   = 1e10
        condense_sp         = []
        non_gas_sp          = []
        fix_species         = []      # fixed the condensable species after condensation-evapoation EQ has reached
        fix_species_time    = 0  # after this time to fix the condensable species

        # ====== steady state check ======
        st_factor = 0.5
        conv_step = 100

        # ====== Setting up numerical parameters for the ODE solver ======
        ode_solver      = 'Ros2' # case sensitive
        trun_min        = 1e2
        runtime         = 1.E22
        use_print_prog  = True
        use_print_delta = False
        print_prog_num  = 20  # print the progress every x steps
        dttry           = 1.E-6
        dt_min          = 1.E-8
        dt_max          = runtime*1e-4
        dt_var_max      = 2.
        dt_var_min      = 0.5

        count_min       = 120
        count_max       = int(3E4)
        atol            = 5.E-2 # Try decreasing this if the solutions are not stable
        mtol            = 1.E-22
        mtol_conv       = 1.E-20
        pos_cut         = 0
        nega_cut        = -1.
        loss_eps        = 1e-1
        yconv_cri       = 0.04  # for checking steady-state
        slope_cri       = 1.e-4
        yconv_min       = 0.5
        flux_cri        = 0.1
        flux_atol       = 1. # the tol for actinc flux (# photons cm-2 s-1 nm-1)
        conver_ignore   = [] # added 2023. to get rid off non-convergent species, e.g. HC3N without sinks

        # ====== Setting up numerical parameters for Ros2 ODE solver ======
        rtol             = 0.7 # relative tolerence for adjusting the stepsize
        post_conden_rtol = 0.1 # switched to this value after fix_species_time

        # ====== Setting up for output and plotting ======
        plot_TP         = {config.atmos_chem.vulcan.save_frames}
        use_live_plot   = {config.atmos_chem.vulcan.save_frames}
        use_live_flux   = False
        use_plot_end    = False
        use_plot_evo    = False
        use_save_movie  = {config.atmos_chem.vulcan.save_frames}
        use_flux_movie  = False
        plot_height     = False
        use_PIL         = False
        live_plot_frq   = 50
        save_movie_rate = live_plot_frq
        y_time_freq     = 1  #  storing data for every 'y_time_freq' step
        plot_spec       = {plt_str}
        # output:
        output_humanread = False
        use_shark        = False
        save_evolution   = False   # save the evolution of chemistry (y_time and t_time) for every save_evo_frq step
        save_evo_frq     = 10
        """


    # Write config file
    vulcan_fpath  = work_dir + "vulcan_cfg.txt"
    with open(vulcan_fpath, 'w') as hdl:
        hdl.write(dedent(vulcan_config))

    # Copy config file to VULCAN directory
    shutil.copyfile(vulcan_fpath, os.path.join(dirs["vulcan"], "vulcan_cfg.py")),

    # ------------------------------------------------------------
    # RUN VULCAN
    # ------------------------------------------------------------

    cmd = [sys.executable, "vulcan.py"]
    if not config.atmos_chem.vulcan.make_funs:
        cmd.append(" -n")
        log.debug("Skipping VULCAN make_chem_funs step")

    logfile = vulcan_out + "recent.log"

    log.info("Running VULCAN subprocess")
    with open(logfile, 'w') as hdl:
        proc = sp.run(cmd, cwd=dirs["vulcan"], stdout=hdl, stderr=hdl, bufsize=1, text=True)

    log.debug("Return code: " + str(proc.returncode))
    if proc.returncode:
        success = False
        log.error("VULCAN subprocess failed")

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
    result_dict["kzz"] = np.array(result["atm"]["Kzz"])
    result_dict["kzz"] = np.insert(result_dict["kzz"],0,result_dict["kzz"][0])

    # read mixing ratios
    result_gas = result["variable"]["species"]
    for i,gas in enumerate(result_gas):
        result_dict[gas] = np.array(result["variable"]["ymix"][:,i])

    # write csv file
    csv_file = result_file + ".csv"
    log.debug(f"Writing to {csv_file}")
    header = ""
    Xarr = []
    for key in result_dict.keys():
        header += str(key).ljust(14, ' ') + "\t"  # compose header
        Xarr.append(list(result_dict[key]))
    Xarr = np.array(Xarr).T  # transpose such that column=variable
    Xarr = Xarr[::-1]        # flip arrays to match AGNI format
    np.savetxt(csv_file, Xarr, delimiter='\t', fmt="%.8e",
                header=header, comments="")

    log.info("    done")
    return success
