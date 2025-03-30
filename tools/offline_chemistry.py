#!/usr/bin/env python3

"""
Code for running offline chemistry based on PROTEUS output.
Configuration options are near the bottom of this file.
"""

from __future__ import annotations

import glob
import os
from textwrap import dedent
import subprocess as sp
import numpy as np
import pandas as pd

# Import PROTEUS
from proteus import Proteus
from proteus.config import Config
from proteus.utils.constants import vol_list, R_sun, AU, element_list
from proteus.utils.helper import find_nearest, safe_rm
from proteus.atmos_clim.common import read_ncdf_profile

def run_once(dirs:dict, config:Config) -> bool:
    """
    Run VULCAN as a subprocess, postprocessing the final PROTEUS output state.
    """

    success = True

    # ------------------------------------------------------------
    # CHECK CONFIG
    # ------------------------------------------------------------

    if config.atmos_clim.module != 'agni':
        print("Offline chemistry is only supported for `atmos_clim.module = 'agni'`")
        return False

    # ------------------------------------------------------------
    # READ DATA FROM PROTEUS RUN
    # ------------------------------------------------------------

    # Get final time for which we have data
    files = glob.glob(os.path.join(dirs["output"], "data", "*_atm.nc"))
    times = [int(f.split("/")[-1].split("_atm.nc")[0]) for f in files]
    year  = sorted(times)[-1]
    print("Locating data for t=%.2e yr..."%year, end='')

    # Read helpfile
    hf_all = pd.read_csv(os.path.join(dirs['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # Get helpfile row for this year
    hf_row = hf_all.iloc[(hf_all['Time']-year).abs().argsort()[:1]]
    hf_row = hf_row.to_dict(orient='records')[0]

    # Read atmosphere data
    atmos = read_ncdf_profile(os.path.join(dirs["output"], "data", "%d_atm.nc"%year),
                              extra_keys=["pl","tmpl", "x_gas", "Kzz"])
    print("  ok")

    # ------------------------------------------------------------
    # WRITE VULCAN INPUT FILES
    # ------------------------------------------------------------

    # Output folder
    vulcan_out = os.path.join(dirs["vulcan"], "output") + "/"
    vulcan_plt = os.path.join(dirs["vulcan"], "plot") + "/"
    print("Writing input data...", end='')

    # Find a reasonable file for the stellar flux
    ls = glob.glob(dirs["output"]+"/data/*.sflux")
    years = [int(f.split("/")[-1].split(".")[0]) for f in ls]
    sflux_fpath = dirs["output"]+"/data/%d.sflux"%find_nearest(years,year)[0]

    # Read spectrum and scale to surface of the star
    sflux_data = np.loadtxt(sflux_fpath, skiprows=1).T
    star_wl = np.array(sflux_data[0])
    star_fl = np.array(sflux_data[1]) * hf_row["separation"]**2 / hf_row["R_star"]**2

    # Remove small values
    min_fl = 1e-20
    star_wl = star_wl[star_fl > min_fl]
    star_fl = star_fl[star_fl > min_fl]

    # Write spectrum
    star_write = vulcan_out + "star.txt"
    np.savetxt(star_write, np.array([star_wl, star_fl]).T,
                header="# WL(nm)    Flux(ergs/cm**2/s/nm)",
                fmt=["%.4f","%.4e"])

    # Write TP profile
    p_arr = np.array(atmos["pl"]) * 10.0 # dyne/cm^2
    t_arr = np.clip(atmos["tmpl"], a_min=180.0, a_max=None)
    header  = "#(dyne/cm2)\t(K)\n"
    header += "Pressure\tTemp"
    prof_write = vulcan_out + "profile.txt"
    np.savetxt(prof_write, np.array([p_arr[::-1], t_arr[::-1]]).T,
               delimiter="\t", header=header, comments='', fmt="%1.5e")
    print("  ok")

    # ------------------------------------------------------------
    # CREATE VULCAN CONFIG
    # ------------------------------------------------------------

    # Which elements are included?
    e_incl = []
    for e in element_list:
        if hf_row[e+"_kg_atm"] > 1e10:
            e_incl.append(e)
    print(f"Elements: {e_incl}")

    # Determine network and species of interest
    network = "CHO_photo_network.txt"
    ele_str = "['H', 'O', 'C']"
    plt_str = "['H2',  'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2']"
    sct_str = "['H2', 'O2']"

    if 'N' in e_incl:
        network = "NCHO_photo_network.txt"
        plt_str = "['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'N2']"
        sct_str = "['H2', 'O2', 'N2']"
        ele_str = "['H', 'O', 'C', 'N']"

        if 'S' in e_incl:
            network = "SNCHO_photo_network.txt"
            plt_str = "['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'SO2', 'H2S', 'S2']"
            ele_str = "['H', 'O', 'C', 'N', 'S']"

    print(f"Using '{network}' ")

    # Parse atmospheric composition into string, work out background gas
    backs = {'H2':0.0, 'N2':0.0, 'O2':0.0, 'CO2':0.0}
    vmr_as_str = ""
    for gas in vol_list:
        vmr = hf_row[gas+"_vmr"]
        if gas in backs.keys():
            backs[gas] = vmr
        if vmr > 1e-8:
            vmr_as_str += "'%s':%.8e, "%(gas, vmr)
    vmr_as_str = vmr_as_str[:-2]

    # Set background gas
    background = max(backs, key=backs.get)
    print(f"Background gas is '{background}'")

    print("Writing VULCAN config...", end='')
    config = f"""\
        # VULCAN CONFIGURATION FILE
        # CREATED AUTOMATICALLY BY PROTEUS

        atom_list               = {ele_str}
        network                 = 'thermo/{network}'
        use_lowT_limit_rates    = False
        gibbs_text              = 'thermo/gibbs_text.txt' # (all the nasa9 files must be placed in the folder: thermo/NASA9/)
        cross_folder            = 'thermo/photo_cross/'
        com_file                = 'thermo/all_compose.txt'

        atm_base                = '{background}'
        rocky                   = True  # for the surface gravity
        nz                      = 80   # number of vertical layers
        P_b                     = {np.amax(atmos["p"])*10}  # pressure at the bottom (dyne/cm^2)
        P_t                     = {np.amin(atmos["p"])*10}  # pressure at the top (dyne/cm^2)
        atm_type                = 'file'
        atm_file                = '{prof_write}'

        sflux_file              = '{star_write}'
        top_BC_flux_file        = 'atm/BC_top.txt' # the file for the top boundary conditions
        bot_BC_flux_file        = 'atm/BC_bot.txt' # the file for the lower boundary conditions

        output_dir              = '{vulcan_out}'
        plot_dir                = '{vulcan_plt}'
        movie_dir               = '{vulcan_plt}/movie/'
        out_name                = 'recent.vul'

        # ====== Setting up the elemental abundance ======
        ini_mix = 'const_mix'
        const_mix = {{ {vmr_as_str} }}


        # ====== Setting up photochemistry ======
        use_ion         = False
        use_photo       = True
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
        use_moldiff = True

        use_vz      = False
        vz_prof     = 'const'  # Options: 'const' or 'file'
        const_vz    = 0 # (cm/s) Only reads when use_vz = True and vz_prof = 'const'

        use_Kzz     = True
        Kzz_prof    = 'Pfunc' # Options: 'const','file' or 'Pfunc' (Kzz increased with P^-0.4)
        const_Kzz   = 1.E10 # (cm^2/s) Only reads when use_Kzz = True and Kzz_prof = 'const'
        K_max       = 1e5        # for Kzz_prof = 'Pfunc'
        K_p_lev     = 0.1      # for Kzz_prof = 'Pfunc'

        update_frq  = 200    # frequency for updating dz and dzi due to change of mu

        # ====== Setting up the boundary conditions ======
        use_topflux     = False
        use_botflux     = False
        use_fix_sp_bot  = {{ {vmr_as_str} }} # fixed mixing ratios at the lower boundary
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
        conv_step = 200

        # ====== Setting up numerical parameters for the ODE solver ======
        ode_solver      = 'Ros2' # case sensitive
        trun_min        = 1e2
        runtime         = 1.E22
        use_print_prog  = True
        use_print_delta = False
        print_prog_num  = 100  # print the progress every x steps
        dttry           = 1.E-3
        dt_min          = 1.E-9
        dt_max          = runtime*1e-4
        dt_var_max      = 2.
        dt_var_min      = 0.5

        count_min       = 120
        count_max       = int(3E4)
        atol            = 1.E-1 # Try decreasing this if the solutions are not stable
        mtol            = 1.E-22
        mtol_conv       = 1.E-20
        pos_cut         = 0
        nega_cut        = -1.
        loss_eps        = 1e-1
        yconv_cri       = 0.01 # for checking steady-state
        slope_cri       = 1.e-4
        yconv_min       = 0.1
        flux_cri        = 0.1
        flux_atol       = 1. # the tol for actinc flux (# photons cm-2 s-1 nm-1)
        conver_ignore   = [] # added 2023. to get rid off non-convergent species, e.g. HC3N without sinks

        # ====== Setting up numerical parameters for Ros2 ODE solver ======
        rtol             = 0.6 # relative tolerence for adjusting the stepsize
        post_conden_rtol = 0.1 # switched to this value after fix_species_time

        # ====== Setting up for output and plotting ======
        plot_TP         = False
        use_live_plot   = True
        use_live_flux   = False
        use_plot_end    = False
        use_plot_evo    = False
        use_save_movie  = False
        use_flux_movie  = False
        plot_height     = False
        use_PIL         = True
        live_plot_frq   = 10
        save_movie_rate = live_plot_frq
        y_time_freq     = 1  #  storing data for every 'y_time_freq' step
        plot_spec       = {plt_str}
        # output:
        output_humanread = True
        use_shark        = False
        save_evolution   = False   # save the evolution of chemistry (y_time and t_time) for every save_evo_frq step
        save_evo_frq     = 10
        """

    config = dedent(config)
    with open(os.path.join(dirs["vulcan"], "vulcan_cfg.py"), 'w') as hdl:
        hdl.write(config)

    print("  ok")

    # ------------------------------------------------------------
    # RUN VULCAN
    # ------------------------------------------------------------

    cmd = [f"cd {dirs["vulcan"]}; python vulcan.py"]
    logfile = vulcan_out + "offline.log"

    with open(logfile,'w') as hdl:
        print("Running VULCAN...")
        proc = sp.run(cmd, shell=True, stdout=hdl, stderr=hdl)

    print("Return code: " + str(proc.returncode))
    if proc.returncode:
        success = False

    # ------------------------------------------------------------
    # MAKE PLOTS
    # ------------------------------------------------------------

    # Outsource?

    print("Done!")
    return success

# If run directly
if __name__ == '__main__':

    # Parameters
    cfgfile =       "output/default/init_coupler.toml"  # Config file used for PROTEUS

    # Read config and dirs
    handler = Proteus(config_path=cfgfile)
    config  = handler.config
    dirs    = handler.directories

    # Do it
    run_once(dirs, config)

