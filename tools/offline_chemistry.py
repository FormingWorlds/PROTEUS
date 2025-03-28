#!/usr/bin/env python3

"""
Code for running offline chemistry based on PROTEUS output.
Configuration options are near the bottom of this file.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess as sp
import numpy as np
import pandas as pd

# Import PROTEUS
from proteus import Proteus
from proteus.plot._cpl_helpers import get_handler_from_argv
from proteus.utils.constants import AU_cm, R_sun_cm, element_list, vol_list
from proteus.utils.helper import find_nearest, safe_rm
from proteus.atmos_clim.common import read_ncdf_profile

# Constants
mixing_ratio_floor = 1e-50      # Minimum mixing ratio (at all)
mixing_ratio_fixed = 1e-0       # Species with mixing ratio greater than this value have fixed abundances at the surface


def run_once(dirs:dict, ini_method:int) -> bool:
    """
    Run VULCAN as a subprocess, postprocessing the final PROTEUS output state.
    """

    success = True

    # ------------------------------------------------------------
    # READ DATA FROM PROTEUS RUN
    # ------------------------------------------------------------

    # Get final time for which we have data
    files = glob.glob(os.path.join(dirs["output"], "data", "*_atm.nc"))
    times = [int(f.split("/")[-1].split("_atm.nc")[0]) for f in files]
    year  = sorted(times)[-1]

    # Read helpfile
    hf_all = pd.read_csv(os.path.join(dirs['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # Get helpfile row for this year
    hf_row = hf_all.iloc[(hf_all['Time']-year).abs().argsort()[:1]]
    hf_row = hf_row.to_dict(orient='records')[0]

    # Read atmosphere data
    atmos = read_ncdf_profile(os.path.join(dirs["output"], "data", "%d_atm.nc"%year))

    # Parse atmospheric composition into dict
    x_gas = {}
    for gas in vol_list:
        x_gas[gas] = hf_row[gas+"_vmr"]

    # Get pressure limits [Pa]
    p_bot = np.max(atmos.p)
    p_top = np.min(atmos.p)

    # ------------------------------------------------------------
    # WRITE VULCAN INPUT FILES
    # ------------------------------------------------------------

    # Copy a reasonable stellar flux file to the location where VULCAN will
    #   read it. This is then scaled to the stellar surface
    ls = glob.glob(dirs["output"]+"/data/*.sflux")
    years = [int(f.split("/")[-1].split(".")[0]) for f in ls]
    sflux_near = dirs["output"]+"/data/%d.sflux"%find_nearest(years,year)[0]

    sflux_read = np.loadtxt(sflux_near, skiprows=1).T
    f_sf = float(hf_row["R_star"]) * R_sun_cm / (OPTIONS["semimajoraxis"] * AU_cm)
    sflux_scaled = sflux_read[1] * f_sf * f_sf       # scale to surface of star

    sflux_write = [sflux_read[0], sflux_scaled]
    np.savetxt(dirs["vulcan"]+"output/%s_SF.txt"%screen_name, np.array(sflux_write).T, header="Stellar spectrum at R_star")

    # Write PT profile
    minT = 120.0
    tmpl = np.clip(tmpl,minT,None)
    if np.any(np.array(tmpl) < minT):
        logger.warn("Temperature is unreasonably low (< %f)!" % minT)
    vul_PT = np.array(
        [np.array(pl)  [::-1] * 10.0,
         np.array(tmpl)[::-1]
        ]
    ).T
    header = "#(dyne/cm2)\t (K) \n Pressure\t Temp"
    np.savetxt(     dirs["vulcan"]+"output/%s_PT.txt"%screen_name, vul_PT,  delimiter="\t",header=header,comments='',fmt="%1.5e")
    shutil.copyfile(dirs["vulcan"]+"output/%s_PT.txt"%screen_name, dirs["output"]+"offchem/%d/PT.txt"%year)

    # ------------------------------------------------------------
    # DEFINE VULCAN VARIABLES
    # ------------------------------------------------------------

    atom_list               = ['H', 'O', 'C', 'N']
    network                 = 'thermo/NCHO_photo_network.txt'
    use_lowT_limit_rates    = False
    gibbs_text              = 'thermo/gibbs_text.txt' # (all the nasa9 files must be placed in the folder: thermo/NASA9/)
    cross_folder            = 'thermo/photo_cross/'
    com_file                = 'thermo/all_compose.txt'
    atm_type                = 'file'  # Options: 'isothermal', 'analytical', 'file', or 'vulcan_ini' 'table'
    atm_file                = 'atm/atm_HD189_Kzz.txt' # TP and Kzz (optional) file
    sflux_file              = 'atm/stellar_flux/sflux-HD189_Moses11.txt' # sflux-HD189_B2020.txt This is the flux density at the stellar surface
    top_BC_flux_file        = 'atm/BC_top.txt' # the file for the top boundary conditions
    bot_BC_flux_file        = 'atm/BC_bot.txt' # the file for the lower boundary conditions
    output_dir              = 'VULCAN/output/'
    plot_dir                = 'plot/'
    movie_dir               = 'plot/movie/'
    out_name                =  'HD189.vul' # output file name

    # ====== Setting up the elemental abundance ======
    ini_mix = 'const_mix' # Options: 'EQ', 'const_mix', 'vulcan_ini', 'table' (for 'vulcan_ini, the T-P grids have to be exactly the same)
    const_mix = {'CH4':2.7761E-4*2, 'O2':4.807e-4, 'He':0.09691, 'N2':8.1853E-5, 'H2':1. -2.7761E-4*2*4/2}

    # ====== Setting up photochemistry ======
    use_photo       = True
    r_star          = 0.805 # stellar radius in solar radius
    Rp              = 1.138*7.1492E9 # Planetary radius (cm) (for computing gravity)
    orbit_radius    = 0.03142 # planet-star distance in A.U.
    sl_angle        = 48 /180.*3.14159 # the zenith angle of the star in degree (usually 58 deg for the dayside average)
    f_diurnal       = 1. # to account for the diurnal average of solar flux (i.e. 0.5 for Earth; 1 for tidally-locked planets)
    scat_sp         = ['H2', 'He'] # the bulk gases that contribute to Rayleigh scattering
    T_cross_sp      = [] # warning: slower start! available atm: 'CO2','H2O','NH3', 'SH','H2S','SO2', 'S2', 'COS', 'CS2'

    edd             = 0.5 # the Eddington coefficient
    dbin1           = 0.1  # the uniform bin width < dbin_12trans (nm)
    dbin2           = 2.   # the uniform bin width > dbin_12trans (nm)
    dbin_12trans    = 240. # the wavelength switching from dbin1 to dbin2 (nm)

    # the frequency to update the actinic flux and optical depth
    ini_update_photo_frq    = 100
    final_update_photo_frq  = 5

    # ====== Setting up ionchemistry ======
    use_ion = False

    # ====== Setting up parameters for the atmosphere ======
    atm_base    = 'H2' #Options: 'H2', 'N2', 'O2', 'CO2 -- the bulk gas of the atmosphere: changes the molecular diffsion, thermal diffusion factor, and settling velocity
    rocky       = False # for the surface gravity
    nz          = 150   # number of vertical layers
    P_b         = 1e9  # pressure at the bottom (dyne/cm^2)
    P_t         = 1e-2 # pressure at the top (dyne/cm^2)

    use_moldiff = True
    use_vz      = False
    vz_prof     = 'const'  # Options: 'const' or 'file'

    use_Kzz     = True
    Kzz_prof    = 'const' # Options: 'const','file' or 'Pfunc' (Kzz increased with P^-0.4)
    const_Kzz   = 1.E10 # (cm^2/s) Only reads when use_Kzz = True and Kzz_prof = 'const'
    const_vz    = 0 # (cm/s) Only reads when use_vz = True and vz_prof = 'const'
    K_max       = 1e5        # for Kzz_prof = 'Pfunc'
    K_p_lev     = 0.1      # for Kzz_prof = 'Pfunc'

    gs          = 2140.         # surface gravity (cm/s^2)  (HD189:2140  HD209:936)
    update_frq  = 100    # frequency for updating dz and dzi due to change of mu

    # ====== Setting up the boundary conditions ======
    use_topflux     = False
    use_botflux     = False
    use_fix_sp_bot  = {} # fixed mixing ratios at the lower boundary
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
    conv_step = 500

    # ====== Setting up numerical parameters for the ODE solver ======
    ode_solver      = 'Ros2' # case sensitive
    use_print_prog  = True
    use_print_delta = False
    print_prog_num  = 500  # print the progress every x steps
    dttry           = 1.E-10
    dt_min          = 1.E-14
    dt_max          = runtime*1e-5
    dt_var_max      = 2.
    dt_var_min      = 0.5

    trun_min        = 1e2
    runtime         = 1.E22

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
    rtol             = 0.2 # relative tolerence for adjusting the stepsize
    post_conden_rtol = 0.1 # switched to this value after fix_species_time

    # ====== Setting up for output and plotting ======
    plot_TP         = False
    use_live_plot   = False
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
    plot_spec       = ['H2O', 'H', 'CH4', 'CO', 'CO2', 'C2H2', 'HCN', 'NH3' ]
    # output:
    output_humanread = True
    use_shark        = False
    save_evolution   = False   # save the evolution of chemistry (y_time and t_time) for every save_evo_frq step
    save_evo_frq     = 10

    # ------------------------------------------------------------
    # RUN VULCAN
    # ------------------------------------------------------------

    # As subprocess

    # ------------------------------------------------------------
    # MAKE PLOTS
    # ------------------------------------------------------------

    # Outsource?

    return success

# If run directly
if __name__ == '__main__':

    # Parameters
    cfgfile =       "output/l9859d/init_coupler.toml"  # Config file used for PROTEUS
    mkplots =       True                # make plots?
    ini_method =    0                   # Method used to init VULCAN abundances  (0: const_mix, 1: eqm)

    # Read config and dirs
    handler = Proteus(config_path=cfgfile)

    # Do it
    run_once(handler.directories, ini_method, elements, network)

