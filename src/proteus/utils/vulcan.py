# Function used to run VULCAN in online mode
from __future__ import annotations

import os
import pickle as pkl
import re
import shutil
import subprocess

import numpy as np

from proteus.utils.coupler import *


def RunVULCAN( atm, time, loop_counter, dirs, runtime_helpfile, OPTIONS ):
    '''
    Run online atmospheric chemistry via VULCAN kinetics

    This function is deprecated.
    '''

    # Runtime info
    print("VULCAN run... (loop =", loop_counter, ")")

    # Copy template file
    vul_cfg = dirs["vulcan"]+"vulcan_cfg.py"                    # Template configuration file for VULCAN
    if os.path.exists(vul_cfg):
        os.remove(vul_cfg)
    shutil.copyfile(dirs["utils"]+"init_vulcan_cfg.py",vul_cfg)

    # Delete old PT profile
    vul_ptp = dirs["vulcan"]+"output/PROTEUS_PT_input.txt"      # Path to input PT profile
    if os.path.exists(vul_ptp):
        os.remove(vul_ptp)


    hf_recent = runtime_helpfile.iloc[-1]

    # Write missing parameters to cfg file
    with open(vul_cfg, 'a') as vcf:

        vcf.write("# < PROTEUS INSERT > \n")

        # System/planet parameters
        vcf.write("Rp = %1.5e \n"           % (atm.planet_radius*100.0))        # Radius [cm]
        vcf.write("gs = %g \n"              % (atm.grav_s*100.0))               # Surface gravity [cm/s^2]
        vcf.write("sl_angle = %g \n"        % (atm.zenith_angle*3.141/180.0))   # Solar zenith angle [rad]
        vcf.write("orbit_radius = %1.5e \n" % OPTIONS["mean_distance"]) # Semi major axis [AU]
        vcf.write("r_star = %1.5e \n"       % OPTIONS["star_radius"])   # Star's radius [R_sun]

        # Set background gas based on gas with highest mixing ratio from list of options
        bg_gas = np.array([ 'H2', 'N2', 'O2', 'CO2' ])
        bg_val = np.array([ hf_recent["%s_mr"%gas] for gas in bg_gas ])  # Get values
        bg_mask = np.argsort(bg_val)  # Get sorting mask
        bg_val = bg_val[bg_mask]   # Sort gas values
        bg_gas = bg_gas[bg_mask]   # Sort gas names
        if (bg_val[-1] < 2.0*bg_val[-2]) or (bg_val[-1] < 0.5):
            print("Warning: Background gas '%s' is not significantly abundant!"%bg_gas[-1])
            print("         Mixing ratio of '%s' is %g" % (bg_gas[-1],bg_val[-1]))
            print("         Mixing ratio of '%s' is %g" % (bg_gas[-2],bg_val[-2]))
        vcf.write("atm_base = '%s' \n"      % str(bg_gas[-1]))

        # Pressure grid limits
        vcf.write("P_b = %1.5e \n"          % float(hf_recent["P_surf"]*1.0e6))        # pressure at the bottom (dyne/cm^2)
        vcf.write("P_t = %1.5e \n"          % float(OPTIONS["P_top"]*1.0e6))   # pressure at the top (dyne/cm^2)

        # Plotting behaviour
        vcf.write("use_live_plot  = %s \n"  % str(bool(OPTIONS["plot_iterfreq"] > 0)))

        # Make copy of element_list as a set, since it'll be used a lot in the code below
        set_elem_list = set(element_list)

        # Rayleigh scattering gases
        rayleigh_candidates = ['N2','O2', 'H2']
        rayleigh_str = ""
        for rc in rayleigh_candidates:
            if set(re.sub('[1-9]', '', rc)).issubset(set_elem_list):  # Remove candidates which aren't supported by elem_list
                rayleigh_str += "'%s',"%rc
        rayleigh_str = rayleigh_str[:-1]
        vcf.write("scat_sp = [%s] \n" % rayleigh_str)

        # Gases for diffusion-limit escape at TOA
        # escape_candidates = ['H2','H']
        # escape_str = ""
        # for ec in escape_candidates:
        #     if set(re.sub('[1-9]', '', ec)).issubset(set_elem_list):  # Remove candidates which aren't supported by elem_list
        #         escape_str += "'%s',"%ec
        # escape_str = escape_str[:-1]
        # vcf.write("diff_esc = [%s] \n" % escape_str)

        # Atom list
        atom_str = ""
        for elem in element_list:
            atom_str += "'%s',"%elem
        atom_str = atom_str[:-1]
        vcf.write("atom_list  = [%s] \n" % atom_str)

        # Choose most appropriate chemical network and species-to-plot
        oxidising = False

        for inert in ['He','Xe','Ar','Ne','Kr']:  # Remove inert elements from list, since they don't matter for reactions
            set_elem_list.discard(inert)

        if set_elem_list == {'H','C','O'}:
            net_str = 'thermo/CHO_photo_network.txt'
            plt_spe = ['H2', 'H', 'H2O', 'C2H2', 'CH4', 'CO2']

        elif set_elem_list == {'N','H','C','O'}:
            if oxidising:
                net_str = 'thermo/NCHO_full_photo_network.txt'
                plt_spe = ['N2', 'O2', 'H2', 'H2O', 'NH3', 'CH4', 'CO', 'O3']
            else:
                net_str = 'thermo/NCHO_photo_network.txt'
                plt_spe = ['H2', 'H', 'H2O', 'OH', 'CH4', 'HCN', 'N2', 'NH3']

        elif set_elem_list == {'S','N','H','C','O'}:
            if oxidising:
                net_str = 'thermo/SNCHO_full_photo_network.txt'
                plt_spe = ['O2', 'N2', 'O3', 'H2', 'H2O', 'NH3', 'CH4', 'SO2', 'S']
            else:
                net_str = 'thermo/SNCHO_photo_network.txt'
                plt_spe = ['N2', 'H2', 'S', 'H', 'OH', 'NH3', 'CH4', 'HCN']

        vcf.write("network  = '%s' \n" % net_str)
        plt_str = ""
        for spe in plt_spe:
            plt_str += "'%s'," % spe
        vcf.write("plot_spec = [%s] \n" % plt_str[:-1])

        # Bottom boundary mixing ratios are fixed according to SPIDER (??)
        # fix_bb_mr = "{"
        # for v in volatile_species:
        #     fix_bb_mr += " '%s' : %1.5e ," % (v,hf_recent["%s_mr"%v])
        # fix_bb_mr = fix_bb_mr[:-1]+" }"
        # vcf.write("use_fix_sp_bot = %s \n"   % str(fix_bb_mr))
        vcf.write("use_fix_sp_bot = {} \n")


        # Has NOT run atmosphere before
        if (loop_counter["atm"] == 0):

            # PT profile
            # vcf.write("atm_type = 'isothermal' \n")
            # vcf.write("Tiso = %g \n"        % float(OPTIONS["T_eqm"]))

            # Abundances
            vcf.write("ini_mix = 'const_mix' \n")  # other options: 'EQ', 'table', 'vulcan_ini'
            const_mix = "{"
            for v in volatile_species:
                const_mix += " '%s' : %1.5e ," % (v,hf_recent["%s_mr"%v])
            const_mix = const_mix[:-1]+" }"
            vcf.write("const_mix = %s \n"   % str(const_mix))

        # Has run atmosphere before
        # else:

            # PT Profile
            # vcf.write("atm_type = 'file' \n")
            # vcf.write("atm_file = '%s' \n" % vul_ptp)

            # Abundances
            # vcf.write("ini_mix = 'vulcan_ini' \n")
            # vcf.write("vul_ini = 'output/PROTEUS_MX_input.vul' \n")
            # ! WRITE ABUNDANCES HERE


        vcf.write("# </ PROTEUS INSERT > \n")
        vcf.write(" ")

    # Write PT profile (to VULCAN, from JANUS)
    vul_PT = np.array(
        [np.array(atm.pl)  [::-1] * 10.0,
         np.array(atm.tmpl)[::-1]
        ]
    ).T
    header = "#(dyne/cm2)\t (K) \n Pressure\t Temp"
    np.savetxt(vul_ptp,vul_PT,delimiter="\t",header=header,comments='',fmt="%1.5e")

    # Write spectrum (using most recent output spectrum from $COUPLERDIR/output)
    # Spectra in output are scaled to 1 AU, but VULCAN wants it at the star's surface.
    print("")
    print("WARNING: VULCAN IS NOT YET USING THE CORRECT SPECTRUM!")
    print("")


    # Switch to VULCAN directory, run VULCAN, switch back to main directory
    vulcan_run_cmd = "python vulcan.py"
    if (loop_counter["atm"] > 0):      # If not first run, skip building chem_funcs
        vulcan_run_cmd += " -n"

    vulcan_print = open(dirs["output"]+"vulcan_recent.log",'w')

    os.chdir(dirs["vulcan"])
    subprocess.run([vulcan_run_cmd], shell=True, check=True, stdout=vulcan_print)
    os.chdir(dirs["proteus"])

    vulcan_print.close()

    # Copy VULCAN output data file to output folder
    vulcan_recent = dirs["output"]+str(int(time))+"_atm_chemistry.vul"
    shutil.copyfile(dirs["vulcan"]+'output/PROTEUS_MX_output.vul', vulcan_recent )

    # Read in data from VULCAN output
    with (open(vulcan_recent, "rb")) as vof:
        vul_data = pkl.load(vof)

    print(vul_data)

    return atm
