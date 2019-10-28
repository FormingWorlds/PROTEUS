"""
Coupler Main file – SPIDER-SOCRATES-VULCAN
"""

import numpy as np
# import GGRadConv
import SocRadConv
import SocRadModel
import subprocess
import glob
import sys, os, shutil
import math
from natsort import natsorted #https://pypi.python.org/pypi/natsort
import coupler_utils
import spider_coupler_utils as su
from datetime import datetime
import pandas as pd

# Define specific directories
coupler_dir = os.getcwd()+"/"
output_dir  = os.getcwd()+"/output/"
vulcan_dir  = os.getcwd()+"/vulcan_spider/"

# SPIDER settings
SPIDER_options = {
    'SURFACE_BC':            4,       # 4: constant heat flux boundary condition
    'SOLVE_FOR_VOLATILES':   1,       # track volatiles in interior/atmosphere reservoirs
    'tsurf_poststep_change': 100.0,   # maximum absolute surface temperature change in Kelvin
    'R_solid_planet':        6371000, # planet radius / m
    'planet_coresize':       0.55,    # fractional radius of core-mantle boundary
    'start_condition':       1,       # 1: Fresh start | 2: Restart from 'restart_file'
    'restart_filename':      0,       # Restart manually, [yr]+".json"
    'nstepsmacro_init':      0,       # init loop: number of timesteps
    'dtmacro_init':          1,       # init loop: delta time per macrostep [yr]
    'dtmacro':               50000,   # delta time per macrostep to advance by, in years
    'heat_flux':             1.0E4,   # init heat flux, re-calculated during runtime [W/m^2]
    'H2O_ppm':               50,      # init loop: H2O mass relative to mantle [ppm wt]
    'CO2_ppm':               100,     # [ppm wt]
    'H2_ppm':                0,       # [ppm wt]
    'CH4_ppm':               0,       # [ppm wt]
    'CO_ppm':                0,       # [ppm wt]
    'N2_ppm':                0,       # [ppm wt]
    'O2_ppm':                0,       # [ppm wt]
    'S_ppm':                 0,       # [ppm wt]
    'He_ppm':                0,       # [ppm wt]
    'H2O_poststep_change':   0.05,    # fractional H2O melt phase change event trigger
    'CO2_poststep_change':   0.05,    # CO2 melt phase trigger
    'H2_poststep_change':    0.00,    # fraction
    'CH4_poststep_change':   0.00,    # fraction
    'CO_poststep_change':    0.00,    # fraction
    'N2_poststep_change':    0.00,    # fraction
    'O2_poststep_change':    0.00,    # fraction
    'S_poststep_change':     0.00,    # fraction
    'He_poststep_change':    0.00,    # fraction
    }

# Total runtime
time_current          = 0.           # yr
time_target           = 1.0e+6       # yr

# Resart file if start_condition == 2
restart_file = ""

# Define runtime helpfile names and generate dataframes
runtime_helpfile_name = "runtime_helpfile.csv"

# Planetary and magma ocean start configuration
star_mass             = 1.0          # M_sol options: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4
mean_distance         = 1.0          # AU, star-planet distance
time_offset           = 100.         # Myr, start of magma ocean after star formation

# Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
loop_counter = { "total": 0, "init": 0, "atm": 0 }

# Equilibration loops for init and atmosphere loops
init_loops = 2
atm_loops  = 2

# If restart skip init loop
if SPIDER_options["start_condition"] == 2:
    loop_counter["total"] += 1
    loop_counter["init"]  += 1
    loop_counter["atm"]   += 1

# Inform about start of runtime
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print("::::::::::::: START INIT LOOP |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

### TO DO LIST ###
# - Feed atmospheric profile to SOCRATES instead of surface abundances
# - In "ModifiedHenrysLaw" function, take mass conservation from atm chemistry --> calc mass of atmosphere from atm chemistry and insert into function. So far, mass conservation is calculated from former .json file, which may be inconsistent.
# - Fix Psurf stuff.
# - Use Height calculation of updated VULCAN versions instead of own function.
# - Check if "Magma ocean volatile content" needs ot be altered in JSON

if SPIDER_options["start_condition"] == 1:

    # Delete old SPIDER output
    coupler_utils.CleanOutputDir( output_dir )

    # Generate help quantities
    runtime_helpfile, time_current = coupler_utils.UpdateHelpfile(loop_counter, output_dir, runtime_helpfile_name)

    # Init loop for coupled interior-atmosphere system to equilibrate starting conditions
    while loop_counter["init"] < init_loops:

        # Run SPIDER
        SPIDER_options, restart_file = coupler_utils.RunSPIDER( time_current, time_target, output_dir, SPIDER_options, loop_counter, runtime_helpfile, runtime_helpfile_name, restart_file )

        # Update help quantities after each SPIDER run
        runtime_helpfile, time_current = coupler_utils.UpdateHelpfile(loop_counter, output_dir, runtime_helpfile_name, runtime_helpfile)

        # Init loop atmosphere: equilibrate atmosphere starting conditions
        while loop_counter["atm"] < atm_loops:

            # Run VULCAN
            atm_chemistry = coupler_utils.RunVULCAN( time_current, loop_counter, vulcan_dir, coupler_dir, output_dir, runtime_helpfile, SPIDER_options )

            # Run SOCRATES
            SPIDER_options["heat_flux"], stellar_toa_heating, solar_lum = coupler_utils.RunSOCRATES( time_current, time_offset, star_mass, mean_distance, output_dir, runtime_helpfile, atm_chemistry, loop_counter )

            # Plot conditions throughout run for on-the-fly analysis
            coupler_utils.UpdatePlots( output_dir )

            # Increase iteration counter
            loop_counter["atm"] += 1

        # / Init loop atmosphere
        loop_counter["atm"] = 0
        
        # Output during runtime + find last file for SPIDER restart
        coupler_utils.PrintCurrentState(time_current, runtime_helpfile, atm_chemistry.iloc[0]["Pressure"], SPIDER_options["heat_flux"], stellar_toa_heating, solar_lum, loop_counter, output_dir, restart_file)

        # Increase iteration counter
        loop_counter["init"] += 1

    # / Init loop interior-atmosphere

    # Increase iteration counters   
    loop_counter["total"] += 1

    # Reset restart flag
    SPIDER_options["start_condition"] = 2

    # Inform about start of main loops
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("::::::::::::: START MAIN LOOP |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# Interior-Atmosphere loop
while time_current < time_target:

    # Run SPIDER
    SPIDER_options, restart_file = coupler_utils.RunSPIDER( time_current, time_target, output_dir, SPIDER_options, loop_counter, runtime_helpfile, runtime_helpfile_name, restart_file )

    # !! Apply HACK to JSON files until SOCRATES input option available
    coupler_utils.ModifiedHenrysLaw( atm_chemistry, output_dir, restart_file )

    # Update help quantities after each SPIDER run
    runtime_helpfile, time_current = coupler_utils.UpdateHelpfile(loop_counter, output_dir, runtime_helpfile_name, runtime_helpfile)

    # Atmosphere sub-loop
    while loop_counter["atm"] < atm_loops:

        # Run VULCAN
        atm_chemistry = coupler_utils.RunVULCAN( time_current, loop_counter, vulcan_dir, coupler_dir, output_dir, runtime_helpfile, SPIDER_options )

        # Run SOCRATES
        SPIDER_options["heat_flux"], stellar_toa_heating, solar_lum = coupler_utils.RunSOCRATES( time_current, time_offset, star_mass, mean_distance, output_dir, runtime_helpfile, atm_chemistry, loop_counter )

        # Increase iteration counter
        loop_counter["atm"]  += 1

    # / Loop Atmosphere
    loop_counter["atm"] = 0

    # Output during runtime + find last file for SPIDER restart
    coupler_utils.PrintCurrentState(time_current, runtime_helpfile, atm_chemistry.iloc[0]["Pressure"], SPIDER_options["heat_flux"], stellar_toa_heating, solar_lum, loop_counter, output_dir, restart_file)

    # Plot conditions throughout run for on-the-fly analysis
    coupler_utils.UpdatePlots( output_dir )

    # Increase iteration counters   
    loop_counter["total"]   += 1

    # / Interior-Atmosphere loop

# Save files from finished simulation
coupler_utils.SaveOutput( output_dir )

print("\n===> Run finished successfully!")
