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
    'SURFACE_BC':                  4,     # 4: constant heat flux boundary condition
    'tsurf_poststep_change':       100.0, # maximum absolute surface temperature change [K]
    'R_solid_planet':              6371000, # planet radius [m]
    'planet_coresize':             0.55,  # fractional radius of core-mantle boundary
    'IC_INTERIOR':                 1,     # 1: Fresh start | 2: Restart from file
    'IC_ATMOSPHERE':               1,     # 1: Fresh start | 3: read in partial pressures
    'ic_interior_filename':        "",    # Restart manually, [yr]+".json"
    'nstepsmacro':                 1,     # number of timesteps, adjusted during runtime
    'dtmacro':                     50000, # delta time per macrostep to advance by [yr]
    'heat_flux':                   1.0E4, # init heat flux, adjusted during runtime [W/m^2]
    'H2O_initial_total_abundance': 100,   # init loop: H2O mass relative to mantle [ppm wt]
    'CO2_initial_total_abundance': 1,     # init loop [ppm wt]
    'H2_initial_total_abundance':  1,     # init loop [ppm wt]
    'CH4_initial_total_abundance': 1,     # init loop [ppm wt]
    'CO_initial_total_abundance':  1,     # init loop [ppm wt]
    'N2_initial_total_abundance':  1,     # init loop [ppm wt]
    'O2_initial_total_abundance':  1,     # init loop [ppm wt]
    'S_initial_total_abundance':   1,     # init loop [ppm wt]
    'He_initial_total_abundance':  1,     # init loop [ppm wt]
    'H2O_poststep_change':         0.05,  # fractional H2O melt phase change event trigger
    'CO2_poststep_change':         0.05,  # CO2 melt phase trigger
    'H2_poststep_change':          0.00,  # fraction
    'CH4_poststep_change':         0.00,  # fraction
    'CO_poststep_change':          0.00,  # fraction
    'N2_poststep_change':          0.00,  # fraction
    'O2_poststep_change':          0.00,  # fraction
    'S_poststep_change':           0.00,  # fraction
    'He_poststep_change':          0.00,  # fraction
    'H2O_initial_atmos_pressure':  0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'CO2_initial_atmos_pressure':  0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'H2_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'CH4_initial_atmos_pressure':  0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'CO_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'N2_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'O2_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'S_initial_atmos_pressure':    0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'He_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
    'species':                     [ "H2O", "CO2" ]
    }
# volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]
# Total runtime
time_start            = 0.           # yr
time_current          = time_start   # yr
time_target           = 1.0e+6       # yr

# Define runtime helpfile names and generate dataframes
runtime_helpfile_name = "runtime_helpfile.csv"

# Planetary and magma ocean start configuration
star_mass             = 1.0          # M_sol options: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4
mean_distance         = 1.0          # AU, star-planet distance
time_offset           = 100.         # Myr, start of magma ocean after star formation

# Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
loop_counter = { "total": 0, "init": 0, "atm": 0, "init_loops": 3, "atm_loops": 2 }

# # Equilibration loops for init and atmosphere sub-loops
# init_loops = 3
# atm_loops  = 2

# Start conditions and help files depending on restart option
if SPIDER_options["IC_INTERIOR"] == 1: 
        coupler_utils.CleanOutputDir( output_dir )
        runtime_helpfile    = []
        atm_chemistry       = []
if SPIDER_options["IC_INTERIOR"] == 2:
    # If restart skip init loop
    loop_counter["total"] += loop_counter["init_loops"]
    loop_counter["init"]  += loop_counter["init_loops"]

    # Restart file name: automatic last file or specific one
    SPIDER_options["ic_interior_filename"] = str(natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1])
    # SPIDER_options["ic_interior_filename"] = "X.json"

# Inform about start of runtime
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print(":::::::::::: START COUPLER RUN |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# Interior-Atmosphere loop
while time_current < time_target:

    ############### INTERIOR SUB-LOOP

    # Run SPIDER
    SPIDER_options = coupler_utils.RunSPIDER( time_current, time_target, output_dir, SPIDER_options, loop_counter, runtime_helpfile, atm_chemistry )

    # Update help quantities, input_flag: "Interior"
    runtime_helpfile, time_current = coupler_utils.UpdateHelpfile(loop_counter, output_dir, vulcan_dir, runtime_helpfile_name, runtime_helpfile, "Interior", atm_chemistry, SPIDER_options)

    ############### / INTERIOR SUB-LOOP

    ############### ATMOSPHERE SUB-LOOP
    while loop_counter["atm"] < loop_counter["atm_loops"]:

        # Run VULCAN
        atm_chemistry = coupler_utils.RunVULCAN( time_current, loop_counter, vulcan_dir, coupler_dir, output_dir, runtime_helpfile, SPIDER_options )

        # Run SOCRATES
        SPIDER_options["heat_flux"], stellar_toa_heating, solar_lum = coupler_utils.RunSOCRATES( time_current, time_offset, star_mass, mean_distance, output_dir, runtime_helpfile, atm_chemistry, loop_counter )

        loop_counter["atm"] += 1

    # Update help quantities, input_flag: "Atmosphere"
    runtime_helpfile, time_current = coupler_utils.UpdateHelpfile(loop_counter, output_dir, vulcan_dir, runtime_helpfile_name, runtime_helpfile, "Atmosphere", atm_chemistry, SPIDER_options)
    ############### / ATMOSPHERE SUB-LOOP
    
    # Runtime info
    coupler_utils.PrintCurrentState(time_current, runtime_helpfile, atm_chemistry.iloc[0]["Pressure"], SPIDER_options, stellar_toa_heating, solar_lum, loop_counter, output_dir)

    # Plot conditions throughout run for on-the-fly analysis
    coupler_utils.UpdatePlots( output_dir )
    
    # Adjust iteration counters + total time 
    loop_counter["atm"]         = 0
    loop_counter["total"]       += 1
    if loop_counter["init"] < loop_counter["init_loops"]:
        loop_counter["init"]    += 1
        time_current            = time_start

    # Reset restart flag once SPIDER was started w/ ~correct volatile chemistry + heat flux
    if loop_counter["total"] >= loop_counter["init_loops"]:
        SPIDER_options["IC_INTERIOR"] = 2

# Save files from finished simulation
coupler_utils.SaveOutput( output_dir )

print("\n===> COUPLER run finished successfully <===")


### TO DO LIST ###
# - Feed atmospheric profile to SOCRATES instead of surface abundances
# - In "ModifiedHenrysLaw" function, take mass conservation from atm chemistry --> calc mass of atmosphere from atm chemistry and insert into function. So far, mass conservation is calculated from former .json file, which may be inconsistent.
# - Fix Psurf stuff.
# - Use Height calculation of updated VULCAN versions instead of own function.
# - Check if "Magma ocean volatile content" needs ot be altered in JSON
# - Read in total masses from VULCAN during UPDATEHELPFILE function during ATMOS update.

