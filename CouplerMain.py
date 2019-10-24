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
import plot_interior
import plot_global
import plot_stacked
import plot_atmosphere
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
    'start_condition':       1,       # 1: Fresh start | 2: Restart from 'restart_filename'
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

# Define runtime helpfile names and generate dataframes
runtime_helpfile_name = "runtime_properties.csv"

# Planetary and magma ocean start configuration
star_mass             = 1.0          # M_sol options: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4
mean_distance         = 1.0          # AU, star-planet distance
time_offset           = 100.         # Myr, start of magma ocean after star formation

# Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
loop_counter = { "total": 0, "init": 0, "atm": 0 }

# If restart skip init loop
if SPIDER_options["start_condition"] == 2:
    loop_counter["total"] += 1
    loop_counter["init"]  += 1
    loop_counter["atm"]   += 1

# Inform about start of runtime
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print("::::::::::::: START INIT LOOP |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

if SPIDER_options["start_condition"] == 1:

    # Delete old SPIDER output
    coupler_utils.CleanOutputDir( output_dir )

    # Generate help quantities
    runtime_helpfile = coupler_utils.UpdateHelpfile(loop_counter, output_dir, runtime_helpfile_name)

    # Init loop for coupled interior-atmosphere system to equilibrate starting conditions
    while loop_counter["init"] < 1:

        # Run SPIDER
        SPIDER_options, runtime_helpfile = coupler_utils.RunSPIDER( time_current, time_target, output_dir, SPIDER_options, loop_counter, runtime_helpfile, runtime_helpfile_name )

        # Init loop atmosphere: equilibrate atmosphere starting conditions
        while loop_counter["atm"] < 1:

            # Run VULCAN
            atm_chemistry = coupler_utils.RunVULCAN( time_current, loop_counter, vulcan_dir, coupler_dir, output_dir, 'spider_input/spider_elements.dat', runtime_helpfile, SPIDER_options["R_solid_planet"] )

            # Run SOCRATES
            SPIDER_options["heat_flux"], stellar_toa_heating, solar_lum = coupler_utils.RunSOCRATES( time_current, time_offset, star_mass, mean_distance, output_dir, runtime_helpfile, atm_chemistry, loop_counter )

            # Increase iteration counter
            loop_counter["atm"]  += 1

            # / init loop atmosphere

        # Find last file for SPIDER restart
        ic_filename = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

        # Output during runtime
        coupler_utils.PrintCurrentState(time_current, runtime_helpfile, atm_chemistry.iloc[0]["Pressure"], SPIDER_options["heat_flux"], ic_filename, stellar_toa_heating, solar_lum)

        # Increase iteration counter
        loop_counter["init"] += 1

         # / init loop interior-atmosphere

    # Increase iteration counters   
    loop_counter["total"]   += 1

    # Reset restart flag
    SPIDER_options["start_condition"] = 2

    # Inform about start of main loops
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("::::::::::::: START MAIN LOOP |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# Ping-pong between SPIDER and SOCRATES until target time is reached
while time_current < time_target:

    # Save surface temperature to file in 'output_dir'
    runtime_helpfile = coupler_utils.UpdateHelpfile(loop_counter, output_dir, runtime_helpfile_name, runtime_helpfile)

    # Interpolate TOA heating from Baraffe models and distance from star
    M_solid_planet = runtime_helpfile.iloc[-1]["M_core"] + runtime_helpfile.iloc[-1]["M_mantle"]
    grav_s      = su.gravity( M_solid_planet, float(R_solid_planet) )
    M_vol_tot   = runtime_helpfile.iloc[-1]["M_atm"]
    p_s = ( M_vol_tot * grav_s / ( 4. * np.pi * (float(R_solid_planet)**2.) ) ) * 1e-2 # mbar
    stellar_toa_heating, solar_lum = coupler_utils.InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance)

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SOCRATES run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    coupler_utils.PrintSeparator()

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    ## TO DO: Adapt to read in atmospheric profile from VULCAN
    SPIDER_options["heat_flux"] = str(SocRadConv.RadConvEqm(output_dir, time_current, runtime_helpfile.iloc[-1]["T_surf"], stellar_toa_heating, p_s, atm_chemistry)) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(SPIDER_options["heat_flux"]))+"\n")
        f.close()

    # Find last file for SPIDER restart
    last_filename = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

    # Output during runtime
    coupler_utils.PrintCurrentState(time_current, surfaceT_current, h2o_kg, h2o_ratio, co2_kg, co2_ratio, p_s, heat_flux, last_filename, stellar_toa_heating, solar_lum)

    # Reset number of computing steps to reach time_target
    dtime       = time_target - time_current
    nstepsmacro = str( math.ceil( dtime / float(dtmacro) ) )

    # Increase iteration counter
    loop_counter["total"] += 1

    # Recalculate melt phase volatile abundance for SPIDER restart
    coupler_utils.ModifiedHenrysLaw( atm_chemistry, output_dir, last_filename )

    # SPIDER restart call sequence
    # ---->> HERE THE VOLATILES FROM VULCAN NEED TO BE RE-FED; OPTION DOES NOT EXIST IN SPIDER CURRENTLY!
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-ic_filename", "output/"+last_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-tsurf_poststep_change", tsurf_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro, "-radius", R_solid_planet, "-coresize", planet_coresize ] # , "-outputDirectory", output_dir

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SPIDER run, loop ", loop_counter, "–", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "– flags:")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    coupler_utils.PrintSeparator()

    # Restart SPIDER
    subprocess.call(call_sequence)

    ### / ping-pong

    # Plot conditions throughout run for on-the-fly analysis
    print("*** Plot current evolution,", end=" ")
    output_times = su.get_all_output_times()
    if len(output_times) <= 8:
        plot_times = output_times
    else:
        plot_times = [ output_times[0]]         # first snapshot
        for i in [ 2, 15, 22, 30, 45, 66 ]:     # distinct timestamps
            plot_times.append(output_times[int(round(len(output_times)*(i/100.)))])
        plot_times.append(output_times[-1])     # last snapshot
    print("snapshots:", plot_times)
    plot_interior.plot_interior(plot_times)     # specific time steps
    plot_atmosphere.plot_atmosphere(output_dir, plot_times) # specific time steps
    plot_stacked.plot_stacked(plot_times)       # specific time steps
    plot_global.plot_global(output_dir)         # all time steps

# Copy old files to separate folder
sim_dir = coupler_utils.make_output_dir() #
print("===> Copy files to separate dir for this run to:", sim_dir)
shutil.copy(os.getcwd()+"/bu_input.opts", sim_dir+"bu_input.opts")
for file in natsorted(glob.glob(output_dir+"*.*")):
    shutil.copy(file, sim_dir+os.path.basename(file))
    print(os.path.basename(file), end =" ")

# Print final statement
coupler_utils.PrintCurrentState(time_current, surfaceT_current, h2o_kg, h2o_ratio, co2_kg, co2_ratio, p_s, heat_flux, last_filename, stellar_toa_heating, solar_lum)

print("\n===> Run finished successfully!")
