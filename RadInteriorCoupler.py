"""
RadInteriorCoupler.py
"""

import numpy as np
# import GGRadConv
import SocRadConv
import ReadInterior
import subprocess
import glob
import os
import math
from natsort import natsorted #https://pypi.python.org/pypi/natsort
import coupler_utils

# SPIDER start input options
ic_filename         = "0.json"       # JSON file to read in initial condition
SURFACE_BC          = "4"            # 4: constant heat flux boundary condition
SOLVE_FOR_VOLATILES = "1"            # track evolution of volatiles in interior/atmosphere reservoirs
H2O_poststep_change = "0.05"         # fractional change in melt phase H2O concentration that triggers event
CO2_poststep_change = "0.05"         # as above for CO2 to trigger an event
nstepsmacro         = "20"           # number of timesteps
dtmacro             = "50000"        # delta time per macrostep to advance by, in years
heat_flux           = "1.0E4"        # prescribed start surface heat flux (e.g., 10^4 W/m^2)

# Define output output output directory
output_dir = coupler_utils.make_output_dir() #os.getcwd()+"/output/"
print("===> Output dir for this run:", output_dir)

# Restart flag
start_condition     = "1"            # 1: Start from beginning, 2: Restart from file "ic_filename"

# Total runtime
time_current = 0
time_target  = 1000000

# Count SPIDER-SOCRATES iterations
pingpong_no = 0

# Initialize SPIDER to generate first output file
if start_condition == "1":

    # # Reset heat flux and surface history and delete old SPIDER output
    # open('OLRFlux.dat', 'w').close()
    # open('surface_atmosphere.dat', 'w').close()
    # print("Remove old output files:", end =" ")
    # for file in natsorted(glob.glob("./output_dir/*.*")):
    #     os.remove(file)
    #     print(os.path.basename(file), end =" ")
    # print("... Done!")

    # SPIDER initialization call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", "1", "-outputDirectory", output_dir, "-ic_filename", ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", "0", "-dtmacro", "1" ]

    # Runtime info
    print("SPIDER run flags:", call_sequence)

    # Call SPIDER
    subprocess.call(call_sequence)

    # Reset restart flag
    start_condition     = "2"

# Ping-pong between SPIDER and SOCRATES until target time is reached
while time_current < time_target:

    # Save surface temperature to file in 'output_dir'
    ReadInterior.write_surface_quantitites(output_dir)

    # Load surface temperature and volatiles released, to be fed to SOCRATES
    surface_quantities  = np.loadtxt(output_dir+'/surface_atmosphere.dat')
    if pingpong_no > 0: # After first iteration more than one entry
        surface_quantities = surface_quantities[-1]
    print(len(surface_quantities), surface_quantities)
    time_current        = surface_quantities[0]  # K
    surfaceT_current    = surface_quantities[1]  # K
    h2o_current         = surface_quantities[2]  # kg
    co2_current         = surface_quantities[3]  # kg

    print(time_current, surfaceT_current)

    print("::::::::::::::::::::: SOCRATES :::::::::::::::::::::")

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    heat_flux = str(SocRadConv.RadConvEqm(output_dir, time_current, surfaceT_current)) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"/OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(heat_flux))+"\n")
        f.close()

    # Find last file for SPIDER restart
    ic_filename = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"./*.json")])[-1]

    # Print current values
    print("_______________________________")
    print("Time [Myr]:", str(float(time_current)/1e6))
    print ("T_surf [K]:", surfaceT_current)
    print ("H2O [kg]:", h2o_current)
    print ("CO2 [kg]:", co2_current)
    print ("Heat flux [W/m^2]:", heat_flux)
    print ("Last file name:", ic_filename)
    print("_______________________________")

    # Reset number of computing steps to reach time_target
    dtime       = time_target - time_current
    nstepsmacro = str( math.ceil( dtime / float(dtmacro) ) )

    # # Set heat_flux class to str for subprocess call
    # heat_flux = str(heat_flux)

    # Increase iteration counter
    pingpong_no += 1

    # SPIDER restart call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-outputDirectory", output_dir, "-ic_filename", ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro ]

    # Runtime info
    print("SPIDER run flags:", call_sequence)

    # Restart SPIDER
    subprocess.call(call_sequence)

    ### / ping-pong

# Print final statement
print("########## Target time reached, final values:")
print("Time [Myr]:", str(float(time_current)/1e6))
print ("T_surf [K]:", surfaceT_current)
print ("H2O [kg]:", h2o_current)
print ("CO2 [kg]:", co2_current)
print ("Heat flux [W/m^2]:", heat_flux)
print ("Last file name:", ic_filename)
print("########## / FIN ")
