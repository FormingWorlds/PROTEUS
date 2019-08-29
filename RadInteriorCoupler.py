"""
RadInteriorCoupler.py
"""

import numpy as np
# import GGRadConv
import SocRadConv
import SocRadModel
import ReadInterior
import subprocess
import glob
import os, shutil
import math
from natsort import natsorted #https://pypi.python.org/pypi/natsort
import coupler_utils
import spider_coupler_utils as su
import plot_interior
import plot_time_evolution
import plot_stacked_interior_atmosphere
from datetime import datetime

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
output_dir = os.getcwd()+"/output/"

# Restart flag
start_condition     = "1"            # 1: Start from beginning, 2: Restart from file "ic_filename"

# Total runtime
time_current = 0
time_target  = 1000000

# Planetary and magma ocean start configuration
star_mass       = 1.0       # M_sol: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4
mean_distance   = 1.0       # AU, star-planet distance
time_offset     = 100.      # Myr, start of magma ocean after star formation


# Count SPIDER-SOCRATES iterations
pingpong_no = 0
if start_condition == "2":
    pingpong_no += 1

# Inform about start of runtime
print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print(":::::::::::::: START COUPLER RUN –", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "::::::::::::::")
print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# Initialize SPIDER to generate first output file
if start_condition == "1":

    # Reset heat flux and surface history and delete old SPIDER output
    # open('OLRFlux.dat', 'w').close()
    # open('surface_atmosphere.dat', 'w').close()
    print("____________________________________________")
    print("Remove old output files:", end =" ")
    for file in natsorted(glob.glob(output_dir+"*.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print("==> Done.")

    # SPIDER initialization call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", "1", "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", "0", "-dtmacro", "1" ]
    # , "-outputDirectory", output_dir

    # Runtime info
    print("____________________________________________")
    print("SPIDER run flags:", end =" ")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    print("____________________________________________")

    # Call SPIDER
    subprocess.call(call_sequence)

    # Reset restart flag
    start_condition     = "2"

# Ping-pong between SPIDER and SOCRATES until target time is reached
while time_current < time_target:

    # Save surface temperature to file in 'output_dir'
    ReadInterior.write_surface_quantitites(output_dir)

    # Load surface temperature and volatiles released, to be fed to SOCRATES
    surface_quantities  = np.loadtxt(output_dir+'surface_atmosphere.dat')
    if pingpong_no > 0: # After first iteration more than one entry
        surface_quantities = surface_quantities[-1]
    print("Surface quantities – [time, T_s, H2O, CO2]: ", surface_quantities)
    time_current        = surface_quantities[0]  # K
    surfaceT_current    = surface_quantities[1]  # K
    h2o_current         = surface_quantities[2]  # kg
    co2_current         = surface_quantities[3]  # kg

    print(time_current, surfaceT_current)

    print("::::::::::::: START SOCRATES ITERATION -", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "::::::::::::::")

    # Interpolate TOA heating from Baraffe models and distance from star
    stellar_toa_heating, solar_lum = coupler_utils.InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance)

    print("STELLAR TOA HEATING:", stellar_toa_heating, solar_lum)

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    heat_flux = str(SocRadConv.RadConvEqm(output_dir, time_current, surfaceT_current, stellar_toa_heating)) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(heat_flux))+"\n")
        f.close()

    # Find last file for SPIDER restart
    ic_filename = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

    # Print current values
    print("_______________________________")
    print("      ==> RUNTIME INFO <==")
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
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro ]
    # , "-outputDirectory", output_dir

    # Runtime info
    print("SPIDER run flags:", end =" ")
    for flag in call_sequence:
        print(flag, end =" ")
    print()

    # Restart SPIDER
    subprocess.call(call_sequence)

    ### / ping-pong

    # Plot conditions throughout run for analysis
    print("*** Plot current evolution,", end=" ")
    plot_time_evolution.plot_evolution(output_dir) # all times
    output_times = su.get_all_output_times()
    if len(output_times) <= 8:
        plot_times = output_times
    else:
        plot_times = [ output_times[0]]     # first snapshot
        for i in [ 2, 15, 22, 30, 45, 66 ]: # distinct timestamps
            plot_times.append(output_times[int(round(len(output_times)*(i/100.)))])
        plot_times.append(output_times[-1]) # last snapshot
    print("snapshots:", plot_times)
    plot_interior.mantle_evolution(plot_times)
    plot_stacked_interior_atmosphere.stacked_evolution(plot_times)
    plt.close()

# Copy files to separate folder
sim_dir = coupler_utils.make_output_dir() #
print("===> Copy files to separate dir for this run to:", sim_dir)
shutil.copy(os.getcwd()+"/bu_input.opts", sim_dir+"bu_input.opts")
for file in natsorted(glob.glob(output_dir+"*.*")):
    shutil.copy(file, sim_dir+os.path.basename(file))
    print(os.path.basename(file), end =" ")
print("\n===> Done!")

# Print final statement
print("########## Target time reached, final values:")
print("Time [Myr]:", str(float(time_current)/1e6))
print ("T_surf [K]:", surfaceT_current)
print ("H2O [kg]:", h2o_current)
print ("CO2 [kg]:", co2_current)
print ("Heat flux [W/m^2]:", heat_flux)
print ("Last file name:", ic_filename)
print("########## / FIN ")
