"""
RadInteriorCoupler.py
"""

import numpy as np
# import GGRadConv
import SocRadConv
import SocRadModel
import subprocess
import glob
import os, shutil
import math
from natsort import natsorted #https://pypi.python.org/pypi/natsort
import coupler_utils
import spider_coupler_utils
import plot_interior
import plot_global
import plot_stacked
import plot_atmosphere
from datetime import datetime

# SPIDER start input options
ic_filename             = "0.json"       # JSON file to read in initial condition
SURFACE_BC              = "4"            # 4: constant heat flux boundary condition
SOLVE_FOR_VOLATILES     = "1"            # track evolution of volatiles in interior/atmosphere reservoirs
H2O_poststep_change     = "0.05"         # fractional change in melt phase H2O concentration that triggers event
CO2_poststep_change     = "0.05"         # as above for CO2 to trigger an event
nstepsmacro             = "20"           # number of timesteps
dtmacro                 = "50000"        # delta time per macrostep to advance by, in years
heat_flux               = "1.0E4"        # prescribed start surface heat flux (e.g., 10^4 W/m^2)
tsurf_poststep_change   = "100.0"        # maximum absolute surface temperature change in Kelvin
planet_radius           = "6371000.0"    # planet radius / m
planet_coresize         = "0.55"         # fractional radius of core-mantle boundary

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

    # Delete old SPIDER output
    print("____________________________________________")
    print("Remove old output files:", end =" ")
    for file in natsorted(glob.glob(output_dir+"*.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print("==> Done.")

    # SPIDER initialization call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", "1", "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-tsurf_poststep_change", tsurf_poststep_change, "-nstepsmacro", "0", "-dtmacro", "1", "-radius", planet_radius, "-coresize", planet_coresize ]
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
    coupler_utils.write_surface_quantitites(output_dir)

    # Load surface + global properties released, to be fed to SOCRATES
    surface_quantities  = np.loadtxt(output_dir+'runtime_properties.dat')
    if pingpong_no > 0: # After first iteration more than one entry
        surface_quantities = surface_quantities[-1]
    time_current        = surface_quantities[0]  # K
    surfaceT_current    = surface_quantities[1]  # K
    h2o_kg              = surface_quantities[2]  # kg
    co2_kg              = surface_quantities[3]  # kg
    h2_kg               = 0  # kg
    ch4_kg              = 0  # kg
    co_kg               = 0  # kg
    n2_kg               = 0  # kg
    o2_kg               = 0  # kg
    he_kg               = 0  # kg
    planet_mass         = surface_quantities[4]  # kg

    # Interpolate TOA heating from Baraffe models and distance from star
    grav_s      = spider_coupler_utils.gravity( planet_mass, float(planet_radius) )
    M_vol_tot   = h2o_kg + co2_kg + h2_kg + ch4_kg + co_kg + n2_kg + o2_kg + he_kg 
    p_s = ( M_vol_tot * grav_s / ( 4. * np.pi * (float(planet_radius)**2.) ) ) * 1e-2 # mbar
    stellar_toa_heating, solar_lum = coupler_utils.InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance)

    # Print statements for development
    # print("Surface quantities – [time, T_s, H2O, CO2]: ", surface_quantities)
    # print("STELLAR TOA HEATING:", stellar_toa_heating, solar_lum)
    # coupler_utils.PrintCurrentState(time_current, surfaceT_current, h2o_kg, "NaN", co2_kg, "NaN", p_s, heat_flux, ic_filename, stellar_toa_heating, solar_lum)

    # Calculate volatile mol ratios from volatile masses in atmosphere
    h2o_ratio, co2_ratio, h2_ratio, ch4_ratio, co_ratio, n2_ratio, o2_ratio, he_ratio = coupler_utils.CalcMolRatios(h2o_kg, co2_kg, h2_kg, ch4_kg, co_kg, n2_kg, o2_kg, he_kg)

    print("::::::::::::: START SOCRATES ITERATION -", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "::::::::::::::")

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    heat_flux = str(SocRadConv.RadConvEqm(output_dir, time_current, surfaceT_current, stellar_toa_heating, p_s, h2o_ratio, co2_ratio, h2_ratio, ch4_ratio, co_ratio, n2_ratio, o2_ratio, he_ratio)) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(heat_flux))+"\n")
        f.close()

    # Find last file for SPIDER restart
    ic_filename = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

    # Output during runtime
    coupler_utils.PrintCurrentState(time_current, surfaceT_current, h2o_kg, h2o_ratio, co2_kg, co2_ratio, p_s, heat_flux, ic_filename, stellar_toa_heating, solar_lum)

    # Reset number of computing steps to reach time_target
    dtime       = time_target - time_current
    nstepsmacro = str( math.ceil( dtime / float(dtmacro) ) )

    # Increase iteration counter
    pingpong_no += 1

    # SPIDER restart call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-tsurf_poststep_change", tsurf_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro, "-radius", planet_radius, "-coresize", planet_coresize ] # , "-outputDirectory", output_dir

    # Runtime info
    print("SPIDER run flags:", end =" ")
    for flag in call_sequence:
        print(flag, end =" ")
    print()

    # Restart SPIDER
    subprocess.call(call_sequence)

    ### / ping-pong

    # Plot conditions throughout run for on-the-fly analysis
    print("*** Plot current evolution,", end=" ")
    output_times = spider_coupler_utils.get_all_output_times()
    if len(output_times) <= 8:
        plot_times = output_times
    else:
        plot_times = [ output_times[0]]         # first snapshot
        for i in [ 2, 15, 22, 30, 45, 66 ]:     # distinct timestamps
            plot_times.append(output_times[int(round(len(output_times)*(i/100.)))])
        plot_times.append(output_times[-1])     # last snapshot
    print("snapshots:", plot_times)
    plot_interior.plot_interior(plot_times)     # specific time steps
    plot_atmosphere.plot_atmosphere(plot_times)   # specific time steps
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
coupler_utils.PrintCurrentState(time_current, surfaceT_current, h2o_kg, h2o_ratio, co2_kg, co2_ratio, p_s, heat_flux, ic_filename, stellar_toa_heating, solar_lum)

print("\n===> Run finished successfully!")
