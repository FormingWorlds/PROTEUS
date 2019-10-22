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

# SPIDER start input options
SURFACE_BC            = "4"          # 4: constant heat flux boundary condition
SOLVE_FOR_VOLATILES   = "1"          # track volatiles in interior/atmosphere reservoirs
H2O_poststep_change   = "0.05"       # fractional H2O melt phase change event trigger
CO2_poststep_change   = "0.05"       # CO2 melt phase trigger
nstepsmacro           = "20"         # number of timesteps
dtmacro               = "50000"      # delta time per macrostep to advance by, in years
heat_flux             = "1.0E4"      # prescribed start surface heat flux (e.g., 10^4 W/m^2)
tsurf_poststep_change = "100.0"      # maximum absolute surface temperature change in Kelvin
R_solid_planet        = "6371000.0"  # planet radius / m
planet_coresize       = "0.55"       # fractional radius of core-mantle boundary

# Restart flags
start_condition       = "1"          # 1: Fresh start | 2: Restart from 'ic_filename'
ic_filename           = "0.json"     # JSON restart file if 'start_condition == 2'
nstepsmacro_init      = "0"          # number of timesteps for initialization loop
dtmacro_init          = "1"          # initialization loop delta time per macrostep [yr]

# Total runtime
time_current          = 0.           # yr
time_target           = 1.0e+6       # yr

# Initial volatile budget, in ppm relative to solid mantle mass
H2O_ppm               = 50.0       # ppm
CO2_ppm               = 100.0      # ppm
H2_ppm                = 0.0        # ppm
CH4_ppm               = 0.0        # ppm
CO_ppm                = 0.0        # ppm
N2_ppm                = 0.0        # ppm
O2_ppm                = 0.0        # ppm
He_ppm                = 0.0        # ppm
S_ppm                 = 0.0        # ppm

# Define runtime helpfile names and generate dataframes
runtime_helpfile_name = "runtime_properties.csv"
# runtime_helpfile = coupler_utils.GenerateRuntimeHelpfiles(output_dir, runtime_helpfile_name)

# Planetary and magma ocean start configuration
star_mass             = 1.0          # M_sol options: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4
mean_distance         = 1.0          # AU, star-planet distance
time_offset           = 100.         # Myr, start of magma ocean after star formation

# Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
loop_no = 0
if start_condition == "2":
    loop_no += 1

    # myjson_o = su.MyJSON( output_dir+'{}.json'.format(time) )
    # core_mass   = myjson_o.get_dict_values(['atmosphere','mass_core'])
    # mantle_mass = myjson_o.get_dict_values(['atmosphere','mass_mantle'])
    # planet_mass = core_mass + mantle_mass

# Inform about start of runtime
print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print("::::::::::::: START INIT LOOP |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# Initialize SPIDER to generate first output file
if start_condition == "1":

    # Delete old SPIDER output
    coupler_utils.PrintSeparator()
    print("Remove old output files:", end =" ")
    for file in natsorted(glob.glob(output_dir+"*.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print("==> Done.")

    # SPIDER initialization call sequence 
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-tsurf_poststep_change", tsurf_poststep_change, "-nstepsmacro", nstepsmacro_init, "-dtmacro", dtmacro_init, "-radius", R_solid_planet, "-coresize", planet_coresize, "-H2O_initial", str(H2O_ppm), "-CO2_initial", str(CO2_ppm), "-H2_initial", str(H2_ppm), "-N2_initial", str(N2_ppm), "-CH4_initial", str(CH4_ppm), "-O2_initial", str(O2_ppm), "-CO_initial", str(CO_ppm), "-S_initial", str(S_ppm), "-He_initial", str(He_ppm) ]

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SPIDER run, loop ", loop_no, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "| flags:")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    coupler_utils.PrintSeparator()

    # Call SPIDER
    subprocess.call(call_sequence)

    # Save surface temperature to file in 'output_dir'
    runtime_helpfile = coupler_utils.UpdateHelpfile(loop_no, output_dir, runtime_helpfile_name)

    # Run VULCAN
    atm_chemistry = coupler_utils.RunVulcan( time_current, loop_no, vulcan_dir, coupler_dir, output_dir, 'spider_input/spider_elements.dat', runtime_helpfile, R_solid_planet )

    # # Generate/adapt VULCAN input files
    # coupler_utils.GenerateVulcanInputFile( time_current, loop_no, vulcan_dir, 'spider_input/spider_elements.dat', runtime_helpfile )

    # # Runtime info
    # coupler_utils.PrintSeparator()
    # print("VULCAN run, loop ", loop_no, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # coupler_utils.PrintSeparator()

    # # Switch to VULCAN directory; run VULCAN, switch back to main directory
    # os.chdir(vulcan_dir)
    # subprocess.run(["python", "vulcan.py", "-n"], shell=False)
    # os.chdir(coupler_dir)

    # # Copy VULCAN dumps to output folder
    # shutil.copy(vulcan_dir+'output/vulcan_EQ.txt', output_dir+str(int(time_current))+"_atm_chemistry.dat")

    # # Read in data from VULCAN output
    # atm_chemistry = pd.read_csv(output_dir+str(int(time_current))+"_atm_chemistry.dat", skiprows=1, delim_whitespace=True)
    # print(atm_chemistry.iloc[:, 0:5])

    plot_atmosphere.plot_mixing_ratios(output_dir, atm_chemistry, int(time_current)) # specific time steps

    # Interpolate TOA heating from Baraffe models and distance from star
    p_s = atm_chemistry.iloc[0]["Pressure"]
    stellar_toa_heating, solar_lum = coupler_utils.InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance)

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SOCRATES run, loop ", loop_no, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    coupler_utils.PrintSeparator()
    # print(volatiles_mixing_ratio)

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    ## TO DO: Adapt to read in atmospheric profile from VULCAN
    heat_flux = str(SocRadConv.RadConvEqm(output_dir, time_current, runtime_helpfile.iloc[-1]["T_surf"], stellar_toa_heating, p_s, atm_chemistry)) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(heat_flux))+"\n")
        f.close()

    # Find last file for SPIDER restart
    ic_filename = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

    # Output during runtime
    coupler_utils.PrintCurrentState(time_current, runtime_helpfile, p_s, heat_flux, ic_filename, stellar_toa_heating, solar_lum)

    # Reset number of computing steps to reach time_target
    dtime       = time_target - time_current
    nstepsmacro = str( math.ceil( dtime / float(dtmacro) ) )

    # Increase iteration counter
    loop_no += 1

    # Restart SPIDER with self-consistent atmospheric composition
    M_mantle_liquid = runtime_helpfile.iloc[-1]["M_mantle_liquid"]
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-tsurf_poststep_change", tsurf_poststep_change, "-nstepsmacro", nstepsmacro_init, "-dtmacro", dtmacro_init, "-radius", R_solid_planet, "-coresize", planet_coresize, "-H2O_initial", str(runtime_helpfile.iloc[-1]["H2O_atm_kg"]/M_mantle_liquid), "-CO2_initial", str(runtime_helpfile.iloc[-1]["CO2_atm_kg"]/M_mantle_liquid), "-H2_initial", str(runtime_helpfile.iloc[-1]["H2_atm_kg"]/M_mantle_liquid), "-N2_initial", str(runtime_helpfile.iloc[-1]["N2_atm_kg"]/M_mantle_liquid), "-CH4_initial", str(runtime_helpfile.iloc[-1]["CH4_atm_kg"]/M_mantle_liquid), "-O2_initial", str(runtime_helpfile.iloc[-1]["O2_atm_kg"]/M_mantle_liquid), "-CO_initial", str(runtime_helpfile.iloc[-1]["CO_atm_kg"]/M_mantle_liquid), "-S_initial", str(runtime_helpfile.iloc[-1]["S_atm_kg"]/M_mantle_liquid), "-He_initial", str(runtime_helpfile.iloc[-1]["He_atm_kg"]/M_mantle_liquid) ]

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SPIDER run, loop ", loop_no, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "| flags:")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    coupler_utils.PrintSeparator()

    # Restart SPIDER
    subprocess.call(call_sequence)

    # Reset restart flag
    start_condition     = "2"

    # Inform about start of main loops
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("::::::::::::: START MAIN LOOP |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# Ping-pong between SPIDER and SOCRATES until target time is reached
while time_current < time_target:

    # Save surface temperature to file in 'output_dir'
    runtime_helpfile = coupler_utils.UpdateHelpfile(loop_no, output_dir, runtime_helpfile_name, runtime_helpfile)

    # Interpolate TOA heating from Baraffe models and distance from star
    M_solid_planet = runtime_helpfile.iloc[-1]["M_core"] + runtime_helpfile.iloc[-1]["M_mantle"]
    grav_s      = su.gravity( M_solid_planet, float(R_solid_planet) )
    M_vol_tot   = runtime_helpfile.iloc[-1]["M_atm"]
    p_s = ( M_vol_tot * grav_s / ( 4. * np.pi * (float(R_solid_planet)**2.) ) ) * 1e-2 # mbar
    stellar_toa_heating, solar_lum = coupler_utils.InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance)

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SOCRATES run, loop ", loop_no, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    coupler_utils.PrintSeparator()

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    ## TO DO: Adapt to read in atmospheric profile from VULCAN
    heat_flux = str(SocRadConv.RadConvEqm(output_dir, time_current, runtime_helpfile.iloc[-1]["T_surf"], stellar_toa_heating, p_s, atm_chemistry)) # W/m^2

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
    loop_no += 1

    # Recalculate melt phase volatile abundance for SPIDER restart
    coupler_utils.ModifiedHenrysLaw( atm_chemistry, output_dir, ic_filename )

    # SPIDER restart call sequence
    # ---->> HERE THE VOLATILES FROM VULCAN NEED TO BE RE-FED; OPTION DOES NOT EXIST IN SPIDER CURRENTLY!
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-tsurf_poststep_change", tsurf_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro, "-radius", R_solid_planet, "-coresize", planet_coresize ] # , "-outputDirectory", output_dir

    # Runtime info
    coupler_utils.PrintSeparator()
    print("SPIDER run, loop ", loop_no, "–", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "– flags:")
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
coupler_utils.PrintCurrentState(time_current, surfaceT_current, h2o_kg, h2o_ratio, co2_kg, co2_ratio, p_s, heat_flux, ic_filename, stellar_toa_heating, solar_lum)

print("\n===> Run finished successfully!")
