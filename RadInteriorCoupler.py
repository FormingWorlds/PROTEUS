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

# SPIDER start input options
ic_filename         = "58796.json"      # JSON file to read in initial condition
SURFACE_BC          = "4"            # 4: constant heat flux boundary condition
SOLVE_FOR_VOLATILES = "1"            # track evolution of volatiles in interior/atmosphere reservoirs
H2O_poststep_change = "1.00"         # fractional change in melt phase H2O concentration that triggers event
CO2_poststep_change = "1.00"         # as above for CO2 to trigger an event
nstepsmacro         = "50"           # number of timesteps
dtmacro             = "20000"        # delta time per macrostep to advance by, in years
heat_flux           = "1.0E4"        # prescribed start surface heat flux (e.g., 10^4 W/m^2)

# Restart flag
start_condition     = "2"            # 1: Start from beginning, 2: Restart from file "ic_filename"

# Total runtime
time_current = 0
time_target  = 1000000

# Initialize SPIDER to generate first output file
if start_condition == "1":

    # SPIDER initialization call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", "1", "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", "0", "-dtmacro", "1" ]

    # Runtime info
    print("SPIDER run flags:", call_sequence)

    # Call SPIDER
    subprocess.call(call_sequence)

    # Reset restart flag
    start_condition     = "2"

# Ping-poing between SPIDER and SOCRATES until target time is reached
while time_current < time_target:

    # SPIDER restart call sequence
    call_sequence = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", start_condition, "-ic_filename", "output/"+ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "-SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "-activate_rollback", "-activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro ]

    # Runtime info
    print("SPIDER run flags:", call_sequence)

    # Restart SPIDER
    subprocess.call(call_sequence)

    # save surface T to file
    ReadInterior.write_surface_quantitites()

    # load surface temperature to be fed to SOCRATES
    surfaceT = np.loadtxt('surfaceT.dat')
    time_current = surfaceT[-1][0]      # K
    surfaceT_current = surfaceT[-1][1]  # K

    # load volatiles released from interior to be fed to SOCRATES
    volatiles_out = np.loadtxt('volatiles_out.dat')
    h2o_current = volatiles_out[-1][1]  # kg
    co2_current = volatiles_out[-1][2]  # kg

    # calculate OLR flux given surface T w/ SOCRATES
    heat_flux = SocRadConv.RadConvEqm(surfaceT_current) # W/m^2

    # save OLR flux for interior code
    np.savetxt('OLRFlux.txt', heat_flux*np.ones(1))

    # find last file for restarting
    ic_filename = natsorted([os.path.basename(x) for x in glob.glob("./output/*.json")])[-1]

    # print current values
    # print("time – T_surf – h2o – co2 – heat_flux – last_file")
    # print(time_current, surfaceT_current, h2o_current, co2_current, heat_flux, ic_filename)
    print("Time [Myr]:", str(float(time_current)/1e6))
    print ("T_surf [K]:", surfaceT_current)
    print ("H2O [kg]:", h2o_current)
    print ("CO2 [kg]:", co2_current)
    print ("OLR [W]:", heat_flux)
    print ("Last file name:", ic_filename)

    # Set to str for subprocess
    heat_flux = str(heat_flux)

    # Reset number of computing steps to reach time_target
    dtime       = time_target - time_current
    nstepsmacro = str( math.ceil( dtime / float(dtmacro) ) )

# Print final statement
print("########## Target time reached, final values:")
print("Time [Myr]:", str(float(time_current)/1e6))
print ("T_surf [K]:", surfaceT_current)
print ("H2O [kg]:", h2o_current)
print ("CO2 [kg]:", co2_current)
print ("OLR [W]:", heat_flux)
print ("Last file name:", ic_filename)
print("########## / FIN ")
