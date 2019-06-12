"""
RadInteriorCoupler.py

"""

import numpy as np
# import GGRadConv
import SocRadConv
import ReadInterior
import subprocess

# SPIDER start input options
ic_filename         = "1.json"      # JSON file to read in initial condition
SURFACE_BC          = "4"           # 4: constant heat flux boundary condition
SOLVE_FOR_VOLATILES = "1"           # track evolution of volatiles in interior/atmosphere reservoirs
H2O_poststep_change = "0.05"        # fractional change in melt phase H2O concentration that triggers event
CO2_poststep_change = "0.05"        # as above for CO2, also a 5% change to trigger an event
nstepsmacro         = "20"          # number of timesteps
dtmacro             = "50000"       # delta time per macrostep to advance by, in years
heat_flux           = "1.0E30"      # prescribed start surface heat flux (e.g., 10^4 W/m^2)

# Restart flag
restart = 0

# Total runtime
time_current = 0

while time_current < 1000000:

    call_sequence_start = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", 1, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "activate_rollback", "activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro ]

    call_sequence_restart = [ "spider", "-options_file", "bu_input.opts", "-initial_condition", 2, "-ic_filename", ic_filename, "-SURFACE_BC", SURFACE_BC, "-surface_bc_value", heat_flux, "SOLVE_FOR_VOLATILES", SOLVE_FOR_VOLATILES, "activate_rollback", "activate_poststep", "-H2O_poststep_change", H2O_poststep_change, "-CO2_poststep_change", CO2_poststep_change, "-nstepsmacro", nstepsmacro, "-dtmacro", dtmacro ]

    if restart == 0:
        call_sequence = call_sequence_start
    if restart == 1:
        call_sequence = call_sequence_restart

    # run SPIDER
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
    OLRFlux = SocRadConv.RadConvEqm(surfaceT_current) # W/m^2

    # save OLR flux for interior code
    np.savetxt('OLRFlux.txt', OLRFlux*np.ones(1))

    # # Heat flux, placeholder, needs to be read-in from SPIDER later
    # rad_planet = 6371000.0                      # m, Earth radius
    # surf_planet = 4.*np.pi*(rad_planet**2.)     # m^2
    # heat_flux = str(OLRFlux * surf_planet)      # W

    # print current values
    print("time – T_surf – h2o – co2 – heat flux")
    print(time_current, surfaceT_current, h2o_current, co2_current, OLRFlux)
