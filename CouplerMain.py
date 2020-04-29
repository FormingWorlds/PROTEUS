#!/usr/bin/env python3

"""
Coupler Main file – SPIDER-SOCRATES-VULCAN
"""

from utils.modules_coupler import *

# Volatile list tracked in radiative-convective model
vol_list = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]

#====================================================================
def main():

    # SPIDER settings
    SPIDER_options = {
        'IC_INTERIOR':                 1,     # 1: Fresh start | 2: Restart from file
        'IC_ATMOSPHERE':               1,     # 1: Fresh start | 3: read in partial pressures
        'SURFACE_BC':                  4,     # 4: constant heat flux boundary condition
        'tsurf_poststep_change':       50.0,  # maximum absolute surface temperature change [K]
        'R_solid_planet':              6371000, # planet radius [m]
        'planet_coresize':             0.55,  # fractional radius of core-mantle boundary
        'ic_interior_filename':        "",    # Restart manually, [yr]+".json"
        'nstepsmacro':                 1,     # number of timesteps, adjusted during runtime
        'dtmacro':                     50000, # delta time per macrostep to advance by [yr]
        'heat_flux':                   1.0E4, # init heat flux, adjusted during runtime [W/m^2]
        'H2O_initial_total_abundance': 0,     # init loop: H2O mass relative to mantle [ppm wt]
        'CO2_initial_total_abundance': 100,   # init loop [ppm wt]
        'H2_initial_total_abundance':  0,     # init loop [ppm wt]
        'CH4_initial_total_abundance': 0,     # init loop [ppm wt]
        'CO_initial_total_abundance':  0,     # init loop [ppm wt]
        'N2_initial_total_abundance':  0,     # init loop [ppm wt]
        'O2_initial_total_abundance':  0,     # init loop [ppm wt]
        'S_initial_total_abundance':   0,     # init loop [ppm wt]
        'He_initial_total_abundance':  0,     # init loop [ppm wt]
        'H2O_poststep_change':         0.05,  # fractional H2O melt phase change event trigger
        'CO2_poststep_change':         0.05,  # CO2 melt phase trigger
        'H2_poststep_change':          0.05,  # fraction
        'CH4_poststep_change':         0.05,  # fraction
        'CO_poststep_change':          0.05,  # fraction
        'N2_poststep_change':          0.05,  # fraction
        'O2_poststep_change':          0.05,  # fraction
        'S_poststep_change':           0.05,  # fraction
        'He_poststep_change':          0.05,  # fraction
        'H2O_initial_atmos_pressure':  0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'CO2_initial_atmos_pressure':  0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'H2_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'CH4_initial_atmos_pressure':  0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'CO_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'N2_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'O2_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'S_initial_atmos_pressure':    0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'He_initial_atmos_pressure':   0.0,   # restart w/ p_i from VULCAN/SPIDER [Pa]
        'use_vulcan':                  0,     # 0: none 1: VULCAN->SOCRATES & 2: V->SOC+SPI
        }
    
    # Planetary and magma ocean start configuration
    star_mass     = 1.0                      # M_sun, [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4 ]
    mean_distance = 1.0                      # AU, star-planet distance
    time_offset   = 1e+8                     # yr, start of magma ocean after star formation
    time_dict     = { 
                      "planet": 0.,          # yr, time since MO start
                      "target": 1e+7,        # yr, target time for MO evolution
                      "offset": time_offset, # yr, time since star formation
                      "star":   time_offset, # yr, time since star formation
                    }

    # Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
    loop_counter = { "total": 0, "init": 0, "atm": 0, "init_loops": 3, "atm_loops": 1 }

    # Start conditions and help files depending on restart option
    if SPIDER_options["IC_INTERIOR"] == 1: 
        cu.CleanOutputDir( dirs["output"] )
        runtime_helpfile    = []
    # If restart skip init loop
    if SPIDER_options["IC_INTERIOR"] == 2:
        loop_counter["total"] += loop_counter["init_loops"]
        loop_counter["init"]  += loop_counter["init_loops"]
        SPIDER_options["IC_ATMOSPHERE"] = 3

        # Restart file name: automatic last file or specific one
        if SPIDER_options["ic_interior_filename"] == "":
            SPIDER_options["ic_interior_filename"] = str(natsorted([os.path.basename(x) for x in glob.glob(dirs["output"]+"/"+"*.json")])[-1])
        # SPIDER_options["ic_interior_filename"] = "X.json"

    # Parse optional argument from console
    if args.H2O_ppm:
        print("Set H2O_ppm from command line:", args.H2O_ppm)
        SPIDER_options["H2O_initial_total_abundance"] = float(args.H2O_ppm)
    elif args.H2O_bar:
        print("Set H2O_bar from command line:", args.H2O_bar)
        SPIDER_options["H2O_initial_atmos_pressure"] = float(args.H2O_bar)
    if args.CO2_ppm:
        print("Set CO2_ppm from command line:", args.CO2_ppm)
        SPIDER_options["CO2_initial_total_abundance"] = float(args.CO2_ppm)
    elif args.CO2_bar:
        print("Set CO2_bar from command line:", args.CO2_bar)
        SPIDER_options["CO2_initial_atmos_pressure"] = float(args.CO2_bar)
    if args.H2_ppm:
        print("Set H2_ppm from command line:", args.H2_ppm)
        SPIDER_options["H2_initial_total_abundance"] = float(args.H2_ppm)
    elif args.H2_bar:
        print("Set H2_bar from command line:", args.H2_bar)
        SPIDER_options["H2_initial_atmos_pressure"] = float(args.H2_bar)
    if args.CO_ppm:
        print("Set CO_ppm from command line:", args.CO_ppm)
        SPIDER_options["CO_initial_total_abundance"] = float(args.CO_ppm)
    elif args.CO_bar:
        print("Set CO_bar from command line:", args.CO_bar)
        SPIDER_options["CO_initial_atmos_pressure"] = float(args.CO_bar)
    if args.N2_ppm:
        print("Set N2_ppm from command line:", args.N2_ppm)
        SPIDER_options["N2_initial_total_abundance"] = float(args.N2_ppm)
    elif args.N2_bar:
        print("Set N2_bar from command line:", args.N2_bar)
        SPIDER_options["N2_initial_atmos_pressure"] = float(args.N2_bar)
    if args.CH4_ppm:
        print("Set CH4_ppm from command line:", args.CH4_ppm)
        SPIDER_options["CH4_initial_total_abundance"] = float(args.CH4_ppm)
    elif args.CH4_bar:
        print("Set CH4_bar from command line:", args.CH4_bar)
        SPIDER_options["CH4_initial_atmos_pressure"] = float(args.CH4_bar)
    if args.O2_ppm:
        print("Set O2_ppm from command line:", args.O2_ppm)
        SPIDER_options["O2_initial_total_abundance"] = float(args.O2_ppm)
    elif args.O2_bar:
        print("Set O2_bar from command line:", args.O2_bar)
        SPIDER_options["O2_initial_atmos_pressure"] = float(args.O2_bar)

    # Inform about start of runtime
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print(":::::::::::: START COUPLER RUN |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    
    # Interior-Atmosphere loop
    while time_dict["planet"] < time_dict["target"]:

        ############### INTERIOR SUB-LOOP

        # Run SPIDER
        SPIDER_options = cu.RunSPIDER( time_dict, dirs, SPIDER_options, loop_counter, runtime_helpfile )

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict = cu.UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", SPIDER_options)

        ############### / INTERIOR SUB-LOOP

        ############### ATMOSPHERE SUB-LOOP

        # Initialize atmosphere structure
        # if loop_counter["total"] == 0 and loop_counter["init"] == 0:
        atm = cu.StructAtm( time_dict, loop_counter, dirs, runtime_helpfile )

        while loop_counter["atm"] < loop_counter["atm_loops"]:

            # Run VULCAN (settings-dependent): update atmosphere mixing ratios
            atm = cu.RunAtmChemistry( atm, time_dict, loop_counter, dirs, runtime_helpfile, SPIDER_options )

            # Run SOCRATES: update TOA heating and MO heat flux
            atm, SPIDER_options = cu.RunSOCRATES( atm, time_dict, star_mass, mean_distance, dirs, runtime_helpfile, loop_counter, SPIDER_options )

            loop_counter["atm"] += 1

        # Update help quantities, input_flag: "Atmosphere"
        runtime_helpfile, time_dict = cu.UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", SPIDER_options)

        ############### / ATMOSPHERE SUB-LOOP
        
        # Print info, save atm to file, update plots
        cu.PrintCurrentState(time_dict, runtime_helpfile, SPIDER_options, atm, loop_counter, dirs)

        # Plot conditions throughout run for on-the-fly analysis
        cu.UpdatePlots( dirs["output"], SPIDER_options["use_vulcan"] )

        # # After very first timestep, starting w/ 2nd init loop: read in partial pressures
        # if loop_counter["init"] >= 1:
        #     SPIDER_options["IC_ATMOSPHERE"] = 3
        
        # Adjust iteration counters + total time 
        loop_counter["atm"]         = 0
        loop_counter["total"]       += 1
        if loop_counter["init"] < loop_counter["init_loops"]:
            loop_counter["init"]    += 1
            time_dict["planet"]     = 0.

        # Reset restart flag once SPIDER was started w/ ~correct volatile chemistry + heat flux
        if loop_counter["total"] >= loop_counter["init_loops"]:
            SPIDER_options["IC_INTERIOR"] = 2

    # Save files from finished simulation
    cu.SaveOutput( dirs["output"] )

    print("\n===> COUPLER run finished successfully <===")

#====================================================================
main()

### TO DO LIST ###
# - Feed atmospheric profile to SOCRATES instead of just surface abundances
# - Feed p_i back to SPIDER
# - Use Height calculation of updated VULCAN versions instead of own function
# - Compare total masses from VULCAN during UPDATEHELPFILE function during ATMOS update

