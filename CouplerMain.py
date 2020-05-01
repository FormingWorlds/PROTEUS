#!/usr/bin/env python3

"""
COUPLER Main file – SPIDER-SOCRATES(-VULCAN)
"""

from utils.modules_coupler import *

# Volatile list tracked in radiative-convective model
vol_list = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]

#====================================================================
def main():

    # Read in COUPLER input file
    SPIDER_options, time_dict = cu.ReadInitFile( dirs, args.init_file )
    print(SPIDER_options, time_dict)

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
            atm, SPIDER_options = cu.RunSOCRATES( atm, time_dict, dirs, runtime_helpfile, loop_counter, SPIDER_options )

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

