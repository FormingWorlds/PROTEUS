#!/usr/bin/env python3

"""
PROTEUS Main file
"""

from utils.modules_coupler import *

# Volatile list tracked in radiative-convective model
vol_list = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]

#====================================================================
def main():

    # Parse console arguments
    args = parse_console_arguments()

    # Read in COUPLER input file
    COUPLER_options, time_dict = cu.ReadInitFile( dirs, args.init_file )

    # Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
    loop_counter = { "total": 0, "init": 0, "atm": 0, "init_loops": 3, "atm_loops": 10 }

    # Parse console arguments
    COUPLER_options = assign_parse_arguments(args, COUPLER_options)

    # Start conditions and help files depending on restart option
    if COUPLER_options["IC_INTERIOR"] == 1: 
        cu.CleanOutputDir( dirs["output"] )
        runtime_helpfile    = []
  
    # If restart skip init loop # args.r or args.rf or 
    if COUPLER_options["IC_INTERIOR"] == 2:
        loop_counter["total"] += loop_counter["init_loops"]
        loop_counter["init"]  += loop_counter["init_loops"]

        # Restart file name if not specified: use last file output
        if (COUPLER_options["IC_INTERIOR"] == 2 and COUPLER_options["ic_interior_filename"] == 0):
            COUPLER_options["ic_interior_filename"] = str(cu.natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/"+"*.json")])[-1])

        # Clean all overtimes from present helpfile
        runtime_helpfile = pd.read_csv(dirs["output"]+"/"+"runtime_helpfile.csv", index_col=False, header=0, sep=" ")
        t_curr = COUPLER_options["ic_interior_filename"][:-5]
        print("Clean helpfile from overtimes >", t_curr, "yr")
        runtime_helpfile = runtime_helpfile.loc[runtime_helpfile["Time"] <= int(t_curr)]

        COUPLER_options["F_int"] = runtime_helpfile.iloc[-1]["F_int"]
        COUPLER_options["F_atm"] = runtime_helpfile.iloc[-1]["F_atm"]
        COUPLER_options["F_net"] = runtime_helpfile.iloc[-1]["F_net"]

    # Inform about start of runtime
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print(":::::::::::: START COUPLER RUN |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    # Interior-Atmosphere loop
    while time_dict["planet"] < time_dict["target"]:

        ############### INTERIOR SUB-LOOP

        # Run SPIDER
        COUPLER_options = cu.RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile )

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict, COUPLER_options = cu.UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", COUPLER_options)

        ############### / INTERIOR SUB-LOOP

        ############### ATMOSPHERE SUB-LOOP

        while loop_counter["atm"] == 0 or (loop_counter["atm"] < loop_counter["atm_loops"] and abs(COUPLER_options["F_net"]) > COUPLER_options["F_eps"]):

            # Initialize atmosphere structure
            atm, COUPLER_options = cu.StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options )

            # Run VULCAN (settings-dependent): update atmosphere mixing ratios
            atm = cu.RunAtmChemistry( atm, time_dict, loop_counter, dirs, runtime_helpfile, COUPLER_options )

            # Run SOCRATES: update TOA heating and MO heat flux
            atm, COUPLER_options = cu.RunSOCRATES( atm, time_dict, dirs, runtime_helpfile, loop_counter, COUPLER_options )

            # Update help quantities, input_flag: "Atmosphere"
            runtime_helpfile, time_dict, COUPLER_options = cu.UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", COUPLER_options)

            loop_counter["atm"] += 1

        ############### / ATMOSPHERE SUB-LOOP
        
        # Print info, save atm to file, update plots
        cu.PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options, atm, loop_counter, dirs)

        # Plot conditions throughout run for on-the-fly analysis
        cu.UpdatePlots( dirs["output"], COUPLER_options, time_dict )
        
        # Adjust iteration counters + total time 
        loop_counter["atm"]         = 0
        loop_counter["total"]       += 1
        if loop_counter["init"] < loop_counter["init_loops"]:
            loop_counter["init"]    += 1
            time_dict["planet"]     = 0.

        # Reset restart flag once SPIDER was started w/ ~correct volatile chemistry + heat flux
        if loop_counter["total"] >= loop_counter["init_loops"]:
            COUPLER_options["IC_INTERIOR"] = 2

        # Stop simulation when planet is completely solidified
        if COUPLER_options["solid_stop"] == 1 and runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]:
            print("\n===> Planet solidified! <===\n")
            break

    # Save files from finished simulation in extra folder
    # cu.SaveOutput( dirs["output"] )

    # Plot conditions at the end
    cu.UpdatePlots( dirs["output"], COUPLER_options, time_dict )

    print("\n\n===> PROTEUS run finished successfully <===")

#====================================================================
main()
