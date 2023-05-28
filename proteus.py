#!/usr/bin/env python3

"""
PROTEUS Main file
"""

from utils.modules_ext import *
from utils.constants import *
from utils.coupler import *
from utils.mors import *
from utils.baraffe import *
from utils.vulcan import RunVULCAN
from utils.aeolus import RunAEOLUS, StructAtm
from utils.spider import RunSPIDER

import utils.constants

#====================================================================
def main():

    print("===> Start PROTEUS <===")

    # Parse console arguments
    args = parse_console_arguments()

    # Read in COUPLER input file
    COUPLER_options, time_dict = ReadInitFile( args.cfg_file )

    # Set directories dictionary
    utils.constants.dirs = SetDirectories(COUPLER_options)
    from utils.constants import dirs

    # Check if output directory exists, otherwise create
    if not os.path.exists(dirs["output"]):
        os.makedirs(dirs["output"])
        print("--> Create output directory:", dirs["output"])

    # Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
    loop_counter = { "total": 0, "init": 0, "atm": 0, "init_loops": 3, "atm_loops": 10 }
    
    # If restart skip init loop # args.r or args.rf or 
    if COUPLER_options["IC_INTERIOR"] == 2:
        loop_counter["total"] += loop_counter["init_loops"]
        loop_counter["init"]  += loop_counter["init_loops"]

        # Restart file name if not specified: use last file output
        if (COUPLER_options["IC_INTERIOR"] == 2 and COUPLER_options["ic_interior_filename"] == 0):
            COUPLER_options["ic_interior_filename"] = str(natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/"+"*.json")])[-1])

        # Clean all overtimes from present helpfile
        runtime_helpfile = pd.read_csv(dirs["output"]+"/"+"runtime_helpfile.csv", index_col=False, header=0, sep="\t")
        t_curr = COUPLER_options["ic_interior_filename"][:-5]
        print("Clean helpfile from overtimes >", t_curr, "yr")
        runtime_helpfile = runtime_helpfile.loc[runtime_helpfile["Time"] <= int(t_curr)]

        COUPLER_options["F_int"] = runtime_helpfile.iloc[-1]["F_int"]
        COUPLER_options["F_atm"] = runtime_helpfile.iloc[-1]["F_atm"]
        COUPLER_options["F_net"] = runtime_helpfile.iloc[-1]["F_net"]

    # Start conditions and help files depending on restart option
    else:
        CleanOutputDir( dirs["output"] )
        CleanOutputDir( dirs["vulcan"]+"/output/" )
        runtime_helpfile    = []

        # Work out which volatiles are involved
        for vol in volatile_species:
            match COUPLER_options["IC_ATMOSPHERE"]:
                case 3:
                    key = vol+"_initial_atmos_pressure"
                case 1:
                    key = vol+"_initial_total_abundance"
            if (key in COUPLER_options):
                COUPLER_options[vol+"_included"] = (COUPLER_options[key]>0.0)
                continue

    # Check that the current configuration is reasonable
    if not ('H' in element_list):
        print("Error: The element list must include hydrogen!")
        print("       Currently the element list includes: ",element_list)
        exit(1)

    
    # Calculate band-integrated fluxes for modern stellar spectrum (1 AU)
    match COUPLER_options['star_model']:
        case 1:
            COUPLER_options = ModernSpectrumFband(dirs, COUPLER_options)
        case 2:
            track = LoadBaraffeTrack(COUPLER_options)
    
    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)
    time_dict['sflux_prev'] = -1.0e99

    # Inform about start of runtime
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print(":::::::::::: START COUPLER RUN |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    # Interior-Atmosphere loop
    while time_dict["planet"] < time_dict["target"]:

        # Calculate stellar luminosity and planetary eqm temperature
        # S_old, _ = SocRadConv.InterpolateStellarLuminosity(COUPLER_options["star_mass"], time_dict, COUPLER_options["mean_distance"], COUPLER_options["albedo_pl"], COUPLER_options["Sfrac"])
        S_0, _ = SolarConstant(time_dict, COUPLER_options)

        # Calculate historical spectrum (1 AU)
        if ( abs( time_dict['planet'] - time_dict['sflux_prev'] ) > COUPLER_options['sflux_dt_update'] ):
            print("Updating stellar spectrum")
            match COUPLER_options['star_model']:
                case 1:
                    MorsSpectrumWrite(time_dict, StellarFlux_wl, StellarFlux_fl,dirs,COUPLER_options)
                case 2:
                    BaraffeSpectrumWrite(time_dict, StellarFlux_wl, StellarFlux_fl,dirs,COUPLER_options, track)
            time_dict['sflux_prev'] = time_dict['planet'] 

        # print(S_old, S_0)
        COUPLER_options["T_eqm"] = calc_eqm_temperature(S_0,  COUPLER_options["albedo_pl"])

        ############### INTERIOR SUB-LOOP
        print("Start interior")

        # Run SPIDER
        COUPLER_options = RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile )

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", COUPLER_options)

        print("End interior")
        ############### / INTERIOR SUB-LOOP

        ############### ATMOSPHERE SUB-LOOP

        print("Start atmosphere")
        while (loop_counter["atm"] == 0) or (loop_counter["atm"] < loop_counter["atm_loops"] and abs(COUPLER_options["F_net"]) > COUPLER_options["F_eps"]):

            # Initialize atmosphere structure
            atm, COUPLER_options = StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options )

            # Run SOCRATES: update TOA heating and MO heat flux
            atm, COUPLER_options = RunAEOLUS( atm, time_dict, dirs, runtime_helpfile, loop_counter, COUPLER_options )
             
            # Run VULCAN (settings-dependent): update atmosphere mixing ratios
            if (COUPLER_options["use_vulcan"] == 1):
                atm = RunVULCAN( atm, time_dict, loop_counter, dirs, runtime_helpfile, COUPLER_options)

            # Update help quantities, input_flag: "Atmosphere"
            runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", COUPLER_options)

            loop_counter["atm"] += 1

        ############### / ATMOSPHERE SUB-LOOP
        print("End atmosphere")
        
        # Print info, save atm to file, update plots
        PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options, atm, loop_counter, dirs)

        # Plot conditions throughout run for on-the-fly analysis
        UpdatePlots( dirs["output"], COUPLER_options, time_dict )
        
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
    # SaveOutput( dirs["output"] )

    # Plot conditions at the end
    UpdatePlots( dirs["output"], COUPLER_options, time_dict )

    print("\n\n===> PROTEUS run finished successfully <===")

#====================================================================
if __name__ == '__main__':
    main()

# End of file
