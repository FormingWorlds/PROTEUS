#!/usr/bin/env python3

"""
PROTEUS Main file
"""

from utils.modules_ext import *
from utils.constants import *
from utils.coupler import *
from utils.stellar import *
from utils.vulcan import RunVULCAN
from utils.aeolus import RunAEOLUS, StructAtm
from utils.spider import RunSPIDER

import utils.constants

from AEOLUS.modules.stellar_luminosity import InterpolateStellarLuminosity
from AEOLUS.utils.StellarSpectrum import PrepareStellarSpectrum,InsertStellarSpectrum

#====================================================================
def main():

    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print(":::::::::::: START PROTEUS RUN |", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    # Parse console arguments
    args = parse_console_arguments()

    # Read in COUPLER input file
    COUPLER_options, time_dict = ReadInitFile( args.cfg_file , verbose=False )

    # Set directories dictionary
    utils.constants.dirs = SetDirectories(COUPLER_options)
    from utils.constants import dirs

    # Check if output directory exists, otherwise create
    if not os.path.exists(dirs["output"]):
        os.makedirs(dirs["output"])

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
        CleanDir( dirs["output"] )
        CleanDir( dirs["vulcan"]+"/output/" )
        runtime_helpfile    = []

        # Copy config file to output directory, for future reference
        shutil.copyfile( args.cfg_file, dirs["output"]+"/init_coupler.cfg")

        # Solve for initial partial pressures of volatiles
        # Using parameterised method
        if (COUPLER_options['initial_pp_method'] == 1):

            solvepp_dict = solvepp_doit(COUPLER_options)

            for s in volatile_species:
                key_pp = str(s+"_initial_atmos_pressure")
                key_in = str(s+"_included")

                # This volatile calculated by solvepp
                if s in solvepp_dict:
                    COUPLER_options[key_in] = 1
                    COUPLER_options[key_pp] = solvepp_dict[s]

                # Volatile is included but is not parameterised and no pressure provided
                elif ( key_in in COUPLER_options):
                    if (COUPLER_options[key_in] == 1) and ( key_pp not in COUPLER_options):
                        COUPLER_options[key_pp] = 0.0

                # Otherwise, volatile is neglected
                else:
                    COUPLER_options[key_in] = 0
                    COUPLER_options[key_pp] = 0.0

        # Use prescribed partial pressures
        else:
            for s in volatile_species:
                key_pp = str(s+"_initial_atmos_pressure")
                key_in = str(s+"_included")

                # Volatile is included but no pressure provided
                if (key_in in COUPLER_options):
                    if ( key_pp not in COUPLER_options):
                        COUPLER_options[key_pp] = 0.0
                
                # Volatile neglected
                else:
                    COUPLER_options[key_in] = 0
                    COUPLER_options[key_pp] = 0.0


    # Check that all partial pressures are positive
    inc_vols = []
    for s in volatile_species:
        key_pp = str(s+"_initial_atmos_pressure")
        key_in = str(s+"_included")
        if (COUPLER_options[key_in] == 1):
            if (COUPLER_options[key_pp] < 0.0):
                print("ERROR: Partial pressures must be positive or zero!")
                exit(1)
            inc_vols.append(s)
    print("Included volatiles:",inc_vols)

    # Handle stellar spectrum...

    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)
    time_dict['sflux_prev'] = -1.0e99

    # Calculate band-integrated fluxes for modern stellar spectrum (1 AU)
    match COUPLER_options['star_model']:
        case 1:
            COUPLER_options = MorsCalculateFband(dirs, COUPLER_options)
        case 2:
            track = BaraffeLoadtrack(COUPLER_options)

    # Spectrum for SOCRATES
    star_spec_src = dirs["output"]+"socrates_star.txt"
    if COUPLER_options["star_model"] == 0:  # Will not be updated during the loop, so just need to write an initial one
        PrepareStellarSpectrum(StellarFlux_wl,StellarFlux_fl,star_spec_src)
        InsertStellarSpectrum(
                                # dirs["rad_conv"]+"/spectral_files/sp_b318_HITRAN_a16/sp_b318_HITRAN_a16_no_spectrum",
                                dirs["rad_conv"]+"/spectral_files/Mallard/sp_b318_HITRAN_a8",
                                star_spec_src,
                                dirs["output"]+"runtime_spectral_file"
                            )
    
    # Main loop
    while time_dict["planet"] < time_dict["target"]:

        PrintSeparator()
        print("Loop counters:", loop_counter)

        ############### STELLAR FLUX MANAGEMENT
        print("Stellar flux management...")
        match COUPLER_options['star_model']:
            case 0:
                S_0, toa_heating = InterpolateStellarLuminosity(time_dict, COUPLER_options)
            case 1:
                S_0, toa_heating = MorsSolarConstant(time_dict, COUPLER_options)
            case 2:
                S_0, toa_heating = BaraffeSolarConstant(time_dict, COUPLER_options, track)

        # Include SW radiation from star in heating rate calculations?
        if (COUPLER_options["stellar_heating"] > 0):
            COUPLER_options["TOA_heating"] = toa_heating
        else:
            COUPLER_options["TOA_heating"] = 0.0

        # Calculate a new (historical) stellar spectrum 
        if (COUPLER_options['star_model'] > 0) and \
           ( abs( time_dict['planet'] - time_dict['sflux_prev'] ) > COUPLER_options['sflux_dt_update'] ):
            
            print("Updating stellar spectrum") 
            match COUPLER_options['star_model']:   # Calculate arrays at 1 AU
                case 1:
                    fl,fls = MorsSpectrumCalc(time_dict['star'], StellarFlux_wl, StellarFlux_fl,COUPLER_options)
                case 2:
                    fl,fls = BaraffeSpectrumCalc(time_dict['star'], StellarFlux_fl,COUPLER_options, track)

            # Write stellar spectra to disk
            print("Writing spectrum to disk")
            writessurf = (COUPLER_options["atmosphere_chem_type"] > 0)
            SpectrumWrite(time_dict,StellarFlux_wl,fl,fls,dirs,write_surf=writessurf)

            # Generate a new SOCRATES spectral file containing this new spectrum
            print("Inserting spectrum into SOCRATES spectral file")
            PrepareStellarSpectrum(StellarFlux_wl,fl,star_spec_src)
            InsertStellarSpectrum(
                                    # dirs["rad_conv"]+"/spectral_files/sp_b318_HITRAN_a16/sp_b318_HITRAN_a16_no_spectrum",
                                    dirs["rad_conv"]+"/spectral_files/Mallard/sp_b318_HITRAN_a8",
                                    star_spec_src,
                                    dirs["output"]+"runtime_spectral_file"
                                )

            time_dict['sflux_prev'] = time_dict['planet'] 

        else:
            print("New spectrum not required at this time")

        COUPLER_options["T_eqm"] = calc_eqm_temperature(S_0,  COUPLER_options["albedo_pl"])
        ############### / STELLAR FLUX MANAGEMENT



        ############### INTERIOR SUB-LOOP

        # Run SPIDER
        COUPLER_options = RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile )

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", COUPLER_options)

        # During init loop: update initial guesses for partial pressure and mantle mass
        if (loop_counter["init"] < loop_counter["init_loops"]) and (COUPLER_options['initial_pp_method'] == 1):

            # Store new guesses based on last init iteration
            run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior']
            COUPLER_options["mantle_mass_guess"] =  run_int.iloc[-1]["M_mantle"]
            COUPLER_options["T_surf_guess"] =       run_int.iloc[-1]["T_surf"]

            # Solve for partial pressures again
            solvepp_dict = solvepp_doit(COUPLER_options)

            # Store results
            for s in volatile_species:
                if s in solvepp_dict.keys():
                    COUPLER_options[s+"_initial_atmos_pressure"] = solvepp_dict[s]


        ############### / INTERIOR SUB-LOOP




        ############### ATMOSPHERE SUB-LOOP
        while (loop_counter["atm"] == 0) or (loop_counter["atm"] < loop_counter["atm_loops"] and abs(COUPLER_options["F_net"]) > COUPLER_options["F_eps"]):

            # Initialize atmosphere structure
            atm, COUPLER_options = StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options )

            # Run SOCRATES: update TOA heating and MO heat flux
            atm, COUPLER_options = RunAEOLUS( atm, time_dict, dirs, runtime_helpfile, loop_counter, COUPLER_options )
             
            # Run VULCAN (settings-dependent): update atmosphere mixing ratios
            if (COUPLER_options["atmosphere_chem_type"] == 2):
                atm = RunVULCAN( atm, time_dict, loop_counter, dirs, runtime_helpfile, COUPLER_options)

            # Update help quantities, input_flag: "Atmosphere"
            runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", COUPLER_options)

            loop_counter["atm"] += 1

        ############### / ATMOSPHERE SUB-LOOP
        


        ############### LOOP ITERATION MANAGEMENT
        # Print info, save atm to file, update plots
        PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options, atm, loop_counter, dirs)

        # Plot conditions throughout run for on-the-fly analysis
        if (COUPLER_options["plot_iterfreq"] > 0) and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0):
            UpdatePlots( dirs["output"] )
        
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

        ############### / LOOP ITERATION MANAGEMENT

    # Plot conditions at the end
    UpdatePlots( dirs["output"] )

    print("\n\n===> PROTEUS run finished successfully <===")
    print("     "+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

#====================================================================
if __name__ == '__main__':
    main()

# End of file
