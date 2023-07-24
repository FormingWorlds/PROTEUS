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


    # Count Interior (SPIDER) <-> Atmosphere (SOCRATES+VULCAN) iterations
    loop_counter = { "total": 0, "init": 0, "atm": 0, "init_loops": 3, "atm_loops": 10 }
    
    # If restart skip init loop # args.r or args.rf or 
    if COUPLER_options["IC_INTERIOR"] == 2:

        # Check if output directory actually exists
        if (not os.path.exists(dirs["output"])) or (not os.path.exists(dirs["output"]+'/data/')):
            print("ERROR: Cannot resume run because directory doesn't exist!")
            exit(1)

        loop_counter["total"] += loop_counter["init_loops"]
        loop_counter["init"]  += loop_counter["init_loops"]

        # Restart file name if not specified: use last file output
        if (COUPLER_options["IC_INTERIOR"] == 2 and COUPLER_options["ic_interior_filename"] == 0):
            COUPLER_options["ic_interior_filename"] = str(natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/data/*.json")])[-1])

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
        CleanDir( dirs['output']+'/data/')
        CleanDir( dirs["vulcan"]+"/output/" )
        runtime_helpfile    = []

        # Copy config file to output directory, for future reference
        shutil.copyfile( args.cfg_file, dirs["output"]+"/init_coupler.cfg")

        # Zero-out all volatiles, and ensure that they are all tracked
        for s in volatile_species:
            key_pp = str(s+"_initial_atmos_pressure")
            COUPLER_options[key_pp] = 0.0

            key_in = str(s+"_included")
            if (key_in in COUPLER_options):
                COUPLER_options[key_in] = int(COUPLER_options[key_in])
            else:
                COUPLER_options[key_in] = 0


        # Solve for initial partial pressures of volatiles, using parameterised method
        solvepp_dict = {"dummy_item"}
        if (COUPLER_options['solvepp_enabled'] == 1):

            solvepp_dict = solvepp_doit(COUPLER_options)

            for s in volatile_species:
                key_pp = str(s+"_initial_atmos_pressure")
                key_in = str(s+"_included")

                # This volatile calculated by solvepp
                if s in solvepp_dict:
                    if solvepp_dict[s] > 1e-1:
                        COUPLER_options[key_in] = 1
                        COUPLER_options[key_pp] = solvepp_dict[s]
                    else:
                        COUPLER_options[key_in] = 0
                        COUPLER_options[key_pp] = 0.0

        # Prepare to use prescribed additional partial pressures
        for s in volatile_species:
            key_pp = str(s+"_initial_atmos_pressure")
            key_ab = str(s+"_add_bars")
            key_in = str(s+"_included")

            # Ensure that additional bars are recorded, even if zero
            if (key_ab not in COUPLER_options):
                COUPLER_options[key_ab] = 0.0

            # Inject additional bars now, if solvepp is disabled or doesn't handle this volatile
            if (COUPLER_options[key_in] == 1) and ( (COUPLER_options["solvepp_enabled"] == 0) or (s not in solvepp_dict)):
                    COUPLER_options[key_pp] = float(COUPLER_options[key_ab] * 1.0e5)  # Convert bar -> Pa

    # Check that all partial pressures are positive
    inc_vols = []
    for s in volatile_species:
        key_pp = str(s+"_initial_atmos_pressure")
        key_in = str(s+"_included")
        if (COUPLER_options[key_in] == 1):
            inc_vols.append(s)

            # Check positive
            if (COUPLER_options[key_pp] <= 0.0):
                print("ERROR: Partial pressures of included volatiles must be positive!")
                print("       Consider assigning volatile '%s' a small positive value" % s)
                exit(1)

            # Ensure numerically reasonable
            if (COUPLER_options[key_pp] <= 1e-3):
                COUPLER_options[key_pp] = 1e-3
            

    print("Included volatiles:",inc_vols)

    # Check that spectral file exists
    spectral_file_nostar = dirs["rad_conv"]+"/"+COUPLER_options["spectral_file"]
    if not os.path.exists(spectral_file_nostar):
        print("ERROR: Spectral file does not exist at '%s'!" % spectral_file_nostar)
        exit(1)

    # Handle stellar spectrum...

    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)
    time_dict['sflux_prev'] = -1.0e99

    # Calculate band-integrated fluxes for modern stellar spectrum (1 AU)
    match COUPLER_options['star_model']:
        case 1:
            MorsSolveUV(dirs,COUPLER_options,StellarFlux_wl,StellarFlux_fl)  # Solve for UV-PL band interface
            COUPLER_options = MorsCalculateFband(dirs, COUPLER_options)
        case 2:
            track = BaraffeLoadtrack(COUPLER_options)

    # Spectrum for SOCRATES
    star_spec_src = dirs["output"]+"socrates_star.txt"
    if COUPLER_options["star_model"] == 0:  # Will not be updated during the loop, so just need to write an initial one
        PrepareStellarSpectrum(StellarFlux_wl,StellarFlux_fl,star_spec_src)
        InsertStellarSpectrum(
                                spectral_file_nostar,
                                star_spec_src,
                                dirs["output"]+"runtime_spectral_file"
                            )
        os.remove(star_spec_src)
    
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
            COUPLER_options["T_eqm"]  = calc_eqm_temperature(S_0,  COUPLER_options["albedo_pl"])
        else:
            print("Stellar heating is disabled")
            COUPLER_options["TOA_heating"] = 0.0
            COUPLER_options["T_eqm"]  = 0.0
        COUPLER_options["T_skin"] = COUPLER_options["T_eqm"] * (0.5**0.25)


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
            SpectrumWrite(time_dict,StellarFlux_wl,fl,fls,dirs['output']+'/data/',write_surf=writessurf)

            # Generate a new SOCRATES spectral file containing this new spectrum
            print("Inserting star into SOCRATES spectral file")
            PrepareStellarSpectrum(StellarFlux_wl,fl,star_spec_src)
            InsertStellarSpectrum(
                                    spectral_file_nostar,
                                    star_spec_src,
                                    dirs["output"]+"runtime_spectral_file"
                                )
            os.remove(star_spec_src)

            time_dict['sflux_prev'] = time_dict['planet'] 

        else:
            print("New spectrum not required at this time")

        ############### / STELLAR FLUX MANAGEMENT



        ############### INTERIOR SUB-LOOP

        # Run SPIDER
        COUPLER_options = RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile )

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", COUPLER_options)

        # Sanity check output
        if (runtime_helpfile.iloc[-1]["T_surf"] < COUPLER_options["min_temperature"]):
            print("WARNING: Surface temperature is very low!")

        # Update initial guesses for partial pressure and mantle mass
        if (COUPLER_options['solvepp_enabled'] == 1):

            # Before injecting additional atmosphere
            if (loop_counter["init"] < loop_counter["init_loops"]):

                # Store new guesses based on last init iteration
                run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior']
                COUPLER_options["mantle_mass_guess"] =      run_int.iloc[-1]["M_mantle"]
                COUPLER_options["T_surf_guess"] =           run_int.iloc[-1]["T_surf"]
                COUPLER_options["melt_fraction_guess"] =    run_int.iloc[-1]["Phi_global"]

                # Solve for partial pressures again
                solvepp_dict = solvepp_doit(COUPLER_options)

                # Store results
                for s in volatile_species:
                    if (s in solvepp_dict.keys()) and (COUPLER_options[s+"_included"] == 1):
                        COUPLER_options[s+"_initial_atmos_pressure"] = solvepp_dict[s]

            # Penultimate init loop: inject additional atmosphere
            if (loop_counter["init"] == max(0,loop_counter["init_loops"]-1)):
                print("Including additional volatile bars in addition to solvepp result")
                for s in volatile_species:
                    key_pp = str(s+"_initial_atmos_pressure")
                    key_ab = str(s+"_add_bars")
                    key_in = str(s+"_included")
                    if (COUPLER_options[key_in] == 1):
                        COUPLER_options[key_pp] += float(COUPLER_options[key_ab] * 1.0e5)  # Convert bar -> Pa
                    

        ############### / INTERIOR SUB-LOOP




        ############### ATMOSPHERE SUB-LOOP
        while (loop_counter["atm"] == 0) or (loop_counter["atm"] < loop_counter["atm_loops"] and abs(COUPLER_options["F_net"]) > COUPLER_options["F_eps"]):

            # Initialize atmosphere structure
            atm, COUPLER_options = StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options )

            # Run SOCRATES: update TOA heating and MO heat flux
            atm, COUPLER_options = RunAEOLUS( atm, time_dict, dirs, COUPLER_options )
             
            # Run VULCAN (settings-dependent): update atmosphere mixing ratios
            if (COUPLER_options["atmosphere_chem_type"] == 2):
                atm = RunVULCAN( atm, time_dict, loop_counter, dirs, runtime_helpfile, COUPLER_options)

            # Update help quantities, input_flag: "Atmosphere"
            runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", COUPLER_options)

            # Iterate
            loop_counter["atm"] += 1

        ############### / ATMOSPHERE SUB-LOOP
        


        ############### LOOP ITERATION MANAGEMENT
        # Print info, save atm to file, update plots
        PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options, atm, loop_counter, dirs)

        # Plot conditions throughout run for on-the-fly analysis
        if (COUPLER_options["plot_iterfreq"] > 0) and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0):
            UpdatePlots( dirs["output"], COUPLER_options )
        
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
    UpdatePlots( dirs["output"], COUPLER_options )

    print("\n\n===> PROTEUS run finished successfully <===")
    print("     "+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

#====================================================================
if __name__ == '__main__':
    main()

# End of file
