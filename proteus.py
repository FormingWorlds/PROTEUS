#!/usr/bin/env python3

"""
PROTEUS Main file
"""

from utils.modules_ext import *
import utils.constants

from utils.constants import *
from utils.coupler import *
from utils.stellar_common import *
from utils.stellar_mors import *
from utils.stellar_baraffe import *
from utils.aeolus import RunAEOLUS, PrepAtm, StructAtm
from utils.agni import RunAGNI
from utils.spider import RunSPIDER

from plot.cpl_fluxes import *
from plot.cpl_heatingrates import *

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

    os.chdir(dirs["coupler"])

    print("Hostname: " + str(os.uname()[1]))

    print("Output directory: '%s'" % dirs["output"])

    # Count iterations
    loop_counter = { 
                    "total": 0,            # Total number of itersperformed
                    "init": 0,             # Number of init iters performed
                    "atm": 0,              # Number of atmosphere sub-iters performed
                    "total_loops": 2000,   # Maximum number of total loops
                    "init_loops": 3,       # Maximum number of init iters
                    "atm_loops":  20,      # Maximum number of atmosphere sub-iters
                    }
    
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
        CleanDir( dirs['output']+'data/')
        
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
                print("       Consider assigning volatile '%s' a small positive value." % s)
                print("       Check that solvepp_enabled has the intended setting.")
                exit(1)

            # Ensure numerically reasonable
            if (COUPLER_options[key_pp] <= 1e-3):
                COUPLER_options[key_pp] = 1e-3
            

    print("Included volatiles:",inc_vols)

    # Check that spectral file exists
    spectral_file_nostar = COUPLER_options["spectral_file"]
    if not os.path.exists(spectral_file_nostar):
        print("ERROR: Spectral file does not exist at '%s'!" % spectral_file_nostar)
        exit(1)

    # Handle stellar spectrum...

    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)
    time_dict['sspec_prev'] = -math.inf
    time_dict['sinst_prev'] = -math.inf

    # Prepare stellar models
    match COUPLER_options['star_model']:
        case 0:
            COUPLER_options["star_radius"] = COUPLER_options["star_radius_modern"]  # Legacy stellar model doesn't update this
        case 1:
            MorsSolveUV(dirs,COUPLER_options,StellarFlux_wl,StellarFlux_fl)  # Solve for UV-PL band interface
            COUPLER_options = MorsCalculateFband(dirs, COUPLER_options)
        case 2:
            track = BaraffeLoadtrack(COUPLER_options)
        case _:
            print("ERROR: Invalid stellar model '%d'" % COUPLER_options['star_model'])
            exit(1)
    

    # Main loop
    while time_dict["planet"] < time_dict["target"]:

        PrintSeparator()
        print("Loop counters:", loop_counter)

        ############### STELLAR FLUX MANAGEMENT
        print("Stellar flux management...")
        
        # Calculate new instellation and radius
        if (abs( time_dict['planet'] - time_dict['sinst_prev'] ) > COUPLER_options['sinst_dt_update']) \
            or (loop_counter["total"] == 0):
            
            time_dict['sinst_prev'] = time_dict['planet'] 

            if (COUPLER_options["stellar_heating"] > 0):
                print("Updating instellation and radius")

                match COUPLER_options['star_model']:
                    case 0:
                        S_0, toa_heating = InterpolateStellarLuminosity(time_dict, COUPLER_options)
                    case 1:
                        COUPLER_options["star_radius"] = MorsStellarRadius(time_dict, COUPLER_options)
                        S_0, toa_heating = MorsSolarConstant(time_dict, COUPLER_options)
                    case 2:
                        COUPLER_options["star_radius"] = BaraffeStellarRadius(time_dict, COUPLER_options, track)
                        S_0, toa_heating = BaraffeSolarConstant(time_dict, COUPLER_options, track)

                # Calculate new eqm temperature
                T_eqm_new = calc_eqm_temperature(S_0,  COUPLER_options["albedo_pl"])
                
                # Get old eqm temperature
                if (loop_counter["total"] > 0):
                    T_eqm_prev = COUPLER_options["T_eqm"]
                else:
                    T_eqm_prev = 0.0
                
            else:
                print("Stellar heating is disabled")
                toa_heating = 0.0
                T_eqm_new   = 0.0
                T_eqm_prev  = 0.0

            COUPLER_options["TOA_heating"]  = toa_heating
            COUPLER_options["T_eqm"]        = T_eqm_new
            COUPLER_options["T_skin"]       = T_eqm_new * (0.5**0.25) # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)

            print("T_eqm change: %.3f K (to 3dp)" % abs(T_eqm_new - T_eqm_prev))

        # Calculate a new (historical) stellar spectrum 
        if (COUPLER_options['star_model'] > 0  \
            and ( abs( time_dict['planet'] - time_dict['sspec_prev'] ) > COUPLER_options['sspec_dt_update'] ) \
            or (loop_counter["total"] == 0) ):
            
            time_dict['sspec_prev'] = time_dict['planet'] 

            print("Updating stellar spectrum") 
            match COUPLER_options['star_model']: 
                case 1:
                    fl,fls = MorsSpectrumCalc(time_dict['star'], StellarFlux_wl, StellarFlux_fl, COUPLER_options)
                case 2:
                    fl,fls = BaraffeSpectrumCalc(time_dict['star'], StellarFlux_fl, COUPLER_options, track)

            # Write stellar spectra to disk
            writessurf = bool(COUPLER_options["atmosphere_chem_type"] > 0)
            SpectrumWrite(time_dict,StellarFlux_wl,fl,fls,dirs['output']+'/data/',write_surf=writessurf)

            # Generate a new SOCRATES spectral file containing this new spectrum
            star_spec_src = dirs["output"]+"socrates_star.txt"
            PrepareStellarSpectrum(StellarFlux_wl,fl,star_spec_src)
            InsertStellarSpectrum(
                                    spectral_file_nostar,
                                    star_spec_src,
                                    dirs["output"]+"runtime_spectral_file"
                                )
            os.remove(star_spec_src)

        else:
            print("New spectrum not required at this time")

        ############### / STELLAR FLUX MANAGEMENT



        ############### INTERIOR SUB-LOOP

        # Run SPIDER
        COUPLER_options = RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile )

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", COUPLER_options)

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
        while (loop_counter["atm"] == 0):

            # Initialize atmosphere structure
            if (loop_counter["atm"] == 0) or (loop_counter["total"] <= 2):
                COUPLER_options = PrepAtm(loop_counter, runtime_helpfile, COUPLER_options)

                if COUPLER_options["atmosphere_model"] == 0:
                    # Run AEOLUS: use the general adiabat to create a PT profile, then calculate fluxes
                    atm = StructAtm( runtime_helpfile, COUPLER_options )
                    COUPLER_options = RunAEOLUS( atm, time_dict, dirs, COUPLER_options, runtime_helpfile )

                elif COUPLER_options["atmosphere_model"] == 1:
                    # Run AGNI 
                    COUPLER_options = RunAGNI(time_dict, dirs, COUPLER_options, runtime_helpfile)
                    
                else:
                    raise Exception("Invalid atmosphere model")

            
            # Update help quantities, input_flag: "Atmosphere"
            runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", COUPLER_options)

            # Iterate
            loop_counter["atm"] += 1

        ############### / ATMOSPHERE SUB-LOOP
        


        ############### LOOP ITERATION MANAGEMENT

        # Advance current time in main loop
        time_dict["planet"] = runtime_helpfile.iloc[-1]["Time"]
        time_dict["star"]   = time_dict["planet"] + time_dict["offset"]

        # Print info, save atm to file
        PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options)
        
        # Update init loop counter
        if loop_counter["init"] < loop_counter["init_loops"]: # Next init iter
            loop_counter["init"]    += 1
            time_dict["planet"]     = 0.
        if loop_counter["total"] >= loop_counter["init_loops"]: # Reset restart flag once SPIDER was started w/ ~correct volatile chemistry + heat flux
            COUPLER_options["IC_INTERIOR"] = 2

        # Adjust total iteration counters
        loop_counter["atm"]         = 0
        loop_counter["total"]       += 1


        # Stop simulation when planet is completely solidified
        if (COUPLER_options["solid_stop"] == 1) \
            and (runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]):
            print("\n===> Planet solidified! <===\n")
            break
            
        # Stop simulation if maximum loops reached
        if (loop_counter["total"] > loop_counter["total_loops"]):
            print("\n Maximum number of iterations reached. Stopping. \n")
            break
        
        # Make plots if required and go to next iteration
        if (COUPLER_options["plot_iterfreq"] > 0) \
            and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0):
            UpdatePlots( dirs["output"], COUPLER_options )


        ############### / LOOP ITERATION MANAGEMENT

    # Plot conditions at the end
    UpdatePlots( dirs["output"], COUPLER_options )
    print("     "+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

#====================================================================
if __name__ == '__main__':
    main()
    print("Goodbye")

# End of file
