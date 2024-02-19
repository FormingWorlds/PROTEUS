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
from utils.logging import setup_logger

from plot.cpl_fluxes import *
from plot.cpl_heatingrates import *

from AEOLUS.modules.stellar_luminosity import InterpolateStellarLuminosity
from AEOLUS.utils.StellarSpectrum import PrepareStellarSpectrum,InsertStellarSpectrum

#====================================================================
def main():

    # Parse console arguments
    args = parse_console_arguments()

    # Read in COUPLER input file
    COUPLER_options, time_dict = ReadInitFile( args.cfg_file , verbose=False )

    # Set directories dictionary
    utils.constants.dirs = SetDirectories(COUPLER_options)
    from utils.constants import dirs
    UpdateStatusfile(dirs, 0)
    
    # Switch to logger 
    setup_logger(logpath=dirs["output"]+"std.log", logterm=True, level=1)
    log = logging.getLogger(__name__)

    log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    log.info("            PROTEUS framework (version 0.1)            ")
    log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    os.chdir(dirs["coupler"])

    start_time = datetime.now()
    log.info("Current time:     " + start_time.strftime('%Y-%m-%d_%H:%M:%S'))
    log.info("Hostname:         " + str(os.uname()[1]))
    log.info("Output directory: " + dirs["output"])

    # Count iterations
    loop_counter = { 
                    "total": 0,            # Total number of iters performed
                    "total_min"  : 30,     # Minimum number of total loops
                    "total_loops": 2000,   # Maximum number of total loops

                    "init": 0,             # Number of init iters performed
                    "init_loops": 2,       # Maximum number of init iters

                    "atm": 0,              # Number of atmosphere sub-iters performed
                    "atm_loops":  20,      # Maximum number of atmosphere sub-iters

                    "steady": 0,           # Number of iterations passed since steady-state declared
                    "steady_loops": 3,     # Number of iterations to perform post-steady state
                    "steady_check": 25     # Number of iterations to look backwards when checking steady state
                    }
    
    # Model has completed?
    complete = False
    
    # Check options are compatible
    if COUPLER_options["atmosphere_surf_state"] == 2: # Not all surface treatments are mutually compatible
        if COUPLER_options["flux_convergence"] == 1:
            UpdateStatusfile(dirs, 20)
            raise Exception("Shallow mixed layer scheme is incompatible with the conductive lid scheme! Turn one of them off")
        if COUPLER_options["PARAM_UTBL"] == 1:
            UpdateStatusfile(dirs, 20)
            raise Exception("SPIDER's UTBL is incompatible with the conductive lid scheme! Turn one of them off")
        
    if COUPLER_options["atmosphere_model"] == 1:  # Julia required for AGNI
        if shutil.which("julia") is None:
            UpdateStatusfile(dirs, 20)
            raise Exception("Could not find Julia in current environment")
        
    if COUPLER_options["atmosphere_nlev"] < 15:
        UpdateStatusfile(dirs, 20)
        raise Exception("Atmosphere must have at least 15 levels")
    
    # If restart skip init loop # args.r or args.rf or 
    if COUPLER_options["IC_INTERIOR"] == 2:

        # Check if output directory actually exists
        if (not os.path.exists(dirs["output"])) or (not os.path.exists(dirs["output"]+'/data/')):
            UpdateStatusfile(dirs, 20)
            raise Exception("Cannot resume run because directory doesn't exist")

        loop_counter["total"] += loop_counter["init_loops"]
        loop_counter["init"]  += loop_counter["init_loops"]

        # Restart file name if not specified: use last file output
        if (COUPLER_options["IC_INTERIOR"] == 2 and COUPLER_options["ic_interior_filename"] == 0):
            COUPLER_options["ic_interior_filename"] = str(natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/data/*.json")])[-1])

        # Clean all overtimes from present helpfile
        runtime_helpfile = pd.read_csv(dirs["output"]+"/"+"runtime_helpfile.csv", index_col=False, header=0, sep="\t")
        t_curr = COUPLER_options["ic_interior_filename"][:-5]
        log.debug("Clean helpfile from overtimes " + str(t_curr) + " yr")
        runtime_helpfile = runtime_helpfile.loc[runtime_helpfile["Time"] <= int(t_curr)]

        COUPLER_options["F_int"] = runtime_helpfile.iloc[-1]["F_int"]
        COUPLER_options["F_atm"] = runtime_helpfile.iloc[-1]["F_atm"]
        COUPLER_options["F_net"] = runtime_helpfile.iloc[-1]["F_net"]

    # Start conditions and help files depending on restart option
    else:
        CleanDir( dirs["output"] , keep_stdlog=True)
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
                    if solvepp_dict[s] > 1e-3:
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
                UpdateStatusfile(dirs, 20)
                raise Exception("Partial pressures of included volatiles must be positive. Consider assigning volatile '%s' a small positive value. Check that solvepp_enabled has the intended setting." % s)

            # Ensure numerically reasonable
            if (COUPLER_options[key_pp] <= 1e-3):
                COUPLER_options[key_pp] = 1e-3
            

    log.info("Included volatiles: " + str(inc_vols))

    # Check that spectral file exists
    spectral_file_nostar = COUPLER_options["spectral_file"]
    if not os.path.exists(spectral_file_nostar):
        UpdateStatusfile(dirs, 20)
        raise Exception("Spectral file does not exist at '%s'" % spectral_file_nostar)

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
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid stellar model '%d'" % COUPLER_options['star_model'])
    
    # Main loop
    UpdateStatusfile(dirs, 1)
    while time_dict["planet"] < time_dict["target"]:

        PrintSeparator()
        log.info("Loop counters: " +  str(loop_counter))

        ############### STELLAR FLUX MANAGEMENT
        log.info("Stellar flux management...")
        
        # Calculate new instellation and radius
        if (abs( time_dict['planet'] - time_dict['sinst_prev'] ) > COUPLER_options['sinst_dt_update']) \
            or (loop_counter["total"] == 0):
            
            time_dict['sinst_prev'] = time_dict['planet'] 

            # Get previous instellation
            if (loop_counter["total"] > 0):
                F_inst_prev = S_0
            else:
                F_inst_prev = 0.0

            if (COUPLER_options["stellar_heating"] > 0):
                log.info("Updating instellation and radius")

                match COUPLER_options['star_model']:
                    case 0:
                        S_0 = InterpolateStellarLuminosity(time_dict, COUPLER_options)
                    case 1:
                        COUPLER_options["star_radius"] = MorsStellarRadius(time_dict, COUPLER_options)
                        S_0 = MorsSolarConstant(time_dict, COUPLER_options)
                    case 2:
                        COUPLER_options["star_radius"] = BaraffeStellarRadius(time_dict, COUPLER_options, track)
                        S_0 = BaraffeSolarConstant(time_dict, COUPLER_options, track)

                # Calculate new eqm temperature
                T_eqm_new = calc_eqm_temperature(S_0, COUPLER_options["asf_scalefactor"], COUPLER_options["albedo_pl"])
                
            else:
                log.info("Stellar heating is disabled")
                T_eqm_new   = 0.0
                S_0 = 0.0

            COUPLER_options["F_ins"]        = S_0          # instellation 
            COUPLER_options["T_eqm"]        = T_eqm_new
            COUPLER_options["T_skin"]       = T_eqm_new * (0.5**0.25) # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)

            log.info("Instellation change: %+.4e W m-2 (to 4dp)" % abs(S_0 - F_inst_prev))

        # Calculate a new (historical) stellar spectrum 
        if (COUPLER_options['star_model'] > 0  \
            and ( abs( time_dict['planet'] - time_dict['sspec_prev'] ) > COUPLER_options['sspec_dt_update'] ) \
            or (loop_counter["total"] == 0) ):
            
            time_dict['sspec_prev'] = time_dict['planet'] 

            log.info("Updating stellar spectrum") 
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
            #    Update stdout
            old_stdout , old_stderr = sys.stdout , sys.stderr
            sys.stdout = StreamToLogger(log, logging.INFO)
            sys.stderr = StreamToLogger(log, logging.ERROR)
            #    Spectral file stuff
            PrepareStellarSpectrum(StellarFlux_wl,fl,star_spec_src)
            InsertStellarSpectrum(
                                    spectral_file_nostar,
                                    star_spec_src,
                                    dirs["output"]+"runtime_spectral_file"
                                )
            os.remove(star_spec_src)
            #    Restore stdout
            sys.stdout , sys.stderr = old_stdout , old_stderr

        else:
            log.info("New spectrum not required at this time")

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
                log.info("Including additional volatile bars in addition to solvepp result")
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
                    atm = StructAtm( dirs, runtime_helpfile, COUPLER_options )
                    COUPLER_options = RunAEOLUS( atm, time_dict, dirs, COUPLER_options, runtime_helpfile )

                elif COUPLER_options["atmosphere_model"] == 1:
                    # Run AGNI 
                    COUPLER_options = RunAGNI(loop_counter, time_dict, dirs, COUPLER_options, runtime_helpfile)
                    
                else:
                    UpdateStatusfile(dirs, 20)
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
        if (COUPLER_options["solid_stop"] == 1) and (runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]):
            UpdateStatusfile(dirs, 10)
            log.info("\n===> Planet solidified! <===\n")
            complete = True

        # Determine when the simulation enters a steady state
        if (COUPLER_options["steady_stop"] == 1) and (loop_counter["total"] > loop_counter["steady_check"]*2+5) and (loop_counter["steady"] == 0):
            # How many iterations to look backwards
            lb1 = -int(loop_counter["steady_check"])
            lb2 = -1

            # Get data
            df_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
            arr_t = np.array(df_atm["Time"])
            arr_f = np.array(df_atm["F_atm"])
            arr_p = np.array(df_atm["Phi_global"])

            # Time samples
            t1 = arr_t[lb1]; t2 = arr_t[lb2]

            # Flux samples
            flx_m = max(arr_f[lb1],arr_f[lb2])

            # Check melt fraction rate
            phi_1 =  arr_p[lb1]
            phi_2 =  arr_p[lb2]
            phi_r = abs(phi_2 - phi_1) / (t2 - t1)
                        
            # Stop when flux is small and melt fraction is unchanging
            if (flx_m < COUPLER_options["steady_flux"]) and (phi_r < COUPLER_options["steady_dprel"]):
                log.info("Steady state declared")
                loop_counter["steady"] = 1

        # Steady-state handling
        if loop_counter["steady"] > 0:
            if loop_counter["steady"] <= loop_counter["steady_loops"]:
                loop_counter["steady"] += 1
            else:
                UpdateStatusfile(dirs, 11)
                log.info("\n===> Planet has entered a steady state! <===\n")
                complete = True
            
        # Stop simulation if maximum loops reached
        if (loop_counter["total"] > loop_counter["total_loops"]):
            UpdateStatusfile(dirs, 12)
            log.info("\n===> Maximum number of iterations reached. <===\n")
            complete = True
        
        # Make plots if required and go to next iteration
        if (COUPLER_options["plot_iterfreq"] > 0) and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0):
            UpdatePlots( dirs["output"], COUPLER_options )

        # Check if the minimum number of loops have been performed
        if complete and (loop_counter["total"] < loop_counter["total_min"]):
            log.info("Minimum number of iterations not yet attained")
            complete = False

        # If marked as complete, exit
        if complete:
            break

        ############### / LOOP ITERATION MANAGEMENT
            
    # Plot conditions at the end
    UpdatePlots( dirs["output"], COUPLER_options, end=True)
    end_time = datetime.now()
    run_time = end_time - start_time
    log.info("Simulation completed at: "+end_time.strftime('%Y-%m-%d_%H:%M:%S'))
    log.info("Total runtime: %.2f hours" % ( run_time.total_seconds()/(60.0 * 60.0)  ))

#====================================================================
if __name__ == '__main__':
    # Check that environment variables are set 
    if os.environ.get('COUPLER_DIR') == None:
        raise Exception("Environment variables not set! Have you sourced PROTEUS.env?")
    # Start main function
    main()
    print("Goodbye")
    exit(0)

# End of file
