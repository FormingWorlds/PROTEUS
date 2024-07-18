#!/usr/bin/env python3

"""
PROTEUS Main file
"""

from utils.modules_ext import *
import utils.constants

from utils.constants import *
from utils.coupler import *
from utils.janus import RunJANUS, PrepAtm, StructAtm
from utils.agni import RunAGNI
from utils.dummy_atmosphere import RunDummyAtm
from utils.spider import RunSPIDER
from utils.surface_gases import *
from utils.logging import setup_logger

from plot.cpl_fluxes import *
from plot.cpl_heatingrates import *

from janus.utils.StellarSpectrum import PrepareStellarSpectrum,InsertStellarSpectrum

import mors 

#====================================================================
def main():

    # Parse console arguments
    args = parse_console_arguments()

    # Read in COUPLER input file
    COUPLER_options, time_dict = ReadInitFile( args["cfg"] , verbose=False )

    # Set directories dictionary
    utils.constants.dirs = SetDirectories(COUPLER_options)
    from utils.constants import dirs
    UpdateStatusfile(dirs, 0)
    
    # Switch to logger 
    setup_logger(logpath=dirs["output"]+"std.log", logterm=True, level=COUPLER_options["log_level"])
    log = logging.getLogger("PROTEUS")

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
                    "total_min"  : 10,     # Minimum number of total loops
                    "total_loops": COUPLER_options["iter_max"],   # Maximum number of total loops

                    "init": 0,             # Number of init iters performed
                    "init_loops": 1,       # Maximum number of init iters

                    "atm": 0,              # Number of atmosphere sub-iters performed
                    "atm_loops":  20,      # Maximum number of atmosphere sub-iters

                    "steady": 0,           # Number of iterations passed since steady-state declared
                    "steady_loops": 3,     # Number of iterations to perform post-steady state
                    "steady_check": 15     # Number of iterations to look backwards when checking steady state
                    }
    
    # Model has completed?
    finished = False

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
    
    if COUPLER_options["interior_nlev"] < 40:
        UpdateStatusfile(dirs, 20)
        raise Exception("Interior must have at least 40 levels")
    
    # Clean output folders
    COUPLER_options["IC_INTERIOR"] = 1  # start from initial condition (not resuming)
    CleanDir( dirs["output"] , keep_stdlog=True)
    CleanDir( dirs['output']+'data/')
    runtime_helpfile    = []
    
    # Copy config file to output directory, for future reference
    shutil.copyfile( args["cfg"], dirs["output"]+"/init_coupler.cfg")

    # Calculate mantle mass (liquid + solid)
    COUPLER_options["mantle_mass"] = calc_mantle_mass(COUPLER_options)
    
    # Zero-out volatiles, and ensure that they are all tracked
    for s in volatile_species:
        key_pp = str(s+"_initial_bar")

        key_in = str(s+"_included")
        if (key_in in COUPLER_options) and (key_pp in COUPLER_options):
            COUPLER_options[key_in] = int(COUPLER_options[key_in])
            COUPLER_options[key_pp] = float(COUPLER_options[key_pp])
        else:
            COUPLER_options[key_in] = 0
            COUPLER_options[key_pp] = 0.0

    # Required vols
    for s in ["H2O","CO2","N2","S2"]:
        COUPLER_options[s+"_included"] = 1

    # Work out which vols are included
    solvevol_warnboth = False
    for s in volatile_species:
        key_pp = str(s+"_initial_bar")
        key_in = str(s+"_included")

        if (COUPLER_options[key_pp] > 0.0) and (COUPLER_options[key_in] == 0):
            raise Exception("Volatile %s has non-zero pressure but is disabled in cfg"%s)
        
        solvevol_warnboth = solvevol_warnboth or ((COUPLER_options[key_pp] > 0.0) and (COUPLER_options["solvevol_use_params"] > 0))
        log.info("%s\t: %s,  %.2f bar"%(s, str(COUPLER_options[key_in]>0), COUPLER_options[key_pp]))

        if COUPLER_options["solvevol_use_params"] > 0:
            COUPLER_options[key_pp] = 0.0

    # Warn (once)
    if solvevol_warnboth:
        log.warning("Parameterised solvevol is enabled but initial_bar is non-zero for at least one volatile")
        log.warning("Ignoring pressure provided in cfg file")

    # Check that all partial pressures are positive
    inc_vols = []
    for s in volatile_species:
        key_pp = str(s+"_initial_bar")
        key_in = str(s+"_included")
        if (COUPLER_options[key_in] > 0):

            # Ensure numerically reasonable
            COUPLER_options[key_pp] = max( COUPLER_options[key_pp], 1.0e-30)

            # Store
            inc_vols.append(s)
            
    log.info("Included volatiles: " + str(inc_vols))

    # Check that spectral file exists
    spectral_file_nostar = COUPLER_options["spectral_file"]
    if not os.path.exists(spectral_file_nostar):
        UpdateStatusfile(dirs, 20)
        raise Exception("Spectral file does not exist at '%s'" % spectral_file_nostar)

    # Handle stellar spectrum...

    # Store copy of modern spectrum in memory (1 AU)
    time_dict['sspec_prev'] = -math.inf
    time_dict['sinst_prev'] = -math.inf
    shutil.copyfile(COUPLER_options["star_spectrum"], os.path.join(dirs["output"],"-1.sflux"))

    # Prepare stellar models
    match COUPLER_options['star_model']:
        case 0: # SPADA (MORS)
            # download evolution track data if not present
            mors.DownloadEvolutionTracks("/Spada")

            # load modern spectrum 
            star_struct_modern = mors.spec.Spectrum()
            star_struct_modern.LoadTSV(COUPLER_options["star_spectrum"])
            star_struct_modern.CalcBandFluxes()

            # get best rotation percentile 
            star_pctle, _ = mors.synthesis.FitModernProperties(star_struct_modern, COUPLER_options["star_mass"], COUPLER_options["star_age_modern"]/1e6)

            # modern properties 
            star_props_modern = mors.synthesis.GetProperties(COUPLER_options["star_mass"], star_pctle, COUPLER_options["star_age_modern"]/1e6)

        case 1:  # BARAFFE
            modern_wl, modern_fl = mors.ModernSpectrumLoad(dirs["coupler"]+"/"+COUPLER_options["star_spectrum"], #path to input spectral file
                                                           dirs['output']+'/-1.sflux') #path to copied spectral file

            mors.DownloadEvolutionTracks("/Baraffe")
            baraffe = mors.BaraffeTrack(COUPLER_options["star_mass"])

        case _:
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid stellar model '%d'" % COUPLER_options['star_model'])
        
    # Create lockfile 
    keepalive_file = os.path.join(dirs["output"],"keepalive")
    safe_rm(keepalive_file)
    with open(keepalive_file, 'w') as fp:
        fp.write("Removing this file will be interpreted by PROTEUS as a request to stop the simulation loop\n")
    
    
    # Main loop
    UpdateStatusfile(dirs, 1)
    while not finished:

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
                        COUPLER_options["star_radius"] = mors.Value(COUPLER_options["star_mass"],time_dict["star"]/1e6, 'Rstar') * mors.const.Rsun * 1.0e-2
                        S_0 =  mors.Value(COUPLER_options["star_mass"], time_dict["star"]/1e6, 'Lbol') * L_sun / ( 4. * np.pi * AU * AU * COUPLER_options["mean_distance"]**2.0 )
                    case 1:
                        COUPLER_options["star_radius"] = baraffe.BaraffeStellarRadius(time_dict["star"])
                        S_0 = baraffe.BaraffeSolarConstant(time_dict["star"], COUPLER_options["mean_distance"])

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
        if ( ( abs( time_dict['planet'] - time_dict['sspec_prev'] ) > COUPLER_options['sspec_dt_update'] ) \
            or (loop_counter["total"] == 0) ):
            
            time_dict['sspec_prev'] = time_dict['planet'] 

            log.info("Updating stellar spectrum") 
            match COUPLER_options['star_model']: 
                case 0:
                    synthetic = mors.synthesis.CalcScaledSpectrumFromProps(star_struct_modern, star_props_modern, time_dict["star"]/1e6)
                    fl = synthetic.fl   # at 1 AU
                    wl = synthetic.wl
                case 1:
                    fl = baraffe.BaraffeSpectrumCalc(time_dict["star"], COUPLER_options["star_luminosity_modern"], modern_fl)
                    wl = modern_wl

            # Scale fluxes from 1 AU to TOA 
            fl *= (1.0 / COUPLER_options["mean_distance"])**2.0
            mors.SpectrumWrite(time_dict,wl,fl,dirs['output']+'/data/')

            # Prepare spectral file for JANUS 
            if COUPLER_options["atmosphere_model"] == 0:
                # Generate a new SOCRATES spectral file containing this new spectrum
                star_spec_src = dirs["output"]+"socrates_star.txt"
                #    Update stdout
                old_stdout , old_stderr = sys.stdout , sys.stderr
                sys.stdout = StreamToLogger(log, logging.INFO)
                sys.stderr = StreamToLogger(log, logging.ERROR)
                #    Spectral file stuff
                PrepareStellarSpectrum(wl,fl,star_spec_src)
                InsertStellarSpectrum(  spectral_file_nostar,
                                        star_spec_src,
                                        dirs["output"]
                                    )
                os.remove(star_spec_src)
                #    Restore stdout
                sys.stdout , sys.stderr = old_stdout , old_stderr

            # Other cases...
            #  - AGNI will prepare the file itself
            #  - dummy_atmosphere does not require this file

        else:
            log.info("New spectrum not required at this time")

        ############### / STELLAR FLUX MANAGEMENT



        ############### INTERIOR SUB-LOOP

        # Run SPIDER
        COUPLER_options = RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile )

        # Run outgassing model

        #    get info from spider
        if loop_counter["total"] > 0:
            run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior']
            COUPLER_options["T_outgas"] =    run_int.iloc[-1]["T_surf"]
            COUPLER_options["Phi_global"] =  run_int.iloc[-1]["Phi_global"]

        #    reset target if during init phase 
        #    since these target masses will be adjusted depending on the true melt fraction and T_outgas
        if loop_counter["init"] < loop_counter["init_loops"]:

            # calculate target mass of atoms
            if COUPLER_options["solvevol_use_params"] > 0:
                solvevol_target = solvevol_get_target_from_params(COUPLER_options)
            else:
                solvevol_target = solvevol_get_target_from_pressures(COUPLER_options)

            # prevent numerical issues
            for key in solvevol_target.keys():
                if solvevol_target[key] < 1.0e4:
                    solvevol_target[key] = 0.0
                log.debug("Solvevol target %s = %g kg"%(key, solvevol_target[key]))
        
        #    do calculation
        solvevol_dict = solvevol_equilibrium_atmosphere(solvevol_target, COUPLER_options)

        # Update help quantities, input_flag: "Interior"
        runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Interior", COUPLER_options, solvevol_dict=solvevol_dict)

        ############### / INTERIOR SUB-LOOP
                        


        ############### UPDATE TIME 
                                   
        # Advance current time in main loop according to interior step
        time_dict["planet"] = runtime_helpfile.iloc[-1]["Time"]
        time_dict["star"]   = time_dict["planet"] + time_dict["offset"]

        # Update init loop counter
        if loop_counter["init"] < loop_counter["init_loops"]: # Next init iter
            loop_counter["init"]    += 1
            time_dict["planet"]     = 0.
        if loop_counter["total"] >= loop_counter["init_loops"]: # Reset restart flag once SPIDER has correct heat flux
            COUPLER_options["IC_INTERIOR"] = 2

        # Adjust total iteration counters
        loop_counter["atm"]         = 0
        loop_counter["total"]       += 1

        ############### / UPDATE TIME



        ############### ATMOSPHERE SUB-LOOP
        while (loop_counter["atm"] == 0):

            # Initialize atmosphere structure
            if (loop_counter["atm"] == 0) or (loop_counter["total"] <= 2):
                COUPLER_options = PrepAtm(loop_counter, runtime_helpfile, COUPLER_options)

                if COUPLER_options["atmosphere_model"] == 0:
                    # Run JANUS: use the general adiabat to create a PT profile, then calculate fluxes
                    atm = StructAtm( dirs, runtime_helpfile, COUPLER_options )
                    COUPLER_options = RunJANUS( atm, time_dict, dirs, COUPLER_options, runtime_helpfile)

                elif COUPLER_options["atmosphere_model"] == 1:
                    # Run AGNI 
                    COUPLER_options = RunAGNI(loop_counter, time_dict, dirs, COUPLER_options, runtime_helpfile)

                elif COUPLER_options["atmosphere_model"] == 2:
                    # Run dummy atmosphere model 
                    COUPLER_options = RunDummyAtm(time_dict, dirs, COUPLER_options, runtime_helpfile)
                    
                else:
                    UpdateStatusfile(dirs, 20)
                    raise Exception("Invalid atmosphere model")

            
            # Update help quantities, input_flag: "Atmosphere"
            runtime_helpfile, time_dict, COUPLER_options = UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, "Atmosphere", COUPLER_options)

            # Iterate
            loop_counter["atm"] += 1

        ############### / ATMOSPHERE SUB-LOOP
        


        ############### CONVERGENCE CHECK 

        # Print info to terminal and log file
        PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options)

        # Stop simulation when planet is completely solidified
        if (COUPLER_options["solid_stop"] == 1) and (runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]):
            UpdateStatusfile(dirs, 10)
            log.info("")
            log.info("===> Planet solidified! <===")
            log.info("")
            finished = True

        # Stop simulation when flux is small
        if (COUPLER_options["emit_stop"] == 1) and (loop_counter["total"] > loop_counter["init_loops"]+1) \
            and ( abs(runtime_helpfile.iloc[-1]["F_atm"]) <= COUPLER_options["F_crit"]):
            UpdateStatusfile(dirs, 14)
            log.info("")
            log.info("===> Planet is no longer cooling! <===")
            log.info("")
            finished = True

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
            flx_m = max(abs(arr_f[lb1]),abs(arr_f[lb2]))

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
                log.info("")
                log.info("===> Planet has entered a steady state! <===")
                log.info("")
                finished = True
            
        # Stop simulation if maximum time reached
        if (time_dict["planet"] >= time_dict["target"]):
            UpdateStatusfile(dirs, 13)
            log.info("")
            log.info("===> Target time reached! <===")
            log.info("")
            finished = True
        
        # Check if the minimum number of loops have been performed
        if finished and (loop_counter["total"] < loop_counter["total_min"]):
            log.info("Minimum number of iterations not yet attained; continuing...")
            finished = False
            UpdateStatusfile(dirs, 1)

        # Stop simulation if maximum loops reached
        if (loop_counter["total"] > loop_counter["total_loops"]):
            UpdateStatusfile(dirs, 12)
            log.info("")
            log.info("===> Maximum number of iterations reached! <===")
            log.info("")
            finished = True

        # Check if keepalive file has been removed - this means that the model should exit ASAP
        if not os.path.exists(keepalive_file):
            UpdateStatusfile(dirs, 25)
            log.info("")
            log.info("===> Model exit was requested by user! <===")
            log.info("")
            finished=True

        # Make plots if required and go to next iteration
        if (COUPLER_options["plot_iterfreq"] > 0) and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0) and (not finished):
            UpdatePlots( dirs["output"], COUPLER_options )

        ############### / CONVERGENCE CHECK 

    # ----------------------
    # FINAL THINGS BEFORE EXIT

    # Clean up files
    safe_rm(keepalive_file)
    for file in glob.glob(dirs["output"]+"/runtime_spectral_file*"):
        os.remove(file)

    # Plot conditions at the end
    UpdatePlots( dirs["output"], COUPLER_options, end=True)
    end_time = datetime.now()
    run_time = end_time - start_time
    log.info("Simulation completed at: "+end_time.strftime('%Y-%m-%d_%H:%M:%S'))
    log.info("Total runtime: %.2f hours" % ( run_time.total_seconds()/(60.0 * 60.0)  ))

    # EXIT
    return

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
