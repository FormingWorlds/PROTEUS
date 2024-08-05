#!/usr/bin/env python3

"""
PROTEUS Main file
"""

import utils.constants
from utils.constants import *

from utils.modules_ext import *
from utils.coupler import *
from utils.spider import RunSPIDER, ReadSPIDER
from utils.surface_gases import get_target_from_params, get_target_from_pressures, equilibrium_atmosphere, CalculateMantleMass
from utils.logs import SetupLogger, GetLogfilePath, GetCurrentLogfileIndex, StreamToLogger

from janus.utils import DownloadSpectralFiles, DownloadStellarSpectra
import mors 

#====================================================================
def main():

    # Parse console arguments
    args = parse_console_arguments()
    resume = bool(args["resume"])

    # Read in COUPLER input file
    cfgsrc = os.path.abspath(str(args["cfg"]))
    OPTIONS = ReadInitFile( cfgsrc , verbose=False )

    # Set directories dictionary
    utils.constants.dirs = SetDirectories(OPTIONS)
    from utils.constants import dirs
    UpdateStatusfile(dirs, 0)

    # Validate options
    ValidateInitFile(dirs, OPTIONS)

    # Clean output directory 
    if not resume:
        CleanDir(dirs["output"])
        CleanDir(os.path.join(dirs['output'], 'data'))

    # Get next logfile path 
    logindex = 1 + GetCurrentLogfileIndex(dirs["output"])
    logpath = GetLogfilePath(dirs["output"], logindex)
    
    # Switch to logger 
    SetupLogger(logpath=logpath, logterm=True, level=OPTIONS["log_level"])
    log = logging.getLogger("PROTEUS")

    # Print information to logger
    log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    log.info("            PROTEUS framework (version 0.1)            ")
    log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    log.info(" ")
    start_time = datetime.now()
    log.info("Current time: " + start_time.strftime('%Y-%m-%d_%H:%M:%S'))
    log.info("Hostname    : " + str(os.uname()[1]))
    log.info("PROTEUS hash: " + GitRevision(dirs["proteus"])) 
    log.info("Py version  : " + sys.version.split(' ')[0]) 
    log.info("Config file : " + cfgsrc)
    log.info("Output dir  : " + dirs["output"])
    log.info("FWL data dir: " + dirs["fwl"])
    if OPTIONS["atmosphere_model"] in [0,1]:
        log.info("SOCRATES dir: " + dirs["rad"])
    log.info(" ")

    # Count iterations
    loop_counter = { 
                    "total": 0,            # Total number of iters performed
                    "total_min"  : 5,      # Minimum number of total loops
                    "total_loops": OPTIONS["iter_max"],   # Maximum number of total loops

                    "init": 0,             # Number of init iters performed
                    "init_loops": 2,       # Maximum number of init iters

                    "steady": 0,           # Number of iterations passed since steady-state declared
                    "steady_loops": 3,     # Number of iterations to perform post-steady state
                    "steady_check": 15     # Number of iterations to look backwards when checking steady state
                    }
    
    # Model has completed?
    finished = False

    # Config file paths
    cfgbak = os.path.join(dirs["output"],"init_coupler.cfg")

    # Import the appropriate atmosphere module 
    if OPTIONS["atmosphere_model"] == 0:
        from utils.janus import RunJANUS, StructAtm, ShallowMixedOceanLayer
        from janus.utils.StellarSpectrum import PrepareStellarSpectrum,InsertStellarSpectrum

    elif OPTIONS["atmosphere_model"] == 1:
        from utils.agni import RunAGNI, InitAtmos, UpdateProfile, ActivateEnv, DeallocAtmos
        ActivateEnv(dirs)
        atm = None

    elif OPTIONS["atmosphere_model"] == 2:
        from utils.dummy_atmosphere import RunDummyAtm
        
    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid atmosphere model")
    
    # Import the appropriate escape module 
    if OPTIONS["escape_model"] == 0:
        pass 
    elif OPTIONS["escape_model"] == 1:
        from utils.escape import RunZEPHYRUS
    elif OPTIONS["escape_model"] == 2:
        from utils.escape import RunDummyEsc
    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid escape model")

    # Is the model resuming from a previous state?
    if not resume:
        # New simulation

        # SPIDER initial condition
        IC_INTERIOR = 1

        # Copy config file to output directory, for future reference
        shutil.copyfile(args["cfg"], cfgbak)

        # No previous iterations to be stored. It is therefore important that the 
        #    submodules do not try to read data from the 'past' iterations, since they do
        #    not yet exist.
        hf_all = None

        # Create an empty initial row for helpfile 
        hf_row = ZeroHelpfileRow()

        # Initial time 
        hf_row["Time"] =    0.0
        hf_row["age_star"]= OPTIONS["time_star"]

        # Initial guess for flux 
        hf_row["F_atm"]   = OPTIONS["F_atm"]
        hf_row["F_int"]   = hf_row["F_atm"]
        hf_row["T_eqm"]   = 2000.0

        # Planet size conversion, and calculate mantle mass (= liquid + solid)
        hf_row["M_planet"] = OPTIONS["mass"] * M_earth
        hf_row["R_planet"] = OPTIONS["radius"] * R_earth
        hf_row["gravity"] = const_G * hf_row["M_planet"] / (hf_row["R_planet"]**2.0)
        hf_row["M_mantle"] = CalculateMantleMass(hf_row["R_planet"], 
                                                 hf_row["M_planet"], 
                                                 OPTIONS["planet_coresize"])


        # Store partial pressures and list of included volatiles
        log.info("Input partial pressures:")
        inc_vols = []
        for s in volatile_species:
            key_pp = str(s+"_initial_bar")
            key_in = str(s+"_included")

            log.info("    %-6s : %-5.2f bar (included = %s)"%(s, OPTIONS[key_pp], 
                                                            str(OPTIONS[key_in]>0)))

            if OPTIONS[key_in] > 0:
                inc_vols.append(s)
                hf_row[s+"_bar"] = max(1.0e-30, float(OPTIONS[key_pp]))
            else:
                hf_row[s+"_bar"] = 0.0
        log.info("Included volatiles: " + str(inc_vols))

    else:
        # Resuming from disk
        log.info("Resuming the simulation from the disk")

        # Copy cfg file 
        if os.path.exists(cfgbak):
            if cfgbak == cfgsrc:
                # resuming from backed-up cfg file, not the original
                pass
            else:
                # delete old and copy new 
                safe_rm(cfgbak)
                shutil.copyfile(cfgsrc, cfgbak)

        # SPIDER initial condition
        IC_INTERIOR = 2

        # Read helpfile from disk 
        hf_all = ReadHelpfileFromCSV(dirs["output"])

        # Check length 
        if len(hf_all) <= loop_counter["init_loops"]+1:
            raise Exception("Simulation is too short to be resumed")

        # Get last row 
        hf_row = hf_all.iloc[-1].to_dict()

        # Set loop counters 
        loop_counter["total"] = len(hf_all)
        loop_counter["init"] = loop_counter["init_loops"]+1

        # Set instellation
        S_0 = hf_row["F_ins"]
        F_inst_prev = S_0

        # Set volatile mass targets f
        solvevol_target = {}
        for e in element_list:
            if e == 'O': continue
            solvevol_target[e] = hf_row[e+"_kg_total"]

    # Download all basic data.
    # (to be improved such that we only download the one we need)
    DownloadSpectralFiles()
    DownloadStellarSpectra()

    spectral_file_nostar = os.path.join(dirs["fwl"] , OPTIONS["spectral_file"])
    if not os.path.exists(spectral_file_nostar):
        UpdateStatusfile(dirs, 20)
        raise Exception("Spectral file does not exist at '%s'" % spectral_file_nostar)
    
    # Runtime spectral file path
    spfile_path = os.path.join(dirs["output"] , "runtime.sf")

    # Handle stellar spectrum...

    # Store copy of modern spectrum in memory (1 AU)
    sspec_prev = -np.inf
    sinst_prev = -np.inf
    star_modern_path = os.path.join(dirs["fwl"],OPTIONS["star_spectrum"])
    shutil.copyfile(star_modern_path, os.path.join(dirs["output"],"-1.sflux"))

    # Prepare stellar models
    match OPTIONS['star_model']:
        case 0: # SPADA (MORS)
            # download evolution track data if not present
            mors.DownloadEvolutionTracks("/Spada")

            # load modern spectrum 
            star_struct_modern = mors.spec.Spectrum()
            star_struct_modern.LoadTSV(star_modern_path)
            star_struct_modern.CalcBandFluxes()

            # get best rotation percentile 
            # star_pctle, _ = mors.synthesis.FitModernProperties(star_struct_modern, 
            #                                                    OPTIONS["star_mass"], 
            #                                                    OPTIONS["star_age_modern"]/1e6)

            # modern properties 
            star_props_modern = mors.synthesis.GetProperties(OPTIONS["star_mass"], 
                                                             OPTIONS["star_rot_pctle"], 
                                                             OPTIONS["star_age_modern"]/1e6)

        case 1:  # BARAFFE
            modern_wl, modern_fl = mors.ModernSpectrumLoad(star_modern_path, 
                                                           dirs['output']+'/-1.sflux') 

            mors.DownloadEvolutionTracks("/Baraffe")
            baraffe = mors.BaraffeTrack(OPTIONS["star_mass"])

        case _:
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid stellar model '%d'" % OPTIONS['star_model'])
        
    # Create lockfile 
    keepalive_file = CreateLockFile(dirs["output"])

    # Main loop
    UpdateStatusfile(dirs, 1)
    while not finished:

        # New rows
        if loop_counter["total"] > 0:
            # Create new row to hold the updated variables. This will be
            #    overwritten by the routines below.
            hf_row = hf_all.iloc[-1].to_dict()
            
        log.info(" ")
        PrintSeparator()
        log.info("Loop counters")
        log.info("init    total     steady")
        log.info("%1d/%1d   %04d/%04d   %02d/%02d" % (
                 loop_counter["init"],  loop_counter["init_loops"], 
                 loop_counter["total"], loop_counter["total_loops"],
                 loop_counter["steady"],loop_counter["steady_loops"]
                 ))


        ############### INTERIOR
        PrintHalfSeparator()

        # Previous magma temperature 
        if loop_counter["init"] < loop_counter["init_loops"]:
            prev_T_magma = 9000.0
        else:
            prev_T_magma = hf_all.iloc[-1]["T_magma"]

        # Run SPIDER
        RunSPIDER( dirs, OPTIONS, IC_INTERIOR, 
                        loop_counter, hf_all, hf_row )
        sim_time, spider_result = ReadSPIDER(dirs, OPTIONS, 
                                             hf_row["Time"], prev_T_magma, 
                                             hf_row["R_planet"])

        for k in spider_result.keys():
            if k in hf_row.keys():
                hf_row[k] = spider_result[k]

        # Do not allow melt fraction to increase
        if (OPTIONS["prevent_warming"] == 1) \
            and (loop_counter["init"] >= loop_counter["init_loops"]):
            hf_row["Phi_global"] = min(hf_row["Phi_global"], hf_all.iloc[-1]["Phi_global"])

        # Advance current time in main loop according to interior step
        dt = float(sim_time) - hf_row["Time"]
        hf_row["Time"]     += dt # in years 
        hf_row["age_star"] += dt # in years

        ############### / INTERIOR


        ############### STELLAR FLUX MANAGEMENT
        PrintHalfSeparator()
        log.info("Stellar flux management...")

        # Calculate new instellation and radius
        if (abs( hf_row["Time"] - sinst_prev ) > OPTIONS['sinst_dt_update']) \
            or (loop_counter["total"] == 0):
            
            sinst_prev = hf_row['Time'] 

            # Get previous instellation
            if (loop_counter["total"] > 0):
                F_inst_prev = S_0
            else:
                F_inst_prev = 0.0

            if (OPTIONS["stellar_heating"] > 0):
                log.info("Updating instellation and radius")

                match OPTIONS['star_model']:
                    case 0:
                        hf_row["R_star"] = mors.Value(OPTIONS["star_mass"],
                                                      hf_row["age_star"]/1e6, 'Rstar') * mors.const.Rsun * 1.0e-2
                        S_0 =  mors.Value(OPTIONS["star_mass"], 
                                          hf_row["age_star"]/1e6, 'Lbol') * L_sun / ( 4. * np.pi * AU * AU * OPTIONS["mean_distance"]**2.0 )
                    case 1:
                        hf_row["R_star"] = baraffe.BaraffeStellarRadius(hf_row["age_star"])
                        S_0 = baraffe.BaraffeSolarConstant(hf_row["age_star"], 
                                                           OPTIONS["mean_distance"])

                # Calculate new eqm temperature
                T_eqm_new = CalculateEqmTemperature(S_0,
                                                 OPTIONS["asf_scalefactor"], 
                                                 OPTIONS["albedo_pl"])
                
            else:
                log.info("Stellar heating is disabled")
                T_eqm_new   = 0.0
                S_0 = 0.0

            hf_row["F_ins"]  =  S_0          # instellation 
            hf_row["T_eqm"]  =  T_eqm_new
            hf_row["T_skin"] =  T_eqm_new * (0.5**0.25) # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)

            log.debug("Instellation change: %+.4e W m-2 (to 4dp)" % abs(S_0 - F_inst_prev))

        # Calculate a new (historical) stellar spectrum 
        if ( ( abs( hf_row["Time"] - sspec_prev ) > OPTIONS['sspec_dt_update'] ) \
            or (loop_counter["total"] == 0) ):
            
            sspec_prev = hf_row["Time"]

            # Remove old spectral file if it exists
            safe_rm(spfile_path)
            safe_rm(spfile_path+"_k")

            log.info("Updating stellar spectrum") 
            match OPTIONS['star_model']: 
                case 0:
                    synthetic = mors.synthesis.CalcScaledSpectrumFromProps(star_struct_modern, 
                                                                           star_props_modern, 
                                                                           hf_row["age_star"]/1e6)
                    fl = synthetic.fl   # at 1 AU
                    wl = synthetic.wl
                case 1:
                    fl = baraffe.BaraffeSpectrumCalc(hf_row["age_star"], 
                                                     OPTIONS["star_luminosity_modern"], 
                                                     modern_fl)
                    wl = modern_wl

            # Scale fluxes from 1 AU to TOA 
            fl *= (1.0 / OPTIONS["mean_distance"])**2.0

            # Save spectrum to file
            header = '# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = %.2e yr'%hf_row["age_star"]
            np.savetxt(os.path.join(dirs["output"],"data","%d.sflux"%hf_row["Time"]), 
                       np.array([wl,fl]).T, 
                       header=header,comments='',fmt="%.8e",delimiter='\t')
    

            # Prepare spectral file for JANUS 
            if OPTIONS["atmosphere_model"] == 0:
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


        ############### ESCAPE

        if (loop_counter["total"] >= loop_counter["init_loops"]) \
            and (OPTIONS["escape_model"] > 0):
            
            PrintHalfSeparator()

            if OPTIONS["escape_model"] == 1:
                esc_result = RunZEPHYRUS()

            elif OPTIONS["escape_model"] == 2:
                esc_result = RunDummyEsc(hf_row, dt, OPTIONS["escape_dummy_rate"])
            
            # store total escape rate 
            hf_row["esc_rate_total"] = esc_result["rate_bulk"]
            log.info("Bulk escape rate: %.2e kg yr-1"%(hf_row["esc_rate_total"] * secs_per_year))

            # update elemental mass targets
            for e in element_list:
                if e=='O': continue

                # store change, for statistics
                esc_m  = esc_result[e+"_kg_total"]
                esc_dm = esc_m - solvevol_target[e]

                # update total elemental inventory
                solvevol_target[e] = esc_m

                # print info to user
                log.debug("    escape %s: m=%.2e kg,  dm=%.2e (%.3f%%)"%
                                    (e, esc_m, esc_dm, 100*esc_dm/esc_m))

                # do not allow negative masses
                solvevol_target[e] = max(0.0, solvevol_target[e])

        ############### / ESCAPE


        ############### OUTGASSING
        PrintHalfSeparator()
        solvevol_inp = copy.deepcopy(OPTIONS)
        solvevol_inp["M_mantle"] =   hf_row["M_mantle"]
        solvevol_inp["T_magma"] =    hf_row["T_magma"]
        solvevol_inp["Phi_global"] = hf_row["Phi_global"]
        solvevol_inp["gravity"]  =   hf_row["gravity"]
        solvevol_inp["mass"]  =      hf_row["M_planet"]
        solvevol_inp["radius"]  =    hf_row["R_planet"]

        #    recalculate mass targets during init phase, since these will be adjusted 
        #    depending on the true melt fraction and T_magma found by SPIDER at runtime.
        if (loop_counter["init"] < loop_counter["init_loops"]):

            # calculate target mass of atoms (except O, which is derived from fO2)
            if OPTIONS["solvevol_use_params"] > 0:
                solvevol_target = get_target_from_params(solvevol_inp)
            else:
                solvevol_target = get_target_from_pressures(solvevol_inp)

            # prevent numerical issues
            for key in solvevol_target.keys():
                if solvevol_target[key] < 1.0e4:
                    solvevol_target[key] = 0.0
        
        #   do calculation
        with warnings.catch_warnings():
            # Suppress warnings from surface_gases solver, since they are triggered when 
            # the model makes a poor guess for the composition. These are then discarded, 
            # so the warning should not propagate anywhere. Errors are still printed.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            solvevol_result = equilibrium_atmosphere(solvevol_target, solvevol_inp)

        #    store results
        for k in solvevol_result.keys():
            if k in hf_row.keys():
                hf_row[k] = solvevol_result[k]

        #    calculate total atmosphere mass 
        hf_row["M_atm"] = 0.0
        for s in volatile_species:
            hf_row["M_atm"] += hf_row[s+"_kg_atm"]
        ############### / OUTGASSING
        

        ############### ATMOSPHERE SUB-LOOP
        PrintHalfSeparator()
        if OPTIONS["shallow_ocean_layer"] == 1:
            hf_row["T_surf"] = ShallowMixedOceanLayer(hf_all.iloc[-1].to_dict(), hf_row)

        if OPTIONS["atmosphere_model"] == 0:
            # Run JANUS: 
            hf_row["T_surf"] = hf_row["T_magma"]
            atm = StructAtm( dirs, hf_row, OPTIONS )
            atm_output = RunJANUS( atm, hf_row["Time"], dirs, OPTIONS, hf_all)

        elif OPTIONS["atmosphere_model"] == 1:
            # Run AGNI 

            # Initialise atmosphere struct
            no_spfile = not os.path.exists(spfile_path)
            no_atm    = bool(atm == None)
            if no_atm or no_spfile:
                log.debug("Initialise new atmosphere struct")

                # first run?
                if no_atm:
                    # surface temperature guess
                    hf_row["T_surf"] = hf_row["T_magma"]
                else:
                    # deallocate old atmosphere 
                    DeallocAtmos(atm)

                # allocate new 
                atm = InitAtmos(dirs, OPTIONS, hf_row)
            
            # Update profile 
            atm = UpdateProfile(atm, hf_row, OPTIONS)

            # Run solver
            atm, atm_output = RunAGNI(atm, loop_counter["total"], dirs, OPTIONS, hf_row)

        elif OPTIONS["atmosphere_model"] == 2:
            # Run dummy atmosphere model 
            atm_output = RunDummyAtm(dirs, OPTIONS, 
                                     hf_row["T_magma"], hf_row["F_ins"], hf_row["R_planet"])
            
        
        # Store atmosphere module output variables
        hf_row["z_obs"]  = atm_output["z_obs"] 
        hf_row["F_atm"]  = atm_output["F_atm"] 
        hf_row["F_olr"]  = atm_output["F_olr"] 
        hf_row["F_sct"]  = atm_output["F_sct"] 
        hf_row["T_surf"] = atm_output["T_surf"]
        hf_row["F_net"]  = hf_row["F_int"] - hf_row["F_atm"]

        # Calculate observables (measured at infinite distance)
        hf_row["transit_depth"] =  (hf_row["z_obs"] / hf_row["R_star"])**2.0
        hf_row["contrast_ratio"] = ((hf_row["F_olr"]+hf_row["F_sct"])/hf_row["F_ins"]) * \
                                     (hf_row["z_obs"] / (OPTIONS["mean_distance"]*AU))**2.0


        ############### HOUSEKEEPING AND CONVERGENCE CHECK 

        PrintHalfSeparator()

        # Update init loop counter
        # Next init iter
        if loop_counter["init"] < loop_counter["init_loops"]: 
            loop_counter["init"]    += 1
            hf_row["Time"]     = 0.
        # Reset restart flag once SPIDER has correct heat flux
        if loop_counter["total"] >= loop_counter["init_loops"]: 
            IC_INTERIOR = 2

        # Adjust total iteration counters
        loop_counter["total"]       += 1

        # Update full helpfile
        if loop_counter["total"]>1:
            # append row
            hf_all = ExtendHelpfile(hf_all, hf_row)
        else:
            # first iter => generate new HF from dict
            hf_all = CreateHelpfileFromDict(hf_row)

        # Write helpfile to disk
        WriteHelpfileToCSV(dirs["output"], hf_all)

        # Print info to terminal and log file
        PrintCurrentState(hf_row)

        # Stop simulation when planet is completely solidified
        if (OPTIONS["solid_stop"] == 1) and (hf_row["Phi_global"] <= OPTIONS["phi_crit"]):
            UpdateStatusfile(dirs, 10)
            log.info("")
            log.info("===> Planet solidified! <===")
            log.info("")
            finished = True

        # Stop simulation when flux is small
        if (OPTIONS["emit_stop"] == 1) and (loop_counter["total"] > loop_counter["init_loops"]+1) \
            and ( abs(hf_row["F_atm"]) <= OPTIONS["F_crit"]):
            UpdateStatusfile(dirs, 14)
            log.info("")
            log.info("===> Planet no longer cooling! <===")
            log.info("")
            finished = True

        # Determine when the simulation enters a steady state
        if (OPTIONS["steady_stop"] == 1) and (loop_counter["total"] > loop_counter["steady_check"]*2+5) and (loop_counter["steady"] == 0):
            # How many iterations to look backwards
            lb1 = -int(loop_counter["steady_check"])
            lb2 = -1

            # Get data
            arr_t = np.array(hf_all["Time"])
            arr_f = np.array(hf_all["F_atm"])
            arr_p = np.array(hf_all["Phi_global"])

            # Time samples
            t1 = arr_t[lb1]; t2 = arr_t[lb2]

            # Flux samples
            flx_m = max(abs(arr_f[lb1]),abs(arr_f[lb2]))

            # Check melt fraction rate
            phi_1 =  arr_p[lb1]
            phi_2 =  arr_p[lb2]
            phi_r = abs(phi_2 - phi_1) / (t2 - t1)
                        
            # Stop when flux is small and melt fraction is unchanging
            if (flx_m < OPTIONS["steady_flux"]) and (phi_r < OPTIONS["steady_dprel"]):
                log.debug("Steady state declared")
                loop_counter["steady"] = 1

        # Steady-state handling
        if loop_counter["steady"] > 0:
            if loop_counter["steady"] <= loop_counter["steady_loops"]:
                loop_counter["steady"] += 1
            else:
                UpdateStatusfile(dirs, 11)
                log.info("")
                log.info("===> Planet entered a steady state! <===")
                log.info("")
                finished = True
        
        # Atmosphere has escaped
        if hf_row["M_atm"] <= OPTIONS["escape_stop"]*hf_all.iloc[0]["M_atm"]:
            UpdateStatusfile(dirs, 15)
            log.info("")
            log.info("===> Atmosphere has escaped! <===")
            log.info("")
            finished = True

        # Maximum time reached
        if (hf_row["Time"] >= OPTIONS["time_target"]):
            UpdateStatusfile(dirs, 13)
            log.info("")
            log.info("===> Target time reached! <===")
            log.info("")
            finished = True

        # Maximum loops reached
        if (loop_counter["total"] > loop_counter["total_loops"]):
            UpdateStatusfile(dirs, 12)
            log.info("")
            log.info("===> Maximum number of iterations reached! <===")
            log.info("")
            finished = True

        # Check if the minimum number of loops have been performed
        if finished and (loop_counter["total"] < loop_counter["total_min"]):
            log.info("Minimum number of iterations not yet attained; continuing...")
            finished = False
            UpdateStatusfile(dirs, 1)

        # Check if keepalive file has been removed - this means that the model should exit ASAP
        if not os.path.exists(keepalive_file):
            UpdateStatusfile(dirs, 25)
            log.info("")
            log.info("===> Model exit was requested! <===")
            log.info("")
            finished=True

        # Make plots if required and go to next iteration
        if (OPTIONS["plot_iterfreq"] > 0) and (loop_counter["total"] % OPTIONS["plot_iterfreq"] == 0) and (not finished):
            PrintHalfSeparator()
            log.info("Making plots")
            UpdatePlots( dirs["output"], OPTIONS )

        ############### / HOUSEKEEPING AND CONVERGENCE CHECK 

    # ----------------------
    # FINAL THINGS BEFORE EXIT
    PrintHalfSeparator()

    # Deallocate atmosphere 
    if OPTIONS["atmosphere_model"] == 1:
        DeallocAtmos(atm)

    # Clean up files
    safe_rm(keepalive_file)

    # Plot conditions at the end
    UpdatePlots( dirs["output"], OPTIONS, end=True)
    end_time = datetime.now()
    log.info("Simulation stopped at: "+end_time.strftime('%Y-%m-%d_%H:%M:%S'))

    run_time = end_time - start_time 
    run_time = run_time.total_seconds()/60 # minutes
    if run_time > 60:
        run_time /= 60 # hours
        log.info("Total runtime: %.2f hours" % run_time )
    else:
        log.info("Total runtime: %.2f minutes" % run_time )

    # EXIT
    return

#====================================================================
if __name__ == '__main__':
    main()
    print("Goodbye")
    exit(0)

# End of file
