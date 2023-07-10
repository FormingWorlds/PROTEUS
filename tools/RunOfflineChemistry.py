#!/usr/bin/env python3

"""
Code for running offline chemistry based on PROTEUS output
"""

# Import libraries
import numpy as np
import pickle as pkl
import os, glob, shutil, subprocess, time, pathlib
from datetime import datetime
import numpy.random as nrand
import logging
import multiprocessing as mp
import pandas as pd

# Import PROTEUS stuff
from utils.coupler import *
from utils.plot import offchem_read_year
from plot.cpl_offchem_species import *
from plot.cpl_offchem_time import *
from plot.cpl_offchem_year import *

# ------------------------------------------------------------------------------------------------

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

def run_once(year:int, now:int, first_run:bool, COUPLER_options:dict, helpfile_df:pd.DataFrame, ini_method:int) -> bool:
    """Run VULCAN once for a given PROTEUS output year
    
    Runs VULCAN in a screen instance so that lots processes may still be
    managed individually. Does not copy final output to PROTEUS folder, but 
    leaves it in VULCAN/output for the master thread to copy later on.

    Parameters
    ----------
        year : int
            Planet age corresponding to a data dump from PROTEUS [yr]
        now : int
            Time at which this run was started (format: HHMMSS)
        first_run : bool
            Is this the first time that VULCAN is run for this configuration?
        COUPLER_options : dict
            PROTEUS options dictionary read from cfg file
        helpfile_df : DataFrame
            Helpfile contents loaded as a Pandas DataFrame object
        ini_method : int
            Method used to set abundances (0: const_mix homogeneous, 1: eqm chemistry with derived metallicities)
    Returns
    ----------
        success : bool
            Did everything dispatch okay?
    """

    success = True                  # Flag for if dispatch worked
    mixing_ratio_floor = 1e-50      # Minimum mixing ratio (at all)
    mixing_ratio_fixed = 1e-0       # Species with mixing ratio greater than this value have fixed abundances at the surface

    # Copy template config file
    shutil.copyfile(dirs["utils"]+"/vulcan_offline_template.py",dirs["vulcan"]+"vulcan_cfg.py")

    # Make results folder for this run
    this_results = dirs["output"]+"offchem/%d/"%year
    if os.path.exists(this_results):
        raise Exception("Offline chemistry already running for this year!")
    os.makedirs(this_results)

    # Read atm object
    with open(dirs["output"]+"%d_atm.pkl"%year,'rb') as atm_handle:
        atm = pkl.load(atm_handle)
        p_bot   = atm.pl[-1]        # Pressure
        p_top   = atm.pl[0]         # Pressure
        v_mx    = atm.vol_list      # Mixing ratios
        
        volatile_species = v_mx.keys()

    # Read helpfile to get data for this year
    helpfile_thisyear = helpfile_df.loc[helpfile_df['Time'] == year].iloc[0]

    # Write specifics to config file
    with open(dirs["vulcan"]+"vulcan_cfg.py",'a') as vcf:
        # Output file to use
        vcf.write("out_name = '%d_offchem_%d.vul' \n" % (now,year))

        # Pressure grid limits
        vcf.write("P_b = %1.5e \n" % float(p_bot * 10.0))  # pressure at the bottom (dyne/cm^2)
        vcf.write("P_t = %1.5e \n" % float(p_top * 10.0))  # pressure at the top (dyne/cm^2)

        # Set initial abundances using selected method
        # Constant mixing ratios
        if ini_method == 0: 
            const_mix = "{"
            for v in volatile_species:
                this_mx = max(float(v_mx[v]),mixing_ratio_floor)     
                const_mix += " '%s' : %1.5e ," % (v,this_mx)
            const_mix = const_mix[:-1]+" }"
            vcf.write("const_mix = %s \n" % str(const_mix))
            vcf.write("ini_mix = 'const_mix' \n")

        # Eqm chemistry
        elif (ini_method == 1):

            # Work out metallicities
            tot = {}
            for e in element_list:
                tot[e] = 0
            for v in volatile_species:
                this_mx = max(float(v_mx[v]),mixing_ratio_floor)    
                elems = mol_to_ele(v)
                for e in elems.keys():
                    tot[e] += (elems[e]*this_mx)
            for e in tot.keys():
                per = tot[e]/tot['H']
                vcf.write("%s_H = %1.8e \n" % (e,per))

            vcf.write("fastchem_met_scale = 1e-60 \n")  # Try to remove elements not set by method above
            vcf.write("use_solar = False \n")
            vcf.write("ini_mix = 'EQ' \n")
            vcf.write("use_solar = False \n")
        else:
            raise Exception("ini_method = %d is invalid!" % ini_method)

        # Bottom boundary condition
        fix_bb_mr = "{ "
        for v in volatile_species:

            this_mx = max(float(v_mx[v]),mixing_ratio_floor)  # Mixing ratio floor to prevent issues with VULCAN

            if (this_mx > mixing_ratio_fixed):
                fix_bb_mr += " '%s' : %1.5e ," % (v,this_mx)

        fix_bb_mr = fix_bb_mr[:-1]+" }"
        vcf.write("use_fix_sp_bot = %s \n" % str(fix_bb_mr))

        # Background component of atmosphere
        bkgrnd_s = "H2"
        bkgrnd_v = -1
        for v in volatile_species:
            this_mx = float(v_mx[v])
            if (this_mx > bkgrnd_v) and (v in ["N2","H2","CO2","O2"]): # Compare against list of viable background gases
                bkgrnd_s = v
                bkgrnd_v = this_mx

        vcf.write("atm_base = '%s' \n" % bkgrnd_s)

        # PT profile
        vcf.write("atm_file = 'output/%d_offchem_%d_PT.txt' \n"%(now,year))

        # Stellar flux (at surface of star)
        vcf.write("sflux_file = 'output/%d_offchem_%d_SF.txt' \n"%(now,year))

        # Stellar radius
        vcf.write("r_star = %1.6e \n" %         float(helpfile_thisyear["R_star"]))

        # Other planet parameters
        vcf.write("Rp = %1.6e \n" %             float(COUPLER_options["radius"]*100.0))
        vcf.write("gs = %1.6e \n" %             float(COUPLER_options["gravity"]*100.0))
        vcf.write("sl_angle = %1.6e \n" %       float(COUPLER_options["zenith_angle"]*np.pi/180.0))
        vcf.write("orbit_radius = %1.6e \n" %   float(COUPLER_options["mean_distance"]))

    
    # Also copy config file to results dir for posterity
    shutil.copyfile(dirs["vulcan"]+"vulcan_cfg.py",this_results+"vulcan_cfg.py")

    # Copy a reasonable stellar flux file to the location where VULCAN will read it
    ls = glob.glob(dirs["output"]+"*.sfluxsurf")
    years = [int(f.split("/")[-1].split(".")[0]) for f in ls]
    year_near = years[find_nearest_idx(years,year)]
    shutil.copyfile(dirs["output"]+"%d.sfluxsurf"%year_near,dirs["vulcan"]+"output/%d_offchem_%d_SF.txt"%(now,year))

    # Write PT profile
    vul_PT = np.array(
        [np.array(atm.pl)  [::-1] * 10.0,
         np.array(atm.tmpl)[::-1]
        ]
    ).T
    header = "#(dyne/cm2)\t (K) \n Pressure\t Temp"
    np.savetxt(dirs["vulcan"]+"output/%d_offchem_%d_PT.txt"%(now,year),vul_PT,  delimiter="\t",header=header,comments='',fmt="%1.5e")
    shutil.copyfile(dirs["vulcan"]+"output/%d_offchem_%d_PT.txt"%(now,year),    dirs["output"]+"offchem/%d/PT.txt"%year)

    # Run vulcan
    # Note that the screen name ends with an 'x' so that accurate matching can be done with grep
    os.chdir(dirs["vulcan"])
    screen_name = "%d_offchem_%dx" % (now,year)
    logfile = dirs["output"]+"offchem/%d/vulcan.log"%year
    vulcan_flag = " " if first_run else "-n"
    vulcan_run = 'screen -S %s -L -Logfile %s -dm bash -c "python3 vulcan.py %s"' % (screen_name,logfile,vulcan_flag)
    subprocess.run([vulcan_run], shell=True, check=True)
    os.chdir(dirs["coupler"])

    return success


# Handle exceptions with logging library, so that they are written to the log file
# https://stackoverflow.com/a/16993115
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == '__main__':
    print("Started main process")

    # Parameters
    cfgfile =       "output/trap1b/init_coupler.cfg"  # Config file used for PROTEUS
    samples =       -1                  # How many samples to use from /output/ (set to -1 if all are requested)
    threads =       45                  # How many threads to use
    mkfuncs =       False               # Compile reaction functions again?
    s_width =       1e5                 # Width of sampling distribution [yr]
    s_centre =      3e4                 # Centre of sampling distribution [yr]
    runtime_sleep = 30                  # Sleep seconds per iter
    ini_method =    1                   # Method used to init VULCAN abundances

    # Read in PROTEUS config file
    COUPLER_options, time_dict = ReadInitFile( cfgfile )

    # Set directories
    dirs = SetDirectories(COUPLER_options)
    offchem_dir = dirs["output"]+"offchem/"

    # Delete old files
    if os.path.exists(offchem_dir):
        shutil.rmtree(offchem_dir)
    pathlib.Path(offchem_dir).mkdir()
    if os.path.isdir(offchem_dir):
        if os.listdir(offchem_dir):
            raise Exception("Could not clear directory for new run")
    else:
        raise Exception("Could not make directory '%s'" % offchem_dir)

    # Copy cfg file for posterity
    shutil.copy2(cfgfile, offchem_dir)

    # Set up logging
    logfile = offchem_dir+"parent.log"
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler()
        ]
    )
    sys.excepthook = handle_exception

    # Log info to user
    time_start = datetime.now()
    logging.info("Date: " +str(time_start.strftime("%Y-%m-%d")))

    now = int(str(time_start.strftime("%H%M%S")))
    logging.info("Time: %d"%now)

    logging.info(" ")
    logging.info("This program will generate several screen sessions")
    logging.info("To kill all of them, run `pkill -f %d_offchem_`" % now)   # (from: https://stackoverflow.com/a/8987063)
    logging.info("Take care to avoid orphaned instances by using `screen -ls`")
    logging.info(" ")

    # Read helpfile
    helpfile_df = pd.read_csv(dirs["output"]+"runtime_helpfile.csv",sep='\t')

    # Find out which years we have both SPIDER and AEOLUS data for
    evolution_json = glob.glob(dirs["output"]+"*.json")
    json_years = np.array([int(f.split("/")[-1].split(".")[0]) for f in evolution_json])
    json_years = np.sort(json_years)
    logging.info("Number of json files: %d"%len(json_years))

    evolution_pkl = glob.glob(dirs["output"]+"*_atm.pkl")
    pkl_years = np.array([int(f.split("/")[-1].split("_atm.")[0]) for f in evolution_pkl])
    pkl_years = np.sort(pkl_years)
    logging.info("Number of pkl files: %d"%len(pkl_years))

    years_all = []
    for y in json_years:
        if y in pkl_years: 
            years_all.append(y)

    # All requested
    if samples == -1:
        samples = len(years_all)

    # Select samples...
    if samples < 1:
        raise Exception("Too few samples requested! (Less than zero)") 
    if samples > len(years_all):
        raise Exception("Too many samples requested! (Duplicates expected)") 
    if threads < 1:
        raise Exception("Too few threads requested! (Less than one)")
    if threads > 62:
        raise Exception("Too many threads requested! (Count is capped at 40)")
    
    logging.info("Choosing sample years... (May take a while in some cases)")
    years = [ ]
    yfirst = years_all[0]
    ylast = years_all[-1]
    if samples >= 1:
        years.append(yfirst) # Include first run
    if samples >= 2:
        years.append(ylast) # Include last run

    # Draw samples
    if len(years_all) == samples:
        # If number of samples = number of data points, don't need to use sampling method
        years = years_all
    else:
        # Otherwise:
        # Draw other runs randomly from a distribution. This is set-up to sample
        # the end of the run more densely than the start, because the evolution at 
        # the start isn't so interesting compared to the end.
        sample_itermax = 30000*samples
        sample_iter = 0
        while (len(years) < samples): 

            sample_iter += 1
            if sample_iter > sample_itermax:
                raise Exception("Maximum iterations reached in selecting sample years!")

            # sample = np.abs(nrand.laplace(loc=ylast,scale=float(ylast*swidth)))
            sample = nrand.gumbel(loc=s_centre,scale=s_width)
            if (sample <= yfirst) or (sample >= ylast):
                continue  # Try again

            inear = find_nearest_idx(years_all,sample)
            ynear = years_all[inear]
            if ynear in years:
                continue  # Try again
            
            years.append(ynear)


    years = sorted(set(years))  # Ensure that there are no duplicates and sort ascending
    samples = len(years)
    logging.info("Years selected: " +str(years))

    # Start processes
    logging.info(" ")
    logging.info("Starting process manager (%d samples, %d threads) in..." % (samples,threads))
    for w in range(5,0,-1):
        logging.info("\t %d seconds" % w)
        time.sleep(1.0)

    # Track statuses
    # 0: queued
    # 1: running
    # 2: done
    # 3: done (no output)
    status = np.zeros((samples))
    done = False                # All done?

    # Dispatch and track processes
    runtime_itermax = int(1e6)  # Max iters
    runtime_iter = 0            # Current iters
    while (not done) and (runtime_iter < runtime_itermax):

        # Sleep if not first
        if runtime_iter > 0:
            time.sleep(runtime_sleep)  # Wait a while before checking again

        # Update iter counter
        runtime_iter += 1

        # Update statuses
        screen_ls = subprocess.run(['screen','-ls'], stdout=subprocess.PIPE)
        screen_out = str(screen_ls.stdout.decode("utf-8"))
        count_threads = 0
        for i in range(samples):
            y = years[i]
            running = bool(str("%d_offchem_%dx"%(now,y)) in screen_out)
            
            if running:
                count_threads += 1
            
            # Currently tagged as running, but check if done
            if (not running) and (status[i] == 1):
                to_copy = dirs["vulcan"]+"/output/%d_offchem_%d.vul" % (now,y)
                if os.path.exists(to_copy):
                    status[i] = 2
                    shutil.copyfile(to_copy, dirs["output"]+"/offchem/%d/output.vul"%y) # Copy output
                    for f in glob.glob(dirs["vulcan"]+"output/%d_offchem_%d*"%(now,y)):  # Tidy VULCAN output folder
                        os.remove(f)
                else:
                    status[i] = 3
                    logging.info("WARNING: Output file missing for year = %d" % y)

        # How many are queued?
        count_queued = np.count_nonzero(status == 0)

        # How many are running?
        count_current = np.count_nonzero(status == 1)

        # Check how many have finished/exited
        count_completed = np.count_nonzero(status > 1)

        # Check how many have been dispatched so far
        count_dispatched = np.count_nonzero(status > 0)

        # Print status
        logging.info("Status: %d queued (%.1f%%), %d running (%.1f%%), %d completed (%.1f%%), %d threads active" % (
                    count_queued, 100.0*count_queued/samples,
                    count_current, 100.0*count_current/samples,
                    count_completed, 100.0*count_completed/samples,
                    count_threads)
                    )

        # If there's space, dispatch a queued process
        if count_current < threads:

            # Find a queued process
            y = -1
            for i in range(samples):
                if status[i]==0:
                    y = years[i]
                    break
            
            # Did we find one?
            if (y > -1):
                
                # Wait for chem_funs.py to finish making the network py file
                if mkfuncs and (count_dispatched == 1):
                    time.sleep(60.0)  

                # Run new process
                logging.info("Dispatching job %03d: %08d yrs..." % (i,y))
                fr = bool( (count_dispatched == 0) and mkfuncs)
                this_success = run_once(years[i], now, fr,COUPLER_options,helpfile_df,ini_method)
                status[i] = 1
                if this_success:
                    logging.info("\t (succeeded)")
                else:
                    logging.info("\t (failed)")


        # Check if all are done
        done = (status > 1).all()
        
    # Check if exited early
    if not done:
        logging.info("WARNING: Master process loop terminated early!")
        logging.info("         This could be because it timed-out or an error occurred.")

    time_end = datetime.now()
    logging.info("All processes finished at: "+str(time_end.strftime("%Y-%m-%d %H%M%S")))
    logging.info("Total runtime: %.1f hours "%((time_end-time_start).total_seconds()/3600.0))

    # Plot
    logging.info("Plotting results...")

    plot_aeolus = bool(ini_method == 0)

    species = ["H2", "H2O", "H", "OH", "CO2", "CO", "CH4", "HCN", "NH3", "N2", "NO","C2H6","CH3OH"]
    logging.info("\t timeline")
    plot_offchem_time(dirs["output"],species,plot_init_mx=plot_aeolus,tmin=1e3)

    ls = glob.glob(dirs["output"]+"offchem/*/output.vul")
    years = [int(f.split("/")[-2]) for f in ls]
    years_data = [offchem_read_year(dirs["output"],y,read_const=plot_aeolus) for y in years]

    if len(years) == 0:
        raise Exception('In attempting to make plots, no VULCAN output files were found')

    for yd in years_data:
        logging.info("\t year = %d" % yd["year"])
        plot_offchem_year(dirs["output"],yd,species,plot_init_mx=plot_aeolus)

    for s in species:
        print("\t species = %s" % s)
        plot_offchem_species(dirs["output"],s,plot_init_mx=plot_aeolus,tmin=1e3)

    # Done
    logging.info("Done!")
