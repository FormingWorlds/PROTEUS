#!/usr/bin/env python3

"""
Code for running offline chemistry based on PROTEUS output.
Configuration options are near the bottom of this file.
"""

# Import libraries
import numpy as np
import netCDF4 as nc
import os, glob, shutil, subprocess, time, pathlib
from datetime import datetime
import numpy.random as nrand
import logging
import pandas as pd

# Import PROTEUS stuff
from utils.coupler import *
from utils.plot_offchem import offchem_read_year
from plot.cpl_offchem_species import *
from plot.cpl_offchem_time import *
from plot.cpl_offchem_year import *
from utils.constants import R_sun_cm, AU_cm

# ------------------------------------------------------------------------------------------------

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

# Custom logger instance 
# https://stackoverflow.com/a/61457119
def setup_logger(name:str, logpath:str="new.log", level=logging.INFO, logterm=True)->logging.Logger:

    custom_logger = logging.getLogger(name)    
    custom_logger.handlers.clear()
    
    if os.path.exists(logpath):
        os.remove(logpath)

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    if logterm:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(level)
        custom_logger.addHandler(sh)

    fh = logging.FileHandler(logpath)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    custom_logger.addHandler(fh)

    custom_logger.setLevel(level)
    # sys.excepthook = handle_exception

    return custom_logger 


# Draw samples from all of the years following a distribution
def get_sample_years(years_all,samples,s_centre,s_width):
    years = [ ]
    yfirst = years_all[1]
    ylast = years_all[-1]

    # Draw samples
    if len(years_all) == samples:
        # If number of samples = number of data points, don't need to use sampling method
        years = years_all
    else:
        # Otherwise:
        # Draw other runs randomly from a distribution. This is set-up to sample
        # the end of the run more densely than the start, because the evolution at 
        # the start isn't so interesting compared to the end.

        years.append(ylast) # Include last run

        if samples > 1:
            years.append(yfirst) # Include first run

        sample_itermax = 30000*samples
        sample_iter = 0
        while (len(years) < samples): 

            sample_iter += 1
            if sample_iter > sample_itermax:
                raise Exception("Maximum iterations reached in selecting sample years!")

            # sample = np.abs(nrand.laplace(loc=ylast,scale=float(ylast*swidth)))
            sample = -1.0 * nrand.gumbel(loc=-1.0*s_centre,scale=s_width)
            if (sample <= yfirst) or (sample >= ylast):
                continue  # Try again

            inear = find_nearest_idx(years_all,sample)
            ynear = years_all[inear]
            if ynear in years:
                continue  # Try again
            
            years.append(ynear)

    return sorted(set(years))  # Ensure that there are no duplicates and sort ascending

def run_once(logger:logging.Logger, year:int, screen_name:str, first_run:bool, dirs:dict, COUPLER_options:dict, helpfile_df:pd.DataFrame, ini_method:int) -> bool:
    """Run VULCAN once for a given PROTEUS output year
    
    Runs VULCAN in a screen instance so that lots processes may still be
    managed individually. Does not copy final output to PROTEUS folder, but 
    leaves it in VULCAN/output for the master thread to copy later on.

    Parameters
    ----------
        logger : logging.Logger
            Logger class to use
        year : int
            Planet age corresponding to a data dump from PROTEUS [yr]
        screen_name : str 
            Screen name to use
        first_run : bool
            Is this the first time that VULCAN is run for this configuration?
        dirs : dict
            Dictionary of useful directories
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
    shutil.copyfile(os.path.join(dirs["utils"], "templates", "vulcan_offline_template.py"),
                    os.path.join(dirs["vulcan"],"vulcan_cfg.py"))

    # Make results folder for this run
    this_results = dirs["output"]+"offchem/%d/"%year
    if os.path.exists(this_results):
        raise Exception("Offline chemistry already running for this year (%s)!" % screen_name)
    os.makedirs(this_results)

    # Read atm data
    nc_fpath = dirs["output"]+"/data/%d_atm.nc"%year
    ds = nc.Dataset(nc_fpath)
    tmpl =  np.array(ds.variables["tmpl"][:])
    pl  =   np.array(ds.variables["pl"][:])
    n_gas = np.array(ds.variables["gases"][:])  # gas names, as read
    r_gas = np.array(ds.variables["x_gas"][:])  # gas mixing ratios
    ds.close()

    # Decode gas names
    gases = []
    x_gas = {}
    for i,g_read in enumerate(n_gas):

        g = "".join([c.decode("utf-8") for c in g_read]).strip()

        # if not(g in volatile_species):
        #     raise Exception("Volatile (%s) present in NetCDF is not present in volatile_species list"%g)
        
        x_arr = np.array(r_gas[:,i])
        x_arr = np.clip(x_arr, 0, None)

        if max(x_arr) < 1e-25:  # neglect species with zero mixing ratio.
            continue 
        
        gases.append(g)
        x_gas[g] = x_arr

    p_bot   = np.max(pl) 
    p_top   = np.min(pl) 

    v_mx    = dict()      # Mixing ratios
    for i,g in enumerate(gases):
        val = float(x_gas[g][-1])
        v_mx[g] = val

    # Read helpfile to get data for this year
    helpfile_thisyear = helpfile_df.loc[helpfile_df['Time'] == year].iloc[0]

    # Write specifics to config file
    with open(dirs["vulcan"]+"vulcan_cfg.py",'a') as vcf:
        # Output file to use
        vcf.write("out_name = '%s.vul' \n" % (screen_name))

        # Pressure grid limits
        vcf.write("P_b = %1.5e \n" % float(p_bot * 10.0))  # pressure at the bottom (dyne/cm^2)
        vcf.write("P_t = %1.5e \n" % float(p_top * 10.0))  # pressure at the top (dyne/cm^2)

        # Constant mixing ratios
        const_mix = "{"
        for v in v_mx.keys():
            this_mx = max(float(v_mx[v]),mixing_ratio_floor)     
            const_mix += " '%s' : %1.5e ," % (v,this_mx)
        const_mix = const_mix[:-1]+" }"
        vcf.write("const_mix = %s \n" % str(const_mix))

        # Work out metallicities
        tot = {}
        for e in element_list:
            tot[e] = 0
        for v in v_mx.keys():
            this_mx = max(float(v_mx[v]),mixing_ratio_floor)    
            elems = mol_to_ele(v)
            for e in elems.keys():
                tot[e] += (elems[e]*this_mx)
        for e in tot.keys():
            per = tot[e]/tot['H']
            vcf.write("%s_H = %1.8e \n" % (e,per))

        # Set initial abundances using selected method
        if ini_method == 0: 
            vcf.write("ini_mix = 'const_mix' \n")
        elif (ini_method == 1):
            vcf.write("fastchem_met_scale = 1e-60 \n")  # Try to remove elements not set by method above
            vcf.write("use_solar = False \n")
            vcf.write("ini_mix = 'EQ' \n")
            vcf.write("use_solar = False \n")
        else:
            raise Exception("ini_method = %d is invalid!" % ini_method)

        # Bottom boundary condition
        fix_bb_mr = "{ "
        for v in gases:
            this_mx = max(float(v_mx[v]),mixing_ratio_floor)  # Mixing ratio floor to prevent issues with VULCAN

            if (this_mx > mixing_ratio_fixed):
                fix_bb_mr += " '%s' : %1.5e ," % (v,this_mx)

        fix_bb_mr = fix_bb_mr[:-1]+" }"
        vcf.write("use_fix_sp_bot = %s \n" % str(fix_bb_mr))

        # Background component of atmosphere
        bkgrnd_s = "H2"
        bkgrnd_v = -1
        for v in gases:
            this_mx = float(v_mx[v])
            if (this_mx > bkgrnd_v) and (v in ["N2","H2","CO2","O2"]): # Compare against list of viable background gases
                bkgrnd_s = v
                bkgrnd_v = this_mx

        vcf.write("atm_base = '%s' \n" % bkgrnd_s)

        # PT profile
        vcf.write("atm_file = 'output/%s_PT.txt' \n"%screen_name)

        # Stellar flux (at surface of star)
        vcf.write("sflux_file = 'output/%s_SF.txt' \n"%screen_name)

        # Stellar radius
        vcf.write("r_star = %1.6e \n" %         float(helpfile_thisyear["R_star"]))

        # Other planet parameters
        vcf.write("Rp = %1.6e \n" %             float(COUPLER_options["radius"]*100.0))
        vcf.write("gs = %1.6e \n" %             float(COUPLER_options["gravity"]*100.0))
        vcf.write("sl_angle = %1.6e \n" %       float(COUPLER_options["zenith_angle"]*np.pi/180.0))
        vcf.write("orbit_radius = %1.6e \n" %   float(COUPLER_options["mean_distance"]))

        vcf.flush()
        os.fsync(vcf.fileno())
    os.sync()
    
    # Also copy config file to results dir for posterity
    shutil.copyfile(dirs["vulcan"]+"vulcan_cfg.py",this_results+"vulcan_cfg.py")

    # Copy a reasonable stellar flux file to the location where VULCAN will 
    #   read it. This is scaled to the stellar surface
    ls = glob.glob(dirs["output"]+"/data/*.sflux")
    years = [int(f.split("/")[-1].split(".")[0]) for f in ls]
    sflux_near = dirs["output"]+"/data/%d.sflux"%years[find_nearest_idx(years,year)]

    sflux_read = np.loadtxt(sflux_near, skiprows=1).T
    f_sf = float(helpfile_thisyear["R_star"]) * R_sun_cm / (COUPLER_options["mean_distance"] * AU_cm)
    sflux_scaled = sflux_read[1] * f_sf * f_sf       # scale to surface of star 

    sflux_write = [sflux_read[0], sflux_scaled]
    np.savetxt(dirs["vulcan"]+"output/%s_SF.txt"%screen_name, np.array(sflux_write).T, header="Stellar spectrum at R_star")

    # Write PT profile
    minT = 120.0
    tmpl = np.clip(tmpl,minT,None)
    if np.any(np.array(tmpl) < minT):
        logger.warn("Temperature is unreasonably low (< %f)!" % minT)
    vul_PT = np.array(
        [np.array(pl)  [::-1] * 10.0,
         np.array(tmpl)[::-1]
        ]
    ).T
    header = "#(dyne/cm2)\t (K) \n Pressure\t Temp"
    np.savetxt(     dirs["vulcan"]+"output/%s_PT.txt"%screen_name, vul_PT,  delimiter="\t",header=header,comments='',fmt="%1.5e")
    shutil.copyfile(dirs["vulcan"]+"output/%s_PT.txt"%screen_name, dirs["output"]+"offchem/%d/PT.txt"%year)

    # Run vulcan
    # Note that the screen name ends with an 'x' so that accurate matching can be done with grep
    os.chdir(dirs["vulcan"])
    logfile = dirs["output"]+"offchem/%d/vulcan.log"%year
    vulcan_flag = " " if first_run else "-n"
    vulcan_run = "screen -dmLS %s python3 vulcan.py %s" % (screen_name,vulcan_flag)
    subprocess.run([vulcan_run], shell=True, check=True)
    os.chdir(dirs["coupler"])

    return success


def parent(cfgfile, samples, threads, s_width, s_centre, 
           mkfuncs=True, runtime_sleep=30, ini_method=1, mkplots=True, logterm=True):
    """Parent process for handing offline chemistry.

    Reads configuration from cfgfile, finds the files, takes samples, and dispatches 
    each sample as required. Allows processes to be run in-parallel inside 
    screen sessions. Multiple instances of this script should not be run at 
    the same time because of inflexibilities in VULCAN.
    
    Tries to take useful samples across the simulation time by using a 
    distribution with a peak and width, which allows regions of interest to be 
    better resolved. You might need to play with s_width and s_centre in order
    to sample the part of the planet's evolution that you're interested in.

    Parameters
    ----------
        cfgfile : str
            Path to config file used for PROTEUS as stored in the output 
            subdirectory for a completed run.
        samples : int
            Number of samples to use from output dir (set to -1 if all are 
            requested, but beware of long runtimes in this case).
        threads : int
            Maximum number of threads to use when running offline chemistry.
        s_width : float
            Width of sampling function [years]
        s_centre : float
            Centre of sampling function [years]

        mkfuncs : bool
            Compile VULCAN reaction functions from network on first run?
        runtime_sleep : float
            Wall clock time to sleep between dispatches of VULCAN [seconds]
        ini_method : int
            Method to use to intialise abundances in VULCAN.
            0: constant mixing ratios based on SPIDER's outgassing products,
            1: calculate elemental abundances and run FastChem eqm chemistry.
        mkplots : bool
            Make plots at end?
        logterm : bool
            Log output to terminal? Will always log to a file.

    """

    # Read in PROTEUS config file
    COUPLER_options, _ = ReadInitFile( cfgfile )

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
    logger_name = "logger_%d" % nrand.randint(low=1e9, high=9e9)
    log_file = offchem_dir+"parent.log"
    logger = setup_logger(logger_name,logpath=log_file,logterm=logterm)

    # Log info to user
    time_start = datetime.now()
    logger.info("Date: " +str(time_start.strftime("%Y-%m-%d")))

    now = int(str(time_start.strftime("%H%M%S")))
    logger.info("Time: %d"%now)

    logger.info(" ")
    logger.info("This program will generate several screen sessions")
    logger.info("To kill all of them, run `pkill -f %d_offchem_`" % now)   # (from: https://stackoverflow.com/a/8987063)
    logger.info("Take care to avoid orphaned instances by using `screen -ls`")
    logger.info(" ")

    # Read helpfile
    helpfile_df = pd.read_csv(dirs["output"]+"runtime_helpfile.csv",sep=r'\s+')

    # Find out which years we have both SPIDER and JANUS data for
    evolution_json = glob.glob(dirs["output"]+"/data/*.json")
    json_years = np.array([int(f.split("/")[-1].split(".")[0]) for f in evolution_json])
    json_years = np.sort(json_years)
    logger.info("Number of json files: %d"%len(json_years))

    evolution_nc = glob.glob(dirs["output"]+"/data/*_atm.nc")
    nc_years = np.array([int(f.split("/")[-1].split("_atm.")[0]) for f in evolution_nc])
    nc_years = np.sort(nc_years)
    logger.info("Number of nc files: %d"%len(nc_years))

    years_all = []
    for y in json_years:
        if y in nc_years: 
            years_all.append(y)

    # All requested
    if samples == -1:
        samples = len(years_all)

    threads = min(threads, samples)
    max_threads = 1000
    if samples < 1:
        raise Exception("Too few samples requested! (Less than zero)") 
    if samples > len(years_all):
        raise Exception("Too many samples requested! (Duplicates expected)") 
    if threads < 1:
        raise Exception("Too few threads requested! (Less than one)")
    if threads > max_threads:
        raise Exception("Too many threads requested! (More than %d)" % max_threads)
    
    # Select samples...
    logger.info("Choosing sample years... (May take a while in some cases)")
    years = get_sample_years(years_all, samples, s_centre, s_width)
    samples = len(years)
    logger.info("Years selected: " +str(years))

    # Start processes
    logger.info(" ")
    logger.info("Starting process manager (%d samples, %d threads) in..." % (samples,threads))
    for w in range(3,0,-1):
        logger.info("\t %d seconds" % w)
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

            sname = "%d_offchem_%d" % (now,y)
            running = bool(str("%sx"%sname) in screen_out)
            
            if running:
                count_threads += 1
            
            # Currently tagged as running, but check if done
            if (not running) and (status[i] == 1):
                to_copy = dirs["vulcan"]+"/output/%s.vul" % sname
                if os.path.exists(to_copy): # Is done?
                    status[i] = 2
                    shutil.copyfile(to_copy, dirs["output"]+"/offchem/%d/output.vul"%y) # Copy output

                else:
                    status[i] = 3
                    logger.info("WARNING: Output file missing for year = %d" % y)

        # How many are queued?
        count_queued = np.count_nonzero(status == 0)

        # How many are running?
        count_current = np.count_nonzero(status == 1)

        # Check how many have finished/exited
        count_completed = np.count_nonzero(status > 1)

        # Check how many have been dispatched so far
        count_dispatched = np.count_nonzero(status > 0)

        # Print status
        logger.info("Status: %d queued (%.1f%%), %d running (%.1f%%), %d completed (%.1f%%), %d threads active" % (
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
                
                # Wait for VULCAN to finish making the network py file
                if mkfuncs and (count_dispatched == 1):
                    time.sleep(60.0)  

                # Run new process
                logger.info("Dispatching job %03d: %08d yrs..." % (i,y))
                fr = bool( (count_dispatched == 0) and mkfuncs)
                sname = "%d_offchem_%d" % (now,y)
                this_success = run_once(logger, years[i], sname, fr, dirs, COUPLER_options, helpfile_df, ini_method)
                status[i] = 1
                if this_success:
                    logger.info("\t (dispatched)")
                else:
                    logger.info("\t (failed)")


        # Check if all are done
        done = (status > 1).all()
        
    # Check if exited early
    if not done:
        logger.info("WARNING: Master process loop terminated early!")
        logger.info("         This could be because it timed-out or an error occurred.")

    # Tidy VULCAN output folder
    for f in glob.glob(dirs["vulcan"]+"output/%d_offchem_*"%now): 
        os.remove(f)

    time_end = datetime.now()
    logger.info("All processes finished at: "+str(time_end.strftime("%Y-%m-%d %H%M%S")))
    logger.info("Total runtime: %.1f hours "%((time_end-time_start).total_seconds()/3600.0))

    # Plot
    if mkplots:
        logger.info("Plotting results...")

        # plot_janus = bool(ini_method == 0)
        plot_janus = True

        species = ["H2", "H2O", "H", "OH", "CO2", "CO", "CH4", "HCN", "NH3", "N2", "NO"]
        logger.info("\t timeline")
        plot_offchem_time(dirs["output"],species,plot_init_mx=plot_janus,tmin=1e4)

        ls = glob.glob(dirs["output"]+"offchem/*/output.vul")
        years = [int(f.split("/")[-2]) for f in ls]
        years_data = [offchem_read_year(dirs["output"],y,read_const=plot_janus) for y in years]

        if len(years) > 0:
            for yd in years_data:
                logger.info("\t year = %d" % yd["year"])
                plot_offchem_year(dirs["output"],yd,species,plot_init_mx=plot_janus)
        else:
            logger.error('In attempting to make plots, no VULCAN output files were found')

        for s in species:
            logger.info("\t species = %s" % s)
            plot_offchem_species(dirs["output"],s,plot_init_mx=plot_janus)

    # Done
    logger.info("Done running offline chemistry!")


# If run directly
if __name__ == '__main__':

    # Parameters
    cfgfile =       "output/agni_mixed_hot/init_coupler.cfg"  # Config file used for PROTEUS
    samples =       4                  # How many samples to use from output dir (set to -1 if all are requested)
    threads =       4                  # How many threads to use
    mkfuncs =       True                # Compile reaction functions again?
    mkplots =       True                # make plots?
    s_width =       1e7                 # Scale width of sampling distribution [yr]
    s_centre =      8e5                 # Centre of sampling distribution [yr]
    runtime_sleep = 40                  # Sleep seconds per iter (required for VULCAN warm up periods to pass)
    ini_method =    1                   # Method used to init VULCAN abundances  (0: const_mix, 1: eqm)

    # Do it
    parent(cfgfile, samples, threads, s_width, s_centre, 
           mkfuncs=mkfuncs, runtime_sleep=runtime_sleep, ini_method=ini_method, mkplots=mkplots)
    
