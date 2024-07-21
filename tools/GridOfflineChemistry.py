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
from plot.cpl_offchem_species import *
from plot.cpl_offchem_time import *
from plot.cpl_offchem_year import *
import tools.RunOfflineChemistry as ROC

# ------------------------------------------------------------------------------------------------


def parent(folder, samples, threads, s_width, s_centre, 
           mkfuncs=True, runtime_sleep=30, ini_method=1, mkplots=True, logterm=True):
    """Parent process for handing a grid of offline chemistry runs.

    Runs offline chemistry for each point in the GridPROTEUS output, searching
    and sampling with each folder. Allows processes to be run in-parallel inside 
    screen sessions. Multiple instances of this script should NOT be run at 
    the same time because of inflexibilities in VULCAN.

    Parameters
    ----------
        folder : str
            Path to base folder containing all of the GridPROTEUS runs.
        samples : int
            Number of samples to use for each run.
        threads : int
            Maximum total number of threads to use, across all grid points.
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
        logterm : bool
            Log output to terminal? Will always log to a file.
    """

    folder = os.path.abspath(folder) + "/"

    # Set up logging
    logger_name = "logger_%d" % nrand.randint(low=1e9, high=9e9)
    log_file = folder+"parent.log"
    logger = ROC.setup_logger(logger_name,logpath=log_file,logterm=logterm)

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

    # We need to reconstruct the grid points in memory
    # These are 2D arrays, the first dimension being each grid point
    # and the second dimension being the samples within that grid point

    grid_opts = []  # COUPLER_options
    grid_hfdf = []  # Helpfile dataframe
    grid_cfgf = []  # Config file
    grid_dirs = []  # Directories dict
    grid_syrs = []  # Sample years
    grid_stat = []  # Status of each run 

    gpoints = 0 # number of grid points

    grid_folders = glob.glob(folder+"case_*")
    
    for gi,gf in enumerate(grid_folders):

        logger.info("Reading gridpoint %d" % gi)

        # Read in PROTEUS config file
        cfgfile = gf+"/init_coupler.cfg"
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

        shutil.copy2(cfgfile, offchem_dir)

        # Read helpfile
        helpfile_df = pd.read_csv(dirs["output"]+"runtime_helpfile.csv",sep=r"\s+")

        # Find out which years we have both SPIDER and JANUS data for
        evolution_json = glob.glob(dirs["output"]+"/data/*.json")
        json_years = np.array([int(f.split("/")[-1].split(".")[0]) for f in evolution_json])
        json_years = np.sort(json_years)
        logger.info("    number of json files: %d"%len(json_years))

        evolution_nc = glob.glob(dirs["output"]+"/data/*_atm.nc")
        nc_years = np.array([int(f.split("/")[-1].split("_atm.")[0]) for f in evolution_nc])
        nc_years = np.sort(nc_years)
        logger.info("    number of nc files: %d"%len(nc_years))

        years_all = []
        for y in json_years:
            if y in nc_years: 
                years_all.append(y)

        # All requested
        if samples == -1:
            samples = len(years_all)

        if samples < 1:
            raise Exception("Too few samples requested! (Less than zero)") 
        if samples > len(years_all):
            raise Exception("Too many samples requested! (Duplicates expected)") 

        # Select samples...
        years = ROC.get_sample_years(years_all, samples, s_centre, s_width)
        samples = len(years)
        logger.info("    years selected: " +str(years))

        # Save to grid-in-memory
        gpoints += 1
        grid_opts.append( [ COUPLER_options ] * samples  ) 
        grid_hfdf.append( [ helpfile_df     ] * samples  )
        grid_cfgf.append( [ cfgfile         ] * samples  )
        grid_dirs.append( [ dirs            ] * samples  )
        grid_syrs.append( list(years)  )
        grid_stat.append( [ 0               ] * samples  )

    tot_runs = samples*gpoints
    threads = min(threads, samples*tot_runs)
    max_threads = 1000
    if threads > max_threads:
        raise Exception("Too many threads requested! (More than %d)" % max_threads)

    # Convert to numpy arrays
    grid_opts = np.array(grid_opts, dtype=dict)
    grid_hfdf = np.array(grid_hfdf, dtype=pd.DataFrame)
    grid_cfgf = np.array(grid_cfgf, dtype=str)
    grid_dirs = np.array(grid_dirs, dtype=dict)
    grid_syrs = np.array(grid_syrs, dtype=int)
    grid_stat = np.array(grid_stat, dtype=int)

    # Start processes
    logger.info(" ")
    logger.info("Starting process manager (%d grid points, %d samples, %d threads) in..." % (gpoints, samples, threads))
    for w in range(5,0,-1):
        logger.info("\t %d seconds" % w)
        time.sleep(1.0)

    # Track statuses
    # 0: queued
    # 1: running
    # 2: done
    # 3: done (no output)
    done = False                # All done?

    # Dispatch and track processes
    runtime_iter = 0            # Current iters
    while not done:

        # Sleep if not first
        if runtime_iter > 0:
            time.sleep(runtime_sleep)  # Wait a while before checking again

        # Update iter counter
        runtime_iter += 1

        # Update statuses
        screen_ls = subprocess.run(['screen','-ls'], stdout=subprocess.PIPE)
        screen_out = str(screen_ls.stdout.decode("utf-8"))
        count_threads = 0

        for gi in range(gpoints):
            for i,y in enumerate(grid_syrs[gi]):
                
                sname = "%d_offchem_%d_%d" % (now,gi,y)

                running = bool(str(sname) in screen_out)
                
                if running:
                    count_threads += 1

                # Currently tagged as running, but check if done
                if (not running) and (grid_stat[gi][i] == 1):

                    dirs = grid_dirs[gi][i]
                    to_copy = dirs["vulcan"]+"/output/%s.vul" % sname
                    if os.path.exists(to_copy): # Is done?
                        grid_stat[gi][i] = 2
                        shutil.copyfile(to_copy, dirs["output"]+"/offchem/%d/output.vul"%y) # Copy output

                    else:
                        grid_stat[gi][i] = 3
                        logger.info("WARNING: Output file missing for year = %d" % y)

        # How many are queued?
        count_queued = np.count_nonzero(grid_stat == 0)

        # How many are running?
        count_current = np.count_nonzero(grid_stat == 1)

        # Check how many have finished/exited
        count_completed = np.count_nonzero(grid_stat > 1)

        # Check how many have been dispatched so far
        count_dispatched = np.count_nonzero(grid_stat > 0)

        # Print status
        logger.info("Status: %d queued (%.1f%%), %d running (%.1f%%), %d completed (%.1f%%), %d threads active" % (
                    count_queued,    100.0*count_queued/tot_runs,
                    count_current,   100.0*count_current/tot_runs,
                    count_completed, 100.0*count_completed/tot_runs,
                    count_threads)
                    )

        # If there's space, dispatch a queued process
        if count_current < threads:

            # Find a queued process
            use_y =  -1
            use_gi = -1
            use_i =  -1
            breakout = False
            for gi in range(gpoints):
                for i,y in enumerate(grid_syrs[gi]):
                    if grid_stat[gi][i]==0:
                        use_y = y
                        use_gi = gi
                        use_i = i
                        breakout = True
                        break
                if breakout:
                    break

            y  = use_y 
            gi = use_gi
            i  = use_i 
            
            # Did we find one?
            if (y > -1):
                
                # Wait for VULCAN to finish making the network py file
                if mkfuncs and (count_dispatched == 1):
                    time.sleep(60.0)  

                grid_stat[gi][i] = 1

                # Run new process
                logger.info("Dispatching job %03d,%03d: %08d yrs..." % (gi,i,y))
                fr = bool( (count_dispatched == 0) and mkfuncs)
                sname = "%d_offchem_%d_%d" % (now,gi,y)
                this_success = ROC.run_once(logger, y, sname, fr, grid_dirs[gi][i], grid_opts[gi][i], grid_hfdf[gi][i], ini_method)

                if this_success:
                    logger.info("\t (dispatched)")
                else:
                    logger.info("\t (failed)")


        # Check if all are done
        done = bool(count_completed == tot_runs)
        if done:
            print(grid_stat)
        
    # Check if exited early
    if not done:
        logger.info("WARNING: Master process loop terminated early!")
        logger.info("         This could be because it timed-out or an error occurred.")

    # Tidy VULCAN output folder
    # doesn't matter which 'dirs' is used, because VULCAN is always in the same place
    for f in glob.glob(dirs["vulcan"]+"output/%d_offchem_*"%now): 
        os.remove(f)

    time_end = datetime.now()
    logger.info("All processes finished at: "+str(time_end.strftime("%Y-%m-%d %H%M%S")))
    logger.info("Total runtime: %.1f hours "%((time_end-time_start).total_seconds()/3600.0))

    # Done
    logger.info("Done running GridOfflinehemistry!")


# If run directly
if __name__ == '__main__':

    # Parameters
    folder =        "output/pspace_redox"  # GridPROTEUS output folder
    samples =       1                   # How many samples to use per grid point
    threads =       70                  # How many threads to use in total
    mkfuncs =       True                # Compile reaction functions again?
    s_width =       1e9                 # Scale width of sampling distribution [yr]
    s_centre =      1e6                 # Centre of sampling distribution [yr]
    runtime_sleep = 40                  # Sleep seconds per iter (required for VULCAN warm up periods to pass)
    ini_method =    1                   # Method used to init VULCAN abundances  (0: const_mix, 1: eqm)

    # Do it
    parent(folder, samples, threads, s_width, s_centre, 
           mkfuncs=mkfuncs, runtime_sleep=runtime_sleep, ini_method=ini_method)
    
